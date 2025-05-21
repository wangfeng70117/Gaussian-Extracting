/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#define BOX_SIZE 1024

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "simple_knn.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

struct CustomMin
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
	}
};

struct CustomMax
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};

__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	//计算坐标值x相对于范围[minn, maxx]的比例，然后左移10位并减去1，即映射到0-1023的整数范围
	//用于生成莫顿码所需的32位整数
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2);
}

__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
{
	//获取当前线程在网格上的索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	//code idx 莫顿码  可以将三维坐标映射到1维，彼此相近的坐标具有彼此接近的莫顿码
	//code[0]是point[0]的莫顿码
	codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax
{
	float3 minn;
	float3 maxx;
};

__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
{
	//indices 排序后的数据下标
	auto idx = cg::this_grid().thread_rank();

	MinMax me;
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	else
	{
		me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
		me.maxx = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	}

	__shared__ MinMax redResult[BOX_SIZE];
	//me: 当前点的位置坐标
	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)
	{
		//ThreadId.x：当前线程块中的索引，比如当前线程块使用100个线程，idx就是0-99
		if (threadIdx.x < 2 * off)
			redResult[threadIdx.x] = me;
		__syncthreads();

		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		boxes[blockIdx.x] = me;
}

__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	float3 diff = { 0, 0, 0 };
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
{
	//knn 长度是3   ref原点   point  相近点
	//d: 距离向量  dist两点距离
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}

__global__ void boxMeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
	//根据莫顿编码找到相近的6个点
	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		//在临近的6个点中找到最相近的三个点
		updateKBest<3>(point, points[indices[i]], best);
	}

	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;
	//b 包围盒个数
	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		//计算当前点到包围盒的距离， 如果点在包围盒内则距离是0
		float dist = distBoxPoint(box, point);
		//reject：距离当前点最近的三个点其中最远的那个距离
		//点到包围盒的距离大于最近点距离，就continue
		if (dist > reject || dist > best[2])
			continue;
		//在点到包围盒距离小于最近点距离时
		//i：包围盒中的所有粒子
		//之前只计算了排序后临近六个，没有计算其他位置的，所以需要计算一个包围盒
		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			//在包围盒中的所有粒子中找到最近的
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	//dist：每个点到距离最近的三个点的平均距离
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;
}

/// <summary>
///
/// </summary>
/// <param name="P">粒子个数</param>
/// <param name="points">float3的指针，就是P个float3 </param>
/// <param name="meanDists">P个flaot</param>
void SimpleKNN::knn(int P, float3* points, float* meanDists)
{
	//result: 
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	//获取所需的临时存储空间大小，该值存储在temp_storage_bytes中
	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	//找到所有点集中的xyz最小值
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);
	//找到所有点集中的xyz最大值
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	//在GPU上创建P大小的 uint32_t类型向量
	thrust::device_vector<uint32_t> morton(P);
	//在GPU上创建P大小的 uint32_t类型向量
	thrust::device_vector<uint32_t> morton_sorted(P);
	//morton是莫顿编码的值\
	//morton[0]是第0个Point的莫顿编码
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	//生成一个递增的序列  每个值都比前一个值大1
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);
	//排序                                                                         输入键数组           输出键数组                  输入值数组（递增序列） 输出值数组
	//根据morton中的值进行排序 排序后的index存储在indices_sorted
	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	//BOX_SIZE： 每个包围盒中有多少个粒子
	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	//num_boxes： 需要创建多少个包围盒
	//创建num_boxes个box
	thrust::device_vector<MinMax> boxes(num_boxes);
	//找到每个box的最大值和最小值
	//线程块数： 包围盒个数   线程数： 粒子数
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	boxMeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}


__global__ void boxMeanDist10(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	float best[10] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
	//根据莫顿编码找到相近的6个点
	for (int i = max(0, idx - 10); i <= min(P - 1, idx + 10); i++)
	{
		if (i == idx)
			continue;
		//在临近的6个点中找到最相近的三个点
		updateKBest<10>(point, points[indices[i]], best);
	}

	float reject = best[9];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;
	best[3] = FLT_MAX;
	best[4] = FLT_MAX;
	best[5] = FLT_MAX;
	best[6] = FLT_MAX;
	best[7] = FLT_MAX;
	best[8] = FLT_MAX;
	best[9] = FLT_MAX;
	//b 包围盒个数
	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		//计算当前点到包围盒的距离， 如果点在包围盒内则距离是0
		float dist = distBoxPoint(box, point);
		//reject：距离当前点最近的三个点其中最远的那个距离
		//点到包围盒的距离大于最近点距离，就continue
		if (dist > reject || dist > best[9])
			continue;
		//在点到包围盒距离小于最近点距离时
		//i：包围盒中的所有粒子
		//之前只计算了排序后临近六个，没有计算其他位置的，所以需要计算一个包围盒
		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			//在包围盒中的所有粒子中找到最近的
			updateKBest<10>(point, points[indices[i]], best);
		}
	}
	//dist：每个点到距离最近的三个点的平均距离
	dists[indices[idx]] = (best[0] + best[1] + best[2] + best[3] + best[4] + best[5] + best[6] + best[7] + best[8] + best[9] ) / 10.0f;
}

__global__ void boxMeanDist30(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	float best[30];
	for (int i = 0; i < 30; ++i) {
        best[i] = FLT_MAX;
    }
	//根据莫顿编码找到相近的6个点
	for (int i = max(0, idx - 10); i <= min(P - 1, idx + 10); i++)
	{
		if (i == idx)
			continue;
		//在临近的6个点中找到最相近的三个点
		updateKBest<50>(point, points[indices[i]], best);
	}

	float reject = best[29];
	for (int i = 0; i < 30; ++i) {
        best[i] = FLT_MAX;
    }
	//b 包围盒个数
	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		//计算当前点到包围盒的距离， 如果点在包围盒内则距离是0
		float dist = distBoxPoint(box, point);
		//reject：距离当前点最近的三个点其中最远的那个距离
		//点到包围盒的距离大于最近点距离，就continue
		if (dist > reject || dist > best[9])
			continue;
		//在点到包围盒距离小于最近点距离时
		//i：包围盒中的所有粒子
		//之前只计算了排序后临近六个，没有计算其他位置的，所以需要计算一个包围盒
		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			//在包围盒中的所有粒子中找到最近的
			updateKBest<10>(point, points[indices[i]], best);
		}
	}
	//dist：每个点到距离最近的三个点的平均距离
	dists[indices[idx]] = (best[0] + best[1] + best[2] + best[3] + best[4] + best[5] + best[6] + best[7] + best[8] + best[9] ) / 10.0f;
}

void SimpleKNN::knn10(int P, float3* points, float* meanDists)
{
	//result: 
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	//获取所需的临时存储空间大小，该值存储在temp_storage_bytes中
	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	//找到所有点集中的xyz最小值
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);
	//找到所有点集中的xyz最大值
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	//在GPU上创建P大小的 uint32_t类型向量
	thrust::device_vector<uint32_t> morton(P);
	//在GPU上创建P大小的 uint32_t类型向量
	thrust::device_vector<uint32_t> morton_sorted(P);
	//morton是莫顿编码的值\
	//morton[0]是第0个Point的莫顿编码
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	//生成一个递增的序列  每个值都比前一个值大1
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);
	//排序                                                                         输入键数组           输出键数组                  输入值数组（递增序列） 输出值数组
	//根据morton中的值进行排序 排序后的index存储在indices_sorted
	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	//BOX_SIZE： 每个包围盒中有多少个粒子
	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	//num_boxes： 需要创建多少个包围盒
	//创建num_boxes个box
	thrust::device_vector<MinMax> boxes(num_boxes);
	//找到每个box的最大值和最小值
	//线程块数： 包围盒个数   线程数： 粒子数
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	boxMeanDist10 << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}

void SimpleKNN::knn30(int P, float3* points, float* meanDists)
{
	//result: 
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	//获取所需的临时存储空间大小，该值存储在temp_storage_bytes中
	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	//找到所有点集中的xyz最小值
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);
	//找到所有点集中的xyz最大值
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	//在GPU上创建P大小的 uint32_t类型向量
	thrust::device_vector<uint32_t> morton(P);
	//在GPU上创建P大小的 uint32_t类型向量
	thrust::device_vector<uint32_t> morton_sorted(P);
	//morton是莫顿编码的值\
	//morton[0]是第0个Point的莫顿编码
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	//生成一个递增的序列  每个值都比前一个值大1
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);
	//排序                                                                         输入键数组           输出键数组                  输入值数组（递增序列） 输出值数组
	//根据morton中的值进行排序 排序后的index存储在indices_sorted
	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	//BOX_SIZE： 每个包围盒中有多少个粒子
	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	//num_boxes： 需要创建多少个包围盒
	//创建num_boxes个box
	thrust::device_vector<MinMax> boxes(num_boxes);
	//找到每个box的最大值和最小值
	//线程块数： 包围盒个数   线程数： 粒子数
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	boxMeanDist30 << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}