# Gaussian Extracting: A Lightweight and Interactive  Gaussian Segmentation Method

# Introduction
The recently developed 3D Gaussian Splatting (3DGS) stands out as an exceptional approach for visualizing 3D representations, outperforming the Neural Radiance Field (NeRF) in both training time and reconstruction quality. These advantages imply that 3DGS could be of broader use in the fields of 3D comprehension and editing. Meanwhile, 3DGS segmentation methods are still in the nascent stage. We propose an lightweight and interactive 3D Gaussian segmentation pipeline with the ability to easily integrate with any existing trained 2D segmentation model. This integration enhances the accuracy and efficiency of segmenting specific objects from 3D scenes, allowing for rapid and precise object identification. We train a category feature for each Gaussian and concurrently train an MLP decoder to predict the category information of models. Consequently, models of the specified category can be efficiently extracted. In order to solve the problem of artifacts in model segmentation, we use the KNN algorithm to refine the segmentation results and reduce the noise generated in model segmentation. Ultimately, the target object in the 3D scene can be segmented from any designated view-point. The experiments demonstrate that our proposed method can be effectively applied to diverse complex scenes, effectively segmenting 3D models within milliseconds while saving considerable memory without dropping the quality of segmentation results.
# Application pipeline
![pipeline](https://github.com/user-attachments/assets/2a393935-f533-45d5-a9ae-d149ec07b25f)

# 3D Object Segmentation
## Single Point Prompt
Our interactive single-point prompt segmentation method excels in speed, completing accurate object segmentation in less than a second, with segmentation times in the millisecond range greatly enhancing the interactivity of our approach.

https://github.com/user-attachments/assets/5cc240a2-61e1-40a5-8ac5-07d240cc06dc

## Multi-Point Prompt
Our method enhances both the robustness and interactivity of scene segmentation, ensuring accurate results even in the presence of overlapping or densely packed objects.

https://github.com/user-attachments/assets/df236cbd-66ef-4d72-b7a2-2bd9c0d3bce1

## Text Prompt
Our method enhances the usability and applicability of 3D segmentation in various complex scenarios, providing a robust solution for accurate model extraction from textual prompts.

https://github.com/user-attachments/assets/bf2ad407-b8cc-4035-b863-8a007686dc64

# Dataset
For evaluating the segmentation quality and reconstruction quality in the paper, you can refer to [the dataset documentation](doc/dataset.md).

# Source Code
Our source code will be released after the paper is accepted.

# Config & Running
## Tracking Anything with DEVA - Automatic Segmentation and 3DGS Integration

This repository demonstrates how to perform automatic video object segmentation using [Tracking Anything with DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA), and how to integrate the results with 3D Gaussian Splatting (3DGS) pipelines.

---

## 🛠️ Setup Instructions

### Step 1: Install Tracking Anything with DEVA

```bash
git clone https://github.com/showlab/Tracking-Anything-with-DEVA.git
cd Tracking-Anything-with-DEVA
pip install -e .
bash scripts/download_models.sh
```

Move demo_automatic.py into the Tracking-Anything-with-DEVA folder to avoid import errors:
```
mv path/to/demo_automatic.py Tracking-Anything-with-DEVA/
```
### Step 2: Install Grounded Segment Anything
```
git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
cd ../..
```
### Grayscale Segmentation
```
python demo_automatic_gray.py \
  --chunk_size 4 \
  --img_path ../data/horns/images \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "./data/horns/gray_seg_output" \
  --suppress_small_objects \
  --SAM_PRED_IOU_THRESHOLD 0.7
```
## 📸 Integrating with 3D Gaussian Splatting
### Step 1: Generate COLMAP Camera Poses
```
python convert.py -s data/bear
```
### Step 2: Train Gaussian Splats
```
python train.py -s data/counter -m data/counter/output
```
### Step 3: Generate Grayscale Segmentation Masks
```
python demo_automatic_gray.py \
  --chunk_size 4 \
  --img_path ../data/counter/images \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "./data/counter/gray_seg_output" \
  --suppress_small_objects \
  --SAM_PRED_IOU_THRESHOLD 0.7
```
Move the generated grayscale masks to /data/category.
### Step 4: Train
```
python train.py -s data/counter -m data/counter/output
```
### Step 5: Launch the Web UI
```
python webui.py --gs_source data/caijian/output/point_cloud/iteration_30000/point_cloud.ply --colmap_path data/caijian --pth_path data/caijian/output/point_cloud/iteration_30000/classifier.pth
```
