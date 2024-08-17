import os
import numpy as np
from PIL import Image
import cv2
import sys

dataset_name = sys.argv[1]
gt_folder_path = os.path.join('result', 'gt_' + dataset_name)
pred_folder_path = os.path.join('result', 'pred_'+ dataset_name)

iou_scores = []
biou_scores = []
class_counts = {}

print(f'gt_folder_path: {gt_folder_path}')
print(f'pred_folder_path: {pred_folder_path}')


# General util function to get the boundary of a binary mask.
# https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    dt = (dt > 128).astype('uint8')
    gt = (gt > 128).astype('uint8')

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou

def calculate_iou(mask1, mask2):
    """Calculate IoU between two boolean masks."""
    mask1_bool = mask1 > 128
    mask2_bool = mask2 > 128
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def load_mask(mask_path):
    assert os.path.exists(mask_path), f'Mask file not found: {mask_path}'
    return np.array(Image.open(mask_path).convert('L'))

for image_name in os.listdir(gt_folder_path):
    gt_image_path = os.path.join(gt_folder_path, image_name)
    pred_image_path = os.path.join(pred_folder_path, image_name)

    gt_mask = load_mask(gt_image_path)
    pred_mask = load_mask(pred_image_path)
    assert gt_mask.shape == pred_mask.shape

    iou = calculate_iou(gt_mask, pred_mask)
    biou = boundary_iou(gt_mask, pred_mask)

    iou_scores.append(iou)
    biou_scores.append(biou)

miou = np.mean(iou_scores)
mbiou = np.mean(biou_scores)
print(biou_scores)
print(f'mIoU of {dataset_name} is {miou:.4f}')
print(f'mBIoU of {dataset_name} is {mbiou:.4f}')
