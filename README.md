# Gaussian-Extracting
Gaussian Extracting: A Lightweight and Interactive  Gaussian Segmentation Method

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
We used scenes from datasets such as LERF, MIP-NERF, and LLFF, and processed some of the images into additional grayscale images to create the dataset required for this paper. The created dataset is available at 
https://huggingface.co/datasets/wfysu/GaussianExtracting/tree/main.
