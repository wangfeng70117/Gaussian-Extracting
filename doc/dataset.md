# Dataset
We used scenes from datasets such as LERF, MIP-NERF, and LLFF, and processed some of the images into additional grayscale images to create the dataset required for this paper. The created dataset is available at [huggingface](https://huggingface.co/datasets/wfysu/GaussianExtracting/tree/main).

The original datasets such as LERF, MIP-NERF, and LLFF can be downloaded from [NeRF Data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

In particular, in the data set figurines, we provide a pre-trained model trained using original 3DGS and Gaussian Grouping and images from various perspectives of 3D scenes trained by the two methods. These images can be used for the evaluation of reconstruction quality

## 1. Segmentation Quality
We also provide a script for evaluating IoU and Boundary-IoU. You can change the output path to your output folder and run the script.

For example,
```
python eval_gaussian_extracting.py figurines
python eval_gaussian_extracting.py teatime
