# Zolly: Zoom Focal Length Correctly for Perspective-Distorted Human Mesh Reconstruction

## The first work aims to solve 3D Human Mesh Reconstruction task in **perspective-distorted images**. 
![teaser](assets/teaser.png)



## News:
* 2023.Mar.27, [arxiv link](https://arxiv.org/abs/2303.13796) is released.


## Abstract
>As it is hard to calibrate single-view RGB images in the wild, existing 3D human mesh reconstruction (3DHMR) methods either use a constant large focal length or estimate one based on the background environment context, which can not tackle the problem of the torso, limb, hand or face distortion caused by perspective camera projection when the camera is close to the human body. The naive focal length assumptions can harm this task with the incorrectly formulated projection matrices. To solve this, we propose Zolly, the first 3DHMR method focusing on perspective-distorted images. Our approach begins with analysing the reason for perspective distortion, which we find is mainly caused by the relative location of the human body to the camera center. We propose a new camera model and a novel 2D representation, termed distortion image, which describes the 2D dense distortion scale of the human body. We then estimate the distance from distortion scale features rather than environment context features. Afterwards, we integrate the distortion feature with image features to reconstruct the body mesh. To formulate the correct projection matrix and locate the human body position, we simultaneously use perspective and weak-perspective projection loss. Since existing datasets could not handle this task, we propose the first synthetic dataset PDHuman and extend two real-world datasets tailored for this task, all containing perspective-distorted human images. Extensive experiments show that Zolly outperforms existing state-of-the-art methods on both perspective-distorted datasets and the standard benchmark (3DPW).

## Run the code
### Environments
- Requirements
    - Pytorch3D
    - MMHuman3D
### Dataset Preparation
- Preprocessed npzs. (Will be released soon)
(Besides our proposed datasets, we will also provide Human3.6M and 3DPW annotations with ground-truth camera parameters and translations.)
- Images. (Will be released soon)
    - HuMMan
    - SPEC-MTP
    - PDHuman
### Train

### Test & Demo

### Add Your Algorithm


### Results of Zolly
---


- 3DPW

| Config       | PA-MPJPE | MPJPE |  PVE  |
| ------------ | -------- | ----- |  ---  |
| Zolly(R50)   |   44.1   | 72.5  |  84.3 |
| Zolly(H48)   |   **39.8**   | **65.0**  |  **76.3** |

- H36M

| Config       | PA-MPJPE | MPJPE |
| ------------ | -------- | ----- |
| Zolly(R50)   |   34.2   |  52.7 |
| Zolly(H48)   |   **32.3**   |  **49.4** |


- SPEC-MTP (full)

| Config       | PA-MPJPE | MPJPE | PVE |
| ------------ | -------- | ----- | --- |
| [SPEC(R50)](https://github.com/mkocabas/SPEC)     |    71.8   |   116.1  | 136.4|
| Zolly(R50)   |      66.9    | 109.6      | 124.4    |
| Zolly(H48)   |  **65.8**     | **108.2**  | **121.9** |

## Quality Results (For more please refer to our paper.)
![sota](assets/demo_sota.jpg)