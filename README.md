# pytorch-video-recognition
This repository is cloned from [pytorch-video-recognition](https://github.com/jfzhang95/pytorch-video-recognition) and fixed the bugs of the original code.

![image](https://github.com/brianlin314/pytorch-video-classification/blob/master/assets/basketballdunk.gif)
![image](https://github.com/brianlin314/pytorch-video-classification/blob/master/assets/basketballdunk_p.gif)

## Introduction
This repo contains several models for video action recognition,
including C3D, R2Plus1D, R3D, inplemented using PyTorch (0.4.0).
Currently, we train these models on UCF101 and HMDB51 datasets.

- Clone the repo:
    ```Shell
    git clone https://github.com/brianlin314/pytorch-video-classification.git
    cd pytorch-video-classification
    ```

- Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install opencv-python
    pip install tqdm scikit-learn tensorboardX
    ```

- Download pretrained model from [BaiduYun](https://pan.baidu.com/s/1saNqGBkzZHwZpG-A5RDLVw) or 
[GoogleDrive](https://drive.google.com/file/d/19NWziHWh1LgCcHU34geoKwYezAogv9fX/view?usp=sharing).
   Currently only support pretrained model for C3D.

- Configure your dataset and pretrained model path in
[mypath.py](https://github.com/brianlin314/pytorch-video-classification/blob/master/mypath.py).

- You can choose different models and datasets in
[train.py](https://github.com/brianlin314/pytorch-video-classification/blob/master/train.py).

- To train the model, please do:
    ```Shell
    python train.py
    ```

## Datasets:

I used two different datasets: UCF101 and HMDB.

Dataset directory tree is shown below

- **UCF101**

Make sure to put the files as the following structure:
  ```
  UCF-101
  ├── ApplyEyeMakeup
  │   ├── v_ApplyEyeMakeup_g01_c01.avi
  │   └── ...
  ├── ApplyLipstick
  │   ├── v_ApplyLipstick_g01_c01.avi
  │   └── ...
  └── Archery
  │   ├── v_Archery_g01_c01.avi
  │   └── ...
  ```
After pre-processing, the output dir's structure is as follows:
  ```
  ucf101
  ├── train
  │    └── ├── ApplyEyeMakeup
  │        │   ├── v_ApplyEyeMakeup_g01_c01
  │        │   │   ├── 00001.jpg
  │        │   │   └── ...
  │        │   └── ...
  │        ├── ApplyLipstick
  │        │   ├── v_ApplyLipstick_g01_c01
  │        │   │   ├── 00001.jpg
  │        │   │   └── ...
  │        │   └── ...
  │
  ├── test
  │    └── ...
  ├── val
  │    └──...
  ```

Note: HMDB dataset's directory tree is similar to UCF101 dataset's.

## Experiments
These models were trained in machine with NVIDIA TITAN X 12gb GPU. Note that I splited
train/val/test data for each dataset using sklearn. If you want to train models using
official train/val/test data, you can look in [dataset.py](https://github.com/brianlin314/pytorch-video-classification/blob/master/dataloaders/dataset.py), and modify it to your needs.

## tensorboard
可透過輸入指令 `tensorboard --logdir './run'` 開啟 tensorboard 

## 2D and 3D convolution operations
在圖像領域，通常都是針對一張靜態圖像進行卷積，即使用 2D 卷積網路就足夠，但在影片領域，雖然也可以使用 2D 卷積網路進行辨識，但為了保留時間信息，就需要模型學習空間特徵，若使用 2D 卷積來處理影片，那麼就沒辦法考慮連續多幀之間的運動信息，所以提出了 C3D 網路。

![image](https://github.com/brianlin314/pytorch-video-classification/blob/master/assets/3Dconvolution.png)

使用 conv2d 進行影片辨識，可參考[這篇](https://debuggercafe.com/action-recognition-in-videos-using-deep-learning-and-pytorch/)手把手教學

**具體操作**: 通過同時堆疊多個連續幀形成的立方體與一個 3D 核進行卷積。通過這個方法，卷積層上的特徵圖連接到了前一層的多個連續幀，從而捕捉動作信息。

**Notations**: c * l * h * w where c is the number of channels, l is length in number of frames, h and w are the height and width of the frame, respectively. We also refer 3D convolution and pooling kernel size by d * k * k, where d is kernel temporal depth and k is kernel spatial size.

## C3D 
以下為 [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767) 所提及之架構

![image](https://github.com/brianlin314/pytorch-video-classification/blob/master/assets/C3D_structure.png)

## R(2+1)D
[A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html) 是 C3D 作者發表在 CVPR2018 的新作，本文的動機為: **對單幀影片進行 2D 卷積能然可以獲得接近 3D 卷積方法中 SOTA 的結果**。

**What is 2+1 ?**
把 3D 時空卷積分解成 2D 空間卷積和 1D 時間卷積，這個分解的好處:
1. 分解成2個子卷積相較於1個卷積多了一個非線性操作
2. 分解後，讓優化的過程也分解開來，降低了 3D 時空卷積把空間信息和動態信息擰在一起不易優化的困境
經結果顯示，R(2+1)D 更加容易優化，loss也相對更低 
![image](https://github.com/brianlin314/pytorch-video-classification/blob/master/assets/R(2%2B1)D_loss.png)

