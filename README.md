# OmniZoomer
The official implementation of [**OmniZoomer**](https://arxiv.org/abs/2308.08114): Learning to Move and Zoom in on Sphere at High-Resolution.

<br>
Zidong Cao, Hao Ai, Yan-Pei Cao, Ying Shan, Xiaohu Qie, Lin Wang
<br>
ICCV 2023

The project page with video is at https://vlislab22.github.io/OmniZoomer/.

### Citation

If you find our work useful in your research, please cite:

```
@inproceedings{cao2023omnizoomer,
  title={OmniZoomer: Learning to Move and Zoom in on Sphere at High-Resolution},
  author={Cao, Zidong and Ai, Hao and Cao, Yan-Pei and Shan, Ying and Qie, Xiaohu and Wang, Lin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12897--12907},
  year={2023}
}
```

### Environment
Our code is based on LIIF, and the basic requirements include:
- Python 3.9
- Pytorch 1.13.0, cuda-toolkit 11.7.1, torchvision 0.14.0
- TensorboardX
- yaml, numpy, tqdm, imageio

## Quick demo for the transformation

- Without pre-trained models, just try the Mobius transformation:

```
python demo.py --input [] --scale [] --output [] --gpu 0
```

- With pre-trained models, try the Mobius transformation:

```
python demo.py --input [] --model [] --scale [] --output [] --gpu 0
```

- We currently support two types of Mobius transformation, i.e., the north pole and the equator. You can spefify it in the Line 48 of the demo.py.

## Evaluation

1. Download ODIM dataset

- Our constructed ODIM dataset is based on ODI-SR dataset proposed by [LAU-Net](https://github.com/wangh-allen/LAU-Net). You can download the ODI-SR dataset in the following link:

```
https://drive.google.com/drive/folders/1w7m1r-yCbbZ7_xMGzb6IBplPe4c89rH9?usp=sharing
```

Please note that the training set of ODI-SR contains 9 panoramas whose spatial resolution is not 1024x2048. In our training process, we delete these 9 panoramas.

- Then, please download the transformation matrices (Total number of 100) in our ODIM dataset in this link:

```
https://drive.google.com/drive/folders/1QbR-8JoqcytY5T1uXPFEmgzu05q44-GH?usp=drive_link
```

2. Download pre-trained models on ODIM dataset. We also provide the updated models in our journal version, called OmniVR.

Model|Up-sampling factor|File size|Download
:-:|:-:|:-:|:-:
OmniZoomer-RCAN|x8|184M|[Google Drive](https://drive.google.com/drive/folders/122iMokJZNrmsUP1-NBRaElzqR5wcs0Pa?usp=sharing)
OmniZoomer-RCAN|x16|186M|[Google Drive](https://drive.google.com/drive/folders/123-vujUO-9AsD_mTgdGkNBBUkzhLbqB-?usp=sharing)
OmniVR-RCAN|x8|184M|[Google Drive](https://drive.google.com/drive/folders/127D6fyQRH134EjTfCYj0mVtiEIarNQaP?usp=sharing)
OmniVR-RCAN|x16|186M|[Google Drive](https://drive.google.com/drive/folders/12Lgd8v7RHh1bv4BBBVzVNnYcCrNRxRtY?usp=sharing)

3. Download the ERP weights for WS-PSNR and WS-SSIM.

- It can be skipped by following the evaluation metric in [LAU-Net](https://github.com/wangh-allen/LAU-Net). However, it could be very slow due to repetitive calculations. We recommend to download the weights file (.npy) in the follwing link:


```
https://drive.google.com/drive/folders/12ahw3N2p9t8w67L4LDRzrMZb_VH42_Sp?usp=sharing
```

4. Run the following code (Take ODI-SR dataset and x8 up-sample for example):

```
python test.py --config configs/test-osr/test-osr-8.yaml --model save/_OmniZoomer_rcan-x8/epoch-best.pth --gpu 0
```

## Acknowledgement

We sincerely thank the following open-source works!

- [LIIF](https://github.com/yinboc/liif): An implicit super-resolution method
- [LTE](https://github.com/jaewon-lee-b/lte): An implicit super-resolution method in the Fourier space
- [Mobius transformation (Numpy version)](https://github.com/henryseg/spherical_image_editing)

## Other links

- This [video](https://www.youtube.com/watch?v=oVwmF_vrZh0) could better understand the Mobius transformation.
- A online [app](https://community.theta360.guide/t/apply-rotation-and-zoom-to-stills-in-seconds-other-mobius-transformations-too/1479) that supports to update your own panoramas for transformation.
