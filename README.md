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

## Quick Start

1. Down ODIM dataset

- Our constructed ODIM dataset is based on ODI-SR dataset proposed by LAU-Net.

1. Download pre-trained models on ODIM dataset.

Model|File size|Download
:-:|:-:|:-:
EDSR-baseline-LIIF|18M|[Dropbox](https://www.dropbox.com/s/6f402wcn4v83w2v/edsr-baseline-liif.pth?dl=0) &#124; [Google Drive](https://drive.google.com/file/d/1wBHSrgPLOHL_QVhPAIAcDC30KSJLf67x/view?usp=sharing)
RDN-LIIF|256M|[Dropbox](https://www.dropbox.com/s/mzha6ll9kb9bwy0/rdn-liif.pth?dl=0) &#124; [Google Drive](https://drive.google.com/file/d/1xaAx6lBVVw_PJ3YVp02h3k4HuOAXcUkt/view?usp=sharing)

2. Convert your image to LIIF and present it in a given resolution (with GPU 0, `[MODEL_PATH]` denotes the `.pth` file)

```
python demo.py --input xxx.png --model [MODEL_PATH] --resolution [HEIGHT],[WIDTH] --output output.png --gpu 0
```

## Reproducing Experiments

### Data

`mkdir load` for putting the dataset folders.

- **DIV2K**: `mkdir` and `cd` into `load/div2k`. Download HR images and bicubic validation LR images from [DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (i.e. [Train_HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), [Valid_HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip), [Valid_LR_X2](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip), [Valid_LR_X3](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip), [Valid_LR_X4](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip)). `unzip` these files to get the image folders.

- **benchmark datasets**: `cd` into `load/`. Download and `tar -xf` the [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (provided by [this repo](https://github.com/thstkdgus35/EDSR-PyTorch)), get a `load/benchmark` folder with sub-folders `Set5/, Set14/, B100/, Urban100/`.

- **celebAHQ**: `mkdir load/celebAHQ` and `cp scripts/resize.py load/celebAHQ/`, then `cd load/celebAHQ/`. Download and `unzip` data1024x1024.zip from the [Google Drive link](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P?usp=sharing) (provided by [this repo](github.com/suvojit-0x55aa/celebA-HQ-dataset-download)). Run `python resize.py` and get image folders `256/, 128/, 64/, 32/`. Download the [split.json](https://www.dropbox.com/s/2qeijojdjzvp3b9/split.json?dl=0).

### Running the code

**0. Preliminaries**

- For `train_liif.py` or `test.py`, use `--gpu [GPU]` to specify the GPUs (e.g. `--gpu 0` or `--gpu 0,1`).

- For `train_liif.py`, by default, the save folder is at `save/_[CONFIG_NAME]`. We can use `--name` to specify a name if needed.

- For dataset args in configs, `cache: in_memory` denotes pre-loading into memory (may require large memory, e.g. ~40GB for DIV2K), `cache: bin` denotes creating binary files (in a sibling folder) for the first time, `cache: none` denotes direct loading. We can modify it according to the hardware resources before running the training scripts.

**1. DIV2K experiments**

**Train**: `python train_liif.py --config configs/train-div2k/train_edsr-baseline-liif.yaml` (with EDSR-baseline backbone, for RDN replace `edsr-baseline` with `rdn`). We use 1 GPU for training EDSR-baseline-LIIF and 4 GPUs for RDN-LIIF.

**Test**: `bash scripts/test-div2k.sh [MODEL_PATH] [GPU]` for div2k validation set, `bash scripts/test-benchmark.sh [MODEL_PATH] [GPU]` for benchmark datasets. `[MODEL_PATH]` is the path to a `.pth` file, we use `epoch-last.pth` in corresponding save folder.

**2. celebAHQ experiments**

**Train**: `python train_liif.py --config configs/train-celebAHQ/[CONFIG_NAME].yaml`.

**Test**: `python test.py --config configs/test/test-celebAHQ-32-256.yaml --model [MODEL_PATH]` (or `test-celebAHQ-64-128.yaml` for another task). We use `epoch-best.pth` in corresponding save folder.
