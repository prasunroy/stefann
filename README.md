<h1 align="center">
  <img src="https://raw.githubusercontent.com/prasunroy/stefann/master/docs/static/imgs/logo.png">
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/build-passing-00c800?style=flat-square">
  <img src="https://img.shields.io/badge/dependencies-up%20to%20date-00c800?style=flat-square">
  <img src="https://img.shields.io/badge/contributions-welcome-ff40ff?style=flat-square">
  <img src="https://img.shields.io/badge/license-Apache--2.0-6464ff?style=flat-square">
  <img src="https://img.shields.io/badge/accepted-CVPR%202020-6464ff?style=flat-square">
</p>

<p align="center">
  <a href="https://github.com/prasunroy/stefann#getting-started">Getting Started</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://github.com/prasunroy/stefann#training-networks">Training Networks</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://github.com/prasunroy/stefann#external-links">External Links</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://github.com/prasunroy/stefann#citation">Citation</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://github.com/prasunroy/stefann#license">License</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/prasunroy/stefann/master/docs/static/imgs/teaser.jpg">
  <br>
  <br>
  <b><i>The official GitHub repository for the paper on <a href="https://prasunroy.github.io/stefann">STEFANN: Scene Text Editor using Font Adaptive Neural Network.</a></i></b>
</p>

<br>

## Getting Started
### 1. Installing Dependencies
| Package    | Source | Version | Tested version<br>(Updated on April 14, 2020) |
| :--------- | :----: | :-----: | :-------------------------------------------: |
| Python     | Conda  | 3.7.7   | :heavy_check_mark: |
| Pip        | Conda  | 20.0.2  | :heavy_check_mark: |
| Numpy      | Conda  | 1.18.1  | :heavy_check_mark: |
| Requests   | Conda  | 2.23.0  | :heavy_check_mark: |
| TensorFlow | Conda  | 2.1.0   | :heavy_check_mark: |
| Keras      | Conda  | 2.3.1   | :heavy_check_mark: |
| Pillow     | Conda  | 7.0.0   | :heavy_check_mark: |
| Colorama   | Conda  | 0.4.3   | :heavy_check_mark: |
| OpenCV     | PyPI   | 4.2.0   | :heavy_check_mark: |
| PyQt5      | PyPI   | 5.14.2  | :heavy_check_mark: |

### :boom: Quick installation
#### Step 1: Install [Git](https://git-scm.com/) and [Conda](https://docs.conda.io/) package manager (Miniconda / Anaconda)
#### Step 2: Update and configure Conda
```bash
conda update conda
conda config --set env_prompt "({name}) "
```
#### Step 3: Clone this repository and change directory to repository root
```bash
git clone https://github.com/prasunroy/stefann.git
cd stefann
```
#### Step 4: Create an environment and install depenpencies

#### On Linux and Windows
* To create **CPU** environment: `conda env create -f release/env_cpu.yml`
* To create **GPU** environment: `conda env create -f release/env_gpu.yml`

#### On macOS
* To create **CPU** environment: `conda env create -f release/env_osx.yml`

### :boom: Quick test
#### Step 1: [Download](https://drive.google.com/open?id=16-mq3MOR1zmOsxNgegRmGDeVRyeyQ0_H) models and pretrained checkpoints into `release/models` directory
#### Step 2: [Download](https://drive.google.com/uc?export=download&id=1Gzb-VYeQJNXwDnkoEI4iAskOGYmWR6Rk) sample images and extract into `release/sample_images` directory
```
stefann/
├── ...
├── release/
│   ├── models/
│   │   ├── colornet.json
│   │   ├── colornet_weights.h5
│   │   ├── fannet.json
│   │   └── fannet_weights.h5
│   ├── sample_images/
│   │   ├── 01.jpg
│   │   ├── 02.jpg
│   │   └── ...
│   └── ...
└── ...
```
#### Step 3: Activate environment
To activate **CPU** environment: `conda activate stefann-cpu`
<br>
To activate **GPU** environment: `conda activate stefann-gpu`
#### Step 4: Change directory to `release` and run STEFANN
```bash
cd release
python stefann.py
```

### 2. Editing Results :satisfied:
<p align="center">
  <img src="https://raw.githubusercontent.com/prasunroy/stefann/master/docs/static/imgs/results.jpg">
  <br>
  <b>Each image pair consists of the original image (Left) and the edited image (Right).</b>
</p>

<br>

## Training Networks
### 1. Downloading Datasets
#### [Download](https://drive.google.com/open?id=1dOl4_yk2x-LTHwgKBykxHQpmqDvqlkab) datasets and extract the archives into `datasets` directory under repository root.
```
stefann/
├── ...
├── datasets/
│   ├── fannet/
│   │   ├── pairs/
│   │   ├── train/
│   │   └── valid/
│   └── colornet/
│       ├── test/
│       ├── train/
│       └── valid/
└── ...
```

#### :pushpin: Description of `datasets/fannet`
<p align="justify">
  This dataset is used to train <b>FANnet</b> and it consists of 3 directories: <code>fannet/pairs</code>, <code>fannet/train</code> and <code>fannet/valid</code>. The directories <code>fannet/train</code> and <code>fannet/valid</code> consist of 1015 and 300 sub-directories respectively, each corresponding to one specific font. Each font directory contains 64x64 grayscale images of 62 English alphanumeric characters (10 numerals + 26 upper-case letters + 26 lower-case letters). The filename format is <code>xx.jpg</code> where <code>xx</code> is the ASCII value of the corresponding character (e.g. "48.jpg" implies an image of character "0"). The directory <code>fannet/pairs</code> contains 50 image pairs, each corresponding to a random font from <code>fannet/valid</code>. Each image pair is horizontally concatenated to a dimension of 128x64. The filename format is <code>id_xx_yy.jpg</code> where <code>id</code> is the image identifier, <code>xx</code> and <code>yy</code> are the ASCII values of source and target characters respectively (e.g. "00_65_66.jpg" implies a transformation from source character "A" to target character "B" for the image with identifier "00").
</p>

#### :pushpin: Description of `datasets/colornet`
<p align="justify">
  This dataset is used to train <b>Colornet</b> and it consists of 3 directories: <code>colornet/test</code>, <code>colornet/train</code> and <code>colornet/valid</code>. Each directory consists of 5 sub-directories: <code>_color_filters</code>, <code>_mask_pairs</code>, <code>input_color</code>, <code>input_mask</code> and <code>output_color</code>. The directory <code>_color_filters</code> contains synthetically generated color filters of dimension 64x64 including both solid and gradient colors. The directory <code>_mask_pairs</code> contains a set of 64x64 grayscale image pairs selected at random from 1315 available fonts in <code>datasets/fannet</code>. Each image pair is horizontally concatenated to a dimension of 128x64. For <code>colornet/train</code> and <code>colornet/valid</code> each color filter is applied on each mask pair. This results in 64x64 image triplets of color source image, binary target image and color target image in <code>input_color</code>, <code>input_mask</code> and <code>output_color</code> directories respectively. For <code>colornet/test</code> one color filter is applied only on one mask pair to generate similar image triplets. With a fixed set of 100 mask pairs, 80000 <code>colornet/train</code> and 20000 <code>colornet/valid</code> samples are generated from 800 and 200 color filters respectively. With another set of 50 mask pairs, 50 <code>colornet/test</code> samples are generated from 50 color filters.
</p>

### 2. Training FANnet and Colornet
#### Step 1: Activate environment
To activate **CPU** environment: `conda activate stefann-cpu`
<br>
To activate **GPU** environment: `conda activate stefann-gpu`
#### Step 2: Change directory to project root
```bash
cd stefann
```
#### Step 3: Configure and train FANnet
To configure training options edit `configurations` section `(line 40-72)` of `fannet.py`
<br>
To start training: `python fannet.py`
###### :cloud: Check [this notebook](https://www.kaggle.com/prasunroy/starter-1-font-generation-stefann-cvpr-2020) hosted at Kaggle for an interactive demonstration of FANnet.
#### Step 4: Configure and train Colornet
To configure training options edit `configurations` section `(line 38-65)` of `colornet.py`
<br>
To start training: `python colornet.py`
###### :cloud: Check [this notebook](https://www.kaggle.com/prasunroy/starter-2-color-transfer-stefann-cvpr-2020) hosted at Kaggle for an interactive demonstration of Colornet.

<br>

## External Links
<h3 align="center">
  <a href="https://prasunroy.github.io/stefann">Project</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://prasunroy.github.io/stefann/static/docs/08915.pdf">Paper</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://prasunroy.github.io/stefann/static/docs/08915-supp.pdf">Supplementary Materials</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://drive.google.com/open?id=1dOl4_yk2x-LTHwgKBykxHQpmqDvqlkab">Datasets</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://drive.google.com/open?id=16-mq3MOR1zmOsxNgegRmGDeVRyeyQ0_H">Models</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://drive.google.com/uc?export=download&id=1Gzb-VYeQJNXwDnkoEI4iAskOGYmWR6Rk">Sample Images</a>
</h3>

<br>

## Citation
```
@InProceedings{Roy_2020_CVPR,
  title     = {STEFANN: Scene Text Editor using Font Adaptive Neural Network},
  author    = {Roy, Prasun and Bhattacharya, Saumik and Ghosh, Subhankar and Pal, Umapada},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2020}
}
```

<br>

## License
```
Copyright 2020 by the authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```




##### Made with :heart: and :pizza: on Earth.
