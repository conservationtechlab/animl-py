# animl-py
AniML comprises a variety of machine learning tools for analyzing ecological data. This Python package includes a set of functions to classify subjects within camera trap field data and can handle both images and videos. 
This package is also available in R: [animl](https://github.com/conservationtechlab/animl)

Table of Contents
1. Installation
2. [Usage](#usage)

## Installation Instructions

It is recommended that you set up a conda environment using the included environment.yml folder.
See **Dependencies** below for more detail. You will have to activate the conda environment first each
time you want to run AniML from a new terminal.

```
git clone https://github.com/conservationtechlab/animl-py.git
cd animl-py
conda env create --file environment.yml
conda activate animl-gpu
pip install -e .
```

### From PyPi
With NVIDIA GPU
```
conda create -n animl-gpu python=3.7
conda activate animl-gpu
conda install cudatoolkit=11.3.1 cudnn=8.2.1
pip install animl
```
CPU only
```
conda create -n animl-cpu python=3.7
conda activate animl
pip install animl
```

### Dependencies
We recommend running AniML on GPU-enabled hardware. **If using an NVIDIA GPU, ensure driviers, cuda-toolkit and cudnn are installed.
The /models/ and /utils/ modules are from the YOLOv5 repository.  https://github.com/ultralytics/yolov5

Python Package Dependencies
- pandas = 1.3.5
- tensorflow = 2.6
- torch = 1.13.1
- torchvision = 0.14.1
- numpy = 1.19.5
- cudatoolkit = 11.3.1 **
- cudnn = 8.2.1 **

A full list of dependencies can be found in environment.yml

### Verify Install 
We recommend you download the [examples](https://github.com/conservationtechlab/animl-py/blob/main/examples/Southwest.zip) folder within this repository.
Download and unarchive the zip folder. Then with the conda environment active:
```
python3 -m animl /path/to/example/folder
```
This should create an Animl-Directory subfolder within
the example folder.

## Usage

### Inference



### Training
