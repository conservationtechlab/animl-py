# animl-py
AniML comprises a variety of machine learning tools for analyzing ecological data. This Python package includes a set of functions to classify subjects within camera trap field data and can handle both images and videos. 
This package is also available in R: [animl](https://github.com/conservationtechlab/animl)

Table of Contents
1. Installation
2. [Usage](#usage)

## Installation Instructions

### From GitHub
```
git clone https://github.com/conservationtechlab/animl-py.git
```
It is recommended that you set up a conda environment using the included environment.yml folder.
See **Dependencies** below for more detail.
```
cd animl-py
conda env create --file environment.yml
conda activate animl-gpu
```

### From PyPi
Note: for the latest version of animl, follow instructions for github


### Dependencies
We recommend running AniML on GPU-enabled hardware. **If using an NVIDIA GPU, ensure driviers, cuda-toolkit and cudnn are installed.

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
From the animl-py/src directory, test AniML on the included examples by running the test script.
This will download MegaDetector 
```
python3 animl
```


## Usage

### Inference



### Training
