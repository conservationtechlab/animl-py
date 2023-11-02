# animl-py
AniML comprises a variety of machine learning tools for analyzing ecological data. This Python package includes a set of functions to classify subjects within camera trap field data and can handle both images and videos. 
This package is also available in R: [animl](https://github.com/conservationtechlab/animl)

Table of Contents
1. Installation
2. [Usage](#usage)
3. [Models](#models)

## Installation Instructions

It is recommended that you set up a conda environment for using animl.
See **Dependencies** below for more detail. You will have to activate the conda environment first each
time you want to run AniML from a new terminal.

### From GitHub
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
python -m animl /path/to/example/folder
```
This should create an Animl-Directory subfolder within
the example folder.

Or, if using your own data/models, animl can be given the paths to those files:
Download and unarchive the zip folder. Then with the conda environment active:
```
python -m animl /path/to/example/folder /path/to/megadetector /path/to/classifier /path/to/classlist.txt
```

## Usage

### Inference
The functionality of animl can be parcelated into its individual functions to suit your data and scripting needs.
The sandbox.ipynb notebook has all of these steps available for further exploration.

1. It is recommended that you use the animl working directory for storing intermediate steps.
```python
from animl import file_management
workingdir = file_management.WorkingDirectory(imagedir)
```

2. Build the file manifest of your given directory. This will find both images and videos.
```python
files = file_management.build_file_manifest('/path/to/images', out_file = workingdir.filemanifest)
```

3. If there are videos, extract individual frames for processing.
   Select either the number of frames or fps using the argumments.
   The other option can be set to None or removed.
```python
from animl import video_processing
allframes = video_processing.images_from_videos(files, out_dir=workingdir.vidfdir,
                                                out_file=workingdir.imageframes,
                                                parallel=True, frames=3, fps=None)
```
4. Pass all images into MegaDetector. We recommend [MDv5a](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt)
   parseMD will merge detections with the original file manifest, if provided.

```python
from animl import detectMD, megadetector, parse_results
detector = megadetector.MegaDetector('/path/to/mdmodel.pt')
mdresults = detectMD.detect_MD_batch(detector, allframes["Frame"], quiet=True)
mdres = parse_results.from_MD(mdresults, manifest=allframes, out_file = workingdir.mdresults)
```
5. For speed and efficiency, extract the empty/human/vehicle detections before classification.
```python
from animl import split
animals = split.getAnimals(mdres)
empty = split.getEmpty(mdres)
```
6. Classify using the appropriate species model. Merge the output with the rest of the detections
   if desired.
```python
from animl import classify, parse_results
classifier = classify.load_classifier('/path/to/classifier/')
predresults = classify.predict_species(animals, classifier, batch = 4)
animals = parse_results.from_classifier(animals, predresults, '/path/to/classlist.txt',
                                         out_file=workingdir.predictions)
manifest = pd.concat([animals,empty])
```


---
### Training

Training workflows are available in the repo but still under development.

# Models

The Conservation Technology Lab has several models available for use. 

* Southwest United States [v2](https://sandiegozoo.box.com/s/mzhv08cxcbunsjuh2yp6o7aa5ueetua6) [v3](https://sandiegozoo.box.com/s/p4ws6v5qnoi87otsie0ckmie0izxzqwo)
* [Amazon](https://sandiegozoo.box.com/s/dfc3ozdslku1ekahvz635kjloaaeopfl)
* [Savannah](https://sandiegozoo.box.com/s/ai6yu45jgvc0to41xzd26moqh8amb4vw)
* [MegaDetector](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt)
