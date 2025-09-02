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
pip install -e .
```

### From PyPi
```
pip install animl
```

### Dependencies
We recommend running AniML on GPU-enabled hardware. **If using an NVIDIA GPU, ensure driviers, cuda-toolkit and cudnn are installed.
PyTorch will install these automatically if using a conda environment. 
The /models/ and /utils/ modules are from the YOLOv5 repository.  https://github.com/ultralytics/yolov5

**Python** <= 3.9

**PyTorch** <br>
Animl currently depends on torch <= 2.5.0.
To enable GPU, install the [CUDA-enabled version](https://pytorch.org/get-started/previous-versions/)

Python Package Dependencies
* numpy >= 1.26.4
* onnxruntime-gpu == 1.18.0
* pandas >= 2.2.2
* panoptes_client >= 1.6.2
* pillow > 10.3.0
* opencv-python >= 4.10.0.82
* scikit-learn >= 1.5.0
* timm >= 1.0,
* tqdm >= 4.66.4
* wget >= 3.2



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
python -m animl /example/folder --detector /path/to/megadetector --classifier /path/to/classifier --classlist /path/to/classlist.txt
```
You can use animl in this fashion on any image directory.

Finally you can use the animl.yml config file to specify parameters:
```
python -m animl /path/to/animl.yml
```

## Usage

### Inference
The functionality of animl can be parcelated into its individual functions to suit your data and scripting needs.
The sandbox.ipynb notebook has all of these steps available for further exploration.

1. It is recommended that you use the animl working directory for storing intermediate steps.
```python
import animl
workingdir = animl.WorkingDirectory('/path/to/save/data')
```

2. Build the file manifest of your given directory. This will find both images and videos.
```python
files = animl.build_file_manifest('/path/to/images', out_file=workingdir.filemanifest, exif=True)
```

3. If there are videos, extract individual frames for processing.
   Select either the number of frames or fps using the argumments.
   The other option can be set to None or removed.
```python
allframes = animl.extract_frames(files, out_dir=workingdir.vidfdir, out_file=workingdir.imageframes,
                                 parallel=True, frames=3, fps=None)
```

4. Pass all images into MegaDetector. We recommend [MDv5a](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt).
   The function parse_MD will convert the json to a pandas DataFrame and merge detections with the original file manifest, if provided.

```python
detector = animl.load_detector('/path/to/mdmodel.pt', model_type="MDV5", device='cuda:0')
mdresults = animl.detect(detector, allframes, file_col="frame", checkpoint_path=working_dir.mdraw, quiet=True)
detections = animl.parse_detections(mdresults, manifest=all_frames, out_file=workingdir.detections)
```

5. For speed and efficiency, extract the empty/human/vehicle detections before classification.
```python
animals = animl.get_animals(detections)
empty = animl.get_empty(detections)
```
6. Classify using the appropriate species model. Merge the output with the rest of the detections
   if desired.
```python
class_list = animl.load_class_list('/path/to/classlist.txt')
classifier = animl.load_classifier('/path/to/model', len(class_list), device='cuda:0')
raw_predictions = animl.classify(classifier, animals, file_col="frame",
                         batch_size=4, out_file=working_dir.predictions)
```

7. Apply labels from class list with or without utilizing timestamp-based sequences.
```python
animals = animl.single_classification(animals, raw_predictions, class_list['class'])
manifest = pd.concat([animals if not animals.empty else None, empty if not empty.empty else None]).reset_index(drop=True)
```
or 
```python
manifest = animl.sequence_classification(animals, empty, 
                                         raw_predictions,
                                         class_list['class'],
                                         station_col='station',
                                         empty_class="",
                                         sort_columns=None,
                                         file_col="filepath",
                                         maxdiff=60)
```

8. (OPTIONAL) Save the Pandas DataFrame's required columns to csv and then use it to create json for TimeLapse compatibility
```python
csv_loc = animl.csv_converter(animals, empty, imagedir, only_animl = True)
animl.animl_results_to_md_results(csv_loc, imagedir + "final_result.json")
```

9. (OPTIONAL) Create symlinks within a given directory for file browser access.
```python
manifest = animl.export_folders(manifest, out_dir=working_dir.linkdir, out_file=working_dir.results)
```

---
### Training

Training workflows are still under development. Please submit Issues as you come upon them.

1. Assuming a file manifest of training data with species labels, first split the data into training, validation and test splits.
   This function splits each label proportionally by the given percentages, by default 0.7 training, 0.2 validation, 0.1 Test.
```python
train, val, test, stats = animl.train_val_test(manifest, out_dir='path/to/save/data/', label_col="species",
                   percentage=(0.7, 0.2, 0.1), seed=None)
```

2. Set up training configuration file. Specify the paths to the data splits from the previous step. See [config README]()

3. (Optional) Update train.py to include MLOPS connection. 

4. Using the config file, begin training
```bash
python -m animl.train --config /path/to/config.yaml
```
Every 10 epochs (or define custom 'checkpoint_frequency'), the model will be checkpointed to the 'experiment_folder' parameter in the config file, and will contain performance metrics for selection.


5. Testing of a model checkpoint can be done with the "test.py" module.  Add an 'active_model' parameter to the config file that contains the path of the checkpoint to test.
   This will produce a confusion matrix of the test dataset as well as a csv containing predicted and ground truth labels for each image.
```bash
python -m animl.test --config /path/to/config.yaml
```

# Models

The Conservation Technology Lab has several models available for use. 

* Southwest United States [v3](https://sandiegozoo.box.com/s/0mait8k3san3jvet8251mpz8svqyjnc3)
* [Amazon](https://sandiegozoo.box.com/s/dfc3ozdslku1ekahvz635kjloaaeopfl)
* [Savannah](https://sandiegozoo.box.com/s/ai6yu45jgvc0to41xzd26moqh8amb4vw)
* [Andes](https://sandiegozoo.box.com/s/kvg89qh5xcg1m9hqbbvftw1zd05uwm07)
* [MegaDetector](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt)
