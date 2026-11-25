---
layout: default
title: Home
description: Developing open-source technology and machine learning tools for wildlife conservation and ecological research
---

# Getting Started

<br>

{: #about}
## About AniML

**Version 3.4.0**

[GitHub Repository](https://github.com/conservationtechlab/animl-py)

The AniML package is available in Python and R for AI-assisted camera trap image processing.

The AniML package provides functions for ingesting raw image and video files and outputs predictions for species using region-specific species classifier models. We provide several species models including for the African Savanna, the Peruvian Amazon, the Andes mountains, and the Western US. AniML provides the results in a number of export formats, including TimeLapse and CamTrapDP. The package also includes AI-based re-indentification tools and custom species model training. 


<br>

{: #installation}
### Installation

  
Install via the command line:

```bash
$ pip install animl
```

<br>

{: #requirements}
### Requirements

<ins>Required dependencies</ins>:
* pytorch
* ultralytics
* onnx-runtime
* pandas

__Recommended__:
* [ExifTool](https://exiftool.org/)
* [CUDA/cuDNN](https://developer.nvidia.com/cuda/toolkit) (for GPU)


We recommend using AniML with a GPU. To use with an Nvidia GPU, be sure that to install the CUDA-compatible
version of [PyTorch](https://pytorch.org/get-started/locally/)

<br><br>

---
{: #examples}
# Examples and Usage

{: #command-line}
### Command-line Execution

Once installed, AniML can be run from the command line:

```bash
$ python -m animl /path/to/data/folder --detector /path/to/megadetector --classifier /path/to/classifier --classlist /path/to/classlist.txt
```

The **-s** flag will sort the images into species folders. <br>
The **-v** flag will create copies of the images with bounding boxes drawn around the animal detections.

You can use animl in this fashion on any image directory.

If you want more fine-tuned control of certain parameters, you can use the animl.yml config file to specify parameters:

```bash
$ python -m animl /path/to/animl.yml
```

An example configuration .yml file can be found [here](https://github.com/conservationtechlab/animl-py/blob/main/src/animl/config/animl.yml).

---
### Species Classification Inference
{: #inference}

The functionality of animl can be parcelated into its individual functions to suit your data and scripting needs.

1. It is recommended that you use the AniML Working Directory for storing intermediate steps.
    ```python
    import animl
    workingdir = animl.WorkingDirectory('/path/to/save/data')
    ```

2. Build the file manifest of your given directory. This will find both images and videos.
    ```python
    files = animl.build_file_manifest('/path/to/images',
                                      out_file=workingdir.filemanifest,
                                      exif=True,
                                      data_timezone='America/Los_Angeles')
    ```

    The argument `data_timezone` indicates the timezone in which the data was collected, so timestamps are correctly 
    interpreted relative to the local timezone.

3. If there are videos, extract individual frames for processing.
   Select either the number of frames or fps using the argumments.
   The other option can be set to None or removed.

    ```python
    allframes = animl.extract_frames(files, frames=3, out_file=workingdir.imageframes, parallel=True)
    ```

4. Pass all images into MegaDetector. We recommend [MDv5a](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt).
   The function `parse_detections()` will convert the detector output to a pandas DataFrame and merge detections with the original file manifest, if provided.

    ```python
    detector = animl.load_detector('/path/to/mdmodel.pt', model_type="mdv5", device='cuda:0')

    mdresults = animl.detect(detector,
                             allframes,
                             resize_width=animl.MEGADETECTORv5_SIZE,
                             resize_height=animl.MEGADETECTORv5_SIZE,
                             letterbox=True,
                             file_col="filepath",
                             device='cuda:0',
                             checkpoint_path=workingdir.mdraw,
                             checkpoint_frequency=1000)

    detections = animl.parse_detections(mdresults, manifest=allframes, out_file=workingdir.detections)
    ```

5. For speed and efficiency, extract the empty/human/vehicle detections before classification.
    ```python
    animals = animl.get_animals(detections)
    empty = animl.get_empty(detections)
    ```

6. Classify using the appropriate species model. Merge the output with the rest of the detections if desired.
    ```python
    classifier, class_list = animl.load_classifier('/path/to/model', '/path/to/classlist.txt', device='cuda:0')

    raw_predictions = animl.classify(classifier,
                                     animals,
                                     resize_width=480,
                                     resize_height=480, 
                                     file_col="filepath",
                                     batch_size=4,
                                     out_file=workingdir.predictions)
    ```

7. Apply labels from class list with or without utilizing timestamp-based sequences.
    ```python
    manifest = animl.single_classification(animals, empty, raw_predictions, class_list['class'])
    ```

    or, after defining a station column named "station",

    ```python
    manifest = animl.sequence_classification(animals,
                                             empty, 
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
    csv_loc = animl.export_timelapse(manifest, '/path/to/output', only_animal=True)
    animl.export_megadetector(manifest, out_file="final_result.json", detector='MegaDetector v5a')
    ```

9. (OPTIONAL) Create links within a given directory, sorted by species, for file browser access.
    ```python
    manifest = animl.export_folders(manifest, out_dir=workingdir.linkdir, out_file=workingdir.results)
    ```

---
### Classifier Model Training
{: #training-example}

1. Assuming a file manifest of training data with species labels, first split the data into training, validation and test splits.
   This function splits each label proportionally by the given fractions, by default 0.8 training, 0.1 validation, 0.1 test.

    ```python
    train, val, test = animl.export_train_val_test(manifest,
                                                   label_col="species",
                                                   out_dir='path/to/save/data/',
                                                   val_size=0.2,
                                                   test_size=0.1,
                                                   seed=42)
    ```

2. Set up training configuration file. Specify the paths to the data splits from the previous step. See [config README](https://github.com/conservationtechlab/animl-py/blob/main/src/animl/config/README.md).

3. (Optional) Update train.py to include MLOPS connection. 

4. Using the config file, begin training
    ```bash
    python -m animl.train --config /path/to/config.yaml
    ```

    Every 10 epochs (or define custom 'checkpoint_frequency'), the model will be checkpointed to the 'experiment_folder' parameter in the config file, and will contain performance metrics for selection.

5. Testing of a model checkpoint can be done with the "test.py" module.  Add an 'active_model' parameter to the config 
file that contains the path of the checkpoint to test. This will produce a confusion matrix of the test dataset as well 
as a csv containing predicted and ground truth labels for each image.

    ```bash
    python -m animl.test --config /path/to/config.yaml
    ```

---
### Re-Identification
{: #re-id-example}

<br><br>

---
### Exports
{: #export-example}


<br><br>

---
# API Reference
  
<br>

## Full Pipeline
{: #full-pipeline}

### animl.from_paths(image_dir, detector_file, classifier_file, classlist_file, ...)
{: #from-paths}

Runs the full detection + classification pipeline on a directory of images or videos. <br>
AniML will add a Animl-Directory folder to the image_dir to store the outputs.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir`       | str | required | Path to image/video directory |
| `detector_file`   | str | required | Path to MegaDetector model |
| `classifier_file` | str | required | Path to classifier model |
| `classlist_file`  | str | required | Path to classifier class definitions (.csv) |
| `class_label`     | str | 'class' | column in the class list that contains the label to use for prediction output (default "class") |
| `batch_size`      | int | 4       | Batch size for inference |
| `sort`            | bool | False  | Toggle to create symlinks of data sorted by species |
| `visualize`       | bool | False  | Toggle to save bounding box visualizations |
| `sequence`        | bool | False  | Toggle to use sequence-level classification |
| `detect_only`     | bool | False  | Skip classification step |

**Returns:** `pandas.DataFrame` — results of detection and classification, including file paths, 
             detection categories, and predicted classes.

<br><br>

### animl.from_config(config)
{: #from-config}

Runs the full detection + classification pipeline on a directory of images or videos.<br>
AniML will add a Animl-Directory folder to the working_dir to store the outputs.

An example configuration .yml file can be found [here](https://github.com/conservationtechlab/animl-py/blob/main/src/animl/config/animl.yml).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | str | required | Path to config yml file.

**Returns:** `pandas.DataFrame` — results of detection and classification, including file paths, 
             detection categories, and predicted classes.

<br><br>
  
---
## Data Ingestion and Processing
{: #data-ingestion}
<br>

### class animl.WorkingDirectory(working_dir)
{: #workingdirectory}

A WorkingDirectory object creates a folder called "Animl-Directory" within the working_dir. 
Attributes include output file paths to save the outputs of intermediary steps.

* self.filemanifest = "FileManifest.csv", typically used with `build_file_manifest()`
* self.imageframes = "ImageFrames.csv", typically used with `extract_frames()`
* self.mdraw = "MD_Raw.json", typically used with `detect()`
* self.detections = "Detections.csv", typically used with `parse_detections()`
* self.predictions = "Predictions.csv", typically used with `classify()`
* self.results = "Results.csv", typically used with the export functions.

* self.linkdir = "Sorted/", typically used as the `out_dir` of `export_folders()`
* self.visdir = "Plots/", typically used as the `out_dir` of `plot_all_bounding_boxes()`

The "Sorted" and "Plots" folders are not created automatically; call `activate_linkdir()` or `activate_visdir()` to create them.

<br><br>

### animl.build_file_manifest(image_dir, exif=True, out_file=None, ...)
{: #build_file_manifest}

Traverse a directory and find image/video files and gather metadata.

To correctly adjust timestamps from exif data, the argument `data_timezone` should be set to the timezone in which the data was collected. 
If you are unsure of the timezone, you can list all with zoneinfo.available_timezones() to find the best match, or leave as None to default to the local timezone.


| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir`       | str   | required | Path to image/video directory |
| `exif`            | bool  | True  | Returns date and time info from exif data |
| `out_file`        | str   | None  | File path to which the dataframe should be saved |
| `data_timezone`   | str   | None  | Timezone of the data, e.g., 'UTC', 'America/New_York', defaults to local timezone if None|
| `station_depth`   | int   | None  | Depth of station directory from the image_dir root in file path, if applicable.* |
| `camera_depth`    | int   | None  | Depth of camera directory from the image_dir root in file path, if applicable.* |
| `recursive`       | bool  | True  | Recursively search through all child directories |

**Returns:** `pandas.DataFrame` — object containing file manifest

Output manifest will have the following columns:
* filepath
* filename
* extension
* width
* height
* createdate (if exif = True)
* filemodifydate (if exif = True)
* datetime (if exif = True, contains createdate or filemodifydate as a fallback)
* station (if station_depth is not None)
* camera (if camera_depth is not None)

\* For `station_depth`, if file paths are in the format "image_dir/station/date/file.jpg",
`station_depth` would be 1 (0 indexed). If None, station column will not be created.
Likewise for `camera_depth`, if file paths are in the format "image_dir/station/camera/date/file.jpg",
`camera_depth` would be 2 (0 indexed). If None, camera column will not be created.


<br><br>
  
### animl.active_times(manifest, file_col="filepath", camera_depth=0, timestamp_col="datetime")
{: #active_times}

Get start and stop dates for each camera folder.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`        | pandas DataFrame  | required      | File manifest dataframe with file paths and timestamps |
| `file_col`        | str               | "filepath"    | Column in manifest to use for file paths, defaults to "filepath" |
| `camera_depth`    | int               | 0             | Directory depth from which to split cameras, with 0 being the root of the manifest_dir |
| `timestamp_col`   | str               | "datetime"    | Column name representing the timestamp in format "%Y-%m-%d %H:%M:%S", defaults to "datetime" |

**Returns:** `pandas.DataFrame` with a row for each camera and the earliest and latest timestamp of data taken at that camera

<br><br>
  
### animl.sequence_calculation(manifest, station_col, sort_columns=None, file_col="filepath", timestamp_col="datetime", maxdiff=60)
{: #sequence_calculation}

Simple sequence calculation based on time differences between consecutive images from the same station. <br>
Unlike sequence_classification(), does not apply any classification or labeling to the sequences.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`        | pandas DataFrame  | required      | File manifest dataframe with file paths and timestamps |
| `station_col`     | str               | required      | Column name in the DataFrame representing the station or camera |
| `sort_columns`    | list[str]         | None          | List of columns to sort by before calculating sequences. Defaults to None, which sorts by `station_col` and `timestamp_col` |
| `file_col`        | str               | "filepath"    | Column in manifest to use for file paths, defaults to `"filepath"` |
| `timestamp_col`   | str               | "datetime"    | Column name representing the timestamp in format "%Y-%m-%d %H:%M:%S", defaults to "datetime" |
| `maxdiff`         | int               | 60            | Maximum time difference in seconds between consecutive images to be considered part of the same sequence |

**Returns:** `pandas.DataFrame` — the input DataFrame with an additional 'sequence' column indicating sequence membership.

<br><br>  

### animl.extract_frames(manifest, frames=5, fps=None, out_file=None, out_dir=None, file_col="filepath", parallel=True, num_workers=NUM_THREADS, quiet=False)
{: #extract_frames}

Extract frames from video files in a given DataFrame.
Can sample frames based on a specified number of frames or frames per second (fps).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | pandas DataFrame  | required      | File manifest dataframe with file paths and timestamps |
| `frames`      | int               | 5             | Number of frames to sample from each video (default is 5) |
| `fps`         | int               | None          | Frames per second to sample from each video. If specified, overrides frames |
| `out_file`    | str               | None          | Path to save the extracted frames manifest as a .csv file |
| `out_dir`     | str               | None          | Directory to save extracted frame images. If None, frames are not saved as images |
| `file_col`    | str               | "filepath"    | Column in manifest to use for file paths, defaults to "filepath" |
| `parallel`    | bool              | True          | Toggle to use multiprocessing for frame extraction (default is True) |
| `num_workers` | int               | NUM_THREADS   | Number of worker processes to use for parallel processing (default is NUM_THREADS) |
| `quiet`       | bool              | False         | Suppress progress bars and print statements |

**Returns:** `pandas.DataFrame` — the input dataframe with and additional "frame" column. The value of frame is 0 for images,
             while videos will now be represented with multiple rows as indicated by frames or fps, with each row containing the sampled frame number.
  
<br><br>

---
## Detection
{: #detection}
<br>

### animl.load_detector(model_path, model_type, device=None)
{: #load_detector}

Loads a detector model from a file path.

Model types accepted: <br>
["mdv5", "mdv6", "mdv1000-cedar", "mdv1000-larch", "mdv1000-sorrel",
"mdv1000-redwood", "mdv1000-spruce", "yolov5", "yolo", "onnx"] <br>
For yolo models v6+, use "yolo", for v5, use "yolov5". For MDv1000 models, specify the version (cedar, larch, sorrel, redwood, spruce).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path`  | str | required | Path to model file |
| `model_type`  | str | required | Type of model|
| `device`      | str | None    | Device to run model on, i.e. `"cpu"` or `"cuda"` |

**Returns:** loaded model object
  
<br><br>

### animl.detect(detector, image_file_names, resize_width, resize_height, ...)
{: #detect}

Runs a detector model on batches of image files.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `detector`                | object                    | required | Preloaded detector model |
| `image_file_names`        | str / list / DataFrame    | required | Single image path, list of paths, or manifest DataFrame |
| `resize_width`            | int                       | required | Width to resize images to |
| `resize_height`           | int                       | required | Height to resize images to |
| `letterbox`               | bool                      | True | Resize and pad to preserve aspect ratio |
| `category_map`            | dict                      | MD_LABELS | Mapping of category IDs to human-readable labels |
| `confidence_threshold`    | float                     | 0.1 | Minimum confidence score to retain a detection |
| `file_col`                | str                       | "filepath" | Column name in manifest containing file paths |
| `batch_size`              | int | 1 | Number of images per batch |
| `num_workers`             | int | 1 | Number of dataloader workers |
| `device`                  | str | None | Device to run inference on: `"cpu"` or `"cuda"` |
| `checkpoint_path`         | str | None | Path to save intermediate checkpoint JSON. Checkpoint will be saved after every N batches as specified by checkpoint_frequency. |
| `checkpoint_frequency`    | int | -1 | Save checkpoint every N batches; -1 disables checkpointing |

**Returns:** `tuple` — (`detections`, `failed_files`)  
- `detections`: list[dict] of detection results in MegaDetector format, one dict per image  
- `failed_files`: list of files that failed to load during processing (if any)  

<br><br>
  
### animl.parse_detections(results, manifest=None, out_file=None, threshold=0, file_col="filepath")
{: #parse_detections}

Converts detector output into a detections DataFrame.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results`     | list[dict] or tuple | required      | Detector output dicts or tuple of (output dicts, failed files) as returned by `detect()`. Failed files are excluded from the results |
| `manifest`    | DataFrame     | None          | Original file manifest, if not None, merge MD predictions automatically |
| `out_file`    | str           | None          | Path to save detections .csv |
| `threshold`   | float         | 0             | Minimum confidence score; detections below are not returned |
| `file_col`    | str           | "filepath"    | Column name containing file paths, will merge results to manifest on this column |

**Returns:** `pandas.DataFrame` — one row per detection with columns:
             `filepath`, `category`, `category_label`, `conf`, `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`, `max_detection_conf`

<br><br>

### animl.get_animals(manifest)
{: #get_animals}

Pulls out MD animal detections for classification

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | DataFrame | required | DataFrame containing one row for every MD detection |

**Returns:** `pandas.DataFrame` — subset of manifest containing only animal detections

<br><br>

### animl.get_empty(manifest)
{: #get_empty}

Pulls out MD non-animal detections and adds prediction and confidence columns

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | DataFrame | required | DataFrame containing one row for every MD detection |

**Returns:** `pandas.DataFrame` — subset of manifest containing empty, vehicle and human detections
        with added prediction and confidence columns

<br><br>

---
## Classification
{: #classification}
<br>

### animl.load_classifier(model_path, classes, device=None, architecture="efficientnet_v2_m", quiet=True)
{: #load_classifier}

Creates and loads a classifier model of the given architecture from disk, with the associated class list.

| Parameter      | Type  | Default                | Description                                                      |
|----------------|--------|------------------------|------------------------------------------------------------------|
| `model_path`   | str    | required               | File or directory path to the model weights                      |
| `classes`      | int \| str \| Path \| pd.DataFrame     | required               | Number of classes, class list file, or DataFrame                 |
| `device`       | str    | None                   | Device to load model on ("cpu" or "cuda")                        |
| `architecture` | str    | "efficientnet_v2_m"    | Expected architecture name for .pt weights ("efficientnet_v2_m", "convnext_base" or "bioclip_2"). Ignored for .pth and .onnx models |
| `quiet`        | bool                                   | True                   | Toggles suppression of device info messages                       |

**Returns:** `(model, class_list)` — loaded model (of given architecture) and class list DataFrame, or `None` if `classes` is an int.
For .onnx models with a `class_dict` in their metadata, the class list is read from the model.

<br><br>
  
### animl.load_class_list(classlist_file)
{: #load_class_list}

Returns classlist file as DataFrame.

| Parameter         | Type   | Default | Description                 |
|-------------------|--------|---------|-----------------------------|
| `classlist_file`  | str    | required| File path to class list CSV |

**Returns:** `pandas.DataFrame` — the class list file data  

<br><br>


### animl.class_list_to_dict(class_list, id_col="id", class_col="class")
{: #class_list_to_dict}

Converts a classification or detection class list DataFrame into a dictionary mapping class IDs to labels.

| Parameter      | Type            | Default      | Description                                                  |
|----------------|-----------------|--------------|--------------------------------------------------------------|
| `class_list`   | pd.DataFrame    | required     | DataFrame containing class list with at least label and id columns |
| `id_col`       | str             | "id"         | Column name in class_list DataFrame containing class IDs (integers corresponding to model output indices) |
| `class_col`    | str             | "class"      | Column name in class_list DataFrame containing class labels   |

**Returns:** `dict` — mapping of class IDs to labels, e.g. {0: "empty", 1: "species_a", 2: "species_b"}

<br><br>

### animl.classify(model, detections, resize_width=480, resize_height=480, file_col="filepath", ...)
{: #classify}

Runs prediction for input detections using a preloaded classifier model

| Parameter      | Type            | Default      | Description                                                  |
|----------------|-----------------|--------------|--------------------------------------------------------------|
| `model`        | nn.Module       | required     | Preloaded classifier model                                   |
| `detections`   | DataFrame/list/str | required | Animal detections: can be DataFrame, list of filepaths, or a filepath string |
| `resize_width` | int             | 480          | Image width input size (pixels)                              |
| `resize_height`| int             | 480          | Image height input size (pixels)                             |
| `file_col`     | str             | "filepath"   | Column name for file paths                                   |
| `crop`         | bool            | True         | Whether to crop images based on bounding boxes               |
| `normalize`    | bool            | True         | Normalize tensors before inference                           |
| `batch_size`   | int             | 1            | Data generator batch size                                    |
| `num_workers`  | int             | NUM_THREADS  | Number of workers (CPU threads or processes)                 |
| `device`       | str             | None         | Device for inference ("cpu" or "cuda")                       |
| `out_file`     | str             | None         | Output file path to save prediction results                  |

**Returns:** `tuple` — (`predictions`, `failed_rows`)  
- `predictions`: `np.array` of softmaxed logits for each class, one row per detection that loaded  
- `failed_rows`: list of positions (in `detections`) of rows that failed to load (if any)  

If `out_file` already exists, the user is prompted to load the saved predictions instead of re-running the model.
Raises a `ValueError` if none of the detections could be loaded.
  
<br><br>

### animl.single_classification(animals, empty, predictions_output, class_list, best=False, count=False, file_col="filepath", failed_rows=None)
{: #single_classification}

Assigns the maximum likelihood class label and confidence to each row in a detection DataFrame, handling rows that failed to load and "empty" detections.

| Parameter          | Type                          | Default     | Description                                                         |
|--------------------|-------------------------------|-------------|---------------------------------------------------------------------|
| `animals`          | pd.DataFrame                  | required    | Detections with animals (from manifest)                             |
| `empty`            | pd.DataFrame or None          | required    | Detections with no animals (from manifest); pass `None` if there are none |
| `predictions_output`| np.array or tuple            | required    | Softmaxed logits, or (logits, failed_rows), i.e. the output of `classify()` |
| `class_list`       | list or pd.Series             | required    | List/series of class labels                                         |
| `best`             | bool                          | False       | If True, returns best prediction for each file only                 |
| `count`            | bool                          | False       | If True, adds a count column with the number of detections of each species per file |
| `file_col`         | str                           | "filepath"  | Column for file paths                                               |
| `failed_rows`      | list or None                  | None        | Positions in `animals` of rows that failed to load during classification, when `predictions_output` is only the predictions |

**Returns:** `pandas.DataFrame` — DataFrame with columns `prediction`, `confidence`, and associated metadata  

<br><br>
  
### animl.sequence_classification(animals, empty, predictions_output, class_list, station_col, empty_class="", ...)
{: #sequence_classification}

Applies class labels to images based on sequential information.

This function applies image classifications at a sequence level by leveraging information from
multiple images. A sequence is defined as all images at the same camera and station where the
time between consecutive images is <=maxdiff. This can improve classification accuracy, but
assumes that only one species is present in each sequence. If you regularly expect multiple
species to occur in an image or sequence don't use this function.

| Parameter        | Type                    | Default    | Description                                                                |
|------------------|-------------------------|------------|----------------------------------------------------------------------------|
| `animals`        | pd.DataFrame            | required   | Detections with animals                                                    |
| `empty`          | pd.DataFrame or None    | required   | Detections with no animals; pass `None` if there are none                  |
| `predictions_output`| np.array or tuple    | required   | Softmaxed logits, or (logits, failed_rows), i.e. the output of `classify()` |
| `class_list`     | pd.Series               | required   | Class labels associated with classifier model, e.g. `class_list['class']`   |
| `station_col`    | str                     | required   | Column indicating station/camera                                            |
| `empty_class`    | str                     | ""         | Value of "empty" label in class list, empty string if not specified         |
| `sort_columns`   | list[str] or None       | None       | Columns to sort groups by, if not specified defaults to `station_col` and `timestamp_col` |
| `file_col`       | str                     | "filepath" | Column indicating image file paths                                         |
| `timestamp_col`  | str                     | "datetime" | Column with file timestamps                                           |
| `failed_rows`    | list or None            | None       | Positions in `animals` of rows that failed to load during classification, when `predictions_output` is only the predictions |
| `maxdiff`        | int                     | 60         | Maximum time (sec) separating images in the same burst/sequence            |

**Returns:** `pandas.DataFrame` — sequence-classified results with columns including `prediction`, `confidence`, `sequence`

<br><br>
  
---
## Re-Identification
{: #re-id}
<br>

### animl.load_miew(file_path, device=None)
{: #load_miew}

Loads a MiewID model from a file path.

| Parameter      | Type            | Default      | Description                                                  |
|----------------|-----------------|--------------|--------------------------------------------------------------|
| `file_path`    | str             | required     | file path to model file                                      |
| `device`       | str             | None         | Device for inference ("cpu" or "cuda")                       |

**Returns:** MiewID model object 

<br><br>

### animl.extract_miew_embeddings(miew_model, manifest, file_col="filepath", batch_size=1, num_workers=1, device=None)
{:#extract_miew_embeddings}

Extracts MiewID embeddings for a given set of images.

| Parameter      | Type            | Default      | Description                                                  |
|----------------|-----------------|--------------|--------------------------------------------------------------|
| `miew_model`   | model object    | required     | MiewID model object                                          |
| `manifest`     | pd.DataFrame    | required     | DataFrame with columns 'filepath', 'emb_id'                 |
| `file_col`     | str             | "filepath"   | Column indicating image file paths                           |
| `batch_size`   | int             | 1            | Data generator batch size                                    |
| `num_workers`  | int             | 1            | Number of workers (CPU threads or processes)                 |
| `device`       | str             | None         | Device for inference ("cpu" or "cuda")                       |

**Returns:** `numpy.ndarray` — one embedding per manifest row; rows whose image failed to load are filled with NaN.
Raises a `ValueError` if none of the images could be loaded.

<br><br>

### animl.remove_diagonal(A)
{:#remove_diagonal}

Removes the diagonal elements from a square matrix.

| Parameter      | Type            | Default      | Description                                                  |
|----------------|-----------------|--------------|--------------------------------------------------------------|
| `A`            | torch.Tensor    | required     | Input square matrix                                          |

**Returns:** `torch.Tensor` - Matrix with diagonal elements removed

<br><br>

### animl.euclidean_squared_distance(input1, input2)
{:#euclidean_squared_distance}

Computes the Euclidean squared distance between two feature matrices.

| Parameter      | Type            | Default      | Description                                                  |
|----------------|-----------------|--------------|--------------------------------------------------------------|
| `input1`       | torch.Tensor    | required     | 2-D feature matrix                                           |
| `input2`       | torch.Tensor    | required     | 2-D feature matrix                                           |

**Returns:** `torch.Tensor` - Euclidean squared distance matrix

<br><br>

### animl.cosine_distance(input1, input2)
{:#cosine_distance}

Computes the cosine distance between two feature matrices.

| Parameter      | Type            | Default      | Description                                                  |
|----------------|-----------------|--------------|--------------------------------------------------------------|
| `input1`       | torch.Tensor    | required     | 2-D feature matrix                                           |
| `input2`       | torch.Tensor    | required     | 2-D feature matrix                                           |

**Returns:** `torch.Tensor` - Cosine distance matrix

<br><br>

### animl.compute_distance_matrix(input1, input2, metric='euclidean')
{:#compute_distance_matrix}

Computes a distance matrix between two feature matrices using the specified metric.

| Parameter      | Type            | Default      | Description                                                  |
|----------------|-----------------|--------------|------------------------------------------------------|
| `input1`       | torch.Tensor or np.ndarray | required | 2-D feature matrix                                           |
| `input2`       | torch.Tensor or np.ndarray | required | 2-D feature matrix                                           |
| `metric`       | str             | 'euclidean'  | Distance metric: "euclidean" or "cosine"                   |

**Returns:** `numpy.ndarray` - Distance matrix

<br><br>

### animl.compute_batched_distance_matrix(input1, input2, metric='cosine', batch_size=10)
{:#compute_batched_distance_matrix}

Computes a distance matrix between two feature matrices in batches, using the specified metric.
This is useful for large datasets that may not fit in memory when computing the full distance matrix at once.

| Parameter      | Type            | Default      | Description                                                  |
|----------------|-----------------|--------------|--------------------------------------------------------------|
| `input1`       | np.ndarray or torch.Tensor | required | 2-D array of query features                                  |
| `input2`       | np.ndarray or torch.Tensor | required | 2-D array of database features                               |
| `metric`       | str             | 'cosine'     | Distance metric (e.g., 'euclidean', 'cosine')               |
| `batch_size`   | int             | 10           | Number of rows from input1 to process at a time              |

**Returns:** `numpy.ndarray` - Computed distance matrix

<br><br>

### animl.predict_viewpoints(model, manifest, device=None)
{:#predict_viewpoints}

Runs a viewpoint model on two-camera sequences, combining the predictions from each camera to improve viewpoint accuracy.

| Parameter      | Type            | Default      | Description                                                  |
|----------------|-----------------|--------------|--------------------------------------------------------------|
| `model`        | model object    | required     | Viewpoint model object                                       |
| `manifest`     | pd.DataFrame    | required     | DataFrame with columns 'filepath', 'camera' and 'sequence'   |
| `device`       | str             | None         | Device for inference ("cpu" or "cuda"); selected automatically if None |

**Returns:** `tuple` — (`pred_labels`, `filepaths`): the predicted viewpoint for each image and the corresponding file paths

<br><br>

---
## Counting
{: #counting}
<br>

### animl.count_detections(detections, station_col="station", confidence_threshold=0.5, iou_threshold=0.9, maxdiff=60, max_n=None, count_type="max", classes=None)
{: #count_detections}

Counts detections per category across image sequences.
Sequences are calculated with `sequence_calculation()`, so `detections` needs a `datetime` column as well as `station_col`.
Within each image, overlapping detections are removed with `deduplicate()`, then the per-image counts are combined for each sequence.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `detections` | pd.DataFrame | required | Output of `parse_detections()` merged with a manifest containing station and timestamp columns |
| `station_col` | str | "station" | Column name representing the station or camera |
| `confidence_threshold` | float | 0.5 | Minimum confidence (`conf`) for a detection to be counted |
| `iou_threshold` | float | 0.9 | IoU above which two detections in the same image are considered the same animal |
| `maxdiff` | int | 60 | Maximum time difference in seconds between consecutive images in a sequence |
| `max_n` | int or None | None | Maximum number of images per sequence to consider; all if None |
| `count_type` | str | "max" | How to combine per-image counts in a sequence: "max" or "mean" |
| `classes` | list or None | None | Values of `category_label` to count; if None, all labels except "empty" |

**Returns:** `pandas.DataFrame` — one row per sequence, with a `sequence` column and a count column for each class

<br><br>

### animl.deduplicate(image_detections, iou_threshold=0.9)
{: #deduplicate}

Removes duplicate detections within a single image. When two detections overlap above the IoU threshold, the one with higher confidence is kept.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_detections` | pd.DataFrame | required | Detections for a single image, with `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` and `conf` columns |
| `iou_threshold` | float | 0.9 | IoU above which two detections are considered the same animal |

**Returns:** `pandas.DataFrame` — deduplicated detections

<br><br>

---
## Model Training
{: #training}
<br>

### animl.train_classifier(cfg)
{:#train_classifier}

Trains a classifier model based on the provided configuration.
For details on the configuration parameters, see the [config README](https://github.com/conservationtechlab/animl-py/blob/main/src/animl/config/README.md).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cfg` | str | required | Path to config yml file containing training parameters and data paths |

<br><br>

### animl.test_classifier(cfg)
{:#test_classifier}

Tests a classifier model based on the provided configuration, evaluating performance on a test dataset and generating a confusion matrix.
For details on the configuration parameters, see the [config README](https://github.com/conservationtechlab/animl-py/blob/main/src/animl/config/README.md).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cfg` | str | required | Path to config yml file containing training parameters and data paths |

<br><br>

### animl.save_classifier(model, out_dir, epoch, stats, optimizer=None, scheduler=None, scaler=None)
{:#save_classifier}

Saves model state weights and optional optimizer/scheduler/scaler states to disk.

| Parameter   | Type             | Default   | Description                                                  |
|-------------|------------------|-----------|--------------------------------------------------------------|
| `model`     | torch.nn.Module  | required  | The PyTorch model instance to save                           |
| `out_dir`   | str              | required  | Directory path where model weights will be saved             |
| `epoch`     | int              | required  | Current training epoch (used as filename)                    |
| `stats`     | dict             | required  | Training/validation stats/metrics to save with the model     |
| `optimizer` | torch.optim.Optimizer | None      | (Optional) Optimizer state to save                      |
| `scheduler` | torch.optim.lr_scheduler._LRScheduler | None      | (Optional) Scheduler state to save      |
| `scaler`    | torch.amp.GradScaler | None      | (Optional) GradScaler state to save, for mixed-precision training |

**Returns:** `None`  

<br><br>

### animl.load_classifier_checkpoint(model_path, model, optimizer, scheduler, scaler, device)
{:#load_classifier_checkpoint}

Loads the latest checkpoint to resume model training, restoring weights and optimizer/scheduler/scaler states.

| Parameter    | Type                      | Default   | Description                                               |
|--------------|---------------------------|-----------|-----------------------------------------------------------|
| `model_path` | str or Path               | required  | Path containing saved model `.pt` checkpoints             |
| `model`      | torch.nn.Module           | required  | Model object to load weights into                         |
| `optimizer`  | torch.optim.Optimizer     | required  | Optimizer object to load state into                       |
| `scheduler`  | torch.optim.lr_scheduler._LRScheduler | required  | Scheduler to load state into               |
| `scaler`     | torch.amp.GradScaler or None | required  | GradScaler to load state into, or `None` if not using mixed precision |
| `device`     | str                       | required  | Device to map tensors onto ("cpu" or "cuda")              |

**Returns:** `int` — starting epoch restored from the latest checkpoint  

<br><br>

---
## Visualization
{: #visualization}
<br>

### animl.get_frame_as_image(video_path, frame=0)
{:#get_frame_as_image}

Given a video path, return a specific frame as an RGB image.

| Parameter      | Type | Default   | Description                 |
|----------------|------|-----------|-----------------------------|
| `video_path`   | str  | required  | File path to video                    |
| `frame`        | int  | 0         | Frame number to extract, default is 0 |

**Returns:** `numpy.ndarray` - Matrix representing the RGB image

<br><br>

### animl.plot_box(rows, file_col="filepath", min_conf=0, classifier_label_col=None, detector_category_col="category", show_confidence=False,...)
{:#plot_box}

Plot bounding box(es) for a single image based on the input rows of a DataFrame.

`plot_box()` is designed for plotting boxes on a single image, while `plot_all_bounding_boxes()` can handle multiple images and has additional options for saving outputs.

`rows` must contain the bounding box coordinates (`bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`), and filepath (`filepath`) for the image to be plotted. 

If `classifier_label_col` is specified, it will also display the predicted class label on the box. The color of the box(es) can be determined by the detector category column specified by `detector_category_col` and the `colors` dictionary. 

If `show_confidence` is True, `rows` must contain `confidence` or `conf` column, and the confidence score will also be displayed on the box.


| Parameter | Type | Default | Description |
|---|---|---|---|
| `rows`                     | pandas.DataFrame or pandas.Series | required | manifest rows to plot bounding boxes for |
| `file_col`                 | str | "filepath" | Column name containing file paths |
| `min_conf`                 | float | 0 | Minimum confidence threshold to display a bounding box |
| `classifier_label_col` | str or None | None | Column name containing classifier labels to display on boxes, if applicable |
| `detector_category_col` | str | "category" | Column name containing detector category (e.g., 'category') to determine box color |
| `show_confidence` | bool | False | If true, show confidence score on box |
| `colors` | dict | MD_COLORS | Dictionary mapping class labels to BGR color tuples for the bounding boxes |
| `detector_labels` | dict | MD_LABELS | Dictionary mapping detector categories to human-readable labels |
| `return_img` | bool | False | If true, return the plotted image as a numpy array instead of displaying it with `cv2.imshow` |

**Returns:** None or `numpy.ndarray` (if `return_img` is True)

<br><br>

### animl.plot_all_bounding_boxes(manifest, out_dir, file_col="filepath", min_conf=0, classifier_label_col=None, detector_category_col="category", show_confidence=False,...)
{:#plot_all_bounding_boxes}

Plot bounding boxes for all rows in a manifest DataFrame and save the plotted images.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`                 | pd.DataFrame | required | DataFrame containing detection results and file paths |
| `out_dir`                  | str | required | Directory to save plotted images with bounding boxes |
| `file_col`                 | str | "filepath" | Column name containing file paths |
| `min_conf`                 | float | 0 | Minimum confidence threshold to display a bounding box |
| `classifier_label_col` | str or None | None | Column name containing classifier labels to display on boxes, if applicable |
| `detector_category_col` | str | "category" | Column name containing detector category (e.g., 'category') to determine box color |
| `show_confidence` | bool | False | If true, show confidence score on box |
| `colors` | dict or None | None | Dictionary mapping detector category labels to BGR color tuples for the bounding boxes; if None, uses MD_COLORS |
| `detector_labels` | dict or None | None | Dictionary mapping detector categories to human-readable labels; if None, uses MD_LABELS |

**Returns:** None 

<br><br>

---
## Export
{: #export}
<br>

### animl.export_folders(manifest, out_dir, out_file=None, label_col="prediction", file_col="filepath", timestamp_col="datetime", ...)
{: #export_folders}

Sorts images into one folder per label within `out_dir`, for browsing and human review.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | pd.DataFrame | required | DataFrame containing file paths and labels |
| `out_dir`     | str or Path  | required | Root directory for the label folders |
| `out_file`    | str or None  | None     | Optional file path to save a .csv of the returned manifest |
| `label_col`   | str          | "prediction" | Column name in manifest containing labels to use for folder names, e.g. `"category_label"` for detection categories or `"prediction"` for species |
| `file_col`    | str          | "filepath" | Column name in manifest containing file paths |
| `timestamp_col` | str        | "datetime" | Column name in manifest containing timestamps in format "%Y-%m-%d %H:%M:%S" |
| `station_col` | str          | "station"  | Column name in manifest containing station identifiers |
| `unique_name` | str          | "uniquename" | Column name in manifest to use for unique file names in exported folders; if not in manifest, they will be created from `station_col`, `timestamp_col` and the original file name |
| `copy`        | bool         | False    | If True, files are hard copied to the new folders; if False, they are hard linked |

**Returns:** `pandas.DataFrame` — copy of manifest with additional column `link` containing the exported file paths, and the
`unique_name` column if it was created

<br><br>

### animl.remove_link(manifest, link_col="link")
{: #remove_link}

Deletes the linked/copied images created by `export_folders()` or `export_camtrapR()`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | pd.DataFrame | required | DataFrame containing a column with file paths to remove |
| `link_col`    | str          | "link"   | Column name in manifest containing file paths to remove |

**Returns:** `pandas.DataFrame` — copy of manifest with  column `link_col` removed

<br><br>

### animl.update_labels_from_folders(manifest, export_dir, unique_name="uniquename")
{: #update_labels_from_folders}

Update manifest after human review of the folders created by `export_folders()`.
Each file's label is taken from the name of the folder it is in after review.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | pd.DataFrame | required | DataFrame containing file paths and labels to update |
| `export_dir`  | str          | required | Root directory containing the reviewed label folders |
| `unique_name` | str          | "uniquename" | Column name in manifest containing the unique file names, which must match the file names within `export_dir` |

**Returns:** `pandas.DataFrame` — manifest merged with a new `label` column containing the reviewed label (the folder name) for each file

<br><br>

### animl.export_train_val_test(manifest, label_col="class", file_col="filepath", conf_col="confidence", out_dir=None, val_size=0.1, test_size=0.1, seed=42, groupby_col=None)
{: #export_train_val_test}

Returns train_df, val_df, test_df with `label_col` stratified.
`test_size` and `val_size` are fractions of the whole dataset (e.g., 0.2 -> 20%), and must sum to less than 1.

Only one row is kept per file: the highest confidence detection according to `conf_col`, or the first row if `conf_col` is not in the manifest. This keeps different detections from the same image from ending up in different splits, which would lead to data leakage and overly optimistic performance estimates.

If `groupby_col` is given (e.g. a station or sequence column), the data is split so that no group appears in more than one split.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | pd.DataFrame | required | DataFrame containing file paths, labels, and confidence scores |
| `label_col`   | str          | "class"   | Column name in manifest containing class labels to stratify on |
| `file_col`    | str          | "filepath" | Column name in manifest containing file paths |
| `conf_col`    | str          | "confidence" | Column name in manifest containing confidence scores, used to keep the best detection per file |
| `out_dir`     | str or None  | None     | Directory to save `train_data.csv`, `validate_data.csv` and `test_data.csv`; if None, CSVs are not saved |
| `val_size`    | float         | 0.1      | Fraction of dataset to use for validation set (e.g., 0.1 for 10%) |
| `test_size`   | float         | 0.1      | Fraction of dataset to use for test set (e.g., 0.1 for 10%) |
| `seed`        | int           | 42       | Random seed for reproducibility of splits |
| `groupby_col` | str or None   | None     | Column containing group labels; if given, train/val/test share no groups |

**Returns:** `tuple` — (`train_df`, `val_df`, `test_df`) DataFrames for training, validation, and testing, stratified by `label_col` 

<br><br>

### animl.export_yolo(train_manifest, val_manifest, test_manifest, class_list, out_dir, label_col="prediction", file_col="filepath", hard_copy=False)
{: #export_yolo}

Export a manifest to YOLO format for model training.
Saves a .txt file for each image with bounding box coordinates and class labels.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train_manifest` | pd.DataFrame | required | DataFrame containing training samples with file paths, labels, and bounding box coordinates |
| `val_manifest`   | pd.DataFrame | required | DataFrame containing validation samples with file paths, labels, and bounding box coordinates |
| `test_manifest`  | pd.DataFrame or None | required | DataFrame containing test samples with file paths, labels, and bounding box coordinates; pass `None` to skip the test split |
| `class_list`     | pd.DataFrame  | required | DataFrame with `class` and `id` columns mapping class names to YOLO class IDs |
| `out_dir`        | str           | required | Directory to save YOLO formatted images, .txt label files and `dataset.yaml` |
| `label_col`     | str           | "prediction"   | Column name in manifest containing class labels |
| `file_col`       | str           | "filepath" | Column name in manifest containing file paths |
| `hard_copy`      | bool          | False     | If True, image files will be hard copied to the YOLO output directory; if False, they will be symlinked |

**Returns** `dict` — contents of the `dataset.yaml` saved to `out_dir`: paths to the YOLO formatted splits, number of classes, and class names, e.g.:
```python
{
    "path": "path/to/yolo/",
    "train": "path/to/yolo/images/train",
    "val": "path/to/yolo/images/val",
    "test": "path/to/yolo/images/test",   # None if test_manifest is None
    "nc": 3,
    "names": ["empty", "species_a", "species_b"]
}
```

<br><br>

### animl.export_coco(manifest, class_dict, out_file, info=None, licenses=None)
{: #export_coco}

Export a manifest to COCO format.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | pd.DataFrame  | required | DataFrame containing detection results and file paths |
| `class_dict`  | dict          | required | Dictionary mapping category IDs to class names (e.g., {0: "empty", 1: "species_a", 2: "species_b"}). Each annotation's category is looked up from its `prediction`; predictions not in `class_dict` get category ID -1 |
| `out_file`    | str or Path   | required | File path to save the COCO JSON output |
| `info`        | dict or None  | None     | Optional dictionary containing dataset info to include in COCO output (e.g., {"description": "My Dataset", "version": "1.0"}) |
| `licenses`    | list of dict or None | None     | Optional list of license dictionaries to include in COCO output (e.g., [{"id": 1, "name": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}]) |

**Returns:** `dict` — COCO format dictionary containing `info`, `licenses`, `categories`, `images`, and `annotations` based on the input manifest and class_dict, and saves it to `out_file` as JSON


<br><br>

### animl.export_camtrapdp(manifest, out_dir, file_public=False, classifier_name=None)
{: #export_camtrapdp}

Export a manifest to CamtrapDP format. Saves `media.csv`, `observations.csv` and `datapackage.json` to `out_dir`.

The previous misspelled name, `export_camptrapdp`, still works but is deprecated.
Requires scientific name for the species prediction label and bounding box coordinates for each detection.
Assumes MegaDetector category labels and uses `category` column to determine which rows are "empty" vs "animal" detections.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | pd.DataFrame  | required | DataFrame containing classification results and file paths |
| `out_dir`     | str  | required | Directory to save the CamtrapDP formatted output |
| `file_public` | bool          | False     | Record whether the media files are publicly available |
| `classifier_name` | str or None | None     | Optional name of the classifier model used for predictions, to include in the output metadata |

**Returns** `tuple` - media_df, observations_df, and datapackage dict
The media_df contains metadata for each media file, the observations_df contains metadata for each observation (detection), and the datapackage dict contains the overall structure and metadata for the CamtrapDP package. 

<br><br>

### animl.export_camtrapR(manifest, out_dir, out_file=None, label_col='prediction', file_col='filepath', ... )
{: #export_camtrapR}

Export into species-labeled folders organized by station.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | pd.DataFrame  | required      | DataFrame containing classification results and file paths |
| `out_dir`     | str           | required      | Directory to export sorted images |
| `out_file`    | str or None   | None          | Optional file path to save a .csv manifest of the exported data |
| `label_col`   | str           | "prediction"  | Column name in manifest species labels to use for folder names |
| `file_col`    | str           | "filepath"    | Column name in manifest containing file paths |
| `timestamp_col` | str         | "datetime" | Column name in manifest containing timestamps |
| `station_col` | str           | "station"     | Column name in manifest containing station identifiers |
| `unique_name` | str           | "uniquename" | Column name in manifest to use for unique file names in exported folders; if not in manifest, they will be created from `station_col` and `timestamp_col` |
| `copy` | bool | False | If True, files are hard copied to the new folders; if False, they are hard linked |

**Returns** `pandas.DataFrame` — copy of manifest with additional column `link` for exported file paths

<br><br>

### animl.export_timelapse(manifest, out_dir, only_animal=True)
{: #export_timelapse}

Converts a manifest to .csv files that contain the columns needed for TimeLapse conversion.
Saves `animals.csv` (and `non-animals.csv` if `only_animal` is False) plus the full `manifest.csv` to `out_dir`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | pd.DataFrame  | required      | DataFrame containing classification results and file paths |
| `out_dir`     | str           | required      | Directory to save the TimeLapse formatted output |
| `only_animal` | bool          | True          | Whether to export only animal detections (True) or also non-animal detections (False) |

**Returns** `pathlib.Path` — path to the saved `manifest.csv`, for use with `export_megadetector()`

<br><br>

### animl.export_megadetector(manifest, out_file=None, detector="MegaDetector v5a", prompt=True)
{: #export_megadetector}

Converts a manifest DataFrame back into MegaDetector format and saves as a .json file.

If `out_file` is None, the output is saved to `detections.json` in the current directory.

Author: Dan Morris https://github.com/agentmorris/MegaDetector/tree/main

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest` | pd.DataFrame | required | DataFrame containing images and associated detections |
| `out_file` | str or None | None | Path to save the MD formatted file |
| `detector` | str | "MegaDetector v5a" | Name of the detector used |
| `prompt` | bool | True | Whether to prompt before overwriting existing file |

**Returns:** `None` — saves a .json file in MegaDetector format

<br><br>

---
## File Input/Output
{: #file-io}
<br>

### animl.save_data(data, out_file, prompt=True, index=False)
{: #save_data}

Save a DataFrame to a .csv file.


| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | pd.DataFrame | required | DataFrame containing data to be saved |
| `out_file` | str or Path | required | Full path to save the data, must include .csv extension |
| `prompt` | bool | True | Whether to prompt before overwriting existing file |
| `index` | bool | False | If True, saves the DataFrame index in the output file |

<br><br>

### animl.load_data(file)
{: #load_data}

Load a .csv file into a DataFrame.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | Union[Path, str] | required | Full path of the file to load |

**Returns:** `pandas.DataFrame`

<br><br>

### animl.save_json(data, out_file, prompt=True)
{: #save_json}

Save a dictionary as a JSON file.


| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | dict | required | Dictionary containing data to be saved |
| `out_file` | str | required | Full path to save the JSON file |
| `prompt` | bool | True | Whether to prompt before overwriting existing file |

<br><br>

### animl.load_json(file)
{: #load_json}

Load data from a JSON file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | Union[Path, str] | required | Full path of the JSON file to load |

**Returns:** `dict`

<br><br>

### animl.save_yaml(data, out_file, prompt=True)
{: #save_yaml}

Save a dictionary as a YAML file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | dict | required | Dictionary containing data to be saved |
| `out_file` | str or Path | required | Full path to save the YAML file |
| `prompt` | bool | True | Whether to prompt before overwriting existing file |

<br><br>

### animl.load_yaml(file)
{: #load_yaml}

Load data from a YAML file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | Union[Path, str] | required | Full path of the YAML file to load |

**Returns:** `dict`

<br><br>

### animl.save_failed_rows(manifest, failed_rows, out_file, file_col="filepath")
{: #save_failed_rows}

Save the manifest rows that failed to load (e.g. during classification), or remove an old file if none failed.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest` | pd.DataFrame | required | Manifest passed to the dataloader |
| `failed_rows` | list | required | Positions in `manifest` of the rows that failed to load |
| `out_file` | str or Path | required | Full path to save file to |
| `file_col` | str | "filepath" | Column name containing file paths |

**Returns:** `None`

<br><br>

### animl.load_failed_rows(file)
{: #load_failed_rows}

Load the positions of manifest rows that failed to load, as saved by `save_failed_rows()`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | Union[Path, str] | required | Full path to the failed rows file |

**Returns:** `list` — positions of the rows that failed to load; empty if the file does not exist

<br><br>

### animl.check_file(file, output_type=None)
{: #check_file}

Check for file existence and prompt user if they want to load.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | Union[Path, str] | required | Full path of the file to check |
| `output_type` | Union[Path, str] | None | Type of output file (e.g., "Manifest", "Detections") for prompt to user|

**Returns:** `bool` — True if file exists and user wants to load, False otherwise

<br><br>

---
## Utilities
{: #utilities}
<br>

### animl.list_models()
{: #list_models}

Prints the names of the MegaDetector models, SDZWA classifier models and classifier class lists available for download.
The download URLs are stored in the dictionaries `animl.MEGADETECTOR`, `animl.CLASSIFIER` and `animl.CLASS_LIST`.

**Returns:** `None`

<br><br>

### animl.download_model(model_url, out_dir="models")
{: #download_model}

Download a model to the given directory.

```python
animl.download_model(animl.MEGADETECTOR['MDV5a'], out_dir='models')
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_url` | str | required | URL of the model to download, e.g. a value from `animl.MEGADETECTOR`, `animl.CLASSIFIER` or `animl.CLASS_LIST` |
| `out_dir` | str | "models" | Directory to save the model to; created if it does not exist |

**Returns:** `None`

<br><br>

### animl.get_torch_device(user_set=None, quiet=False)
{: #get_torch_device}

Get the PyTorch device to run on. Uses CUDA if available, unless the user requests otherwise.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `user_set` | str or None | None | Requested device, e.g. `"cpu"`, `"cuda"` or `"cuda:1"`; if None, selected automatically |
| `quiet` | bool | False | Suppress the device info message |

**Returns:** `torch.device`

<br><br>

### animl.get_onnx_device(user_set=None, quiet=False)
{: #get_onnx_device}

Get the ONNX Runtime execution providers to run on. Uses CUDA if available, unless the user requests otherwise.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `user_set` | str or None | None | Requested device, e.g. `"cpu"`, `"cuda"` or `"cuda:1"`; if None, selected automatically |
| `quiet` | bool | False | Suppress the device info message |

**Returns:** `list` — ONNX Runtime execution providers

<br><br>

### animl.check_torch_cuda(), animl.check_onnx_cuda(), animl.check_exiftool()
{: #checks}

Check whether CUDA is available to PyTorch, whether CUDA is available to ONNX Runtime, and whether ExifTool is installed and accessible, respectively.

**Returns:** `bool` for the CUDA checks; for `check_exiftool()`, the ExifTool version string, or `False` if it could not be run.

<br><br>

### animl.get_version()
{: #get_version}

**Returns:** `str` — the installed animl-py version

<br><br>

### animl.init_seed(seed)
{: #init_seed}

Initialize the seed for all random number generators, for reproducible results.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seed` | int | required | Seed for the random number generators |

<br><br>

### animl.get_iou(bb1, bb2)
{: #get_iou}

Calculates the intersection over union (IoU) of two bounding boxes.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bb1` | list | required | Bounding box as [x_min, y_min, width, height] |
| `bb2` | list | required | Bounding box as [x_min, y_min, width, height] |

**Returns:** `float` — IoU in [0, 1]

<br><br>

### Constants
{: #constants}

| Name | Value |
|---|---|
| `NUM_THREADS` | Default number of worker processes: CPU count minus one, between 1 and 8 |
| `MD_LABELS` | `{0: 'empty', 1: 'animal', 2: 'human', 3: 'vehicle'}` |
| `MD_COLORS` | Default BGR box colors for each MegaDetector category |
| `MEGADETECTORv5_SIZE` | Input size for MegaDetector v5 models (1280) |
| `SDZWA_CLASSIFIER_SIZE` | Input size for SDZWA classifier models (480) |
| `BIOCLIP_CLASSIFIER_SIZE` | Input size for BioCLIP models (224) |
| `MIEWID_SIZE` | Input size for MiewID models (440) |
| `IMAGE_EXTENSIONS`, `VIDEO_EXTENSIONS`, `VALID_EXTENSIONS` | File extensions recognized by `build_file_manifest()` |

<br><br>

---
# Troubleshooting