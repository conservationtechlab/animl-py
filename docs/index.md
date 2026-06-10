---
layout: default
title: Home
description: Developing open-source technology and machine learning tools for wildlife conservation and ecological research
---

# Getting Started

<br>

{: #about}
## About AniML

**Version 3.3.0**

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
   The function parse_MD will convert the json to a pandas DataFrame and merge detections with the original file manifest, if provided.

    ```python
    detector = animl.load_detector('/path/to/mdmodel.pt', model_type="mdv5", device='cuda:0')

    mdresults = animl.detect(detector,
                             allframes,
                             resize_width=animl.MEGADETECTORv5_SIZE,
                             resize_height=animl.MEGADETECTORv5_SIZE,
                             letterbox=True,
                             file_col="frame",
                             device='cuda:0',
                             checkpoint_path=working_dir.mdraw,
                             quiet=True)

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
                                     out_file=working_dir.predictions)
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
    csv_loc = animl.export_timelapse(manifest, imagedir, only_animal = True)
    animl.export_megadetector(manifest, out_file ="final_result.json", detector = 'MegaDetector v5a')
    ```

9. (OPTIONAL) Create symlinks within a given directory for file browser access.
    ```python
    manifest = animl.export_folders(manifest, out_dir=working_dir.linkdir, out_file=working_dir.results)
    ```

---
### Classifer Model Training

1. Assuming a file manifest of training data with species labels, first split the data into training, validation and test splits.
   This function splits each label proportionally by the given percentages, by default 0.7 training, 0.2 validation, 0.1 Test.

    ```python
    train, val, test, stats = animl.train_val_test(manifest,
                                                   out_dir='path/to/save/data/', 
                                                   label_col="species",
                                                   val_size: float = 0.2,
                                                   test_size: float = 0.1,
                                                   random_state: int = 42)
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
{: re-id}

---
### Exports
{: #export}


<br><br>

---
# API Reference

## Full Pipeline
{: #full-pipeline}
<br>

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

**Returns:** a dataframe containing the results of the detection and classification, including file paths, 
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

**Returns:** a dataframe containing the results of the detection and classification, including file paths, 
             detection categories, and predicted classes.

<br><br>
  
---
## Data Ingestion and Processing
{: #data-ingestion}
<br>

### class animl.WorkingDirectory(working_dir)
{: #workingdirectory}

A WorkingDirectory object creates a folder called "Animl-Directory within the working_dir. 
Attributes include output file paths to save the outputs of intermediary steps.

* self.filemanifest = "FileManifest.csv", typically used with `build_file_manifest()`
* self.imageframes = "ImageFrames.csv", typically used with `extract_frames()`
* self.mdraw = "MD_Raw.json", typically used with `detect()`
* self.detections = "Detections.csv", typically used with `parse_detections()`
* self.predictions = "Predictions.csv", typically used with `classify()`
* self.results = "Results.csv", typically used with the export functions.

If `export_folders()` or `plot_all_bounding_boxes()` are used with a WorkingDirectory,
it will create a "Sorted" or "Plots" folder respectively within "Animl-Directory".

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
| `data_timezone`   | str   | 4     | Timezone of the data, e.g., 'UTC', 'America/New_York', defaults to local timezone if None|
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

\* For station_depth, if file paths are in the format "image_dir/station/date/file.jpg",
station_depth would be 1 (0 indexed). If None, station column will not be created.
Likewise for camera_depth, if file paths are in the format "image_dir/station/camera/date/file.jpg",
camera_depth would be 2 (0 indexed). If None, camera column will not be created.


<br><br>
  
### animl.active_times(manifest, file_col="filepath", camera_depth=0, timestamp_col="datetime")
{: #active_times}

Get start and stop dates for each camera folder.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`        | pandas DataFrame  | required      | File manifest dataframe with file paths and timestamps |
| `file_col`        | str               | "filepath"    | Column in manifest to use for file paths, defaults to "filepath" |
| `camera_depth`    | int               | 0             | Directory depth from which to split cameras,
                                                          with 0 being the root of the manifest_dir |
| `timestamp_col`   | str               | "datetime"    | Column name representing the timestamp in format "%Y-%m-%d %H:%M:%S", defaults to "datetime" |

**Returns:** `pandas.DataFrame` with a row for each camera and the earliest and latest timestamp of data taken at that camera

<br><br>
  
### animl.sequence_calculation(manifest, station_col="station", sort_columns=None, file_col="filepath", timestamp_col="datetime", maxdiff=60)
{: #sequence_calculation}

Simple sequence calculation based on time differences between consecutive images from the same station. <br>
Unlike sequence_classification(), does not apply any classification or labeling to the sequences.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`        | pandas DataFrame  | required      | File manifest dataframe with file paths and timestamps |
| `station_col`     | str               | "station"     | Column name in the DataFrame representing the station or camera |
| `sort_columns`    | list[str]         | None          | List of columns to sort by before calculating sequences.
                                                          Defaults to None, which sorts by station_col and timestamp_col |
| `file_col`        | str               | "filepath"    | Column in manifest to use for file paths, defaults to "filepath" |
| `timestamp_col`   | str               | "datetime"    | Column name representing the timestamp in format "%Y-%m-%d %H:%M:%S", defaults to "datetime". |
| `maxdiff`         | int               | 60            | Maximum time difference in seconds between consecutive images to be
                                                          considered part of the same sequence. Defaults to 60 |

**Returns:** `pandas.DataFrame` — the input DataFrame with an additional 'sequence' column indicating sequence membership.

<br><br>  

### animl.extract_frames(manifest, frames=5, fps=None, out_file=None, out_dir=None, file_col="filepath", parallel=True, num_workers=NUM_THREADS)
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
| `parallel`    | str               | True          | Toggle to use multiprocessing for frame extraction (default is True) |
| `num_workers` | int               | NUM_THREADS   | Number of worker processes to use for parallel processing (default is NUM_THREADS) |

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
Model types accepted: ["mdv5", "mdv6", "mdv1000-cedar", "mdv1000-larch", "mdv1000-sorrel",
"mdv1000-redwood", "mdv1000-spruce", "yolov5", "yolo", "onnx"]
For yolo models v6+, use "yolo", for v5, use "yolov5". 

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
  
### animl.parse_detections(results, manifest=None, out_file=None, threshold=0.1, file_col="filepath")
{: #parse_detections}

Converts detector output into a detections DataFrame.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results`     | list[dict], list    | required      | Detector output dicts or tuple of (output dicts, failed files) |
| `manifest`    | DataFrame     | None          | Original file manifest, if not None, merge MD predictions automatically |
| `out_file`    | str           | None          | Path to save detections .csv |
| `threshold`   | float         | 0             | Minimum confidence score; detections below are not returned |
| `file_col`    | str           | "filepath"    | Column name containing file paths, will merge results to manifest on this column |

**Returns:** `pd.DataFrame` — one row per detection with columns:
             `filepath`, `category`, `category_label`, `conf`, `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`, `max_detection_conf`

<br><br>

### animl.get_animals(manifest)
{: #get_animals}

Pulls out MD animal detections for classification

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | DataFrame | None  | DataFrame containing one row for every MD detection |

**Returns:** `pd.DataFrame` — subset of manifest containing only animal detections

<br><br>

### animl.get_empty(manifest)
{: #get_empty}

Pulls out MD non-animal detections and adds prediction and confidence columns

| Parameter | Type | Default | Description |
|---|---|---|---|
| `manifest`    | DataFrame | None  | DataFrame containing one row for every MD detection |

**Returns:** `pd.DataFrame` — subset of manifest containing empty, vehicle and human detections
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
| `architecture` | str    | "efficientnet_v2_m"    | Expected architecture name ("efficientnet_v2_m" or "convnext_base")       |
| `quiet`        | bool                                   | True                   | Toggles suppression of device info messages                       |

**Returns:** `(model, class_list)` — loaded model (of given architecture) and class list  

<br><br>
  
### animl.load_class_list(classlist_file)
{: #load_class_list}

Returns classlist file as DataFrame.

| Parameter         | Type   | Default | Description                 |
|-------------------|--------|---------|-----------------------------|
| `classlist_file`  | str    | required| File path to class list CSV |

**Returns:** `pd.DataFrame` — the class list file data  

<br><br>


### animl.class_list_to_dict(class_list, label_col="class", id_col="id")
{: #class_list_to_dict}

Converts a class list DataFrame into a dictionary mapping class IDs to labels.

| Parameter      | Type            | Default      | Description                                                  |
|----------------|-----------------|--------------|--------------------------------------------------------------|
| `class_list`   | pd.DataFrame    | required     | DataFrame containing class list with at least label and id columns |
| `label_col`    | str             | "class"      | Column name in class_list DataFrame containing class labels   |
| `id_col`       | str             | "id"         | Column name in class_list DataFrame containing class IDs (integers corresponding to model output indices) |

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

**Returns:** `tuple` — (`predictions`, `failed_files`)  
- `predictions`: `np.array` of softmaxed logits for each class/image  
- `failed_files`: list of files that failed during processing (if any)  
  
<br><br>

### animl.single_classification(animals, empty, predictions_output, class_list, best=False, file_col="filepath", failed_files=None)
{: #single_classification}

Assigns predicted class labels and confidences to each row in a detection DataFrame, handling failed files and "empty" detections.

| Parameter          | Type                          | Default     | Description                                                         |
|--------------------|-------------------------------|-------------|---------------------------------------------------------------------|
| `animals`          | pd.DataFrame                  | required    | Detections with animals (from manifest)                             |
| `empty`            | pd.DataFrame or None          | None        | Detections with no animals (from manifest)                          |
| `predictions_output`| np.array or tuple            | required    | Softmaxed logits or (logits, failed_files) from `classify()`        |
| `class_list`       | list or pd.Series             | required    | List/series of class labels                                         |
| `best`             | bool                          | False       | If True, returns best prediction for each file only                 |
| `count`             | bool                          | False       | If True, returns count of predicted class for each file             |
| `file_col`         | str                           | "filepath"  | Column for file paths                                               |
| `failed_files`     | list or None                  | None        | List of files that failed during classification                     |

**Returns:** `pd.DataFrame` — DataFrame with columns `prediction`, `confidence`, and associated metadata  

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
| `empty`          | pd.DataFrame or None    | None       | Detections with no animals                                                 |
| `predictions_output`| np.array or tuple    | required   | Softmaxed logits, or (logits, failed_files), from `classify()`             |
| `class_list`     | pd.DataFrame            | required   | Class labels associated with classifier model                               |
| `station_col`    | str                     | required   | Column indicating station/camera                                            |
| `empty_class`    | str                     | ""         | Value of "empty" label in class list, empty string if not specified         |
| `sort_columns`   | list[str] or None       | None       | Columns to sort groups by, if not specified defaults to `station_col` and `timestamp_col` |
| `file_col`       | str                     | "filepath" | Column indicating image file paths                                         |
| `timestamp_col`  | str                     | "datetime" | Column with file timestamps                                           |
| `failed_files`   | list or None            | None       | List of files that failed to classify                                      |
| `maxdiff`        | int                     | 60         | Maximum time (sec) separating images in the same burst/sequence            |

**Returns:** `pd.DataFrame` — sequence-classified results with columns including `prediction`, `confidence`, `sequence`

<br><br>
  
---
## Re-Identification
{: #re-id}
<br>

### animl.load_miew(file_path, device)
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

**Returns:** `np.ndarray` — array of extracted embeddings

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

**Returns:** `np.ndarray` - Distance matrix

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

**Returns:** `np.ndarray` - Computed distance matrix

<br><br>

---
## Model Training
{: #training}
<br>

### animl.train_classifier(config)
{:#train_classifier}

Trains a classifier model based on the provided configuration.
For details on the configuration parameters, see the [config README](https://github.com/conservationtechlab/animl-py/blob/main/src/animl/config/README.md).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | str | required | Path to config yml file containing training parameters and data paths |

<br><br>

### animl.test_classifier(config)
{:#test_classifier}

Tests a classifier model based on the provided configuration, evaluating performance on a test dataset and generating a confusion matrix.
For details on the configuration parameters, see the [config README](https://github.com/conservationtechlab/animl-py/blob/main/src/animl/config/README.md).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | str | required | Path to config yml file containing training parameters and data paths |

<br><br>

### animl.save_classifier(model, out_dir, epoch, stats, optimizer=None, scheduler=None)
Saves model state weights and optional optimizer/scheduler states to disk.

| Parameter   | Type             | Default   | Description                                                  |
|-------------|------------------|-----------|--------------------------------------------------------------|
| `model`     | torch.nn.Module  | required  | The PyTorch model instance to save                           |
| `out_dir`   | str              | required  | Directory path where model weights will be saved             |
| `epoch`     | int              | required  | Current training epoch (used as filename)                    |
| `stats`     | dict             | required  | Training/validation stats/metrics to save with the model     |
| `optimizer` | torch.optim.Optimizer | None      | (Optional) Optimizer state to save                      |
| `scheduler` | torch.optim.lr_scheduler._LRScheduler | None      | (Optional) Scheduler state to save      |

**Returns:** `None`  

<br><br>

### animl.load_classifier_checkpoint(model_path, model, optimizer, scheduler, device)
Loads the latest checkpoint to resume model training, restoring weights and optimizer/scheduler states.

| Parameter    | Type                      | Default   | Description                                               |
|--------------|---------------------------|-----------|-----------------------------------------------------------|
| `model_path` | str or Path               | required  | Path containing saved model `.pt` checkpoints             |
| `model`      | torch.nn.Module           | required  | Model object to load weights into                         |
| `optimizer`  | torch.optim.Optimizer     | required  | Optimizer object to load state into                       |
| `scheduler`  | torch.optim.lr_scheduler._LRScheduler | required  | Scheduler to load state into               |
| `device`     | str                       | required  | Device to map tensors onto ("cpu" or "cuda")              |

**Returns:** `int` — starting epoch restored from the latest checkpoint  

<br><br>

---
## Visualization
{: #visualization}
<br>

### animl.get_frame_as_image(video_path, frame=0)
{:#get_frame_as_image}

| Parameter      | Type | Default   | Description                 |
|----------------|------|-----------|-----------------------------|
| `video_path`   | str  | required  | File path to video                    |
| `frame`        | int  | 0         | Frame number to extract, default is 0 |

**Returns:** np.ndarray - Matrix representing the cv2 image

<br><br>

---
## Export
{: #export}

### class `animl.save_data()`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |


### class `animl.load_data()`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |

### class `animl.save_json()`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |

### class `animl.load_data()`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |

### class `animl.check_file()`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |


### class `animl.get_frame_as_image()`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |

<br><br>

---
## Utilities
{: #utilities}


---
# Troubleshooting