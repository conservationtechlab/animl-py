---
layout: default
title: Home
description: Developing open-source technology and machine learning tools for wildlife conservation and ecological research
---

# Getting Started

## About AniML
{: #about}

The Conservation Tech Lab develops cutting-edge technology solutions for wildlife conservation and ecological research. 
Our work spans machine learning for camera trap analysis, edge-AI field devices, bioacoustics tools, and animal tracking systems.

All of our projects are open-source, promoting collaboration and knowledge sharing within the conservation technology community.
We focus on practical, field-deployable solutions that help researchers and conservationists better understand and protect wildlife.

## Installation
{: #installation}
To Install:
pip install animl

### Requirements
{: #requirements}
ExifTool, PyTorch, Ultralytics, ONNX Runtime, pandas</p>


# Examples
{: #examples}


# Reference

## Full Pipeline
{: #full-pipeline}
### `animl.from_paths(image_dir, detector_file, classifier_file, classlist_file, ...)`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |
| `batch_size` | int | 4 | Batch size for inference |
| `sort` | bool | False | Create symlinks sorted by species |
| `visualize` | bool | False | Save bounding box visualizations |
| `sequence` | bool | False | Use sequence-level classification |
| `detect_only` | bool | False | Skip classification step |
\
\
### `animl.from_config(config)`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | str | required | Path to config yml file.

The config yml must contain the following fields:
  
  
## Data Ingestion and Processing
{: #data-ingestion}
### class `animl.WorkingDirectory()`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |
| `batch_size` | int | 4 | Batch size for inference |
| `sort` | bool | False | Create symlinks sorted by species |
| `visualize` | bool | False | Save bounding box visualizations |
| `sequence` | bool | False | Use sequence-level classification |
| `detect_only` | bool | False | Skip classification step |
  
  
### `animl.build_file_manifest()`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |
| `batch_size` | int | 4 | Batch size for inference |
| `sort` | bool | False | Create symlinks sorted by species |
| `visualize` | bool | False | Save bounding box visualizations |
| `sequence` | bool | False | Use sequence-level classification |
| `detect_only` | bool | False | Skip classification step |

**Returns:** pandas DataFrame object containing file manifest

Example manifest:

  
  
### class `animl.active_times()`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |


  
  
### class `animl.sequence_calculation()`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |
  
  
### class `animl.extract_frames()`
Runs the full detection + classification pipeline on a directory of images or videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | str | required | Path to image/video directory |
| `detector_file` | str | required | Path to MegaDetector model |
| `batch_size` | int | 4 | Batch size for inference |
| `sort` | bool | False | Create symlinks sorted by species |
| `visualize` | bool | False | Save bounding box visualizations |
| `sequence` | bool | False | Use sequence-level classification |
| `detect_only` | bool | False | Skip classification step |
  
  
---
## Detection
{: #detection}

### `animl.load_detector(model_path, model_type, device=None)`
Loads a detector model from a file path.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | str | required | Path to model file |
| `model_type` | str | required | Type of model: `"mdv5"`, `"mdv6"`, `"mdv1000-cedar"`, `"mdv1000-larch"`, `"mdv1000-sorrel"`, `"mdv1000-redwood"`, `"mdv1000-spruce"`, `"yolov5"`, `"yolo"`, `"onnx"` |
| `device` | str | None | Device to run model on: `"cpu"` or `"cuda"` |

**Returns:** loaded model object
  
  
### `animl.detect(detector, image_file_names, resize_width, resize_height, ...)`
Runs a detector model on batches of image files.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `detector` | object | required | Preloaded detector model |
| `image_file_names` | str / list / DataFrame | required | Single image path, list of paths, or manifest DataFrame |
| `resize_width` | int | required | Width to resize images to |
| `resize_height` | int | required | Height to resize images to |
| `letterbox` | bool | True | Resize and pad to preserve aspect ratio |
| `confidence_threshold` | float | 0.1 | Minimum confidence score to retain a detection |
| `file_col` | str | `"filepath"` | Column name in manifest containing file paths |
| `batch_size` | int | 1 | Number of images per batch |
| `num_workers` | int | 1 | Number of dataloader workers |
| `device` | str | None | Device to run inference on: `"cpu"` or `"cuda"` |
| `checkpoint_path` | str | None | Path to save intermediate checkpoint JSON |
| `checkpoint_frequency` | int | -1 | Save checkpoint every N batches; -1 disables checkpointing |

**Returns:** `list[dict]` — MegaDetector-format results, one dict per image
  
  
### `animl.parse_detections(results, manifest=None, out_file=None, threshold=0.1, file_col="filepath")`
Converts detector output into a detections DataFrame.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | list[dict] | required | Raw detector output from `detect()` |
| `manifest` | DataFrame | None | Original file manifest to join metadata onto |
| `out_file` | str | None | Path to save detections CSV |
| `threshold` | float | 0.1 | Minimum confidence score; detections below are set to category 0 |
| `file_col` | str | `"filepath"` | Column name containing file paths |

**Returns:** `pd.DataFrame` — one row per detection with columns `filepath`, `category`, `conf`, `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`, `max_detection_conf`
  
  
---
## Classification
{: #classification}

### `save_classifier(model, out_dir, epoch, stats, optimizer=None, scheduler=None)`
Saves model state weights and optional optimizer/scheduler states to disk.

| Parameter   | Type             | Default   | Description                                                  |
|-------------|------------------|-----------|--------------------------------------------------------------|
| `model`     | torch.nn.Module  | required  | The PyTorch model instance to save                           |
| `out_dir`   | str              | required  | Directory path where model weights will be saved             |
| `epoch`     | int              | required  | Current training epoch (used as filename)                    |
| `stats`     | dict             | required  | Training/validation stats/metrics to save with the model     |
| `optimizer` | torch.optim.Optimizer | None      | (Optional) Optimizer state to save                           |
| `scheduler` | torch.optim.lr_scheduler._LRScheduler | None      | (Optional) Scheduler state to save                          |

**Returns:** `None`  
  
  
### `load_classifier(model_path, classes, device=None, architecture="efficientnet_v2_m", quiet=True)`
Creates and loads a classifier model of the given architecture from disk, with the associated class list.

| Parameter      | Type                                   | Default                | Description                                                      |
|----------------|----------------------------------------|------------------------|------------------------------------------------------------------|
| `model_path`   | str                                    | required               | File or directory path to the model weights                      |
| `classes`      | int \| str \| Path \| pd.DataFrame      | required               | Number of classes, class list file, or DataFrame                 |
| `device`       | str                                    | None                   | Device to load model on ("cpu" or "cuda")                        |
| `architecture` | str                                    | "efficientnet_v2_m"    | Architecture name ("efficientnet_v2_m" or "convnext_base")       |
| `quiet`        | bool                                   | True                   | If `True`, suppresses device info messages                       |

**Returns:** `(model, class_list)` — loaded model (of given architecture) and class list  
  
  
### `load_classifier_checkpoint(model_path, model, optimizer, scheduler, device)`
Loads the latest checkpoint to resume model training, restoring weights and optimizer/scheduler states.

| Parameter    | Type                      | Default   | Description                                               |
|--------------|---------------------------|-----------|-----------------------------------------------------------|
| `model_path` | str or Path               | required  | Path containing saved model `.pt` checkpoints             |
| `model`      | torch.nn.Module           | required  | Model object to load weights into                         |
| `optimizer`  | torch.optim.Optimizer     | required  | Optimizer object to load state into                       |
| `scheduler`  | torch.optim.lr_scheduler._LRScheduler | required  | Scheduler to load state into               |
| `device`     | str                       | required  | Device to map tensors onto ("cpu" or "cuda")              |

**Returns:** `int` — starting epoch restored from the latest checkpoint  
  
  
### `load_class_list(classlist_file)`
Returns classlist file as DataFrame.

| Parameter         | Type   | Default | Description                 |
|-------------------|--------|---------|-----------------------------|
| `classlist_file`  | str    | required| File path to class list CSV |

**Returns:** `pd.DataFrame` — the class list file data  
  
  
### `classify(model, detections, resize_width=480, resize_height=480, file_col="filepath", crop=True, normalize=True, batch_size=1, num_workers=NUM_THREADS, device=None, out_file=None)`
Runs prediction for input detections using a preloaded classifier model, managing batching and output saving.

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
| `device`       | str             | None         | Device for inference ("cpu" or "cuda")                      |
| `out_file`     | str             | None         | Output file path to save prediction results                  |

**Returns:** `tuple` — (`predictions`, `failed_files`)  
- `predictions`: `np.array` of softmaxed logits for each class/image  
- `failed_files`: list of files that failed during processing (if any)  
  
### `single_classification(animals, empty, predictions_output, class_list, best=False, file_col="filepath", failed_files=None)`
Assigns predicted class labels and confidences to each row in a detection DataFrame, handling failed files and "empty" detections.

| Parameter          | Type                          | Default     | Description                                                         |
|--------------------|-------------------------------|-------------|---------------------------------------------------------------------|
| `animals`          | pd.DataFrame                  | required    | Detections with animals (from manifest)                             |
| `empty`            | pd.DataFrame or None          | None        | Detections with no animals (from manifest)                          |
| `predictions_output`| np.array or tuple            | required    | Softmaxed logits or (logits, failed_files) from `classify()`        |
| `class_list`       | list or pd.Series             | required    | List/series of class labels                                         |
| `best`             | bool                          | False       | If True, returns best prediction for each file only                 |
| `file_col`         | str                           | "filepath"  | Column for file paths                                               |
| `failed_files`     | list or None                  | None        | List of files that failed during classification                     |

**Returns:** `pd.DataFrame` — DataFrame with columns `prediction`, `confidence`, and associated metadata  
  
  
### `sequence_classification(animals, empty, predictions_output, class_list, station_col, empty_class="", sort_columns=None, file_col="filepath", timestamp_col="datetime", failed_files=None, maxdiff=60)`
Assigns class labels to detections at a sequence level (camera trap burst) using both spatial and temporal context, improving classification accuracy for image bursts.

| Parameter        | Type                    | Default    | Description                                                                |
|------------------|-------------------------|------------|----------------------------------------------------------------------------|
| `animals`        | pd.DataFrame            | required   | Detections with animals                                                    |
| `empty`          | pd.DataFrame or None    | None       | Detections with no animals                                                 |
| `predictions_output`| np.array or tuple    | required   | Softmaxed logits, or (logits, failed_files), from `classify()`             |
| `class_list`     | pd.DataFrame            | required   | Class labels associated with classifier model                               |
| `station_col`    | str                     | required   | Column indicating station/camera                                            |
| `empty_class`    | str                     | ""         | Value of "empty" label in class list                                       |
| `sort_columns`   | list[str] or None       | None       | Columns to sort groups by                                                  |
| `file_col`       | str                     | "filepath" | Column indicating image file paths                                         |
| `timestamp_col`  | str                     | "datetime" | Column with detection timestamps                                           |
| `failed_files`   | list or None            | None       | List of files that failed to classify                                      |
| `maxdiff`        | int                     | 60         | Maximum time (sec) separating images in the same burst/sequence            |

**Returns:** `pd.DataFrame` — sequence-classified results with columns including `prediction`, `confidence`, `sequence`
  
  
---
## Re-Identification
{: #re-id}



---
## Model Training
{: #training}

---
## Visualization
{: #visualization}
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





                        <nav>

                            <a href="#getting-started">class WorkingDirectory</a>
                            <a href="#build_file_manifest">build_file_manifest()</a>
                            <a href="#usage">extract_frames()</a>
                            <a href="#usage">load_detector()</a>
                            <a href="#usage">detect()</a>
                            <a href="#usage">parse_detections()</a>
                            <a href="#usage">load_classifier()</a>
                            <a href="#usage">load_class_list()</a>
                            <a href="#getting-started">classify()</a>
                            <a href="#getting-started">single_classification()</a>
                            <a href="#getting-started">sequence_classification()</a>
                            <a href="#usage">Re-ID</a>
                            <a href="#usage">Training</a>
                            <a href="#export">Export</a>
                        </nav>