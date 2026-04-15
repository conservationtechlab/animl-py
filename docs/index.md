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


## Detection
{: #detection}

## Classification
{: #classification}

## Re-Identification
{: #re-id}

## Model Training
{: #training}

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