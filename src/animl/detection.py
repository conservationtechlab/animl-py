"""
Object Detection Module

Functions for loading MegaDetector, as well as custom YOLO models
parse_detections() converts json output into a dataframe

"""
import argparse
from typing import Optional, Union
import time
from shutil import copyfile
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from ultralytics import YOLO

from animl import file_management
from animl.model_architecture import MEGADETECTORv5_SIZE, MD_LABELS, MD_MODELS
from animl.generator import manifest_dataloader, image_to_tensor
from animl.utils.general import (normalize_bbox, _xyxy_to_xywh, scale_letterbox,
                                 non_max_suppression, get_torch_device, get_onnx_device)


def load_detector(model_path: str,
                  model_type: str,
                  device: Optional[str] = None):
    """
    Load Detector model from filepath.

    Args:
        model_path (str): path to model file
        model_type (str): type of model expected ["mdv5", "mdv6", "mdv1000-cedar", "mdv1000-larch", "mdv1000-sorrel",
                                                "mdv1000-redwood", "mdv1000-spruce", "yolov5", "yolo", "onnx"]
                        for yolo models v6+, use "yolo", for v5, use "yolov5".
                        for mdv1000 models, specify the version (cedar, larch, sorrel, redwood, spruce)
        device (str): specify to run on cpu or gpu

    Returns:
        object: loaded model object
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model_type = model_type.lower()

    # YOLOv5/MDv5
    if model_type in {"mdv5", "yolov5", "mdv1000-redwood", "mdv1000-spruce"}:
        # check to make sure GPU is available if chosen
        device = get_torch_device(user_set=device)
        # load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Compatibility fix that allows older YOLOv5 models with
        # newer versions of YOLOv5/PT
        if hasattr(checkpoint['model'], 'modules'):
            for m in checkpoint['model'].modules():
                t = type(m)
                if t is torch.nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                    m.recompute_scale_factor = None
        model = checkpoint['model'].float().fuse().eval()  # FP32 model
        model.model_type = model_type
        model.to(device)
        return model
    # YOLOv6+
    elif model_type in {"yolo", "mdv6", "mdv1000-cedar", "mdv1000-larch", "mdv1000-sorrel"}:
        # check to make sure GPU is available if chosen
        device = get_torch_device(user_set=device)
        model = YOLO(model_path, task='detect')
        model.model_type = model_type
        model.to(device)
        return model
    # ONNX model
    elif model_type in {"onnx"}:
        import onnxruntime as ort
        # check to make sure GPU is available if chosen
        providers = get_onnx_device(user_set=device)
        model = ort.InferenceSession(model_path, providers=providers)
        model.model_type = model_type
        return model
    else:
        print(f"Please chose a supported model. Version {model_type} is not supported.")
        print("Expected model_type to be one of ['mdv5', 'mdv6', 'mdv1000-cedar', 'mdv1000-larch', 'mdv1000-sorrel',",
              "'mdv1000-redwood', 'mdv1000-spruce', 'yolov5', 'yolo', 'onnx']")
        return None


def detect(detector,
           image_file_names,
           resize_width: int,
           resize_height: int,
           letterbox: bool = True,
           category_map: Optional[dict] = MD_LABELS,
           confidence_threshold: float = 0.1,
           file_col: str = 'filepath',
           batch_size: int = 1,
           num_workers: int = 1,
           device: Optional[str] = None,
           checkpoint_path: Optional[str] = None,
           checkpoint_frequency: int = -1) -> list[dict]:
    """
    Runs Detector model on a batches of image files.

    Args:
        detector (object): preloaded detector model
        image_file_names (mult): list of image filenames, a single image filename, or manifest
                                    containing a list of images.
        resize_width (int): width to resize images to
        resize_height (int): height to resize images to
        letterbox (bool): if True, resize and pad image to keep aspect ratio, else resize without padding
        category_map (dict): mapping of category IDs to human-readable labels
        confidence_threshold (float): only detections above this threshold are returned
        file_col (str): column name containing file paths
        batch_size (int): size of each batch
        num_workers (int): number of processes to handle the data
        device (str): specify to run on cpu or gpu
        checkpoint_path (str): path to checkpoint file
        checkpoint_frequency (int): write results to checkpoint file every N images

    Returns:
        list: list of dicts, each dict represents detections on one image
    """
    if checkpoint_frequency != -1:
        checkpoint_frequency = max(1, round(checkpoint_frequency/batch_size, None))

    # assume yolo model if not already specified
    if 'model_type' not in detector.__dict__:
        detector.model_type = "yolo"

    # convert map keys to int if they are string (ie from reticulate)
    category_map = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in category_map.items()}

    # Single image filepath
    if isinstance(image_file_names, str):
        # convert img path to tensor
        batch_from_dataloader = image_to_tensor(image_file_names, letterbox=letterbox,
                                                resize_width=resize_width, resize_height=resize_height)
        if batch_from_dataloader is None:
            print(f"Error loading image {image_file_names}. Skipping.")
            return []
        # yolov5
        if detector.model_type in {"yolov5", "mdv5", "mdv1000-redwood", "mdv1000-spruce"}:
            # check to make sure GPU is available if chosen
            device = get_torch_device(user_set=device)
            # letterboxing should be true
            prediction = detector(batch_from_dataloader[0].to(device))
            pred: list = prediction[0]
            pred = non_max_suppression(prediction=pred, conf_thres=confidence_threshold)
        # onnx
        elif detector.model_type == "onnx":
            input_name = detector.get_inputs()[0].name
            providers = get_onnx_device(user_set=device)
            if 'CUDAExecutionProvider' in providers:
                pred = detector.run(None, {input_name: batch_from_dataloader[0].numpy()})[0]
            # default to cpu
            else:
                pred = detector.run(None, {input_name: batch_from_dataloader[0].cpu().numpy()})[0]
        # standard yolo model (v6+)
        else:
            pred = detector.predict(source=batch_from_dataloader[0].to(device),
                                    conf=confidence_threshold, verbose=False)
        # convert predictions to expected output format and append to results
        results = _convert_detections(pred, batch_from_dataloader, letterbox, detector.model_type, category_map)
        return results

    # list of image filepaths
    elif isinstance(image_file_names, list):
        # create a data frame from list of image paths
        manifest = pd.DataFrame(image_file_names, columns=[file_col])
        # no frame column, assume all images and set to 0
        manifest['frame'] = 0

    # full manifest, select file_col
    elif isinstance(image_file_names, pd.DataFrame):
        if file_col not in image_file_names.columns:
            raise ValueError(f"file_col {file_col} not found in manifest columns")
        # no frame column, assume all images and set to 0
        if 'frame' not in image_file_names.columns:
            print("Warning: 'frame' column not found in manifest columns. Defaulting to 0 assuming images.")
            image_file_names['frame'] = 0
        # create a list of image paths
        manifest = image_file_names[[file_col, 'frame']]

    # single row pd.Series, select file_col
    elif isinstance(image_file_names, pd.Series):
        if file_col not in image_file_names.index:
            raise ValueError(f"file_col {file_col} not found in Series index")
        if 'frame' not in image_file_names.index:
            print("Warning: 'frame' column not found in Series index. Defaulting to 0 assuming images.")
            image_file_names['frame'] = 0
        # create a data frame from image_file_names
        manifest = pd.DataFrame(image_file_names).T
    # column from pd.DataFrame, expected input
    else:
        raise ValueError('image_file_names is not a recognized object')

    # load checkpoint
    if file_management.check_file(checkpoint_path, output_type="Megadetector raw output"):
        results = file_management.load_json(checkpoint_path).get('images')
        already_processed = set([r['filepath'] for r in results])
        manifest = image_file_names[~image_file_names[file_col].isin(already_processed)][[file_col, 'frame']].reset_index(drop=True)
        if manifest.empty:
            print("All images have already been processed. Exiting.")
            return results
    else:
        results = []
        image_file_names = set(image_file_names)

    if detector.model_type == "onnx":
        device = get_onnx_device(user_set=device, quiet=True)
    else:
        device = get_torch_device(user_set=device, quiet=True)

    # create dataloader
    dataloader = manifest_dataloader(manifest, batch_size=batch_size,
                                     num_workers=num_workers, crop=False,
                                     normalize=True, letterbox=letterbox,
                                     file_col=file_col,
                                     resize_width=resize_width,
                                     resize_height=resize_height)

    start_time = time.time()
    failed_files = []

    count = 0  # counter to track number of batches processed for checkpointing
    for _, batch_from_dataloader in tqdm(enumerate(dataloader), total=len(dataloader)):
        successes, failed = batch_from_dataloader
        failed_files.extend(failed)

        if successes is None:  # entire batch was bad
            continue
        count += 1

        # Run inference on the current batch of image_tensors
        if detector.model_type in {"yolov5", "mdv5", "mdv1000-redwood", "mdv1000-spruce"}:
            # letterboxing should be true
            prediction = detector(successes[0].to(device))
            pred: list = prediction[0]
            pred = non_max_suppression(prediction=pred, conf_thres=confidence_threshold)
        # 'onnx'
        elif detector.model_type == "onnx":
            input_name = detector.get_inputs()[0].name
            if device == "cpu":
                pred = detector.run(None, {input_name: successes[0].cpu().numpy()})[0]
            else:
                pred = detector.run(None, {input_name: successes[0].numpy()})[0]
        # standard yolo model (v6+)
        else:
            pred = detector.predict(source=successes[0].to(device), conf=confidence_threshold, verbose=False)

        # convert predictions to expected output format and append to results
        results.extend(_convert_detections(pred, successes, letterbox, detector.model_type, category_map))

        # Write a checkpoint if necessary
        if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
            print(f"Writing a new checkpoint after having processed {count*batch_size} images since last restart")
            _save_detection_checkpoint(checkpoint_path, results)

    # final checkpoint save
    if checkpoint_path:
        _save_detection_checkpoint(checkpoint_path, results)

    print(f"\nFinished detection. Total images processed: {len(results)} at ",
          f"{round(len(results)/(time.time() - start_time), 1)} img/s.")

    return results, failed_files


def _convert_detections(predictions: list,
                        batch_from_dataloader: list,
                        letterbox: bool,
                        model_type: str,
                        category_map: dict = MD_LABELS,) -> pd.DataFrame:
    # Converts output into nested list with categories, conf, and bboxes in expected format for parsing function.
    # Supports YOLOv5/MDv5, YOLOv6+, and ONNX models with either relative or absolute bounding box outputs.
    # If letterbox=True, rescales bboxes back to original image size.

    # unpack batch dataloader output
    image_tensors = batch_from_dataloader[0]
    image_paths = batch_from_dataloader[1]
    image_frames = batch_from_dataloader[2]
    image_sizes = batch_from_dataloader[3]

    if model_type != "onnx":
        # convert to numpy if needed
        if isinstance(image_sizes, torch.Tensor):
            image_sizes = image_sizes.cpu().numpy()
        if isinstance(image_tensors, torch.Tensor):
            image_tensors = image_tensors.cpu().numpy()
        if isinstance(image_frames, torch.Tensor):
            image_frames = image_frames.cpu().numpy()

    # if no category map provided, default to MD_LABELS
    if category_map is None:
        print("No category map provided, defaulting to MD_LABELS. ",
              "This may lead to incorrect category labels if using a custom model.")
        category_map = MD_LABELS

    results = []
    for i, pred in enumerate(predictions):
        # extract boxes and conf
        # YOLOv5/MDv5
        if model_type in {"onnx", "mdv5", "yolov5", "mdv1000-redwood", "mdv1000-spruce"}:
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            boxes = pred[:, :4]  # Bounding box coordinates
            conf = pred[:, 4]  # Confidence scores
            category = pred[:, 5]  # Class labels as integers
            max_detection_conf = float(round(conf.max(), 4)) if len(conf) > 0 else None
        # YOLOv6+
        elif model_type in {"yolo", "mdv6", "mdv1000-cedar", "mdv1000-larch", "mdv1000-sorrel"}:
            boxes = pred.boxes.xyxyn.cpu().numpy()  # Bounding box coordinates
            conf = pred.boxes.conf.cpu().numpy()  # Confidence scores
            category = pred.boxes.cls.cpu().numpy()  # Class labels as integers
            max_detection_conf = float(round(conf.max(), 4)) if len(conf) > 0 else None
        else:
            print(f"Please chose a supported model. Version {model_type} is not supported.")
            return None

        # no detections
        if len(conf) == 0:
            # category is []
            data = {'filepath': str(image_paths[i]),
                    'frame': int(image_frames[i]),
                    'max_detection_conf': max_detection_conf,
                    # for MD models, set category to 0 (empty) if no detections, for other models set to None
                    'category': 0 if model_type in MD_MODELS else None,
                    'category_label': 'empty',
                    'detections': []}
            results.append(data)
        # detections
        else:
            detections = []
            for j in range(len(conf)):
                # YOLOv5/MDv5
                if model_type in {'onnx', 'mdv5', 'yolov5', "mdv1000-redwood", "mdv1000-spruce"}:  # xyxy absolute
                    bbox = normalize_bbox(boxes[j], image_tensors[i].shape[1:])
                    bbox = _xyxy_to_xywh(bbox)
                # YOLOv6+
                elif model_type in {'yolo', "mdv6", "mdv1000-cedar", "mdv1000-larch", "mdv1000-sorrel"}:  # xyxy relative
                    bbox = _xyxy_to_xywh(boxes[j])
                else:
                    print(f"Please chose a supported model. Version {model_type} is not supported.")
                    return None
                # rescale bboxes if letterbox was used in preprocessing
                if letterbox:
                    bbox = scale_letterbox(bbox, image_tensors[i].shape[1:], image_sizes[i, :])

                # increase md categories by 1
                if model_type in MD_MODELS:
                    category[j] += 1

                # build detection dict
                detection = {'category': int(category[j]),
                             'category_label': category_map.get(int(category[j]), "unknown"),
                             'conf': float(round(conf[j], 4)),
                             'bbox_x': float(round(bbox[0], 4)),
                             'bbox_y': float(round(bbox[1], 4)),
                             'bbox_w': float(round(bbox[2], 4)),
                             'bbox_h': float(round(bbox[3], 4))}
                detections.append(detection)

            data = {'filepath': str(image_paths[i]),
                    'frame': int(image_frames[i]),
                    'max_detection_conf': max_detection_conf,
                    'detections': detections}
            results.append(data)

    return results


def parse_detections(results: Union[list, tuple],
                     manifest: Optional[pd.DataFrame] = None,
                     out_file: Optional[str] = None,
                     threshold: float = 0,
                     file_col: str = "filepath"):
    """
    Converts listed output from detector to DataFrame.

    Args:
        results (Union[list, tuple]): md output dicts or tuple of (md output dicts, failed files)
        manifest (pd.DataFrame): full file manifest, if not None, merge md predictions automatically
        out_file (str): path to save dataframe
        threshold (float): parse only detections above given confidence threshold
        file_col (str): if manifest, merge results onto file_col

    Returns:
        df (pd.DataFrame): formatted md outputs, one row per detection
    """
    if manifest is not None and file_col not in manifest.columns:
        raise ValueError(f"file_col '{file_col}' not found in manifest columns")

    if manifest is not None and 'frame' not in manifest.columns:
        print("""Warning: 'frame' column not found in manifest columns. Defaulting to 0 for all rows.""")
        manifest['frame'] = 0

    # unpack results
    if isinstance(results, (tuple, list)) and len(results) == 2 and isinstance(results[0], list):
        detections, failed_files = results
        if len(failed_files) > 0:
            print(f"Warning: {len(failed_files)} files failed to load during detection and will be excluded from results.")
            if out_file is not None:
                with (Path(out_file).parent / "detection_failed_files.txt").open("w") as f:
                    for item in failed_files:
                        f.write(f"{item}\n")
    else:
        detections, failed_files = results, None

    # check results format
    if not isinstance(detections, list):
        raise TypeError("MD results input must be list")
    if len(detections) == 0:
        raise AssertionError("'results' contains no detections")

    # load results from file if they have already been parsed
    if file_management.check_file(out_file, output_type="Detections"):
        return file_management.load_data(out_file)

    lst = []
    for frame in tqdm(detections):
        try:
            frame_detections = frame['detections']
        except KeyError:
            print('File error ', frame['filepath'])
            continue

        if len(frame_detections) == 0:
            data = {'filepath': frame['filepath'],
                    'frame': frame['frame'],
                    'max_detection_conf': frame['max_detection_conf'],
                    'category': frame['category'] if 'category' in frame else None,
                    'category_label': frame['category_label'] if 'category_label' in frame else 'empty',
                    'conf': None, 'bbox_x': None, 'bbox_y': None, 'bbox_w': None, 'bbox_h': None}
            lst.append(data)

        else:
            for detection in frame_detections:
                if (detection['conf'] > threshold):
                    data = {'filepath': frame['filepath'],
                            'frame': frame['frame'],
                            'max_detection_conf': frame['max_detection_conf'],
                            'category': detection['category'],
                            'category_label': detection['category_label'],
                            'conf': detection['conf'],
                            'bbox_x': np.clip(detection['bbox_x'], 0, 1),
                            'bbox_y': np.clip(detection['bbox_y'], 0, 1),
                            'bbox_w': np.clip(detection['bbox_w'], 0, 1),
                            'bbox_h': np.clip(detection['bbox_h'], 0, 1)}
                    lst.append(data)

    df = pd.DataFrame(lst)

    if manifest is not None:
        if file_col in manifest.columns:
            df = manifest.merge(df, left_on=[file_col, 'frame'], right_on=["filepath", "frame"], how='left')
        else:
            raise ValueError("Please provide a manifest with a valid file_col to merge results onto.")

    if out_file:
        file_management.save_data(df, out_file)

    return df


def get_animals(manifest: pd.DataFrame):
    """
    Pulls MD animal detections for classification

    Args:
        manifest (pd.DataFrame): DataFrame containing one row for every MD detection

    Returns:
        subset of manifest containing only animal detections
    """
    if "category_label" in manifest.columns:
        return manifest[manifest["category_label"] == "animal"].reset_index(drop=True)
    # Removes all images that MegaDetector gave no detection for
    else:
        # make sure category column is int and fill NaN with 0 (empty)
        manifest["category"] = manifest["category"].fillna(0)
        # Pulls only the animal detections
        return manifest[manifest["category"].astype(int) == 1].reset_index(drop=True)


def get_empty(manifest: pd.DataFrame):
    """
    Pulls MD non-animal detections

    Args:
        manifest (pd.DataFrame): DataFrame containing one row for every MD detection

    Returns:
        otherdf: subset of manifest containing empty, vehicle and human detections
        with added prediction and confidence columns
    """
    if "category_label" in manifest.columns:
        otherdf = manifest[manifest["category_label"] != "animal"].reset_index(drop=True)

    else:
        # Convert category column to int and fill NaN with 0 (empty) if necessary
        manifest["category"] = manifest["category"].fillna(0)
        manifest["category"] = manifest["category"].astype(int)
        manifest["category_label"] = manifest["category"].replace(MD_LABELS)
        otherdf = manifest[manifest["category"] != 1].reset_index(drop=True)

    if not otherdf.empty:
        otherdf['prediction'] = otherdf["category_label"]
        otherdf['confidence'] = otherdf['conf'].fillna(1)  # correct empty conf

    return otherdf


def _save_detection_checkpoint(checkpoint_path: str, results: dict) -> None:
    """
    Save a checkpoint of the detection results to a JSON file.

    Args:
        checkpoint_path (str): the path to the checkpoint file
        results (list): a list of detection results to save
    """
    assert checkpoint_path is not None
    # Back up any previous checkpoints, to protect against crashes while we're writing
    # the checkpoint file.
    checkpoint_tmp_path = None
    if Path(checkpoint_path).is_file():
        checkpoint_tmp_path = str(checkpoint_path) + '_tmp'
        copyfile(checkpoint_path, checkpoint_tmp_path)

    # Write the new checkpoint
    file_management.save_json({'images': results}, checkpoint_path, prompt=False)

    # Remove the backup checkpoint if it exists
    if checkpoint_tmp_path is not None:
        Path(checkpoint_tmp_path).unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deep learning model.')

    parser.add_argument('detector', help='Path to detector file')
    parser.add_argument('manifest', help='Path to manifest file')
    parser.add_argument('output_path', help='Path to output file')
    parser.add_argument('detector_labels', nargs='?', help='Path to detector labels file')

    parser.add_argument('--model_type', nargs='?', help='Path to detector file', default='MDv5')
    parser.add_argument('--resize_width', nargs='?', help='Path to config file', default=MEGADETECTORv5_SIZE)
    parser.add_argument('--resize_height', nargs='?', help='Path to config file', default=MEGADETECTORv5_SIZE)
    parser.add_argument('--letterbox', nargs='?', help='Path to config file', default=True)
    parser.add_argument('--confidence_threshold', nargs='?', help='Path to config file', default=0.1)
    parser.add_argument('--file_col', nargs='?', help='Path to config file', default='frame')
    parser.add_argument('--batch_size', nargs='?', help='Path to config file', default=4)
    parser.add_argument('--num_workers', nargs='?', help='Path to config file', default=4)
    parser.add_argument('--device', nargs='?', help='Path to config file', default=get_torch_device())

    args = parser.parse_args()

    detector = load_detector(args.detector, args.model_type, device=args.device)
    manifest = file_management.load_data(args.manifest)

    if args.detector_labels:
        category_map = file_management.load_json(args.detector_labels)
        category_map = file_management.class_list_to_dict(category_map)
    else:
        category_map = MD_LABELS

    mdresults, failed_files = detect(detector, manifest, args.resize_width, args.resize_height,
                                     category_map, args.letterbox, confidence_threshold=args.confidence_threshold,
                                     file_col=args.file_col, batch_size=args.batch_size,
                                     num_workers=args.num_workers, device=args.device)
    results = parse_detections(mdresults, manifest=manifest, out_file=args.output_path)
