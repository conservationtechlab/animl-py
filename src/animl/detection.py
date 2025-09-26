"""
Object Detection Module

Functions for loading MegaDetector, as well as custom YOLO models
parse_detections() converts json output into a dataframe

"""
import argparse
from typing import Optional
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from ultralytics import YOLO

from animl import file_management
from animl.model_architecture import MEGADETECTORv5_SIZE
from animl.generator import manifest_dataloader, image_to_tensor
from animl.utils.general import normalize_boxes, xyxy2xywh, scale_letterbox, non_max_suppression, get_device


def load_detector(model_path: str,
                  model_type: str,
                  device: Optional[str] = None):
    """
    Load Detector model from filepath.

    Args:
        model_path (str): path to model file
        model_type (str): type of model expected ["MDV5", "MDV6", "YOLO"]
        device (str): specify to run on cpu or gpu

    Returns:
        object: loaded model object
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    if device is None:
        device = get_device()
    print('Device set to', device)

    if model_type.lower() in {"mdv5", "yolov5"}:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Compatibility fix that allows older YOLOv5 models with
        # newer versions of YOLOv5/PT
        if hasattr(checkpoint['model'], 'modules'):
            for m in checkpoint['model'].modules():
                t = type(m)
                if t is torch.nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                    m.recompute_scale_factor = None
        model = checkpoint['model'].float().fuse().eval()  # FP32 model
        model.model_type = "yolov5"
    elif model_type.lower() in {"yolo", "mdv6", "mdv1000"}:
        model = YOLO(model_path, task='detect')
        model.model_type = "yolo"
    else:
        print(f"Please chose a supported model. Version {model_type} is not supported.")
        return None

    model.to(device)
    return model


def detect(detector,
           image_file_names,
           resize_width: int,
           resize_height: int,
           letterbox: bool = True,
           confidence_threshold: float = 0.1,
           file_col: str = 'frame',
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
        checkpoint_frequency = round(checkpoint_frequency/batch_size, 0)

    # check to make sure GPU is available if chosen
    if device is None:
        device = get_device()
    print('Device set to', device)

    # Single image filepath
    if isinstance(image_file_names, str):
        # convert img path to tensor
        batch_from_dataloader = image_to_tensor(image_file_names, letterbox=letterbox,
                                                resize_width=resize_width, resize_height=resize_height)
        image_tensors = batch_from_dataloader[0]  # Tensor of images for the current batch
        current_image_paths = batch_from_dataloader[1]  # List of image names for the current batch
        image_sizes = batch_from_dataloader[2]  # List of original image sizes for the current batch
        if detector.model_type == "yolov5":
            # letterboxing should be true
            prediction = detector(image_tensors.to(device))
            pred: list = prediction[0]
            pred = non_max_suppression(prediction=pred, conf_thres=confidence_threshold)
        else:
            pred = detector.predict(source=image_tensors.to(device), conf=confidence_threshold, verbose=False)
        results = convert_yolo_detections(pred, image_tensors, current_image_paths, image_sizes, letterbox, detector.model_type)
        return results
    # Full manifest, select file_col
    elif isinstance(image_file_names, pd.DataFrame):
        if file_col not in image_file_names.columns:
            raise ValueError(f"file_col {file_col} not found in manifest columns")
        image_file_names = image_file_names[file_col]
    # single row pd.Series, select file_col
    elif isinstance(image_file_names, pd.Series):
        if file_col not in image_file_names.index:
            raise ValueError(f"file_col {file_col} not found in Series index")
        image_file_names = [image_file_names[file_col]]
    # column from pd.DataFrame, expected input
    elif isinstance(image_file_names, list):
        pass
    else:
        raise ValueError('image_file_names is not a recognized object')

    # load checkpoint
    if file_management.check_file(checkpoint_path):
        results = file_management.load_json(checkpoint_path)
    else:
        results = []

    # remove loaded images
    already_processed = set([r['file'] for r in results.get('images')]) 
    image_file_names = set(image_file_names) - already_processed

    count = 0

    # create a data frame from image_file_names
    manifest = pd.DataFrame(image_file_names, columns=['file'])
    # create dataloader
    dataloader = manifest_dataloader(manifest, batch_size=batch_size,
                                     num_workers=num_workers, crop=False,
                                     normalize=True, letterbox=letterbox,
                                     resize_width=resize_width,
                                     resize_height=resize_height)

    print("Starting batch processing...")
    start_time = time.time()
    for batch_idx, batch_from_dataloader in tqdm(enumerate(dataloader), total=len(dataloader)):
        count += 1

        image_tensors = batch_from_dataloader[0]  # Tensor of images for the current batch
        current_image_paths = batch_from_dataloader[1]  # List of image names for the current batch
        image_sizes = batch_from_dataloader[2]  # List of original image sizes for the current batch

        # Run inference on the current batch of image_tensors
        if detector.model_type == "yolov5":
            # letterboxing should be true
            prediction = detector(image_tensors.to(device))
            pred: list = prediction[0]
            pred = non_max_suppression(prediction=pred, conf_thres=confidence_threshold)
        else:
            pred = detector.predict(source=image_tensors.to(device), conf=confidence_threshold, verbose=False)
        # convert to normalized xywh
        results.extend(convert_yolo_detections(pred, image_tensors, current_image_paths, image_sizes, letterbox, detector.model_type))
        # Write a checkpoint if necessary
        if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
            print('Writing a new checkpoint after having processed {} images since last restart'.format(count*batch_size))
            file_management.save_detection_checkpoint(checkpoint_path, results)

    print(f"\nFinished batch processing. Total images processed: {len(results)} at {round(len(results)/(time.time() - start_time), 1)} img/s.")

    return results


def convert_yolo_detections(predictions: list,
                            image_tensors: list,
                            image_paths: list,
                            image_sizes: list,
                            letterbox: bool,
                            model_type: str,) -> pd.DataFrame:
    """
    Converts YOLO output into a nested list.

    Args:
        predictions (list): YOLO detection output (list of dictionaries with detections for each file)
        image_tensors (list): array of image tensors from mdv6 output
        image_paths (list): List of image file paths corresponding to predictions
        image_sizes (list): List of original image sizes corresponding to predictions
        letterbox (bool): whether letterboxing was used during preprocessing
        model_type (str): type of model expected ["MDV5", "MDV6", "YOLO"]

    Returns:
        results (list): Formatted YOLO outputs, nested list of dictionaries
    """
    # convert to numpy if needed
    if isinstance(image_sizes, torch.Tensor):
        image_sizes = image_sizes.cpu().numpy()
    if isinstance(image_tensors, torch.Tensor):
        image_tensors = image_tensors.cpu().numpy()

    results = []

    # loop over all predictions
    for i, pred in enumerate(predictions):
        file = image_paths[i]

        # extract boxes and conf
        # YOLOv5/MDv5
        if model_type.lower() in {"mdv5", "yolov5"}:
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            boxes = pred[:, :4]  # Bounding box coordinates
            conf = pred[:, 4]  # Confidence scores
            category = pred[:, 5]  # Class labels as integers
            max_detection_conf = conf.max() if len(conf) > 0 else 0
        # YOLOv6+
        elif model_type.lower() in {"yolo", "mdv6", "mdv1000"}:
            boxes = pred.boxes.xyxyn.cpu().numpy()  # Bounding box coordinates
            conf = pred.boxes.conf.cpu().numpy()  # Confidence scores
            category = pred.boxes.cls.cpu().numpy()  # Class labels as integers
            max_detection_conf = conf.max() if len(conf) > 0 else 0
        else:
            print(f"Please chose a supported model. Version {model_type} is not supported.")
            return None

        # no detections
        if len(conf) == 0:
            data = {'file': file,
                    'max_detection_conf': float(round(max_detection_conf, 4)),
                    'detections': []}
            results.append(data)
        # detections
        else:
            detections = []
            for j in range(len(conf)):
                # YOLOv5/MDv5
                if model_type.lower() in {'mdv5', 'yolov5'}:  # xyxy absolute
                    bbox = normalize_boxes(boxes[j], image_tensors[i].shape[1:])
                    bbox = xyxy2xywh(bbox)
                # YOLOv6+
                elif model_type.lower() in {'yolo', "mdv6", "mdv1000"}:  # xyxy relative
                    bbox = xyxy2xywh(boxes[j])
                else:
                    print(f"Please chose a supported model. Version {model_type} is not supported.")
                    return None

                if letterbox:
                    bbox = scale_letterbox(bbox, image_tensors[i].shape[1:], image_sizes[i, :])

                data = {'category': int(category[j]+1),
                        'conf': float(round(conf[j], 4)),
                        'bbox_x': float(round(bbox[0], 4)),
                        'bbox_y': float(round(bbox[1], 4)),
                        'bbox_w': float(round(bbox[2], 4)),
                        'bbox_h': float(round(bbox[3], 4))}
                detections.append(data)

            data = {'file': file,
                    'max_detection_conf': float(round(max_detection_conf, 4)),
                    'detections': detections}
            results.append(data)

    return results


def parse_detections(results: list,
                     manifest: Optional[pd.DataFrame] = None,
                     out_file: Optional[str] = None,
                     threshold: float = 0,
                     file_col: str = "frame"):
    """
    Converts listed output from detector to DataFrame.

    Args:
        results (list): md output dicts
        manifest (pd.DataFrame): full file manifest, if not None, merge md predictions automatically
        out_file (str): path to save dataframe
        threshold (float): parse only detections above given confidence threshold
        file_col (str): if manifest, merge results onto file_col

    Returns:
        df (pd.DataFrame): formatted md outputs, one row per detection
    """
    # load checkpoint
    if file_management.check_file(out_file):  # checkpoint comes back empty
        df = file_management.load_data(out_file)
        already_processed = set([row['file'] for row in df])

    else:
        df = pd.DataFrame(columns=('file', 'max_detection_conf', 'category', 'conf',
                                   'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'))
        already_processed = set()

    if not isinstance(results, list):
        raise AssertionError("MD results input must be list")

    if len(results) == 0:
        raise AssertionError("'results' contains no detections")

    lst = []

    for frame in tqdm(results):
        # pass if already analyzed
        if frame['file'] in already_processed:
            continue

        try:
            detections = frame['detections']
        except KeyError:
            print('File error ', frame['file'])
            continue

        if len(detections) == 0:
            data = {'file': frame['file'],
                    'max_detection_conf': frame['max_detection_conf'],
                    'category': 0, 'conf': None, 'bbox_x': None,
                    'bbox_y': None, 'bbox_w': None, 'bbox_h': None}
            lst.append(data)

        else:
            for detection in detections:
                if (detection['conf'] > threshold):
                    data = {'file': frame['file'],
                            'max_detection_conf': frame['max_detection_conf'],
                            'category': detection['category'], 'conf': detection['conf'],
                            'bbox_x': np.clip(detection['bbox_x'], 0, 1),
                            'bbox_y': np.clip(detection['bbox_y'], 0, 1),
                            'bbox_w': np.clip(detection['bbox_w'], 0, 1),
                            'bbox_h': np.clip(detection['bbox_h'], 0, 1)}
                    lst.append(data)

    df = pd.DataFrame(lst)

    if manifest is not None:
        if file_col in manifest.columns:
            df = manifest.merge(df, left_on=file_col, right_on="file")
        else:
            raise ValueError("Please provide a manifest with a valid file_col to merge results onto.")

    if out_file:
        file_management.save_data(df, out_file)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deep learning model.')

    parser.add_argument('detector', help='Path to detector file')
    parser.add_argument('manifest', help='Path to manifest file')
    parser.add_argument('output_path', help='Path to output file')

    parser.add_argument('--model_type', nargs='?', help='Path to detector file', default='MDv5')
    parser.add_argument('--resize_width', nargs='?', help='Path to config file', default=MEGADETECTORv5_SIZE)
    parser.add_argument('--resize_height', nargs='?', help='Path to config file', default=MEGADETECTORv5_SIZE)
    parser.add_argument('--letterbox', nargs='?', help='Path to config file', default=True)
    parser.add_argument('--confidence_threshold', nargs='?', help='Path to config file', default=0.1)
    parser.add_argument('--file_col', nargs='?', help='Path to config file', default='frame')
    parser.add_argument('--batch_size', nargs='?', help='Path to config file', default=4)
    parser.add_argument('--num_workers', nargs='?', help='Path to config file', default=4)
    parser.add_argument('--device', nargs='?', help='Path to config file', default=get_device())

    args = parser.parse_args()

    detector = load_detector(args.detector, args.model_type)
    manifest = file_management.load_data(args.manifest)

    mdresults = detect(detector, manifest, args.resize_width, args.resize_height, args.letterbox,
           confidence_threshold=args.confidence_threshold, file_col=args.file_col,
           batch_size=args.batch_size, num_workers=args.num_workers, device=args.device)
    results = parse_detections(mdresults, manifest=manifest, out_file=args.output_path)
