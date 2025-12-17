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
        model_type (str): type of model expected ["MDV5", "MDV6", "YOLO", "ONNX"]
        device (str): specify to run on cpu or gpu

    Returns:
        object: loaded model object
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    if device is None:
        device = get_device()

    # YOLOv5/MDv5
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
        model.to(device)
        return model
    # YOLOv6+
    elif model_type.lower() in {"yolo", "mdv6"}:
        model = YOLO(model_path, task='detect')
        model.model_type = "yolo"
        model.to(device)
        return model
    # ONNX model
    elif model_type.lower() in {"onnx"}:
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        model = ort.InferenceSession(model_path, providers=providers)
        model.model_type = "onnx"
        return model
    else:
        print(f"Please chose a supported model. Version {model_type} is not supported.")
        return None


def detect(detector,
           image_file_names,
           resize_width: int,
           resize_height: int,
           letterbox: bool = True,
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

    # check to make sure GPU is available if chosen
    if device is None:
        device = get_device()

    # Single image filepath
    if isinstance(image_file_names, str):
        # convert img path to tensor
        batch_from_dataloader = image_to_tensor(image_file_names, letterbox=letterbox,
                                                resize_width=resize_width, resize_height=resize_height)
        batch_tensors = batch_from_dataloader[0]  # Tensor of images for the current batch
        batch_paths = batch_from_dataloader[1]  # List of image names for the current batch
        batch_sizes = batch_from_dataloader[2]  # List of original image sizes for the current batch

        batch_frames = [0]  # single image, frame 0

        if detector.model_type == "yolov5":
            # letterboxing should be true
            prediction = detector(batch_tensors.to(device))
            pred: list = prediction[0]
            pred = non_max_suppression(prediction=pred, conf_thres=confidence_threshold)
            results = convert_yolo_detections(pred, batch_tensors, batch_paths, batch_frames,
                                              batch_sizes, letterbox, detector.model_type)
        elif detector.model_type == "onnx":
            input_name = detector.get_inputs()[0].name
            if device == "cpu":
                outputs = detector.run(None, {input_name: batch_tensors.cpu().numpy()})[0]
            else:
                outputs = detector.run(None, {input_name: batch_tensors.numpy()})[0]
            # Process outputs to match expected format
            results = convert_onnx_detections(outputs, batch_tensors, batch_paths,
                                              batch_frames, batch_sizes, letterbox)
        else:
            pred = detector.predict(source=batch_tensors.to(device), conf=confidence_threshold, verbose=False)
            results = convert_yolo_detections(pred, batch_tensors, batch_paths, batch_frames,
                                              batch_sizes, letterbox, detector.model_type)
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

    count = 0

    # create dataloader
    dataloader = manifest_dataloader(manifest, batch_size=batch_size,
                                     num_workers=num_workers, crop=False,
                                     normalize=True, letterbox=letterbox,
                                     resize_width=resize_width,
                                     resize_height=resize_height)

    start_time = time.time()
    for _, batch_from_dataloader in tqdm(enumerate(dataloader), total=len(dataloader)):
        count += 1

        batch_tensors = batch_from_dataloader[0]  # Tensor of images for the current batch
        batch_paths = batch_from_dataloader[1]  # List of image names for the current batch
        batch_frames = batch_from_dataloader[2]  # List of frame numbers for the current batch
        batch_sizes = batch_from_dataloader[3]  # List of original image sizes for the current batch

        # Run inference on the current batch of image_tensors
        if detector.model_type == "yolov5":
            # letterboxing should be true
            prediction = detector(batch_tensors.to(device))
            pred: list = prediction[0]
            pred = non_max_suppression(prediction=pred, conf_thres=confidence_threshold)
            # convert to normalized xywh
            results.extend(convert_yolo_detections(pred, batch_tensors, batch_paths, batch_frames,
                                                   batch_sizes, letterbox, detector.model_type))
        elif detector.model_type == "onnx":
            input_name = detector.get_inputs()[0].name
            if device == "cpu":
                outputs = detector.run(None, {input_name: batch_tensors.cpu().numpy()})[0]
            else:
                outputs = detector.run(None, {input_name: batch_tensors.numpy()})[0]

            # Process outputs to match expected format
            results.extend(convert_onnx_detections(outputs, batch_tensors, batch_paths, batch_frames,
                                                   batch_sizes, letterbox))
        # standard yolo model (v6+)
        else:
            pred = detector.predict(source=batch_tensors.to(device), conf=confidence_threshold, verbose=False)
            # convert to normalized xywh
            results.extend(convert_yolo_detections(pred, batch_tensors, batch_paths, batch_frames,
                                                   batch_sizes, letterbox, detector.model_type))

        # Write a checkpoint if necessary
        if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
            print('Writing a new checkpoint after having processed {} images since last restart'.format(count*batch_size))
            file_management.save_detection_checkpoint(checkpoint_path, results)

    print(f"\nFinished detection. Total images processed: {len(results)} at {round(len(results)/(time.time() - start_time), 1)} img/s.")
    if checkpoint_path:
        file_management.save_detection_checkpoint(checkpoint_path, results)

    return results


def convert_onnx_detections(predictions: list,
                            image_tensors: list,
                            image_paths: list,
                            image_frames: list,
                            image_sizes: list,
                            letterbox: bool) -> pd.DataFrame:
    # Process ONNX predictions
    results = []

    for i, pred in enumerate(predictions):

        boxes = pred[:, :4]  # Bounding box coordinates
        conf = pred[:, 4]  # Confidence scores
        category = pred[:, 5]  # Class labels as integers
        max_detection_conf = conf.max() if len(conf) > 0 else 0

        if len(conf) == 0:
            data = {'filepath': str(image_paths[i]),
                    'frame': int(image_frames[i]),
                    'max_detection_conf': float(round(max_detection_conf, 4)),
                    'detections': []}
            results.append(data)
        else:
            detections = []
            for j in range(len(conf)):
                bbox = normalize_boxes(boxes[j], image_tensors[i].shape[1:])
                bbox = xyxy2xywh(bbox)
                if bbox.all() == 0:
                    continue

                if letterbox:
                    bbox = scale_letterbox(bbox, image_tensors[i].shape[1:], image_sizes[i, :])

                detection = {
                    'bbox_x': float(round(bbox[0], 4)),
                    'bbox_y': float(round(bbox[1], 4)),
                    'bbox_w': float(round(bbox[2], 4)),
                    'bbox_h': float(round(bbox[3], 4)),
                    'conf': float(round(conf[j], 4)),
                    'category': int(category[j]+1)
                }
                detections.append(detection)
            data = {'filepath': str(image_paths[i]),
                    'frame': int(image_frames[i]),
                    'max_detection_conf': float(round(max_detection_conf, 4)),
                    'detections': detections}
            results.append(data)

    return results


def convert_yolo_detections(predictions: list,
                            image_tensors: list,
                            image_paths: list,
                            image_frames: list,
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
    if isinstance(image_frames, torch.Tensor):
        image_frames = image_frames.cpu().numpy()

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
            data = {'filepath': str(file),
                    'frame': int(image_frames[i]),
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

            data = {'filepath': str(file),
                    'frame': int(image_frames[i]),
                    'max_detection_conf': float(round(max_detection_conf, 4)),
                    'detections': detections}
            results.append(data)

    return results


def parse_detections(results: list,
                     manifest: Optional[pd.DataFrame] = None,
                     out_file: Optional[str] = None,
                     threshold: float = 0,
                     file_col: str = "filepath"):
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
    if file_management.check_file(out_file, output_type="Detections"):  # checkpoint comes back empty
        df = file_management.load_data(out_file)
        already_processed = set([row['filepath'] for row in df])

    else:
        df = pd.DataFrame(columns=('filepath', 'frame', 'max_detection_conf', 'category', 'conf',
                                   'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'))
        already_processed = set()

    if not isinstance(results, list):
        raise AssertionError("MD results input must be list")

    if len(results) == 0:
        raise AssertionError("'results' contains no detections")

    lst = []

    for frame in tqdm(results):

        # pass if already analyzed
        if frame['filepath'] in already_processed:
            continue

        try:
            detections = frame['detections']
        except KeyError:
            print('File error ', frame['filepath'])
            continue

        if len(detections) == 0:
            data = {'filepath': frame['filepath'],
                    'frame': frame['frame'],
                    'max_detection_conf': frame['max_detection_conf'],
                    'category': 0, 'conf': None, 'bbox_x': None,
                    'bbox_y': None, 'bbox_w': None, 'bbox_h': None}
            lst.append(data)

        else:
            for detection in detections:
                if (detection['conf'] > threshold):
                    data = {'filepath': frame['filepath'],
                            'frame': frame['frame'],
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
            df = manifest.merge(df, left_on=[file_col, 'frame'], right_on=["filepath", "frame"], how='left')
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
