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
from animl.model_architecture import MEGADETECTORv5_SIZE
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
        model_type (str): type of model expected ["MDV5", "MDV6", "YOLO", "ONNX"]
        device (str): specify to run on cpu or gpu

    Returns:
        object: loaded model object
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model_type = model_type.lower()

    # YOLOv5/MDv5
    if model_type in {"mdv5", "yolov5"}:
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
    elif model_type in {"yolo", "mdv6"}:
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

    # assume yolo model if not already specified
    if 'model_type' not in detector.__dict__:
        detector.model_type = "yolo"

    # Single image filepath
    if isinstance(image_file_names, str):
        # convert img path to tensor
        batch_from_dataloader = image_to_tensor(image_file_names, letterbox=letterbox,
                                                resize_width=resize_width, resize_height=resize_height)
        if batch_from_dataloader is None:
            print(f"Error loading image {image_file_names}. Skipping.")
            return []
        batch_tensors = batch_from_dataloader[0]  # Tensor of images for the current batch
        batch_paths = batch_from_dataloader[1]  # List of image names for the current batch
        batch_sizes = batch_from_dataloader[2]  # List of original image sizes for the current batch

        batch_frames = [0]  # single image, frame 0

        if detector.model_type in {"yolov5", "mdv5"}:
            # check to make sure GPU is available if chosen
            device = get_torch_device(user_set=device)
            # letterboxing should be true
            prediction = detector(batch_tensors.to(device))
            pred: list = prediction[0]
            pred = non_max_suppression(prediction=pred, conf_thres=confidence_threshold)
            results = _convert_yolo_detections(pred, batch_tensors, batch_paths, batch_frames,
                                               batch_sizes, letterbox, detector.model_type)
        elif detector.model_type == "onnx":
            input_name = detector.get_inputs()[0].name
            providers = get_onnx_device(user_set=device)
            if 'CUDAExecutionProvider' in providers:
                outputs = detector.run(None, {input_name: batch_tensors.numpy()})[0]
            # default to cpu
            else:
                outputs = detector.run(None, {input_name: batch_tensors.cpu().numpy()})[0]

            # Process outputs to match expected format
            results = _convert_onnx_detections(outputs, batch_tensors, batch_paths,
                                               batch_frames, batch_sizes, letterbox)
        else:
            pred = detector.predict(source=batch_tensors.to(device), conf=confidence_threshold, verbose=False)
            results = _convert_yolo_detections(pred, batch_tensors, batch_paths, batch_frames,
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

    for _, batch_from_dataloader in tqdm(enumerate(dataloader), total=len(dataloader)):
        collated, failed = batch_from_dataloader
        failed_files.extend(failed)

        if collated is None:  # entire batch was bad
            continue
        count += 1

        batch_tensors = collated[0]  # Tensor of images for the current batch
        batch_paths = collated[1]  # List of image names for the current batch
        batch_frames = collated[2]  # List of frame numbers for the current batch
        batch_sizes = collated[3]  # List of original image sizes for the current batch

        # Run inference on the current batch of image_tensors
        if detector.model_type in {"yolov5", "mdv5"}:
            # letterboxing should be true
            prediction = detector(batch_tensors.to(device))
            pred: list = prediction[0]
            pred = non_max_suppression(prediction=pred, conf_thres=confidence_threshold)
            # convert to normalized xywh
            results.extend(_convert_yolo_detections(pred, batch_tensors, batch_paths, batch_frames,
                                                    batch_sizes, letterbox, detector.model_type))
        elif detector.model_type == "onnx":
            input_name = detector.get_inputs()[0].name
            if device == "cpu":
                outputs = detector.run(None, {input_name: batch_tensors.cpu().numpy()})[0]
            else:
                outputs = detector.run(None, {input_name: batch_tensors.numpy()})[0]

            # Process outputs to match expected format
            results.extend(_convert_onnx_detections(outputs, batch_tensors, batch_paths, batch_frames,
                                                    batch_sizes, letterbox))
        # standard yolo model (v6+)
        else:
            pred = detector.predict(source=batch_tensors.to(device), conf=confidence_threshold, verbose=False)
            # convert to normalized xywh
            results.extend(_convert_yolo_detections(pred, batch_tensors, batch_paths, batch_frames,
                                                    batch_sizes, letterbox, detector.model_type))

        # Write a checkpoint if necessary
        if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
            print('Writing a new checkpoint after having processed {} images since last restart'.format(count*batch_size))
            _save_detection_checkpoint(checkpoint_path, results)

    print(f"\nFinished detection. Total images processed: {len(results)} at {round(len(results)/(time.time() - start_time), 1)} img/s.")
    if checkpoint_path:
        _save_detection_checkpoint(checkpoint_path, results)

    return results, failed_files


def _convert_onnx_detections(predictions: list,
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
        max_detection_conf = float(round(conf.max(), 4)) if len(conf) > 0 else None

        if len(conf) == 0:
            data = {'filepath': str(image_paths[i]),
                    'frame': int(image_frames[i]),
                    'max_detection_conf': max_detection_conf,
                    'detections': []}
            results.append(data)
        else:
            detections = []
            for j in range(len(conf)):
                bbox = normalize_bbox(boxes[j], image_tensors[i].shape[1:])
                bbox = _xyxy_to_xywh(bbox)
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
                    'category': int(category[j])
                }
                detections.append(detection)
            data = {'filepath': str(image_paths[i]),
                    'frame': int(image_frames[i]),
                    'max_detection_conf': max_detection_conf,
                    'detections': detections}
            results.append(data)

    return results


def _convert_yolo_detections(predictions: list,
                             image_tensors: list,
                             image_paths: list,
                             image_frames: list,
                             image_sizes: list,
                             letterbox: bool,
                             model_type: str) -> pd.DataFrame:
    """
    Converts YOLO output into a nested list.

    Args:
        predictions (list): YOLO detection output (list of dictionaries with detections for each file)
        image_tensors (list): array of image tensors from mdv6 output
        image_paths (list): List of image file paths corresponding to predictions
        image_frames (list): List of frame numbers corresponding to predictions
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
        if model_type in {"mdv5", "yolov5"}:
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            boxes = pred[:, :4]  # Bounding box coordinates
            conf = pred[:, 4]  # Confidence scores
            category = pred[:, 5]  # Class labels as integers
            max_detection_conf = float(round(conf.max(), 4)) if len(conf) > 0 else None
        # YOLOv6+
        elif model_type in {"yolo", "mdv6", "mdv1000"}:
            boxes = pred.boxes.xyxyn.cpu().numpy()  # Bounding box coordinates
            conf = pred.boxes.conf.cpu().numpy()  # Confidence scores
            category = pred.boxes.cls.cpu().numpy()  # Class labels as integers
            max_detection_conf = float(round(conf.max(), 4)) if len(conf) > 0 else None
        else:
            print(f"Please chose a supported model. Version {model_type} is not supported.")
            return None

        # no detections
        if len(conf) == 0:
            data = {'filepath': str(file),
                    'frame': int(image_frames[i]),
                    'max_detection_conf': max_detection_conf,
                    'detections': []}
            results.append(data)
        # detections
        else:
            detections = []
            for j in range(len(conf)):
                # YOLOv5/MDv5
                if model_type in {'mdv5', 'yolov5'}:  # xyxy absolute
                    bbox = normalize_bbox(boxes[j], image_tensors[i].shape[1:])
                    bbox = _xyxy_to_xywh(bbox)
                # YOLOv6+
                elif model_type in {'yolo', "mdv6", "mdv1000"}:  # xyxy relative
                    bbox = _xyxy_to_xywh(boxes[j])
                else:
                    print(f"Please chose a supported model. Version {model_type} is not supported.")
                    return None

                # increase md categories by 1
                if model_type in {"mdv5", "mdv6", "mdv1000"}:
                    category[j] += 1

                if letterbox:
                    bbox = scale_letterbox(bbox, image_tensors[i].shape[1:], image_sizes[i, :])

                detection = {'category': int(category[j]),
                             'conf': float(round(conf[j], 4)),
                             'bbox_x': float(round(bbox[0], 4)),
                             'bbox_y': float(round(bbox[1], 4)),
                             'bbox_w': float(round(bbox[2], 4)),
                             'bbox_h': float(round(bbox[3], 4))}
                detections.append(detection)

            data = {'filepath': str(file),
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
    if isinstance(results, tuple):
        failed_files = results[1]
        results = results[0]
        if len(failed_files) > 0:
            print(f"Warning: {len(failed_files)} files failed to load during detection and will be excluded from results.")
            if out_file is not None:
                with (Path(out_file).parent / "detection_failed_files.txt").open("w") as f:
                    for item in failed_files:
                        f.write(f"{item}\n")
    else:
        failed_files = None

    # check results format
    if not isinstance(results, list):
        raise TypeError("MD results input must be list")
    if len(results) == 0:
        raise AssertionError("'results' contains no detections")

    # load results from file if they have already been parsed
    if file_management.check_file(out_file, output_type="Detections"):
        return file_management.load_data(out_file)

    lst = []
    for frame in tqdm(results):
        try:
            detections = frame['detections']
        except KeyError:
            print('File error ', frame['filepath'])
            continue

        if len(detections) == 0:
            data = {'filepath': frame['filepath'],
                    'frame': frame['frame'],
                    'max_detection_conf': frame['max_detection_conf'],
                    'category': None, 'conf': None, 'bbox_x': None,
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

    mdresults, failed_files = detect(detector, manifest, args.resize_width, args.resize_height,
                                     args.letterbox, confidence_threshold=args.confidence_threshold,
                                     file_col=args.file_col, batch_size=args.batch_size,
                                     num_workers=args.num_workers, device=args.device)
    results = parse_detections(mdresults, manifest=manifest, out_file=args.output_path)
