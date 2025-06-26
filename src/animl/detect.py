import json
import os
import time
import typing
import torch
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
from torch import tensor

from animl import file_management
from animl.generator import manifest_dataloader, image_to_tensor
from animl.utils import general

from ultralytics import YOLO


def load_detector(model_path, model_type, device=None):
    """
    Load Detector model from filepath

    Args:
        model_path (str): path to model file
        model_type (str): type of model expected ["MDV5", "MDV6", "YOLO"]
        device (str): specify to run on cpu or gpu

    Returns:
        object: loaded model object
    """
    if device is None:
        device = general.get_device()
    print('Device set to', device)

    if model_type == "MDV5":
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Compatibility fix that allows older YOLOv5 models with
        # newer versions of YOLOv5/PT
        if hasattr(checkpoint['model'], 'modules'):
            for m in checkpoint['model'].modules():
                t = type(m)
                if t is torch.nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                    m.recompute_scale_factor = None
        model = checkpoint['model'].float().fuse().eval()  # FP32 model
        model.model_type = "MDV5"
    elif model_type == "YOLO" or model_type == "MDV6":
        model = YOLO(model_path, task='detect')
        model.model_type = "YOLO"
    else:
        print(f"Please chose a supported model. Version {model_type} is not supported.")

    model.to(device)
    return model


def detect_batch(detector, image_file_names,
                 batch_size=1,
                 workers=1,
                 device=None,
                 checkpoint_path: typing.Optional[str] = None,
                 checkpoint_frequency: int = -1,
                 confidence_threshold: float = 0.1,
                 image_size: typing.Optional[int] = 1280,
                 file_col: str = 'Frame') -> typing.List[typing.Dict]:
    """
    Runs Detector model on a batches of image files

        Args:
            detector (object): preloaded detector model
            image_file_names (mult): list of image filenames, a single image filename,
                                a folder to recursively search for images in, or a .json file
                                containing a list of images.
            batch_size (int): size of each batch
            workers (int): number of processes to handle the data
            device (str): specify to run on cpu or gpu
            checkpoint_path (str): path to checkpoint file
            checkpoint_frequency (int): write results to checkpoint file every N images
            confidence_threshold (float): only detections above this threshold are returned
            image_size (int): overrides default image size, 1280
            file_col (str): column name containing file paths

        Returns:
            list: list of dicts, each dict represents detections on one image
    """
    if confidence_threshold is None:
        confidence_threshold = 0.01  # Defult from MegaDetector
    if checkpoint_frequency is None:
        checkpoint_frequency = -1
    elif checkpoint_frequency != -1:
        checkpoint_frequency = round(checkpoint_frequency/batch_size, 0)

    # check to make sure GPU is available if chosen
    if device is None:
        device = general.get_device()
    print('Device set to', device)

    # Handle the case where image_file_names is not yet actually a list
    if isinstance(image_file_names, str):
        # Find the images to score; images can be a directory, may need to recurse
        if os.path.isdir(image_file_names):
            image_dir = image_file_names
            image_file_names = file_management.build_file_manifest(image_dir, True)
            print('{} image files found in folder {}'.format(len(image_file_names), image_dir))

        # A json list of image paths
        elif os.path.isfile(image_file_names) and image_file_names.endswith('.json'):
            list_file = image_file_names
            with open(list_file) as f:
                image_file_names = json.load(f)
            print('Loaded {} image filenames from list file {}'.format(len(image_file_names), list_file))

        # A single image file
        elif os.path.isfile(image_file_names):
            image_file_names = [image_file_names]

        else:
            raise ValueError('image_file_names is a string, but is not a directory, a json ' +
                             'list (.json), or an image file (png/jpg/jpeg/gif)')

    # full manifest, select file_col
    elif isinstance(image_file_names, pd.DataFrame):
        image_file_names = image_file_names[file_col]

    # column from pd.DataFrame, expected input
    elif isinstance(image_file_names, pd.Series):
        pass
    else:
        raise ValueError('image_file_names is not a recognized object')

    # load checkpoint
    if file_management.check_file(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
            results = data['images']
    else:
        results = []

    # remove loaded images
    already_processed = set([i['file'] for i in results])
    image_file_names = set(image_file_names) - already_processed

    count = 0

    # create a data frame from image_file_names
    manifest = pd.DataFrame(image_file_names, columns=['file'])
    # create dataloader
    # TODO: letterbox if mdv5
    dataloader = manifest_dataloader(manifest, batch_size=batch_size,
                                     workers=workers, crop=False, normalize=True,
                                     resize_width=image_size,
                                     resize_height=image_size)

    print("Starting batch processing...")
    start_time = time.time()
    for batch_idx, batch_from_dataloader in tqdm(enumerate(dataloader), total=len(dataloader)):
        count += 1

        image_tensors = batch_from_dataloader[0]  # Tensor of images for the current batch
        current_image_paths = batch_from_dataloader[1]  # List of image names for the current batch

        # Run inference on the current batch of image_tensors
        if detector.model_type == "MDV5":
            prediction = detector(image_tensors.to(device))
            pred: list = prediction[0]
            pred = general.non_max_suppression(prediction=pred, conf_thres=0.1)
            results.extend(convert_raw_detections(pred, image_tensors, current_image_paths))

        else:
            prediction = detector.predict(source=image_tensors.to(device), conf=0.1, verbose=False)
            results.extend(convert_yolo_detections(prediction, current_image_paths))

        # Write a checkpoint if necessary
        if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
            print('Writing a new checkpoint after having processed {} images since last restart'.format(count*batch_size))

            assert checkpoint_path is not None
            # Back up any previous checkpoints, to protect against crashes while we're writing
            # the checkpoint file.
            checkpoint_tmp_path = None
            if os.path.isfile(checkpoint_path):
                checkpoint_tmp_path = str(checkpoint_path) + '_tmp'
                copyfile(checkpoint_path, checkpoint_tmp_path)

            # Write the new checkpoint
            with open(checkpoint_path, 'w') as f:
                json.dump({'images': results}, f, indent=1)

            # Remove the backup checkpoint if it exists
            if checkpoint_tmp_path is not None:
                os.remove(checkpoint_tmp_path)

    print(f"\nFinished batch processing. Total images processed: {len(results)} at {round(len(results)/(time.time() - start_time), 1)} img/s.")

    return results


def detect_single(detector, file_path, device=None,
                  confidence_threshold: float = 0.1,
                  image_size: typing.Optional[int] = 1280) -> typing.Dict:
    """
    Run Detector model on a single image

    Args:
        detector (obj): loaded model
        file_path (str): path to image file
        device (str): device used for inference "cuda", "cpu", etc
        confidence_threshold (float): only detections above this threshold are returned
        image_size (int): overrides default image size, 1280

    Returns:
        dict: model detections on one image
        see the 'images' key in
        https://github.com/agentmorris/MegaDetector/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    if not isinstance(file_path, str):
        raise TypeError(f"Expected str for file_path, got {type(file_path)}")
    if not isinstance(confidence_threshold, float):
        raise TypeError(f"Expected float for confidence_threshold, got {type(confidence_threshold)}")

    # check to make sure GPU is available if chosen
    if device is None:
        device = general.get_device()
    print('Device set to', device)

    # convert img path to tensor
    image_tensor = image_to_tensor(file_path, resize_width=image_size, resize_height=image_size)

    # Run inference on the image
    if detector.model_type == "MDV5":
        prediction = detector(image_tensor.to(device))
        pred: list = prediction[0]
        pred = general.non_max_suppression(prediction=pred, conf_thres=confidence_threshold)
        results = convert_raw_detections(pred, image_tensor, [file_path])

    else:
        prediction = detector.predict(source=image_tensor.to(device), conf=confidence_threshold, verbose=False)
        results = convert_yolo_detections(prediction, [file_path])

    return results


def convert_yolo_detections(predictions, image_paths):
    """
    Converts YOLO output into a nested list

    Args:
        predictions (list): YOLO detection output (list of dictionaries with detections for each file)
        image_paths (list): List of image file paths corresponding to predictions

    Returns:
        df (pd.DataFrame): Formatted YOLO outputs, one row per detection
    """
    results = []

    for i, pred in enumerate(predictions):
        file = image_paths[i]

        boxes = pred.boxes.xyxyn.cpu().numpy()  # Bounding box coordinates
        conf = pred.boxes.conf.cpu().numpy()  # Confidence scores
        category = pred.boxes.cls.cpu().numpy()  # Class labels as integers
        max_detection_conf = conf.max() if len(conf) > 0 else 0

        if len(conf) == 0:
            data = {'file': file,
                    'max_detection_conf': float(round(max_detection_conf, 4)),
                    'detections': []}
            results.append(data)

        else:
            detections = []
            for i in range(len(conf)):
                data = {'category': int(category[i]+1),
                        'conf': float(round(conf[i], 4)),
                        'bbox1': float(round(boxes[i][0], 4)),
                        'bbox2': float(round(boxes[i][1], 4)),
                        'bbox3': float(round(boxes[i][2]-boxes[i][0], 4)),
                        'bbox4': float(round(boxes[i][3]-boxes[i][1], 4))}
                detections.append(data)

            data = {'file': file,
                    'max_detection_conf': float(round(max_detection_conf, 4)),
                    'detections': detections}
            results.append(data)

    return results


def convert_raw_detections(predictions, image_tensors, image_paths):
    """
    Converts MDv5 output into a nested list

    Args:
        predictions (list): YOLO detection output (list of dictionaries with detections for each file)
        image_paths (list): List of image file paths corresponding to predictions

    Returns:
        df (pd.DataFrame): Formatted YOLO outputs, one row per detection
    """
    results = []

    # This is a loop over detection batches, which will always be length 1
    for i, det in enumerate(predictions):
        detections = []
        max_conf = 0.0

        if len(det):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = general.scale_coords(image_tensors[i].shape[1:], det[:, :4], image_tensors[i].shape[1:]).round()

            for *xyxy, conf, cls in reversed(det):
                # normalized center-x, center-y, width and height
                xywh = (general.xyxy2xywh(tensor(xyxy).view(1, 4)) / tensor(image_tensors[i].shape[1:])[[1, 0, 1, 0]]).view(-1).tolist()

                api_box = general.convert_yolo_to_xywh(xywh)
                api_box = general.truncate_float_array(api_box, precision=3)

                conf = general.truncate_float(conf.tolist(), precision=3)

                cls = int(cls.tolist()) + 1
                if cls not in (1, 2, 3):
                    raise KeyError(f'{cls} is not a valid class.')

                detections.append({
                    'category': str(cls),
                    'conf': conf,
                    'bbox1': api_box[0],
                    'bbox2': api_box[1],
                    'bbox3': api_box[2],
                    'bbox4': api_box[3],
                })
                max_conf = max(max_conf, conf)

            data = {'file': image_paths[i],
                    'max_detection_conf': float(round(max_conf, 4)),
                    'detections': detections}
            results.append(data)
        else:
            data = {'file': image_paths[i],
                    'max_detection_conf': float(round(max_conf, 4)),
                    'detections': []}
            results.append(data)

    return results


def parse_detections(results, manifest=None, out_file=None, buffer=0.02,
                     threshold=0, file_col="Frame"):
    """
    Converts listed output from detector to DataFrame

    Args:
        results (list): md output dicts
        manifest (pd.DataFrame): full file manifest, if not None, merge md predictions automatically
        out_file (str): path to save dataframe
        buffer (float): adjust bbox by percentage of img size to avoid clipping out of bounds
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
                                   'bbox1', 'bbox2', 'bbox3', 'bbox4'))
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
                    'category': 0, 'conf': None, 'bbox1': None,
                    'bbox2': None, 'bbox3': None, 'bbox4': None}
            lst.append(data)

        else:
            for detection in detections:
                if (detection['conf'] > threshold):
                    data = {'file': frame['file'],
                            'max_detection_conf': frame['max_detection_conf'],
                            'category': detection['category'], 'conf': detection['conf'],
                            'bbox1': min(max(detection['bbox1'], buffer), 1 - buffer),
                            'bbox2': min(max(detection['bbox2'], buffer), 1 - buffer),
                            'bbox3': min(max(detection['bbox3'], buffer), 1 - buffer),
                            'bbox4': min(max(detection['bbox4'], buffer), 1 - buffer)}
                    lst.append(data)

    df = pd.DataFrame(lst)

    if isinstance(manifest, pd.DataFrame):
        df = manifest.merge(df, left_on=file_col, right_on="file")

    if out_file:
        file_management.save_data(df, out_file)

    return df


# TODO: check if still necessary
def parse_YOLO(results, manifest=None, out_file=None, buffer=0.02, threshold=0, file_col="Frame", convert=True):
    """
    Converts YOLO detection results to a formatted DataFrame, similar to parse_MD.

    Args:
        - results (list): YOLO detection output (list of dictionaries with detections for each file)
        - manifest (pd.DataFrame): Full file manifest, if not None, merge YOLO predictions automatically
        - out_file (str): Path to save the resulting DataFrame
        - buffer (float): Adjust bbox by percentage of img size to avoid clipping out of bounds
        - threshold (float): Parse only detections above the given confidence threshold
        - file_col (str): Column name in the manifest to match with YOLO detection filenames

    Returns:
        - df (pd.DataFrame): Formatted YOLO outputs, one row per detection
    """
    # Load checkpoint if it exists
    if file_management.check_file(out_file):
        df = file_management.load_data(out_file)
        already_processed = set(df['file'])
    else:
        df = pd.DataFrame(columns=('file', 'max_detection_conf', 'category', 'conf',
                                   'bbox1', 'bbox2', 'bbox3', 'bbox4'))
        already_processed = set()

    # Ensure the results are in the expected format
    if not isinstance(results, list):
        raise AssertionError("YOLO results input must be a list.")

    if len(results) == 0:
        raise AssertionError("'results' contains no detections.")

    # Parse results into a list of dictionaries
    lst = []

    for frame in tqdm(results):
        # Skip already analyzed files
        if frame['file'] in already_processed:
            continue

        # Extract detections for the frame
        try:
            detections = frame['detections']
        except KeyError:
            print(f"File error: {frame['file']}")
            continue

        # Handle files with no detections
        if len(detections) == 0:
            data = {
                'file': frame['file'],
                'max_detection_conf': None,
                'category': 0,
                'conf': None,
                'bbox1': None,
                'bbox2': None,
                'bbox3': None,
                'bbox4': None,
            }
            lst.append(data)
        else:
            # Process each detection
            for detection in detections:
                if detection['conf'] > threshold:
                    data = {
                        'file': frame['file'],
                        'max_detection_conf': max(d['conf'] for d in detections),  # Maximum confidence for the frame
                        'category': detection['class'],  # YOLO uses "class" for category
                        'conf': detection['conf'],  # Confidence score
                        'bbox1': detection['bbox1'],
                        'bbox2': detection['bbox2'],
                        'bbox3': detection['bbox3'],
                        'bbox4': detection['bbox4'],
                    }
                    if convert:
                        image_size = general.get_image_size(frame['file'])
                        data['bbox1'], data['bbox2'], data['bbox3'], data['bbox4'] = general.absolute_to_relative(
                            [data['bbox1'], data['bbox2'], data['bbox3'], data['bbox4']], image_size
                        )
                    lst.append(data)

    # Create DataFrame from the parsed data
    df = pd.DataFrame(lst)

    # Merge with manifest if provided
    if isinstance(manifest, pd.DataFrame):
        df = manifest.merge(df, left_on=file_col, right_on="file")

    # Save to file if specified
    if out_file:
        file_management.save_data(df, out_file)

    return df
