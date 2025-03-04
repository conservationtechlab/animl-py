'''
    Tools for Using MegaDetector

    Modified from https://github.com/agentmorris/MegaDetector/tree/master
'''
import typing
from tqdm import tqdm
import json
import os
import pandas as pd
from shutil import copyfile
from PIL import Image

from animl import file_management
import torch
from PytorchWildlife.models import detection as pw_detection


def process_image(im_file: str,
                  detector: object,
                  confidence_threshold: float,
                  quiet: bool = True,
                  image_size: typing.Optional[int] = None,
                  skip_image_resize: bool = False) -> typing.Dict:
    """
    Runs MegaDetector on a single image file.
    """
    if not isinstance(im_file, str):
        raise TypeError(f"Expected str for im_file, got {type(im_file)}")
    if not isinstance(confidence_threshold, float):
        raise TypeError(f"Expected float for confidence_threshold, got {type(confidence_threshold)}")

    if not quiet:
        print('Processing image {}'.format(im_file))
    # open the file
    try:
        image = Image.open(im_file).convert(mode='RGB')
        image.load()
    except Exception as e:
        if not quiet:
            print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
        result = {
            'img_id': im_file,
            'failure': 'Failure image access'
        }
        return result
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detection_model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-e")

        result = detection_model.single_image_detection(im_file)
        print(result)
        if result is None:
            raise ValueError("No result returned from detector")
        if 'file' not in result:
            result['file'] = im_file
    except Exception as e:
        if not quiet:
            print('Image {} cannot be processed. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': 'Failure inference'
        }
        return result

    # Explicitly return the result on successful processing
    return result





def detect_MD_batch(detector: object,
                    image_file_names: typing.List[str],
                    checkpoint_path: typing.Optional[str] = None,
                    checkpoint_frequency: int = -1,
                    confidence_threshold: float = 0.1,
                    quiet: bool = True,
                    image_size: typing.Optional[int] = None,
                    file_col: str = 'Frame') -> typing.List[typing.Dict]:
    """
    From AgentMorris/MegaDetector
    Runs MegaDetector on a batch of image files.

        Args:
            - detector: preloaded md model
            - image_file_names (mult): list of image filenames, a single image filename,
                                a folder to recursively search for images in, or a .json file
                                containing a list of images.
            - checkpoint_path (str): path to checkpoint file
            - checkpoint_frequency (int): write results to checkpoint file every N images
            - confidence_threshold (float): only detections above this threshold are returned
            - quiet (bool): print debugging statements when false, defaults to true
            - image_size (int): overrides default image size, 1280
            - file_col (str): column name containing file paths

        Returns:
            - results: list of dict, each dict represents detections on one image
    """
    if confidence_threshold is None:
        confidence_threshold = 0.005  # Defult from MegaDetector

    if checkpoint_frequency is None:
        checkpoint_frequency = -1

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

    if file_management.check_file(checkpoint_path):  # checkpoint comes back empty
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
            results = data['images']
    else:
        results = []

    # remove loaded images
    already_processed = set([i['img_id'] for i in results])
    image_file_names = set(image_file_names) - already_processed

    count = 0

    for im_file in tqdm(image_file_names):
        # process single image
        count += 1
        result = process_image(im_file, detector,
                               confidence_threshold, quiet=quiet,
                               image_size=image_size)
        results.append(result)

        # Write a checkpoint if necessary
        if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
            print('Writing a new checkpoint after having processed {} images since last restart'.format(count))

            assert checkpoint_path is not None
            # Back up any previous checkpoints, to protect against crashes while we're writing
            # the checkpoint file.
            checkpoint_tmp_path = None
            if os.path.isfile(checkpoint_path):
                checkpoint_tmp_path = checkpoint_path + '_tmp'
                copyfile(checkpoint_path, checkpoint_tmp_path)

            # Write the new checkpoint
            with open(checkpoint_path, 'w') as f:
                json.dump({'images': results}, f, indent=1)

            # Remove the backup checkpoint if it exists
            if checkpoint_tmp_path is not None:
                os.remove(checkpoint_tmp_path)
    print(results)
    return results

def parse_MD(results: typing.List[dict],
             manifest: typing.Optional[pd.DataFrame] = None,
             out_file: typing.Optional[str] = None,
             buffer: float = 0.02,
             threshold: float = 0,
             file_col: str = "Frame") -> pd.DataFrame:
    """
    Converts numerical output from classifier to common name species label.
    """
    if results is None:
        raise ValueError("No detection results provided (results is None).")

    # Convert new format to expected format if necessary
    if results and 'file' in results[0]:
        results = convert_new_format(results)

    # Load checkpoint if available
    if file_management.check_file(out_file):
        df = file_management.load_data(out_file)
        already_processed = set([row['file'] for row in df])
    else:
        # Predefine the DataFrame columns
        df = pd.DataFrame(columns=['file', 'max_detection_conf', 'category', 'conf', 'bbox1', 'bbox2', 'bbox3', 'bbox4'])
        already_processed = set()

    if not isinstance(results, list):
        raise AssertionError("MD results input must be list")

    if len(results) == 0:
        raise AssertionError("'results' contains no detections")

    lst = []
    for data in results:
        lst.append(data)
    # Create DataFrame with defined columns so that even if lst is empty,
    # the DataFrame will have the 'file' column
    df = pd.DataFrame(lst, columns=['file', 'max_detection_conf', 'category', 'conf', 'bbox1', 'bbox2', 'bbox3', 'bbox4'])

    if isinstance(manifest, pd.DataFrame):
        df = manifest.merge(df, left_on=file_col, right_on="file")

    if out_file:
        file_management.save_data(df, out_file)

    return df

def convert_new_format(new_results):
    """
    Converts new output format into a list of dictionaries compatible with parse_MD.

    Args:
        new_results (list): List of dicts in the new format.

    Returns:
        List of dicts in the format expected by parse_MD.
    """
    converted = []
    for frame in new_results:
        # Map 'img_id' to 'file'
        file_path = frame.get('img_id')
        detections_obj = frame.get('detections')

        # If no detections available, add an entry with empty values.
        if detections_obj is None or detections_obj.xyxy.size == 0:
            converted.append({
                'file': file_path,
                'max_detection_conf': None,
                'category': 0,
                'conf': None,
                'bbox1': None,
                'bbox2': None,
                'bbox3': None,
                'bbox4': None
            })
            continue


        xyxy = detections_obj.xyxy  # shape: (N, 4)
        conf_array = detections_obj.confidence.flatten()  # shape: (N,)
        class_ids = detections_obj.class_id.flatten()       # shape: (N,)

        max_conf = float(conf_array.max()) if conf_array.size > 0 else None

        # Create one dict per detection.
        for i in range(len(conf_array)):
            detection_dict = {
                'file': file_path,
                'max_detection_conf': max_conf,
                'category': int(class_ids[i]),
                'conf': float(conf_array[i]),

                'bbox1': float(xyxy[i][0]),
                'bbox2': float(xyxy[i][1]),
                'bbox3': float(xyxy[i][2]),
                'bbox4': float(xyxy[i][3])
            }
            print(detection_dict    )
            converted.append(detection_dict)
    return converted
