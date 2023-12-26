"""
Object Detection Module

This module provides functions for using MegaDetector

Modified from https://github.com/agentmorris/MegaDetector/tree/main
"""
from tqdm import tqdm
import json
import os
import pandas as pd
from shutil import copyfile
from PIL import Image
from . import file_management


def process_image(im_file, detector, confidence_threshold, quiet=True,
                  image_size=None, skip_image_resize=False):
    """
        Runs MegaDetector on a single image file.

        Args
            - im_file: str, path to image file
            - detector: loaded model
            - confidence_threshold: float, only detections above this threshold are returned
            - quiet: toggle for warning statements, defaults to true
            - image_size: if resizing, width in px
            - skip_image_resize: whether to skip internal image resizing and rely on external resizing

        Returns:
            - result: dict representing detections on one image
            see the 'images' key in
            https://github.com/agentmorris/MegaDetector/tree/master/api/batch_processing#batch-processing-api-output-format
    """
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
            'file': im_file,
            'failure': 'Failure image access'
        }
        return result
    # run MD
    try:
        result = detector.generate_detections_one_image(image, im_file,
                                                        confidence_threshold=confidence_threshold,
                                                        image_size=image_size,
                                                        skip_image_resize=skip_image_resize)
    except Exception as e:
        if not quiet:
            print('Image {} cannot be processed. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': 'Failure inference'
        }
        return result

    return result


def detect_MD_batch(detector, image_file_names, checkpoint_path=None, checkpoint_frequency=-1,
                    confidence_threshold=0.005, results=None, quiet=True, image_size=None):
    """
        Runs MegaDetector on a batch of images

        Args
            - detector: preloaded md model
            - image_file_names: list of strings (image filenames), a single image filename,
                                a folder to recursively search for images in, or a .json file
                                containing a list of images.
            - checkpoint_path: str, path to JSON checkpoint file
            - checkpoint_frequency: int, write results to JSON checkpoint file every N images
            - confidence_threshold: float, only detections above this threshold are returned
            - results: list of dict, existing results loaded from checkpoint
            - quiet: toggle for warning statements, defaults to true
            - image_size: if resizing, width in px

        Returns
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

        # Dataframe, expected input
        elif isinstance(image_file_names, pd.Series):
            pass

        # A single image file
        elif os.path.isfile(image_file_names):
            image_file_names = [image_file_names]
        else:
            raise ValueError('image_file_names is a string, but is not a directory, a json ' +
                             'list (.json), or an image file (png/jpg/jpeg/gif)')

    if results is None:
        results = []

    already_processed = set([i['Frame'] for i in results])

    count = 0
    for im_file in tqdm(image_file_names):
        # Will not add additional entries not in the starter checkpoint
        if im_file in already_processed:
            if not quiet:
                print('Bypassing image {}'.format(im_file))
            continue

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

    return results


def parse_MD(results, out_file=None, buffer=0.02, threshold=0):
    """
        Converts numerical output from classifier to common name species label

        Args:
            - results: dataframe of animal detections
            - out_file: path to save dataframe
            - buffer: adjust bbox by fraction of total width to avoid clipping out of image
            - threshold: lower bound for bbox confidence threshold

        Returns:
            - results: dataframe containing species labels
    """
    if file_management.check_file(out_file):
        return file_management.load_data(out_file)

    if not isinstance(results, list):
        raise AssertionError("MD results input must be list")

    if len(results) == 0:
        raise AssertionError("'results' contains no detections")

    df = pd.DataFrame(columns=('file', 'max_detection_conf', 'category', 'conf',
                               'bbox1', 'bbox2', 'bbox3', 'bbox4'))
    for frame in tqdm(results):

        try:
            detections = frame['detections']
        except KeyError:
            print('File error ', frame['file'])
            continue
        if len(detections) == 0:
            data = {'file': [frame['file']],
                    'max_detection_conf': [frame['max_detection_conf']],
                    'category': [0], 'conf': [None], 'bbox1': [None],
                    'bbox2': [None], 'bbox3': [None], 'bbox4': [None]}
            df = pd.concat([df, pd.DataFrame(data)]).reset_index(drop=True)

        else:
            for detection in detections:
                bbox = detection['bbox']
                if (detection['conf'] > threshold):
                    data = {'file': [frame['file']],
                            'max_detection_conf': [frame['max_detection_conf']],
                            'category': [detection['category']], 'conf': [detection['conf']],
                            'bbox1': [bbox[0]], 'bbox2': [bbox[1]],
                            'bbox3': [bbox[2]], 'bbox4': [bbox[3]]}
                    df = pd.concat([df, pd.DataFrame(data)]).reset_index(drop=True)

    # adjust boxes with 2% buffer from image edge
    df.loc[df["bbox1"] > (1 - buffer), "bbox1"] = (1 - buffer)
    df.loc[df["bbox2"] > (1 - buffer), "bbox2"] = (1 - buffer)
    df.loc[df["bbox3"] > (1 - buffer), "bbox3"] = (1 - buffer)
    df.loc[df["bbox4"] > (1 - buffer), "bbox4"] = (1 - buffer)

    df.loc[df["bbox1"] < buffer, "bbox1"] = buffer
    df.loc[df["bbox2"] < buffer, "bbox2"] = buffer
    df.loc[df["bbox3"] < buffer, "bbox3"] = buffer
    df.loc[df["bbox4"] < buffer, "bbox4"] = buffer

    df['category'] = df['category'].astype(int)

    if out_file:
        file_management.save_data(df, out_file)

    return df
