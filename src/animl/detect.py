'''
    Tools for Using MegaDetector

    Modified from https://github.com/agentmorris/MegaDetector/tree/master
'''
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
    From AgentMorris/MegaDetector
    Runs MegaDetector on a single image file.

    Args:
        - im_file (str): path to image file
        - detector: loaded model
        - confidence_threshold (float): only detections above this threshold are returned
        - quiet (bool): print debugging statements when false, defaults to true
        - image_size (int): overrides default image size, 1280
        - skip_image_resizing (bool): skip internal image resizing and rely on external resizing

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
                    confidence_threshold=0.005, quiet=True, image_size=None):
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

        # column from pd.DataFrame, expected input
        elif isinstance(image_file_names, pd.Series):
            pass

        # A single image file
        elif os.path.isfile(image_file_names):
            image_file_names = [image_file_names]
        else:
            raise ValueError('image_file_names is a string, but is not a directory, a json ' +
                             'list (.json), or an image file (png/jpg/jpeg/gif)')

    results = file_management.check_file(checkpoint_path)
    if not results:  # checkpoint comes back empty
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


def parse_MD(results, manifest=None, out_file=None, buffer=0.02, threshold=0):
    """
    Converts numerical output from classifier to common name species label

    Args:
        - results (list): md output dicts
        - manifest (pd.DataFrame): full file manifest, if not None, merge md predictions automatically
        - out_file (str): path to save dataframe
        - buffer (float): adjust bbox by percentage of img size to avoid clipping out of bounds
        - threshold (float): parse only detections above given confidence threshold

    Returns:
        - df (pd.DataFrame): formatted md outputs, one row per detection
    """
    if file_management.check_file(out_file):
        return file_management.load_data(out_file)

    if not isinstance(results, list):
        raise AssertionError("MD results input must be list")

    if len(results) == 0:
        raise AssertionError("'results' contains no detections")

    df = pd.DataFrame(columns=('file', 'max_detection_conf',
                               'category', 'conf', 'bbox1',
                               'bbox2', 'bbox3', 'bbox4'))
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

    if isinstance(manifest, pd.DataFrame):
        df = manifest.merge(df, left_on="Frame", right_on="file")

    if out_file:
        file_management.save_data(df, out_file)

    return df
