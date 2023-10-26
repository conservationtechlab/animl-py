import pandas as pd
from os.path import isfile
import file_management
import numpy as np


def from_MD(results, manifest=None, out_file=None, buffer=0.02):
    """
    Converts numerical output from classifier to common name species label

    Args:
        - animals: dataframe of animal detections
        - predictions: output of the classifier model
        - class_file: species list associated with classifier outputs
        - out_file: path to save dataframeft
    Returns:
        - animals: dataframe containing species labels
    """
    if file_management.check_file(out_file):
        return file_management.load_data(out_file)

    if not isinstance(results, list):
        raise AssertionError("MD results input must be list")

    if len(results) == 0:
        raise AssertionError("'results' contains no detections")

    df = pd.DataFrame()
    for dictionary in results:

        detections = dictionary['detections']
        if len(detections) == 0:
            data = {'file': dictionary['file'],
                    'max_detection_conf': dictionary['max_detection_conf'],
                    'category': 0, 'conf': None, 'bbox1': None,
                    'bbox2': None, 'bbox3': None, 'bbox4': None}
            df = df.append(data, ignore_index=True)
        else:
            for detection in detections:
                bbox = detection['bbox']
                data = {'file': dictionary['file'],
                        'max_detection_conf': dictionary['max_detection_conf'],
                        'category': detection['category'], 'conf': detection['conf'],
                        'bbox1': bbox[0], 'bbox2': bbox[1],
                        'bbox3': bbox[2], 'bbox4': bbox[3]}
                df = df.append(data, ignore_index=True)

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


def from_classifier(animals, predictions, class_file, out_file=None):
    """
    Converts numerical output from classifier to common name species label

    Args:
        - animals: dataframe of animal detections
        - predictions: output of the classifier model
        - class_file: species list associated with classifier outputs
        - out_file: path to save dataframe
    Returns:
        - animals: dataframe containing species labels
    """
    if file_management.check_file(out_file):
        return file_management.load_data(out_file)

    if not isinstance(animals, pd.DataFrame):
        raise AssertionError("'animals' must be DataFrame.")

    if not isfile(class_file):
        raise AssertionError("The given class file does not exist.")

    # Format Classification results
    table = pd.read_table(class_file, sep=" ", index_col=0)

    animals['prediction'] = [table['x'].values[int(np.argmax(x))] for x in predictions]
    animals['confidence'] = [np.max(x) for x in predictions] * animals["conf"]

    return animals
