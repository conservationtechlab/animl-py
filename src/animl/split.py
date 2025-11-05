"""
Tools for splitting the data for different workflows

@ Kyra Swanson 2023
"""
import pandas as pd
import numpy as np
from math import fsum
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split

from animl.file_management import save_data


def get_animals(manifest: pd.DataFrame):
    """
    Pulls MD animal detections for classification

    Args:
        manifest (pd.DataFrame): DataFrame containing one row for every MD detection

    Returns:
        subset of manifest containing only animal detections
    """
    return manifest[manifest['category'].astype(int) == 1].reset_index(drop=True)


def get_empty(manifest: pd.DataFrame):
    """
    Pulls MD non-animal detections

    Args:
        manifest (pd.DataFrame): DataFrame containing one row for every MD detection

    Returns:
        otherdf: subset of manifest containing empty, vehicle and human detections
        with added prediction and confidence columns
    """
    # Removes all images that MegaDetector gave no detection for
    otherdf = manifest[manifest['category'].astype(int) != 1].reset_index(drop=True)
    otherdf['prediction'] = otherdf['category'].astype(int)

    # Numbers the class of the non-animals correctly
    if not otherdf.empty:
        otherdf['prediction'] = otherdf['prediction'].replace(2, "human")
        otherdf['prediction'] = otherdf['prediction'].replace(3, "vehicle")
        otherdf['prediction'] = otherdf['prediction'].replace(0, "empty")
        otherdf['confidence'] = otherdf['conf']
        otherdf['confidence'] = otherdf['confidence'].replace(np.nan, 1)  # correct empty conf

    else:
        otherdf = pd.DataFrame(columns=manifest.columns.values)

    return otherdf


# TODO: IMPROVE
def train_val_test(manifest: pd.DataFrame,
                   out_dir: Optional[str] = None,
                   label_col: str = "class",
                   file_col: str = 'filepath',
                   groupby_col: Optional[list] = None,
                   percentage: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                   seed: Optional[int] = None):
    '''
    Splits the manifest into Training, Validation, and Test Datasets for training

    Credit: Unduwap Kandage-Don

    Args:
        manifest (DataFrame): list of files to split for training
        out_dir (str): location to save split lists to
        label_col (str): column name containing class labels
        file_col (str): column containing file paths
        groupby_col (list): columns to group by before splitting, ie station
        percentage (tuple): fraction of data dedicated to train-val-test
        seed (int): RNG seed, if none will pick one at random within [0,100]

    Returns:
        train manifest
        validate manifest
        test manifest
        stats file
    '''
    if seed is None:
        seed = np.random.randint(0, 100)
    print(f"RNG seed: {seed}")

    # check percentages add up to 1
    if not (fsum(percentage) == 1):
        print("Invalid percentages")

    # only one label per file 
    manifest.drop_duplicates(subset=[file_col], inplace=True)

    labelCt = manifest[label_col].value_counts()
    # downsampling based on label counts
    median = np.median(labelCt.values)
    

    if groupby_col:
        groupby_col = groupby_col + [label_col]
    else:
        groupby_col = [label_col]


    # split groups into train, val, test
    trainGroups, tempGroups = train_test_split(manifest,
                                               test_size=(1 - percentage[0]),
                                               shuffle=False,
                                               random_state=seed, stratify=groupby_col)

    valGroups, testGroups = train_test_split(tempGroups,
                                             test_size=(percentage[2] / (percentage[1] + percentage[2])),
                                             shuffle=False,
                                             random_state=seed, stratify=groupby_col)

    # get all data for each split based on groups
    train = pd.merge(manifest, trainGroups, on=groupby_col, how='inner')
    val = pd.merge(manifest, valGroups, on=groupby_col, how='inner')
    test = pd.merge(manifest, testGroups, on=groupby_col, how='inner')

    # save stats
    stats = {"label": list(labelCt.keys()), "total images": len(manifest),
             "train": len(train), "test": len(test), "validation": len(val)}
    if out_dir is not None:
        save_data(pd.DataFrame(stats), out_dir + "/data_split.csv")

        # save to csv
        save_data(train, out_dir + "/train_data.csv")
        save_data(val, out_dir + "/validate_data.csv")
        save_data(test, out_dir + "/test_data.csv")

    return train, val, test, stats