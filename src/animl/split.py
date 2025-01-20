"""
Tools for splitting the data for different workflows
"""
import pandas as pd
import numpy as np
from math import fsum
from typing import Optional, Tuple


def get_animals(manifest: pd.DataFrame):
    """
    Pulls MD animal detections for classification

    Args:
        - manifest: DataFrame containing one row for ever MD detection

    Returns:
        subset of manifest containing only animal detections
    """
    if not isinstance(manifest, pd.DataFrame):
        raise AssertionError("'manifest' must be DataFrame.")
    return manifest[manifest['category'].astype(int) == 1].reset_index(drop=True)


def get_empty(manifest: pd.DataFrame):
    """
    Pulls MD non-animal detections

    Args:
        - manifest: DataFrame containing one row for ever MD detection

    Returns:
        subset of manifest containing empty, vehicle and human detections
        with added prediction and confidence columns
    """
    if not isinstance(manifest, pd.DataFrame):
        raise AssertionError("'manifest' must be DataFrame.")

    # Removes all images that MegaDetector gave no detection for
    otherdf = manifest[manifest['category'].astype(int) != 1].reset_index(drop=True)
    otherdf['prediction'] = otherdf['category'].astype(int)

    # Numbers the class of the non-animals correctly
    if not otherdf.empty:
        otherdf['prediction'] = otherdf['prediction'].replace(2, "human")
        otherdf['prediction'] = otherdf['prediction'].replace(3, "vehicle")
        otherdf['prediction'] = otherdf['prediction'].replace(0, "empty")
        otherdf['confidence'] = otherdf['conf']

    else:
        otherdf = pd.DataFrame(columns=manifest.columns.values)

    return otherdf


def train_val_test(manifest: pd.DataFrame,
                   out_dir: Optional[str] = None,
                   label_col: str = "species",
                   percentage: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                   seed: Optional[int] = None):
    '''
    Splits the manifest into training. validation and test dataets for training

    Credit: Unduwap Kandage-Don

    Args:
        - manifest (DataFrame): list of files to split for training
        - out_dir (str): location to save split lists to
        - label_col (str): column name containing class labels
        - percentage (tuple): fraction of data dedicated to train-val-test
        - seed (int): RNG seed, if none will pick one at random within [0,100]

    Returns:
        - train
        - validate
        - test
        - stats
    '''
    if seed is None:
        seed = np.random.randint(0, 100)

    # check percentages add up to 1
    if not (fsum(percentage) == 1):
        print("Invalid percentages")

    # create blank dataframes
    train = pd.DataFrame()
    validate = pd.DataFrame()
    test = pd.DataFrame()

    # stats
    totCtArr = []
    trainCtArr = []
    valCtArr = []
    testCtArr = []

    # group the data based on label column
    manifest_by_label = manifest.groupby(label_col)
    labelCt = manifest[label_col].value_counts()

    print("seed =", seed)

    for label in labelCt.keys():
        # calc how much of each data belongs to each category
        # test gets the remainder due to rounding percentages
        catCt = labelCt[label]
        trainCt = round(catCt * percentage[0])
        valCt = round(catCt * percentage[1])
        testCt = catCt - (trainCt + valCt)

        totCtArr.append(catCt)
        trainCtArr.append(trainCt)
        valCtArr.append(valCt)
        testCtArr.append(testCt)

        # shuffle based on seed without re-sample
        currLabel = manifest_by_label.get_group(label).sample(frac=1, replace=False, random_state=seed)

        # split group into train, test, val
        trainLabel = currLabel[0:trainCt]
        valLabel = currLabel[trainCt:trainCt+valCt]
        testLabel = currLabel[trainCt+valCt:]

        # save to combined data frame
        train = pd.concat([train, trainLabel], ignore_index=True)
        validate = pd.concat([validate, valLabel], ignore_index=True)
        test = pd.concat([test, testLabel], ignore_index=True)

    # save stats
    stats = {"label": list(labelCt.keys()), "total images": totCtArr,
             "train": trainCtArr, "test": testCtArr, "validation": valCtArr}

    if out_dir is not None:
        statsdf = pd.DataFrame(stats)
        statsdf.to_csv(out_dir + "/data_split.csv")

        # save to csv
        train.to_csv(out_dir + "/train_data.csv")
        validate.to_csv(out_dir + "/validate_data.csv")
        test.to_csv(out_dir + "/test_data.csv")

    return train, validate, test, stats
