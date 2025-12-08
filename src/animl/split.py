"""
Tools for splitting the data for different workflows

@ Kyra Swanson 2023
"""
import pandas as pd
import numpy as np
from typing import Optional
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


def train_val_test(manifest: pd.DataFrame,
                   label_col: str = "class",
                   file_col: str = 'filepath',
                   conf_col: str = "conf",
                   out_dir: Optional[str] = None,
                   test_size: float = 0.1,
                   val_size: float = 0.1,
                   random_state: int = 42):
    """
    Returns train_df, val_df, test_df with label_col stratified.
    test_size and val_size are fractions of the whole dataset (e.g., 0.2 -> 20%).
    """
    assert 0 <= test_size < 1
    assert 0 <= val_size < 1
    assert test_size + val_size < 1

    if label_col not in manifest.columns:
        raise ValueError(f"label_col '{label_col}' not found in dataframe columns")
    if file_col not in manifest.columns:
        raise ValueError(f"file_col '{file_col}' not found in dataframe columns")

    # Keep only the highest confidence entry for each file, or one entry per file if no conf_col
    if conf_col not in manifest.columns:
        manifest = manifest.drop_duplicates(subset=[file_col])
    else:
        idx = manifest.groupby(file_col)[conf_col].idxmax()
        manifest = manifest.loc[idx].reset_index(drop=True)

    # Stage 1: split off test
    trainval_df, test_df = train_test_split(manifest,
                                            test_size=test_size,
                                            stratify=manifest[label_col],
                                            random_state=random_state)

    # Stage 2: split train/val from trainval (val_size is relative to the original dataset)
    # Compute val fraction relative to trainval size
    rel_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(trainval_df,
                                        test_size=rel_val_size,
                                        stratify=trainval_df[label_col],
                                        random_state=random_state + 1)
    # save to csv
    if out_dir is not None:
        save_data(train_df, out_dir + "/train_data.csv")
        save_data(val_df, out_dir + "/validate_data.csv")
        save_data(test_df, out_dir + "/test_data.csv")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
