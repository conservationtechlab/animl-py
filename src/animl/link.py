"""
    Symlink Module

    Provides functions for creating, removing, and updating sorted symlinks.

    @ Kyra Swanson 2023
"""
import os
import pandas as pd
from shutil import copy2
from random import randrange
from pathlib import Path
from pandas import DataFrame

from animl import file_management


def sort_species(manifest: DataFrame,
                 link_dir: str,
                 file_col: str = "FilePath",
                 unique_name: str = 'UniqueName',
                 copy: bool = False) -> DataFrame:
    """
    Creates symbolic links of images into species folders

    Args
        - manifest (DataFrame): dataframe containing images and associated predictions
        - link_dir (str): root directory for species folders
        - file_col (str): column containing source paths
        - unique_name (str): column containing unique file name
        - copy (bool): if true, hard copy

    Returns
        copy of manifest with link path column
    """
    link_dir = Path(link_dir)
    # Create species folders
    for species in manifest['prediction'].unique():
        path = link_dir / Path(species)
        path.mkdir(exist_ok=True)

    # create new column
    manifest['Link'] = link_dir

    for i, row in manifest.iterrows():
        if unique_name in manifest.columns:
            name = row[unique_name]
        else:  # create a unique name
            uniqueid = '{:05}'.format(randrange(1, 10 ** 5))
            filename = os.path.basename(row[file_col])
            filename, extension = os.path.splitext(filename)
            name = "_".join([filename, uniqueid]) + extension

        link = link_dir / Path(row['prediction']) / Path(name)
        manifest.loc[i, 'Link'] = str(link)

        if copy:  # make a hard copy
            copy2(row[file_col], link)
        else:  # make a hard
            os.link(row[file_col], link)

    return manifest


def sort_MD(manifest: DataFrame,
            link_dir: str,
            file_col: str = "file",
            unique_name: str = 'UniqueName',
            copy: bool = False) -> DataFrame:
    """
    Creates symbolic links of images into species folders

    Args
        - manifest (DataFrame): dataframe containing images and associated predictions
        - link_dir (str): root directory for species folders
        - file_col (str): column containing source paths
        - copy (bool): if true, hard copy

    Returns
        copy of manifest with link path column
    """
    link_dir = Path(link_dir)
    # Create class subfolders
    classes = ["empty", "animal", "human", "vehicle"]
    for i in range(classes):
        path = link_dir / Path(classes)
        path.mkdir(exist_ok=True)

    # create new column
    manifest['Link'] = link_dir
    for i, row in manifest.iterrows():
        if unique_name in manifest.columns:
            name = row[unique_name]
        else:
            uniqueid = '{:05}'.format(randrange(1, 10 ** 5))
            filename = os.path.basename(row[file_col])
            filename, extension = os.path.splitext(filename)
            name = "_".join([filename, uniqueid]) + extension

        link = link_dir / Path(row['category']) / Path(name)
        manifest.loc[i, 'Link'] = str(link)

        if copy:  # make a hard copy
            copy2(row[file_col], link)
        else:  # make a hard link
            os.link(row[file_col], link)

    return manifest


def remove_link(manifest: DataFrame,
                link_col: str = 'Link') -> DataFrame:
    """
    Deletes symbolic links of images

    Args
        - manifest: dataframe containing images and associated predictions
        - link_col: column name of paths to remove
    """
    # delete files
    for _, row in manifest.iterrows():
        os.remove(row[link_col])
    # remove column
    manifest.drop(columns=[link_col])
    return manifest


def update_labels(manifest: DataFrame,
                  link_dir: str,
                  unique_name: str = 'UniqueName') -> DataFrame:
    """
    Update manifest after human review of symlink directories

    Args
        - manifest: dataframe containing images and associated predictions
        - link_dir: root directory for species folders
        - unique_name: column to merge sorted labels onto manifest

    Return
        - manifest: dataframe with updated
    """
    if unique_name not in manifest.columns:
        raise AssertionError("Manifest does not have unique names, cannot match to sorted directories.")

    ground_truth = file_management.build_file_manifest(link_dir, exif=False)

    if len(ground_truth) != len(manifest):
        print(f"Warning, found {len(ground_truth)} files in link dir but {len(manifest)} files in manifest.")

    # last level should be label level
    ground_truth['label'] = ground_truth["FilePath"].apply(
        lambda x: os.path.split(os.path.split(x)[0])[1])

    return pd.merge(manifest, ground_truth, left_on=unique_name, right_on="FileName")
