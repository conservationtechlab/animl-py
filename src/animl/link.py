"""
    Symlink Module

    Provides functions for creating, removing, and updating sorted symlinks.

    @ Kyra Swanson 2023
"""
import os
import argparse
import pandas as pd
from shutil import copy2
from random import randrange
from pathlib import Path
from pandas import DataFrame
from tqdm import tqdm

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
        path = link_dir / Path(str(species))
        path.mkdir(exist_ok=True)

    # create new column
    manifest['Link'] = link_dir

    for i, row in tqdm(manifest.iterrows()):
        try:
            name = row[unique_name]
        except KeyError:
            filename = os.path.basename(str(row[file_col]))
            filename, extension = os.path.splitext(filename)

            # get datetime
            if "DateTime" in manifest.columns:
                reformat_date = pd.to_datetime(row['DateTime'], format="%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d_%H%M%S")
            else:
                reformat_date = '{:04}'.format(randrange(1, 10 ** 5))
            # get station
            if "Station" in manifest.columns:
                station = row['Station']
                name = "_".join([station, reformat_date, filename]) + extension
            else:
                name = "_".join([reformat_date, filename]) + extension

            manifest.loc[i, unique_name] = name

        link = link_dir / Path(row['prediction']) / Path(name)
        manifest.loc[i, 'Link'] = str(link)

        if not link.is_file():
            if copy:  # make a hard copy
                copy2(row[file_col], link)
            else:  # make a hard
                os.link(row[file_col], link,)

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
    for i, row in tqdm(manifest.iterrows()):
        try:
            name = row[unique_name]
        except KeyError:
            filename = os.path.basename(str(row[file_col]))
            filename, extension = os.path.splitext(filename)

            # get datetime
            if "DateTime" in manifest.columns:
                reformat_date = pd.to_datetime(row['DateTime'], format="%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d_%H%M%S")
            else:
                reformat_date = '{:04}'.format(randrange(1, 10 ** 5))
            # get station
            if "Station" in manifest.columns:
                station = row['Station']
                name = "_".join([station, reformat_date, filename]) + extension
            else:
                name = "_".join([reformat_date, filename]) + extension

            manifest.loc[i, unique_name] = name

        link = link_dir / Path(row['category']) / Path(name)
        manifest.loc[i, 'Link'] = str(link)

        if not link.is_file():
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

    print("Searching directory...")
    ground_truth = file_management.build_file_manifest(link_dir, exif=False)

    if len(ground_truth) != len(manifest):
        print(f"Warning, found {len(ground_truth)} files in link dir but {len(manifest)} files in manifest.")

    # last level should be label level
    ground_truth = ground_truth.rename(columns={'FileName': unique_name})
    ground_truth['label'] = ground_truth["FilePath"].apply(lambda x: os.path.split(os.path.split(x)[0])[1])

    return pd.merge(manifest, ground_truth[[unique_name, 'label']], on=unique_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update manifest')
    parser.add_argument('--manifest', help='Path to manifest file')
    parser.add_argument('--link_dir', help='Sorted directory')
    parser.add_argument('--unique', help='Column referring to unique file name', default='UniqueName')
    args = parser.parse_args()

    if not Path(args.manifest).is_file():
        raise FileNotFoundError(f'Manifest "{args.manifest}" not found.')

    manifest = pd.read_csv(args.manifest)

    new_manifest = update_labels(manifest, args.link_dir, args.unique)
    print("Rewriting manifest...")
    new_manifest.to_csv(os.path.split(args.manifest)[0] + "/Results_corrected.csv", index=False)
