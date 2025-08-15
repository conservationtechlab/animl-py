"""
Symlink Module

Provides functions for creating, removing, and updating sorted symlinks.

@ Kyra Swanson 2023
"""
import os
import pandas as pd
from typing import Optional
from shutil import copy2
from random import randrange
from pathlib import Path
from pandas import DataFrame
from tqdm import tqdm

from animl import file_management


def sort_species(manifest: DataFrame,
                 out_dir: str,
                 out_file: Optional[str] = None,
                 file_col: str = "FilePath",
                 unique_name: str = 'UniqueName',
                 copy: bool = False) -> DataFrame:
    """
    Creates symbolic links of images into species folders.

    Args:
        manifest (DataFrame): dataframe containing images and associated predictions
        out_dir (str): root directory for species folders
        out_file (Optional[str]): if provided, save the manifest to this file
        file_col (str): column containing source paths
        unique_name (str): column containing unique file name
        copy (bool): if true, hard copy

    Returns:
        copy of manifest with link path column
    """
    out_dir = Path(out_dir)
    # Create species folders
    for species in manifest['prediction'].unique():
        path = out_dir / Path(str(species))
        path.mkdir(exist_ok=True)

    # create new column
    manifest['Link'] = out_dir

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

        link = out_dir / Path(row['prediction']) / Path(name)
        manifest.loc[i, 'Link'] = str(link)

        if not link.is_file():
            if copy:  # make a hard copy
                copy2(row[file_col], link)
            else:  # make a hard
                os.link(row[file_col], link,)

    if out_file:
        manifest.to_csv(out_file, index=False)

    return manifest


def sort_MD(manifest: DataFrame,
            out_dir: str,
            out_file: Optional[str] = None,
            file_col: str = "file",
            unique_name: str = 'UniqueName',
            copy: bool = False) -> DataFrame:
    """
    Creates symbolic links of images into MegaDetector class folders

    Args:
        manifest (DataFrame): dataframe containing images and associated predictions
        out_dir (str): root directory for species folders
        out_file (Optional[str]): if provided, save the manifest to this file
        file_col (str): column containing source paths
        unique_name (str): column containing unique file name
        copy (bool): if true, hard copy

    Returns:
        copy of manifest with link path column
    """
    out_dir = Path(out_dir)
    # Create class subfolders
    classes = ["empty", "animal", "human", "vehicle"]
    for i in range(classes):
        path = out_dir / Path(classes)
        path.mkdir(exist_ok=True)

    # create new column
    manifest['Link'] = out_dir
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

        link = out_dir / Path(row['category']) / Path(name)
        manifest.loc[i, 'Link'] = str(link)

        if not link.is_file():
            if copy:  # make a hard copy
                copy2(row[file_col], link)
            else:  # make a hard link
                os.link(row[file_col], link)

    if out_file:
        manifest.to_csv(out_file, index=False)

    return manifest


def remove_link(manifest: DataFrame,
                link_col: str = 'Link') -> DataFrame:
    """
    Deletes symbolic links of images.

    Args:
        manifest (pd.DataFrame): dataframe containing images and associated predictions
        link_col (str): column name of paths to remove

    Returns:
        manifest without link column
    """
    # delete files
    for _, row in manifest.iterrows():
        os.remove(row[link_col])
    # remove column
    manifest.drop(columns=[link_col])
    return manifest


def update_link_labels(manifest: DataFrame,
                       out_dir: str,
                       unique_name: str = 'UniqueName') -> DataFrame:
    """
    Update manifest after human review of symlink directories.

    Args:
        manifest (pd.DataFrame): dataframe containing images and associated predictions
        out_dir (str): root directory for species folders
        unique_name (str): column to merge sorted labels onto manifest

    Returns:
        manifest: dataframe with updated predictions
    """
    if unique_name not in manifest.columns:
        raise AssertionError("Manifest does not have unique names, cannot match to sorted directories.")

    print("Searching directory...")
    ground_truth = file_management.build_file_manifest(out_dir, exif=False)

    if len(ground_truth) != len(manifest):
        print(f"Warning, found {len(ground_truth)} files in link dir but {len(manifest)} files in manifest.")

    # last level should be label level
    ground_truth = ground_truth.rename(columns={'FileName': unique_name})
    ground_truth['label'] = ground_truth["FilePath"].apply(lambda x: os.path.split(os.path.split(x)[0])[1])

    return pd.merge(manifest, ground_truth[[unique_name, 'label']], on=unique_name)


def export_coco(manifest: DataFrame,
                out_file: str):
    """
    Export a manifest to COCO format.

    Args:
        manifest (pd.DataFrame): dataframe containing images and associated predictions
        out_file (str): path to save the COCO formatted file

    Returns:
        coco formatted json file saved to out_file
    """
    # TODO
    return None


def export_timelapse(animals, empty, imagedir, only_animl=True):
    '''
    Converts the Pandas DataFrame created by running the animl classsifier to a csv file that contains columns needed for TimeLapse conversion in later step

    Credit: Sachin Gopal Wani

    Args:
        animals - a DataFrame that has entries of anuimal classification \
        empty - a DataFrame that has detection of non-animal objects in images \
        imagedir - location of root directory where all images are stored (can contain subdirectories) \
        only_animl - A bool that confirms whether we want only animal detctions or all (animal + non-animal detection from MegaDetector + classifier)

    Returns:
        animals.csv - A csv file containing all the detection and classification information for animal detections \
        non-anim.csv - A csv file containing detections of all non-animals made to be similar to animals.csv in columns \
        csv_loc - Location of the stored animals csv file
    '''
    if not imagedir.endswith("/"):
        imagedir += "/"

    # Create directory
    ICdir = os.path.join(imagedir, "Animl-Directory", "IC")
    os.makedirs(ICdir, exist_ok=True)

    expected_columns = ('FilePath', 'FileName', 'FileModifyDate', 'Frame', 'file',
                        'max_detection_conf', 'category', 'conf', 'bbox_x', 'bbox_y', 'bbox_w',
                        'bbox_h', 'prediction', 'confidence')

    for s in expected_columns:
        assert s in animals.columns, 'Expected column {} not found in animals DataFrame'.format(s)

    # Dropping unnecessary columns (Refer to columns numbers above for expected columns - 0 indexed).
    animals.drop(['FilePath', 'FileName', 'FileModifyDate', 'Frame', 'max_detection_conf'], axis=1, inplace=True)

    # Keep relative path only
    animals['file'] = animals['file'].apply(lambda x: x[len(imagedir):])
    # ALT: copy_ani['file'] = copy_ani['file'].str.slice(start=len(imagedir))

    # Rename column names for clarity
    animals.rename(columns={'conf': 'detection_conf', 'prediction': 'class', 'confidence': 'classification_conf'}, inplace=True)

    if only_animl:
        # Saving animal results to csv file for conversion to timelapse compatible json
        csv_loc = os.path.join(ICdir, "animals.csv")
        animals.to_csv(csv_loc, index=False)

        # Saving non-animal csv entries for manual perusal
        empty.to_csv(os.path.join(ICdir, "non-anim.csv"), index=False)

    else:
        # Checking if the columns match the expected DataFrame
        for s in expected_columns:
            assert s in empty.columns, 'Expected column {} not found in empty (non-animals) DataFrame'.format(s)

        # Doing the same process for non-animal results
        empty.drop(['FilePath', 'FileName', 'FileModifyDate', 'Frame', 'max_detection_conf'], axis=1, inplace=True)
        empty['file'] = empty['file'].apply(lambda x: x[len(imagedir):])
        empty.rename(columns={'conf': 'detection_conf', 'prediction': 'class'}, inplace=True)

        # Adding prediction as person and human
        empty['class'].replace({'0': 'empty', '2': 'person', '3': 'vehicle'}, inplace=True)

        # Changing classification conf = detection_conf instead of max_detection_conf
        empty['classification_conf'] = empty.loc[:, 'detection_conf']

        # Combining DataFrames and saving it to csv file for further use
        csv_loc = os.path.join(ICdir, "manifest.csv")
        manifest = pd.concat([animals, empty])
        manifest.to_csv(csv_loc, index=False)

    # Return the location of csv for json conversion
    return csv_loc


def export_megadetector(manifest, output_file=None):
    """
    Converts the .csv file [input_file] to the MD-formatted .json file [output_file].

    If [output_file] is None, '.json' will be appended to the input file.

    # Credit goes to Dan Morris https://github.com/agentmorris/MegaDetector/tree/main
    # Adding a modified script to animl-py repo
    """

    detection_category_id_to_name = {'0': 'empty', '1': 'animal', '2': 'person', '3': 'vehicle'}

    if output_file is None:
        output_file = 'detections.json'

    expected_columns = ('file', 'category', 'detection_conf',
                        'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                        'class', 'classification_conf')

    for s in expected_columns:
        assert s in manifest.columns, \
            'Expected column {} not found'.format(s)

    classification_category_name_to_id = {}
    filename_to_results = {}

    for i_row, row in manifest.iterrows():

        if str(row['category']) == '0':
            continue

        # Is this the first detection we've seen for this file?
        if row['file'] in filename_to_results:
            im = filename_to_results[row['file']]
        else:
            im = {}
            im['detections'] = []
            im['file'] = row['file']
            filename_to_results[im['file']] = im

        assert isinstance(row['category'], int), 'Invalid category identifier in row {}'.format(im['file'])
        detection_category_id = str(row['category'])
        assert detection_category_id in detection_category_id_to_name, \
            'Unrecognized detection category ID {}'.format(detection_category_id)

        detection = {}
        detection['category'] = detection_category_id
        detection['conf'] = row['detection_conf']
        bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
        detection['bbox'] = bbox
        classification_category_name = row['class']

        # Have we seen this classification category before?
        if classification_category_name in classification_category_name_to_id:
            classification_category_id = \
                classification_category_name_to_id[classification_category_name]
        else:
            classification_category_id = str(len(classification_category_name_to_id))
            classification_category_name_to_id[classification_category_name] = \
                classification_category_id

        classifications = [[classification_category_id, row['classification_conf']]]
        detection['classifications'] = classifications

        im['detections'].append(detection)

    # ...for each row

    info = {}
    info['format_version'] = '1.3'
    info['detector'] = 'Animl'
    info['classifier'] = 'Animl'

    results = {}
    results['info'] = info
    results['detection_categories'] = detection_category_id_to_name
    results['classification_categories'] = \
        {v: k for k, v in classification_category_name_to_id.items()}
    results['images'] = list(filename_to_results.values())

    # Save the results to a JSON file
    file_management.save_json(results, output_file)
