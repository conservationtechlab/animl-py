"""
Symlink Module

Provides functions for creating, removing, and updating sorted symlinks.

@ Kyra Swanson 2023
"""
import json
import os
import pandas as pd
from typing import Optional
from shutil import copy2
from random import randrange
from pathlib import Path
from tqdm import tqdm

from animl import file_management, __version__
from animl.utils.general import convert_minxywh_to_absxyxy


def export_folders(manifest: pd.DataFrame,
                   out_dir: str,
                   out_file: Optional[str] = None,
                   label_col: str = 'prediction',
                   file_col: str = "filepath",
                   unique_name: str = 'uniquename',
                   copy: bool = False) -> pd.DataFrame:
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

    if label_col not in manifest.columns:
        raise AssertionError(f"Label column {label_col} not found in manifest.")

    if label_col == 'category':
        classes = {"0": "empty", "1": "animal", "2": "human", "3": "vehicle"}
        for i in classes.values():
            path = out_dir / str(i)
            path.mkdir(exist_ok=True)
    else:
        classes = manifest[label_col].unique()
        for i in classes:
            path = out_dir / str(i)
            path.mkdir(exist_ok=True)

    # create new column
    manifest['link'] = out_dir

    for i, row in tqdm(manifest.iterrows()):
        try:
            name = row[unique_name]
        except KeyError:
            filename = Path(row[file_col]).stem
            extension = Path(row[file_col]).suffix

            # get datetime
            if "datetime" in manifest.columns:
                reformat_date = pd.to_datetime(row['datetime'], format="%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d_%H%M%S")
            else:
                reformat_date = '{:04}'.format(randrange(1, 10 ** 5))
            # get station
            if "station" in manifest.columns:
                station = row['station']
                name = "_".join([station, reformat_date, filename]) + extension
            else:
                name = "_".join([reformat_date, filename]) + extension

            manifest.loc[i, unique_name] = name

        if label_col == 'category':
            link = out_dir / str(classes[str(row['category'])]) / str(name)
        else:
            link = out_dir / str(row[label_col]) / str(name)

        manifest.loc[i, 'link'] = str(link)

        if not link.is_file():
            if copy:  # make a hard copy
                copy2(row[file_col], link)
            else:  # make a hard
                os.link(row[file_col], link)

    if out_file:
        manifest.to_csv(out_file, index=False)

    return manifest


def remove_link(manifest: pd.DataFrame,
                link_col: str = 'link') -> pd.DataFrame:
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
        Path(row[link_col]).unlink(missing_ok=True)
    # remove column
    manifest.drop(columns=[link_col])
    return manifest


def update_labels_from_folders(manifest: pd.DataFrame,
                               export_dir: str,
                               unique_name: str = 'uniquename') -> pd.DataFrame:
    """
    Update manifest after human review of symlink directories.

    Args:
        manifest (pd.DataFrame): dataframe containing images and associated predictions
        export_dir (str): root directory for species folders
        unique_name (str): column to merge sorted labels onto manifest

    Returns:
        manifest: dataframe with updated predictions
    """
    if unique_name not in manifest.columns:
        raise AssertionError("Manifest does not have unique names, cannot match to sorted directories.")

    print("Searching directory...")
    ground_truth = file_management.build_file_manifest(export_dir, exif=False)

    if len(ground_truth) != len(manifest):
        print(f"Warning, found {len(ground_truth)} files in link dir but {len(manifest)} files in manifest.")

    # last level should be label level
    ground_truth = ground_truth.rename(columns={'filename': unique_name})
    ground_truth['label'] = ground_truth["filepath"].apply(lambda x: Path(x).parent.name)

    return pd.merge(manifest, ground_truth[[unique_name, 'label']], on=unique_name)


def export_coco(manifest: pd.DataFrame,
                class_list: pd.DataFrame,
                out_file: str,
                info: Optional[dict] = None,
                licenses: Optional[list] = None):
    """
    Export a manifest to COCO format.

    Args:
        manifest (pd.DataFrame): dataframe containing images and associated predictions
        class_list (pd.DataFrame): dataframe containing class names and their corresponding IDs
        out_file (str): path to save the COCO formatted file
        info (Optional[dict]): info section of COCO file
        licenses (Optional[list]): licenses section of COCO file

    Returns:
        coco formatted json file saved to out_file
    """
    expected_columns = ('filepath', 'filename', 'filemodifydate', 'frame',
                        'max_detection_conf', 'category', 'conf', 'bbox_x', 'bbox_y', 'bbox_w',
                        'bbox_h', 'prediction', 'confidence')

    for s in expected_columns:
        assert s in manifest.columns, f'Expected column {s} not found in results DataFrame'

    if info is None:
        info = {'description': 'COCO Export from animl',
                'version': __version__,
                'date_created': pd.Timestamp.now().strftime("%Y/%m/%d")}

    if licenses is None:
        licenses = []

    # build categories from class list
    class_dict = {row['class']: int(row['id']) for _, row in class_list.iterrows()}
    categories = []
    for _, row in class_list.iterrows():
        category = {'id': int(row['id']),
                    'name': row['class'],
                    'supercategory': 'none'}
        categories.append(category)

    # create image id based on filepath
    manifest['image_id'] = manifest.groupby('filepath').ngroup()

    images = []
    annotations = []
    for i_row, row in manifest.iterrows():

        width = int(row['width']) if not pd.isna(row['width']) else 0
        height = int(row['height']) if not pd.isna(row['height']) else 0

        image = {'id': row['image_id'],
                 'file_name': Path(row['filepath']).name,
                 'width': width,
                 'height': height}
        images.append(image)

        # convert bbox to abs coordinates
        bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
        # skip annotation if bbox is NaN
        if pd.isna(bbox).any():
            continue
        bbox = convert_minxywh_to_absxyxy(bbox, width, height)
        area = bbox[2] * bbox[3]

        # get category id
        category_id = class_dict.get(row['prediction'], -1)

        annotation = {'id': i_row,
                      'image_id': row['image_id'],
                      'category_id': category_id,
                      'frame': int(row.get('frame', 0)),
                      'bbox': bbox,
                      'area': area,
                      'iscrowd': 0}
        annotations.append(annotation)

    coco_format = {'info': info,
                   'licenses': licenses,
                   'images': images,
                   'annotations': annotations,
                   'categories': categories}

    with open(out_file, 'w') as f:
        json.dump(coco_format, f)

    return coco_format


def export_camtrapR(manifest: pd.DataFrame,
                    out_dir: str,
                    out_file: Optional[str] = None,
                    label_col: str = 'prediction',
                    file_col: str = "filepath",
                    station_col: str = 'station',
                    unique_name: str = 'uniquename',
                    copy: bool = False) -> pd.DataFrame:
    
    expected_columns = (file_col, station_col, label_col)
    for s in expected_columns:
        assert s in manifest.columns, f'Expected column {s} not found in results DataFrame'

    manifest['link'] = out_dir

    stations = manifest.groupby(station_col)

    for station_name, station in tqdm(stations):
        for i, row in station.iterrows():
            try:
                name = row[unique_name]
            except KeyError:
                filename = Path(row[file_col]).stem
                extension = Path(row[file_col]).suffix

                # get datetime
                if "datetime" in manifest.columns:
                    reformat_date = pd.to_datetime(row['datetime'], format="%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d_%H%M%S")
                else:
                    reformat_date = '{:04}'.format(randrange(1, 10 ** 5))
                # get station
                if "station" in manifest.columns:
                    station = row['station']
                    name = "_".join([station, reformat_date, filename]) + extension
                else:
                    name = "_".join([reformat_date, filename]) + extension

                manifest.loc[i, unique_name] = name

            link = out_dir / str(station_name) / str(row[label_col]) / str(name)

            manifest.loc[i, 'link'] = str(link)

            if not link.is_file():
                if copy:  # make a hard copy
                    copy2(row[file_col], link)
                else:  # make a hard
                    os.link(row[file_col], link)

    if out_file:
        manifest.to_csv(out_file, index=False)

    return manifest

def export_timelapse(results: pd.DataFrame,
                     image_dir: str,
                     only_animl: bool = True) -> Path:
    '''
    Converts the Pandas DataFrame created by running the animl classsifier to a csv file that contains columns needed for TimeLapse conversion in later step

    Credit: Sachin Gopal Wani

    Args:
        results - a DataFrame that contains classifications \
        image_dir - location of root directory where all images are stored (can contain subdirectories) \
        only_animl - A bool that confirms whether we want only animal detctions or all (animal + non-animal detection from MegaDetector + classifier)

    Returns:
        animals.csv - A csv file containing all the detection and classification information for animal detections \
        non-anim.csv - A csv file containing detections of all non-animals made to be similar to animals.csv in columns \
        csv_loc - Location of the stored animals csv file
    '''
    # Create directory
    export_dir = Path(image_dir) / "Export"
    Path(export_dir).mkdir(exist_ok=True)

    expected_columns = ('filepath', 'filename', 'filemodifydate', 'frame',
                        'max_detection_conf', 'category', 'conf', 'bbox_x', 'bbox_y', 'bbox_w',
                        'bbox_h', 'prediction', 'confidence')

    for s in expected_columns:
        assert s in results.columns, f'Expected column {s} not found in results DataFrame'

    # Dropping unnecessary columns (Refer to columns numbers above for expected columns - 0 indexed).
    results = results.drop(['filepath', 'filemodifydate', 'max_detection_conf'], axis=1)

    # Keep relative path only
    results['file'] = results['filename']

    # Rename column names for clarity
    results = results.rename(columns={'conf': 'detection_conf', 'prediction': 'class', 'confidence': 'classification_conf'})
    csv_loc = Path(export_dir / "timelapse_manifest.csv")
    results.to_csv(csv_loc, index=False)

    if only_animl:
        animals = results[results['category'] == 1]
        animals.to_csv(Path(export_dir / "animals.csv"), index=False)

    # Return the location of csv for json conversion
    return csv_loc


def export_megadetector(manifest: pd.DataFrame,
                        output_file: Optional[str] = None,
                        detector: str = 'MegaDetector v5a',
                        prompt: bool = True):
    """
    Converts the .csv file [input_file] to the MD-formatted .json file [output_file].

    If [output_file] is None, '.json' will be appended to the input file.

    # Credit goes to Dan Morris https://github.com/agentmorris/MegaDetector/tree/main
    # Adding a modified script to animl-py repo

    Args:
        manifest (pd.DataFrame): dataframe containing images and associated detections
        output_file (Optional[str]): path to save the MD formatted file
        detector (str): name of the detector used
        prompt (bool): whether to prompt before overwriting existing file

    Returns:
        None, saves a json file in MD format
    """

    detection_category_id_to_name = {'0': 'empty', '1': 'animal', '2': 'person', '3': 'vehicle'}

    if output_file is None:
        output_file = 'detections.json'

    if not {'filepath', 'category', 'conf', 'bbox_x', 'bbox_y',
            'bbox_w', 'bbox_h', 'prediction', 'confidence'}.issubset(manifest.columns):
        raise ValueError("DataFrame must contain bounding boxes and confidence.")

    classification_category_name_to_id = {}
    filename_to_results = {}

    for i_row, row in manifest.iterrows():

        if str(row['category']) == '0':
            continue

        # Is this the first detection we've seen for this file?
        if row['filepath'] in filename_to_results:
            im = filename_to_results[row['filepath']]
        else:
            im = {}
            im['detections'] = []
            im['file'] = row['filepath']
            filename_to_results[im['file']] = im

        assert isinstance(row['category'], int), 'Invalid category identifier in row {}'.format(im['file'])
        detection_category_id = str(row['category'])
        assert detection_category_id in detection_category_id_to_name, \
            'Unrecognized detection category ID {}'.format(detection_category_id)

        detection = {}
        detection['category'] = detection_category_id
        detection['conf'] = row['conf']
        detection['frame'] = int(row.get('frame', 0))
        bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
        detection['bbox'] = bbox
        classification_category_name = row['prediction']

        # Have we seen this classification category before?
        if classification_category_name in classification_category_name_to_id:
            classification_category_id = \
                classification_category_name_to_id[classification_category_name]
        else:
            classification_category_id = str(len(classification_category_name_to_id))
            classification_category_name_to_id[classification_category_name] = \
                classification_category_id

        classifications = [[classification_category_id, row['confidence']]]
        detection['classifications'] = classifications

        im['detections'].append(detection)

    # ...for each row

    info = {}
    info['format_version'] = '3.0'
    info['detector'] = detector
    info['classifier'] = 'Animl'

    results = {}
    results['info'] = info
    results['detection_categories'] = detection_category_id_to_name
    results['classification_categories'] = \
        {v: k for k, v in classification_category_name_to_id.items()}
    results['images'] = list(filename_to_results.values())

    # Save the results to a JSON file
    file_management.save_json(results, output_file, prompt=prompt)
