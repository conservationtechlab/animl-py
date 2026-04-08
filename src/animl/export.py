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
import yaml


from animl import file_management, __version__
from animl.utils.general import _xywh_to_xywhc, _xywh_to_absxyxy


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
        label_col (str): column containing species labels,
                        'category' for MD categories or 'prediction' for species labels
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
        unique_name (str): column containing unique file names

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
        bbox = _xywh_to_absxyxy(bbox, width, height)
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


def export_yolo(train_manifest: pd.DataFrame,
                val_manifest: pd.DataFrame,
                test_manifest: Optional[pd.DataFrame],
                class_list: pd.DataFrame,
                out_dir: str,
                label_col: str = 'prediction',
                file_col: str = 'filepath'):
    """
    Export a manifest to YOLO format for model training.
    Saves a .txt file for each image with bounding box coordinates and class labels.

    Args:
        manifest (pd.DataFrame): dataframe containing images and associated predictions
        class_list (pd.DataFrame): dataframe containing class names and their corresponding IDs
        out_dir (str): directory to save YOLO formatted files
        label_col (str): column containing species labels,
                        'category' for MD categories or 'prediction' for species labels
        file_col (str): column containing source paths
    """
    expected_columns = (file_col, label_col, 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h')
    for s in expected_columns:
        assert s in train_manifest.columns, f'Expected column {s} not found in train_manifest DataFrame'
        assert s in val_manifest.columns, f'Expected column {s} not found in val_manifest DataFrame'
        if test_manifest is not None:
            assert s in test_manifest.columns, f'Expected column {s} not found in test_manifest DataFrame'

    # create output directories
    out_dir = Path(out_dir)
    image_dir = out_dir / 'images'
    image_train_dir = image_dir / 'train'
    image_val_dir = image_dir / 'val'
    image_test_dir = image_dir / 'test'

    label_dir = out_dir / 'labels'
    label_train_dir = label_dir / 'train'
    label_val_dir = label_dir / 'val'
    label_test_dir = label_dir / 'test'

    for d in [image_train_dir, image_val_dir, image_test_dir,
              label_train_dir, label_val_dir, label_test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def convert_to_yolo(row):
        # convert bbox to abs coordinates
        bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
        # skip annotation if bbox is NaN
        if pd.isna(bbox).any():
            return None
        bbox = _xywh_to_xywhc(bbox)
        # get class id
        class_id = class_list[class_list['class'] == row[label_col]]['id'].values[0]
        return f"{class_id} {' '.join(map(str, bbox))}"

    # symlink images to train/val/test folders
    for _, row in tqdm(train_manifest.iterrows()):
        file = Path(row[file_col]).name
        link = image_train_dir / file
        if not link.is_file():
            link.symlink_to(file)

        label = file.stem + '.txt'
        label_path = label_train_dir / label
        yolo_annotation = convert_to_yolo(row)
        if yolo_annotation is not None:
            with open(label_path, 'w') as f:
                f.write(yolo_annotation)

    for _, row in tqdm(val_manifest.iterrows()):
        file = Path(row[file_col])
        link = image_val_dir / file.name
        if not link.is_file():
            link.symlink_to(file)

        label = file.stem + '.txt'
        label_path = label_val_dir / label
        yolo_annotation = convert_to_yolo(row)
        if yolo_annotation is not None:
            with open(label_path, 'w') as f:
                f.write(yolo_annotation)

    if test_manifest is not None:
        for _, row in tqdm(test_manifest.iterrows()):
            file = Path(row[file_col])
            link = image_test_dir / file.name
            if not link.is_file():
                link.symlink_to(file)

            label = file.stem + '.txt'
            label_path = label_test_dir / label
            yolo_annotation = convert_to_yolo(row)
            if yolo_annotation is not None:
                with open(label_path, 'w') as f:
                    f.write(yolo_annotation)

    output_yml = {
        'path': str(out_dir),
        'train': str(image_train_dir),
        'val': str(image_val_dir),
        'test': str(image_test_dir) if test_manifest is not None else None,
        'nc': len(class_list),
        'names': class_list['class'].tolist()
    }
    output_yml_path = out_dir / 'dataset.yaml'
    with open(output_yml_path, 'w') as f:
        yaml.dump(output_yml, f)


def export_camptrapdp(manifest: pd.DataFrame,
                      out_dir: str,
                      file_public: bool = False,
                      classifier_name: str = None):
    """
    Export a manifest to camtrapdp format.
    Requires scientific name for the species prediction label and bounding box coordinates for each detection.

    Args:
        manifest (pd.DataFrame): dataframe containing images and associated predictions
        out_file (str): path to save the camtrapdp formatted file
        file_public (bool): whether media files are publicly accessible
        classifier_name (str): name of the classifier used for predictions
    """
    # convert MD categories to camtrapdp categories
    category_conversion = {0: 'blank', 1: 'animal', 2: 'human', 3: 'vehicle'}
    # reset index to ensure unique media and observation ids when creating media and observation tables
    manifest = manifest.reset_index(drop=True)

    datapackage = {
        "name": "camtrapdp_export",
        "profile": "tabular-data-package",
        "resources": [
            {
                "name": "observations",
                "schema": {
                }
            }

        ]
    }

    # create media_id column based on filepath, which is required for media table
    manifest["media_id"] = manifest["filepath"].factorize()[0]

    def convert_media(row):
        media = {'mediaID': row['media_id'] if 'media_id' in row else None,
                 'deploymentID': row['deployment_id'] if 'deployment_id' in row else None,
                 'filePublic': file_public,
                 'timestamp': row['datetime'] if 'datetime' in row else None,
                 'filePath': row['filepath'] if 'filepath' in row else None,
                 'fileName': row['filename'] if 'filename' in row else None,
                 'fileMediaType': row['extension'] if 'extension' in row else None}
        return media
    # create media table
    media = [convert_media(row) for _, row in manifest.drop_duplicates(subset=['filepath']).iterrows()]

    def convert_observation(id, row):
        observation = {'observationID': id,
                       'deploymentID': row['deployment_id'] if 'deployment_id' in row else None,
                       'mediaID': row['media_id'] if 'media_id' in row else None,
                       'eventStart': row['datetime'] if 'datetime' in row else None,
                       'eventEnd': row['datetime'] if 'datetime' in row else None,
                       'observationLevel': 'media',
                       'observationType':  category_conversion.get(row['category'], 'unknown'),
                       'scientificName': row['prediction'] if 'prediction' in row else None,
                       'bboxX': row['bbox_x'] if 'bbox_x' in row else None,
                       'bboxY': row['bbox_y'] if 'bbox_y' in row else None,
                       'bboxWidth': row['bbox_w'] if 'bbox_w' in row else None,
                       'bboxHeight': row['bbox_h'] if 'bbox_h' in row else None,
                       'classificationMethod': 'machine',
                       'classifiedBy': classifier_name,
                       'classificationTimestamp': pd.Timestamp.now().strftime("%Y-%m-%d"),
                       'classificationProbability': row['confidence'] if 'confidence' in row else None,
                       }
        return observation
    # create observations table
    observations = [convert_observation(id, row) for id, row in manifest.iterrows()]

    # save media and observations to separate csv files
    media_df = pd.DataFrame(media)
    observations_df = pd.DataFrame(observations)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    media_df.to_csv(out_dir / 'media.csv', index=False)
    observations_df.to_csv(out_dir / 'observations.csv', index=False)

    # create datapackage.json file
    with open(out_dir / 'datapackage.json', 'w') as f:
        json.dump(datapackage, f, indent=4)

    return


def export_camtrapR(manifest: pd.DataFrame,
                    out_dir: str,
                    out_file: Optional[str] = None,
                    label_col: str = 'prediction',
                    file_col: str = "filepath",
                    station_col: str = 'station',
                    unique_name: str = 'uniquename',
                    copy: bool = False) -> pd.DataFrame:
    """
    Export data into sorted folders organized by station

    Args:
        - manifest (pd.DataFrame): dataframe containing images and associated predictions
        - out_dir (str): directory to export sorted images
        - out_file (Optional[str]): if provided, save the manifest to this file
        - label_col (str): column containing species labels
        - file_col (str): column containing source paths
        - station_col (str): column containing station names
        - unique_name (str): column containing unique file name
        - copy (bool): if true, hard copy
    """
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


def export_timelapse(manifest: pd.DataFrame,
                     out_dir: str,
                     only_animal: bool = True) -> Path:
    '''
    Converts the Pandas DataFrame created by running the animl classsifier to a csv file that contains columns needed for TimeLapse conversion in later step

    Credit: Sachin Gopal Wani

    Args:
        manifest - a DataFrame that contains classifications
        out_dir - location of directory where csv files will be saved
        only_animl - A bool that confirms whether we want only animal detctions or all (animal + non-animal detection from MegaDetector + classifier)

    Returns:
        animals.csv - A csv file containing all the detection and classification information for animal detections
        non-anim.csv - A csv file containing detections of all non-animals made to be similar to animals.csv in columns
        csv_loc - Location of the stored animals csv file
    '''
    # Create directory
    Path(out_dir).mkdir(exist_ok=True)

    expected_columns = ('filepath', 'filename', 'filemodifydate', 'frame',
                        'max_detection_conf', 'category', 'conf', 'bbox_x', 'bbox_y', 'bbox_w',
                        'bbox_h', 'prediction', 'confidence')

    for s in expected_columns:
        assert s in manifest.columns, f'Expected column {s} not found in manifest DataFrame'

    # Dropping unnecessary columns (Refer to columns numbers above for expected columns - 0 indexed).
    manifest = manifest.drop(['filemodifydate', 'frame' 'max_detection_conf'], axis=1)

    # Rename column names for clarity
    manifest = manifest.rename(columns={'filename': 'file', 'conf': 'detection_conf', 
                                        'prediction': 'class', 'confidence': 'classification_conf'})
    csv_loc = Path(out_dir / "timelapse_manifest.csv")
    manifest.to_csv(csv_loc, index=False)

    # remove erroneous detections
    manifest = manifest[manifest['category'].notna()]

    animals = manifest[manifest['category'] == 1]

    if only_animal:
        animals.to_csv(Path(out_dir / "animals.csv"), index=False)
    
    else:
        empty = manifest[manifest['category'] != 1]
        # Adding prediction as person and human
        empty['class'].replace({'0': 'empty', '2': 'person', '3': 'vehicle'}, inplace=True)
        # Changing classification conf = detection_conf instead of max_detection_conf
        empty['classification_conf'] = empty.loc[:, 'detection_conf']

        # Combining DataFrames and saving it to csv file for further use
        csv_loc = Path(out_dir / "manifest.csv")
        manifest = pd.concat([animals, empty])
        manifest.to_csv(csv_loc, index=False)
        animals.to_csv(Path(out_dir / "animals.csv"), index=False)
        empty.to_csv(Path(out_dir / "non-animals.csv"), index=False)
    # Return the location of csv for json conversion
    return csv_loc


def export_megadetector(manifest: pd.DataFrame,
                        out_file: Optional[str] = None,
                        detector: str = 'MegaDetector v5b',
                        prompt: bool = True):
    """
    Converts the .csv file [input_file] to the MD-formatted .json file [out_file].

    If [out_file] is None, '.json' will be appended to the input file.

    # Credit goes to Dan Morris https://github.com/agentmorris/MegaDetector/tree/main
    # Adding a modified script to animl-py repo

    Args:
        manifest (pd.DataFrame): dataframe containing images and associated detections
        out_file (Optional[str]): path to save the MD formatted file
        detector (str): name of the detector used
        prompt (bool): whether to prompt before overwriting existing file

    Returns:
        None, saves a json file in MD format
    """

    detection_category_id_to_name = {'0': 'empty', '1': 'animal', '2': 'person', '3': 'vehicle'}

    if out_file is None:
        out_file = 'detections.json'

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
    file_management.save_json(results, out_file, prompt=prompt)
