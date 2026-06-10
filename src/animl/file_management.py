"""
File Management Module

This module provides functions and classes for managing files and directories.

@ Kyra Swanson 2023
"""
import json
from pathlib import Path, PosixPath
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import PIL
import cv2
import exiftool
import yaml
from typing import Optional, Union

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', ".tiff", '.tif"'}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv",
                    ".mpg", ".mpeg", ".asf", ".m4v"}
VALID_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def build_file_manifest(image_dir: str,
                        exif: bool = True,
                        out_file: Optional[str] = None,
                        data_timezone: Optional[str] = None,
                        station_depth: Optional[int] = None,
                        camera_depth: Optional[int] = None,
                        recursive: bool = True):
    """
    Find Image/Video Files and Gather exif Data.

    Args:
        image_dir (str): directory of files to analyze
        exif (bool): returns date and time info from exif data, defaults to True
        out_file (str): file path to which the dataframe should be saved
        data_timezone (str): timezone of the data, e.g., 'UTC', 'America/New_York', defaults to local timezone if None
                             if you are unsure of the timezone, you can list all with zoneinfo.available_timezones()
        station_depth (int): depth of station directory from the image_dir root in file path, if applicable.
                             For example, if file paths are in the format "image_dir/station/date/file.jpg",
                             station_depth would be 1 (0 indexed). If None, station column will not be created.
        camera_depth (int): depth of camera directory from the image_dir root in file path, if applicable.
                            For example, if file paths are in the format "image_dir/station/camera/date/file.jpg",
                            camera_depth would be 2 (0 indexed). If None, camera column will not be created.
        recursive (bool): recursively search through all child directories

    Returns:
        files (pd.DataFrame): list of files with or without file modify dates
    """
    image_dir = Path(image_dir)
    if check_file(out_file, output_type="Manifest"):
        return load_data(out_file)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"The given directory: {image_dir}, does not exist.")

    files = Path(image_dir).rglob("*.*") if recursive else Path(image_dir).glob("*.*")

    # only keep images and videos
    files = [str(f) for f in files if Path(f).suffix.lower() in VALID_EXTENSIONS]

    # no files found, return empty dataframe
    if not files:
        return pd.DataFrame()

    files = pd.DataFrame(files, columns=["filepath"])
    files["filename"] = files["filepath"].apply(lambda x: Path(x).name)
    files["extension"] = files["filepath"].apply(lambda x: Path(x).suffix.lower())

    if station_depth is not None:
        if recursive is False and station_depth >= 1:
            raise ValueError("station_depth must be less than 1 if recursive is False")
        root_depth = len(Path(image_dir).parts) - 1
        station_depth = root_depth + int(station_depth)
        files["station"] = files["filepath"].apply(
            lambda x: Path(x).parts[station_depth] if len(Path(x).parts) > station_depth else None)

    if camera_depth is not None:
        if recursive is False and camera_depth >= 1:
            raise ValueError("camera_depth must be less than 1 if recursive is False")
        root_depth = len(Path(image_dir).parts) - 1
        camera_depth = root_depth + int(camera_depth)
        files["camera"] = files["filepath"].apply(
            lambda x: Path(x).parts[camera_depth] if len(Path(x).parts) > camera_depth else None)

    invalid = []

    if exif:
        for i, row in files.iterrows():
            if row["extension"] in IMAGE_EXTENSIONS:
                try:
                    img = PIL.Image.open(row['filepath'])
                    files.loc[i, "width"] = img.size[0]
                    files.loc[i, "height"] = img.size[1]
                    files.loc[i, "createdate"] = img.getexif().get(0x0132)
                except PIL.UnidentifiedImageError:
                    print(f"Error processing image file {row['filepath']}")
                    invalid.append(i)

            elif row["extension"] in VIDEO_EXTENSIONS:
                try:
                    vid = cv2.VideoCapture(row['filepath'])
                    # check if video opened successfully
                    if not vid.isOpened():
                        print(f"Error opening video file {row['filepath']}")
                        invalid.append(i)
                        continue
                    files.loc[i, "width"] = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                    files.loc[i, "height"] = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                except Exception as e:
                    print(f"Error processing file {row['filepath']}: {e}")
                    invalid.append(i)
                # for videos try to get createdate from exiftool, but use filemodifydate as backup
                try:
                    with exiftool.ExifToolHelper() as et:
                        metadata = et.get_metadata(row['filepath'])[0]
                        if "QuickTime:CreateDate" in metadata:
                            files.loc[i, "createdate"] = metadata["QuickTime:CreateDate"]
                        elif "EXIF:DateTimeOriginal" in metadata:
                            files.loc[i, "createdate"] = metadata["EXIF:DateTimeOriginal"]
                        else:
                            files.loc[i, "createdate"] = None
                except Exception:
                    print("pyexiftool failed, is exiftool installed and in PATH? \n",
                          "createdate cannot be determined for videos, falling back to filemodifydate.")
                    files.loc[i, "createdate"] = None

        # determine local timezone for conversion
        local_tz = datetime.now().astimezone().tzinfo
        if data_timezone is not None:
            try:
                data_tz = ZoneInfo(data_timezone)
            except Exception as e:
                print(f"Error with timezone: {e}. Defaulting to local timezone.")
                data_tz = local_tz
        else:
            data_tz = local_tz

        # get filemodifydate as backup (videos, etc)
        def get_modify_date(x):
            local = datetime.fromtimestamp(Path(x).stat().st_mtime, tz=local_tz)
            adjusted = local.astimezone(data_tz)
            return adjusted.strftime('%Y-%m-%d %H:%M:%S')

        # function to convert multiple string formats to desired format, returns None if not recognized
        def check_time(timestamp, tzinfo=data_tz):
            input_formats = ['%Y:%m:%d %H:%M:%S', "%d-%m-%Y %H:%M", "%Y/%m/%d %H:%M:%S"]
            desired_format = '%Y-%m-%d %H:%M:%S'
            try:
                timestamp = datetime.strptime(timestamp, desired_format).replace(tzinfo=tzinfo)
                return timestamp.strftime(desired_format)
            except ValueError:
                pass
            # Try other input formats
            for fmt in input_formats:
                try:
                    newtimestamp = datetime.strptime(timestamp, fmt).replace(tzinfo=tzinfo)
                    return newtimestamp.strftime(desired_format)
                except ValueError:
                    continue
            # timestamp not recognized
            return None
        files["filemodifydate"] = files["filepath"].apply(get_modify_date)

        if "createdate" in files.columns:
            # convert multiple string formats to datetime
            files['createdate'] = files['createdate'].replace(r'^\s*$', None, regex=True)
            files["createdate"] = files['createdate'].apply(lambda x: check_time(x) if isinstance(x, str) else x)
            # select createdate if not none, else choose filemodify date
            files["datetime"] = files["createdate"].fillna(files["filemodifydate"])
        else:
            files["datetime"] = files["filemodifydate"]

    files = files.drop(index=invalid).reset_index(drop=True)

    if out_file:
        save_data(files, out_file)

    return files


class WorkingDirectory():
    """
    Set Working Directory and save file global variables.

    Constructor requires root working_directory
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, working_dir):
        if not isinstance(working_dir, PosixPath):
            working_dir = Path(working_dir)  # OS-agnostic path
        if not working_dir.is_dir():
            raise FileNotFoundError(f"The given directory: {working_dir}, does not exist.")

        self.basedir = working_dir / Path("Animl-Directory/")
        self.linkdir = self.basedir / Path("Sorted/")
        self.visdir = self.basedir / Path("Plots/")

        # Create directories if they do not already exist
        self.basedir.mkdir(exist_ok=True)

        # Assign specific file paths
        self.filemanifest = self.basedir / Path("FileManifest.csv")
        self.imageframes = self.basedir / Path("ImageFrames.csv")
        self.results = self.basedir / Path("Results.csv")
        self.predictions = self.basedir / Path("Predictions.csv")
        self.detections = self.basedir / Path("Detections.csv")
        self.mdraw = self.basedir / Path("MD_Raw.json")

    def activate_visdir(self):
        self.visdir.mkdir(exist_ok=True)

    def activate_linkdir(self):
        self.linkdir.mkdir(exist_ok=True)


def save_data(data: pd.DataFrame,
              out_file: Union[Path, str],
              prompt: bool = True) -> None:
    """
    Save data to given file.

    Args:
        data (pd.DataFrame): the dataframe to be saved
        out_file (Union[Path, str]): full path to save file to
        prompt (bool): prompts the user to confirm overwrite

    Returns:
        None
    """
    if Path(out_file).is_file() and (prompt is True):
        prompt = "Output file exists, would you like to overwrite? y/n: "
        if input(prompt).lower() != "y":
            return
    else:
        if Path(out_file).parent.exists():
            data.to_csv(out_file, index=False)
        else:
            raise AssertionError('Cannot save, directory does not exis.')


def load_data(file: Union[Path, str]) -> pd.DataFrame:
    """
    Load .csv File.

    Args:
        file (Union[Path, str]): the full path of the file to load

    Returns:
        data extracted from the file. pd.dataframe form
    """
    if Path(file).suffix.lower() == ".csv":
        return pd.read_csv(file)
    else:
        raise AssertionError("Expecting a .csv file.")


def save_json(data: dict,
              out_file: Union[Path, str],
              prompt: bool = True) -> None:
    """
    Save data to a JSON file.

    Args:
        data (dict): the dictionary to be saved
        out_file (Union[Path, str]): full path to save file to
        prompt (bool): prompt user to confirm overwrite

    Returns:
        None
    """
    if Path(out_file).is_file() and (prompt is True):
        prompt = "Output file exists, would you like to overwrite? y/n: "
        if input(prompt).lower() != "y":
            return
    with open(out_file, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(file: Union[Path, str]) -> dict:
    """
    Load data from a JSON file.

    Args:
        file (Union[Path, str]): the full path of the file to load

    Returns:
        data extracted from the file. dict form
    """
    if Path(file).suffix.lower() == ".json":
        with open(file, 'r') as f:
            return json.load(f)
    else:
        raise AssertionError("Error. Expecting a .json file.")


def save_yaml(data: dict,
              out_file: Union[Path, str],
              prompt: bool = True) -> None:
    """Save data to a YAML file.

    Args:
        data (dict): the dictionary to be saved
        out_file (Union[Path, str]): full path to save file to
        prompt (bool): prompt user to confirm overwrite

    Returns:
        None
    """
    if Path(out_file).is_file() and (prompt is True):
        prompt = "Output file exists, would you like to overwrite? y/n: "
        if input(prompt).lower() != "y":
            return
    with open(out_file, 'w') as f:
        yaml.dump(data, f)


def load_yaml(file: Union[Path, str]) -> dict:
    """Load data from a YAML file.

    Args:
        file (Union[Path, str]): the full path of the file to load
    Returns:
        data extracted from the file. dict form
    """
    if Path(file).suffix.lower() in {".yaml", ".yml"}:
        with open(file, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise AssertionError("Error. Expecting a .yaml or .yml file.")


def check_file(file: Union[Path, str], output_type: Union[Path, str] = None) -> bool:
    """
    Check for files existence and prompt user if they want to load.

    Args:
        file (Union[Path, str]): the full path of the file to check
        output_type (Union[Path, str]): type of output file (e.g., "Manifest", "Detections")

    Returns:
        a boolean indicating whether a file was found and the user wants to load or not
    """

    if file is not None and Path(file).is_file():
        date = datetime.fromtimestamp(Path(file).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        if output_type is None:
            output_type = "Output"
        prompt = f"{output_type} file already exists and was last modified {date}, would you like to load it? y/n: "
        response = input(prompt)
        if response.lower() == "y":
            return True
        elif response.lower() == "n":
            return False
        else:
            print("Invalid input, proceeding without loading file.")
    return False


def class_list_to_dict(class_list: pd.DataFrame,
                       id_col: str = 'id',
                       class_col: str = 'class') -> dict:
    """
    Convert classification or detection class list dataframe to dictionary.

    Args:
        class_list (pd.DataFrame): dataframe with 'class' and 'id' columns
        id_col (str): column name for class ids, defaults to 'id'
        class_col (str): column name for class names, defaults to 'class'

    Returns:
        class_dict (dict): dictionary mapping ids to class names
    """
    if not {class_col, id_col}.issubset(class_list.columns):
        raise ValueError(f"DataFrame must contain '{class_col}' and '{id_col}' columns.")
    return {int(row[id_col]): row[class_col] for _, row in class_list.iterrows()}


def active_times(manifest,
                 file_col: str = "filepath",
                 camera_depth: int = 0,
                 timestamp_col: str = "datetime") -> pd.DataFrame:
    """
    Get start and stop dates for each camera folder.

    Args:
        manifest (pd.DataFrame): file manifest dataframe with file paths and timestamps
        camera_depth (int): directory depth from which to split cameras,
            with 0 being the root of the manifest_dir, defaults to 0
        file_col (str): column in manifest to use for file paths, defaults to "filepath"
        timestamp_col (str): column in manifest to use for timestamps, defaults to "datetime"

    Returns:
        times (pd.DataFrame): list of files with or without file modify dates
    """
    # from manifest file
    if not isinstance(manifest, pd.DataFrame):
        raise ValueError("Manifest must be a pandas DataFrame.")

    if not {file_col}.issubset(manifest.columns):
        raise ValueError(f"DataFrame must contain '{file_col}' filepath column.")

    # get filemodifydate timestamps if dne
    if timestamp_col not in manifest.columns:
        manifest[timestamp_col] = manifest[file_col].apply(lambda x: datetime.fromtimestamp(Path(x).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'))

    # get camera names if dne
    if "camera" not in manifest.columns:
        root_depth = len(Path(manifest[file_col].iloc[0]).parts) - 1
        camera_depth = root_depth + int(camera_depth)
        manifest["camera"] = manifest[file_col].apply(lambda x: Path(x).parts[camera_depth])

    times = manifest.groupby("camera").agg({timestamp_col: ['min', 'max']})

    return times


def sequence_calculation(manifest,
                         station_col: str,
                         sort_columns: list[str] = None,
                         file_col: str = "filepath",
                         timestamp_col: str = "datetime",
                         maxdiff: int = 60):
    """
    Simple sequence calculation based on time differences between consecutive images from the same station.
    Unlike sequence_classification(), does not apply any classification or labeling to the sequences.

    Args:
        - manifest (pd.DataFrame): DataFrame containing image file information,
            including 'file_col' and 'timestamp_col' columns
        - station_col (str): column name in the DataFrame representing the station or camera
        - sort_columns (list[str]): list of columns to sort by before calculating sequences.
                                    Defaults to None, which sorts by station_col and timestamp_col.
        - file_col (str): column name representing the file path. Defaults to "filepath".
        - timestamp_col (str): column name representing the timestamp in format "%Y-%m-%d %H:%M:%S", defaults to "datetime".
        - maxdiff (int): maximum time difference in seconds between consecutive images to be
            considered part of the same sequence. Defaults to 60.
    """
    if not isinstance(station_col, str) or station_col == '':
        raise Exception("'station_col' must be a non-empty string")

    # Sanity check to verify that maxdiff is a positive number
    if not isinstance(maxdiff, (int, float)) or maxdiff < 0:
        raise Exception("'maxdiff' must be a number >= 0")

    if not {file_col}.issubset(manifest.columns):
        raise ValueError(f"DataFrame must contain '{file_col}' column.")

    if not {timestamp_col}.issubset(manifest.columns):
        raise ValueError(f"DataFrame must contain '{timestamp_col}' column.")

    if sort_columns is None:
        sort_columns = [station_col, timestamp_col]

    # convert timestamp column to datetime if it's not already
    manifest[timestamp_col] = pd.to_datetime(manifest[timestamp_col], format="%Y-%m-%d %H:%M:%S")

    # sort by station and timestamp to ensure correct sequence calculation
    sort = manifest.sort_values(by=sort_columns).index
    manifest_sort = manifest.loc[sort].reset_index(drop=True)

    # Initialize sequence placeholder with zeros
    sequence_placeholder = np.zeros(len(manifest_sort))

    i = 0
    s = 0
    while i < len(manifest_sort):
        rows = [i]
        last_index = i+1

        while (last_index < len(manifest_sort) and not pd.isna(manifest_sort.loc[i, timestamp_col]) and
               not pd.isna(manifest_sort.loc[last_index, timestamp_col]) and
               manifest_sort.loc[last_index, station_col] == manifest_sort.loc[i, station_col] and
               (manifest_sort.loc[last_index, timestamp_col] - manifest_sort.loc[i, timestamp_col]).total_seconds() <= maxdiff):
            rows.append(last_index)
            last_index += 1

        sequence_placeholder[np.array(rows)] = int(s)

        i = last_index
        s += 1

    manifest_sort['sequence'] = sequence_placeholder

    return manifest_sort
