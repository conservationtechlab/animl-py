"""
    File Management Module

    This module provides functions and classes for managing files and directories.

    @ Kyra Swanson 2023
"""
import os
from pathlib import Path
from glob import glob
from datetime import datetime, timedelta
import pandas as pd
from exiftool import ExifToolHelper
from typing import Optional


VALID_EXTENSIONS = {'.png', '.jpg', ',jpeg', ".tiff",
                    ".mp4", ".avi", ".mov", ".wmv",
                    ".mpg", ".mpeg", ".asf", ".m4v"}


def build_file_manifest(image_dir: str,
                        exif: bool = True,
                        out_file: Optional[str] = None,
                        offset: int = 0,
                        recursive: bool = True):
    """
    Find Image/Video Files and Gather exif Data

    Args:
        - image_dir (str): directory of files to analyze
        - exif (bool): returns date and time info from exif data, defaults to True
        - out_file (str): file path to which the dataframe should be saved
        - offset (int): add timezone offset in hours to datetime column
        - recursive (bool): recursively search thhrough all child directories

    Returns:
        - files (pd.DataFrame): list of files with or without file modify dates
    """
    image_dir = Path(image_dir)
    if check_file(out_file):
        return load_data(out_file)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"The given directory: {image_dir}, does not exist.")

    files = glob(os.path.join(image_dir, '**', '*.*'), recursive=recursive)

    # only keep images and videos
    files = [f for f in files if os.path.splitext(os.path.basename(f))[1].lower() in VALID_EXTENSIONS]

    # no files found, return empty dataframe
    if not files:
        return pd.DataFrame()

    files = pd.DataFrame(files, columns=["FilePath"])
    files["FileName"] = files["FilePath"].apply(
        lambda x: os.path.split(x)[1])

    if exif:
        et = ExifToolHelper()
        file_exif = et.get_tags(files["FilePath"].tolist(), tags=["CreateDate", "ImageWidth", "ImageHeight"])
        et.terminate()  # close exiftool
        # merge exif data with manifest
        file_exif = pd.DataFrame(file_exif).rename(columns={"EXIF:CreateDate": "CreateDate",
                                                            "File:ImageWidth": "Width",
                                                            "File:ImageHeight": "Height"})

        # adjust for windows if necessary
        file_exif["SourceFile"] = file_exif["SourceFile"].apply(lambda x: os.path.normpath(x))
        files = files.merge(pd.DataFrame(file_exif), left_on="FilePath", right_on="SourceFile")
        # get filemodifydate as backup (videos, etc)
        files["FileModifyDate"] = files["FilePath"].apply(lambda x: datetime.fromtimestamp(os.path.getmtime(x)))
        files["FileModifyDate"] = files["FileModifyDate"] + timedelta(hours=offset)
        try:
            # select createdate if exists, else choose filemodify date
            files['CreateDate'] = files['CreateDate'].replace(r'^\s*$', None, regex=True)
            files["CreateDate"] = files['CreateDate'].apply(lambda x: datetime.strptime(str(x), '%Y:%m:%d %H:%M:%S') if isinstance(x, str) else x)
            files["DateTime"] = files['CreateDate'].combine_first(files['FileModifyDate'])
        except KeyError:
            files["DateTime"] = files["FileModifyDate"]

    if out_file:
        save_data(files, out_file)

    return files


class WorkingDirectory():
    """
    Set Working Directory and Save File Global Variables

    Constructor requires root working_directory
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, working_dir):
        working_dir = Path(r"" + working_dir)  # OS-agnostic path
        if not working_dir.is_dir():
            raise FileNotFoundError(f"The given directory: {working_dir}, does not exist.")

        self.basedir = working_dir / Path("Animl-Directory/")
        self.datadir = self.basedir / Path("Data/")
        self.vidfdir = self.basedir / Path("Frames/")
        self.linkdir = self.basedir / Path("Sorted/")

        # Create directories if they do not already exist
        self.basedir.mkdir(exist_ok=True)
        self.datadir.mkdir(exist_ok=True)
        self.vidfdir.mkdir(exist_ok=True)
        self.linkdir.mkdir(exist_ok=True)

        # Assign specific file paths
        self.filemanifest = self.datadir / Path("FileManifest.csv")
        self.imageframes = self.datadir / Path("ImageFrames.csv")
        self.results = self.datadir / Path("Results.csv")
        self.predictions = self.datadir / Path("Predictions.csv")
        self.detections = self.datadir / Path("Detections.csv")
        self.mdraw = self.datadir / Path("MD_Raw.json")


def save_data(data: pd.DataFrame, out_file: str, prompt: bool = True) -> None:
    """
    Save Data to Given File

    Args:
        - data (pd.DataFrame): the dataframe to be saved
        - out_file (str): full path to save file to
        - prompt (bool): prompts the user to confirm overwrite

    Returns:
        None
    """
    if os.path.exists(out_file) and (prompt is True):
        prompt = "Output file exists, would you like to overwrite? y/n: "
        if input(prompt).lower() == "y":
            data.to_csv(out_file, index=False)
    else:
        data.to_csv(out_file, index=False)


def load_data(file: str) -> pd.DataFrame:
    """
    Load .csv File

    Args:
        - file (str): the full path of the file to load

    Returns:
        - data extracted from the file. pd.dataframe form
    """
    ext = os.path.splitext(file)[1]
    if ext == ".csv":
        return pd.read_csv(file)
    else:
        raise AssertionError("Error. Expecting a .csv file.")


def check_file(file: str) -> bool:
    """
    Check for files existence and prompt user if they want to load

    Args:
        - file (str): the full path of the file to check

    Returns:
        - a boolean indicating whether a file was found and
          the user wants to load or not
    """

    if file is not None and os.path.isfile(file):
        date = os.path.getmtime(file)
        date = datetime.fromtimestamp(date)
        prompt = "Output file already exists and was last modified {}, \
                 would you like to load it? y/n: ".format(date)
        if input(prompt).lower() == "y":
            return True
    return False


def active_times(manifest_dir: str,
                 depth: int = 1,
                 recursive: bool = True,
                 offset: int = 0) -> pd.DataFrame:
    """
    Get start and stop dates for each camera folder

    Args:
        - manifest_dir (str): either file manifest or directory of files to analyze
        - depth (int): directory depth from which to split cameras
        - recursive (bool): recursively search thhrough all child directories
        - offset (int): add timezone offset in hours to datetime column

    Returns:
        - times (pd.DataFrame): list of files with or without file modify dates

    """
    # from manifest file
    if check_file(manifest_dir):
        files = load_data(manifest_dir)  # load_data(outfile) load file manifest

    # from manifest dataframe
    elif isinstance(manifest_dir, pd.DataFrame):
        # get time stamps if dne
        if "FileModifyDate" not in manifest_dir.columns:
            files = manifest_dir
            files["FileModifyDate"] = files["FilePath"].apply(lambda x: datetime.fromtimestamp(os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M:%S'))

    # from scratch
    elif os.path.isdir(manifest_dir):
        files = build_file_manifest(manifest_dir, exif=True, offset=offset, recursive=recursive)

    else:
        raise FileNotFoundError("Requires a file manifest or image directory.")

    files["Camera"] = files["FilePath"].apply(lambda x: x.split(os.sep)[depth])

    times = files.groupby("Camera").agg({'FileModifyDate': ['min', 'max']})

    return times
