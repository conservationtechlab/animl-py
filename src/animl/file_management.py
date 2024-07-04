"""
    File Management Module

    This module provides functions and classes for managing files and directories.

    @ Kyra Swanson 2023
"""
import os
from glob import glob
from datetime import datetime, timedelta
import pandas as pd
from exiftool import ExifToolHelper


def build_file_manifest(image_dir, exif=True, out_file=None, offset=0, recursive=True):
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
    if check_file(out_file):
        return load_data(out_file)  # load_data(outfile) load file manifest
    if not os.path.isdir(image_dir):
        raise FileNotFoundError("The given directory does not exist.")

    files = glob(os.path.join(image_dir, '**', '*.*'), recursive=recursive)

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
        # select createdate if exists, else choose filemodify date
        files['CreateDate'] = files['CreateDate'].replace(r'^\s*$', None, regex=True)
        files["DateTime"] = files['CreateDate'].combine_first(files['FileModifyDate'])

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

        if not os.path.isdir(working_dir):
            raise FileNotFoundError("The given directory does not exist.")
        if not working_dir.endswith("/"):
            working_dir = working_dir + "/"

        self.basedir = working_dir + "Animl-Directory/"
        self.datadir = self.basedir + "Data/"
        self.vidfdir = self.basedir + "Frames/"
        self.linkdir = self.basedir + "Sorted/"

        # Create directories if they do not already exist
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
        if not os.path.exists(self.vidfdir):
            os.makedirs(self.vidfdir)
        if not os.path.exists(self.linkdir):
            os.makedirs(self.linkdir)

        # Assign specific file paths
        self.filemanifest = self.datadir + "FileManifest.csv"
        self.imageframes = self.datadir + "ImageFrames.csv"
        self.results = self.datadir + "Results.csv"
        self.predictions = self.datadir + "Predictions.csv"
        self.mdresults = self.datadir + "MD_Results.csv"
        self.mdraw = self.datadir + "MD_Raw.csv"


def save_data(data, out_file, prompt=True):
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


def load_data(file):
    """
    Load .csv File

    Args:
        - file (str): the full path of the file to load

    Returns:
        - data extracted from the file
    """
    ext = os.path.splitext(file)[1]
    if ext == ".csv":
        return pd.read_csv(file)
    else:
        raise AssertionError("Error. Expecting a .csv file.")


def check_file(file):
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


def active_times(manifest_dir, depth=1, recursive=True, offset=0):
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
