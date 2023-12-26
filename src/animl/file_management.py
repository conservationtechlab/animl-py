"""
File Management Module

This module provides functions and classes for managing files and directories.

"""
import os
from glob import glob
from datetime import datetime
import pandas as pd


def build_file_manifest(image_dir, exif=True, out_file=None):
    """
    Extract exif Data and Create File Manifest

    Args
        - image_dir: directory to analyze
        - exif: returns date and time info from exif data, defaults to true
        - out_file: file path to which the data frame should be saved

    Returns
        - files: dataframe with or without file modify dates
    """
    if check_file(out_file):
        return load_data(out_file)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError("The given directory does not exist.")

    files = glob(os.path.join(image_dir, '**', '*.*'), recursive=True)

    files = pd.DataFrame(files, columns=["FilePath"])
    files["FileName"] = files["FilePath"].apply(
        lambda x: os.path.split(x)[1])

    if exif:
        files["FileModifyDate"] = files["FilePath"].apply(
            lambda x: datetime.fromtimestamp(
                os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M:%S'))

    if out_file:
        save_data(files, out_file)

    return files


class WorkingDirectory():
    """
    # Set Working Directory and Save File Global Variables
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, working_dir):
        if not os.path.isdir(working_dir):
            raise FileNotFoundError("The given directory does not exist.")
        if not working_dir.endswith("/"):
            working_dir = working_dir + "/"

        self.basedir = working_dir + "Animl-Directory/"
        self.datadir = self.basedir + "Data/"
        self.cropdir = self.basedir + "Crops/"
        self.vidfdir = self.basedir + "Frames/"
        self.linkdir = self.basedir + "Sorted/"

        # Create directories if they do not already exist
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
        if not os.path.exists(self.cropdir):
            os.makedirs(self.cropdir)
        if not os.path.exists(self.vidfdir):
            os.makedirs(self.vidfdir)
        if not os.path.exists(self.linkdir):
            os.makedirs(self.linkdir)

        # Assign specific file paths
        self.filemanifest = self.datadir + "FileManifest.csv"
        self.imageframes = self.datadir + "ImageFrames.csv"
        self.mdresults = self.datadir + "mdResults.csv"
        self.predictions = self.datadir + "Predictions.csv"
        self.results = self.datadir + "Results.csv"
        self.crops = self.datadir + "Crops.csv"


def save_data(data, file, prompt=True):
    """
    Save Data to Given File

    Args
        - data: the dataframe to be saved
        - file: the full path of the saved file
        - prompt: if true, prompts the user to confirm overwrite

    Returns
        None
    """
    if os.path.exists(file) and (prompt is True):
        prompt = "Output file exists, would you like to overwrite? y/n: "
        if input(prompt).lower() == "y":
            data.to_csv(file, index=False)
    else:
        data.to_csv(file, index=False)


def load_data(file):
    """
    Load .csv File

    Args
        - file: the full path of the file to load

    Returns
        data extracted from the file
    """
    ext = os.path.splitext(file)[1]
    if ext == ".csv":
        return pd.read_csv(file)
    else:
        raise AssertionError("Error. Expecting a .csv file.")


def check_file(file):
    """
    Check for files existence and prompt user if they want to load

    Args
        - file: the full path of the file to check

    Returns
        a boolean indicating wether a file was found
        and the user wants to load or not
    """
    if file is not None and os.path.isfile(file):
        date = os.path.getmtime(file)
        date = datetime.fromtimestamp(date)
        if input(f"Output file already exists and was last modified {date}, \
                 would you like to load it? y/n: ").lower() == "y":
            return True
    return False
