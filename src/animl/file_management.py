"""
File Management Module

This module provides functions and classes for managing files and directories.

@ Kyra Swanson 2023
"""
import json
from shutil import copyfile
from pathlib import Path, PosixPath
from datetime import datetime, timedelta
import pandas as pd
import PIL
from typing import Optional


IMAGE_EXTENSIONS = {'.png', '.jpg', ',jpeg', ".tiff", '.tif"'}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv",
                    ".mpg", ".mpeg", ".asf", ".m4v"}
VALID_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def build_file_manifest(image_dir: str,
                        exif: bool = True,
                        out_file: Optional[str] = None,
                        offset: int = 0,
                        recursive: bool = True):
    """
    Find Image/Video Files and Gather exif Data.

    Args:
        image_dir (str): directory of files to analyze
        exif (bool): returns date and time info from exif data, defaults to True
        out_file (str): file path to which the dataframe should be saved
        offset (int): add timezone offset in hours to datetime column
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

    invalid = []

    def check_time(timestamp):
        input_formats = ['%Y:%m:%d %H:%M:%S', "%d-%m-%Y %H:%M", "%Y/%m/%d %H:%M:%S"]
        desired_format = '%Y-%m-%d %H:%M:%S'
        try:
            # If it already matches, return as is
            if datetime.strptime(timestamp, desired_format).strftime(desired_format) == timestamp:
                return timestamp
        except ValueError:
            pass
        # Try other input formats
        for fmt in input_formats:
            try:
                newtimestamp = datetime.strptime(timestamp, fmt)
                return newtimestamp.strftime(desired_format)
            except ValueError:
                continue
        # timestamp not recognized
        return None

    if exif:
        for i, row in files.iterrows():
            if row["extension"] in IMAGE_EXTENSIONS:
                try:
                    img = PIL.Image.open(row['filepath'])
                    files.loc[i, "width"] = img.size[0]
                    files.loc[i, "height"] = img.size[1]
                    files.loc[i, "createdate"] = img.getexif().get(0x0132)
                except PIL.UnidentifiedImageError:
                    invalid.append(i)

            elif row["extension"] in VIDEO_EXTENSIONS:
                try:
                    import cv2
                    vid = cv2.VideoCapture(row['filepath'])
                    if vid.isOpened():
                        files.loc[i, "width"] = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                        files.loc[i, "height"] = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid.release()
                except Exception:
                    invalid.append(i)

        # get filemodifydate as backup (videos, etc)
        files["filemodifydate"] = files["filepath"].apply(lambda x: datetime.fromtimestamp(Path(x).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'))
        files["filemodifydate"] = pd.to_datetime(files["filemodifydate"]) + timedelta(hours=offset)
        try:
            # select createdate if exists, else choose filemodify date
            files['createdate'] = files['createdate'].replace(r'^\s*$', None, regex=True)
            files["createdate"] = files['createdate'].apply(lambda x: check_time(x) if isinstance(x, str) else x)
            files["datetime"] = files['createdate'].fillna(files['filemodifydate'])
        except KeyError:
            files["datetime"] = files["filemodifydate"]

        # convert to datetime
        files["datetime"] = pd.to_datetime(files["datetime"])

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
              out_file: str,
              prompt: bool = True) -> None:
    """
    Save data to given file.

    Args:
        data (pd.DataFrame): the dataframe to be saved
        out_file (str): full path to save file to
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


def load_data(file: str) -> pd.DataFrame:
    """
    Load .csv File.

    Args:
        file (str): the full path of the file to load

    Returns:
        data extracted from the file. pd.dataframe form
    """
    if Path(file).suffix.lower() == ".csv":
        return pd.read_csv(file)
    else:
        raise AssertionError("Expecting a .csv file.")


def save_json(data: dict,
              out_file: str,
              prompt: bool = True) -> None:
    """
    Save data to a JSON file.

    Args:
        data (dict): the dictionary to be saved
        out_file (str): full path to save file to
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


def load_json(file: str) -> dict:
    """
    Load data from a JSON file.

    Args:
        file (str): the full path of the file to load

    Returns:
        data extracted from the file. dict form
    """
    if Path(file).suffix.lower() == ".json":
        with open(file, 'r') as f:
            return json.load(f)
    else:
        raise AssertionError("Error. Expecting a .json file.")


def check_file(file: str, output_type: str = None) -> bool:
    """
    Check for files existence and prompt user if they want to load.

    Args:
        file (str): the full path of the file to check
        output_type (str): type of output file (e.g., "Manifest", "Detections")

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


def save_detection_checkpoint(checkpoint_path: str, results: dict) -> None:
    """
    Save a checkpoint of the detection results to a JSON file.

    Args:
        checkpoint_path (str): the path to the checkpoint file
        results (list): a list of detection results to save
    """
    assert checkpoint_path is not None
    # Back up any previous checkpoints, to protect against crashes while we're writing
    # the checkpoint file.
    checkpoint_tmp_path = None
    if Path(checkpoint_path).is_file():
        checkpoint_tmp_path = str(checkpoint_path) + '_tmp'
        copyfile(checkpoint_path, checkpoint_tmp_path)

    # Write the new checkpoint
    save_json({'images': results}, checkpoint_path, prompt=False)

    # Remove the backup checkpoint if it exists
    if checkpoint_tmp_path is not None:
        Path(checkpoint_tmp_path).unlink()


def active_times(manifest_dir,
                 depth: int = 1,
                 recursive: bool = True,
                 offset: int = 0) -> pd.DataFrame:
    """
    Get start and stop dates for each camera folder.

    Args:
        manifest_dir (str): either file manifest or directory of files to analyze
        depth (int): directory depth from which to split cameras
        recursive (bool): recursively search thhrough all child directories
        offset (int): add timezone offset in hours to datetime column

    Returns:
        times (pd.DataFrame): list of files with or without file modify dates
    """
    # from manifest file
    if isinstance(manifest_dir, str):
        if check_file(manifest_dir):
            files = load_data(manifest_dir)  # load_data(outfile) load file manifest

    # from manifest dataframe
    elif isinstance(manifest_dir, pd.DataFrame):
        # get time stamps if dne
        if "filemodifydate" not in manifest_dir.columns:
            files = manifest_dir
            files["filemodifydate"] = files["filepath"].apply(lambda x: datetime.fromtimestamp(Path(x).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'))

    # from scratch
    elif Path(manifest_dir).is_dir():
        files = build_file_manifest(manifest_dir, exif=True, offset=offset, recursive=recursive)
    else:
        raise FileNotFoundError("Requires a file manifest or image directory.")

    files["camera"] = files["filepath"].apply(lambda x: Path(x).parts[depth])

    times = files.groupby("camera").agg({'filemodifydate': ['min', 'max']})

    return times
