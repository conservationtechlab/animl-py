import pandas as pd
import os
from glob import glob
from datetime import datetime
from imageUtils import is_image


# Extract exif Data and Create File Manifest
#
# @param imagedir file path
# @param exif returns date and time info from exif data, defaults to true
# @param offset add offset to videos, defaults to 0
# @param outfile file path to which the data frame should be saved
#
# @return files dataframe with or without file modify dates
def build_file_manifest(image_dir, exif=True, offset=0, out_file=None):
    if check_file(out_file):
        return load_data(out_file)  # load_data(outfile) load file manifest
    if not os.path.isdir(image_dir):
        raise FileNotFoundError("The given directory does not exist.")

    files = glob(os.path.join(image_dir, '**', '*.*'), recursive=True)

    files = pd.DataFrame(files, columns=["FilePath"])
    files["FileName"] = files["FilePath"].apply(
        lambda x: os.path.split(x)[1])
    files["FileModifyDate"] = files["FilePath"].apply(
        lambda x: datetime.fromtimestamp(
            os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M:%S'))

    if(out_file):
        save_data(files, out_file)

    return files


# Set Working Directory and Save File Global Variables
#
# @param workingdir local directory that contains data to process
# @param pkg.env environment to create global variables in
#
# @return None
class WorkingDirectory():
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
        self.results = self.datadir + "Results.csv"
        self.crops = self.datadir + "Crops.csv"
        self.predictions = self.datadir + "Predictions.csv"
        self.mdresults = self.datadir + "mdresults.csv"


# Save Data to Given File
#
# @param data the dataframe to be saved
# @param file the full path of the saved file
# @param prompt if true, prompts the user to confirm overwrite
#
# @return none
def save_data(data, file, prompt=True):
    if os.path.exists(file) and (prompt is True):
        prompt = "Output file exists, would you like to overwrite? y/n: "
        if input(prompt).lower() == "y":
            data.to_csv(file, index=False)
    else:
        data.to_csv(file, index=False)


# Load .csv File
#
# @param file the full path of the file to load
#
# @return data extracted from the file
def load_data(file):
    ext = os.path.splitext(file)[1]
    if ext == ".csv":
        return pd.read_csv(file)
    else:
        raise AssertionError("Error. Expecting a .csv file.")


# Check for files existence and prompt user if they want to load
#
# @param file the full path of the file to check
#
# @return a boolean indicating wether a file was found
#             and the user wants to load or not
def check_file(file):
    if (file is not None) and os.path.isfile(file):
        date = os.path.getmtime(file)
        date = datetime.fromtimestamp(date)
        prompt = "Output file already exists and was last modified {}, \
                 would you like to load it? y/n: ".format(date)
        if input(prompt).lower() == "y":
            return(True)
    return(False)
