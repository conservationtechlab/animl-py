'''
    Main script.

    Runs full animl workflow on a given directory.
    User must provide MegaDetector, Classifier, and Class list files,
    otherwise will pull MDv5 and the CTL Southwest v2 models by default.

    Usage example
    > python -m animl /home/usr/animl-py/examples/southwest/

    OR

    > python -m animl /image/dir megadetector.pt classifier.h5 class_file.csv

    Paths to model files must be edited to local machine.

    @ Kyra Swanson, 2023
'''
import argparse
import os
import wget
import torch
import pandas as pd
from . import (file_management, video_processing, megadetector,
               detect, split, classifiers, inference, symlink)


def main(image_dir, detector_file, classifier_file, class_list, sort=True):
    """
    This function is the main method to invoke all the sub functions
    to create a working directory for the image directory.

    Args:
        - image_dir (str): directory path containing the images or videos.
        - model_file (str): file path of the MegaDetector model.
        - class_model (str): file path of the classifier model.
        - class_list (list): list of classes or species for classification.
        - sort (bool): toggle option to create symlinks

    Returns:
        pandas.DataFrame: Concatenated dataframe of animal and empty detections
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = 'cpu'

    device = 'cpu'
    print("Searching directory...")
    # Create a working directory, build the file manifest from img_dir
    working_dir = file_management.WorkingDirectory(image_dir)
    files = file_management.build_file_manifest(image_dir,
                                                out_file=working_dir.filemanifest,
                                                exif=True)
    print("Found %d files." % len(files))

    # Video-processing to extract individual frames as images in to directory
    print("Processing videos...")
    all_frames = video_processing.extract_frames(files,out_dir=working_dir.vidfdir,
                                                 out_file=working_dir.imageframes,
                                                 parallel=True, frames=1)

    # Run all images and video frames through MegaDetector
    print("Running images and video frames through MegaDetector...")
    if (file_management.check_file(working_dir.mdresults)):
        detections = file_management.load_data(working_dir.mdresults)
    else:
        detector = megadetector.MegaDetector(detector_file, device=device)
        md_results = detect.detect_MD_batch(detector, all_frames["Frame"], 
                                            checkpoint_path=working_dir.mdraw, quiet=True)
        # Convert MD JSON to pandas dataframe, merge with manifest
        print("Converting MD JSON to dataframe and merging with manifest...")
        detections = detect.parse_MD(md_results, manifest=all_frames,
                                     out_file=working_dir.mdresults)

    # Extract animal detections from the rest
    print("Extracting animal detections...")
    animals = split.get_animals(detections)
    empty = split.get_empty(detections)

    # Use the classifier model to predict the species of animal detections
    print("Predicting species of animal detections...")
    print(class_list)
    classifier, classes = classifiers.load_model(classifier_file, class_list, device=device)
    animals = inference.predict_species(animals.reset_index(drop=True), classifier, classes, device=device,
                                        file_col="Frame", batch_size=4, out_file=working_dir.predictions)

    # merge animal and empty, create symlinks
    print("Concatenating animal and empty dataframes...")
    manifest = pd.concat([animals if not animals.empty else None, empty if not empty.empty else None]).reset_index(drop=True)
    if sort:
        manifest = symlink.symlink_species(manifest, working_dir.linkdir)

    file_management.save_data(manifest, working_dir.results)
    print("Final Results in " + working_dir.results)

    return manifest


# IF RUN FROM COMMAND LINE
# Create an argument parser
parser = argparse.ArgumentParser(description='Folder locations for the main script')
home = os.path.join(os.getcwd(), 'models')
# Create and parse arguements
parser.add_argument('image_dir', type=str,
                    help='Path to Image Directory')
parser.add_argument('--detector', type=str, nargs='?',
                    help='Path to MD model',
                    default=os.path.join(home, 'md_v5a.0.0.pt'))
parser.add_argument('--classifier', type=str, nargs='?',
                    help='Path to Class model',
                    default=os.path.join(home, 'southwest_v3.pt'))
parser.add_argument('--classlist', type=str, nargs='?',
                    help='Path to class list',
                    default=os.path.join(home, 'southwest_v3_classes.csv'))
# Parse the command-line arguments

args = parser.parse_args()

if not os.path.isfile(args.detector):
    prompt = "MegaDetector not found, would you like to download? y/n: "
    if input(prompt).lower() == "y":
        if not os.path.isdir(home):
            os.mkdir(home)
        print('Saving to', home)
        wget.download('https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt',
                      out=home)

if not os.path.isfile(args.classifier):
    prompt = "Classifier not found, would you like to download Southwest_v3? y/n: "
    if input(prompt).lower() == "y":
        if not os.path.isdir(home):
            os.mkdir(home)
        print('Saving to', home)
        wget.download('https://sandiegozoo.box.com/shared/static/ucbk8kc2h3qu15g4xbg0nvbghvo1cl97.pt',
                      out=home)

if not os.path.isfile(args.classlist):
    prompt = "Class list not found, would you like to download Southwest_v3? y/n: "
    if input(prompt).lower() == "y":
        if not os.path.isdir(home):
            os.mkdir(home)
        print('Saving to', home)
        wget.download('https://sandiegozoo.box.com/shared/static/tetfkotf295espoaw8jyco4tk1t0trtt.csv',
                      out=home)

# Call the main function
main(args.image_dir, args.detector, args.classifier, args.classlist)
