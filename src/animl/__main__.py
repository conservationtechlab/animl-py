import argparse
import os
import wget
import pandas as pd
from . import (file_management, video_processing, megadetector,
               detect, split, inference)


def main(image_dir, detector_file, classifier_file, class_list):
    """
        This function is the main method to invoke all the sub functions
        to create a working directory for the image directory.

        Args:
            image_dir (str): The directory path containing the images or videos.
            model_file (str): The file path of the MegaDetector model.
            class_model (str): The file path of the classifier model.
            class_list (list): A list of classes or species for classification.

        Returns:
            pandas.DataFrame: Concatenated dataframe of animal and empty detections
    """
    print("Setting up working directory...")
    # Create a working directory, build the file manifest from img_dir
    working_dir = file_management.WorkingDirectory(image_dir)
    files = file_management.build_file_manifest(
        image_dir, out_file=working_dir.filemanifest
        )
    print("Found %d files." % len(files))
    print("Processing videos...")
    # Video-processing to extract individual frames as images in to directory
    all_frames = video_processing.images_from_videos(
        files, out_dir=working_dir.vidfdir,
        out_file=working_dir.imageframes, parallel=True, frames=2
        )
    print("Running images and video frames through MegaDetector...")
    # Run all images and video frames through MegaDetector
    if (file_management.check_file(working_dir.mdresults)):
        detections = file_management.load_data(working_dir.mdresults)
    else:
        detector = megadetector.MegaDetector(detector_file)
        md_results = detect.detect_MD_batch(detector,
                                            all_frames["Frame"],
                                            results=None, quiet=True)
        print("Converting MD JSON to dataframe and merging with manifest...")
        # Convert MD JSON to pandas dataframe, merge with manifest
        detections = detect.parse_MD(md_results, out_file=working_dir.mdresults)
    detections = all_frames.merge(detections, left_on="Frame", right_on="file")
    print("Extracting animal detections...")
    # Extract animal detections from the rest
    animals = split.get_animals(detections)
    empty = split.get_empty(detections)
    print(empty)
    print("Predicting species of animal detections...")
    classifier, classes = inference.load_classifier(classifier_file, class_list)
    # Use the classifier model to predict the species of animal detections
    animals = inference.predict_species(animals, classifier, classes,
                                        batch=4, out_file=working_dir.predictions)
    print("Concatenating animal and empty dataframes...")
    manifest = pd.concat([animals, empty])
    manifest.to_csv(working_dir.results)
    print("Final Results in " + working_dir.results)
    return manifest


# Create an argument parser
parser = argparse.ArgumentParser(description='Folder locations for the main script')
home = os.path.join(os.getcwd(), 'models')
# Create and parse arguements
parser.add_argument('image_dir', type=str,
                    help='Path to Image Directory')
parser.add_argument('detector_file', type=str, nargs='?',
                    help='Path to MD model',
                    default=os.path.join(home, 'md_v5a.0.0.pt'))
parser.add_argument('classifier_file', type=str, nargs='?',
                    help='Path to Class model',
                    default=os.path.join(home, 'southwest_v2.h5'))
parser.add_argument('class_list', type=str, nargs='?',
                    help='Path to class list',
                    default=os.path.join(home, 'southwest_v2_classes.txt'))
# Parse the command-line arguments

args = parser.parse_args()

if not os.path.isfile(args.detector_file):
    prompt = "MegaDetector not found, would you like to download? y/n: "
    if input(prompt).lower() == "y":
        if not os.path.isdir(home):
            os.mkdir(home)
        print('Saving to', home)
        wget.download('https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt',
                      out=home)

if not os.path.isfile(args.classifier_file):
    prompt = "Classifier not found, would you like to download Southwest_v2? y/n: "
    if input(prompt).lower() == "y":
        if not os.path.isdir(home):
            os.mkdir(home)
        print('Saving to', home)
        wget.download('https://sandiegozoo.box.com/shared/static/x63lnaxw8hag39mczeommqy9tw4t0ht9.h5',
                      out=home)

if not os.path.isfile(args.class_list):
    prompt = "Class list not found, would you like to download Southwest_v2? y/n: "
    if input(prompt).lower() == "y":
        if not os.path.isdir(home):
            os.mkdir(home)
        print('Saving to', home)
        wget.download('https://sandiegozoo.box.com/shared/static/hn8nput5pxjc3toao57gfn4h6zo1lyng.txt',
                      out=home)

# Call the main function
main(args.image_dir, args.detector_file, args.classifier_file, args.class_list)
