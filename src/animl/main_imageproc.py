'''
This module contains the main function for the species detection using
the MD, yolo5 and ai4eutils models
'''
import argparse
#import sys
import pandas as pd
import file_management
import videoProcessing
import detectMD
import parseResults
import splitData
import predictSpecies


def main(image_dir, model_file, class_model, class_list):
    '''
    This function is the main method to invoke all the sub functions
    to create a working directory for the image directory.

    Parameters-
    image_dir (str): The directory path containing the images or videos.
    model_file (str): The file path of the MegaDetector model.
    class_model (str): The file path of the classifier model.
    class_list (list): A list of classes or species for classification.

    Returns:
    pandas.DataFrame: Concatenated dataframe of animal and empty detections.

    '''

    print("Setting up working directory...")
    # Create a working directory, build the file manifest from img_dir
    working_dir = file_management.WorkingDirectory(image_dir)
    files = file_management.build_file_manifest(
        image_dir, out_file=working_dir.filemanifest
        )
    print("Processing videos...")
    # Video-processing to extract individual frames as images in to directory
    all_frames = videoProcessing.images_from_videos(
        files, out_dir=working_dir.vidfdir,
        out_file=working_dir.imageframes, parallel=True, frames=2
        )
    print("Running images and video frames through MegaDetector...")
    # Run all images and video frames through MegaDetector
    md_results = detectMD.detect_MD_batch(
        model_file, all_frames["Frame"],
        checkpoint_path=None, checkpoint_frequency=-1,
        results=None, n_cores=1, quiet=True
        )
    print("Converting MD JSON to pd dataframe and merging with manifest...")
    # Convert MD JSON to pandas dataframe, merge with manifest
    md_res = parseResults.parseMD(
        md_results, manifest=all_frames, out_file=working_dir.mdresults
        )
    print("Extracting animal detections...")
    # Extract animal detections from the rest
    animals = splitData.getAnimals(md_res)
    empty = splitData.getEmpty(md_res)
    print("Predicting species of animal detections...")
    # Use the classifier model to predict the species of animal detections
    pred_results = predictSpecies.predictSpecies(animals, class_model, batch=4)
    print("Applying predictions to animal detections...")
    animals = parseResults.applyPredictions(
        animals, pred_results, class_list, out_file=working_dir.predictions
        )
    print("Concatenating animal and empty dataframes...")
    manifest = pd.concat([animals, empty])
    manifest.to_csv(working_dir.results)
    print("Final Results in "+ working_dir.results)

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Folder locations for the main script'
        )

    # Create and parse arguements
    parser.add_argument('image_dir', type=str, help='Path to Image Directory')
    parser.add_argument('model_file', type=str, help='Path to MD model')
    parser.add_argument('class_model', type=str, help='Path to Class model')
    parser.add_argument('class_list', type=str, help='Path to class list')
    parser.add_argument('results_file', type=str, help='Path to Final results')
    parser.add_argument('linkdir', type=str, help='Destination Directory for symlinks')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function
    main(
        args.image_dir, args.model_file, args.class_model, args.class_list
        )
