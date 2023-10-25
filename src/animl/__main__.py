import argparse
import pandas as pd
import file_management
import video_processing
import detectMD
import parse_results 
import split
import classify

def main(image_dir, md_model, class_model, class_list):
    """
    This function is the main method to invoke all the sub functions
    to create a working directory for the image directory.

    Args:
        image_dir (str): The directory path containing the images or videos.
        model_file (str): The file path of the MegaDetector model.
        class_model (str): The file path of the classifier model.
        class_list (list): A list of classes or species for classification.

    Returns:
        pandas.DataFrame: Concatenated dataframe of animal and empty detections.
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
        detector = detectMD.load_MD_model(md_model)
        md_results = detectMD.detect_MD_batch(
            detector, all_frames["Frame"], results=None, 
            )
        print("Converting MD JSON to pd dataframe and merging with manifest...")
        # Convert MD JSON to pandas dataframe, merge with manifest
        detections = parse_results.from_MD(
            md_results, manifest=all_frames, out_file=working_dir.mdresults
            )
    print("Extracting animal detections...")
    # Extract animal detections from the rest
    animals = split.getAnimals(detections)
    empty = split.getEmpty(detections)
    print("Predicting species of animal detections...")
    classifier = classify.load_classifier(class_model)
    # Use the classifier model to predict the species of animal detections
    pred_results = classify.predict_species(animals, classifier, batch=4)
    print("Applying predictions to animal detections...")
    animals = parse_results.from_classifier(
        animals, pred_results, class_list, out_file=working_dir.predictions
        )
    print("Concatenating animal and empty dataframes...")
    manifest = pd.concat([animals, empty])
    manifest.to_csv(working_dir.results)
    print("Final Results in " + working_dir.results)

# Create an argument parser
parser = argparse.ArgumentParser(description='Folder locations for the main script')

# Create and parse arguements
parser.add_argument('image_dir', type=str, help='Path to Image Directory')
parser.add_argument('model_file', type=str, help='Path to MD model')
parser.add_argument('class_model', type=str, help='Path to Class model')
parser.add_argument('class_list', type=str, help='Path to class list')
# Parse the command-line arguments
args = parser.parse_args()

# Call the main function
main(args.image_dir, args.model_file, args.class_model, args.class_list)
