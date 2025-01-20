'''
    Main script.

    Runs full animl workflow on a given directory.
    User must provide MegaDetector, Classifier, and Class list files,
    otherwise will pull MDv5 and the CTL Southwest v2 models by default.

    Usage example
    > python -m animl /home/usr/animl-py/examples/Southwest/

    OR

    > python -m animl /image/dir megadetector.pt classifier.h5 class_file.csv

    Paths to model files must be edited to local machine.

    @ Kyra Swanson, 2023
'''
import argparse
import os
import wget
import yaml
import torch
import pandas as pd
from animl import (file_management, video_processing, detect,
                   split, classification, link)
from animl.models import megadetector
from animl.utils.torch_utils import get_device
import typing


def main_paths(image_dir: str,
               detector_file: str,
               classifier_file: str,
               class_list: typing.List[str],
               sort: bool = True) -> pd.DataFrame:
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
    device = get_device()

    print("Searching directory...")
    # Create a working directory, build the file manifest from img_dir
    working_dir = file_management.WorkingDirectory(image_dir)
    files = file_management.build_file_manifest(image_dir,
                                                out_file=working_dir.filemanifest,
                                                exif=True)
    print("Found %d files." % len(files))

    # Video-processing to extract individual frames as images in to directory
    print("Processing videos...")
    all_frames = video_processing.extract_frames(files, out_dir=working_dir.vidfdir,
                                                 out_file=working_dir.imageframes,
                                                 parallel=True, frames=1)

    # Run all images and video frames through MegaDetector
    print("Running images and video frames through MegaDetector...")
    if (file_management.check_file(working_dir.detections)):
        detections = file_management.load_data(working_dir.detections)
    else:
        detector = megadetector.MegaDetector(detector_file, device=device)
        md_results = detect.detect_MD_batch(detector, all_frames, file_col="Frame",
                                            checkpoint_path=working_dir.mdraw, quiet=True)
        # Convert MD JSON to pandas dataframe, merge with manifest
        print("Converting MD JSON to dataframe and merging with manifest...")
        detections = detect.parse_MD(md_results, manifest=all_frames, out_file=working_dir.detections)

    # Extract animal detections from the rest
    print("Extracting animal detections...")
    animals = split.get_animals(detections)
    empty = split.get_empty(detections)

    # Use the classifier model to predict the species of animal detections
    print("Predicting species of animal detections...")
    print(class_list)
    classifier, classes = classification.load_model(classifier_file, class_list, device=device)
    animals = classification.predict_species(animals, classifier, classes, device=device,
                                             file_col="Frame", batch_size=4, out_file=working_dir.predictions)

    # merge animal and empty, create symlinks
    print("Concatenating animal and empty dataframes...")
    manifest = pd.concat([animals if not animals.empty else None, empty if not empty.empty else None]).reset_index(drop=True)
    if sort:
        manifest = link.sort_species(manifest, working_dir.linkdir)

    file_management.save_data(manifest, working_dir.results)
    print("Final Results in " + str(working_dir.results))

    return manifest


def main_config(config):
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

    print(f'Using config "{config}"')
    cfg = yaml.safe_load(open(config, 'r'))

    # get image dir and cuda defaults
    image_dir = cfg['image_dir']
    device = cfg.get('device', get_device())

    if device != 'cpu' and not torch.cuda.is_available():
        device = 'cpu'

    print("Searching directory...")
    # Create a working directory, default to image_dir
    working_dir = file_management.WorkingDirectory(cfg.get('working_dir', image_dir))
    files = file_management.build_file_manifest(image_dir,
                                                out_file=working_dir.filemanifest,
                                                exif=cfg.get('exif', True))
    print("Found %d files." % len(files))

    # Video-processing to extract individual frames as images in to directory
    print("Processing videos...")
    fps = cfg.get('fps', None)
    if fps == "None":
        fps = None
    all_frames = video_processing.extract_frames(files, out_dir=working_dir.vidfdir,
                                                 out_file=working_dir.imageframes,
                                                 parallel=cfg.get('parallel', True),
                                                 frames=cfg.get('frames', 1), fps=fps)

    # Run all images and video frames through MegaDetector
    print("Running images and video frames through MegaDetector...")
    if (file_management.check_file(working_dir.detections)):
        detections = file_management.load_data(working_dir.detections)
    else:
        detector = megadetector.MegaDetector(cfg['detector_file'], device=device)
        md_results = detect.detect_MD_batch(detector, all_frames, file_col=cfg.get('file_col_detection', 'Frame'),
                                            checkpoint_path=working_dir.mdraw,
                                            checkpoint_frequency=cfg.get('checkpoint_frequency', -1),
                                            quiet=True)
        # Convert MD JSON to pandas dataframe, merge with manifest
        print("Converting MD JSON to dataframe and merging with manifest...")
        detections = detect.parse_MD(md_results, manifest=all_frames, out_file=working_dir.detections)

    # Extract animal detections from the rest
    print("Extracting animal detections...")
    animals = split.get_animals(detections)
    empty = split.get_empty(detections)

    # Use the classifier model to predict the species of animal detections
    print("Predicting species of animal detections...")
    classifier, classes = classification.load_model(cfg['classifier_file'], cfg['class_list'], device=device)
    animals = classification.predict_species(animals, classifier, classes, device=device,
                                             file_col=cfg.get('file_col_classification', 'Frame'),
                                             batch_size=cfg.get('batch_size', 4),
                                             out_file=working_dir.predictions)

    # merge animal and empty, create symlinks
    print("Concatenating animal and empty dataframes...")
    manifest = pd.concat([animals if not animals.empty else None, empty if not empty.empty else None]).reset_index(drop=True)
    if cfg.get('sort', False):
        manifest = link.sort_species(manifest, cfg.get('link_dir', working_dir.linkdir),
                                     copy=cfg.get('copy', False))

    file_management.save_data(manifest, working_dir.results)
    print("Final Results in " + str(working_dir.results))

    return manifest


# IF RUN FROM COMMAND LINE
parser = argparse.ArgumentParser(description='Folder locations for the main script')
home = os.path.join(os.getcwd(), 'models')
# Create and parse arguements
parser.add_argument('imagedir_config', type=str,
                    help='Path to Image Directory or Config File')
parser.add_argument('--detector', type=str, nargs='?',
                    help='Path to MD model',
                    default=os.path.join(home, 'md_v5a.0.0.pt'))
parser.add_argument('--classifier', type=str, nargs='?',
                    help='Path to Class model',
                    default=os.path.join(home, 'sdzwa_southwest_v3.pt'))
parser.add_argument('--classlist', type=str, nargs='?',
                    help='Path to class list',
                    default=os.path.join(home, 'sdzwa_southwest_v3_classes.csv'))
args = parser.parse_args()

# first argument is config file
if os.path.isfile(args.imagedir_config):
    main_config(args.imagedir_config)

# first argument is a directory
else:
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
    main_paths(args.imagedir_config, args.detector, args.classifier, args.classlist)
