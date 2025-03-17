'''
    Main script for Custom YOLO.


    Usage example
    > python -m animl /home/usr/animl-py/examples/Southwest/


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
                   split, link)
from animl.utils.torch_utils import get_device
from animl.models import custom_detector
import typing

def main_paths(image_dir: str,
               detector_file: str,
               class_list: typing.List[str],
               sort: bool = True) -> pd.DataFrame:
    """
    This function is the main method to invoke all the sub functions
    to create a working directory for the image directory.

    Args:
        - image_dir (str): directory path containing the images or videos.
        - model_file (str): file path of the MegaDetector model.
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
        from animl.models import custom_detector
        detector = custom_detector.CustomYOLO(device=device)
        detections = detector.detect_batch(all_frames)
        detections = detect.parse_YOLO(detections, manifest=all_frames, out_file=working_dir.detections)
        print("USING CUSTOM YOLO DETECTOR")
        print("Extracting animal detections...")
        animals = split.get_animals_custom(detections)
        #animals = split.get_animals(detections)
        #empty = split.get_empty_custom(detections)
        # merge animal and empty, create symlinks
        print("Concatenating animal and empty dataframes...")
        manifest = animals
        manifest['confidence'] = manifest['conf']
        if sort:
            manifest = link.sort_species(manifest, working_dir.linkdir)
        file_management.save_data(manifest, working_dir.results)
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

    if (file_management.check_file(working_dir.detections)):
        detections = file_management.load_data(working_dir.detections)


    else:
        detector = custom_detector.CustomYOLO(device=device, config_path=config)
        detections = detector.detect_batch(all_frames)
        detections = detect.parse_YOLO(detections, manifest=all_frames, out_file=working_dir.detections)

        # merge animal and empty, create symlinks
        print("Concatenating animal and empty dataframes...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prediction_file = cfg.get('prediction_file', None)
        if prediction_file == "None":
            prediction_file = os.path.join(script_dir, "models", "custom_yolo_classes.csv")
        else:
            prediction_file = os.path.join(script_dir, prediction_file)
        prediction_dict = pd.read_csv(prediction_file, header=None, index_col=0).to_dict()[1]
        del prediction_dict['id']
        animals = split.get_animals_custom(detections, prediction_dict)
        #animals = split.get_animals(detections)
        empty = split.get_empty_custom(detections)

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


    if not os.path.isfile(args.classlist):
        prompt = "Class list not found, would you like to download Southwest_v3? y/n: "
        if input(prompt).lower() == "y":
            if not os.path.isdir(home):
                os.mkdir(home)
            print('Saving to', home)
            wget.download('https://sandiegozoo.box.com/shared/static/tetfkotf295espoaw8jyco4tk1t0trtt.csv',
                          out=home)
    # Call the main function
    main_paths(args.imagedir_config, args.detector, args.classlist)