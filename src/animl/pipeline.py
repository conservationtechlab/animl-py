"""
Automated Pipeline Functions

@ Kyra Swanson 2023
"""
import os
import yaml
import torch
import pandas as pd
from animl import (file_management, video_processing, megadetector, detect,
                   split, classification, link)
from animl.utils.torch_utils import get_device


def from_paths(image_dir: str,
               detector_file: str,
               classifier_file: str,
               classlist_file: str,
               class_label: str = "Code",
               sort: bool = True,
               sequence: bool = False) -> pd.DataFrame:
    """
    This function is the main method to invoke all the sub functions
    to create a working directory for the image directory.

    Args:
        image_dir (str): directory path containing the images or videos.
        detector_file (str): file path of the MegaDetector model.
        classifier_file (str): file path of the classifier model.
        classlist_file (list): list of classes or species for classification.
        class_label: column in the class list that contains the label wanted
        sort (bool): toggle option to create symlinks
        sequence (bool): if True, run sequence_classification

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
    #files["Station"] = files["FilePath"].apply(lambda x: x.split(os.sep)[-2])
    print("Found %d files." % len(files))

    # Video-processing to extract individual frames as images in to directory
    print("Processing videos...")
    all_frames = video_processing.extract_frames(files, out_dir=working_dir.vidfdir,
                                                 out_file=working_dir.imageframes,
                                                 parallel=True, frames=3)

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
    animals = split.get_animals(detections)
    empty = split.get_empty(detections)

    # Use the classifier model to predict the species of animal detections
    print("Predicting species of animal detections...")
    class_list = pd.read_csv(classlist_file)
    classifier = classification.load_classifier(classifier_file, len(class_list), device=device)
    predictions_raw = classification.classify(classifier, animals,
                                              device=device,
                                              file_col="Frame",
                                              batch_size=4,
                                              num_workers=NUM_THREADS,
                                              out_file=working_dir.predictions)
    if sequence:
        print("Classifying sequences...")
        manifest = classification.sequence_classification(animals, empty, predictions_raw,
                                                          class_list[class_label],
                                                          station_col='Station',
                                                          empty_class="",
                                                          sort_columns=None,
                                                          file_col="FilePath",
                                                          maxdiff=60)
    else:
        print("Classifying individual frames...")
        animals = classification.individual_classification(animals, predictions_raw, class_list[class_label])
        manifest = pd.concat([animals if not animals.empty else None, empty if not empty.empty else None]).reset_index(drop=True)
        # TODO: single output per file

    # create symlinks
    if sort:
        print("Sorting...")
        manifest = link.sort_species(manifest, working_dir.linkdir)

    file_management.save_data(manifest, working_dir.results)
    print("Final Results in " + str(working_dir.results))

    return manifest


def from_config(config: str):
    """
    This function is the main method to invoke all the sub functions
    to create a working directory for the image directory.

    Args:
        config (str): path containing config file for inference

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

    # Station Col
    station_dir = cfg.get('station_dir', None)
    if station_dir:
        files["Station"] = files["FilePath"].apply(lambda x: x.split(os.sep)[station_dir])

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
    animals = split.get_animals(detections)
    empty = split.get_empty(detections)

    # Use the classifier model to predict the species of animal detections
    print("Predicting species...")
    class_list = pd.read_csv(cfg['class_list'])
    classifier = classification.load_model(cfg['classifier_file'], len(class_list), device=device)
    predictions_raw = classification.predict_species(animals, classifier, device=device,
                                                     file_col=cfg.get('file_col_classification', 'Frame'),
                                                     batch_size=cfg.get('batch_size', 4),
                                                     out_file=working_dir.predictions)

    # merge animal and empty, create symlinks
    if station_dir:
        manifest = classification.sequence_classification(animals, empty, predictions_raw,
                                                          class_list[cfg.get('class_label_col', 'Code')],
                                                          station_col='Station',
                                                          empty_class="",
                                                          sort_columns=["Station", "DateTime", "FrameNumber"],
                                                          file_col=cfg.get('file_col_classification', 'Frame'),
                                                          maxdiff=60)
    else:
        animals = classification.single_classification(animals, predictions_raw, class_list[cfg.get('class_label_col', 'Code')])
        # merge animal and empty, create symlinks
        manifest = pd.concat([animals if not animals.empty else None, empty if not empty.empty else None]).reset_index(drop=True)

    if cfg.get('sort', False):
        manifest = link.sort_species(manifest, cfg.get('link_dir', working_dir.linkdir),
                                     copy=cfg.get('copy', False))

    file_management.save_data(manifest, working_dir.results)
    print("Final Results in " + str(working_dir.results))

    return manifest
