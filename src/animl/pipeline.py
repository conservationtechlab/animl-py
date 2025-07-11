import os
import yaml
import torch
import pandas as pd
from animl import (classification, detection, file_management, video_processing, split, link)
from animl.utils.general import get_device, NUM_THREADS


def from_paths(image_dir: str,
               detector_file: str,
               classifier_file: str,
               classlist_file: str,
               class_label: str = "class",
               sort: bool = True,
               simple=True) -> pd.DataFrame:
    """
    This function is the main method to invoke all the sub functions
    to create a working directory for the image directory.

    Args:
        - image_dir (str): directory path containing the images or videos.
        - model_file (str): file path of the MegaDetector model.
        - class_model (str): file path of the classifier model.
        - classlist_file (list): list of classes or species for classification.
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
    print(files)
    # files["Station"] = files["FilePath"].apply(lambda x: x.split(os.sep)[-2])
    print("Found %d files." % len(files))

    # Video-processing to extract individual frames as images in to directory
    print("Processing videos...")
    all_frames = video_processing.extract_frames(files,
                                                 out_dir=working_dir.vidfdir,
                                                 out_file=working_dir.imageframes,
                                                 parallel=True,
                                                 num_workers=NUM_THREADS,
                                                 frames=3)

    # Run all images and video frames through MegaDetector
    print("Running images and video frames through MegaDetector...")
    if (file_management.check_file(working_dir.detections)):
        detections = file_management.load_data(working_dir.detections)
    else:
        detector = detection.load_detector(detector_file, "MDV5", device=device)
        md_results = detection.detect(detector,
                                      all_frames,
                                      file_col="Frame",
                                      batch_size=4,
                                      num_workers=NUM_THREADS,
                                      checkpoint_path=working_dir.mdraw,
                                      checkpoint_frequency=5000)
        # Convert MD JSON to pandas dataframe, merge with manifest
        print("Converting MD JSON to dataframe and merging with manifest...")
        detections = detection.parse_detections(md_results, manifest=all_frames, out_file=working_dir.detections)

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
    if simple:
        print("Classifying individual frames...")
        animals = classification.individual_classification(animals, predictions_raw, class_list[class_label])
        manifest = pd.concat([animals if not animals.empty else None, empty if not empty.empty else None]).reset_index(drop=True)
        # TODO: single output per file
    else:
        print("Classifying sequences...")
        manifest = classification.sequence_classification(animals, empty, predictions_raw,
                                                          class_list[class_label],
                                                          station_col='Station',
                                                          empty_class="",
                                                          sort_columns=None,
                                                          file_col="FilePath",
                                                          maxdiff=60)
    # create symlinks
    if sort:
        print("Sorting...")
        manifest = link.sort_species(manifest, working_dir.linkdir)

    file_management.save_data(manifest, working_dir.results)
    print("Final Results in " + str(working_dir.results))

    return manifest


def from_config(config):
    """
    This function is the main method to invoke all the sub functions
    to create a working directory for the image directory.

    Args:
        - config (str): path containing config file for inference

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
    all_frames = video_processing.extract_frames(files,
                                                 out_dir=working_dir.vidfdir,
                                                 out_file=working_dir.imageframes,
                                                 parallel=cfg.get('parallel', True),
                                                 num_workers=cfg.get('num_workers', NUM_THREADS),
                                                 frames=cfg.get('frames', 1), fps=fps)

    # Run all images and video frames through MegaDetector
    print("Running images and video frames through MegaDetector...")
    if (file_management.check_file(working_dir.detections)):
        detections = file_management.load_data(working_dir.detections)
    else:
        detector = detection.load_detector(cfg['detector_file'], device=device)
        md_results = detection.detect(detector,
                                      all_frames,
                                      file_col=cfg.get('file_col_detection', 'Frame'),
                                      batch_size=cfg.get('batch_size', 4),
                                      num_workers=cfg.get('num_workers', NUM_THREADS),
                                      checkpoint_path=working_dir.mdraw,
                                      checkpoint_frequency=cfg.get('checkpoint_frequency', -1))
        # Convert MD JSON to pandas dataframe, merge with manifest
        print("Converting MD JSON to dataframe and merging with manifest...")
        detections = detection.parse_detections(md_results, manifest=all_frames, out_file=working_dir.detections)

    # Extract animal detections from the rest
    animals = split.get_animals(detections)
    empty = split.get_empty(detections)

    # Use the classifier model to predict the species of animal detections
    print("Predicting species...")
    class_list = pd.read_csv(cfg['class_list'])
    classifier = classification.load_classifier(cfg['classifier_file'], len(class_list), device=device)
    predictions_raw = classification.classify(classifier, animals,
                                              device=device,
                                              file_col=cfg.get('file_col_classification', 'Frame'),
                                              batch_size=cfg.get('batch_size', 4),
                                              num_workers=cfg.get('num_workers', NUM_THREADS),
                                              out_file=working_dir.predictions)

    # Convert predictions to labels
    if station_dir:
        manifest = classification.sequence_classification(animals, empty, predictions_raw,
                                                          class_list[cfg.get('class_label_col', 'class')],
                                                          station_col='Station',
                                                          empty_class="",
                                                          sort_columns=["Station", "DateTime", "FrameNumber"],
                                                          file_col=cfg.get('file_col_classification', 'Frame'),
                                                          maxdiff=60)
    else:
        animals = classification.individual_classification(animals, predictions_raw, class_list[cfg.get('class_label_col', 'class')])
        # merge animal and empty
        manifest = pd.concat([animals if not animals.empty else None, empty if not empty.empty else None]).reset_index(drop=True)

    # Create Symlinks
    if cfg.get('sort', False):
        manifest = link.sort_species(manifest, cfg.get('link_dir', working_dir.linkdir),
                                     copy=cfg.get('copy', False))

    file_management.save_data(manifest, working_dir.results)
    print("Final Results in " + str(working_dir.results))

    return manifest
