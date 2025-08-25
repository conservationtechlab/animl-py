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
import time
import shutil
from pathlib import Path
import pandas as pd

import animl
from animl.utils.general import get_device, NUM_THREADS


# @unittest.skip
def main_test():
    start_time = time.time()

    print(Path.cwd())
    image_dir = Path.cwd() / 'examples' / 'Southwest'
    megadetector = Path.cwd() / 'models/md_v5a.0.0.pt'
    classifier_file = Path.cwd() / 'models/sdzwa_southwest_v3.pt'
    class_list_file = Path.cwd() / 'models/sdzwa_southwest_v3_classes.csv'

    workingdir = Path(image_dir) / 'Animl-Directory'
    if workingdir.exists():
        shutil.rmtree(workingdir)

    # sequence
    sequence = False
    sort = True

    # run pipeline
    device = get_device()

    print("Searching directory...")
    working_dir = animl.WorkingDirectory(image_dir)
    files = animl.build_file_manifest(image_dir,
                                      out_file=working_dir.filemanifest,
                                      exif=True)
    print("Found %d files." % len(files))

    # Video-processing to extract individual frames as images in to directory
    print("Processing videos...")
    all_frames = animl.extract_frames(files,
                                      out_dir=working_dir.vidfdir,
                                      out_file=working_dir.imageframes,
                                      parallel=True,
                                      num_workers=NUM_THREADS,
                                      frames=3)

    # Run all images and video frames through MegaDetector
    print("Running images and video frames through MegaDetector...")
    if (animl.check_file(working_dir.detections)):
        detections = animl.load_data(working_dir.detections)
    else:
        detector = animl.load_detector(megadetector, "MDV5", device=device)
        md_results = animl.detect(detector,
                                  all_frames,
                                  file_col="Frame",
                                  batch_size=4,
                                  num_workers=NUM_THREADS,
                                  checkpoint_path=working_dir.mdraw,
                                  checkpoint_frequency=5)
        # Convert MD JSON to pandas dataframe, merge with manifest
        print("Converting MD JSON to dataframe and merging with manifest...")
        detections = animl.parse_detections(md_results, manifest=all_frames, out_file=working_dir.detections)

    # Extract animal detections from the rest
    animals = animl.get_animals(detections)
    empty = animl.get_empty(detections)

    # Use the classifier model to predict the species of animal detections
    print("Predicting species of animal detections...")
    class_list = animl.load_class_list(class_list_file)
    classifier = animl.load_classifier(classifier_file, len(class_list), device=device)
    predictions_raw = animl.classify(classifier,
                                     animals,
                                     device=device,
                                     file_col="Frame",
                                     batch_size=4,
                                     num_workers=NUM_THREADS,
                                     out_file=working_dir.predictions)

    if sequence:
        print("Classifying sequences...")
        manifest = animl.sequence_classification(animals,
                                                 empty,
                                                 predictions_raw,
                                                 class_list['class'],
                                                 station_col='Station',
                                                 empty_class="",
                                                 sort_columns=None,
                                                 file_col="FilePath",
                                                 maxdiff=60)
    else:
        print("Classifying individual frames...")
        animals = animl.individual_classification(animals, predictions_raw, class_list['class'])
        manifest = pd.concat([animals if not animals.empty else None, empty if not empty.empty else None]).reset_index(drop=True)
        # TODO: single output per file

    # create symlinks
    if sort:
        print("Sorting...")
        working_dir.activate_linkdir()
        manifest = animl.sort_species(manifest, working_dir.linkdir)

    animl.save_data(manifest, working_dir.results)
    print("Final Results in " + str(working_dir.results))

    results_path = Path(image_dir) / 'Animl-Directory' / 'Data' / 'Results.csv'
    gt_path = Path.cwd() / 'tests' / 'GroundTruth' / 'Data' / 'Results.csv'
    if results_path.exists():
        test_manifest = pd.read_csv(results_path)
        gt_manifest = pd.read_csv(gt_path)

        try:
            test_manifest['FilePath'].equals(gt_manifest['FilePath'])
        except ValueError:
            print("FilePath columns do not match. Test Failure :(")
            print(test_manifest.compare(gt_manifest))
            exit(1)

        try:
            test_manifest['prediction'].equals(gt_manifest['prediction'])
        except ValueError:
            print("Prediction columns do not match. Test Failure :(")
            print(test_manifest.compare(gt_manifest))
            exit(1)

        print("Test Successful!")

    print(f"Pipeline took {time.time() - start_time:.2f} seconds")


main_test()
