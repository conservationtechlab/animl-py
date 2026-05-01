"""
Test pipeline functions: from_paths and from_config

"""
import unittest
import time
import shutil
from pathlib import Path
import pandas as pd

import animl


@unittest.skip
def from_paths_test():
    start_time = time.time()

    image_dir = Path.cwd() / 'examples' / 'Southwest'
    workingdir = Path.cwd() / 'examples' / 'Southwest' / 'Animl-Directory'
    shutil.rmtree(workingdir, ignore_errors=True)

    megadetector = Path.cwd() / 'models/md_v1000.0.0-sorrel.onnx'
    classifier_file = Path.cwd() / 'models/sdzwa_southwest_v3.pt'
    class_list_file = Path.cwd() / 'models/sdzwa_southwest_v3_classes.csv'

    results = animl.from_paths(image_dir, megadetector, classifier_file, class_list_file,
                               sort=True, visualize=True, sequence=False)

    results_path = Path(image_dir) / 'Animl-Directory' / 'Results.csv'
    gt_path = Path.cwd() / 'tests' / 'GroundTruth' / 'main' / 'Results.csv'
    if results_path.exists():
        test_manifest = pd.read_csv(results_path)
        gt_manifest = pd.read_csv(gt_path)

        try:
            test_manifest['filepath'].equals(gt_manifest['filepath'])
        except ValueError:
            print("filepath columns do not match. Test Failure :(")
            print(test_manifest.compare(gt_manifest))
            exit(1)

        try:
            test_manifest['prediction'].equals(gt_manifest['prediction'])
        except ValueError:
            print("Prediction columns do not match. Test Failure :(")
            print(test_manifest.compare(gt_manifest))
            exit(1)

        print("from_paths Test Successful!")

    print(f"Test completed in {time.time() - start_time:.2f} seconds")


@unittest.skip
def from_paths_detect_only_test():
    start_time = time.time()

    image_dir = Path.cwd() / 'examples' / 'Southwest'
    workingdir = Path.cwd() / 'examples' / 'Southwest' / 'Animl-Directory'
    shutil.rmtree(workingdir, ignore_errors=True)

    megadetector = Path.cwd() / 'models/md_v1000.0.0-sorrel.onnx'

    detections = animl.from_paths(image_dir, megadetector,
                                  classifier_file=None, classlist_file=None,
                                  detect_only=True)

    assert isinstance(detections, pd.DataFrame), "Expected a DataFrame"
    assert len(detections) > 0, "Expected detections to be non-empty"
    print("from_paths detect_only Test Successful!")
    print(f"Test completed in {time.time() - start_time:.2f} seconds")


@unittest.skip
def from_paths_sequence_test():
    start_time = time.time()

    image_dir = Path.cwd() / 'examples' / 'Southwest'
    workingdir = Path.cwd() / 'examples' / 'Southwest' / 'Animl-Directory'
    shutil.rmtree(workingdir, ignore_errors=True)

    megadetector = Path.cwd() / 'models/md_v1000.0.0-sorrel.onnx'
    classifier_file = Path.cwd() / 'models/sdzwa_southwest_v3.pt'
    class_list_file = Path.cwd() / 'models/sdzwa_southwest_v3_classes.csv'

    results = animl.from_paths(image_dir, megadetector, classifier_file, class_list_file,
                               sort=False, visualize=False, sequence=True)

    assert isinstance(results, pd.DataFrame), "Expected a DataFrame"
    assert 'prediction' in results.columns, "Expected 'prediction' column in results"
    print("from_paths sequence Test Successful!")
    print(f"Test completed in {time.time() - start_time:.2f} seconds")


@unittest.skip
def from_config_test():
    start_time = time.time()

    workingdir = Path.cwd() / 'examples' / 'Southwest' / 'Animl-Directory'
    shutil.rmtree(workingdir, ignore_errors=True)

    config = Path.cwd() / 'examples' / 'animl.yml'

    results = animl.from_config(config)

    assert isinstance(results, pd.DataFrame), "Expected a DataFrame"
    assert 'prediction' in results.columns, "Expected 'prediction' column in results"
    print("from_config Test Successful!")
    print(f"Test completed in {time.time() - start_time:.2f} seconds")


from_paths_test()
from_paths_detect_only_test()
from_paths_sequence_test()
from_config_test()
