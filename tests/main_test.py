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
import unittest
import time
import shutil
from pathlib import Path
import pandas as pd

import animl


@unittest.skip
def main_test():
    start_time = time.time()

    image_dir = Path.cwd() / 'examples' / 'Southwest'
    workingdir = Path.cwd() / 'examples' / 'Southwest' / 'Animl-Directory'
    shutil.rmtree(workingdir, ignore_errors=True)

    megadetector = Path.cwd() / 'models/md_v5a.0.0.pt'
    if megadetector.exists():
        megadetector.unlink()

    #animl.download_model(animl.MEGADETECTOR['MDV5a'], out_dir='models')
    megadetector = Path.cwd() / 'models/md_v1000.0.0-sorrel.onnx'

    classifier_file = Path.cwd() / 'models/sdzwa_southwest_v3.pt'
    class_list_file = Path.cwd() / 'models/sdzwa_southwest_v3_classes.csv'

    animl.from_paths(image_dir, megadetector, classifier_file, class_list_file,
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

        print("Main Test Successful!")

    print(f"Pipeline took {time.time() - start_time:.2f} seconds")


main_test()
