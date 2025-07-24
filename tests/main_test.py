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
start_time = time.time()
import os
from pathlib import Path
import pandas as pd

from animl import pipeline

print(Path.cwd())
image_dir = Path.cwd() / 'examples' / 'Southwest'
megadetector = Path.cwd() / 'models/md_v5a.0.0.pt'
classifier = Path.cwd() / 'models/sdzwa_southwest_v3.pt'
class_list = Path.cwd() / 'models/sdzwa_southwest_v3_classes.csv'

workingdir = Path(image_dir) / 'Animl-Directory'
if workingdir.exists():
    workingdir.rmdir()

pipeline.from_paths(image_dir, megadetector, classifier, class_list, sort=False)


results_path = Path(image_dir) / 'Animl-Directory' / 'Data' / 'Results.csv'
gt_path = Path.cwd() / 'tests' / 'GroundTruth' / 'Data' / 'Results.csv'
if results_path.exists():
    test_manifest = pd.read_csv(results_path)
    gt_manifest = pd.read_csv(results_path)

    if test_manifest.equals(gt_manifest):
        print("Test Successful!")
    else:
        print(test_manifest.ne(gt_manifest))

        print("Test Failure :(")

print(f"Pipeline took {time.time() - start_time:.2f} seconds")