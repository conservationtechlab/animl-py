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

from animl import pipeline

start_time = time.time()

print(Path.cwd())
image_dir = Path.cwd() / 'examples' / 'Southwest'
megadetector = Path.cwd() / 'models/md_v5a.0.0.pt'
classifier = Path.cwd() / 'models/sdzwa_southwest_v3.pt'
class_list = Path.cwd() / 'models/sdzwa_southwest_v3_classes.csv'

workingdir = Path(image_dir) / 'Animl-Directory'
if workingdir.exists():
    shutil.rmtree(workingdir)

# run pipeline
pipeline.from_paths(image_dir, megadetector, classifier, class_list, sort=False)

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
