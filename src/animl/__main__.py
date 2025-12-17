'''
Main script.

Runs full animl workflow on a given directory.
User must provide MegaDetector, Classifier, and Class list files,
otherwise will pull MDv5 and the CTL Southwest v2 models by default.

Usage example
> python -m animl /home/usr/animl-py/examples/Southwest/

OR

> python -m animl /image/dir megadetector.pt classifier.pt class_file.csv

Paths to model files must be edited to local machine.

@ Kyra Swanson, 2023
'''
import time
import argparse
from pathlib import Path

from animl import pipeline
import animl.models.download as models

# Start timer
start_time = time.time()

# IF RUN FROM COMMAND LINE
parser = argparse.ArgumentParser(description='Folder locations for the main script')
home = Path.cwd() / 'models'
# Create and parse arguements
parser.add_argument('imagedir_config', type=str,
                    help='Path to Image Directory or Config File')
parser.add_argument('--detector', type=str, nargs='?',
                    help='Path to MD model',
                    default=Path(home / 'md_v5a.0.0.pt'))
parser.add_argument('--classifier', type=str, nargs='?',
                    help='Path to Class model',
                    default=Path(home / 'sdzwa_southwest_v3.pt'))
parser.add_argument('--classlist', type=str, nargs='?',
                    help='Path to class list',
                    default=Path(home / 'sdzwa_southwest_v3_classes.csv'))
parser.add_argument('--detect_only', '-d', action='store_true',
                    help='Run detection only, skip classification')
parser.add_argument('--sort', '-s', action='store_true',
                    help='Sort images into subfolders based on classification')
parser.add_argument('--visualize', '-v', action='store_true',
                    help='Visualize detections and classifications on images')
args = parser.parse_args()

# first argument is config file
if Path(args.imagedir_config).is_file():
    pipeline.from_config(args.imagedir_config)

# first argument is a directory
else:
    if not Path(args.detector).is_file():
        prompt = "MegaDetector not found, would you like to download? y/n: "
        if input(prompt).lower() == "y":
            models.download_model(models.MEGADETECTOR['MDV5a'], out_dir=home)

    if not Path(args.classifier).is_file():
        prompt = "Classifier not found, would you like to download Southwest_v3? y/n: "
        if input(prompt).lower() == "y":
            models.download_model(models.CLASSIFIER['SDZWA_Southwest_v3'], out_dir=home)

    if not Path(args.classlist).is_file():
        prompt = "Class list not found, would you like to download Southwest_v3? y/n: "
        if input(prompt).lower() == "y":
            models.download_model(models.CLASS_LIST['SDZWA_Southwest_v3'], out_dir=home)

    # Call the main function
    pipeline.from_paths(args.imagedir_config, args.detector, args.classifier, args.classlist,
                        sort=args.sort, visualize=args.visualize, detect_only=args.detect_only)

print(f"Pipeline took {time.time() - start_time:.2f} seconds")
