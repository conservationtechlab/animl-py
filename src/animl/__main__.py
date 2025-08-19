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
import os

from animl import pipeline
import animl.models.download as models

# Start timer
start_time = time.time()

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
    pipeline.from_config(args.imagedir_config)

# first argument is a directory
else:
    if not os.path.isfile(args.detector):
        prompt = "MegaDetector not found, would you like to download? y/n: "
        if input(prompt).lower() == "y":
            models.download_model(models.MEGADETECTOR['MDV5a'], out_dir=home)

    if not os.path.isfile(args.classifier):
        prompt = "Classifier not found, would you like to download Southwest_v3? y/n: "
        if input(prompt).lower() == "y":
            models.download_model(models.CLASSIFIER['SDZWA_Southwest_v3'], out_dir=home)

    if not os.path.isfile(args.classlist):
        prompt = "Class list not found, would you like to download Southwest_v3? y/n: "
        if input(prompt).lower() == "y":
            models.download_model(models.CLASS_LIST['SDZWA_Southwest_v3'], out_dir=home)

    # Call the main function
    pipeline.from_paths(args.imagedir_config, args.detector, args.classifier, args.classlist)

print(f"Pipeline took {time.time() - start_time:.2f} seconds")
