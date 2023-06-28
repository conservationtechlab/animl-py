"""
Module: run_plotboxes

This module provides functionality to read a CSV file and perform box plotting on the images
based on the data in the CSV file.

Usage:
    python run_plotboxes.py csv_file output_dir

Arguments:
    csv_file (str): Path to the CSV file containing image data and coordinates.
    output_dir (str): Directory to save the boxed images.

Example:
    python run_plotboxes.py images.csv boxed_images/

"""

import argparse
import pandas as pd
from plot_boxes import draw_bounding_boxes


def main(csv_file, output_dir):
    """
    Read a CSV file values and perform box plotting on the images.

    Args:
        csv_file (str): Path to the CSV file.
        output_dir (str): Saved location  of boxed images output dir.

    Returns:
        None
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Perform box plotting for each image in the CSV file
    for i, row in data.iterrows():
        draw_bounding_boxes(row, 60 + i, output_dir + '/')


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Plot boxes images-csv')

    # Add the CSV file and output directory arguments
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('output_dir', type=str, help='Path to the output dir')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function
    main(args.csv_file, args.output_dir)
