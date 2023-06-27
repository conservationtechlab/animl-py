import argparse
import pandas as pd
from plot_boxes import drawBoundingBoxes

def main(csv_file, output_dir):
    """
    Read a CSV file values and perform box plotting on the images.

    Args:
        csv_file (str): Path to the CSV file.
        output_dir (str): Path to the output directory where the modified images will be saved.

    Returns:
        None
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Perform box plotting for each image in the CSV file
    for i, row in data.iterrows():
        drawBoundingBoxes(row, 60 + i, output_dir + '/')


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Draw bounding boxes on images based on a CSV file')

    # Add the CSV file and output directory arguments
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function
    main(args.csv_file, args.output_dir)

