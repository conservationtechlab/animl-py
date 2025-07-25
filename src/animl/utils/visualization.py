"""
Plot Bounding Boxes and Save Images

Functionality to draw bounding boxes and labels provided image DataFrame.

@ Kyra Swanson 2023
"""
import cv2
import argparse
import pandas as pd
import os
import numpy as np
from typing import Union, Optional

from animl.utils import general


def plot_box(img, row, prediction=False):
    height, width, _ = img.shape
    bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
    xyxy = general.convert_minxywh_to_absxyxy(bbox, width, height)

    thick = int((height + width) // 900)
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (90, 255, 0), thick)

    # Printing prediction if enabled
    if prediction:
        label = row['prediction']
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
        text_size_width, text_size_height = text_size

        box_right = (xyxy[2] if (xyxy[2] - xyxy[0]) < (text_size_width * 3)
                    else xyxy[0] + (text_size_width * 3))
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (box_right, xyxy[1] - (text_size_height * 2)),
                    (90, 255, 0), -1)

        cv2.putText(img, label, (xyxy[0], xyxy[1] - 12), 0, 1e-3 * height,
                    (0, 0, 0), thick // 3)


def plot_all_bounding_boxes(manifest: pd.DataFrame,
                            out_dir: str,
                            file_col: str,
                            min_conf: Union[int, float] = 0,
                            prediction: bool = False):
    """
    This function takes the data frame output from MegaDetector, makes a copy of each image,
    plots the boxes in the new image, and saves it the specified directory.

    Args:
        data_frame (Pandas DataFrame): Output of Mega Detector
        output_dir (str): Name of the output directory
        file_col (str): Column name containing file paths
        min_conf (Optional) (Int or Float): Confidence threshold to plot the box
        prediction (Optional) (Boolean): Should the prediction be printed alongside bounding box

    Raises:
    - Exception: If 'data_frame' is not a pandas DataFrame
    - Exception: If 'min_conf' is not a number between [0,1]
    - Exception: If 'prediction' is not a boolean
    """
    # If the specified output directory does not exist, make it
    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # iterate through unique file paths
    manifest_filepaths = manifest.groupby(file_col)
    for filepath, detections in manifest_filepaths:
        # ouput name
        file_name_no_ext, file_ext = os.path.splitext(os.path.split(filepath)[1])

        # If the file is not an image, do each frame separately
        if file_ext.lower() not in ['.jpg', '.jpeg', '.png']:
             # Plotting individual boxes in an image
            for i, row in detections.iterrows():
                print(row)
                img = cv2.imread(row['Frame'])
                # Skipping the box if the confidence threshold is not met
                if (row['max_detection_conf']) < min_conf:
                    continue

                # If any of the box isn't defined, jump to next one
                if np.isnan(row['bbox_x']):
                    continue

                # Calculations required for plotting
                plot_box(img, row, prediction=prediction)

                # Saving the image
                new_file_name = f"{file_name_no_ext}_box_{i}.jpg"
                new_file_path = os.path.join(out_dir, new_file_name)
                cv2.imwrite(new_file_path, img)

            cv2.destroyAllWindows()
            
        else:
            img = cv2.imread(filepath)

            # Plotting individual boxes in an image
            for i, row in detections.iterrows():

                # Skipping the box if the confidence threshold is not met
                if (row['max_detection_conf']) < min_conf:
                    continue

                # If any of the box isn't defined, jump to next one
                if np.isnan(row['bbox_x']):
                    continue

                plot_box(img, row, prediction=prediction)

            # Saving the image
            new_file_name = f"{file_name_no_ext}_box_{i}{file_ext}"
            new_file_path = os.path.join(out_dir, new_file_name)
            cv2.imwrite(new_file_path, img)

        cv2.destroyAllWindows()

def draw_bounding_boxes(row: pd.Series,
                        out_file: Optional[str] = None,
                        prediction: bool = False):
    """
    Draws bounding boxes and labels on image DataFrame.

    Args:
        row : DataFrame containing image data - coordinates and predictions.
            The DataFrame should have the following columns:
            - 'Frame': Filename or path to the image file.
            - 'bbox_x': Normalized x-coordinate of the top-left corner.
            - 'bbox_y': Normalized y-coordinate of the top-left corner.
            - 'bbox_w': Normalized width of the bounding box (range: 0-1).
            - 'bbox_h': Normalized height of the bounding box (range: 0-1).
            - 'prediction': Object prediction label for the bounding box.
        box_number (int): Number used for generating the output image filename.
        image_output_path (str): Output directory to saved images.
        prediction (bool): if true, add prediction label
    """
    img = cv2.imread(row["Frame"])
    
    plot_box(img, row, prediction=prediction)

    if out_file is not None:
        cv2.imwrite(out_file, img)

    cv2.destroyAllWindows()


def demo_boxes(manifest: pd.DataFrame, file_col: str, min_conf: float = 0.9, prediction: bool = True):
    """
    Draws bounding boxes and labels on image DataFrame.

    Args:
        manifest : DataFrame containing image data - coordinates and predictions.
            The DataFrame should have the following columns:
            - file_col: Filename or path to the image file.
            - 'bbox_x': Normalized x-coordinate of the top-left corner.
            - 'bbox_y': Normalized y-coordinate of the top-left corner.
            - 'bbox_w': Normalized width of the bounding box (range: 0-1).
            - 'bbox_h': Normalized height of the bounding box (range: 0-1).
            - 'prediction': Object prediction label for the bounding box.
        file_col (str): column containing file paths
        min_conf (float): minimum confidence threshold to plot box
        prediction (bool): if true, add prediction label
    """
    images = manifest[file_col].unique()

    for image_path in images:
        # display the image, wait for key
        img = cv2.imread(image_path)  # Use row directly without indexing
        cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
        cv2.imshow('Display', img)
        cv2.waitKey(0)

        boxes = manifest[manifest[file_col] == image_path]
        print(boxes)
        for _, row in boxes.iterrows():
            confidence = row["confidence"]
            if confidence >= min_conf:
                height, width, _ = img.shape
                left = int(row['bbox_x'] * width)
                top = int(row['bbox_y'] * height)
                right = int((row['bbox_x'] + row['bbox_w']) * width)
                bottom = int((row['bbox_y'] + row['bbox_h']) * height)
                label = row['prediction']
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
                text_size_width, text_size_height = text_size
                thick = int((height + width) // 900)
                box_right = (right if (right - left) < (text_size_width * 3)
                             else left + (text_size_width * 3))

                cv2.rectangle(img, (left, top), (right, bottom), (90, 255, 0), thick)

                if prediction:
                    cv2.rectangle(img, (left, top),
                                  (box_right, top - (text_size_height * 3)),
                                  (90, 255, 0), -1)
                    cv2.putText(img, label, (left, top - 12), 0, 1e-3 * height,
                                (0, 0, 0), thick // 3)
                cv2.imshow('Display', img)
                cv2.waitKey(0)

            else:
                continue

    cv2.destroyAllWindows()


def main(csv_file: str, output_dir: str):
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
