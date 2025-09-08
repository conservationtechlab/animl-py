"""
Plot Bounding Boxes and Save Images

Functionality to draw bounding boxes and labels provided image DataFrame.

@ Kyra Swanson 2023
"""
import cv2
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Union

from animl.utils import general
from animl.file_management import IMAGE_EXTENSIONS


def plot_box(rows, file_col="FilePath", min_conf: Union[int, float] = 0, prediction=False):
    """
    Plot a bounding box on a given (loaded) image

    Args:
        img (numpy.ndarray): Loaded image in which the bounding box will be plotted.
        row (pandas.Series): Row from the DataFrame containing bounding box coordinates and prediction.
            Expected columns:
            - file_col
            - 'conf': Confidence score of the detection.
            - 'bbox_x': x-coordinate of the top-left corner of the bounding box.
            - 'bbox_y': y-coordinate of the top-left corner of the bounding box.
            - 'bbox_w': width of the bounding box.
            - 'bbox_h': height of the bounding box.
            - 'prediction': Prediction label to be displayed alongside the bounding box (optional).
        prediction (bool): If True, display the prediction label alongside the bounding box.

    Returns:
        None
    """
    # If a single row is passed, convert it to a DataFrame for consistency
    if isinstance(rows, pd.Series):
        rows = pd.DataFrame([rows])

    img = cv2.imread(rows.iloc[0][file_col])
    height, width, _ = img.shape

    if not {file_col, 'conf', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'}.issubset(rows.columns):
        raise ValueError(f"DataFrame must contain {file_col}, 'conf', 'bbox_x', 'bbox_y', 'bbox_w', and 'bbox_h' columns.")

    for _, row in rows.iterrows():
        # Skipping the box if the confidence threshold is not met
        if (row['conf']) < min_conf:
            continue

        # If any of the box isn't defined, jump to next one
        if np.isnan(row['bbox_x']):
            continue

        bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
        xyxy = general.convert_minxywh_to_absxyxy(bbox, width, height)

        thick = int((height + width) // 900)
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (90, 255, 0), thick)

        # Printing prediction if enabled
        if prediction:
            if not {'prediction'}.issubset(rows.columns):
                raise ValueError("DataFrame must contain 'prediction' column to display labels.")
            label = row['prediction']
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
            text_size_width, text_size_height = text_size

            box_right = (xyxy[2] if (xyxy[2] - xyxy[0]) < (text_size_width * 3)
                        else xyxy[0] + (text_size_width * 3))
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (box_right, xyxy[1] - (text_size_height * 2)),
                        (90, 255, 0), -1)

            cv2.putText(img, label, (xyxy[0], xyxy[1] - 12), 0, 1e-3 * height,
                        (0, 0, 0), thick // 3)
    return img


def plot_all_bounding_boxes(manifest: pd.DataFrame,
                            out_dir: str,
                            file_col: str = 'frame',
                            min_conf: Union[int, float] = 0.1,
                            prediction: bool = False):
    """
    This function takes the parsed dataframe output from MegaDetector, makes a copy of each image,
    plots the boxes in the new image, and saves it the specified directory.

    Args:
        manifest (Pandas DataFrame): manifest of detections
        out_dir (str): Name of the output directory
        file_col (str): Column name containing file paths
        min_conf (Optional) (Int or Float): Confidence threshold to plot the box
        prediction (Optional) (Boolean): Should the prediction be printed alongside bounding box

    Returns:
        None
    """
    if not {file_col}.issubset(manifest.columns):
        raise ValueError(f"DataFrame must contain '{file_col}' column.")

    # If the specified output directory does not exist, make it
    Path(out_dir).mkdir(exist_ok=True)

    # iterate through unique file paths
    manifest_filepaths = manifest.groupby(file_col)
    for filepath, detections in manifest_filepaths:
        # ouput name
        file_name_no_ext = Path(filepath).stem
        file_ext = Path(filepath).suffix

        # file is an image
        if file_ext.lower() in IMAGE_EXTENSIONS:

            img = plot_box(detections, file_col=file_col, min_conf=min_conf, prediction=prediction)

                # Saving the image
            new_file_path = Path(out_dir) / f"{file_name_no_ext}_box.jpg"
            cv2.imwrite(new_file_path, img)
            cv2.destroyAllWindows()
            
        # file is a video, break up by frames
        else:
            frames = detections.groupby('frame')
            for f, frame_detections in frames:

                img = plot_box(frame_detections, file_col="frame", min_conf=min_conf, prediction=prediction)

                # Saving the image
                new_file_path = Path(out_dir) / f"{file_name_no_ext}_box.jpg"
                cv2.imwrite(new_file_path, img)
                cv2.destroyAllWindows()


def plot_from_file(csv_file: str, out_dir: str):
    """
    Read a CSV manifest file and perform box plotting on the images.

    Args:
        csv_file (str): Path to the CSV file.
        out_dir (str): Saved location  of boxed images output dir.

    Returns:
        None
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Perform box plotting for each image in the CSV file
    for i, row in data.iterrows():
        img = plot_box(row)
        # Save the image with boxes
        file_name_no_ext = Path(row['FilePath']).stem
        file_ext = Path(row['FilePath']).suffix
        new_file_path = Path(out_dir, f"{file_name_no_ext}_{i}_{file_ext}")
        cv2.imwrite(new_file_path, img)


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Plot boxes images-csv')

    # Add the CSV file and output directory arguments
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('out_dir', type=str, help='Path to the output dir')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function'
    plot_from_file(args.csv_file, args.out_dir)
