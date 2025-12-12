"""
Plot Bounding Boxes and Save Images

Functionality to draw bounding boxes and labels provided image DataFrame.

@ Kyra Swanson 2023
"""
import cv2
import argparse
import pandas as pd
import math
import numpy as np
from pathlib import Path
from typing import Union

from animl.utils import general
from animl.file_management import IMAGE_EXTENSIONS
from animl.video_processing import get_frame_as_image



MD_COLORS = {"1": (0, 255, 0), "2": (0, 0, 255),  "3": (255, 0, 0)}
MD_LABELS = {"1": "animal", "2": "human",  "3": "vehicle"}

def plot_box(rows,
             file_col: str = "filepath",
             min_conf: Union[int, float] = 0,
             label_col=None,
             show_confidence=False,
             colors = MD_COLORS,
             detector_labels = MD_LABELS,
             return_img: bool = False):
    """
    Plot a bounding box on a given (loaded) image

    Args:
        rows (pandas.DataFrame): Row from the DataFrame containing bounding box coordinates and prediction.
            Expected columns:
            - file_col
            - 'conf': Confidence score of the detection.
            - 'bbox_x': x-coordinate of the top-left corner of the bounding box.
            - 'bbox_y': y-coordinate of the top-left corner of the bounding box.
            - 'bbox_w': width of the bounding box.
            - 'bbox_h': height of the bounding box.
            - 'prediction': Prediction label to be displayed alongside the bounding box (optional).
        file_col (str): filepath column name in the DataFrame
        min_conf (int or float): Minimum confidence threshold to plot the box
        label_col (str or None): Column name containing class to print above the box. If None, no label is printed.
        show_confidence (bool): If true, show confidence score above the box.
        colors (dict): Dictionary mapping class labels to BGR color tuples for the bounding boxes.
        detector_labels (dict): Dictionary mapping detector categories to human-readable labels.
        return_img (bool): If true, return the image array with boxes overlaid, otherwise display it using cv2.imshow.

    Returns:
        None
    """
    # If a single row is passed, convert it to a DataFrame for consistency
    if isinstance(rows, pd.Series):
        rows = pd.DataFrame([rows])
        
    if not {file_col, 'conf', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'}.issubset(rows.columns):
        raise ValueError(f"DataFrame must contain {file_col}, 'conf', 'bbox_x', 'bbox_y', 'bbox_w', and 'bbox_h' columns.")
    
    if colors is None:
        colors = MD_COLORS
    if detector_labels is None:
        detector_labels = MD_LABELS

    # Load the image
    path = rows.iloc[0][file_col]
    if not Path(path).is_file():
        raise FileNotFoundError(f"The file {path} does not exist.")

    if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
        img = cv2.imread(rows.iloc[0][file_col])
    else:
        frame = rows.iloc[0]['frame'] if 'frame' in rows.columns else 0
        img = get_frame_as_image(rows.iloc[0][file_col], frame)
    
    height, width, _ = img.shape

    font_scale = min(width, height) * 1e-3
    thickness = math.ceil(min(width, height) *  1e-3)

    for _, row in rows.iterrows():
        # Skipping the box if the confidence threshold is not met
        if (row['conf']) < min_conf:
            continue

        # If any of the box isn't defined, jump to next one
        if np.isnan(row['bbox_x']):
            continue

        bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
        xyxy = general.convert_minxywh_to_absxyxy(bbox, width, height)

        color = colors[str(row['category'])]
        thick = int((height + width) // 900)
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thick)

        # Printing prediction if enabled
        if label_col:
            if label_col == "category":
                label = detector_labels[str(int(row['category']))]
            else:
                label = row[label_col]

            if show_confidence:
                if 'confidence' in row and not np.isnan(row['confidence']):
                    label += f" {row['confidence']:.2f}"
                elif 'conf' in row and not np.isnan(row['conf']):
                    label += f" {row['conf']:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
            # move label if it is out of the image
            label_top = max(0, xyxy[1] - text_height - baseline)
            label_bottom = xyxy[1] if label_top > 0 else xyxy[1] + text_height + baseline

            label_right = min(width, xyxy[0] + text_width)
            label_left = xyxy[0] if label_right < width else xyxy[0] - text_width

            img = cv2.rectangle(img, (label_left, label_top), (label_right, label_bottom), color, -1)

            # adjust text if label is at top of image
            text_y_pos = label_bottom - 5 if label_top > 0 else label_bottom
            cv2.putText(img, label, (label_left, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)

    if return_img:
        return img
    else:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def plot_all_bounding_boxes(manifest: pd.DataFrame,
                            out_dir: str,
                            file_col: str = 'filepath',
                            min_conf: Union[int, float] = 0.1,
                            label_col = False,
                            show_confidence: bool = False,
                            colors = MD_COLORS,
                            detector_labels = MD_LABELS):
    """
    This function takes the parsed dataframe output from MegaDetector, makes a copy of each image,
    plots the boxes in the new image, and saves it the specified directory.

    Args:
        manifest (Pandas DataFrame): manifest of detections
        out_dir (str): Name of the output directory
        file_col (str): Column name containing file paths
        min_conf (Optional) (Int or Float): Confidence threshold to plot the box
        label_col (Optional) (str): Column name containing label to print on box
        show_confidence (Optional) (bool): If true, show confidence score on box
        colors (Optional) (dict): Dictionary mapping class labels to BGR color tuples for the bounding boxes.
        detector_labels (Optional) (dict): Dictionary mapping detector categories to human-readable labels.

    Returns:
        None
    """
    if not {file_col}.issubset(manifest.columns):
        raise ValueError(f"DataFrame must contain '{file_col}' column.")
    
    if colors is None:
        colors = MD_COLORS
    if detector_labels is None:
        detector_labels = MD_LABELS

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

            img = plot_box(detections, file_col=file_col, min_conf=min_conf, 
                           label_col=label_col, show_confidence=show_confidence,
                           colors=colors, detector_labels=detector_labels, return_img=True)

                # Saving the image
            new_file_path = Path(out_dir) / f"{file_name_no_ext}_box.jpg"
            cv2.imwrite(new_file_path, img)
            cv2.destroyAllWindows()
            
        # file is a video, break up by frames
        else:
            if not {'frame'}.issubset(manifest.columns):
                raise ValueError("DataFrame must contain 'frame' column for video files.")
            
            frames = detections.groupby('frame')
            for f, frame_detections in frames:

                img = plot_box(frame_detections, file_col=file_col, min_conf=min_conf,
                               label_col=label_col, show_confidence=show_confidence,
                               colors=colors, detector_labels=detector_labels, return_img=True)

                # Saving the image
                new_file_path = Path(out_dir) / f"{file_name_no_ext}_{f}_box.jpg"
                cv2.imwrite(new_file_path, img)
                cv2.destroyAllWindows()


def plot_from_file(csv_file: str, out_dir: str, file_col: str = 'filepath'):
    """
    Read a CSV manifest file and perform box plotting on the images.

    Args:
        csv_file (str): Path to the CSV file.
        out_dir (str): Saved location  of boxed images output dir.
        file_col (str): Column name containing file paths.

    Returns:
        None
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Perform box plotting for each image in the CSV file
    for i, row in data.iterrows():
        img = plot_box(row, return_img=True)
        # Save the image with boxes
        file_name_no_ext = Path(row[file_col]).stem
        file_ext = Path(row[file_col]).suffix
        if file_ext.lower() not in IMAGE_EXTENSIONS:
            file_ext = '.jpg'
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
