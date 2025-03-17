"""
Module: animl.plot_boxes
Functionality to draw bounding boxes and labels provided image DataFrame.
"""
import cv2
import argparse
import pandas as pd
import os
import numpy as np
from typing import Union, Optional


def plot_all_bounding_boxes(data_frame: pd.DataFrame,
                            output_dir: str,
                            file_col: str,
                            min_conf: Union[int, float] = 0,
                            prediction: bool = False):
    """
    This function takes the data frame output from MegaDetector, makes a copy of each image,
    plots the boxes in the new image, and saves it the specified directory.

    Args:
        - data_frame (Pandas DataFrame): Output of Mega Detector
        - output_dir (String): Name of the output directory
        - file_col (str): Column name containing file paths
        - min_conf (Optional) (Int or Float): Confidence threshold to plot the box
        - prediction (Optional) (Boolean): Should the prediction be printed alongside bounding box

    Raises:
    - Exception: If 'data_frame' is not a pandas DataFrame
    - Exception: If 'min_conf' is not a number between [0,1]
    - Exception: If 'prediction' is not a boolean

    Returns:
    - None
    """

    # Sanity check to verify that data_frame is a Pandas DataFrame
    if not isinstance(data_frame, pd.DataFrame):
        raise Exception("'data_frame' must be a DataFrame")

    # If the specified output directory does not exist, make it
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Sanity check to verify that maxdiff is a positive number
    if not isinstance(min_conf, (int, float)) or min_conf < 0 or min_conf > 1:
        raise Exception("'min_conf' must be a number between [0,1]")

    # Sanity check to verify prediction is a boolean
    if not isinstance(prediction, bool):
        raise Exception("'prediction' must be a boolean value")

    # Sorting by file path to accumulate all rows belonging to the same image
    data_frame[file_col] = data_frame[file_col].astype(str)
    data_frame = data_frame.sort_values(by=file_col)

    # List to store all the rows which have the same file name
    curr_picture = []

    # Iterating through all rows and gathering the ones which belong to the same image
    for index, row in data_frame.iterrows():
        # Initializing the list when it's empty
        if len(curr_picture) == 0:
            curr_picture.append(row)

        # Check if row belongs to the same image
        elif row[file_col] == curr_picture[0][file_col]:
            curr_picture.append(row)

        # All rows for the current image have been collected
        else:
            if not os.path.exists(curr_picture[0][file_col]):
                continue
            # Loading the image
            img = cv2.imread(curr_picture[0][file_col])

            # Getting the output destination
            file_path = curr_picture[0][file_col]
            dir_path, file_name = os.path.split(file_path)
            file_name_no_ext, file_ext = os.path.splitext(file_name)
            new_file_name = f"{file_name_no_ext}_box{file_ext}"
            write_dir = output_dir
            new_file_path = os.path.join(write_dir, new_file_name)

            # If the file is not an image, skipping it
            if file_ext.lower() not in ['.jpg', '.jpeg', '.png']:
                curr_picture = [row]
                continue

            # Plotting individual boxes in an image
            for i in curr_picture:
                # Skipping the box if the confidence threshold is not met
                if (i['max_detection_conf']) < min_conf:
                    continue

                # If any of the box isn't defined, jump to next one
                if np.isnan(i['bbox1']).any() or np.isnan(i['bbox2']).any() or np.isnan(i['bbox3']).any() or np.isnan(i['bbox4']).any():
                    continue

                # Calculations required for plotting
                height, width, _ = img.shape
                left = int(i['bbox1'] * width)
                top = int(i['bbox2'] * height)
                right = int((i['bbox1'] + i['bbox3']) * width)
                bottom = int((i['bbox2'] + i['bbox4']) * height)
                thick = int((height + width) // 900)

                # Plotting the box
                cv2.rectangle(img, (left, top), (right, bottom), (90, 255, 0), thick)

                # Printing prediction if enabled
                if prediction:
                    label = i['prediction']
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
                    text_size_width, text_size_height = text_size

                    box_right = (right if (right - left) < (text_size_width * 3) else left + (text_size_width * 3))
                    cv2.rectangle(img, (left, top), (box_right, top - (text_size_height * 2)), (90, 255, 0), -1)
                    cv2.putText(img, label, (left, top - 12), 0, 1e-3 * height, (0, 0, 0), thick // 3)

            # Saving the image
            cv2.imwrite(new_file_path, img)

            # Reset list for next image
            curr_picture = [row]

    if not os.path.exists(curr_picture[0][file_col]):
        return

    # Loading the image
    img = cv2.imread(curr_picture[0][file_col])

    # Getting the output destination
    file_path = curr_picture[0][file_col]
    dir_path, file_name = os.path.split(file_path)
    file_name_no_ext, file_ext = os.path.splitext(file_name)
    new_file_name = f"{file_name_no_ext}_box{file_ext}"
    write_dir = output_dir
    new_file_path = os.path.join(write_dir, new_file_name)

    # If the file is not an image, skipping it
    if file_ext.lower() in ['.jpg', '.jpeg', '.png']:

        # Plotting individual boxes in an image
        for i in curr_picture:
            # Skipping the box if the confidence threshold is not met
            if (i['max_detection_conf']) < min_conf:
                continue

            # If any of the box isn't defined, jump to next one
            if np.isnan(i['bbox1']).any() or np.isnan(i['bbox2']).any() or np.isnan(i['bbox3']).any() or np.isnan(i['bbox4']).any():
                continue

            # Calculations required for plotting
            height, width, _ = img.shape
            left = int(i['bbox1'] * width)
            top = int(i['bbox2'] * height)
            right = int((i['bbox1'] + i['bbox3']) * width)
            bottom = int((i['bbox2'] + i['bbox4']) * height)
            thick = int((height + width) // 900)

            # Plotting the box
            cv2.rectangle(img, (left, top), (right, bottom), (90, 255, 0), thick)

            # Printing prediction if enabled
            if prediction:
                label = i['prediction']
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
                text_size_width, text_size_height = text_size

                box_right = (right if (right - left) < (text_size_width * 3) else left + (text_size_width * 3))
                cv2.rectangle(img, (left, top), (box_right, top - (text_size_height * 2)), (90, 255, 0), -1)
                cv2.putText(img, label, (left, top - 12), 0, 1e-3 * height, (0, 0, 0), thick // 3)

        # Saving the image
        cv2.imwrite(new_file_path, img)


def draw_bounding_boxes(row: pd.Series,
                        box_number: int,
                        image_output_path: Optional[str] = None,
                        prediction: bool = False):
    """
    Draws bounding boxes and labels on image DataFrame.
    Args:
        - row : DataFrame containing image data - coordinates and predictions.
            The DataFrame should have the following columns:
            - 'Frame': Filename or path to the image file.
            - 'bbox1': Normalized x-coordinate of the top-left corner.
            - 'bbox2': Normalized y-coordinate of the top-left corner.
            - 'bbox3': Normalized width of the bounding box (range: 0-1).
            - 'bbox4': Normalized height of the bounding box (range: 0-1).
            - 'prediction': Object prediction label for the bounding box.
        - box_number (int): Number used for generating the output image filename.
        - image_output_path (str): Output directory to saved images.
        - prediction (bool): if true, add prediction label

    Returns:
        None
    """
    img = cv2.imread(row["Frame"])  # Use row directly without indexing
    height, width, _ = img.shape
    left = int(row['bbox1'] * width)
    top = int(row['bbox2'] * height)
    right = int((row['bbox1'] + row['bbox3']) * width)
    bottom = int((row['bbox2'] + row['bbox4']) * height)
    thick = int((height + width) // 900)
    cv2.rectangle(img, (left, top), (right, bottom), (90, 255, 0), thick)

    if prediction:
        label = row['prediction']
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
        text_size_width, text_size_height = text_size

        box_right = (right if (right - left) < (text_size_width * 3)
                     else left + (text_size_width * 3))
        cv2.rectangle(img, (left, top), (box_right, top - (text_size_height * 2)),
                      (90, 255, 0), -1)

        cv2.putText(img, label, (left, top - 12), 0, 1e-3 * height,
                    (0, 0, 0), thick // 3)

    if image_output_path is not None:
        filename = image_output_path + str(box_number) + ".jpg"
        print(filename)
        cv2.imwrite(filename, img)


def demo_boxes(manifest: pd.DataFrame, file_col: str, min_conf: float = 0.9, prediction: bool = True):
    """
    Draws bounding boxes and labels on image DataFrame.

    Args:
        - manifest : DataFrame containing image data - coordinates and predictions.
            The DataFrame should have the following columns:
            - 'Frame': Filename or path to the image file.
            - 'bbox1': Normalized x-coordinate of the top-left corner.
            - 'bbox2': Normalized y-coordinate of the top-left corner.
            - 'bbox3': Normalized width of the bounding box (range: 0-1).
            - 'bbox4': Normalized height of the bounding box (range: 0-1).
            - 'prediction': Object prediction label for the bounding box.
        - file_col (str): column containing file paths
        - min_conf (float): minimum confidence threshold to plot box
        - prediction (bool): if true, add prediction label

    Returns:
        None
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
                left = int(row['bbox1'] * width)
                top = int(row['bbox2'] * height)
                right = int((row['bbox1'] + row['bbox3']) * width)
                bottom = int((row['bbox2'] + row['bbox4']) * height)
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
