"""
Module: animl.plot_boxes
Functionality to draw bounding boxes and labels provided image DataFrame.
"""

import cv2


def draw_bounding_boxes(row, box_number, image_output_path, prediction):
    # pylint: disable=R0914
    """
    Draws bounding boxes and labels on image DataFrame.
    Args:
        image : DataFrame containing image data - coordinates and predictions.
            The DataFrame should have the following columns:
            - 'Frame': Filename or path to the image file.
            - 'bbox1': Normalized x-coordinate of the top-left corner.
            - 'bbox2': Normalized y-coordinate of the top-left corner.
            - 'bbox3': Normalized width of the bounding box (range: 0-1).
            - 'bbox4': Normalized height of the bounding box (range: 0-1).
            - 'prediction': Object prediction label for the bounding box.
        n (int): Number used for generating the output image filename.
        imageOutputPath (str): Output directory to saved images.
    Returns:
        None
    """
    img = cv2.imread(row["Frame"])  # Use row directly without indexing
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
    cv2.rectangle(img, (left, top),
                  (box_right, top - (text_size_height * 2)),
                  (90, 255, 0), -1)
    if prediction:
        cv2.putText(img, label, (left, top - 12), 0, 1e-3 * height,
                    (0, 0, 0), thick // 3)
        
    filename = image_output_path + str(box_number) + ".jpg"
    print(filename)
    cv2.imwrite(filename, img)

    
def demo_boxes(manifest, min_conf = 0.9, prediction = True):
    # pylint: disable=R0914
    """
    Draws bounding boxes and labels on image DataFrame.
    Args:
        image : DataFrame containing image data - coordinates and predictions.
            The DataFrame should have the following columns:
            - 'Frame': Filename or path to the image file.
            - 'bbox1': Normalized x-coordinate of the top-left corner.
            - 'bbox2': Normalized y-coordinate of the top-left corner.
            - 'bbox3': Normalized width of the bounding box (range: 0-1).
            - 'bbox4': Normalized height of the bounding box (range: 0-1).
            - 'prediction': Object prediction label for the bounding box.
        n (int): Number used for generating the output image filename.
        imageOutputPath (str): Output directory to saved images.
    Returns:
        None
    """
    images = manifest["FilePath"].unique()
    
    for image_path in images:
        #display the image, wait for key
        img = cv2.imread(image_path)  # Use row directly without indexing
        cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
        cv2.imshow('Display', img)
        cv2.waitKey(0)
        
        boxes = manifest[manifest["FilePath"] == image_path]
        print(boxes)
        for _,row in boxes.iterrows():
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