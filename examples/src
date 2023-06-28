import cv2
import sys
import pandas as pd

def drawBoundingBoxes(row, n, imageOutputPath):
    """
    Draws bounding boxes and labels on an image based on the provided image DataFrame.

    Args:
        image (DataFrame): DataFrame containing image information, including bounding box coordinates and predictions.
            The DataFrame should have the following columns:
            - 'Frame': Filename or path to the image file.
            - 'bbox1': Normalized x-coordinate of the top-left corner of the bounding box (range: 0-1).
            - 'bbox2': Normalized y-coordinate of the top-left corner of the bounding box (range: 0-1).
            - 'bbox3': Normalized width of the bounding box (range: 0-1).
            - 'bbox4': Normalized height of the bounding box (range: 0-1).
            - 'prediction': Object prediction label for the bounding box.

        n (int): Number used for generating the output image filename.

        imageOutputPath (str): Path to the directory where the output images will be saved.

    Returns:
        None
    """
    '''im = cv2.imread(image.iloc[0]["Frame"])
    h, w, _ = im.shape
    left = int(image['bbox1'] * w)
    top = int(image['bbox2'] * h)
    right = int((image['bbox1'] + image['bbox3']) * w)
    bottom = int((image['bbox2'] + image['bbox4']) * h)
    label = image.iloc[0]['prediction']'''
    im = cv2.imread(row["Frame"])  # Use row directly without indexing
    h, w, _ = im.shape
    left = int(row['bbox1'] * w)
    top = int(row['bbox2'] * h)
    right = int((row['bbox1'] + row['bbox3']) * w)
    bottom = int((row['bbox2'] + row['bbox4']) * h)
    label = row['prediction']
    textSize, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
    textSizeWidth, textSizeHeight = textSize
    
   # print( textSizeWidth, textSizeHeight)

    thick = int((h + w) // 900)
    
    boxright = right if (right - left) < (textSizeWidth*3) else left + (textSizeWidth*3)
    cv2.rectangle(im,(left, top), (right, bottom), (90,255,0), thick)
    
    cv2.rectangle(im, (left, top), (boxright, top - (textSizeHeight*2)), (90,255,0), -1)
    
    cv2.putText(im, label, (left, top - 12), 0, 1e-3 * h, (0,0,0), thick//3)    
        
    filename = imageOutputPath + str(n) + ".jpg"
    print(filename)
    cv2.imwrite(filename, im)


