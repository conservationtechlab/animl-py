import pandas as pd
import numpy as np
from datetime import datetime

""" 
@ Ayush Singh 2024

This function applies image classifications at a sequence level by leveraging information from 
multiple images. A sequence is defined as all images at the same camera and station where the
time between consecutive images is <=maxdiff. This can improve classification accuracy, but 
assumes that only one species is present in each sequence. If you regularly expect multiple 
species to occur in an image or sequence don't use this function.

Parameters:
- animals (Pandas DataFrame): Sub-selection of all images that contain animals
- sortcolumns (List of Strings): Defines sorting order for the DataFrame
- empty (Optional) (Pandas DataFrame): Sub-selection of all images that do not contain animals
- maxdiff (float) (Optional): Maximum time difference between any two images in a sequence

Raises:
- Exception: If 'animals' is not a pandas DataFrame
- Exception: If 'sortcolumns' is not a list or is empty
- Exception: If 'empty' is defined and is not a pandas DataFrame 
- Exception: If maxdiff is defined and is not a positive number

Output:
- Merged DataFrame: If return_flag is 0
- Animals and Empty DataFrame: If return_flag is 1
"""

def sequenceClassification(animals, sortcolumns, empty=None, maxdiff=60):
   
    # Sanity check to verify that animals is a Pandas DataFrame
    if not isinstance(animals, pd.DataFrame):
        raise Exception("'animals' must be a DataFrame")
    
    # Sanity check to verify that sortcolumns is a non-empty list
    if not isinstance(sortcolumns, list) or len(sortcolumns) == 0:
        raise Exception("'sortcolumns' must be a non-empty list of strings")
    
    # Sanity check to verify that empty is a Pandas DataFrame, if defined
    if empty is not None and not isinstance(animals, pd.DataFrame):
        raise Exception("'empty' must be a DataFrame")

    # Sanity check to verify that maxdiff is a positive number
    if not isinstance(maxdiff, (int, float)) or maxdiff < 0:
        raise Exception("'maxdiff' must be a number >= 0")

    # Column to indicate the origin DataFrame of a row, used for distinction between animal and empty rows
    animals['df'] = "animals"

    # DataFrame to hold the merged data from animals and empty
    if empty is not None:
        empty['df'] = "empty"
        empty['confidence'] = empty['confidence'].astype('float32') 
        merged_df = pd.concat([animals, empty])
    else:
        merged_df = animals
    
    # Converting FileModifyDate from a String to a datetime object
    merged_df['FileModifyDate'] = pd.to_datetime(merged_df['FileModifyDate'], format="%Y-%m-%d %H:%M:%S")

    # Sorting the merged data in the order specified
    merged_df = merged_df.sort_values(by=sortcolumns)

    # DataFrame to store the final result after sequence classification
    final_df = pd.DataFrame(columns=merged_df.columns)

    # List to store all rows which are a part of the current sequence
    curr_sequence = []

    # Iterating through all entries, one at a time
    for index, row in merged_df.iterrows():
        # If the current sequence is empty, initialize it with current row 
        if len(curr_sequence) == 0:
            curr_sequence.append(row)

        # Check if current row is a part of the current sequence
        elif (row['Station'] == curr_sequence[0]['Station']) and (row["FileModifyDate"] - curr_sequence[0]["FileModifyDate"]).total_seconds() <= maxdiff:
            curr_sequence.append(row)

        # All rows for the current sequence have been collected
        else:
            print(len(curr_sequence))
            # Flag to handel the corner case of all entries belonging to the empty DataFrame
            flag = True

            # Variables to store the highest confidence seen and its corresponding label
            highest_confidence_value = -float('inf')
            highest_confidence_label = ""

            # Calculating the highest confidence and its label in the sequence, excluding entries from empty DataFrame
            for element in curr_sequence:
                if element['confidence'] > highest_confidence_value and element['df'] != "empty":
                    flag = False
                    highest_confidence_value = element['confidence']
                    highest_confidence_label = element['prediction']
            
            # Corner case when the entire sequence belongs to empty DataFrame, calculating confidence and lable for the case
            if flag == True:
                for element in curr_sequence:
                    if element['confidence'] > highest_confidence_value: 
                        highest_confidence_value = element['confidence']
                        highest_confidence_label = element['prediction']
            
            # Setting the values of confidence and prediction to highest_confidence_value and highest_confidence_label for all elements in a sequence
            for element in curr_sequence:     
                element['confidence'] = highest_confidence_value
                element['prediction'] = highest_confidence_label
                final_df = pd.concat([final_df, pd.DataFrame([element])])   
                
            # Re-initializing the current sequence
            curr_sequence = [row]

    # Handeling the last batch missed by the while loop
    # Flag to handel the corner case of all entries belonging to the empty DataFrame
    flag = True

    # Variables to store the highest confidence seen and its corresponding label
    highest_confidence_value = -float('inf')
    highest_confidence_label = ""

    # Calculating the highest confidence and its label in the sequence, excluding entries from empty DataFrame
    for element in curr_sequence:
        if element['confidence'] > highest_confidence_value and element['df'] != "empty":
            flag = False
            highest_confidence_value = element['confidence']
            highest_confidence_label = element['prediction']
    
    # Corner case when the entire sequence belongs to empty DataFrame, calculating confidence and lable for the case
    if flag == True:
        for element in curr_sequence:
            if element['confidence'] > highest_confidence_value: 
                highest_confidence_value = element['confidence']
                highest_confidence_label = element['prediction']

    # Setting the values of confidence and prediction to highest_confidence_value and highest_confidence_label for all elements in a sequence
    for element in curr_sequence:     
        element['confidence'] = highest_confidence_value
        element['prediction'] = highest_confidence_label
        final_df = pd.concat([final_df, pd.DataFrame([element])])   

    # Returning the result, as per the flag specified
    final_df = final_df.drop(columns=['df'])
    return final_df
