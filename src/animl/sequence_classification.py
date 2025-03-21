"""
@ Ayush Singh 2024
"""

import pandas as pd
import numpy as np
from typing import List, Optional

# TODO: REALIGN WITH R VERSION


def sequence_classification(animals: pd.DataFrame,
                            empty: pd.DataFrame,
                            predictions: np.ndarray,
                            classes: List[str],
                            station_col: str,
                            empty_class: str="empty",
                            sort_columns: List[str]=None,
                            file_col: str="FilePath",
                            maxdiff: float = 60):

    """
    This function applies image classifications at a sequence level by leveraging information from
    multiple images. A sequence is defined as all images at the same camera and station where the
    time between consecutive images is <=maxdiff. This can improve classification accuracy, but
    assumes that only one species is present in each sequence. If you regularly expect multiple
    species to occur in an image or sequence don't use this function.

    Parameters:
    - animals (Pandas DataFrame): Sub-selection of all images that contain animals
    - sort_columns (List of Strings): Defines sorting order for the DataFrame
    - predictions (Numpy Array of Numpy Arrays): Logits of all entries in "animals"
    - species (CSV File): File mapping index to species
    - station_col (String): The name of the station column
    - empty (Optional) (Pandas DataFrame): Sub-selection of all images that do not contain animals
    - maxdiff (float) (Optional): Maximum time difference between any two images in a sequence

    Raises:
    - Exception: If 'animals' is not a pandas DataFrame
    - Exception: If 'sort_columns' is not a list or is empty
    - Exception: If 'station_col' is not a string or is empty
    - Exception: If 'empty' is defined and is not a pandas DataFrame
    - Exception: If maxdiff is defined and is not a positive number

    Output:
    - final_df (Pandas DataFrame): Sequence classified data from both animals and empty
    """

    # Sanity check to verify that animals is a Pandas DataFrame
    if not isinstance(animals, pd.DataFrame):
        raise Exception("'animals' must be a DataFrame")

    # Sanity check to verify that sort_columns is a non-empty list
    if not isinstance(sort_columns, list) or len(sort_columns) == 0:
        raise Exception("'sort_columns' must be a non-empty list of strings")

    if not isinstance(station_col, str) or station_col == '':
        raise Exception("'station_col' must be a non-empty string")

    # Sanity check to verify that empty is a Pandas DataFrame, if defined
    if empty is not None and not isinstance(animals, pd.DataFrame):
        raise Exception("'empty' must be a DataFrame")

    # Sanity check to verify that maxdiff is a positive number
    if not isinstance(maxdiff, (int, float)) or maxdiff < 0:
        raise Exception("'maxdiff' must be a number >= 0")


    # DataFrame to hold the merged data from animals and empty
    if empty is not None:
        empty["ID"] = range(0, empty.shape[0])
        predempty = empty.pivot(index="ID", columns="prediction", values="confidence")

        # Replace NaN with 0
        predempty = predempty.fillna(0)

        # Example 'predictions' shape assumption (adjust based on actual dimensions)
        zero_matrix = np.zeros((empty.shape[0], len(classes)))

        # Add zero matrix to the left (like cbind in R)
        predempty = np.hstack(zero_matrix, predempty)

        # apply temporary predictions to animals





    # Converting FileModifyDate from a String to a datetime object
    merged_df['FileModifyDate'] = pd.to_datetime(merged_df['FileModifyDate'], format="%Y-%m-%d %H:%M:%S")

    # Sorting the merged data in the order specified
    merged_df = merged_df.sort_values(by=sort_columns)

    # DataFrame to store the final result after sequence classification
    final_df = pd.DataFrame(columns=merged_df.columns)

    # List to store all rows which are a part of the current sequence
    curr_sequence = []
    curr_sequence_logits = []

    # Iterating through all entries, one at a time
    for index, row in merged_df.iterrows():
        # If the current sequence is empty, initialize it with current row
        if len(curr_sequence) == 0:
            curr_sequence.append(row)
            curr_sequence_logits.append(row['logits'])

        # Check if current row is a part of the current sequence
        elif (row[station_col] == curr_sequence[0][station_col]) and (row["FileModifyDate"] - curr_sequence[0]["FileModifyDate"]).total_seconds() <= maxdiff:
            curr_sequence.append(row)
            curr_sequence_logits.append(row['logits'])

        # All rows for the current sequence have been collected
        else:
            matrix = np.stack(curr_sequence_logits, axis=0)

            # Getting the max prediction from each row, used in case when all entries are empty in sequence
            max_each_row = np.argmax(matrix, axis=1)

            # Checking if the entire sequence is from empty df
            all_from_empty_df = all(element['df'] != 'animals' for element in curr_sequence)

            # Checking if all entries of the sequence are classified as empty
            all_empty_col = not any(idx + 1 != empty_col for idx in max_each_row)

            # If the entire sequence is from empty dataFrame or all entries are empty, prediction becomes empty
            if (all_from_empty_df or all_empty_col):
                highest_confidence_label = 'empty'

            # Otherwise, pick the highest non-empty prediction
            else:
                # Summing the logits column-wise
                col_sums = np.sum(matrix, axis=0)

                # Getting the sorted order of indices (increasing)
                col_sums = np.argsort(col_sums)

                # Assigning the prediction label
                if (col_sums[-1] + 1) != empty_col:
                    highest_confidence_label = classes[col_sums[-1] + 1]
                else:
                    highest_confidence_label = classes[col_sums[-2] + 1]

            # Adding the sequence to final_df
            for element in curr_sequence:
                element['prediction'] = highest_confidence_label
                final_df = pd.concat([final_df, pd.DataFrame([element])])

            # Re-initializing the current sequence
            curr_sequence = [row]
            curr_sequence_logits = [row['logits']]

    # Handeling the last batch -------------------------------------------------
    matrix = np.stack(curr_sequence_logits, axis=0)

    # Getting the max prediction from each row, used in case when all entries are empty in sequence
    max_each_row = np.argmax(matrix, axis=1)

    # Checking if the entire sequence is from empty df
    all_from_empty_df = all(element['df'] != 'animals' for element in curr_sequence)

    # Checking if all entries of the sequence are classified as empty
    all_empty_col = not any(idx + 1 != empty_col for idx in max_each_row)

    # If the entire sequence is from empty dataFrame or all entries are empty, prediction becomes empty
    if (all_from_empty_df or all_empty_col):
        highest_confidence_label = 'empty'

    # Otherwise, pick the highest non-empty prediction
    else:
        # Summing the logits column-wise
        col_sums = np.sum(matrix, axis=0)

        # Getting the sorted order of indices (increasing)
        col_sums = np.argsort(col_sums)

        # Assigning the prediction label
        if (col_sums[-1] + 1) != empty_col:
            highest_confidence_label = classes.loc[classes['id'] == col_sums[-1] + 1, 'species'].iloc[0]
        else:
            highest_confidence_label = classes.loc[classes['id'] == col_sums[-2] + 1, 'species'].iloc[0]

    # --------------------------------------------------------------------------
    # Adding the sequence to final_df
    for element in curr_sequence:
        element['prediction'] = highest_confidence_label
        final_df = pd.concat([final_df, pd.DataFrame([element])])

    animals = animals.drop(['df', 'logits'], axis=1)

    if empty is not None:
        empty = empty.drop(['df', 'logits'], axis=1)

    final_df = final_df.drop(['df', 'logits'], axis=1)

    return final_df
