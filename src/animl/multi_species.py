"""
@ Ayush Singh 2024
"""
import pandas as pd


def multi_species_detection(animals: pd.DataFrame, threshold: float, file_col: str = "FilePath") -> pd.DataFrame:
    """
    This function applies image classifications at a image level. All images which have multiple
    species present with confidence above threshold, will be returned as a DataFrame

    Parameters:
    - animals (Pandas DataFrame): Sub-selection of all images that contain animals
    - threshold (float): Minimum confidence for the image to be considered

    Raises:
    - Exception: If 'animals' is not a pandas DataFrame
    - Exception: If threshold is not a float or threshold < 0 or threshold > 1

    Output:
    - result_df (Pandas DataFrame): Rows from images having more than one species
    """

    # Sanity check to verify that animals is a Pandas DataFrame
    if not isinstance(animals, pd.DataFrame):
        raise Exception("'animals' must be a DataFrame")

    # Sanity check to verify that threshold is a float and is in range [0,1]
    if (not isinstance(threshold, float)) or (threshold < 0) or (threshold > 1):
        raise Exception("Threshold must be a value between 0 and 1, both inclusive")

    # Sorting by file name to accumulate all rows belonging to the same image
    animals[file_col] = animals[file_col].astype(str)
    animals = animals.sort_values(by=file_col)

    # Initializing data frame to store the result
    result_df = pd.DataFrame()

    # Making a new column for count
    result_df['count'] = []

    # List to store all the rows which have the same file name
    curr_picture = []

    # Iterating through all rows and gathering the ones which belong to the same image
    for index, row in animals.iterrows():
        # Initializing the list when it's empty
        if len(curr_picture) == 0:
            curr_picture.append(row)

        # Check if row belongs to the same image
        elif row[file_col] == curr_picture[0][file_col]:
            curr_picture.append(row)

        # All rows for the current image have been collected
        else:
            # Key is the specie, value is list [row, count]
            dic = {}

            # For all images above threshold, save the one with highest confidence for each class
            for element in curr_picture:
                if element['confidence'] > threshold:
                    if element['prediction'] in dic:
                        dic[element['prediction']][1] += 1
                        if dic[element['prediction']][0]['confidence'] < element['confidence']:
                            dic[element['prediction']][0] = element
                    else:
                        dic[element['prediction']] = [element, 1]

            # Make current image a part of output only if it has more than 1 species
            if len(dic) > 1:
                for key in dic:
                    dic[key][0]['count'] = dic[key][1]
                    result_df = pd.concat([result_df, pd.DataFrame([dic[key][0]])])

            # Reset list for next image
            curr_picture = [row]

    # Handeling the last batch
    dic = {}
    for element in curr_picture:
        if element['confidence'] > threshold and element['prediction'] != 'empty':
            if element['prediction'] in dic:
                dic[element['prediction']][1] += 1
                if dic[element['prediction']][0]['confidence'] < element['confidence']:
                    dic[element['prediction']][0] = element
            else:
                dic[element['prediction']] = [element, 1]

    # Make current image a part of output only if it has more than 1 species
    if len(dic) > 1:
        for key in dic:
            dic[key][0]['count'] = dic[key][1]
            result_df = pd.concat([result_df, pd.DataFrame([dic[key][0]])])

    return result_df
