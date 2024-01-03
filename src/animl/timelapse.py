'''
Functions for compatibility with Timelapse

'''
import pandas as pd
import os


def csv_converter(animals, empty, imagedir, only_animl=True):
    '''
    Converts the Pandas DataFrame created by running the animl classsifier to a csv file that contains columns needed for TimeLapse conversion in later step

    Credit: Sachin Gopal Wani

    Args:
        animals - a DataFrame that has entries of anuimal classification \
        empty - a DataFrame that has detection of non-animal objects in images \
        imagedir - location of root directory where all images are stored (can contain subdirectories) \
        only_animl - A bool that confirms whether we want only animal detctions or all (animal + non-animal detection from MegaDetector + classifier)

    Returns:
        animals.csv - A csv file containing all the detection and classification information for animal detections \
        non-anim.csv - A csv file containing detections of all non-animals made to be similar to animals.csv in columns \
        csv_loc - Location of the stored animals csv file
    '''
    if not imagedir.endswith("/"):
        imagedir += "/"

    # Create directory
    ICdir = os.path.join(imagedir, "Animl-Directory", "IC")
    os.makedirs(ICdir, exist_ok=True)

    expected_columns = ('FilePath', 'FileName', 'FileModifyDate', 'Frame', 'file',
                        'max_detection_conf', 'category', 'conf', 'bbox1', 'bbox2', 'bbox3',
                        'bbox4', 'prediction', 'confidence')

    for s in expected_columns:
        assert s in animals.columns, 'Expected column {} not found in animals DataFrame'.format(s)

    # Dropping unnecessary columns (Refer to columns numbers above for expected columns - 0 indexed).
    animals.drop(['FilePath', 'FileName', 'FileModifyDate', 'Frame', 'max_detection_conf'], axis=1, inplace=True)

    # Keep relative path only
    animals['file'] = animals['file'].apply(lambda x: x[len(imagedir):])
    # ALT: copy_ani['file'] = copy_ani['file'].str.slice(start=len(imagedir))

    # Rename column names for clarity
    animals.rename(columns={'conf': 'detection_conf', 'prediction': 'class', 'confidence': 'classification_conf'}, inplace=True)

    if only_animl:
        # Saving animal results to csv file for conversion to timelapse compatible json
        csv_loc = os.path.join(ICdir, "animals.csv")
        animals.to_csv(csv_loc, index=False)

        # Saving non-animal csv entries for manual perusal
        empty.to_csv(os.path.join(ICdir, "non-anim.csv"), index=False)

    else:
        # Checking if the columns match the expected DataFrame
        for s in expected_columns:
            assert s in empty.columns, 'Expected column {} not found in empty (non-animals) DataFrame'.format(s)

        # Doing the same process for non-animal results
        empty.drop(['FilePath', 'FileName', 'FileModifyDate', 'Frame', 'max_detection_conf'], axis=1, inplace=True)
        empty['file'] = empty['file'].apply(lambda x: x[len(imagedir):])
        empty.rename(columns={'conf': 'detection_conf', 'prediction': 'class'}, inplace=True)

        # Adding prediction as person and human
        empty['class'].replace({'0': 'empty', '2': 'person', '3': 'vehicle'}, inplace=True)

        # Changing classification conf = detection_conf instead of max_detection_conf
        empty['classification_conf'] = empty.loc[:, 'detection_conf']

        # Combining DataFrames and saving it to csv file for further use
        csv_loc = os.path.join(ICdir, "manifest.csv")
        manifest = pd.concat([animals, empty])
        manifest.to_csv(csv_loc, index=False)

    # Return the location of csv for json conversion
    return csv_loc
