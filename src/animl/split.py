import pandas as pd


def get_animals(manifest):
    """
    Pulls MD animal detections for classification

    Args:
        - manifest: DataFrame containing one row for ever MD detection

    Returns:
        subset of manifest containing only animal detections
    """
    if not isinstance(manifest, pd.DataFrame):
        raise AssertionError("'manifest' must be DataFrame.")
    return manifest[manifest['category'].astype(int) == 1].reset_index(drop=True)


def get_empty(manifest):
    """
    Pulls MD non-animal detections

    Args:
        - manifest: DataFrame containing one row for ever MD detection

    Returns:
        subset of manifest containing empty, vehicle and human detections
        with added prediction and confidence columns
    """
    if not isinstance(manifest, pd.DataFrame):
        raise AssertionError("'manifest' must be DataFrame.")

    # Removes all images that MegaDetector gave no detection for
    otherdf = manifest[manifest['category'].astype(int) != 1].reset_index(drop=True)
    print(otherdf)

    # Numbers the class of the non-animals correctly
    if not otherdf.empty:
        otherdf['prediction'] = otherdf['category'].astype(int)
        otherdf['prediction'] = otherdf['prediction'].replace(2, "human")
        otherdf['prediction'] = otherdf['prediction'].replace(3, "vehicle")
        otherdf['prediction'] = otherdf['prediction'].replace(0, "empty")
        otherdf['confidence'] = otherdf['conf']

    else:
        # return blank dataframe
        otherdf = pd.DataFrame(columns=manifest.columns.values)

    return otherdf
