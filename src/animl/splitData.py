import pandas as pd

def getAnimals(manifest):
    if not isinstance(manifest, pd.DataFrame):
        raise AssertionError("'manifest' must be DataFrame.")
    return manifest[manifest['category'].astype(int) == 1].reset_index(drop=True)


def getEmpty(manifest):
    if not isinstance(manifest, pd.DataFrame):
        raise AssertionError("'manifest' must be DataFrame.")
        
    # Removes all images that MegaDetector gave no detection for
    otherdf = manifest[manifest['category'].astype(int) != 1]
    otherdf = otherdf.reset_index(drop=True)
    otherdf['prediction'] = otherdf['category']
    
    # Numbers the class of the non-animals correctly
    if not otherdf.empty:
        otherdf['prediction'] = otherdf['prediction'].replace(2,"human")
        otherdf['prediction'] = otherdf['prediction'].replace(3,"vehicle")
        otherdf['prediction'] = otherdf['prediction'].replace(0,"empty")
        otherdf['confidence'] = otherdf['max_detection_conf']
        
    else:
        otherdf = pd.DataFrame(columns = manifest.columns.values)
                                       
    return otherdf