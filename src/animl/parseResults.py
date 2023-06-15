import pandas as pd
from os.path import isfile
import fileManagement
import numpy as np


def parseMD(results, manifest = None, out_file = None):
    if fileManagement.check_file(out_file):
        return fileManagement.load_data(out_file)
    
    if not isinstance(results, list):
        raise AssertionError("MD results input must be list")
    
    if len(results) == 0:
        raise AssertionError("'results' contains no detections")
        
    df = pd.DataFrame()
    for dictionary in results:
        detections = dictionary['detections']
        for detection in detections:
            bbox = detection['bbox']
            data = {'file': dictionary['file'], 'max_detection_conf': dictionary['max_detection_conf'],
                    'category': detection['category'], 'conf': detection['conf'], 'bbox1': bbox[0],
                    'bbox2': bbox[1],
                    'bbox3': bbox[2], 'bbox4': bbox[3]}
            df = df.append(data, ignore_index=True)
       
    # adjust boxes with 2% buffer from image edge 
    df.loc[df["bbox1"] > 0.98, "bbox1"] = 0.98
    df.loc[df["bbox2"] > 0.98, "bbox2"] = 0.98
    df.loc[df["bbox3"] > 0.98, "bbox3"] = 0.98
    df.loc[df["bbox4"] > 0.98, "bbox4"] = 0.98
    
    df.loc[df["bbox1"] < 0.02, "bbox1"] = 0.02
    df.loc[df["bbox2"] < 0.02, "bbox2"] = 0.02
    df.loc[df["bbox3"] < 0.02, "bbox3"] = 0.02
    df.loc[df["bbox4"] < 0.02, "bbox4"] = 0.02
    
    
    if isinstance(manifest, pd.DataFrame):
        df = manifest.merge(df, left_on="Frame", right_on="file")
        
    if out_file:
        fileManagement.save_data(df, out_file)
            
    return df
    
# def parseMDjson
def applyPredictions(animals, predictions, class_file, out_file = None, counts = False):
    if fileManagement.check_file(out_file): 
        return fileManagement.load_data(out_file)

    if not isinstance(animals, pd.DataFrame):
        raise AssertionError("'animals' must be DataFrame.")

    if not isfile(class_file):
        raise AssertionError("The given class file does not exist.")

    # Format Classification results
    table = pd.read_table(class_file, sep=" ", index_col=0)

    animals['prediction'] = [table['x'].values[int(np.argmax(x))] for x in predictions]
    animals['confidence'] = [np.max(x) for x in predictions] * animals["conf"]
    
    return animals