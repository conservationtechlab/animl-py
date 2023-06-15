from tensorflow.keras.models import load_model
from math import ceil
from os.path import isfile
from pandas import DataFrame
import imageCropGenerator

def predictSpecies(detections, model, resize = 456, standardize = False, batch = 1, workers = 1):
  
    if not isfile(model):
        raise AssertionError("The given model file does not exist.")

    model = load_model(model)
        
    if isinstance(detections, DataFrame):
        steps = ceil(len(detections) / batch)
        print(steps)

        filecol = "Frame" if "Frame" in detections.columns else "file"

        if any(detections.columns.isin(["bbox1"])):

            dataset = imageCropGenerator.GenerateCropsFromFile(detections, filecol = filecol,
                                                               resize = resize,
                                                               standardize = standardize, batch = batch)
        else:
            raise AssertionError("Input must be a data frame of crops or vector of file names.")
  
    else:
        raise AssertionError("Input must be a data frame of crops or vector of file names.")
  
    return model.predict(dataset, workers = workers, verbose = 1)
     
