from tensorflow.keras.models import load_model
from math import ceil
from os.path import isfile
from pandas import DataFrame
import imageCropGenerator

def load_clasifier(model_file):
    if not isfile(model_file):
        raise AssertionError("The given model file does not exist.")
        
    start_time = time.time()
    if model_file.endswith('.h5'):
        model = load_model(model_file)
   # elif model_file.endswith('.pt'):
   #     from detection.pytorch_detector import PTDetector
   #     detector = PTDetector(model_file, force_cpu, USE_MODEL_NATIVE_CLASSES)        
    else:
        raise ValueError('Unrecognized model format: {}'.format(model_file))
    elapsed = time.time() - start_time
    print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))
    return model
        

def predictSpecies(detections, model, resize = 456, standardize = False, batch = 1, workers = 1):
  

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
     
