from tensorflow.keras.models import load_model
from keras.engine.functional import Functional
from os.path import isfile
from pandas import DataFrame
from time import time
from humanfriendly import format_timespan
from animl import generator


def load_classifier(model_file):
    """
    Load classifier model

    Args
        - model_file: path to classifier model

    Returns
        model: model object
    """
    if not isfile(model_file):
        raise AssertionError("The given model file does not exist.")

    start_time = time()
    if model_file.endswith('.h5'):
        model = load_model(model_file)
    elif model_file.endswith('.pt'):
        # TO DO
        raise ValueError('Pytorch models not currently supported.')
    else:
        raise ValueError('Unrecognized model format: {}'.format(model_file))
    elapsed = time() - start_time
    print('Loaded model in {}'.format(format_timespan(elapsed)))
    return model


def predict_species(detections, model, resize=456,
                    standardize=False, batch=1, workers=1):
    """
    Predict species using classifier model

    Args
        - detections: dataframe of animal detections
        - model: preloaded classifier model
        - resize: image input size
        - standardize:
        - batch: data generator batch size
        - workers: number of cores

    Returns
        matrix of model outputs
    """
    if isinstance(detections, DataFrame):
        filecol = "Frame" if "Frame" in detections.columns else "file"

        if any(detections.columns.isin(["bbox1"])):
            if type(model) == dict:  # pytorch
                dataset = generator.create_dataloader(detections, batch, workers, filecol)
            elif type(model) == Functional:  # tensorflow
                dataset = generator.TFGenerator(detections, filecol='file', resize=resize, batch=batch)
            else:
                raise AssertionError("Model architechture not supported.")
        else:
            raise AssertionError("Input must be a data frame of crops or vector of file names.")
    else:
        raise AssertionError("Input must be a data frame of crops or vector of file names.")

    return model.predict(dataset, workers=workers, verbose=1)
