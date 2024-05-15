"""
    Inference Module

    Provides functions for species classifier inference

    @ Kyra Swanson 2023
"""
import torch
import pandas as pd
import numpy as np
from tqdm import trange
from . import generator, file_management
from .classifiers import EfficientNet


def softmax(x):
    '''
    Helper function to softmax 
    '''
    return np.exp(x)/np.sum(np.exp(x),axis=1, keepdims=True)


def to_numpy(tensor):
    '''
    Helper function for onnx
    '''
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



def predict_species(detections, model, classes, device='cpu', out_file=None,
                    file_col='Frame', crop=True, resize=299, batch_size=1, workers=1, raw=False):
    """
    Predict species using classifier model

    Args
        - detections (pd.DataFrame): dataframe of (animal) detections
        - model: preloaded classifier model
        - classes: preloaded class list
        - device (str): specify to run model on cpu or gpu, default to cpu
        - out_file (str): path to save prediction results to
        - file_col (str): column name containing file paths
        - resize (int): image input size
        - batch_size (int): data generator batch size
        - workers (int): number of cores

    Returns
        - detections (pd.DataFrame): MD detections with classifier prediction and confidence
    """
    if file_management.check_file(out_file):
        return file_management.load_data(out_file)

    if isinstance(detections, pd.DataFrame):
        # tensorflow
        if model.framework == "tensorflow":
            
            dataset = generator.TFGenerator(detections, file_col=file_col, resize=resize, batch_size=batch_size)
            output = model.predict(dataset, workers=workers, verbose=1)

            detections['prediction'] = [classes['species'].values[int(np.argmax(x))] for x in output]
            detections['confidence'] = [np.max(x) for x in output]
        
        else:
            predictions = []
            probabilities = []
            if raw:
                raw_output = []

            dataset = generator.manifest_dataloader(detections, batch_size=batch_size, workers=workers, 
                                                    file_col=file_col, crop=crop, resize=resize)
            progressBar = trange(len(dataset))
            with torch.no_grad():
                for _, batch in enumerate(dataset):
                    
                    if model.framework == "pytorch" or model.framework == "EfficientNet":
                        data = batch[0]
                        data = data.to(device)
                        output = model(data)
                        if raw:
                            raw_output.extend(output.cpu().detach().numpy())

                        labels = torch.argmax(output, dim=1).cpu().detach().numpy()
                        pred = classes['species'].values[labels]
                        predictions.extend(pred)

                        probs = torch.max(torch.nn.functional.softmax(output, dim=1), 1)[0]
                        probabilities.extend(probs.cpu().detach().numpy())
                        progressBar.update(1)
                        
                    # onnx 
                    elif model.framework == "onnx":
                        data = batch[0]
                        data = np.swapaxes(data,1,3)
                        data = data.to(device)
                        
                        
                        ort_inputs = {model.get_inputs()[0].name: to_numpy(data)}
                        ort_outs = model.run(None, ort_inputs)[0]
                        if raw:
                            raw_output.extend(ort_outs)

                        
                        onnx_label = np.argmax(ort_outs, axis=1)
                        pred = [classes['species'].values[x] for x in onnx_label]
                        predictions.extend(pred)

                        #onnx_probs = np.max(softmax(ort_outs),axis=1)
                        onnx_probs = np.max(ort_outs,axis=1)
                        probabilities.extend(onnx_probs)
                        
                    else:
                       raise AssertionError("Model architechture not supported.")

                detections['prediction'] = predictions
                detections['confidence'] = probabilities
                progressBar.close()

    else:
        raise AssertionError("Input must be a data frame of crops or vector of file names.")

    if out_file:
        file_management.save_data(detections, out_file)

    if raw:
        return np.vstack(raw_output)
    else:
        return detections
