'''
    Tools for Saving, Loading, and Building Species Classifiers

    @ Kyra Swanson 2023
'''
import argparse
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from time import time
from tqdm import tqdm

import torch
import torch.onnx
import onnxruntime

from animl import generator, file_management, split
from animl.models.species import EfficientNet, ConvNeXtBase
from animl.utils.torch_utils import get_device


def softmax(x: np.ndarray) -> np.ndarray:
    '''
    Helper function to softmax
    '''
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)


def save_model(out_dir: str, epoch: int, model: torch.nn.Module, stats: dict):
    '''
    Saves model state weights.

    Args:
        - out_dir (str): directory to save model to
        - epoch (int): current training epoch
        - model: pytorch model
        - stats (dict): performance metrics of current epoch

    Returns:
        None
    '''
    if not isinstance(out_dir, str):
        raise TypeError(f"Expected string for out_dir, got {type(out_dir)}")
    if not isinstance(epoch, int):
        raise TypeError(f"Expected int for epoch, got {type(epoch)}")
    if not isinstance(stats, dict):
        raise TypeError(f"Expected dict for stats, got {type(dict)}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # get model parameters and add to stats
    stats['model'] = model.state_dict()

    torch.save(stats, open(f'{out_dir}/{epoch}.pt', 'wb'))


def load_model(model_path, class_file, device=None, architecture="CTL"):
    '''
    Creates a model instance and loads the latest model state weights.

    Args:
        - model_path (str): file or directory path to model weights
        - class_file (str): path to associated class list
        - device (str): specify to run on cpu or gpu
        - architecture (str): expected model architecture

    Returns:
        - model: model object of given architecture with loaded weights
        - classes: associated species class list
        - start_epoch (int, optional): current epoch, 0 if not resuming training
    '''
    # read class file
    model_path = Path(model_path)
    classes = pd.read_csv(Path(class_file))

    # check to make sure GPU is available if chosen
    if not torch.cuda.is_available():
        device = 'cpu'
    elif torch.cuda.is_available() and device is None:
        device = 'cuda:0'
    else:
        device = device

    print('Device set to', device)

    # load latest model state from given folder
    if model_path.is_dir():
        model_path = str(model_path)
        start_epoch = 0
        if (architecture == "CTL") or (architecture == "efficientnet_v2_m"):
            model = EfficientNet(len(classes))
        elif architecture == "convnext_base":
            model = ConvNeXtBase(len(classes))
        else:  # can only resume models from a directory at this time
            raise AssertionError('Please provide the correct model')

        model_states = []
        for file in os.listdir(model_path):
            if os.path.splitext(file)[1] == ".pt":
                model_states.append(file)

        if len(model_states):
            # at least one save state found; get latest
            model_epochs = [int(m.replace(model_path, '').replace('.pt', '')) for m in model_states]
            start_epoch = max(model_epochs)

            # load state dict and apply weights to model
            print(f'Resuming from epoch {start_epoch}')
            state = torch.load(open(f'{model_path}/{start_epoch}.pt', 'rb'))
            model.load_state_dict(state['model'])
        else:
            # no save state found; start anew
            print('No model state found, starting new model')

        return model, classes, start_epoch

    # load a specific model file
    elif model_path.is_file():
        print(f'Loading model at {model_path}')
        start_time = time()
        # TensorFlow
        # if model_path.endswith('.h5'):
        #    model = keras.models.load_model(model_path)
        # PyTorch dict
        if model_path.suffix == '.pt':
            if (architecture == "CTL") or (architecture == "efficientnet_v2_m"):
                model = EfficientNet(len(classes), tune=False)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
                model.to(device)
                model.eval()
                model.framework = "EfficientNet"
            elif architecture == "convnext_base":
                model = ConvNeXtBase(len(classes), tune=False)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
                model.to(device)
                model.eval()
                model.framework = "ConvNeXt-Base"
        # PyTorch full model
        elif model_path.suffix == '.pth':
            model = torch.load(model_path, map_location=device)
            model.to(device)
            model.eval()
            model.framework = "pytorch"
        elif model_path.suffix == '.onnx':
            if device == "cpu":
                model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            else:
                model = onnxruntime.InferenceSession(model_path, providers=["CUDAExecutionProvider", 'CPUExecutionProvider'])
            model.framework = "onnx"
        else:
            raise ValueError('Unrecognized model format: {}'.format(model_path))
        elapsed = time() - start_time
        print('Loaded model in %.2f seconds' % elapsed)

        # no need to return epoch
        return model, classes

    # no dir or file found
    else:
        raise ValueError("Model not found at given path")


def tensor_to_onnx(tensor, channel_last=True):
    '''
    Helper function for onnx, shifts dims to BxHxWxC
    '''
    if channel_last:
        tensor = tensor.permute(0, 2, 3, 1)  # reorder BxCxHxW to BxHxWxC

    tensor = tensor.numpy()

    return tensor


def predict_species(detections, model, classes, device='cpu', out_file=None, raw=False,
                    file_col='Frame', crop=True, resize_width=299, resize_height=299,
                    normalize=True, batch_size=1, workers=1):
    """
    Predict species using classifier model

    Args
        - detections (pd.DataFrame): dataframe of (animal) detections
        - model: preloaded classifier model
        - classes: preloaded class list
        - device (str): specify to run model on cpu or gpu, default to cpu
        - out_file (str): path to save prediction results to
        - raw (bool): return raw logits instead of applying labels
        - file_col (str): column name containing file paths
        - crop (bool): use bbox to crop images before feeding into model
        - resize_width (int): image width input size
        - resize_height (int): image height input size
        - normalize (bool): normalize the tensor before inference
        - batch_size (int): data generator batch size
        - workers (int): number of cores

    Returns
        - detections (pd.DataFrame): MD detections with classifier prediction and confidence
    """
    # Typechecking
    if not isinstance(detections, pd.DataFrame):
        raise TypeError(f"Expected pd.Dataframe for detecionts, got {type(detections)}")
    if not isinstance(classes, pd.DataFrame):
        raise TypeError(f"Expected pd.Dataframe for classes, got {type(classes)}")

    if file_management.check_file(out_file):
        return file_management.load_data(out_file)

    if isinstance(detections, pd.DataFrame):
        # initialize lists
        detections = detections.reset_index(drop=True)
        predictions = []
        probabilities = []
        if raw:
            raw_output = []

        dataset = generator.manifest_dataloader(detections, file_col=file_col, crop=crop,
                                                resize_width=resize_width, resize_height=resize_height,
                                                normalize=normalize, batch_size=batch_size, workers=workers)

        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataset)):
                # pytorch
                if model.framework == "pytorch" or model.framework == "EfficientNet":
                    data = batch[0]
                    data = data.to(device)
                    output = model(data)
                    if raw:
                        raw_output.extend(torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy())

                    labels = torch.argmax(output, dim=1).cpu().detach().numpy()
                    pred = classes['Code'].values[labels]
                    predictions.extend(pred)

                    probs = torch.max(torch.nn.functional.softmax(output, dim=1), 1)[0]
                    probabilities.extend(probs.cpu().detach().numpy())

                # onnx
                elif model.framework == "onnx":
                    data = batch[0]
                    data = tensor_to_onnx(data)
                    output = model.run(None, {model.get_inputs()[0].name: data})[0]
                    if raw:
                        raw_output.extend(softmax(output))

                    labels = np.argmax(output, axis=1)
                    pred = classes['Code'].values[labels]
                    predictions.extend(pred)

                    onnx_probs = np.max(softmax(output), axis=1)
                    probabilities.extend(onnx_probs)

                else:
                    raise AssertionError("Model architechture not supported.")

            detections['prediction'] = predictions
            detections['confidence'] = probabilities

    else:
        raise AssertionError("Input must be a data frame of crops or vector of file names.")

    if out_file:
        file_management.save_data(detections, out_file)

    if raw:
        return np.vstack(raw_output)
    else:
        return detections


def classify_with_config(config):
    """
    Run Classification from Config File

    Args:
        - config (str): path to config file

    Returns:
        predictions dataframe
    """
    # get config file
    print(f'Using config "{config}"')
    cfg = yaml.safe_load(open(config, 'r'))

    manifest = pd.read_csv(cfg['manifest'])

    # get available device
    device = get_device()

    classifier, class_list = load_model(cfg['classifier_file'], cfg['class_list'],
                                        device=device, architecture=cfg.get('class_list', "CTL"))

    if cfg.get('split_animals', True):
        manifest = split.get_animals(manifest)

    predictions = predict_species(manifest, classifier, class_list, device=device, out_file=cfg['out_file'],
                                  raw=cfg.get('raw', False), file_col=cfg['file_col'], crop=cfg['crop'],
                                  resize_width=cfg['resize_width'], resize_height=cfg['resize_height'],
                                  normalize=cfg.get('normalize', True), batch_size=cfg.get('batch_size', 1),
                                  workers=cfg.get('workers', 1))
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='exp_resnet18.yaml')
    args = parser.parse_args()
    classify_with_config(args.config)
