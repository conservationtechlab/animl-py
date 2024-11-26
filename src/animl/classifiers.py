'''
    Tools for Saving, Loading, and Building Species Classifiers

    @ Kyra Swanson 2023
'''
import os
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from time import time
from torchvision.models import efficientnet, convnext_base, ConvNeXt_Base_Weights
# import tensorflow.keras
# import onnx
import torch.onnx
import onnxruntime


def save_model(out_dir, epoch, model, stats):
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
    os.makedirs(out_dir, exist_ok=True)

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
        - start_epoch (int): current epoch, 0 if not resuming training
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
    if os.path.isdir(model_path):
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
    elif os.path.isfile(model_path):
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

class EfficientNet(nn.Module):

    def __init__(self, num_classes, tune=True):
        '''
            Construct the EfficientNet model architecture.
        '''
        super(EfficientNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.model = efficientnet.efficientnet_v2_m(weights=efficientnet.EfficientNet_V2_M_Weights.DEFAULT)       # "pretrained": use weights pre-trained on ImageNet
        if tune:
            for params in self.model.parameters():
                params.requires_grad = True

        num_ftrs = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)

    def forward(self, x):
        '''
            Forward pass (prediction)
        '''
        # x.size(): [B x 3 x W x H]
        x = self.model.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        prediction = self.model.classifier(x)  # prediction.size(): [B x num_classes]

        return prediction
    
class ConvNeXtBase(nn.Module):
    def __init__(self, num_classes, tune=True):
        '''
        Construct the ConvNeXt-Base model architecture.
        '''
        super(ConvNeXtBase, self).__init__()

        self.model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT) # load the ConvNeXt-Base model pre-trained on ImageNet 1K
        if not tune:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the last classifier layer
        num_ftrs = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features=num_ftrs, out_features=num_classes)

    def forward(self, x):
        '''
        Forward pass (prediction).
        '''
        return self.model(x)
