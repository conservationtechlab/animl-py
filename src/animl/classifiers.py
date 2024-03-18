'''
    Tools for Saving, Loading, and Building Species Classifiers

    @ Kyra Swanson 2023
'''
import os
import glob
import pandas as pd
import torch
import torch.nn as nn
from time import time
from torchvision.models import efficientnet
from tensorflow import keras


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


def load_model(model_path, class_file, device="cpu", architecture="CTL", overwrite=False):
    '''
    Creates a model instance and loads the latest model state weights.

    Args:
        - model_path (str): file or directory path to model weights
        - class_file (str): path to associated class list
        - device (str): specify to run on cpu or gpu
        - architecture (str): expected model architecture
        - overwrite (bool): overwrite existing model files within path if true

    Returns:
        - model: model object of given architecture with loaded weights
        - classes: associated species class list
        - start_epoch (int): current epoch, 0 if not resuming training
    '''
    # read class file
    classes = pd.read_csv(class_file)

    # check to make sure GPU is available if chosen
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        device = 'cpu'
    else:
        print('Device set to', device)

    # load latest model state from given folder
    if os.path.isdir(model_path):
        start_epoch = 0
        if (architecture == "CTL") or (architecture == "efficientnet_v2_m"):
            model = EfficientNet(len(classes))
        else:  # can only resume CTL models from a directory at this time
            raise AssertionError('Please provide the correct model')

        model_states = glob.glob(model_path + '*.pt')

        if len(model_states) and not overwrite:
            # at least one save state found; get latest
            model_epochs = [int(m.replace(model_path, '').replace('.pt', '')) for m in model_states]
            start_epoch = max(model_epochs)

            # load state dict and apply weights to model
            print(f'Resuming from epoch {start_epoch}')
            state = torch.load(open(f'{model_path}/{start_epoch}.pt', 'rb'))
            model.load_state_dict(state['model'])
        else:
            # no save state found/overwrite; start anew
            print('Model found but overwrite enabled, starting new model')

        return model, classes, start_epoch

    # load a specific model file
    elif os.path.isfile(model_path):
        print(f'Loading model at {model_path}')
        start_time = time()
        # TensorFlow
        if model_path.endswith('.h5'):
            model = keras.models.load_model(model_path)
        # PyTorch dict
        elif model_path.endswith('.pt'):
            model = EfficientNet(len(classes), tune=False)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model'])
            model.to(device)
            model.eval()
        # PyTorch full model
        elif model_path.endswith('.pth'):
            model = torch.load(model_path)
            model.to(device)
            model.eval()
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
            Construct the model architecture.
        '''
        super(EfficientNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.model = efficientnet.efficientnet_v2_m(weights=efficientnet.EfficientNet_V2_M_Weights)       # "pretrained": use weights pre-trained on ImageNet
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
