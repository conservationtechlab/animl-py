"""
Classifier Module

This module provides functions and classes for loading and saving clasifier models.

"""
import os
import glob
import torch.nn as nn
import torch
from torchvision.models import efficientnet


def save_model(dest_folder, model, epoch, stats):
    '''
        Saves model state weights

        Args
            - dest_folder: directory in which to save the model
            - model: model object
            - epoch: current epoch
            - stats: performance metrics for the given epoch
    '''
    # make sure save directory exists; create if not
    os.makedirs(dest_folder, exist_ok=True)

    # get model parameters and add to stats
    stats['model'] = model.state_dict()

    torch.save(stats, open(f'{dest_folder}/{epoch}.pt', 'wb'))


def load_model(path, num_classes, architecture="CTL", overwrite=False):
    '''
        Creates a model instance and loads the latest model state weights.

        Args
            - path: directory or model file path
            - num_classes: number of classes expected by the model
            - architecture: pytorch requires a model architecture to assign weights
            - overwrite: if the given path contains model file(s), overwrite

        Returns
            - model_instance: model object of the given architecture
            - start_epoch: starting epoch number
    '''
    if architecture == "CTL" or architecture == "efficientnet_v2_m":
        model_instance = EfficientNet(num_classes, tune=False)
    else:
        raise AssertionError('Please provide the correct model')

    # load latest model state from given folder
    if os.path.isdir(path):

        model_states = glob.glob(path + '*.pt')

        if len(model_states) and not overwrite:
            # at least one save state found; get latest
            model_epochs = [int(m.replace(path, '').replace('.pt', '')) for m in model_states]
            start_epoch = max(model_epochs)

            # load state dict and apply weights to model
            print(f'Resuming from epoch {start_epoch}')
            state = torch.load(open(f'{path}/{start_epoch}.pt', 'rb'))
            model_instance.load_state_dict(state['model'])

        else:
            print('Overwrite enabled, starting new model')
            start_epoch = 0

    # load from file path
    elif os.path.isfile(path):
        print(f'Loading file {path}')
        state = torch.load(open(path, 'rb'))
        model_instance.load_state_dict(state['model'])
        start_epoch = 0

    # no save state found; start anew
    else:
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch


class EfficientNet(nn.Module):
    '''
        The CTL uses EfficientNet v2 m as the base architecture for its classifier models.
        The class definition to initialize the model is here.
    '''
    def __init__(self, num_classes, tune=True):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
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
            Forward pass
        '''
        x = self.model.features(x)    # features.size(): [B x 3 x W x H]

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        prediction = self.model.classifier(x)  # prediction.size(): [B x num_classes]

        return prediction
