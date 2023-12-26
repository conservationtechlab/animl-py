'''
    Tools for Saving, Loading, and Building Species Classifiers

   @ Kyra Swanson 2023
'''
import os
import glob
import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models import efficientnet


def save_model(out_dir, epoch, model, stats):
    '''
        Saves model state weights.

        Args:
            - out_dir (str): directory to save model to
            - epoch (int): current training epoch
            - model: pytorch model
            - stats (dict): performance metrics of current epoch

        Returns
            None
    '''
    os.makedirs(out_dir, exist_ok=True)

    # get model parameters and add to stats
    stats['model'] = model.state_dict()

    torch.save(stats, open(f'{out_dir}/{epoch}.pt', 'wb'))


def load_model(path, architecture, num_classes, overwrite=False):
    '''
        Creates a model instance and loads the latest model state weights.

        Args:
            - path (str): file path to model weights
            - architecture (str): expected model architecture
            - num_classes (int): number of expected classes to set output layer to
            - overwrite (bool): overwrite existing model files within path if true

        Returns:
            - model_instance: model object of given architecture with loaded weights
            - start_epoch (int): current epoch, 0 if not resuming training
    '''
    if (architecture == "CTL") or (architecture == "efficientnet_v2_m"):
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
            # no save state found/overwrite; start anew
            print('Model found but overwrite enabled, starting new model')
            start_epoch = 0

    # load a specific model file
    elif os.path.isfile(path):
        print(f'Loading model at {path}')
        state = torch.load(open(f'{path}/{start_epoch}.pt', 'rb'))
        model_instance.load_state_dict(state['model'])

    # no dir or file found
    else:
        raise ValueError("Model not found at given path")

    return model_instance, start_epoch


class CTLClassifier(nn.Module):

    def __init__(self, num_classes):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers). NOT IN USE
        '''
        super(CTLClassifier, self).__init__()

        self.feature_extractor = resnet.resnet18(pretrained=True)       # "pretrained": use weights pre-trained on ImageNet

        # replace the very last layer from the original, 1000-class output
        # ImageNet to a new one that outputs num_classes
        last_layer = self.feature_extractor.fc                          # tip: print(self.feature_extractor) to get info on how model is set up
        in_features = last_layer.in_features                            # number of input dimensions to last (classifier) layer
        self.feature_extractor.fc = nn.Identity()                       # discard last layer...

        self.classifier = nn.Linear(in_features, num_classes)           # ...and create a new one

    def forward(self, x):
        '''
            Forward pass (prediction)
        '''
        # x.size(): [B x 3 x W x H]
        features = self.feature_extractor(x)    # features.size(): [B x 512 x W x H]
        prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

        return prediction


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
