'''
    Model implementation.
    We'll be using a "simple" ResNet-18 for image classification here.

    2022 Benjamin Kellenberger
'''
import yaml
import os
import glob
import torch
import torch.nn as nn
from torchvision.models import resnet


def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class
    overwrite = cfg['overwrite']
    exp_folder = cfg['experiment_folder']

    # load latest model state
    model_states = glob.glob(exp_folder + '*.pt')

    if len(model_states) and overwrite==False:
        # at least one save state found; get latest
        model_epochs = [int(m.replace(exp_folder,'').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'{exp_folder}/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch


def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    exp_folder = cfg['experiment_folder']
    os.makedirs(exp_folder, exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'{exp_folder}/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = exp_folder + '/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)


class CustomResNet18(nn.Module):

    def __init__(self, num_classes):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        super(CustomResNet18, self).__init__()

        self.feature_extractor = resnet.resnet18(pretrained=True)       # "pretrained": use weights pre-trained on ImageNet

        # replace the very last layer from the original, 1000-class output
        # ImageNet to a new one that outputs num_classes
        last_layer = self.feature_extractor.fc                          # tip: print(self.feature_extractor) to get info on how model is set up
        in_features = last_layer.in_features                            # number of input dimensions to last (classifier) layer
        self.feature_extractor.fc = nn.Identity()                       # discard last layer...

        self.classifier = nn.Linear(in_features, num_classes)           # ...and create a new one
    

    def forward(self, x):
        '''
            Forward pass. Here, we define how to apply our model. It's basically
            applying our modified ResNet-18 on the input tensor ("x") and then
            apply the final classifier layer on the ResNet-18 output to get our
            num_classes prediction.
        '''
        # x.size(): [B x 3 x W x H]
        features = self.feature_extractor(x)    # features.size(): [B x 512 x W x H]
        prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

        return prediction