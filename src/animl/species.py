"""
Class Definitions for Species Classification
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet, convnext_base, ConvNeXt_Base_Weights

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
        # load the ConvNeXt-Base model pre-trained on ImageNet 1K
        self.model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
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