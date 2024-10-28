"""
Viewpoint Estimators

"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet
import torch.onnx


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 480


def filter(manifest, value=None):
    if value is None:
        filter = manifest[manifest["viewpoint"].isna()]
    else:
        filter = manifest[manifest["viewpoint"] == value]
    return filter.reset_index(drop=True)


def load(file_path, device='cpu'):
    weights = torch.load(file_path, weights_only=False)
    viewpoint_model = ViewpointModel()
    viewpoint_model.to(device)
    viewpoint_model.load_state_dict(weights, strict=False)
    viewpoint_model.eval()
    return viewpoint_model


class ViewpointModel(nn.Module):

    def __init__(self, num_classes=2, tune=True):
        '''
            Construct the model architecture.
        '''
        super(ViewpointModel, self).__init__()
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
        # x.size(): [B x 3 x 480 x 480]
        x = self.model.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        prediction = self.model.classifier(x)  # prediction.size(): [B x num_classes]

        return prediction