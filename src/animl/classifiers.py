import torch.nn as nn
from torch import flatten
from torchvision.models import resnet
from torchvision.models import efficientnet


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
            Forward pass. Here, we define how to apply our model. It's basically
            applying our modified ResNet-18 on the input tensor ("x") and then
            apply the final classifier layer on the ResNet-18 output to get our
            num_classes prediction.
        '''
        # x.size(): [B x 3 x W x H]
        features = self.feature_extractor(x)    # features.size(): [B x 512 x W x H]
        prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

        return prediction


class EfficientNet(nn.Module):

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
            Forward pass.
        '''
        # x.size(): [B x 3 x W x H]
        x = self.model.features(x)
        x = self.avgpool(x)
        x = flatten(x, 1)

        prediction = self.model.classifier(x)  # prediction.size(): [B x num_classes]

        return prediction
