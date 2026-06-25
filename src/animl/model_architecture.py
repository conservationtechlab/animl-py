"""
Class Definitions for Species Classification

@ Kyra Swanson 2023
"""
import open_clip
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
from torchvision.models import efficientnet, convnext_base, ConvNeXt_Base_Weights

MEGADETECTORv5_SIZE = 1280
MEGADETECTORv5_STRIDE = 64
MD_LABELS = {0: "empty", 1: "animal", 2: "human",  3: "vehicle"}
MD_MODELS = {"mdv5", "mdv6", "mdv1000-redwood", "mdv1000-spruce", "mdv1000-cedar", "mdv1000-larch", "mdv1000-sorrel"}
SDZWA_CLASSIFIER_SIZE = 480


class EfficientNet(nn.Module):
    '''
    Construct the EfficientNet model architecture.
    '''
    def __init__(self, num_classes, device=None, tune=False):
        super(EfficientNet, self).__init__()
        self.device = device
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # "pretrained": use weights pre-trained on ImageNet
        self.model = efficientnet.efficientnet_v2_m(weights=efficientnet.EfficientNet_V2_M_Weights.DEFAULT)
        if tune:
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False

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
    '''
    Construct the ConvNeXt-Base model architecture.
    '''
    def __init__(self, num_classes, tune=True):
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


class BioClip(nn.Module):
    '''
    Construct the BioClip2 model architecture.
    '''
    def __init__(self, num_classes, tune=False):
        super(BioClip,self).__init__()
        # load the BioClip2 vision encoder pre-trained on TreeOfLife-200M
        full_model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        )
        self.model = full_model.visual
        # set up low-rank adaptation
        config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["attn","c_fc","c_proj"],
                lora_dropout=0.0,
                bias="none",
                modules_to_save=None
        )
        # this freezes the base visual encoder and injects trainable LoRA layers
        self.model = get_peft_model(self.model, config)

        if not tune:
            for param in self.model.parameters():
                param.requires_grad = False

        # Add a classifier layer
        embedding_dim = self.model.base_model.output_dim
        self.classifier = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self,x):
        '''
        Forward pass (prediction)
        '''
        # only use the visual encoder of the BioClip2 model
        features = self.model(x)
        # features.size(): [B x 768]
        # normalize so the model learns based off direction only
        features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classifier(features)
        return logits
