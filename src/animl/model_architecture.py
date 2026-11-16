"""
Class Definitions for Species Classification

@ Kyra Swanson 2023
"""
import open_clip
from peft import LoraConfig, get_peft_model
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet, convnext_base, ConvNeXt_Base_Weights

MEGADETECTORv5_SIZE = 1280
MEGADETECTORv5_STRIDE = 64
MD_LABELS = {0: "empty", 1: "animal", 2: "human",  3: "vehicle"}
MD_MODELS = {"mdv5", "mdv6", "mdv1000-redwood", "mdv1000-spruce", "mdv1000-cedar", "mdv1000-larch", "mdv1000-sorrel"}

SDZWA_CLASSIFIER_SIZE = 480
BIOCLIP_CLASSIFIER_SIZE = 224
MIEWID_SIZE = 440


class EfficientNet(nn.Module):
    '''
    Construct the EfficientNet model architecture.
    '''
    def __init__(self, num_classes, device=None, tune=False):
        super(EfficientNet, self).__init__()
        self.device = device
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
        return self.model(x)


class ConvNeXtBase(nn.Module):
    '''
    Construct the ConvNeXt-Base model architecture.
    '''
    def __init__(self, num_classes, tune=False):
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


class BioCLIP(nn.Module):
    '''
    Construct the BioClip2 model architecture.
    '''
    def __init__(self, num_classes, tune=False):
        super(BioCLIP, self).__init__()
        # load the BioClip2 vision encoder pre-trained on TreeOfLife-200M
        full_model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        )
        self.model = full_model.visual
        embedding_dim = self.model.output_dim
        # set up low-rank adaptation
        config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["attn", "c_fc", "c_proj"],
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
        self.classifier = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x):
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


class GeM(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x: torch.Tensor, p: torch.Tensor, eps: float) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class MiewIdNet(nn.Module):
    def __init__(self,
                 device=None,
                 n_classes=10,
                 model_name='efficientnetv2_rw_m',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 pretrained=True):

        super(MiewIdNet, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.model_name = model_name
        self.device = device

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if model_name.startswith('efficientnetv2_rw'):
            final_in_features = self.backbone.classifier.in_features
        if model_name.startswith('swinv2'):
            final_in_features = self.backbone.norm.normalized_shape[0]

        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()

        self.pooling = GeM()
        self.bn = nn.BatchNorm1d(final_in_features)
        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.bn = nn.BatchNorm1d(fc_dim)
            self.bn.bias.requires_grad_(False)
            self.fc = nn.Linear(final_in_features, n_classes, bias=False)
            self.bn.apply(self.weights_init_kaiming)
            self.fc.apply(self.weights_init_classifier)
            final_in_features = fc_dim

        self.loss_module = loss_module
        self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self) -> None:
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        feature = self.extract_feat(x)
        return feature

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.backbone.forward_features(x)
        if self.model_name.startswith('swinv2'):
            x = x.permute(0, 3, 1, 2)

        x = self.pooling(x).view(batch_size, -1)
        x = self.bn(x)
        if self.use_fc:
            x1 = self.dropout(x)
            x1 = self.bn(x1)
            x1 = self.fc(x1)

        return x

    def extract_logits(self, x: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        feature = self.extract_feat(x)
        assert label is not None
        if self.loss_module in ('arcface', 'arcface_subcenter_dynamic'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)

        return logits

    def weights_init_kaiming(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)
