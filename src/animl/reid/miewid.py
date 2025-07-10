"""
Code to run Miew_ID

(source)

"""
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from animl.reid.heads import ElasticArcFace, ArcFaceSubCenterDynamic
from animl.utils.general import get_device
from animl.generator import manifest_dataloader

IMAGE_HEIGHT = 440
IMAGE_WIDTH = 440


def filter(rois: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rois that have not yet had embedding extracted

    Args:
        - rois (dataframe): list of rois

    Returns:
        - subset of rois with no extracted embedding
    """
    return rois[rois['emb_id'] == 0].reset_index(drop=True)


def load_miew(file_path, device=None):
    """
    Load MiewID from file path

    Args:
        - file_path (str): file path to model file
        - device (str): device to load model to

    Returns:
        loaded miewid model object
    """
    if device is None:
        device = get_device()
    print('Sending model to %s' % device)
    weights = torch.load(file_path, weights_only=True)
    miew = MiewIdNet(device=device)
    miew.to(device)
    miew.load_state_dict(weights, strict=False)
    miew.eval()
    return miew


def extract_embeddings(manifest, miew_model, file_col="FilePath", batch_size=1, num_workers=1, device=None):
    """
    Wrapper for MiewID embedding extraction within MatchyPatchy
    """
    if device is None:
        device = get_device()
    output = []
    if isinstance(manifest, pd.DataFrame):
        dataloader = manifest_dataloader(manifest, batch_size=batch_size, num_workers=num_workers,
                                         file_col=file_col, crop=True, normalize=True,
                                         resize_width=IMAGE_WIDTH, resize_height=IMAGE_HEIGHT)
        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader)):
                img = batch[0]
                emb = miew_model.extract_feat(img.to(device))
                output.append(emb.cpu().detach().numpy()[0])
    return output


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
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=True,
                 margins=None,
                 k=None):

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
            self.bn.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ElasticArcFace(final_in_features, n_classes,
                                        s=s, m=margin)
        elif loss_module == 'arcface_subcenter_dynamic':
            if margins is None:
                margins = [0.3] * n_classes
            self.final = ArcFaceSubCenterDynamic(embedding_dim=final_in_features,
                                                 output_classes=n_classes,
                                                 margins=margins, s=s, k=k)
        # elif loss_module == 'cosface':
        #     self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        # elif loss_module == 'adacos':
        #     self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self) -> None:
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        feature = self.extract_feat(x)
        return feature
        # if not self.training:
        #     return feature
        # else:
        #     assert label is not None
        # if self.loss_module in ('arcface', 'arcface_subcenter_dynamic'):
        #     logits = self.final(feature, label)
        # else:
        #     logits = self.final(feature)
        #
        # return logits

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
