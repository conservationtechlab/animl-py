"""
From MiewID

Class definitions for MiewID model and head layers
"""
import torch
import torch.nn as nn
import math
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter


MIEWID_SIZE = 440


def l2_norm(input: torch.Tensor, axis: int = 1) -> torch.Tensor:
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)wandb: ERROR Abnormal program exit

        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ElasticArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, std=0.0125, plus=False, k=None):
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std = std
        self.plus = plus

    def forward(self, embeddings, label):
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device)  # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta


# Subcenter Arcface with dynamic margin

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features: torch.Tensor):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins: torch.Tensor, out_dim: int, s: float) -> None:
        super().__init__()
#       self.crit = nn.CrossEntropyLoss()
        self.s = s
        self.register_buffer('margins', torch.tensor(margins))
        self.out_dim = out_dim

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # ms = []
        # ms = self.margins[labels.cpu().numpy()]
        ms = self.margins[labels]
        cos_m = torch.cos(ms)  # torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.sin(ms)  # torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.cos(math.pi - ms)  # torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.sin(math.pi - ms) * ms  # torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        cosine = logits
        sine = torch.sqrt(1.0 - cosine * cosine)
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        return output


class ArcFaceSubCenterDynamic(nn.Module):
    def __init__(self, embedding_dim, output_classes, margins, s, k=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_classes = output_classes
        self.margins = margins
        self.s = s
        self.wmetric_classify = ArcMarginProduct_subcenter(self.embedding_dim, self.output_classes, k=k)

        self.warcface_margin = ArcFaceLossAdaptiveMargin(margins=self.margins,
                                                         out_dim=self.output_classes,
                                                         s=self.s)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.wmetric_classify(features.float())
        logits = self.warcface_margin(logits, labels)
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
            self.bn.apply(self.weights_init_kaiming)
            self.fc.apply(self.weights_init_classifier)
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
