"""
General utils

"""
import cv2
import math
import os
import random
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import onnxruntime as ort
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn


# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')

# Settings
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)  # OpenMP max threads (PyTorch and SciPy)


def softmax(x):
    '''
    Helper function to softmax
    '''
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)


def tensor_to_onnx(tensor, channel_last=False):
    '''
    Helper function for onnx, shifts dims to BxHxWxC
    '''
    if channel_last:
        tensor = tensor.permute(0, 2, 3, 1)  # reorder BxCxHxW to BxHxWxC

    tensor = tensor.numpy()

    return tensor


# ==============================================================================
# CUDA
# ==============================================================================

def get_device(quiet=False):
    """
    Get Torch device if available
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not quiet:
        print(f'Device is set to {device}.')
    return device

def get_device_onnx(quiet=False):
    """
    Get ort device if available
    """
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        device = 'cuda'
    else:
        device = 'cpu'
    if not quiet:
        print(f'Device is set to {device}.')
    return device


def init_seed(seed):
    '''
    Initalize the seed for all random number generators.

    This is important to be able to reproduce results and experiment with different
    random setups of the same code and experiments.

    Args:
        seed (int): seed for RNG
    '''
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.deterministic = True


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# ==============================================================================
# TRAINING
# ==============================================================================

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not Path(p).exists():  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path
 
# ==============================================================================
# COORDINATE CONVERSION
# ==============================================================================
def xyxyc2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywhc2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxyc2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def xywh2xyxy(bbox):
    """
    Converts bounding boxes from xywh to xyxy format.

    Args:
        bbox (list): Bounding box coordinates in the format [x_min, y_min, width, height].

    Returns:
        list: Normalized bounding box coordinates in the format [x_min, y_min, width, height].
    """
    y = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
    y[2] = y[0] + y[2]  # bottom right x
    y[3] = y[1] + y[3]  # bottom right y
    return y


def xyxy2xywh(bbox):
    """
    Converts bounding boxes from xywh to xyxy format.

    Args:
        bbox (list): Bounding box coordinates in the format [x_min, y_min, width, height].
                     x_min,y_min are the top left corner.

    Returns:
        list: Normalized bounding box coordinates in the format [x_min, y_min, width, height].
    """
    y = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
    y[2] = y[2] - y[0]  # width
    y[3] = y[3] - y[1]  # height
    return y


def absolute_to_relative(bbox, img_size):
    """
    Converts absolute bounding box coordinates to relative coordinates.

    Args:
        bbox (list): Bounding box coordinates in the format [x_min, y_min, width, height].
        img_size (tuple): Image size in the format (width, height).

    Returns:
        list: Bounding box coordinates in the format [x_center, y_center, width, height].
    """
    x_min, y_min, width, height = bbox
    img_width, img_height = img_size

    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height

    return [x_center, y_center, width, height]


def convert_minxywh_to_absxyxy(bbox, width, height):
    """
    Converts bounding box from [x_min, y_min, width, height] to [x1, y1, x2, y2] format.

    Args:
        bbox (list): Bounding box in the format [x_min, y_min, width, height].
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        list: Bounding box in the format [x1, y1, x2, y2].
    """
    x_min, y_min, w, h = bbox
    x1 = x_min
    y1 = y_min
    x2 = x_min + w
    y2 = y_min + h

    return [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def normalize_boxes(bbox, image_sizes):
    """
    Converts absolute bounding box coordinates to relative coordinates.

    Args:
        bbox (list): Absolute bounding box coordinates.
        img_size (tuple): Image size in the format (width, height).

    Returns:
        list: Normalized bounding box coordinates.
    """
    img_height, img_width  = image_sizes
    y = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)   
    y[[0,2]] = np.clip(y[[0,2]] / img_width, 0, 1)
    y[[1,3]] = np.clip(y[[1,3]] / img_height, 0, 1)
    return y

# ==============================================================================
# MDV5
# ==============================================================================

def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    

    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywhc2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output

# ==============================================================================
# Augmentations
# ==============================================================================

def letterbox(im: np.ndarray, new_shape = (640, 640),
              color = (114, 114, 114),
              auto: bool = True,
              scaleFill: bool = False,
              scaleup: bool = True,
              stride: int = 32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def scale_letterbox(bbox, resized_shape, original_shape):
    """
    Converts bounding box coordinates from a resized, letterboxed image space
    back to the original image's coordinate space. Assumes input coordinates
    are in normalized [x_corner, y_corner, width, height] format.

    Args:
        bbox (np.ndarray or torch.Tensor): A numpy array or tensor of bounding
                                             boxes, shape (n, 4), in
                                             (x_corner, y_corner, width, height) format.
                                             Coordinates are in pixels relative
                                             to the resized/padded image.
        resized_shape (tuple): The (height, width) of the resized and
                               letterboxed image.
        original_shape (tuple): The (height, width) of the original image.

    Returns:
        np.ndarray: A numpy array of bounding boxes, shape (n, 4), with
                    coordinates in normalized (x_corner, y_corner, width, height)
                    format.
    """
    # Convert to numpy array if it's a tensor
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.cpu().numpy()
    
    if isinstance(resized_shape, torch.Tensor):
        resized_shape = resized_shape.cpu().numpy()

    if isinstance(original_shape, torch.Tensor):
        original_shape = original_shape.cpu().numpy()

    # Convert input xywh (top-left corner) to xyxy
    xyxy_coords = xywh2xyxy(bbox)

    # Calculate the scaling ratio and padding
    ratio = min(resized_shape[0] / original_shape[0], resized_shape[1] / original_shape[1])
    new_unpad_shape = (int(round(original_shape[0] * ratio)), int(round(original_shape[1] * ratio)))
    dw = (resized_shape[1] - new_unpad_shape[1]) / 2  # x-padding
    dh = (resized_shape[0] - new_unpad_shape[0]) / 2  # y-padding

    # Remove padding from coordinates
    xyxy_coords[[0, 2]] -= (dw / resized_shape[1])
    xyxy_coords[[1, 3]] -= (dh /resized_shape[0])

    # Scale to original image size
    xyxy_coords[[0, 2]] = xyxy_coords[[0, 2]] *  resized_shape[1]/new_unpad_shape[1]
    xyxy_coords[[1, 3]] = xyxy_coords[[1, 3]] *  resized_shape[0]/new_unpad_shape[0]

    # Clip coordinates to be within the original image dimensions
    xyxy_coords[[0, 2]] = np.clip(xyxy_coords[[0, 2]], 0, 1)  
    xyxy_coords[[1, 3]] = np.clip(xyxy_coords[[1, 3]], 0, 1) 

    # Convert final xyxy to xywh (top-left corner)
    xywh_coords = xyxy2xywh(xyxy_coords)

    return xywh_coords

def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90, }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image
