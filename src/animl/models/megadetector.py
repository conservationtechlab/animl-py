'''
Based on MegaDetector/detections/pytorch_detector.py

Version 5

'''
import math
import torch
from pathlib import Path
import numpy as np
import traceback
from animl.utils import general, augmentations
from animl.models import yolo


CONF_DIGITS = 3
COORD_DIGITS = 4


class MegaDetector:

    IMAGE_SIZE = 1280  # image size used in training
    STRIDE = 64

    def __init__(self, model_path: str, device=None):
        if not torch.cuda.is_available():
            self.device = 'cpu'
        elif torch.cuda.is_available() and device is None:
            self.device = 'cuda:0'
        else:
            self.device = device

        print('Sending model to %s' % self.device)
        # model_path = str(model_path) if isinstance(model_path, Path) else model_path
        self.model = MegaDetector._load_model(Path(model_path), self.device)
        # self.model = MegaDetector._load_model(Path(r""+model_path), self.device)
        self.model.to(device)

        self.printed_image_size_warning = False

    @staticmethod
    def _load_model(model_pt_path, device):
        checkpoint = torch.load(model_pt_path, map_location=device)
        # Compatibility fix that allows older YOLOv5 models with
        # newer versions of YOLOv5/PT
        for m in checkpoint['model'].modules():
            t = type(m)
            if t is torch.nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                m.recompute_scale_factor = None

        model = checkpoint['model'].float().fuse().eval()  # FP32 model
        return model

    def generate_detections_one_image(self, img_original, image_id,
                                      confidence_threshold, image_size=None,
                                      skip_image_resize=False):
        """
        Apply the detector to an image.

        Args:
            img_original: the PIL Image object with EXIF rotation taken into account
            image_id: a path to identify the image; will be in the "file" field of the output object
            confidence_threshold: confidence above which to include the detection proposal
            skip_image_resizing: whether to skip internal image resizing and rely on external resizing

        Returns:
        A dict with the following fields, see the 'images' key in https://github.com/agentmorris/MegaDetector/tree/master/api/batch_processing#batch-processing-api-output-format
            - 'file' (always present)
            - 'max_detection_conf' (removed from MegaDetector output by default, but generated here)
            - 'detections', which is a list of detection objects containing keys 'category',
              'conf' and 'bbox'
            - 'failure'
        """
        result = {'file': image_id}
        detections = []
        max_conf = 0.0

        try:
            img_original = np.asarray(img_original)

            # padded resize
            target_size = MegaDetector.IMAGE_SIZE

            # Image size can be an int (which translates to a square target size) or (h,w)
            if image_size is not None:

                assert isinstance(image_size, int) or (len(image_size) == 2)

                if not self.printed_image_size_warning:
                    print('Warning: using user-supplied image size {}'.format(image_size))
                    self.printed_image_size_warning = True

                target_size = image_size

            else:
                self.printed_image_size_warning = False

            if skip_image_resize:
                img = img_original
            else:
                img = augmentations.letterbox(img_original, new_shape=target_size,
                                              stride=MegaDetector.STRIDE, auto=True)[0]
            # HWC to CHW; PIL Image is RGB already
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            img = img.to(self.device)
            img = img.float()
            img /= 255

            # In practice this is always true
            if len(img.shape) == 3:
                img = torch.unsqueeze(img, 0)

            pred: list = self.model(img)[0]

            # NMS
            if self.device == 'mps':
                # As of v1.13.0.dev20220824, nms is not implemented for MPS.
                # Send predication back to the CPU to fix.
                pred = general.non_max_suppression(prediction=pred.cpu(), conf_thres=confidence_threshold)
            else:
                pred = general.non_max_suppression(prediction=pred, conf_thres=confidence_threshold)
            # format detections/bounding boxes
            # normalization gain whwh
            gn = torch.tensor(img_original.shape)[[1, 0, 1, 0]]

            # This is a loop over detection batches, which will always be length 1
            for det in pred:
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = general.scale_coords(img.shape[2:], det[:, :4], img_original.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        # normalized center-x, center-y, width and height
                        xywh = (general.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                        api_box = convert_yolo_to_xywh(xywh)
                        api_box = truncate_float_array(api_box, precision=COORD_DIGITS)

                        conf = truncate_float(conf.tolist(), precision=CONF_DIGITS)

                        cls = int(cls.tolist()) + 1
                        if cls not in (1, 2, 3):
                            raise KeyError(f'{cls} is not a valid class.')

                        detections.append({
                            'category': str(cls),
                            'conf': conf,
                            'bbox1': api_box[0],
                            'bbox2': api_box[1],
                            'bbox3': api_box[2],
                            'bbox4': api_box[3],
                        })
                        max_conf = max(max_conf, conf)
                    # ...for each detection in this batch
                # ...if this is a non-empty batch
            # ...for each detection batch
        # ...try
        except Exception as e:
            result['failure'] = 'Failure inference'
            print('PTDetector: image {} failed during inference: {}\n'.format(image_id, str(e)))
            traceback.print_exc(e)

        result['max_detection_conf'] = max_conf
        result['detections'] = detections

        return result


# From MegeDetector/ct_utils
def convert_yolo_to_xywh(yolo_box):
    """
    Converts a YOLO format bounding box to [x_min, y_min, width_of_box, height_of_box].

    Args:
        yolo_box: bounding box of format [x_center, y_center, width_of_box, height_of_box].

    Returns:
        bbox with coordinates represented as [x_min, y_min, width_of_box, height_of_box].
    """
    x_center, y_center, width_of_box, height_of_box = yolo_box
    x_min = x_center - width_of_box / 2.0
    y_min = y_center - height_of_box / 2.0
    return [x_min, y_min, width_of_box, height_of_box]


def truncate_float(x, precision=3):
    """
    Truncates a floating-point value to a specific number of significant digits.
    For example: truncate_float(0.0003214884) --> 0.000321
    This function is primarily used to achieve a certain float representation
    before exporting to JSON.

    Args:
        - x (float): Scalar to truncate
        - precision (int): The number of significant digits to preserve, should be
                      greater or equal 1
    """
    assert precision > 0

    if np.isclose(x, 0):
        return 0
    else:
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit.
        factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
        # Shift decimal point by multiplicatipon with factor, flooring, and
        # division by factor.
        return math.floor(x * factor)/factor


def truncate_float_array(xs, precision=3):
    """
    Vectorized version of truncate_float(...)

    Args:
        - xs (list of float): List of floats to truncate
        - precision (int): The number of significant digits to preserve,
                           should be greater or equal 1
    """
    return [truncate_float(x, precision=precision) for x in xs]
