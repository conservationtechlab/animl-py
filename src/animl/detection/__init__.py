from animl.detection import inference
from animl.detection import train

from animl.detection.inference import (MEGADETECTORv5_SIZE,
                                       MEGADETECTORv5_STRIDE, detect,
                                       load_detector, parse_detections,)

__all__ = ['MEGADETECTORv5_SIZE', 'MEGADETECTORv5_STRIDE', 'detect',
           'inference', 'load_detector', 'parse_detections', 'train']