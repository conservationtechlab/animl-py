__version__ = '3.1.1'

from animl import classification
from animl import detection
from animl import export
from animl import file_management
from animl import generator
from animl import model_architecture
from animl import models
from animl import pipeline
from animl import pose
from animl import reid
from animl import split
from animl import test
from animl import train
from animl import utils
from animl import video_processing

from animl.classification import (classify, load_class_list, load_classifier,
                                  load_classifier_checkpoint, save_classifier,
                                  sequence_classification,
                                  single_classification,)
from animl.detection import (convert_onnx_detections, convert_yolo_detections,
                             detect, load_detector, parse_detections,)
from animl.export import (export_coco, export_folders, export_megadetector,
                          export_timelapse, remove_link,
                          update_labels_from_folders,)
from animl.file_management import (IMAGE_EXTENSIONS, VALID_EXTENSIONS,
                                   VIDEO_EXTENSIONS, WorkingDirectory,
                                   active_times, build_file_manifest,
                                   check_file, load_data, load_json, save_data,
                                   save_detection_checkpoint, save_json,)
from animl.generator import (Letterbox, ManifestGenerator, TrainGenerator,
                             collate_fn, image_to_tensor, manifest_dataloader,
                             train_dataloader,)
from animl.model_architecture import (ConvNeXtBase, EfficientNet,
                                      MEGADETECTORv5_SIZE,
                                      MEGADETECTORv5_STRIDE,
                                      SDZWA_CLASSIFIER_SIZE,)
from animl.models import (AutoShape, BaseModel, Bottleneck, BottleneckCSP, C3,
                          C3Ghost, C3SPP, C3TR, C3x, CLASSIFIER, CLASS_LIST,
                          Classify, Concat, Contract, Conv, CrossConv, DWConv,
                          DWConvTranspose2d, Detect, DetectMultiBackend,
                          DetectionModel, Detections, Expand, FILE, Focus,
                          GhostBottleneck, GhostConv, MD_FILENAMES,
                          MEGADETECTOR, Model, ROOT, SPP, SPPF, Segment,
                          TransformerBlock, TransformerLayer, autopad,
                          check_anchor_order, check_suffix, common, download,
                          download_model, list_models, parse_model, yolo,)
from animl.pipeline import (from_config, from_paths,)
from animl.pose import (predict_viewpoints, viewpoint,)
from animl.reid import (ArcFaceLossAdaptiveMargin, ArcFaceSubCenterDynamic,
                        ArcMarginProduct, ArcMarginProduct_subcenter,
                        ElasticArcFace, GeM, MIEWID_SIZE, MiewIdNet,
                        compute_batched_distance_matrix,
                        compute_distance_matrix, cosine_distance, distance,
                        euclidean_squared_distance, extract_miew_embeddings,
                        inference, l2_norm, load_miew, miewid,
                        remove_diagonal,)
from animl.split import (get_animals, get_empty, train_val_test,)
from animl.test import (test_func, test_main,)
from animl.train import (train_func, train_main, validate_func,)
from animl.utils import (NUM_THREADS, absolute_to_relative, box_area, box_iou,
                         clip_coords, convert_minxywh_to_absxyxy, copy_attr,
                         exif_transpose, fuse_conv_and_bn, general, get_device,
                         increment_path, init_seed, initialize_weights,
                         letterbox, make_divisible, non_max_suppression,
                         normalize_boxes, plot_all_bounding_boxes, plot_box,
                         plot_from_file, scale_coords, scale_img,
                         scale_letterbox, softmax, tensor_to_onnx, time_sync,
                         visualization, xyn2xy, xywh2xyxy, xywhc2xyxy,
                         xywhn2xyxy, xyxy2xywh, xyxyc2xywh, xyxyc2xywhn,)
from animl.video_processing import (count_frames, extract_frames,
                                    get_frame_as_image,)

__all__ = ['ArcFaceLossAdaptiveMargin', 'ArcFaceSubCenterDynamic',
           'ArcMarginProduct', 'ArcMarginProduct_subcenter', 'AutoShape',
           'BaseModel', 'Bottleneck', 'BottleneckCSP', 'C3', 'C3Ghost',
           'C3SPP', 'C3TR', 'C3x', 'CLASSIFIER', 'CLASS_LIST', 'Classify',
           'Concat', 'Contract', 'Conv', 'ConvNeXtBase', 'CrossConv', 'DWConv',
           'DWConvTranspose2d', 'Detect', 'DetectMultiBackend',
           'DetectionModel', 'Detections', 'EfficientNet', 'ElasticArcFace',
           'Expand', 'FILE', 'Focus', 'GeM', 'GhostBottleneck', 'GhostConv',
           'IMAGE_EXTENSIONS', 'Letterbox', 'MD_FILENAMES', 'MEGADETECTOR',
           'MEGADETECTORv5_SIZE', 'MEGADETECTORv5_STRIDE', 'MIEWID_SIZE',
           'ManifestGenerator', 'MiewIdNet', 'Model', 'NUM_THREADS', 'ROOT',
           'SDZWA_CLASSIFIER_SIZE', 'SPP', 'SPPF', 'Segment', 'TrainGenerator',
           'TransformerBlock', 'TransformerLayer', 'VALID_EXTENSIONS',
           'VIDEO_EXTENSIONS', 'WorkingDirectory', 'absolute_to_relative',
           'active_times', 'autopad', 'box_area', 'box_iou',
           'build_file_manifest', 'check_anchor_order', 'check_file',
           'check_suffix', 'classification', 'classify', 'clip_coords',
           'collate_fn', 'common', 'compute_batched_distance_matrix',
           'compute_distance_matrix', 'convert_minxywh_to_absxyxy',
           'convert_onnx_detections', 'convert_yolo_detections', 'copy_attr',
           'cosine_distance', 'count_frames', 'detect', 'detection',
           'distance', 'download', 'download_model',
           'euclidean_squared_distance', 'exif_transpose', 'export',
           'export_coco', 'export_folders', 'export_megadetector',
           'export_timelapse', 'extract_frames', 'extract_miew_embeddings',
           'file_management', 'from_config', 'from_paths', 'fuse_conv_and_bn',
           'general', 'generator', 'get_animals', 'get_device', 'get_empty',
           'get_frame_as_image', 'image_to_tensor', 'increment_path',
           'inference', 'init_seed', 'initialize_weights', 'l2_norm',
           'letterbox', 'list_models', 'load_class_list', 'load_classifier',
           'load_classifier_checkpoint', 'load_data', 'load_detector',
           'load_json', 'load_miew', 'make_divisible', 'manifest_dataloader',
           'miewid', 'model_architecture', 'models', 'non_max_suppression',
           'normalize_boxes', 'parse_detections', 'parse_model', 'pipeline',
           'plot_all_bounding_boxes', 'plot_box', 'plot_from_file', 'pose',
           'predict_viewpoints', 'reid', 'remove_diagonal', 'remove_link',
           'save_classifier', 'save_data', 'save_detection_checkpoint',
           'save_json', 'scale_coords', 'scale_img', 'scale_letterbox',
           'sequence_classification', 'single_classification', 'softmax',
           'split', 'tensor_to_onnx', 'test', 'test_func', 'test_main',
           'time_sync', 'train', 'train_dataloader', 'train_func',
           'train_main', 'train_val_test', 'update_labels_from_folders',
           'utils', 'validate_func', 'video_processing', 'viewpoint',
           'visualization', 'xyn2xy', 'xywh2xyxy', 'xywhc2xyxy', 'xywhn2xyxy',
           'xyxy2xywh', 'xyxyc2xywh', 'xyxyc2xywhn', 'yolo']
