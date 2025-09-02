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
                                  load_classifier_checkpoint,
                                  multispecies_classification, save_classifier,
                                  sequence_classification,
                                  single_classification,)
from animl.detection import (MEGADETECTORv5_SIZE, MEGADETECTORv5_STRIDE,
                             convert_yolo_detections, detect, load_detector,
                             parse_detections,)
from animl.export import (export_coco, export_megadetector, export_timelapse,
                          export_to_folders, export_to_folders_MD, remove_link,
                          update_labels_from_folders,)
from animl.file_management import (IMAGE_EXTENSIONS, VALID_EXTENSIONS,
                                   VIDEO_EXTENSIONS, WorkingDirectory,
                                   active_times, build_file_manifest,
                                   check_file, load_data, load_json, save_data,
                                   save_detection_checkpoint, save_json,)
from animl.generator import (ImageGenerator, Letterbox, TrainGenerator,
                             collate_fn, image_to_tensor, manifest_dataloader,
                             train_dataloader,)
from animl.model_architecture import (ConvNeXtBase, EfficientNet,
                                      MEGADETECTOR_SIZE, MIEWID_SIZE,
                                      SDZWA_CLASSIFIER_SIZE,)
from animl.models import (AutoShape, BaseModel, Bottleneck, BottleneckCSP, C3,
                          C3Ghost, C3SPP, C3TR, C3x, CLASSIFIER, CLASS_LIST,
                          Classify, Concat, Contract, Conv, CrossConv, DWConv,
                          DWConvTranspose2d, Detect, DetectMultiBackend,
                          DetectionModel, Detections, Expand, FILE, Focus,
                          GhostBottleneck, GhostConv, MEGADETECTOR, Model,
                          ROOT, SPP, SPPF, Segment, TransformerBlock,
                          TransformerLayer, autopad, check_anchor_order,
                          check_suffix, common, download, download_model,
                          parse_model, yolo,)
from animl.pipeline import (from_config, from_paths,)
from animl.pose import (predict_viewpoints, viewpoint,)
from animl.reid import (ArcFaceLossAdaptiveMargin, ArcFaceSubCenterDynamic,
                        ArcMarginProduct, ArcMarginProduct_subcenter,
                        ElasticArcFace, GeM, MIEW_HEIGHT, MIEW_WIDTH,
                        MiewIdNet, compute_batched_distance_matrix,
                        compute_distance_matrix, cosine_distance, distance,
                        euclidean_squared_distance, extract_miew_embeddings,
                        inference, l2_norm, load_miew, miewid,
                        remove_diagonal,)
from animl.split import (get_animals, get_empty, train_val_test,)
from animl.test import (test_func, test_main,)
from animl.train import (train_func, train_main, validate_func,)
from animl.utils import (NUM_THREADS, absolute_to_relative, box_area, box_iou,
                         check_img_size, clean_str, clip_coords,
                         convert_minxywh_to_absxyxy, convert_yolo_to_xywh,
                         copy_attr, demo_boxes, device_count, exif_transpose,
                         file_age, file_date, file_size, fuse_conv_and_bn,
                         general, get_device, get_image_size, get_latest_run,
                         increment_path, init_seed, initialize_weights,
                         intersect_dicts, letterbox, main, make_divisible,
                         non_max_suppression, normalize_boxes,
                         plot_all_bounding_boxes, plot_box, scale_coords,
                         scale_img, scale_letterbox, select_device, softmax,
                         tensor_to_onnx, time_sync, visualization, xyn2xy,
                         xywh2xyxy, xywhc2xyxy, xywhn2xyxy, xyxy2xywh,
                         xyxyc2xywh, xyxyc2xywhn,)
from animl.video_processing import (extract_frame_single, extract_frames,)

__all__ = ['ArcFaceLossAdaptiveMargin', 'ArcFaceSubCenterDynamic',
           'ArcMarginProduct', 'ArcMarginProduct_subcenter', 'AutoShape',
           'BaseModel', 'Bottleneck', 'BottleneckCSP', 'C3', 'C3Ghost',
           'C3SPP', 'C3TR', 'C3x', 'CLASSIFIER', 'CLASS_LIST', 'Classify',
           'Concat', 'Contract', 'Conv', 'ConvNeXtBase', 'CrossConv', 'DWConv',
           'DWConvTranspose2d', 'Detect', 'DetectMultiBackend',
           'DetectionModel', 'Detections', 'EfficientNet', 'ElasticArcFace',
           'Expand', 'FILE', 'Focus', 'GeM', 'GhostBottleneck', 'GhostConv',
           'IMAGE_EXTENSIONS', 'ImageGenerator', 'Letterbox', 'MEGADETECTOR',
           'MEGADETECTOR_SIZE', 'MEGADETECTORv5_SIZE', 'MEGADETECTORv5_STRIDE',
           'MIEWID_SIZE', 'MIEW_HEIGHT', 'MIEW_WIDTH', 'MiewIdNet', 'Model',
           'NUM_THREADS', 'ROOT', 'SDZWA_CLASSIFIER_SIZE', 'SPP', 'SPPF',
           'Segment', 'TrainGenerator', 'TransformerBlock', 'TransformerLayer',
           'VALID_EXTENSIONS', 'VIDEO_EXTENSIONS', 'WorkingDirectory',
           'absolute_to_relative', 'active_times', 'autopad', 'box_area',
           'box_iou', 'build_file_manifest', 'check_anchor_order',
           'check_file', 'check_img_size', 'check_suffix', 'classification',
           'classify', 'clean_str', 'clip_coords', 'collate_fn', 'common',
           'compute_batched_distance_matrix', 'compute_distance_matrix',
           'convert_minxywh_to_absxyxy', 'convert_yolo_detections',
           'convert_yolo_to_xywh', 'copy_attr', 'cosine_distance',
           'demo_boxes', 'detect', 'detection', 'device_count', 'distance',
           'download', 'download_model', 'euclidean_squared_distance',
           'exif_transpose', 'export', 'export_coco', 'export_megadetector',
           'export_timelapse', 'export_to_folders', 'export_to_folders_MD',
           'extract_frame_single', 'extract_frames', 'extract_miew_embeddings',
           'file_age', 'file_date', 'file_management', 'file_size',
           'from_config', 'from_paths', 'fuse_conv_and_bn', 'general',
           'generator', 'get_animals', 'get_device', 'get_empty',
           'get_image_size', 'get_latest_run', 'image_to_tensor',
           'increment_path', 'inference', 'init_seed', 'initialize_weights',
           'intersect_dicts', 'l2_norm', 'letterbox', 'load_class_list',
           'load_classifier', 'load_classifier_checkpoint', 'load_data',
           'load_detector', 'load_json', 'load_miew', 'main', 'make_divisible',
           'manifest_dataloader', 'miewid', 'model_architecture', 'models',
           'multispecies_classification', 'non_max_suppression',
           'normalize_boxes', 'parse_detections', 'parse_model', 'pipeline',
           'plot_all_bounding_boxes', 'plot_box', 'pose', 'predict_viewpoints',
           'reid', 'remove_diagonal', 'remove_link', 'save_classifier',
           'save_data', 'save_detection_checkpoint', 'save_json',
           'scale_coords', 'scale_img', 'scale_letterbox', 'select_device',
           'sequence_classification', 'single_classification', 'softmax',
           'split', 'tensor_to_onnx', 'test', 'test_func', 'test_main',
           'time_sync', 'train', 'train_dataloader', 'train_func',
           'train_main', 'train_val_test', 'update_labels_from_folders',
           'utils', 'validate_func', 'video_processing', 'viewpoint',
           'visualization', 'xyn2xy', 'xywh2xyxy', 'xywhc2xyxy', 'xywhn2xyxy',
           'xyxy2xywh', 'xyxyc2xywh', 'xyxyc2xywhn', 'yolo']
