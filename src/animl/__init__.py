from animl import api
from animl import classification
from animl import detection
from animl import file_management
from animl import generator
from animl import link
from animl import model_architecture
from animl import models
from animl import multi_species
from animl import pipeline
from animl import pose
from animl import reid
from animl import split
from animl import test
from animl import train
from animl import utils
from animl import video_processing

from animl.api import (MiewGenerator, animl_results_to_md_results, animl_to_md,
                       csv_converter, detection_category_id_to_name, main,
                       matchypatchy, miew_embedding, reid_dataloader,
                       timelapse, viewpoint_estimator,)
from animl.classification import (classify, individual_classification,
                                  load_class_list, load_classifier,
                                  save_classifier, sequence_classification,)
from animl.detection import (convert_raw_detections, convert_yolo_detections,
                             detect, load_detector, parse_detections,)
from animl.file_management import (IMAGE_EXTENSIONS, VALID_EXTENSIONS,
                                   VIDEO_EXTENSIONS, WorkingDirectory,
                                   active_times, build_file_manifest,
                                   check_file, load_data, save_checkpoint,
                                   save_data,)
from animl.generator import (ImageGenerator, ResizeWithPadding, TrainGenerator,
                             collate_fn, image_to_tensor, manifest_dataloader,
                             train_dataloader,)
from animl.link import (remove_link, sort_MD, sort_species, update_labels,)
from animl.model_architecture import (ConvNeXtBase, EfficientNet,)
from animl.models import (AutoShape, Bottleneck, BottleneckCSP, C3, C3Ghost,
                          C3SPP, C3TR, C3x, Classify, Concat, Contract, Conv,
                          CrossConv, DWConv, DWConvTranspose2d, Detect,
                          DetectMultiBackend, Detections, Expand, FILE, Focus,
                          GhostBottleneck, GhostConv, Model, ROOT, SPP, SPPF,
                          TransformerBlock, TransformerLayer,
                          check_anchor_order, common, parse_model, yolo,)
from animl.multi_species import (multi_species_detection,)
from animl.pipeline import (from_config, from_paths,)
from animl.pose import (predict, predict_by_camera, predict_viewpoints,)
from animl.reid import (ArcFaceLossAdaptiveMargin, ArcFaceSubCenterDynamic,
                        ArcMarginProduct, ArcMarginProduct_subcenter,
                        ElasticArcFace, GeM, IMAGE_HEIGHT, IMAGE_WIDTH,
                        MiewIdNet, extract_embeddings, heads, l2_norm,
                        load_miew, miewid, weights_init_classifier,
                        weights_init_kaiming,)
from animl.split import (get_animals, get_animals_custom, get_empty,
                         get_empty_custom, train_val_test,)
from animl.test import (main, test_func,)
from animl.train import (init_seed, load_model_checkpoint, main, train_func,
                         validate_func,)
from animl.utils import (FILE, NUM_THREADS, RANK, ROOT, absolute_to_relative,
                         autopad, box_area, box_iou, check_file,
                         check_img_size, check_python, check_requirements,
                         check_suffix, check_version, clean_str, clip_coords,
                         convert_minxywh_to_absxyxy, convert_yolo_to_xywh,
                         copy_attr, de_parallel, demo_boxes, device_count,
                         draw_bounding_boxes, exif_transpose, file_age,
                         file_date, file_size, find_modules, fuse_conv_and_bn,
                         general, get_device, get_image_size, get_latest_run,
                         increment_path, init_seeds, initialize_weights,
                         intersect_dicts, is_parallel, labels_to_class_weights,
                         labels_to_image_weights, letterbox, main,
                         make_divisible, model_info, non_max_suppression,
                         one_cycle, plot_all_bounding_boxes, plot_box,
                         print_args, prune, resample_segments, scale_coords,
                         scale_img, segment2box, segments2boxes, select_device,
                         softmax, sparsity, tensor_to_onnx, time_sync,
                         truncate_float, truncate_float_array, visualization,
                         xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywh, xyxy2xywhn,
                         yolo5,)
from animl.video_processing import (extract_frame_single, extract_frames,)

__all__ = ['ArcFaceLossAdaptiveMargin', 'ArcFaceSubCenterDynamic',
           'ArcMarginProduct', 'ArcMarginProduct_subcenter', 'AutoShape',
           'Bottleneck', 'BottleneckCSP', 'C3', 'C3Ghost', 'C3SPP', 'C3TR',
           'C3x', 'Classify', 'Concat', 'Contract', 'Conv', 'ConvNeXtBase',
           'CrossConv', 'DWConv', 'DWConvTranspose2d', 'Detect',
           'DetectMultiBackend', 'Detections', 'EfficientNet',
           'ElasticArcFace', 'Expand', 'FILE', 'Focus', 'GeM',
           'GhostBottleneck', 'GhostConv', 'IMAGE_EXTENSIONS', 'IMAGE_HEIGHT',
           'IMAGE_WIDTH', 'ImageGenerator', 'MiewGenerator', 'MiewIdNet',
           'Model', 'NUM_THREADS', 'RANK', 'ROOT', 'ResizeWithPadding', 'SPP',
           'SPPF', 'TrainGenerator', 'TransformerBlock', 'TransformerLayer',
           'VALID_EXTENSIONS', 'VIDEO_EXTENSIONS', 'WorkingDirectory',
           'absolute_to_relative', 'active_times',
           'animl_results_to_md_results', 'animl_to_md', 'api', 'autopad',
           'box_area', 'box_iou', 'build_file_manifest', 'check_anchor_order',
           'check_file', 'check_img_size', 'check_python',
           'check_requirements', 'check_suffix', 'check_version',
           'classification', 'classify', 'clean_str', 'clip_coords',
           'collate_fn', 'common', 'convert_minxywh_to_absxyxy',
           'convert_raw_detections', 'convert_yolo_detections',
           'convert_yolo_to_xywh', 'copy_attr', 'csv_converter', 'de_parallel',
           'demo_boxes', 'detect', 'detection',
           'detection_category_id_to_name', 'device_count',
           'draw_bounding_boxes', 'exif_transpose', 'extract_embeddings',
           'extract_frame_single', 'extract_frames', 'file_age', 'file_date',
           'file_management', 'file_size', 'find_modules', 'from_config',
           'from_paths', 'fuse_conv_and_bn', 'general', 'generator',
           'get_animals', 'get_animals_custom', 'get_device', 'get_empty',
           'get_empty_custom', 'get_image_size', 'get_latest_run', 'heads',
           'image_to_tensor', 'increment_path', 'individual_classification',
           'init_seed', 'init_seeds', 'initialize_weights', 'intersect_dicts',
           'is_parallel', 'l2_norm', 'labels_to_class_weights',
           'labels_to_image_weights', 'letterbox', 'link', 'load_class_list',
           'load_classifier', 'load_data', 'load_detector', 'load_miew',
           'load_model_checkpoint', 'main', 'make_divisible',
           'manifest_dataloader', 'matchypatchy', 'miew_embedding', 'miewid',
           'model_architecture', 'model_info', 'models', 'multi_species',
           'multi_species_detection', 'non_max_suppression', 'one_cycle',
           'parse_detections', 'parse_model', 'pipeline',
           'plot_all_bounding_boxes', 'plot_box', 'pose', 'predict',
           'predict_by_camera', 'predict_viewpoints', 'print_args', 'prune',
           'reid', 'reid_dataloader', 'remove_link', 'resample_segments',
           'save_checkpoint', 'save_classifier', 'save_data', 'scale_coords',
           'scale_img', 'segment2box', 'segments2boxes', 'select_device',
           'sequence_classification', 'softmax', 'sort_MD', 'sort_species',
           'sparsity', 'split', 'tensor_to_onnx', 'test', 'test_func',
           'time_sync', 'timelapse', 'train', 'train_dataloader', 'train_func',
           'train_val_test', 'truncate_float', 'truncate_float_array',
           'update_labels', 'utils', 'validate_func', 'video_processing',
           'viewpoint_estimator', 'visualization', 'weights_init_classifier',
           'weights_init_kaiming', 'xyn2xy', 'xywh2xyxy', 'xywhn2xyxy',
           'xyxy2xywh', 'xyxy2xywhn', 'yolo', 'yolo5']
