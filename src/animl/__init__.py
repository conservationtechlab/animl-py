from animl import api
from animl import classification
from animl import custom_detector
from animl import custom_yolo_run
from animl import detect
from animl import file_management
from animl import generator
from animl import link
from animl import megadetector
from animl import model_architecture
from animl import models
from animl import multi_species
from animl import pipeline
from animl import plot_boxes
from animl import reid
from animl import split
from animl import test
from animl import train
from animl import utils
from animl import video_processing

from animl.api import (MiewGenerator, animl_results_to_md_results, animl_to_md,
                       classify_mp, csv_converter,
                       detection_category_id_to_name, main, matchypatchy,
                       miew_embedding, reid_dataloader, timelapse,
                       viewpoint_estimator,)
from animl.classification import (classify_with_config, load_model,
                                  predict_species, save_model,
                                  sequence_classification,
                                  single_classification, softmax,
                                  tensor_to_onnx,)
from animl.custom_detector import (CustomYOLO,)
from animl.custom_yolo_run import (main_config, main_paths,)
from animl.detect import (absolute2relative, detect_batch, get_image_size,
                          parse_detections, parse_YOLO, process_image,)
from animl.file_management import (IMAGE_EXTENSIONS, VALID_EXTENSIONS,
                                   WorkingDirectory, active_times,
                                   build_file_manifest, check_file, load_data,
                                   save_data,)
from animl.generator import (ImageGenerator, ResizeWithPadding, TrainGenerator,
                             manifest_dataloader, train_dataloader,)
from animl.link import (remove_link, sort_MD, sort_species, update_labels,)
from animl.megadetector import (CONF_DIGITS, COORD_DIGITS, MegaDetector,
                                convert_yolo_to_xywh, truncate_float,
                                truncate_float_array,)
from animl.model_architecture import (ConvNeXtBase, EfficientNet,)
from animl.models import (AutoShape, Bottleneck, BottleneckCSP, C3, C3Ghost,
                          C3SPP, C3TR, C3x, Classify, Concat, Contract, Conv,
                          CrossConv, DWConv, DWConvTranspose2d, Detect,
                          DetectMultiBackend, Detections, Expand, FILE, Focus,
                          GhostBottleneck, GhostConv, Model, ROOT, SPP, SPPF,
                          TransformerBlock, TransformerLayer, autopad,
                          check_anchor_order, common, parse_model, yolo,)
from animl.multi_species import (multi_species_detection,)
from animl.pipeline import (from_config, from_paths,)
from animl.plot_boxes import (demo_boxes, draw_bounding_boxes, main,
                              plot_all_bounding_boxes,)
from animl.reid import (ArcFaceLossAdaptiveMargin, ArcFaceSubCenterDynamic,
                        ArcMarginProduct, ArcMarginProduct_subcenter,
                        ElasticArcFace, GeM, IMAGE_HEIGHT, IMAGE_WIDTH,
                        MiewIdNet, extract_embeddings, heads, l2_norm,
                        load_miew, miewid, weights_init_classifier,
                        weights_init_kaiming,)
from animl.split import (get_animals, get_animals_custom, get_empty,
                         get_empty_custom, train_val_test,)
from animl.test import (main, test_func,)
from animl.train import (init_seed, load_checkpoint, main, train_func,
                         validate,)
from animl.utils import (AUTOINSTALL, CONFIG_DIR, DATASETS_DIR, EarlyStopping,
                         FILE, FONT, LOGGER, ModelEMA, NCOLS, NUM_THREADS,
                         Profile, RANK, ROOT, Timeout, VERBOSE,
                         WorkingDirectory, apply_classifier, augmentations,
                         box_area, box_iou, check_amp, check_dataset,
                         check_file, check_font, check_git_status,
                         check_img_size, check_imshow, check_online,
                         check_python, check_requirements, check_suffix,
                         check_version, check_yaml, clean_str, clip_coords,
                         coco80_to_coco91_class, colorstr, copy_attr,
                         de_parallel, device_count, download, emojis,
                         exif_transpose, file_age, file_date, file_size,
                         find_modules, fitness, fuse_conv_and_bn, general,
                         get_device, get_latest_run, git_describe,
                         gsutil_getsize, imread, imshow, imshow_, imwrite,
                         increment_path, init_seeds, initialize_weights,
                         intersect_dicts, is_ascii, is_chinese, is_colab,
                         is_docker, is_kaggle, is_parallel, is_pip,
                         is_writeable, labels_to_class_weights,
                         labels_to_image_weights, letterbox, make_divisible,
                         methods, model_info, non_max_suppression, one_cycle,
                         print_args, print_mutation, profile, prune,
                         resample_segments, scale_coords, scale_img,
                         segment2box, segments2boxes, select_device,
                         set_logging, sparsity, strip_optimizer, threaded,
                         time_sync, torch_distributed_zero_first, torch_utils,
                         try_except, url2file, user_config_dir, xyn2xy,
                         xywh2xyxy, xywhn2xyxy, xyxy2xywh, xyxy2xywhn,)
from animl.video_processing import (extract_frame_single, extract_frames,)

__all__ = ['AUTOINSTALL', 'ArcFaceLossAdaptiveMargin',
           'ArcFaceSubCenterDynamic', 'ArcMarginProduct',
           'ArcMarginProduct_subcenter', 'AutoShape', 'Bottleneck',
           'BottleneckCSP', 'C3', 'C3Ghost', 'C3SPP', 'C3TR', 'C3x',
           'CONFIG_DIR', 'CONF_DIGITS', 'COORD_DIGITS', 'Classify', 'Concat',
           'Contract', 'Conv', 'ConvNeXtBase', 'CrossConv', 'CustomYOLO',
           'DATASETS_DIR', 'DWConv', 'DWConvTranspose2d', 'Detect',
           'DetectMultiBackend', 'Detections', 'EarlyStopping', 'EfficientNet',
           'ElasticArcFace', 'Expand', 'FILE', 'FONT', 'Focus', 'GeM',
           'GhostBottleneck', 'GhostConv', 'IMAGE_EXTENSIONS', 'IMAGE_HEIGHT',
           'IMAGE_WIDTH', 'ImageGenerator', 'LOGGER', 'MegaDetector',
           'MiewGenerator', 'MiewIdNet', 'Model', 'ModelEMA', 'NCOLS',
           'NUM_THREADS', 'Profile', 'RANK', 'ROOT', 'ResizeWithPadding',
           'SPP', 'SPPF', 'Timeout', 'TrainGenerator', 'TransformerBlock',
           'TransformerLayer', 'VALID_EXTENSIONS', 'VERBOSE',
           'WorkingDirectory', 'absolute2relative', 'active_times',
           'animl_results_to_md_results', 'animl_to_md', 'api',
           'apply_classifier', 'augmentations', 'autopad', 'box_area',
           'box_iou', 'build_file_manifest', 'check_amp', 'check_anchor_order',
           'check_dataset', 'check_file', 'check_font', 'check_git_status',
           'check_img_size', 'check_imshow', 'check_online', 'check_python',
           'check_requirements', 'check_suffix', 'check_version', 'check_yaml',
           'classification', 'classify_mp', 'classify_with_config',
           'clean_str', 'clip_coords', 'coco80_to_coco91_class', 'colorstr',
           'common', 'convert_yolo_to_xywh', 'copy_attr', 'csv_converter',
           'custom_detector', 'custom_yolo_run', 'de_parallel', 'demo_boxes',
           'detect', 'detect_MD_batch', 'detection_category_id_to_name',
           'device_count', 'download', 'draw_bounding_boxes', 'emojis',
           'exif_transpose', 'extract_embeddings', 'extract_frame_single',
           'extract_frames', 'file_age', 'file_date', 'file_management',
           'file_size', 'find_modules', 'fitness', 'from_config', 'from_paths',
           'fuse_conv_and_bn', 'general', 'generator', 'get_animals',
           'get_animals_custom', 'get_device', 'get_empty', 'get_empty_custom',
           'get_image_size', 'get_latest_run', 'git_describe',
           'gsutil_getsize', 'heads', 'imread', 'imshow', 'imshow_', 'imwrite',
           'increment_path', 'init_seed', 'init_seeds', 'initialize_weights',
           'intersect_dicts', 'is_ascii', 'is_chinese', 'is_colab',
           'is_docker', 'is_kaggle', 'is_parallel', 'is_pip', 'is_writeable',
           'l2_norm', 'labels_to_class_weights', 'labels_to_image_weights',
           'letterbox', 'link', 'load_checkpoint', 'load_data', 'load_miew',
           'load_model', 'main', 'main_config', 'main_paths', 'make_divisible',
           'manifest_dataloader', 'matchypatchy', 'megadetector', 'methods',
           'miew_embedding', 'miewid', 'model_architecture', 'model_info',
           'models', 'multi_species', 'multi_species_detection',
           'non_max_suppression', 'one_cycle', 'parse_MD', 'parse_YOLO',
           'parse_model', 'pipeline', 'plot_all_bounding_boxes', 'plot_boxes',
           'predict_species', 'print_args', 'print_mutation', 'process_image',
           'profile', 'prune', 'reid', 'reid_dataloader', 'remove_link',
           'resample_segments', 'save_data', 'save_model', 'scale_coords',
           'scale_img', 'segment2box', 'segments2boxes', 'select_device',
           'sequence_classification', 'set_logging', 'single_classification',
           'softmax', 'sort_MD', 'sort_species', 'sparsity', 'split',
           'strip_optimizer', 'tensor_to_onnx', 'test', 'test_func',
           'threaded', 'time_sync', 'timelapse',
           'torch_distributed_zero_first', 'torch_utils', 'train',
           'train_dataloader', 'train_func', 'train_val_test',
           'truncate_float', 'truncate_float_array', 'try_except',
           'update_labels', 'url2file', 'user_config_dir', 'utils', 'validate',
           'video_processing', 'viewpoint_estimator',
           'weights_init_classifier', 'weights_init_kaiming', 'xyn2xy',
           'xywh2xyxy', 'xywhn2xyxy', 'xyxy2xywh', 'xyxy2xywhn', 'yolo']
