# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
utils/initialization

Utils for MD/Yolo

"""
from animl.utils import augmentations
from animl.utils import dataloaders
from animl.utils import general
from animl.utils import torch_utils

from animl.utils.augmentations import (Albumentations, augment_hsv, bbox_iou,
                                       box_candidates, copy_paste, cutout,
                                       hist_equalize, letterbox, mixup,
                                       random_perspective, replicate,)
from animl.utils.dataloaders import (BAR_FORMAT, HELP_URL, IMG_FORMATS,
                                     InfiniteDataLoader, LOCAL_RANK,
                                     LoadImages, LoadImagesAndLabels,
                                     LoadStreams, LoadWebcam, VID_FORMATS,
                                     autosplit, create_dataloader,
                                     create_folder, dataset_stats, exif_size,
                                     exif_transpose, extract_boxes,
                                     flatten_recursive, get_hash,
                                     img2label_paths, verify_image_label,)
from animl.utils.general import (AUTOINSTALL, CONFIG_DIR, DATASETS_DIR, FILE,
                                 FONT, LOGGER, NCOLS, NUM_THREADS, Profile,
                                 RANK, ROOT, Timeout, VERBOSE,
                                 WorkingDirectory, apply_classifier, box_area,
                                 box_iou, check_amp, check_dataset, check_file,
                                 check_font, check_git_status, check_img_size,
                                 check_imshow, check_online, check_python,
                                 check_requirements, check_suffix,
                                 check_version, check_yaml, clean_str,
                                 clip_coords, coco80_to_coco91_class, colorstr,
                                 download, emojis, file_age, file_date,
                                 file_size, fitness, get_latest_run,
                                 git_describe, gsutil_getsize, imread, imshow,
                                 imshow_, imwrite, increment_path, init_seeds,
                                 intersect_dicts, is_ascii, is_chinese,
                                 is_colab, is_docker, is_kaggle, is_pip,
                                 is_writeable, labels_to_class_weights,
                                 labels_to_image_weights, make_divisible,
                                 methods, non_max_suppression, one_cycle,
                                 print_args, print_mutation, resample_segments,
                                 scale_coords, segment2box, segments2boxes,
                                 set_logging, strip_optimizer, threaded,
                                 try_except, url2file, user_config_dir, xyn2xy,
                                 xywh2xyxy, xywhn2xyxy, xyxy2xywh, xyxy2xywhn,)
from animl.utils.torch_utils import (EarlyStopping, ModelEMA, copy_attr,
                                     de_parallel, device_count, find_modules,
                                     fuse_conv_and_bn, get_device,
                                     initialize_weights, is_parallel,
                                     model_info, profile, prune, scale_img,
                                     select_device, sparsity, time_sync,
                                     torch_distributed_zero_first,)

__all__ = ['AUTOINSTALL', 'Albumentations', 'BAR_FORMAT', 'CONFIG_DIR',
           'DATASETS_DIR', 'EarlyStopping', 'FILE', 'FONT', 'HELP_URL',
           'IMG_FORMATS', 'InfiniteDataLoader', 'LOCAL_RANK', 'LOGGER',
           'LoadImages', 'LoadImagesAndLabels', 'LoadStreams', 'LoadWebcam',
           'ModelEMA', 'NCOLS', 'NUM_THREADS', 'Profile', 'RANK', 'ROOT',
           'Timeout', 'VERBOSE', 'VID_FORMATS', 'WorkingDirectory',
           'apply_classifier', 'augment_hsv', 'augmentations', 'autosplit',
           'bbox_iou', 'box_area', 'box_candidates', 'box_iou', 'check_amp',
           'check_dataset', 'check_file', 'check_font', 'check_git_status',
           'check_img_size', 'check_imshow', 'check_online', 'check_python',
           'check_requirements', 'check_suffix', 'check_version', 'check_yaml',
           'clean_str', 'clip_coords', 'coco80_to_coco91_class', 'colorstr',
           'copy_attr', 'copy_paste', 'create_dataloader', 'create_folder',
           'cutout', 'dataloaders', 'dataset_stats', 'de_parallel',
           'device_count', 'download', 'emojis', 'exif_size', 'exif_transpose',
           'extract_boxes', 'file_age', 'file_date', 'file_size',
           'find_modules', 'fitness', 'flatten_recursive', 'fuse_conv_and_bn',
           'general', 'get_device', 'get_hash', 'get_latest_run',
           'git_describe', 'gsutil_getsize', 'hist_equalize',
           'img2label_paths', 'imread', 'imshow', 'imshow_', 'imwrite',
           'increment_path', 'init_seeds', 'initialize_weights',
           'intersect_dicts', 'is_ascii', 'is_chinese', 'is_colab',
           'is_docker', 'is_kaggle', 'is_parallel', 'is_pip', 'is_writeable',
           'labels_to_class_weights', 'labels_to_image_weights', 'letterbox',
           'make_divisible', 'methods', 'mixup', 'model_info',
           'non_max_suppression', 'one_cycle', 'print_args', 'print_mutation',
           'profile', 'prune', 'random_perspective', 'replicate',
           'resample_segments', 'scale_coords', 'scale_img', 'segment2box',
           'segments2boxes', 'select_device', 'set_logging', 'sparsity',
           'strip_optimizer', 'threaded', 'time_sync',
           'torch_distributed_zero_first', 'torch_utils', 'try_except',
           'url2file', 'user_config_dir', 'verify_image_label', 'xyn2xy',
           'xywh2xyxy', 'xywhn2xyxy', 'xyxy2xywh', 'xyxy2xywhn']
