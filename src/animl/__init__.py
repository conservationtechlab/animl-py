__version__ = '3.4.0'

from animl import classification
from animl import count
from animl import detection
from animl import export
from animl import file_management
from animl import generator
from animl import model_architecture
from animl import models
from animl import pipeline
from animl import pose
from animl import reid
from animl import test
from animl import train
from animl import utils
from animl import video_processing

from animl.classification import (classify, load_class_list, load_classifier,
                                  sequence_classification,
                                  single_classification,)
from animl.count import (count_detections, deduplicate,)
from animl.detection import (detect, get_animals, get_empty, load_detector,
                             parse_detections,)
from animl.export import (export_camptrapdp, export_camtrapR, export_coco,
                          export_folders, export_megadetector,
                          export_timelapse, export_train_val_test, export_yolo,
                          remove_link, update_labels_from_folders,)
from animl.file_management import (IMAGE_EXTENSIONS, VALID_EXTENSIONS,
                                   VIDEO_EXTENSIONS, WorkingDirectory,
                                   active_times, build_file_manifest,
                                   check_file, class_list_to_dict, load_data,
                                   load_json, load_yaml, save_data, save_json,
                                   save_yaml, sequence_calculation,)
from animl.generator import (Letterbox, ManifestGenerator, TrainGenerator,
                             collate_fn, image_to_tensor, manifest_dataloader,
                             train_dataloader,)
from animl.model_architecture import (BIOCLIP_CLASSIFIER_SIZE, BioCLIP,
                                      ConvNeXtBase, EfficientNet, GeM,
                                      MD_LABELS, MD_MODELS,
                                      MEGADETECTORv5_SIZE,
                                      MEGADETECTORv5_STRIDE, MIEWID_SIZE,
                                      MiewIdNet, SDZWA_CLASSIFIER_SIZE,)
from animl.models import (AutoShape, BaseModel, Bottleneck, BottleneckCSP, C3,
                          C3Ghost, C3SPP, C3TR, C3x, CLASSIFIER, CLASS_LIST,
                          Classify, Concat, Contract, Conv, CrossConv, DWConv,
                          DWConvTranspose2d, Detect, DetectMultiBackend,
                          DetectionModel, Detections, Expand, FILE, Focus,
                          GhostBottleneck, GhostConv, MD_FILENAMES,
                          MEGADETECTOR, Model, ROOT, SPP, SPPF, Segment,
                          TransformerBlock, TransformerLayer, common, download,
                          download_model, list_models, parse_model, yolo,)
from animl.pipeline import (from_config, from_paths,)
from animl.pose import (predict_viewpoints, viewpoint,)
from animl.reid import (compute_batched_distance_matrix,
                        compute_distance_matrix, cosine_distance,
                        euclidean_squared_distance, extract_miew_embeddings,
                        load_miew, remove_diagonal,)
from animl.test import (test_classifier,)
from animl.train import (load_classifier_checkpoint, save_classifier,
                         train_classifier,)
from animl.utils import (MD_COLORS, NUM_THREADS, animlr, box_area, box_iou,
                         check_exiftool, check_onnx_cuda, check_torch_cuda,
                         exif_transpose, general, get_iou, get_onnx_device,
                         get_torch_device, get_version, init_seed, letterbox,
                         non_max_suppression, normalize_bbox,
                         plot_all_bounding_boxes, plot_box, scale_letterbox,
                         softmax, tensor_to_onnx, visualization,)
from animl.video_processing import (extract_frames, get_frame_as_image,)

__all__ = ['AutoShape', 'BIOCLIP_CLASSIFIER_SIZE', 'BaseModel', 'BioCLIP',
           'Bottleneck', 'BottleneckCSP', 'C3', 'C3Ghost', 'C3SPP', 'C3TR',
           'C3x', 'CLASSIFIER', 'CLASS_LIST', 'Classify', 'Concat', 'Contract',
           'Conv', 'ConvNeXtBase', 'CrossConv', 'DWConv', 'DWConvTranspose2d',
           'Detect', 'DetectMultiBackend', 'DetectionModel', 'Detections',
           'EfficientNet', 'Expand', 'FILE', 'Focus', 'GeM', 'GhostBottleneck',
           'GhostConv', 'IMAGE_EXTENSIONS', 'Letterbox', 'MD_COLORS',
           'MD_FILENAMES', 'MD_LABELS', 'MD_MODELS', 'MEGADETECTOR',
           'MEGADETECTORv5_SIZE', 'MEGADETECTORv5_STRIDE', 'MIEWID_SIZE',
           'ManifestGenerator', 'MiewIdNet', 'Model', 'NUM_THREADS', 'ROOT',
           'SDZWA_CLASSIFIER_SIZE', 'SPP', 'SPPF', 'Segment', 'TrainGenerator',
           'TransformerBlock', 'TransformerLayer', 'VALID_EXTENSIONS',
           'VIDEO_EXTENSIONS', 'WorkingDirectory', 'active_times', 'animlr',
           'box_area', 'box_iou', 'build_file_manifest', 'check_exiftool',
           'check_file', 'check_onnx_cuda', 'check_torch_cuda',
           'class_list_to_dict', 'classification', 'classify', 'collate_fn',
           'common', 'compute_batched_distance_matrix',
           'compute_distance_matrix', 'cosine_distance', 'count',
           'count_detections', 'deduplicate', 'detect', 'detection',
           'download', 'download_model', 'euclidean_squared_distance',
           'exif_transpose', 'export', 'export_camptrapdp', 'export_camtrapR',
           'export_coco', 'export_folders', 'export_megadetector',
           'export_timelapse', 'export_train_val_test', 'export_yolo',
           'extract_frames', 'extract_miew_embeddings', 'file_management',
           'from_config', 'from_paths', 'general', 'generator', 'get_animals',
           'get_empty', 'get_frame_as_image', 'get_iou', 'get_onnx_device',
           'get_torch_device', 'get_version', 'image_to_tensor', 'init_seed',
           'letterbox', 'list_models', 'load_class_list', 'load_classifier',
           'load_classifier_checkpoint', 'load_data', 'load_detector',
           'load_json', 'load_miew', 'load_yaml', 'manifest_dataloader',
           'model_architecture', 'models', 'non_max_suppression',
           'normalize_bbox', 'parse_detections', 'parse_model', 'pipeline',
           'plot_all_bounding_boxes', 'plot_box', 'pose', 'predict_viewpoints',
           'reid', 'remove_diagonal', 'remove_link', 'save_classifier',
           'save_data', 'save_json', 'save_yaml', 'scale_letterbox',
           'sequence_calculation', 'sequence_classification',
           'single_classification', 'softmax', 'tensor_to_onnx', 'test',
           'test_classifier', 'train', 'train_classifier', 'train_dataloader',
           'update_labels_from_folders', 'utils', 'video_processing',
           'viewpoint', 'visualization', 'yolo']
