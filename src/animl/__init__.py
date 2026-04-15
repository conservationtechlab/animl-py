__version__ = '3.2.1'

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
from animl.detection import (detect, load_detector, parse_detections,)
from animl.export import (export_camptrapdp, export_camtrapR, export_coco,
                          export_folders, export_megadetector,
                          export_timelapse, export_yolo, remove_link,
                          update_labels_from_folders,)
from animl.file_management import (IMAGE_EXTENSIONS, VALID_EXTENSIONS,
                                   VIDEO_EXTENSIONS, WorkingDirectory,
                                   active_times, build_file_manifest,
                                   check_file, load_data, load_json, save_data,
                                   save_json, sequence_calculation,)
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
                          TransformerBlock, TransformerLayer, common, download,
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
from animl.utils import (MD_COLORS, MD_LABELS, NUM_THREADS, box_area, box_iou,
                         exif_transpose, general, get_onnx_device,
                         get_torch_device, init_seed, letterbox,
                         non_max_suppression, normalize_bbox,
                         plot_all_bounding_boxes, plot_box, plot_from_file,
                         scale_letterbox, softmax, tensor_to_onnx,
                         visualization,)
from animl.video_processing import (extract_frames, get_frame_as_image,)

__all__ = ['ArcFaceLossAdaptiveMargin', 'ArcFaceSubCenterDynamic',
           'ArcMarginProduct', 'ArcMarginProduct_subcenter', 'AutoShape',
           'BaseModel', 'Bottleneck', 'BottleneckCSP', 'C3', 'C3Ghost',
           'C3SPP', 'C3TR', 'C3x', 'CLASSIFIER', 'CLASS_LIST', 'Classify',
           'Concat', 'Contract', 'Conv', 'ConvNeXtBase', 'CrossConv', 'DWConv',
           'DWConvTranspose2d', 'Detect', 'DetectMultiBackend',
           'DetectionModel', 'Detections', 'EfficientNet', 'ElasticArcFace',
           'Expand', 'FILE', 'Focus', 'GeM', 'GhostBottleneck', 'GhostConv',
           'IMAGE_EXTENSIONS', 'Letterbox', 'MD_COLORS', 'MD_FILENAMES',
           'MD_LABELS', 'MEGADETECTOR', 'MEGADETECTORv5_SIZE',
           'MEGADETECTORv5_STRIDE', 'MIEWID_SIZE', 'ManifestGenerator',
           'MiewIdNet', 'Model', 'NUM_THREADS', 'ROOT',
           'SDZWA_CLASSIFIER_SIZE', 'SPP', 'SPPF', 'Segment', 'TrainGenerator',
           'TransformerBlock', 'TransformerLayer', 'VALID_EXTENSIONS',
           'VIDEO_EXTENSIONS', 'WorkingDirectory', 'active_times', 'box_area',
           'box_iou', 'build_file_manifest', 'check_file', 'classification',
           'classify', 'collate_fn', 'common',
           'compute_batched_distance_matrix', 'compute_distance_matrix',
           'cosine_distance', 'detect', 'detection', 'distance', 'download',
           'download_model', 'euclidean_squared_distance', 'exif_transpose',
           'export', 'export_camptrapdp', 'export_camtrapR', 'export_coco',
           'export_folders', 'export_megadetector', 'export_timelapse',
           'export_yolo', 'extract_frames', 'extract_miew_embeddings',
           'file_management', 'from_config', 'from_paths', 'general',
           'generator', 'get_animals', 'get_empty', 'get_frame_as_image',
           'get_onnx_device', 'get_torch_device', 'image_to_tensor',
           'inference', 'init_seed', 'l2_norm', 'letterbox', 'list_models',
           'load_class_list', 'load_classifier', 'load_classifier_checkpoint',
           'load_data', 'load_detector', 'load_json', 'load_miew',
           'manifest_dataloader', 'miewid', 'model_architecture', 'models',
           'non_max_suppression', 'normalize_bbox', 'parse_detections',
           'parse_model', 'pipeline', 'plot_all_bounding_boxes', 'plot_box',
           'plot_from_file', 'pose', 'predict_viewpoints', 'reid',
           'remove_diagonal', 'remove_link', 'save_classifier', 'save_data',
           'save_json', 'scale_letterbox', 'sequence_calculation',
           'sequence_classification', 'single_classification', 'softmax',
           'split', 'tensor_to_onnx', 'test', 'test_func', 'test_main',
           'train', 'train_dataloader', 'train_func', 'train_main',
           'train_val_test', 'update_labels_from_folders', 'utils',
           'validate_func', 'video_processing', 'viewpoint', 'visualization',
           'yolo']
