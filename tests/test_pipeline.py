"""
Unit tests for animl/pipeline.py

@ Kyra Swanson 2023
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_wd_mock():
    """Create a WorkingDirectory mock instance with required attributes."""
    mock_wd = MagicMock()
    mock_wd.filemanifest = 'manifest.csv'
    mock_wd.imageframes = 'imageframes.csv'
    mock_wd.detections = 'detections.csv'
    mock_wd.mdraw = 'mdraw.json'
    mock_wd.predictions = 'predictions.npy'
    mock_wd.results = 'results.csv'
    mock_wd.linkdir = 'links/'
    mock_wd.visdir = 'vis/'
    return mock_wd


def _make_parse_detections_result():
    """Return a DataFrame matching what parse_detections() actually returns."""
    return pd.DataFrame({
        'filepath': ['a.jpg', 'a.jpg'],
        'frame': [0, 0],
        'max_detection_conf': [0.9, 0.9],
        'category': [1, 1],
        'category_label': ['animal', 'animal'],
        'conf': [0.9, 0.85],
        'bbox_x': [0.1, 0.5],
        'bbox_y': [0.1, 0.5],
        'bbox_w': [0.2, 0.2],
        'bbox_h': [0.3, 0.3],
    })


def _make_classification_results():
    """Return mock classification results."""
    return pd.DataFrame({
        'filepath': ['a.jpg', 'a.jpg'],
        'frame': [0, 0],
        'category': [1, 1],
        'category_label': ['animal', 'animal'],
        'prediction': ['cat', 'dog'],
        'confidence': [0.95, 0.88],
    })


class TestFromPaths(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image_dir = tempfile.mkdtemp()
        cls.detector_file = str(Path(cls.image_dir) / 'detector.pt')
        cls.classifier_file = str(Path(cls.image_dir) / 'classifier.pt')
        cls.classlist_file = str(Path(cls.image_dir) / 'classes.csv')
        Path(cls.detector_file).touch()
        Path(cls.classifier_file).touch()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.image_dir)

    def _setup_mocks(self, mocks_dict):
        """Configure all mocks with appropriate return values."""
        mock_wd_cls = mocks_dict['mock_wd_cls']
        mock_build = mocks_dict['mock_build']
        mock_check = mocks_dict['mock_check']
        mock_load = mocks_dict['mock_load']
        mock_save = mocks_dict['mock_save']
        mock_extract = mocks_dict['mock_extract']
        mock_load_det = mocks_dict['mock_load_det']
        mock_detect = mocks_dict['mock_detect']
        mock_parse = mocks_dict['mock_parse']
        mock_load_cls = mocks_dict['mock_load_cls']
        mock_classify = mocks_dict['mock_classify']
        mock_single = mocks_dict['mock_single']
        mock_sequence = mocks_dict['mock_sequence']
        mock_export = mocks_dict['mock_export']
        mock_plot = mocks_dict['mock_plot']
        mock_class_list_to_dict = mocks_dict.get('mock_class_list_to_dict')

        # Setup WorkingDirectory
        mock_wd_instance = _make_wd_mock()
        mock_wd_cls.return_value = mock_wd_instance

        # Setup file management
        mock_manifest = pd.DataFrame({
            'filepath': ['a.jpg', 'b.jpg'],
            'frame': [0, 0],
        })
        mock_build.return_value = mock_manifest
        mock_check.return_value = False
        mock_load.return_value = mock_manifest
        mock_extract.return_value = mock_manifest

        # Setup detection
        mock_load_det.return_value = MagicMock()
        mock_detect.return_value = ([], [])  # results, failed_files
        
        # Setup parse_detections - CRITICAL: must have all required columns
        mock_parse.return_value = _make_parse_detections_result()

        # Setup classification
        if mock_class_list_to_dict is not None:
            mock_class_list_to_dict.return_value = {0: 'empty', 1: 'animal'}
        
        mock_class_list = pd.DataFrame({'class': ['empty', 'animal']})
        mock_load_cls.return_value = (MagicMock(), mock_class_list)
        mock_classify.return_value = (np.array([[0.9, 0.1], [0.1, 0.9]]), [])
        mock_single.return_value = _make_classification_results()
        mock_sequence.return_value = _make_classification_results()

        # Setup export
        mock_export.return_value = _make_classification_results()

    def test_returns_dataframe(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.file_management.class_list_to_dict') as mock_class_list_to_dict, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            
            self._setup_mocks({
                'mock_wd_cls': mock_wd_cls,
                'mock_build': mock_build,
                'mock_check': mock_check,
                'mock_load': mock_load,
                'mock_save': mock_save,
                'mock_extract': mock_extract,
                'mock_load_det': mock_load_det,
                'mock_detect': mock_detect,
                'mock_parse': mock_parse,
                'mock_load_cls': mock_load_cls,
                'mock_classify': mock_classify,
                'mock_single': mock_single,
                'mock_sequence': mock_sequence,
                'mock_export': mock_export,
                'mock_plot': mock_plot,
                'mock_class_list_to_dict': mock_class_list_to_dict,
            })
            
            from animl.pipeline import from_paths
            result = from_paths(self.image_dir, self.detector_file,
                                self.classifier_file, self.classlist_file)
            self.assertIsInstance(result, pd.DataFrame)

    def test_calls_build_file_manifest(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.file_management.class_list_to_dict') as mock_class_list_to_dict, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            
            self._setup_mocks({
                'mock_wd_cls': mock_wd_cls,
                'mock_build': mock_build,
                'mock_check': mock_check,
                'mock_load': mock_load,
                'mock_save': mock_save,
                'mock_extract': mock_extract,
                'mock_load_det': mock_load_det,
                'mock_detect': mock_detect,
                'mock_parse': mock_parse,
                'mock_load_cls': mock_load_cls,
                'mock_classify': mock_classify,
                'mock_single': mock_single,
                'mock_sequence': mock_sequence,
                'mock_export': mock_export,
                'mock_plot': mock_plot,
                'mock_class_list_to_dict': mock_class_list_to_dict,
            })
            
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file)
            mock_build.assert_called_once()

    def test_calls_load_detector_for_pt_file(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.file_management.class_list_to_dict') as mock_class_list_to_dict, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            
            self._setup_mocks({
                'mock_wd_cls': mock_wd_cls,
                'mock_build': mock_build,
                'mock_check': mock_check,
                'mock_load': mock_load,
                'mock_save': mock_save,
                'mock_extract': mock_extract,
                'mock_load_det': mock_load_det,
                'mock_detect': mock_detect,
                'mock_parse': mock_parse,
                'mock_load_cls': mock_load_cls,
                'mock_classify': mock_classify,
                'mock_single': mock_single,
                'mock_sequence': mock_sequence,
                'mock_export': mock_export,
                'mock_plot': mock_plot,
                'mock_class_list_to_dict': mock_class_list_to_dict,
            })
            
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file)
            mock_load_det.assert_called_once()

    def test_single_classification_called_by_default(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.file_management.class_list_to_dict') as mock_class_list_to_dict, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            
            self._setup_mocks({
                'mock_wd_cls': mock_wd_cls,
                'mock_build': mock_build,
                'mock_check': mock_check,
                'mock_load': mock_load,
                'mock_save': mock_save,
                'mock_extract': mock_extract,
                'mock_load_det': mock_load_det,
                'mock_detect': mock_detect,
                'mock_parse': mock_parse,
                'mock_load_cls': mock_load_cls,
                'mock_classify': mock_classify,
                'mock_single': mock_single,
                'mock_sequence': mock_sequence,
                'mock_export': mock_export,
                'mock_plot': mock_plot,
                'mock_class_list_to_dict': mock_class_list_to_dict,
            })
            
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file, sequence=False)
            mock_single.assert_called_once()
            mock_sequence.assert_not_called()

    def test_sequence_classification_called_when_flag_set(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.file_management.class_list_to_dict') as mock_class_list_to_dict, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            
            self._setup_mocks({
                'mock_wd_cls': mock_wd_cls,
                'mock_build': mock_build,
                'mock_check': mock_check,
                'mock_load': mock_load,
                'mock_save': mock_save,
                'mock_extract': mock_extract,
                'mock_load_det': mock_load_det,
                'mock_detect': mock_detect,
                'mock_parse': mock_parse,
                'mock_load_cls': mock_load_cls,
                'mock_classify': mock_classify,
                'mock_single': mock_single,
                'mock_sequence': mock_sequence,
                'mock_export': mock_export,
                'mock_plot': mock_plot,
                'mock_class_list_to_dict': mock_class_list_to_dict,
            })
            
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file, sequence=True)
            mock_sequence.assert_called_once()
            mock_single.assert_not_called()

    def test_sort_calls_export_folders(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.file_management.class_list_to_dict') as mock_class_list_to_dict, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            
            self._setup_mocks({
                'mock_wd_cls': mock_wd_cls,
                'mock_build': mock_build,
                'mock_check': mock_check,
                'mock_load': mock_load,
                'mock_save': mock_save,
                'mock_extract': mock_extract,
                'mock_load_det': mock_load_det,
                'mock_detect': mock_detect,
                'mock_parse': mock_parse,
                'mock_load_cls': mock_load_cls,
                'mock_classify': mock_classify,
                'mock_single': mock_single,
                'mock_sequence': mock_sequence,
                'mock_export': mock_export,
                'mock_plot': mock_plot,
                'mock_class_list_to_dict': mock_class_list_to_dict,
            })
            
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file, sort=True)
            mock_export.assert_called_once()

    def test_visualize_calls_plot_bounding_boxes(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.file_management.class_list_to_dict') as mock_class_list_to_dict, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            
            self._setup_mocks({
                'mock_wd_cls': mock_wd_cls,
                'mock_build': mock_build,
                'mock_check': mock_check,
                'mock_load': mock_load,
                'mock_save': mock_save,
                'mock_extract': mock_extract,
                'mock_load_det': mock_load_det,
                'mock_detect': mock_detect,
                'mock_parse': mock_parse,
                'mock_load_cls': mock_load_cls,
                'mock_classify': mock_classify,
                'mock_single': mock_single,
                'mock_sequence': mock_sequence,
                'mock_export': mock_export,
                'mock_plot': mock_plot,
                'mock_class_list_to_dict': mock_class_list_to_dict,
            })
            
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file, visualize=True)
            mock_plot.assert_called_once()


if __name__ == '__main__':
    unittest.main()
