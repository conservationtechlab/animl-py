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
import yaml


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


PATCHES = [
    'animl.pipeline.file_management.WorkingDirectory',
    'animl.pipeline.file_management.build_file_manifest',
    'animl.pipeline.file_management.check_file',
    'animl.pipeline.file_management.load_data',
    'animl.pipeline.file_management.save_data',
    'animl.pipeline.video_processing.extract_frames',
    'animl.pipeline.detection.load_detector',
    'animl.pipeline.detection.detect',
    'animl.pipeline.detection.parse_detections',
    'animl.pipeline.split.get_animals',
    'animl.pipeline.split.get_empty',
    'animl.pipeline.classification.load_classifier',
    'animl.pipeline.classification.classify',
    'animl.pipeline.classification.single_classification',
    'animl.pipeline.classification.sequence_classification',
    'animl.pipeline.export.export_folders',
    'animl.pipeline.visualization.plot_all_bounding_boxes',
]


def _apply_all_patches(test_func):
    """Stack all PATCHES as decorators (bottom-up order)."""
    for target in reversed(PATCHES):
        test_func = patch(target)(test_func)
    return test_func


class TestFromPaths(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image_dir = tempfile.mkdtemp()
        cls.detector_file = str(Path(cls.image_dir) / 'detector.pt')
        cls.classifier_file = str(Path(cls.image_dir) / 'classifier.pt')
        cls.classlist_file = str(Path(cls.image_dir) / 'classes.csv')
        Path(cls.detector_file).touch()
        Path(cls.classifier_file).touch()

        cls.mock_manifest = pd.DataFrame({
            'filepath': ['a.jpg', 'b.jpg'],
            'frame': [0, 0],
            'station': ['cam1', 'cam1'],
            'datetime': ['2023-01-01 10:00:00', '2023-01-01 10:01:00'],
            'extension': ['.jpg', '.jpg'],
        })
        cls.mock_animals = cls.mock_manifest.copy()
        cls.mock_animals['category'] = 1
        cls.mock_animals['conf'] = 0.9
        cls.mock_animals['bbox_x'] = 0.1
        cls.mock_animals['bbox_y'] = 0.1
        cls.mock_animals['bbox_w'] = 0.2
        cls.mock_animals['bbox_h'] = 0.3

        cls.mock_empty = pd.DataFrame(columns=cls.mock_manifest.columns)
        cls.mock_predictions = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        cls.mock_results = cls.mock_animals.copy()
        cls.mock_results['prediction'] = 'cat'
        cls.mock_results['confidence'] = 0.9
        cls.mock_class_list = pd.DataFrame({'class': ['cat', 'dog', 'bird']})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.image_dir)

    def _setup_mocks(self, mock_wd_cls, mock_build, mock_check, mock_load,
                     mock_save, mock_extract, mock_load_det, mock_detect,
                     mock_parse, mock_get_animals, mock_get_empty,
                     mock_load_cls, mock_classify, mock_single, mock_sequence,
                     mock_export, mock_plot):
        """Configure all mocks with sensible return values."""
        mock_wd_instance = _make_wd_mock()
        mock_wd_cls.return_value = mock_wd_instance

        mock_build.return_value = self.mock_manifest
        mock_check.return_value = False
        mock_load.return_value = self.mock_manifest
        mock_extract.return_value = self.mock_manifest

        mock_load_det.return_value = MagicMock()
        mock_detect.return_value = []
        mock_parse.return_value = self.mock_manifest

        mock_get_animals.return_value = self.mock_animals
        mock_get_empty.return_value = self.mock_empty

        mock_model = MagicMock()
        mock_load_cls.return_value = (mock_model, self.mock_class_list)
        mock_classify.return_value = (self.mock_predictions, [])
        mock_single.return_value = self.mock_results
        mock_sequence.return_value = self.mock_results
        mock_export.return_value = self.mock_results

        return mock_wd_instance

    def test_returns_dataframe(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
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
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file)
            mock_build.assert_called_once()

    def test_calls_extract_frames(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file)
            mock_extract.assert_called_once()

    def test_calls_load_detector_for_pt_file(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file)
            mock_load_det.assert_called_once()
            call_kwargs = mock_load_det.call_args
            self.assertEqual(call_kwargs.kwargs.get('model_type') or call_kwargs.args[1], 'mdv5')

    def test_calls_load_detector_for_onnx_file(self):
        detector_onnx = str(Path(self.image_dir) / 'detector.onnx')
        Path(detector_onnx).touch()
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, detector_onnx,
                       self.classifier_file, self.classlist_file)
            mock_load_det.assert_called_once()
            args, kwargs = mock_load_det.call_args
            # onnx branch passes positional arg: load_detector(file, "onnx", ...)
            if args and len(args) > 1:
                self.assertEqual(args[1], 'onnx')
            else:
                self.assertEqual(kwargs.get('model_type'), 'onnx')

    def test_calls_detect(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file)
            mock_detect.assert_called_once()

    def test_calls_parse_detections(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file)
            mock_parse.assert_called_once()

    def test_calls_load_classifier(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file)
            mock_load_cls.assert_called_once()

    def test_calls_classify(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file)
            mock_classify.assert_called_once()

    def test_single_classification_called_by_default(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
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
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file, sequence=True)
            mock_sequence.assert_called_once()
            mock_single.assert_not_called()

    def test_detect_only_returns_early(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            result = from_paths(self.image_dir, self.detector_file,
                                self.classifier_file, self.classlist_file,
                                detect_only=True)
            mock_load_cls.assert_not_called()
            mock_classify.assert_not_called()
            self.assertIsInstance(result, pd.DataFrame)

    def test_sort_calls_export_folders(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file, sort=True)
            mock_export.assert_called_once()

    def test_sort_false_does_not_call_export(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file, sort=False)
            mock_export.assert_not_called()

    def test_visualize_calls_plot_bounding_boxes(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file, visualize=True)
            mock_plot.assert_called_once()

    def test_visualize_false_does_not_plot(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file, visualize=False)
            mock_plot.assert_not_called()

    def test_save_data_called(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file)
            mock_save.assert_called_once()

    def test_uses_cached_detections_when_check_file_true(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            mock_check.return_value = True
            mock_load.return_value = self.mock_manifest
            from animl.pipeline import from_paths
            from_paths(self.image_dir, self.detector_file,
                       self.classifier_file, self.classlist_file)
            mock_load.assert_called()
            mock_detect.assert_not_called()

    def test_missing_image_dir_raises(self):
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            mock_wd_cls.side_effect = FileNotFoundError("Directory not found")
            from animl.pipeline import from_paths
            with self.assertRaises(FileNotFoundError):
                from_paths('/nonexistent/dir', self.detector_file,
                           self.classifier_file, self.classlist_file)


def _write_config(tmp_dir, detector_file, classifier_file, overrides=None):
    """Write a minimal valid YAML config file and return its path."""
    cfg = {
        'image_dir': tmp_dir,
        'detector_file': detector_file,
        'classifier_file': classifier_file,
        'device': 'cpu',
        'exif': False,
    }
    if overrides:
        cfg.update(overrides)
    config_path = str(Path(tmp_dir) / 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)
    return config_path


class TestFromConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image_dir = tempfile.mkdtemp()
        cls.detector_file = str(Path(cls.image_dir) / 'detector.pt')
        cls.classifier_file = str(Path(cls.image_dir) / 'classifier.pt')
        cls.classlist_file = str(Path(cls.image_dir) / 'classes.csv')
        Path(cls.detector_file).touch()
        Path(cls.classifier_file).touch()

        cls.mock_manifest = pd.DataFrame({
            'filepath': ['a.jpg', 'b.jpg'],
            'frame': [0, 0],
            'station': ['cam1', 'cam1'],
            'datetime': ['2023-01-01 10:00:00', '2023-01-01 10:01:00'],
            'extension': ['.jpg', '.jpg'],
        })
        cls.mock_animals = cls.mock_manifest.copy()
        cls.mock_animals['category'] = 1
        cls.mock_animals['conf'] = 0.9
        cls.mock_animals['bbox_x'] = 0.1
        cls.mock_animals['bbox_y'] = 0.1
        cls.mock_animals['bbox_w'] = 0.2
        cls.mock_animals['bbox_h'] = 0.3

        cls.mock_empty = pd.DataFrame(columns=cls.mock_manifest.columns)
        cls.mock_predictions = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        cls.mock_results = cls.mock_animals.copy()
        cls.mock_results['prediction'] = 'cat'
        cls.mock_results['confidence'] = 0.9
        cls.mock_class_list = pd.DataFrame({'class': ['cat', 'dog', 'bird']})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.image_dir)

    def _setup_mocks(self, mock_wd_cls, mock_build, mock_check, mock_load,
                     mock_save, mock_extract, mock_load_det, mock_detect,
                     mock_parse, mock_get_animals, mock_get_empty,
                     mock_load_cls, mock_classify, mock_single, mock_sequence,
                     mock_export, mock_plot):
        mock_wd_instance = _make_wd_mock()
        mock_wd_cls.return_value = mock_wd_instance

        mock_build.return_value = self.mock_manifest
        mock_check.return_value = False
        mock_load.return_value = self.mock_manifest
        mock_extract.return_value = self.mock_manifest

        mock_load_det.return_value = MagicMock()
        mock_detect.return_value = []
        mock_parse.return_value = self.mock_manifest

        mock_get_animals.return_value = self.mock_animals
        mock_get_empty.return_value = self.mock_empty

        mock_model = MagicMock()
        mock_load_cls.return_value = (mock_model, self.mock_class_list)
        mock_classify.return_value = (self.mock_predictions, [])
        mock_single.return_value = self.mock_results
        mock_sequence.return_value = self.mock_results
        mock_export.return_value = self.mock_results

        return mock_wd_instance

    def test_returns_dataframe(self):
        config_path = _write_config(self.image_dir, self.detector_file,
                                    self.classifier_file,
                                    overrides={'empty_class': ''})
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_config
            result = from_config(config_path)
            self.assertIsInstance(result, pd.DataFrame)

    def test_loads_config_file(self):
        config_path = _write_config(self.image_dir, self.detector_file,
                                    self.classifier_file,
                                    overrides={'empty_class': ''})
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_config
            # should not raise
            from_config(config_path)

    def test_invalid_config_path_raises(self):
        from animl.pipeline import from_config
        with self.assertRaises(Exception):
            from_config('/nonexistent/config.yaml')

    def test_station_dir_adds_station_column(self):
        config_path = _write_config(self.image_dir, self.detector_file,
                                    self.classifier_file,
                                    overrides={'station_dir': -1, 'empty_class': ''})
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_config
            from_config(config_path)
            mock_sequence.assert_called_once()

    def test_no_station_dir_calls_single_classification(self):
        config_path = _write_config(self.image_dir, self.detector_file,
                                    self.classifier_file)
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_config
            from_config(config_path)
            mock_single.assert_called_once()
            mock_sequence.assert_not_called()

    def test_sort_true_calls_export(self):
        config_path = _write_config(self.image_dir, self.detector_file,
                                    self.classifier_file,
                                    overrides={'sort': True})
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_config
            from_config(config_path)
            mock_export.assert_called_once()

    def test_visualize_true_calls_plot(self):
        config_path = _write_config(self.image_dir, self.detector_file,
                                    self.classifier_file,
                                    overrides={'visualize': True, 'sort': False})
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_config
            from_config(config_path)
            mock_plot.assert_called_once()

    def test_uses_cached_detections(self):
        config_path = _write_config(self.image_dir, self.detector_file,
                                    self.classifier_file)
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            mock_check.return_value = True
            mock_load.return_value = self.mock_manifest
            from animl.pipeline import from_config
            from_config(config_path)
            mock_load.assert_called()
            mock_detect.assert_not_called()

    def test_device_passed_to_detector(self):
        config_path = _write_config(self.image_dir, self.detector_file,
                                    self.classifier_file,
                                    overrides={'device': 'cpu'})
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_config
            from_config(config_path)
            mock_load_det.assert_called_once()
            _, kwargs = mock_load_det.call_args
            self.assertEqual(kwargs.get('device'), 'cpu')

    def test_custom_batch_size(self):
        config_path = _write_config(self.image_dir, self.detector_file,
                                    self.classifier_file,
                                    overrides={'batch_size': 8})
        with patch('animl.pipeline.file_management.WorkingDirectory') as mock_wd_cls, \
             patch('animl.pipeline.file_management.build_file_manifest') as mock_build, \
             patch('animl.pipeline.file_management.check_file') as mock_check, \
             patch('animl.pipeline.file_management.load_data') as mock_load, \
             patch('animl.pipeline.file_management.save_data') as mock_save, \
             patch('animl.pipeline.video_processing.extract_frames') as mock_extract, \
             patch('animl.pipeline.detection.load_detector') as mock_load_det, \
             patch('animl.pipeline.detection.detect') as mock_detect, \
             patch('animl.pipeline.detection.parse_detections') as mock_parse, \
             patch('animl.pipeline.split.get_animals') as mock_get_animals, \
             patch('animl.pipeline.split.get_empty') as mock_get_empty, \
             patch('animl.pipeline.classification.load_classifier') as mock_load_cls, \
             patch('animl.pipeline.classification.classify') as mock_classify, \
             patch('animl.pipeline.classification.single_classification') as mock_single, \
             patch('animl.pipeline.classification.sequence_classification') as mock_sequence, \
             patch('animl.pipeline.export.export_folders') as mock_export, \
             patch('animl.pipeline.visualization.plot_all_bounding_boxes') as mock_plot:
            self._setup_mocks(mock_wd_cls, mock_build, mock_check, mock_load,
                              mock_save, mock_extract, mock_load_det, mock_detect,
                              mock_parse, mock_get_animals, mock_get_empty,
                              mock_load_cls, mock_classify, mock_single, mock_sequence,
                              mock_export, mock_plot)
            from animl.pipeline import from_config
            from_config(config_path)
            mock_detect.assert_called_once()
            _, kwargs = mock_detect.call_args
            self.assertEqual(kwargs.get('batch_size'), 8)


if __name__ == '__main__':
    unittest.main()
