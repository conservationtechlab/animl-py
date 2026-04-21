"""
Unit tests for animl/detection.py

@ Kyra Swanson 2023
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from animl.detection import (
    _convert_onnx_detections as convert_onnx_detections,
    _convert_yolo_detections as convert_yolo_detections,
    _save_detection_checkpoint as save_detection_checkpoint,
    parse_detections,
    load_detector,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_image_tensor(h=640, w=640, batch=1):
    """Return a (batch, 3, h, w) float tensor."""
    return torch.zeros(batch, 3, h, w)

def _make_detection_result(filepath='a.jpg', frame=0, max_conf=0.9, detections=None):
    """Return a single result dict as produced by detect()."""
    if detections is None:
        detections = [{
            'bbox_x': 0.1, 'bbox_y': 0.1,
            'bbox_w': 0.2, 'bbox_h': 0.3,
            'conf': 0.9, 'category': 1
        }]
    return {
        'filepath': filepath,
        'frame': frame,
        'max_detection_conf': max_conf,
        'detections': detections,
    }


# ---------------------------------------------------------------------------
# load_detector
# ---------------------------------------------------------------------------

class TestLoadDetector(unittest.TestCase):

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_detector('/nonexistent/model.pt', 'MDV5')

    def test_unsupported_model_type_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix='.pt') as f:
            result = load_detector(f.name, 'UNSUPPORTED')
            self.assertIsNone(result)



class TestSaveDetectionCheckpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_creates_checkpoint_file(self):
        path = str(Path(self.tmp_dir) / 'checkpoint.json')
        save_detection_checkpoint(path, [{'image': 'a.jpg'}])
        self.assertTrue(Path(path).exists())

    def test_checkpoint_contains_images_key(self):
        path = str(Path(self.tmp_dir) / 'checkpoint2.json')
        results = [{'image': 'a.jpg', 'detections': []}]
        save_detection_checkpoint(path, results)
        with open(path) as f:
            data = json.load(f)
        self.assertIn('images', data)
        self.assertEqual(data['images'], results)

    def test_overwrites_existing_checkpoint(self):
        path = str(Path(self.tmp_dir) / 'checkpoint3.json')
        save_detection_checkpoint(path, [{'image': 'a.jpg'}])
        save_detection_checkpoint(path, [{'image': 'b.jpg'}])
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data['images'][0]['image'], 'b.jpg')

    def test_none_path_raises(self):
        with self.assertRaises(AssertionError):
            save_detection_checkpoint(None, [])

# ---------------------------------------------------------------------------
# convert_onnx_detections
# ---------------------------------------------------------------------------

class TestConvertOnnxDetections(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image_tensors = _make_image_tensor()
        cls.image_paths = ['a.jpg']
        cls.image_frames = [0]
        cls.image_sizes = np.array([[480, 640]])

    def _make_pred(self, num_detections=2, conf_val=0.9):
        """Build a fake ONNX prediction array (N x 6: x1,y1,x2,y2,conf,class)."""
        pred = np.zeros((num_detections, 6), dtype=np.float32)
        pred[:, 2] = 0.5   # x2
        pred[:, 3] = 0.5   # y2
        pred[:, 4] = conf_val
        pred[:, 5] = 0     # class 0
        return pred

    def test_returns_list(self):
        preds = [self._make_pred()]
        result = convert_onnx_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes, letterbox=False)
        self.assertIsInstance(result, list)

    def test_result_length_matches_input(self):
        preds = [self._make_pred()]
        result = convert_onnx_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes, letterbox=False)
        self.assertEqual(len(result), 1)

    def test_result_has_required_keys(self):
        preds = [self._make_pred()]
        result = convert_onnx_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes, letterbox=False)
        for key in ('filepath', 'frame', 'max_detection_conf', 'detections'):
            self.assertIn(key, result[0])

    def test_no_detections_returns_empty_list(self):
        preds = [np.zeros((0, 6), dtype=np.float32)]
        result = convert_onnx_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes, letterbox=False)
        self.assertEqual(result[0]['detections'], [])
        self.assertIsNone(result[0]['max_detection_conf'])

    def test_detection_keys_present(self):
        preds = [self._make_pred(num_detections=1)]
        result = convert_onnx_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes, letterbox=False)
        detection = result[0]['detections']
        if len(detection) > 0:
            for key in ('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'conf', 'category'):
                self.assertIn(key, detection[0])

    def test_filepath_preserved(self):
        preds = [self._make_pred()]
        result = convert_onnx_detections(preds, self.image_tensors, ['my_image.jpg'],
                                         self.image_frames, self.image_sizes, letterbox=False)
        self.assertEqual(result[0]['filepath'], 'my_image.jpg')

    def test_category_is_zero_indexed(self):
        """ONNX category is passed through as-is (0-indexed, not incremented)."""
        pred = np.zeros((1, 6), dtype=np.float32)
        pred[0, 2] = 0.5
        pred[0, 3] = 0.5
        pred[0, 4] = 0.9
        pred[0, 5] = 0   # class 0 -> expected category 0 (no offset applied)
        result = convert_onnx_detections([pred], self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes, letterbox=False)
        if result[0]['detections']:
            self.assertEqual(result[0]['detections'][0]['category'], 0)

    def test_multiple_images(self):
        tensors = _make_image_tensor(batch=2)
        paths = ['a.jpg', 'b.jpg']
        frames = [0, 0]
        sizes = np.array([[480, 640], [480, 640]])
        preds = [self._make_pred(), self._make_pred()]
        result = convert_onnx_detections(preds, tensors, paths, frames, sizes, letterbox=False)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# convert_yolo_detections
# ---------------------------------------------------------------------------

class TestConvertYoloDetections(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image_tensors = _make_image_tensor().numpy()
        cls.image_paths = ['a.jpg']
        cls.image_frames = np.array([0])
        cls.image_sizes = np.array([[480, 640]])

    def _make_yolov5_pred(self, num_detections=2, conf_val=0.9):
        """Fake YOLOv5 prediction: (N, 6) numpy array [x1,y1,x2,y2,conf,class]."""
        pred = np.zeros((num_detections, 6), dtype=np.float32)
        pred[:, 2] = 320   # x2 absolute pixels
        pred[:, 3] = 240   # y2 absolute pixels
        pred[:, 4] = conf_val
        pred[:, 5] = 0
        return pred

    def test_returns_list(self):
        preds = [self._make_yolov5_pred()]
        result = convert_yolo_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes,
                                         letterbox=False, model_type='yolov5')
        self.assertIsInstance(result, list)

    def test_result_has_required_keys(self):
        preds = [self._make_yolov5_pred()]
        result = convert_yolo_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes,
                                         letterbox=False, model_type='yolov5')
        for key in ('filepath', 'frame', 'max_detection_conf', 'detections'):
            self.assertIn(key, result[0])

    def test_no_detections_returns_empty_list(self):
        preds = [np.zeros((0, 6), dtype=np.float32)]
        result = convert_yolo_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes,
                                         letterbox=False, model_type='yolov5')
        self.assertEqual(result[0]['detections'], [])

    def test_unsupported_model_type_returns_none(self):
        preds = [self._make_yolov5_pred()] 
        result = convert_yolo_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes,
                                         letterbox=False, model_type='UNKNOWN')
        self.assertIsNone(result)

    def test_filepath_preserved(self):
        preds = [self._make_yolov5_pred()]
        result = convert_yolo_detections(preds, self.image_tensors, ['my_img.jpg'],
                                         self.image_frames, self.image_sizes,
                                         letterbox=False, model_type='yolov5')
        self.assertEqual(result[0]['filepath'], 'my_img.jpg')

    def test_detection_bbox_values_are_floats(self):
        preds = [self._make_yolov5_pred(num_detections=1)]
        result = convert_yolo_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes,
                                         letterbox=False, model_type='yolov5')
        if result[0]['detections']:
            det = result[0]['detections'][0]
            for key in ('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'conf'):
                self.assertIsInstance(det[key], float)

    def test_category_is_one_indexed(self):
        """MegaDetector (mdv5) categories are incremented by 1; plain yolov5 is not."""
        preds = [self._make_yolov5_pred(num_detections=1)]
        result = convert_yolo_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes,
                                         letterbox=False, model_type='mdv5')
        if result[0]['detections']:
            self.assertEqual(result[0]['detections'][0]['category'], 1)

    def test_mdv5_alias_works(self):
        preds = [self._make_yolov5_pred()]
        result = convert_yolo_detections(preds, self.image_tensors, self.image_paths,
                                         self.image_frames, self.image_sizes,
                                         letterbox=False, model_type='mdv5')
        self.assertIsInstance(result, list)


# ---------------------------------------------------------------------------
# parse_detections
# ---------------------------------------------------------------------------

class TestParseDetections(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results_with_detections = [
            _make_detection_result('a.jpg', detections=[
                {'bbox_x': 0.1, 'bbox_y': 0.1, 'bbox_w': 0.2, 'bbox_h': 0.3, 'conf': 0.9, 'category': 1}
            ]),
            _make_detection_result('b.jpg', detections=[
                {'bbox_x': 0.2, 'bbox_y': 0.2, 'bbox_w': 0.1, 'bbox_h': 0.1, 'conf': 0.5, 'category': 1}
            ]),
        ]
        cls.results_no_detections = [
            _make_detection_result('c.jpg', max_conf=0.0, detections=[]),
        ]
        cls.results_mixed = cls.results_with_detections + cls.results_no_detections

    def test_returns_dataframe(self):
        result = parse_detections(self.results_with_detections)
        self.assertIsInstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        result = parse_detections(self.results_with_detections)
        for col in ('filepath', 'frame', 'max_detection_conf', 'category', 'conf',
                    'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'):
            self.assertIn(col, result.columns)

    def test_one_row_per_detection(self):
        result = parse_detections(self.results_with_detections)
        self.assertEqual(len(result), 2)

    def test_empty_detections_produces_row_with_none_conf(self):
        result = parse_detections(self.results_no_detections)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result.iloc[0]['conf'])

    def test_mixed_results_correct_length(self):
        result = parse_detections(self.results_mixed)
        self.assertEqual(len(result), 3)

    def test_non_list_input_raises(self):
        with self.assertRaises(TypeError):
            parse_detections("not a list")

    def test_empty_list_raises(self):
        with self.assertRaises(AssertionError):
            parse_detections([])

    def test_confidence_threshold_filters_detections(self):
        result = parse_detections(self.results_with_detections, threshold=0.6)
        # only the 0.9 conf detection should pass
        self.assertEqual(len(result), 1)
        self.assertGreater(result.iloc[0]['conf'], 0.6)

    def test_bbox_values_clipped_between_0_and_1(self):
        results = [_make_detection_result('a.jpg', detections=[
            {'bbox_x': -0.1, 'bbox_y': 1.5, 'bbox_w': 0.2, 'bbox_h': 0.3, 'conf': 0.9, 'category': 1}
        ])]
        result = parse_detections(results)
        self.assertGreaterEqual(result.iloc[0]['bbox_x'], 0.0)
        self.assertLessEqual(result.iloc[0]['bbox_y'], 1.0)

    def test_merge_with_manifest(self):
        manifest = pd.DataFrame({
            'filepath': ['a.jpg', 'b.jpg'],
            'frame': [0, 0],
            'station': ['cam1', 'cam2'],
        })
        result = parse_detections(self.results_with_detections, manifest=manifest)
        self.assertIn('station', result.columns)
        self.assertEqual(len(result), 2)

    def test_merge_with_invalid_file_col_raises(self):
        manifest = pd.DataFrame({'path': ['a.jpg'], 'frame': [0]})
        with self.assertRaises(ValueError):
            parse_detections(self.results_with_detections, manifest=manifest, file_col='filepath')

    def test_saves_to_out_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'detections.csv')
            with patch('animl.file_management.check_file', return_value=False):
                parse_detections(self.results_with_detections, out_file=out_file)
            self.assertTrue(Path(out_file).exists())

    def test_filepath_column_values_correct(self):
        result = parse_detections(self.results_with_detections)
        self.assertIn('a.jpg', result['filepath'].values)
        self.assertIn('b.jpg', result['filepath'].values)


if __name__ == '__main__':
    unittest.main()