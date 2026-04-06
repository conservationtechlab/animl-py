"""
Unit tests for animl/classification.py

@ Kyra Swanson 2023
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from animl.split import get_animals

from animl.classification import (
    classify,
    load_class_list,
    load_classifier,
    single_classification,
    sequence_classification,
    save_classifier,
)


class TestLoadClassList(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.class_file = str(Path(cls.tmp_dir) / 'classes.csv')
        pd.DataFrame({'class': ['cat', 'dog', 'bird']}).to_csv(cls.class_file, index=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_returns_dataframe(self):
        result = load_class_list(self.class_file)
        self.assertIsInstance(result, pd.DataFrame)

    def test_correct_number_of_classes(self):
        result = load_class_list(self.class_file)
        self.assertEqual(len(result), 3)

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_class_list('/nonexistent/classes.csv')


class TestSaveClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_saves_checkpoint_file(self):
        import torch.nn as nn

        model = nn.Linear(4, 2)
        stats = {'loss': 0.5, 'accuracy': 0.9}
        out_dir = str(Path(self.tmp_dir) / 'checkpoints')

        save_classifier(model, out_dir, epoch=1, stats=stats)

        self.assertTrue(Path(out_dir, '1.pt').exists())

    def test_creates_output_directory(self):
        import torch.nn as nn

        model = nn.Linear(4, 2)
        stats = {'loss': 0.3}
        out_dir = str(Path(self.tmp_dir) / 'new_checkpoints')

        save_classifier(model, out_dir, epoch=5, stats=stats)
        self.assertTrue(Path(out_dir).exists())

    def test_checkpoint_contains_model_and_stats(self):
        import torch
        import torch.nn as nn

        model = nn.Linear(4, 2)
        stats = {'loss': 0.1}
        out_dir = str(Path(self.tmp_dir) / 'verify_checkpoints')

        save_classifier(model, out_dir, epoch=2, stats=stats)

        checkpoint = torch.load(Path(out_dir) / '2.pt', weights_only=False)
        self.assertIn('model', checkpoint)
        self.assertIn('stats', checkpoint)

    def test_checkpoint_includes_optimizer_state(self):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        model = nn.Linear(4, 2)
        optimizer = optim.Adam(model.parameters())
        stats = {'loss': 0.2}
        out_dir = str(Path(self.tmp_dir) / 'opt_checkpoints')

        save_classifier(model, out_dir, epoch=3, stats=stats, optimizer=optimizer)

        checkpoint = torch.load(Path(out_dir) / '3.pt', weights_only=False)
        self.assertIn('optimizer', checkpoint)


class TestLoadClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path_pt = Path.cwd() / 'models/sdzwa_southwest_v3.pt'
        cls.model_path_onnx = Path.cwd() / 'models/sdzwa_southwest_v3.onnx'
        cls.class_list_path = Path.cwd() / 'models/sdzwa_southwest_v3_classes.csv'
        cls.classes = load_class_list(cls.class_list_path)

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_classifier('nonexistent_model.pt', self.classes, device='cpu')

    def test_unsupported_architecture_raises(self):
        with self.assertRaises(ValueError):
            load_classifier(self.model_path_pt, self.classes, device='cpu', architecture='unsupported_arch')

    def test_unsupported_device(self):
        model = load_classifier(self.model_path_pt, self.classes, device='unsupported_device', architecture='efficientnet_v2_m')
        self.assertIsNotNone(model)

    def test_loads_pytorch_cpu(self):
        model = load_classifier(self.model_path_pt, self.classes, 
                                device='cpu', architecture='efficientnet_v2_m')
        self.assertIsNotNone(model)

    def test_loads_onnx_cpu_with_classes(self):
        model = load_classifier(self.model_path_onnx, self.classes, device='cpu')
        self.assertIsNotNone(model)

    def test_loads_onnx_cpu_without_classes(self):
        model = load_classifier(self.model_path_onnx, None, device='cpu')
        self.assertIsNotNone(model)

    def test_loads_pytorch_cuda(self):
        model = load_classifier(self.model_path_pt, self.classes, 
                                device='cuda', architecture='efficientnet_v2_m')
        self.assertIsNotNone(model)

    def test_loads_onnx_cuda(self):
        model = load_classifier(self.model_path_onnx, self.classes, device='cuda')
        self.assertIsNotNone(model)    



class TestClassifyPytorch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path_pt = Path.cwd() / 'models/sdzwa_southwest_v3.pt'
        detections = pd.read_csv(Path(__file__).parent / 'GroundTruth/southwest/Detections.csv')
        cls.detections = get_animals(detections)
        cls.class_path = load_class_list(Path.cwd() / 'models/sdzwa_southwest_v3_classes.csv')
        cls.model, cls.classes = load_classifier(cls.model_path_pt, cls.class_path, device='cpu', architecture='efficientnet_v2_m')
        cls.tmpdir = tempfile.mkdtemp()

    def test_returns_array(self):
        result, fnf = classify(self.model, self.detections.copy(), device='cpu', batch_size=1)
        self.assertIsInstance(result, np.ndarray)

    def test_output_shape_matches_input(self):
        result, fnf = classify(self.model, self.detections.copy(), device='cpu', batch_size=1)
        print(result.shape)
        print(len(self.detections), len(self.classes))
        self.assertEqual(result.shape[0], len(self.detections))
        self.assertEqual(result.shape[1], len(self.classes))

    def test_output_values_are_probabilities(self):
        result, fnf = classify(self.model, self.detections.copy(), device='cpu', batch_size=1)
        self.assertTrue((result >= 0).all() and (result <= 1).all())

    def test_empty_detections_returns_empty_array(self):
        empty = pd.DataFrame(columns=self.detections.columns)
        result, fnf = classify(self.model, empty, device='cpu', batch_size=1)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 0)

    def test_invalid_file_col_raises(self):
        with self.assertRaises(ValueError):
            classify(self.model, self.detections.copy(), device='cpu', batch_size=1, file_col='invalid_col')

    def test_invalid_batch_size_raises(self):
        with self.assertRaises(ValueError):
            classify(self.model, self.detections.copy(), device='cpu', batch_size=-1)

    def test_non_dataframe_detections_raises(self):
        with self.assertRaises(ValueError):
            classify(self.model, "not_a_dataframe", device='cpu', batch_size=1)

    def test_non_model_raises(self):
        with self.assertRaises(AttributeError):
            classify("not_a_model", self.detections[0:1].copy(), device='cpu', batch_size=1) 
    
    def test_non_string_file_col_raises(self):
        with self.assertRaises(ValueError):
            classify(self.model, self.detections[0:1].copy(), device='cpu', batch_size=1, file_col=123)
    
    def test_non_boolean_crop_raises(self):
        with self.assertRaises(TypeError):
            classify(self.model, self.detections[0:1].copy(), device='cpu', batch_size=1, crop='not_a_boolean')

    def test_non_boolean_normalize_raises(self):
        with self.assertRaises(TypeError):
            classify(self.model, self.detections[0:1].copy(), device='cpu', batch_size=1, normalize='not_a_boolean')

    def test_non_integer_resize_width_raises(self):
        with self.assertRaises(TypeError):
            classify(self.model, self.detections[0:1].copy(), device='cpu', batch_size=1, resize_width='not_an_integer')

    def test_non_integer_resize_height_raises(self):
        with self.assertRaises(TypeError):
            classify(self.model, self.detections[0:1].copy(), device='cpu', batch_size=1, resize_height='not_an_integer')    

    def test_non_integer_num_workers_raises(self):
        with self.assertRaises(TypeError):
            classify(self.model, self.detections[0:1].copy(), device='cpu', batch_size=1, num_workers='not_an_integer')

    def test_non_string_out_file_raises(self):
        with self.assertRaises(ValueError):
            classify(self.model, self.detections[0:1].copy(), device='cpu', batch_size=1, out_file=123)

    def test_nonexistent_out_file_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            classify(self.model, self.detections[0:1].copy(), device='cpu', batch_size=1, out_file='/nonexistent_dir/results.csv')

    def test_saves_output_file(self):
        out_file = str(Path(self.tmpdir) / 'classification_results.csv')
        classify(self.model, self.detections.copy(), device='cpu', batch_size=1, out_file=out_file)
        self.assertTrue(Path(out_file).exists())

    def test_input_contains_nonexistent_files_raises(self):
        detections = self.detections.copy()
        detections.loc[0, 'filepath'] = 'nonexistent_file.jpg'
        result, fnf = classify(self.model, detections, device='cpu', batch_size=1)
        self.assertEqual(result.shape[0], len(self.detections) - len(fnf)) 

    def test_input_contains_non_image_files_raises(self):
        detections = self.detections.copy()
        detections.loc[0, 'filepath'] = __file__  # this test file is not an image
        result, fnf = classify(self.model, detections, device='cpu', batch_size=1)
        self.assertEqual(result.shape[0], len(self.detections) - len(fnf)) 

    def test_input_contains_empty_filepaths_raises(self):
        detections = self.detections.copy()
        detections.loc[0, 'filepath'] = ''
        result, fnf = classify(self.model, detections, device='cpu', batch_size=1)
        self.assertEqual(result.shape[0], len(self.detections) - len(fnf)) 

    def test_input_missing_bbox_raises(self):
        detections = self.detections.copy()
        detections = detections.drop(columns=['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
        with self.assertRaises(ValueError):
            classify(self.model, detections, device='cpu', batch_size=1)
            


class TestClassifyOnnx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path_onnx = Path.cwd() / 'models/sdzwa_southwest_v3.onnx'
        cls.detections = pd.read_csv(Path(__file__).parent / 'GroundTruth/main/Detections.csv')
        cls.classes = load_class_list(Path.cwd() / 'models/sdzwa_southwest_v3_classes.csv')
        cls.model = load_classifier(cls.model_path_onnx, cls.classes, device='cpu')



class TestSingleClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.class_list = pd.Series(['cat', 'dog', 'bird'])
        cls.animals = pd.DataFrame({
            'filepath': ['a.jpg', 'b.jpg', 'c.jpg'],
            'extension': ['.jpg', '.jpg', '.jpg'],
            'conf': [0.9, 0.8, 0.7],
        })
        # strong prediction for cat, dog, bird respectively
        cls.predictions_raw = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])

    def test_returns_dataframe(self):
        result = single_classification(self.animals.copy(), None, self.predictions_raw, self.class_list)
        self.assertIsInstance(result, pd.DataFrame)

    def test_prediction_column_added(self):
        result = single_classification(self.animals.copy(), None, self.predictions_raw, self.class_list)
        self.assertIn('prediction', result.columns)

    def test_confidence_column_added(self):
        result = single_classification(self.animals.copy(), None, self.predictions_raw, self.class_list)
        self.assertIn('confidence', result.columns)

    def test_correct_predictions(self):
        result = single_classification(self.animals.copy(), None, self.predictions_raw, self.class_list)
        self.assertEqual(result.iloc[0]['prediction'], 'cat')
        self.assertEqual(result.iloc[1]['prediction'], 'dog')
        self.assertEqual(result.iloc[2]['prediction'], 'bird')

    def test_confidence_values_are_positive(self):
        result = single_classification(self.animals.copy(), None, self.predictions_raw, self.class_list)
        self.assertTrue((result['confidence'] >= 0).all())

    def test_empty_animals_with_empty_detections(self):
        empty = pd.DataFrame({
            'filepath': ['d.jpg'],
            'extension': ['.jpg'],
            'conf': [1.0],
            'prediction': ['empty'],
            'confidence': [1.0],
        })
        result = single_classification(pd.DataFrame(), empty, np.array([]).reshape(0, 3), self.class_list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['prediction'], 'empty')

    def test_none_empty_treated_as_empty_dataframe(self):
        result = single_classification(self.animals.copy(), None, self.predictions_raw, self.class_list)
        self.assertEqual(len(result), len(self.animals))

    def test_best_returns_one_per_file(self):
        animals = pd.DataFrame({
            'filepath': ['a.jpg', 'a.jpg'],
            'extension': ['.jpg', '.jpg'],
            'conf': [0.9, 0.6],
        })
        preds = np.array([[0.9, 0.05, 0.05], [0.6, 0.2, 0.2]])
        result = single_classification(animals, None, preds, self.class_list, best=True)
        self.assertEqual(len(result['filepath'].unique()), 1)


class TestSequenceClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.class_list = pd.Series(['cat', 'dog', 'bird'])
        cls.animals = pd.DataFrame({
            'filepath': ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'],
            'extension': ['.jpg', '.jpg', '.jpg', '.jpg'],
            'station': ['cam1', 'cam1', 'cam1', 'cam2'],
            'conf': [0.9, 0.8, 0.85, 0.7],
            'datetime': [
                '2023-01-01 10:00:00',
                '2023-01-01 10:00:30',  # same sequence as first
                '2023-01-01 10:05:00',  # new sequence
                '2023-01-01 10:00:00',  # different camera
            ]
        })
        cls.predictions_raw = np.array([
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])

    def test_returns_dataframe(self):
        result = sequence_classification(
            self.animals.copy(), None, self.predictions_raw, self.class_list, station_col='station'
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_prediction_column_added(self):
        result = sequence_classification(
            self.animals.copy(), None, self.predictions_raw, self.class_list, station_col='station'
        )
        self.assertIn('prediction', result.columns)

    def test_confidence_column_added(self):
        result = sequence_classification(
            self.animals.copy(), None, self.predictions_raw, self.class_list, station_col='station'
        )
        self.assertIn('confidence', result.columns)

    def test_sequence_column_added(self):
        result = sequence_classification(
            self.animals.copy(), None, self.predictions_raw, self.class_list, station_col='station'
        )
        self.assertIn('sequence', result.columns)

    def test_invalid_station_col_raises(self):
        with self.assertRaises(Exception):
            sequence_classification(
                self.animals.copy(), None, self.predictions_raw, self.class_list, station_col=''
            )

    def test_invalid_maxdiff_raises(self):
        with self.assertRaises(Exception):
            sequence_classification(
                self.animals.copy(), None, self.predictions_raw, self.class_list,
                station_col='station', maxdiff=-1
            )

    def test_missing_filepath_col_raises(self):
        df = self.animals.copy().rename(columns={'filepath': 'path'})
        with self.assertRaises(ValueError):
            sequence_classification(
                df, None, self.predictions_raw, self.class_list,
                station_col='station', file_col='filepath'
            )

    def test_missing_datetime_col_raises(self):
        df = self.animals.copy().drop(columns=['datetime'])
        with self.assertRaises(ValueError):
            sequence_classification(
                df, None, self.predictions_raw, self.class_list, station_col='station'
            )

    def test_close_images_get_same_sequence(self):
        result = sequence_classification(
            self.animals.copy(), None, self.predictions_raw, self.class_list, station_col='station'
        )
        cam1 = result[result['station'] == 'cam1'].sort_values('datetime').reset_index(drop=True)
        # first two images are 30s apart — same sequence
        self.assertEqual(cam1.iloc[0]['sequence'], cam1.iloc[1]['sequence'])

    def test_far_images_get_different_sequence(self):
        result = sequence_classification(
            self.animals.copy(), None, self.predictions_raw, self.class_list, station_col='station'
        )
        cam1 = result[result['station'] == 'cam1'].sort_values('datetime').reset_index(drop=True)
        # first and third image are 5min apart — different sequence
        self.assertNotEqual(cam1.iloc[0]['sequence'], cam1.iloc[2]['sequence'])

    def test_output_length_matches_input(self):
        result = sequence_classification(
            self.animals.copy(), None, self.predictions_raw, self.class_list, station_col='station'
        )
        self.assertEqual(len(result), len(self.animals))

    def test_conf_column_defaults_to_one_if_missing(self):
        animals = self.animals.copy().drop(columns=['conf'])
        result = sequence_classification(
            animals, None, self.predictions_raw, self.class_list, station_col='station'
        )
        self.assertIn('confidence', result.columns)


if __name__ == '__main__':
    unittest.main()