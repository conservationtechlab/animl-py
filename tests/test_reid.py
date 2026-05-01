"""
Unit tests for animl/reid/

@ Kyra Swanson 2024
"""
import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from animl.reid.distance import (
    remove_diagonal,
    euclidean_squared_distance,
    cosine_distance,
    compute_distance_matrix,
    compute_batched_distance_matrix,
)
from animl.reid.miewid import MiewIdNet, GeM, l2_norm, MIEWID_SIZE
from animl.reid.inference import load_miew, extract_miew_embeddings


class TestRemoveDiagonal(unittest.TestCase):

    def test_output_shape_3x3(self):
        A = torch.arange(9, dtype=torch.float).reshape(3, 3)
        result = remove_diagonal(A)
        self.assertEqual(result.shape, (3, 2))

    def test_output_shape_4x4(self):
        A = torch.ones(4, 4)
        result = remove_diagonal(A)
        self.assertEqual(result.shape, (4, 3))

    def test_non_square_raises(self):
        A = torch.ones(2, 3)
        with self.assertRaises(ValueError):
            remove_diagonal(A)

    def test_diagonal_values_removed(self):
        # identity matrix — diagonal is 1, off-diagonal is 0
        A = torch.eye(3)
        result = remove_diagonal(A)
        self.assertTrue((result == 0).all())

    def test_2x2_edge_case(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = remove_diagonal(A)
        self.assertEqual(result.shape, (2, 1))


class TestEuclideanSquaredDistance(unittest.TestCase):

    def test_output_shape(self):
        a = torch.randn(3, 4)
        b = torch.randn(5, 4)
        result = euclidean_squared_distance(a, b)
        self.assertEqual(result.shape, (3, 5))

    def test_self_distance_is_zero(self):
        a = torch.tensor([[1.0, 2.0, 3.0]])
        result = euclidean_squared_distance(a, a)
        self.assertAlmostEqual(result[0, 0].item(), 0.0, places=5)

    def test_known_values(self):
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[3.0, 4.0]])
        result = euclidean_squared_distance(a, b)
        self.assertAlmostEqual(result[0, 0].item(), 25.0, places=5)

    def test_non_negative(self):
        a = torch.randn(4, 3)
        result = euclidean_squared_distance(a, a)
        self.assertTrue((result >= -1e-5).all())

    def test_square_output_for_same_inputs(self):
        a = torch.randn(5, 8)
        result = euclidean_squared_distance(a, a)
        self.assertEqual(result.shape, (5, 5))


class TestCosineDistance(unittest.TestCase):

    def test_identical_vectors_distance_is_zero(self):
        a = torch.tensor([[1.0, 0.0, 0.0]])
        result = cosine_distance(a, a)
        self.assertAlmostEqual(result[0, 0].item(), 0.0, places=5)

    def test_orthogonal_vectors_distance_is_one(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        result = cosine_distance(a, b)
        self.assertAlmostEqual(result[0, 0].item(), 1.0, places=5)

    def test_opposite_vectors_distance_is_two(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[-1.0, 0.0]])
        result = cosine_distance(a, b)
        self.assertAlmostEqual(result[0, 0].item(), 2.0, places=5)

    def test_output_shape(self):
        a = torch.randn(3, 4)
        b = torch.randn(2, 4)
        result = cosine_distance(a, b)
        self.assertEqual(result.shape, (3, 2))

    def test_values_in_range(self):
        a = torch.randn(5, 8)
        b = torch.randn(4, 8)
        result = cosine_distance(a, b)
        self.assertTrue((result >= -1e-5).all())
        self.assertTrue((result <= 2 + 1e-5).all())


class TestComputeDistanceMatrix(unittest.TestCase):

    def test_euclidean_returns_numpy(self):
        a = np.random.randn(3, 4).astype(np.float32)
        result = compute_distance_matrix(a, a, metric='euclidean')
        self.assertIsInstance(result, np.ndarray)

    def test_cosine_returns_numpy(self):
        a = np.random.randn(3, 4).astype(np.float32)
        result = compute_distance_matrix(a, a, metric='cosine')
        self.assertIsInstance(result, np.ndarray)

    def test_euclidean_output_shape(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(5, 4).astype(np.float32)
        result = compute_distance_matrix(a, b, metric='euclidean')
        self.assertEqual(result.shape, (3, 5))

    def test_cosine_output_shape(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(2, 4).astype(np.float32)
        result = compute_distance_matrix(a, b, metric='cosine')
        self.assertEqual(result.shape, (3, 2))

    def test_accepts_torch_tensors(self):
        a = torch.randn(3, 4)
        result = compute_distance_matrix(a, a, metric='euclidean')
        self.assertIsInstance(result, np.ndarray)

    def test_unknown_metric_raises(self):
        a = np.random.randn(3, 4).astype(np.float32)
        with self.assertRaises(ValueError):
            compute_distance_matrix(a, a, metric='unknown_metric')

    def test_1d_input_raises(self):
        a = np.random.randn(4).astype(np.float32)
        with self.assertRaises(AssertionError):
            compute_distance_matrix(a, a, metric='euclidean')

    def test_self_euclidean_diagonal_near_zero(self):
        a = np.random.randn(4, 3).astype(np.float32)
        result = compute_distance_matrix(a, a, metric='euclidean')
        for i in range(4):
            self.assertAlmostEqual(result[i, i], 0.0, places=4)


class TestComputeBatchedDistanceMatrix(unittest.TestCase):

    def test_output_shape(self):
        a = np.random.randn(5, 4).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)
        result = compute_batched_distance_matrix(a, b, metric='cosine', batch_size=2)
        self.assertEqual(result.shape, (5, 3))

    def test_matches_unbatched(self):
        np.random.seed(42)
        a = np.random.randn(6, 4).astype(np.float32)
        unbatched = compute_batched_distance_matrix(a, a, metric='cosine', batch_size=100)
        batched = compute_batched_distance_matrix(a, a, metric='cosine', batch_size=2)
        np.testing.assert_allclose(unbatched, batched, atol=1e-5)

    def test_accepts_torch_tensors(self):
        a = torch.randn(4, 3)
        result = compute_batched_distance_matrix(a, a, metric='cosine', batch_size=2)
        self.assertIsInstance(result, np.ndarray)

    def test_batch_size_larger_than_input(self):
        a = np.random.randn(3, 4).astype(np.float32)
        result = compute_batched_distance_matrix(a, a, metric='euclidean', batch_size=100)
        self.assertEqual(result.shape, (3, 3))

    def test_euclidean_metric(self):
        a = np.random.randn(4, 3).astype(np.float32)
        result = compute_batched_distance_matrix(a, a, metric='euclidean', batch_size=2)
        self.assertEqual(result.shape, (4, 4))
        for i in range(4):
            self.assertAlmostEqual(result[i, i], 0.0, places=4)


class TestL2Norm(unittest.TestCase):

    def test_output_is_unit_norm(self):
        x = torch.tensor([[3.0, 4.0]])
        result = l2_norm(x, axis=1)
        norm = torch.norm(result, p=2, dim=1)
        self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_batch_output_all_unit_norm(self):
        x = torch.randn(5, 8)
        result = l2_norm(x, axis=1)
        norms = torch.norm(result, p=2, dim=1)
        for n in norms:
            self.assertAlmostEqual(n.item(), 1.0, places=5)

    def test_output_shape_unchanged(self):
        x = torch.randn(4, 6)
        result = l2_norm(x, axis=1)
        self.assertEqual(result.shape, x.shape)


class TestGeM(unittest.TestCase):

    def test_forward_produces_tensor(self):
        gem = GeM()
        x = torch.randn(2, 16, 7, 7)  # batch=2, channels=16, h=7, w=7
        result = gem(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_output_shape(self):
        gem = GeM()
        x = torch.randn(2, 32, 8, 8)
        result = gem(x)
        # GeM global pools spatial dims → (batch, channels, 1, 1)
        self.assertEqual(result.shape, (2, 32, 1, 1))

    def test_repr_contains_p_and_eps(self):
        gem = GeM()
        r = repr(gem)
        self.assertIn('p=', r)
        self.assertIn('eps=', r)


class TestMiewIdNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = MiewIdNet(device='cpu', pretrained=False)
        cls.model.eval()

    def test_instantiates(self):
        self.assertIsNotNone(self.model)

    def test_has_framework_attribute_after_manual_set(self):
        self.model.framework = 'torch'
        self.assertEqual(self.model.framework, 'torch')

    def test_extract_feat_output_shape(self):
        # MIEWID_SIZE = 440; use a small proxy size to keep the test fast
        x = torch.randn(1, 3, 440, 440)
        with torch.no_grad():
            result = self.model.extract_feat(x)
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result.shape[0], 1)  # batch size

    def test_forward_equals_extract_feat(self):
        x = torch.randn(1, 3, 440, 440)
        with torch.no_grad():
            fwd = self.model(x)
            feat = self.model.extract_feat(x)
        self.assertTrue(torch.allclose(fwd, feat))


class TestLoadMiew(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path_pt = Path.cwd() / 'models/miewid_v3.bin'
        cls.model_path_onnx = Path.cwd() / 'models/miewid_v3.onnx'
        if not cls.model_path_pt.exists() and not cls.model_path_onnx.exists():
            raise unittest.SkipTest(
                "No MiewID model file found at models/miewid_v3.bin or models/miewid_v3.onnx; "
                "skipping TestLoadMiew (requires downloaded model weights)."
            )

    def test_nonexistent_path_raises(self):
        with self.assertRaises(Exception):
            load_miew('nonexistent_model.pt', device='cpu')

    def test_loads_pytorch_model(self):
        if not self.model_path_pt.exists():
            self.skipTest("PyTorch model not found")
        model = load_miew(str(self.model_path_pt), device='cpu')
        self.assertIsNotNone(model)
        self.assertEqual(model.framework, 'torch')

    def test_pytorch_model_is_in_eval_mode(self):
        if not self.model_path_pt.exists():
            self.skipTest("PyTorch model not found")
        model = load_miew(str(self.model_path_pt), device='cpu')
        self.assertFalse(model.training)

    def test_loads_onnx_model(self):
        if not self.model_path_onnx.exists():
            self.skipTest("ONNX model not found")
        model = load_miew(str(self.model_path_onnx), device='cpu')
        self.assertIsNotNone(model)
        self.assertEqual(model.framework, 'onnx')


class TestExtractMiewEmbeddings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path = Path.cwd() / 'models/miewid.bin'
        cls.detections_path = Path(__file__).parent / 'GroundTruth/southwest/Detections.csv'
        if not cls.model_path.exists() or not cls.detections_path.exists():
            raise unittest.SkipTest(
                "MiewID model or ground-truth detections not found; "
                "skipping TestExtractMiewEmbeddings (requires model weights and test data)."
            )
        cls.model = load_miew(str(cls.model_path), device='cpu')
        cls.detections = pd.read_csv(cls.detections_path)

    def test_returns_ndarray(self):
        result = extract_miew_embeddings(self.model, self.detections.head(2), device='cpu', batch_size=1)
        self.assertIsInstance(result, np.ndarray)

    def test_output_shape_matches_input_rows(self):
        n = 3
        result = extract_miew_embeddings(self.model, self.detections.head(n), device='cpu', batch_size=1)
        self.assertEqual(result.shape[0], n)

    def test_embedding_dim_is_positive(self):
        result = extract_miew_embeddings(self.model, self.detections.head(2), device='cpu', batch_size=1)
        self.assertGreater(result.shape[1], 0)

    def test_embeddings_are_finite(self):
        result = extract_miew_embeddings(self.model, self.detections.head(2), device='cpu', batch_size=1)
        self.assertTrue(np.isfinite(result).all())

    def test_missing_file_col_raises(self):
        with self.assertRaises(ValueError):
            extract_miew_embeddings(self.model, self.detections.head(2), file_col='nonexistent_col', device='cpu')


if __name__ == '__main__':
    unittest.main()
