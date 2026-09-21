"""
Unit tests for animl/reid/

@ Kyra Swanson 2024
"""
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from animl.reid import (
    load_miew, 
    extract_miew_embeddings,
    remove_diagonal,
    euclidean_squared_distance,
    cosine_distance,
    compute_distance_matrix,
    compute_batched_distance_matrix,
)
from animl.model_architecture import MiewIdNet, GeM


MIEWID_HF_REPO = "conservationxlabs/miewid-msv3"


def _fetch_and_convert_miewid(out_path: Path) -> bool:
    """
    Fetch MiewID weights from Hugging Face and convert them into a state_dict
    animl's own MiewIdNet can load, caching the result at `out_path`.

    conservationxlabs/miewid-msv3 has no declared license for its weights, so
    rather than redistributing a copy as our own release asset, this fetches
    them from the original source on demand -- which is what Conservation X
    Labs' own model card says to do.

    This intentionally avoids `transformers.AutoModel.from_pretrained(...,
    trust_remote_code=True)`, which would run Conservation X Labs' own
    modeling code. Instead it strict-loads the raw safetensors weights into
    animl's own MiewIdNet re-implementation -- a real compatibility check.
    Any key mismatch is treated as a hard failure (not a skip), since it
    means the two architectures have diverged and any test built on the
    result can't be trusted. Only a genuine inability to reach Hugging Face
    (missing optional deps, no network) results in a skip.

    Returns:
        True if weights were fetched/converted (or already cached) and are
        ready to use; False if this environment can't reach Hugging Face
        right now, in which case the caller should skip.
    """
    if out_path.exists():
        return True

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
        from safetensors.torch import load_file
    except ImportError:
        print(
            "huggingface_hub/safetensors not installed; skipping MiewID tests. "
            "Install with: pip install huggingface_hub safetensors"
        )
        return False

    try:
        config_path = hf_hub_download(MIEWID_HF_REPO, "config.json")
        weights_path = hf_hub_download(MIEWID_HF_REPO, "model.safetensors")
    except (HfHubHTTPError, OSError, ConnectionError) as e:
        print(f"Could not reach Hugging Face ({MIEWID_HF_REPO}); skipping MiewID tests. Error: {e}")
        return False

    with open(config_path) as f:
        config = json.load(f)
    state_dict = load_file(weights_path)


    KEY_REMAP = {"backbone.global_pool.p": "pooling.p"}
    for src_key, dst_key in KEY_REMAP.items():
        if src_key in state_dict and dst_key not in state_dict:
            tensor = state_dict.pop(src_key)
            if tuple(tensor.shape) != (1,):
                raise RuntimeError(
                    f"Refusing to remap '{src_key}' -> '{dst_key}': expected a shape (1,) GeM "
                    f"exponent but got {tuple(tensor.shape)}. The checkpoint's structure may have "
                    "changed in a way this narrow remap no longer accounts for."
                )
            print(f"Remapping checkpoint key '{src_key}' -> '{dst_key}' (shape {tuple(tensor.shape)})")
            state_dict[dst_key] = tensor

    # Infer n_classes from the checkpoint itself: config.json's value is
    # frequently a placeholder unrelated to the actual training run.
    final_weight_keys = [k for k in state_dict if k.startswith("final.") and k.endswith(".weight")]
    if not final_weight_keys:
        raise RuntimeError(
            "Could not find a 'final.*.weight' tensor in the Hugging Face checkpoint to infer "
            f"n_classes from. Top-level key prefixes found: {sorted({k.split('.')[0] for k in state_dict})}. "
            "The remote architecture may no longer match animl's MiewIdNet."
        )
    n_classes = state_dict[final_weight_keys[0]].shape[0]

    model = MiewIdNet(
        device="cpu",
        n_classes=n_classes,
        model_name=config.get("model_name", "efficientnetv2_rw_m"),
        use_fc=config.get("use_fc", False),
        fc_dim=config.get("fc_dim", 512),
        dropout=config.get("dropout", 0.0),
        loss_module=config.get("loss_module", "softmax"),
        pretrained=False,  # every weight is about to be overwritten
    )
    # The actual compatibility test. strict=False only so we can report
    # *which* keys mismatch; any mismatch at all is still a hard failure.
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "animl's MiewIdNet does not match the Hugging Face checkpoint key-for-key.\n"
            f"Missing keys ({len(result.missing_keys)}): {result.missing_keys}\n"
            f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys}\n"
            "Do not trust this conversion -- the implementations have diverged, or this "
            "checkpoint needs an explicit key-remapping step before it can be used."
        )

    # Sanity check: run the same call production code makes
    # (extract_miew_embeddings calls model.extract_feat(...) directly).
    model.eval()
    with torch.no_grad():
        emb = model.extract_feat(torch.randn(1, 3, 224, 224))
    assert emb.ndim == 2 and emb.shape[0] == 1, f"Unexpected embedding shape from extract_feat: {tuple(emb.shape)}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return True


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

    def test_has_architecture_attribute_after_manual_set(self):
        self.model.architecture = 'miewid'
        self.assertEqual(self.model.architecture, 'miewid')

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
            _fetch_and_convert_miewid(cls.model_path_pt)
        if not cls.model_path_pt.exists() and not cls.model_path_onnx.exists():
            raise unittest.SkipTest(
                "No MiewID model file found at models/miewid_v3.bin or models/miewid_v3.onnx, "
                "and it couldn't be fetched from Hugging Face; skipping TestLoadMiew."
            )

    def test_nonexistent_path_raises(self):
        with self.assertRaises(Exception):
            load_miew('nonexistent_model.pt', device='cpu')

    def test_loads_pytorch_model(self):
        if not self.model_path_pt.exists():
            self.skipTest("PyTorch model not found")
        model = load_miew(str(self.model_path_pt), device='cpu')
        self.assertIsNotNone(model)
        self.assertEqual(model.architecture, 'miewid')

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
        self.assertEqual(model.architecture, 'onnx')


class TestExtractMiewEmbeddings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path = Path.cwd() / 'models/miewid_v3.bin'
        cls.detections_path = Path(__file__).parent / 'GroundTruth/southwest/Detections.csv'
        if not cls.model_path.exists():
            _fetch_and_convert_miewid(cls.model_path)
        if not cls.model_path.exists() or not cls.detections_path.exists():
            raise unittest.SkipTest(
                "MiewID model (local or fetched from Hugging Face) or ground-truth detections "
                "not found; skipping TestExtractMiewEmbeddings."
            )
        cls.model = load_miew(str(cls.model_path), device='cpu')
        detections = pd.read_csv(cls.detections_path)
        examples_dir = Path(__file__).parent.parent / 'examples' / 'Southwest'
        detections['filepath'] = detections['filepath'].apply(
            lambda p: str(examples_dir / Path(p).name)
        )
        cls.detections = detections

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
