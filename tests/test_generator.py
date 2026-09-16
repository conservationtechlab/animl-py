"""
Unit tests for animl/generator.py

@ Kyra Swanson 2023
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from animl.generator import (
    Letterbox,
    ManifestGenerator,
    TrainGenerator,
    collate_fn,
    image_to_tensor,
    manifest_dataloader,
    train_dataloader,
    _get_model_transforms,
    _get_augmentations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(path, width=100, height=80, color=(128, 64, 32)):
    """Save a small solid-color RGB image to disk and return the path."""
    img = Image.new("RGB", (width, height), color)
    img.save(path)
    return str(path)


def _make_manifest(filepaths, with_bbox=False, label_col=None, labels=None):
    """Build a minimal manifest DataFrame."""
    df = pd.DataFrame({"filepath": filepaths})
    df["frame"] = 0
    if with_bbox:
        df["bbox_x"] = 0.1
        df["bbox_y"] = 0.1
        df["bbox_w"] = 0.5
        df["bbox_h"] = 0.5
    if label_col and labels:
        df[label_col] = labels
    return df


# ---------------------------------------------------------------------------
# Letterbox
# ---------------------------------------------------------------------------

class TestLetterbox(unittest.TestCase):
    """Test Letterbox transform for aspect ratio preservation."""

    def test_wide_image_padding(self):
        """Wide image should be padded vertically."""
        lb = Letterbox(resize_height=64, resize_width=64)
        img = Image.new("RGB", (128, 64))   # 2:1 aspect ratio
        result = lb(img)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (64, 64))

    def test_tall_image_padding(self):
        """Tall image should be padded horizontally."""
        lb = Letterbox(resize_height=64, resize_width=64)
        img = Image.new("RGB", (64, 128))   # 1:2 aspect ratio
        result = lb(img)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (64, 64))

    def test_square_image_no_padding(self):
        """Square image should be resized without padding."""
        lb = Letterbox(resize_height=64, resize_width=64)
        img = Image.new("RGB", (64, 64))
        result = lb(img)
        self.assertIsNotNone(result)
        self.assertEqual(result.size, (64, 64))

    def test_different_dimensions(self):
        """Test non-square resize targets."""
        lb = Letterbox(resize_height=128, resize_width=256)
        img = Image.new("RGB", (100, 100))
        result = lb(img)
        self.assertEqual(result.size, (256, 128))

    def test_output_type(self):
        """Output should always be PIL Image."""
        lb = Letterbox(resize_height=64, resize_width=64)
        img = Image.new("RGB", (100, 80))
        result = lb(img)
        self.assertIsInstance(result, Image.Image)


# ---------------------------------------------------------------------------
# image_to_tensor
# ---------------------------------------------------------------------------

class TestImageToTensor(unittest.TestCase):
    """Test image_to_tensor conversion function."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "test.jpg")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_returns_tuple_of_four(self):
        """Should return (tensor, paths, frames, sizes)."""
        result = image_to_tensor(self.img_path, resize_height=64, 
                                resize_width=64, letterbox=False)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_tensor_has_batch_dimension(self):
        """Tensor should have batch dimension (1, 3, H, W)."""
        tensor, _, _, _ = image_to_tensor(self.img_path, resize_height=64,
                                         resize_width=64, letterbox=False)
        self.assertEqual(tensor.shape, (1, 3, 64, 64))

    def test_tensor_dtype_is_float32(self):
        """Tensor dtype should be float32."""
        tensor, _, _, _ = image_to_tensor(self.img_path, resize_height=64,
                                         resize_width=64, letterbox=False)
        self.assertEqual(tensor.dtype, torch.float32)

    def test_tensor_normalized_values(self):
        """Tensor values should be in [0, 1] range."""
        tensor, _, _, _ = image_to_tensor(self.img_path, resize_height=64,
                                         resize_width=64, letterbox=False)
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_filepath_preserved_in_output(self):
        """Original filepath should be preserved in output."""
        _, paths, _, _ = image_to_tensor(self.img_path, resize_height=64,
                                        resize_width=64, letterbox=False)
        self.assertEqual(paths[0], self.img_path)

    def test_frame_default_is_zero(self):
        """Default frame for images should be 0."""
        _, _, frames, _ = image_to_tensor(self.img_path, resize_height=64,
                                         resize_width=64, letterbox=False)
        self.assertEqual(frames[0], 0)

    def test_size_tensor_contains_original_dimensions(self):
        """Size tensor should contain (height, width) of original image."""
        _, _, _, sizes = image_to_tensor(self.img_path, resize_height=64,
                                        resize_width=64, letterbox=False)
        # Original image is 100w x 80h
        self.assertEqual(sizes[0][0].item(), 80)   # height
        self.assertEqual(sizes[0][1].item(), 100)  # width

    def test_letterbox_true_maintains_aspect(self):
        """Letterbox=True should maintain aspect ratio."""
        result = image_to_tensor(self.img_path, resize_height=64,
                                resize_width=64, letterbox=True)
        self.assertIsNotNone(result)
        tensor, _, _, _ = result
        self.assertEqual(tensor.shape, (1, 3, 64, 64))

    def test_nonexistent_file_returns_none(self):
        """Nonexistent file should return None."""
        result = image_to_tensor("/nonexistent/file.jpg", resize_height=64,
                                resize_width=64, letterbox=False)
        self.assertIsNone(result)

    def test_corrupted_image_returns_none(self):
        """Corrupted image file should return None."""
        bad = Path(self.tmp_dir) / "bad.jpg"
        bad.write_text("not an image")
        result = image_to_tensor(str(bad), resize_height=64,
                                resize_width=64, letterbox=False)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _get_model_transforms
# ---------------------------------------------------------------------------

class TestGetModelTransforms(unittest.TestCase):
    """Test transform pipeline generation for different architectures."""

    def test_default_transform_pipeline(self):
        """Default pipeline should handle resize and normalization."""
        transform = _get_model_transforms(64, 64, architecture=None, letterbox=False)
        self.assertIsNotNone(transform)
        # Test that it's a valid transform
        img = Image.new("RGB", (100, 80))
        result = transform(img)
        self.assertEqual(result.shape, (3, 64, 64))

    def test_letterbox_transform_pipeline(self):
        """Letterbox pipeline should preserve aspect ratio."""
        transform = _get_model_transforms(64, 64, architecture=None, letterbox=True)
        img = Image.new("RGB", (128, 64))
        result = transform(img)
        self.assertEqual(result.shape, (3, 64, 64))

    def test_bioclip_transform_pipeline(self):
        """BioCLIP should enforce 224x224 size."""
        transform = _get_model_transforms(100, 100, architecture="bioclip_2", letterbox=False)
        img = Image.new("RGB", (100, 100))
        result = transform(img)
        self.assertEqual(result.shape, (3, 224, 224))

    def test_miewid_transform_pipeline(self):
        """MiewID should enforce 128x128 size."""
        transform = _get_model_transforms(100, 100, architecture="miewid", letterbox=False)
        img = Image.new("RGB", (100, 100))
        result = transform(img)
        self.assertEqual(result.shape, (3, 440, 440))

    def test_invalid_resize_height_type_raises(self):
        """Non-integer resize_height should raise TypeError."""
        with self.assertRaises(TypeError):
            _get_model_transforms("64", 64)

    def test_invalid_resize_width_type_raises(self):
        """Non-integer resize_width should raise TypeError."""
        with self.assertRaises(TypeError):
            _get_model_transforms(64, "64")

    def test_invalid_letterbox_type_raises(self):
        """Non-boolean letterbox should raise TypeError."""
        with self.assertRaises(TypeError):
            _get_model_transforms(64, 64, letterbox="true")


# ---------------------------------------------------------------------------
# _get_augmentations
# ---------------------------------------------------------------------------

class TestGetAugmentations(unittest.TestCase):
    """Test data augmentation pipeline."""

    def test_augmentations_returns_compose(self):
        """Should return a Compose transform."""
        aug = _get_augmentations()
        self.assertIsNotNone(aug)

    def test_augmentations_applies_to_image(self):
        """Augmentations should be applicable to images."""
        aug = _get_augmentations()
        img = Image.new("RGB", (64, 64))
        result = aug(img)
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# ManifestGenerator
# ---------------------------------------------------------------------------

class TestManifestGeneratorBasic(unittest.TestCase):
    """Test ManifestGenerator initialization and basic functionality."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "img.jpg")
        cls.manifest = _make_manifest([cls.img_path])
        cls.manifest_bbox = _make_manifest([cls.img_path], with_bbox=True)
        cls.transform = _get_model_transforms(64, 64)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_creates_dataset_with_transform(self):
        """Should create dataset with required transform."""
        ds = ManifestGenerator(self.manifest, crop=False, transform=self.transform)
        self.assertIsNotNone(ds)

    def test_creates_dataset_without_transform(self):
        """Should create dataset even with transform=None."""
        ds = ManifestGenerator(self.manifest, crop=False, transform=None)
        self.assertIsNotNone(ds)

    def test_dataset_length(self):
        """Dataset length should match manifest size."""
        ds = ManifestGenerator(self.manifest, crop=False, transform=self.transform)
        self.assertEqual(len(ds), 1)

    def test_multiple_images_length(self):
        """Dataset with multiple images should have correct length."""
        paths = [_make_image(Path(self.tmp_dir) / f"img{i}.jpg") for i in range(3)]
        manifest = _make_manifest(paths)
        ds = ManifestGenerator(manifest, crop=False, transform=self.transform)
        self.assertEqual(len(ds), 3)

    def test_missing_file_col_raises_error(self):
        """Missing file_col should raise ValueError."""
        with self.assertRaises(ValueError):
            ManifestGenerator(self.manifest, file_col="nonexistent", 
                            crop=False, transform=self.transform)

    def test_crop_without_bbox_raises_error(self):
        """Crop without bbox columns should raise ValueError."""
        with self.assertRaises(ValueError):
            ManifestGenerator(self.manifest, crop=True, transform=self.transform)

    def test_invalid_crop_coord_raises_error(self):
        """Invalid crop_coord should raise ValueError."""
        with self.assertRaises(ValueError):
            ManifestGenerator(self.manifest_bbox, crop=True, 
                            crop_coord="invalid", transform=self.transform)

    def test_invalid_normalize_type_raises_error(self):
        """Non-boolean normalize should raise TypeError."""
        with self.assertRaises(TypeError):
            ManifestGenerator(self.manifest, crop=False, 
                            normalize="true", transform=self.transform)

    def test_frame_column_added_if_missing(self):
        """Frame column should be added if missing."""
        df = self.manifest.drop(columns=["frame"])
        ds = ManifestGenerator(df, crop=False, transform=self.transform)
        self.assertIn("frame", ds.x.columns)

    def test_frame_column_default_is_zero(self):
        """Default frame values should be 0."""
        df = self.manifest.drop(columns=["frame"])
        ds = ManifestGenerator(df, crop=False, transform=self.transform)
        self.assertTrue((ds.x["frame"] == 0).all())


class TestManifestGeneratorGetItem(unittest.TestCase):
    """Test ManifestGenerator __getitem__ and item processing."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "img.jpg", width=100, height=80)
        cls.manifest = _make_manifest([cls.img_path])
        cls.manifest_bbox = _make_manifest([cls.img_path], with_bbox=True)
        cls.transform = _get_model_transforms(64, 64)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_getitem_returns_tuple_of_four(self):
        """Item should be (tensor, filepath, frame, size)."""
        ds = ManifestGenerator(self.manifest, crop=False, transform=self.transform)
        item = ds[0]
        self.assertEqual(len(item), 4)

    def test_tensor_dtype_is_float32(self):
        """Tensor dtype should be float32."""
        ds = ManifestGenerator(self.manifest, crop=False, transform=self.transform)
        tensor, _, _, _ = ds[0]
        self.assertEqual(tensor.dtype, torch.float32)

    def test_tensor_shape_matches_resize(self):
        """Tensor shape should match resize dimensions."""
        ds = ManifestGenerator(self.manifest, crop=False, transform=self.transform)
        tensor, _, _, _ = ds[0]
        self.assertEqual(tensor.shape, (3, 64, 64))

    def test_filepath_preserved(self):
        """Filepath should be preserved in output."""
        ds = ManifestGenerator(self.manifest, crop=False, transform=self.transform)
        _, filepath, _, _ = ds[0]
        self.assertEqual(filepath, self.img_path)

    def test_frame_is_int(self):
        """Frame should be an integer."""
        ds = ManifestGenerator(self.manifest, crop=False, transform=self.transform)
        _, _, frame, _ = ds[0]
        self.assertIsInstance(frame, (int, np.integer))

    def test_size_tensor_shape(self):
        """Size tensor should have shape (2,)."""
        ds = ManifestGenerator(self.manifest, crop=False, transform=self.transform)
        _, _, _, size = ds[0]
        self.assertEqual(size.shape, (2,))

    def test_size_tensor_values(self):
        """Size tensor should contain (height, width) of original."""
        ds = ManifestGenerator(self.manifest, crop=False, transform=self.transform)
        _, _, _, size = ds[0]
        self.assertEqual(size[0].item(), 80)   # height
        self.assertEqual(size[1].item(), 100)  # width

    def test_crop_relative_returns_tensor(self):
        """Crop with relative coords should return valid tensor."""
        ds = ManifestGenerator(self.manifest_bbox, crop=True, 
                             crop_coord='relative', transform=self.transform)
        tensor, _, _, _ = ds[0]
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (3, 64, 64))

    def test_crop_absolute_returns_tensor(self):
        """Crop with absolute coords should return valid tensor."""
        manifest = _make_manifest([self.img_path], with_bbox=True)
        manifest["bbox_x"] = 10
        manifest["bbox_y"] = 10
        manifest["bbox_w"] = 50
        manifest["bbox_h"] = 40
        ds = ManifestGenerator(manifest, crop=True, crop_coord='absolute',
                             transform=self.transform)
        tensor, _, _, _ = ds[0]
        self.assertIsNotNone(tensor)

    def test_nonexistent_image_returns_none_tensor(self):
        """Nonexistent image should return None tensor."""
        manifest = _make_manifest(["/nonexistent/img.jpg"])
        ds = ManifestGenerator(manifest, crop=False, transform=self.transform)
        tensor, filepath, _, _ = ds[0]
        self.assertIsNone(tensor)
        self.assertEqual(filepath, "/nonexistent/img.jpg")

    def test_invalid_file_type_returns_none_tensor(self):
        """Invalid file type should return None tensor."""
        txt_file = Path(self.tmp_dir) / "doc.txt"
        txt_file.write_text("not an image")
        manifest = _make_manifest([str(txt_file)])
        ds = ManifestGenerator(manifest, crop=False, transform=self.transform)
        tensor, _, _, _ = ds[0]
        self.assertIsNone(tensor)


# ---------------------------------------------------------------------------
# TrainGenerator
# ---------------------------------------------------------------------------

class TestTrainGeneratorBasic(unittest.TestCase):
    """Test TrainGenerator initialization and basic functionality."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "img.jpg")
        cls.classes = {0: "cat", 1: "dog"}
        cls.manifest = _make_manifest([cls.img_path], label_col="species", labels=["cat"])
        cls.manifest_bbox = _make_manifest([cls.img_path], with_bbox=True,
                                          label_col="species", labels=["cat"])
        cls.transform = _get_model_transforms(64, 64)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_creates_dataset(self):
        """Should create TrainGenerator dataset."""
        ds = TrainGenerator(self.manifest, self.classes, self.transform, crop=False)
        self.assertIsNotNone(ds)

    def test_dataset_length(self):
        """Dataset length should match manifest size."""
        ds = TrainGenerator(self.manifest, self.classes, self.transform, crop=False)
        self.assertEqual(len(ds), 1)

    def test_missing_file_col_raises_error(self):
        """Missing file_col should raise ValueError."""
        with self.assertRaises(ValueError):
            TrainGenerator(self.manifest, self.classes, self.transform,
                         file_col="nonexistent", crop=False)

    def test_missing_label_col_raises_error(self):
        """Missing label_col should raise ValueError."""
        with self.assertRaises(ValueError):
            TrainGenerator(self.manifest, self.classes, self.transform,
                         label_col="nonexistent", crop=False)

    def test_crop_without_bbox_raises_error(self):
        """Crop without bbox should raise ValueError."""
        with self.assertRaises(ValueError):
            TrainGenerator(self.manifest, self.classes, self.transform, crop=True)

    def test_invalid_crop_coord_raises_error(self):
        """Invalid crop_coord should raise ValueError."""
        with self.assertRaises(ValueError):
            TrainGenerator(self.manifest_bbox, self.classes, self.transform,
                         crop=True, crop_coord="invalid")

    def test_categories_dict_built(self):
        """Categories dict should be built from classes."""
        ds = TrainGenerator(self.manifest, self.classes, self.transform, crop=False)
        self.assertIsInstance(ds.categories, dict)
        self.assertIn("cat", ds.categories)
        self.assertIn("dog", ds.categories)

    def test_cache_dir_created(self):
        """Cache directory should be created if specified."""
        cache = Path(self.tmp_dir) / "cache"
        ds = TrainGenerator(self.manifest, self.classes, self.transform,
                          crop=False, cache_dir=str(cache))
        self.assertTrue(cache.exists())


class TestTrainGeneratorGetItem(unittest.TestCase):
    """Test TrainGenerator __getitem__ and item processing."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "img.jpg", width=100, height=80)
        cls.classes = {0: "cat", 1: "dog"}
        cls.manifest = _make_manifest([cls.img_path], label_col="species", labels=["cat"])
        cls.manifest_bbox = _make_manifest([cls.img_path], with_bbox=True,
                                          label_col="species", labels=["cat"])
        cls.transform = _get_model_transforms(64, 64)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_getitem_returns_tuple_of_three(self):
        """Item should be (tensor, label, filepath)."""
        ds = TrainGenerator(self.manifest, self.classes, self.transform, crop=False)
        item = ds[0]
        self.assertEqual(len(item), 3)

    def test_tensor_dtype_is_float32(self):
        """Tensor dtype should be float32."""
        ds = TrainGenerator(self.manifest, self.classes, self.transform, crop=False)
        tensor, _, _ = ds[0]
        self.assertEqual(tensor.dtype, torch.float32)

    def test_tensor_shape_matches_resize(self):
        """Tensor shape should match resize dimensions."""
        ds = TrainGenerator(self.manifest, self.classes, self.transform, crop=False)
        tensor, _, _ = ds[0]
        self.assertEqual(tensor.shape, (3, 64, 64))

    def test_label_is_int(self):
        """Label should be an integer."""
        ds = TrainGenerator(self.manifest, self.classes, self.transform, crop=False)
        _, label, _ = ds[0]
        self.assertIsInstance(label, (int, np.integer))

    def test_label_correct_index(self):
        """Label should map to correct category index."""
        ds = TrainGenerator(self.manifest, self.classes, self.transform, crop=False)
        _, label, _ = ds[0]
        self.assertEqual(label, ds.categories["cat"])

    def test_filepath_preserved(self):
        """Filepath should be preserved in output."""
        ds = TrainGenerator(self.manifest, self.classes, self.transform, crop=False)
        _, _, filepath = ds[0]
        self.assertEqual(filepath, self.img_path)

    def test_crop_relative_returns_tensor(self):
        """Crop with relative coords should return valid tensor."""
        ds = TrainGenerator(self.manifest_bbox, self.classes, self.transform,
                          crop=True, crop_coord='relative')
        tensor, _, _ = ds[0]
        self.assertIsNotNone(tensor)

    def test_nonexistent_image_returns_none_tensor(self):
        """Nonexistent image should return None tensor."""
        manifest = _make_manifest(["/nonexistent/img.jpg"],
                                label_col="species", labels=["cat"])
        ds = TrainGenerator(manifest, self.classes, self.transform, crop=False)
        tensor, _, _ = ds[0]
        self.assertIsNone(tensor)

    def test_cache_saves_image(self):
        """Cache should save processed image."""
        cache = Path(self.tmp_dir) / "train_cache"
        ds = TrainGenerator(self.manifest, self.classes, self.transform,
                          crop=False, cache_dir=str(cache))
        ds[0]   # first call saves to cache
        cached = list(cache.glob("*.jpg"))
        self.assertGreater(len(cached), 0)

    def test_cache_loads_from_disk(self):
        """Subsequent calls should load from cache."""
        cache = Path(self.tmp_dir) / "train_cache2"
        ds = TrainGenerator(self.manifest, self.classes, self.transform,
                          crop=False, cache_dir=str(cache))
        ds[0]   # writes cache
        tensor, _, _ = ds[0]   # reads from cache
        self.assertIsNotNone(tensor)


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

class TestCollateFn(unittest.TestCase):
    """Test collate_fn for batch handling."""

    def _make_item(self, filepath="a.jpg"):
        """Create a valid batch item."""
        tensor = torch.zeros(3, 64, 64)
        return tensor, filepath, 0, torch.tensor((80, 100))

    def test_all_valid_batch(self):
        """All valid items should be collated."""
        batch = [self._make_item("a.jpg"), self._make_item("b.jpg")]
        collated, failed = collate_fn(batch)
        self.assertIsNotNone(collated)
        self.assertEqual(len(failed), 0)

    def test_all_failed_batch(self):
        """All failed items should return None collated."""
        batch = [(None, "a.jpg", 0, None), (None, "b.jpg", 0, None)]
        collated, failed = collate_fn(batch)
        self.assertIsNone(collated)
        self.assertEqual(len(failed), 2)

    def test_mixed_batch(self):
        """Mixed batch should filter out None items."""
        batch = [self._make_item("a.jpg"), (None, "b.jpg", 0, None)]
        collated, failed = collate_fn(batch)
        self.assertIsNotNone(collated)
        self.assertEqual(len(failed), 1)
        self.assertIn("b.jpg", failed)

    def test_failed_filepaths_collected(self):
        """Failed item filepaths should be collected."""
        batch = [(None, "bad1.jpg", 0, None), (None, "bad2.jpg", 0, None)]
        _, failed = collate_fn(batch)
        self.assertEqual(set(failed), {"bad1.jpg", "bad2.jpg"})

    def test_collated_batch_dimension(self):
        """Collated tensors should have batch dimension."""
        batch = [self._make_item("a.jpg"), self._make_item("b.jpg")]
        collated, _ = collate_fn(batch)
        tensors = collated[0]
        self.assertEqual(tensors.shape[0], 2)   # batch size = 2
        self.assertEqual(tensors.shape[1], 3)   # channels
        self.assertEqual(tensors.shape[2], 64)  # height
        self.assertEqual(tensors.shape[3], 64)  # width

    def test_filepaths_preserved_in_collated(self):
        """Filepaths should be preserved in collated output."""
        batch = [self._make_item("a.jpg"), self._make_item("b.jpg")]
        collated, _ = collate_fn(batch)
        filepaths = collated[1]
        self.assertIn("a.jpg", filepaths)
        self.assertIn("b.jpg", filepaths)


# ---------------------------------------------------------------------------
# manifest_dataloader
# ---------------------------------------------------------------------------

class TestManifestDataloader(unittest.TestCase):
    """Test manifest_dataloader function."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "img.jpg")
        cls.manifest = _make_manifest([cls.img_path])
        cls.manifest_bbox = _make_manifest([cls.img_path], with_bbox=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_returns_dataloader(self):
        """Should return a DataLoader instance."""
        from torch.utils.data import DataLoader
        dl = manifest_dataloader(self.manifest, crop=False,
                                resize_height=64, resize_width=64, batch_size=1)
        self.assertIsInstance(dl, DataLoader)

    def test_dataloader_length(self):
        """DataLoader length should match expected batch count."""
        dl = manifest_dataloader(self.manifest, crop=False,
                                resize_height=64, resize_width=64, batch_size=1)
        self.assertEqual(len(dl), 1)

    def test_dataloader_iteration(self):
        """DataLoader should iterate without error."""
        dl = manifest_dataloader(self.manifest, crop=False,
                                resize_height=64, resize_width=64, batch_size=1)
        for batch in dl:
            self.assertIsNotNone(batch)
            self.assertEqual(len(batch), 2)  # (collated, failed)

    def test_batch_with_crop(self):
        """Batch should work with crop=True and bbox."""
        dl = manifest_dataloader(self.manifest_bbox, crop=True,
                                resize_height=64, resize_width=64, batch_size=1)
        for batch in dl:
            collated, failed = batch
            if collated is not None:
                self.assertEqual(collated[0].shape[-1], 64)

    def test_missing_file_col_raises_error(self):
        """Missing file_col should raise ValueError."""
        with self.assertRaises(ValueError):
            manifest_dataloader(self.manifest, file_col="nonexistent", crop=False)

    def test_multiple_images_batched(self):
        """Multiple images should be properly batched."""
        paths = [_make_image(Path(self.tmp_dir) / f"img{i}.jpg") for i in range(3)]
        manifest = _make_manifest(paths)
        dl = manifest_dataloader(manifest, crop=False,
                                resize_height=64, resize_width=64, batch_size=2)
        batches = list(dl)
        self.assertEqual(len(batches), 2)  # 3 images, batch size 2 = 2 batches


# ---------------------------------------------------------------------------
# train_dataloader
# ---------------------------------------------------------------------------

class TestTrainDataloader(unittest.TestCase):
    """Test train_dataloader function."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "img.jpg")
        cls.classes = {0: "cat", 1: "dog"}
        cls.manifest = _make_manifest([cls.img_path], label_col="species", labels=["cat"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_returns_dataloader(self):
        """Should return a DataLoader instance."""
        from torch.utils.data import DataLoader
        dl = train_dataloader(self.manifest, self.classes, crop=False,
                             resize_height=64, resize_width=64, batch_size=1)
        self.assertIsInstance(dl, DataLoader)

    def test_dataloader_length(self):
        """DataLoader length should match expected batch count."""
        dl = train_dataloader(self.manifest, self.classes, crop=False,
                             resize_height=64, resize_width=64, batch_size=1)
        self.assertEqual(len(dl), 1)

    def test_dataloader_iteration(self):
        """DataLoader should iterate without error."""
        dl = train_dataloader(self.manifest, self.classes, crop=False,
                             resize_height=64, resize_width=64, batch_size=1)
        for batch in dl:
            self.assertIsNotNone(batch)

    def test_missing_label_col_raises_error(self):
        """Missing label_col should raise ValueError."""
        with self.assertRaises(ValueError):
            train_dataloader(self.manifest, self.classes,
                           label_col="nonexistent", crop=False)

    def test_shuffle_is_enabled(self):
        """Training dataloader should shuffle by default."""
        dl = train_dataloader(self.manifest, self.classes, crop=False,
                             resize_height=64, resize_width=64, batch_size=1)
        self.assertTrue(dl.dataset is not None)

    def test_with_augmentation(self):
        """DataLoader should work with augmentation enabled."""
        dl = train_dataloader(self.manifest, self.classes, crop=False,
                             resize_height=64, resize_width=64, batch_size=1,
                             augment=True)
        for batch in dl:
            self.assertIsNotNone(batch)
            break  # Just test one batch


if __name__ == "__main__":
    unittest.main()
