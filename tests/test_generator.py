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

    def test_square_output_from_wide_image(self):
        """Wide image should be padded and resized to square."""
        lb = Letterbox(resize_height=64, resize_width=64)
        img = Image.new("RGB", (128, 64))   # 2:1 aspect ratio
        result = lb(img)
        self.assertIsNotNone(result)

    def test_square_output_from_tall_image(self):
        """Tall image should be padded and resized to square."""
        lb = Letterbox(resize_height=64, resize_width=64)
        img = Image.new("RGB", (64, 128))   # 1:2 aspect ratio
        result = lb(img)
        self.assertIsNotNone(result)

    def test_same_aspect_ratio_no_padding(self):
        """Image with matching aspect ratio should just be resized."""
        lb = Letterbox(resize_height=64, resize_width=64)
        img = Image.new("RGB", (64, 64))    # 1:1 aspect ratio
        result = lb(img)
        self.assertIsNotNone(result)

    def test_output_is_pil_image(self):
        lb = Letterbox(resize_height=64, resize_width=64)
        img = Image.new("RGB", (100, 80))
        result = lb(img)
        self.assertIsInstance(result, Image.Image)


# ---------------------------------------------------------------------------
# image_to_tensor
# ---------------------------------------------------------------------------

class TestImageToTensor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "test.jpg")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_returns_tuple(self):
        result = image_to_tensor(self.img_path, letterbox=False,
                                 resize_width=64, resize_height=64)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_tensor_shape(self):
        tensor, _, _ = image_to_tensor(self.img_path, letterbox=False,
                                       resize_width=64, resize_height=64)
        # (1, 3, H, W)
        self.assertEqual(tensor.shape, (1, 3, 64, 64))

    def test_tensor_dtype_float32(self):
        tensor, _, _ = image_to_tensor(self.img_path, letterbox=False,
                                       resize_width=64, resize_height=64)
        self.assertEqual(tensor.dtype, torch.float32)

    def test_tensor_values_normalised(self):
        """Values should be in [0, 1]."""
        tensor, _, _ = image_to_tensor(self.img_path, letterbox=False,
                                       resize_width=64, resize_height=64)
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_filepath_preserved(self):
        _, paths, _ = image_to_tensor(self.img_path, letterbox=False,
                                      resize_width=64, resize_height=64)
        self.assertEqual(paths[0], self.img_path)

    def test_size_tensor_correct(self):
        _, _, sizes = image_to_tensor(self.img_path, letterbox=False,
                                      resize_width=64, resize_height=64)
        # original image is 100w x 80h → size tensor should be (80, 100)
        self.assertEqual(sizes[0][0].item(), 80)   # height
        self.assertEqual(sizes[0][1].item(), 100)  # width

    def test_letterbox_true_returns_result(self):
        result = image_to_tensor(self.img_path, letterbox=True,
                                 resize_width=64, resize_height=64)
        self.assertIsNotNone(result)

    def test_nonexistent_file_returns_none(self):
        result = image_to_tensor("/nonexistent/file.jpg", letterbox=False,
                                 resize_width=64, resize_height=64)
        self.assertIsNone(result)

    def test_invalid_file_returns_none(self):
        bad = Path(self.tmp_dir) / "bad.jpg"
        bad.write_text("not an image")
        result = image_to_tensor(str(bad), letterbox=False,
                                 resize_width=64, resize_height=64)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# ManifestGenerator
# ---------------------------------------------------------------------------

class TestManifestGeneratorInit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "img.jpg")
        cls.manifest = _make_manifest([cls.img_path])
        cls.manifest_bbox = _make_manifest([cls.img_path], with_bbox=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_creates_dataset(self):
        ds = ManifestGenerator(self.manifest, crop=False)
        self.assertIsNotNone(ds)

    def test_len(self):
        ds = ManifestGenerator(self.manifest, crop=False)
        self.assertEqual(len(ds), 1)

    def test_missing_file_col_raises(self):
        with self.assertRaises(ValueError):
            ManifestGenerator(self.manifest, file_col="nonexistent", crop=False)

    def test_crop_without_bbox_raises(self):
        with self.assertRaises(ValueError):
            ManifestGenerator(self.manifest, crop=True)

    def test_invalid_crop_coord_raises(self):
        with self.assertRaises(ValueError):
            ManifestGenerator(self.manifest_bbox, crop=True, crop_coord="diagonal")

    def test_invalid_crop_type_raises(self):
        with self.assertRaises(TypeError):
            ManifestGenerator(self.manifest, crop=False, normalize="true")

    def test_invalid_letterbox_type_raises(self):
        with self.assertRaises(TypeError):
            ManifestGenerator(self.manifest, crop=False, letterbox=1)

    def test_frame_column_added_if_missing(self):
        df = self.manifest.drop(columns=["frame"])
        ds = ManifestGenerator(df, crop=False)
        self.assertIn("frame", ds.x.columns)

    def test_default_frame_is_zero(self):
        df = self.manifest.drop(columns=["frame"])
        ds = ManifestGenerator(df, crop=False)
        self.assertTrue((ds.x["frame"] == 0).all())


class TestManifestGeneratorGetItem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "img.jpg", width=100, height=80)
        cls.manifest = _make_manifest([cls.img_path])
        cls.manifest_bbox = _make_manifest([cls.img_path], with_bbox=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_getitem_returns_tuple_of_four(self):
        ds = ManifestGenerator(self.manifest, crop=False,
                               resize_height=64, resize_width=64)
        item = ds[0]
        self.assertEqual(len(item), 4)

    def test_tensor_is_float32(self):
        ds = ManifestGenerator(self.manifest, crop=False,
                               resize_height=64, resize_width=64)
        tensor, _, _, _ = ds[0]
        self.assertEqual(tensor.dtype, torch.float32)

    def test_tensor_shape(self):
        ds = ManifestGenerator(self.manifest, crop=False,
                               resize_height=64, resize_width=64)
        tensor, _, _, _ = ds[0]
        self.assertEqual(tensor.shape, (3, 64, 64))

    def test_filepath_preserved(self):
        ds = ManifestGenerator(self.manifest, crop=False,
                               resize_height=64, resize_width=64)
        _, path, _, _ = ds[0]
        self.assertEqual(path, self.img_path)

    def test_frame_is_int(self):
        ds = ManifestGenerator(self.manifest, crop=False,
                               resize_height=64, resize_width=64)
        _, _, frame, _ = ds[0]
        self.assertIsInstance(frame, int)

    def test_size_tensor_shape(self):
        ds = ManifestGenerator(self.manifest, crop=False,
                               resize_height=64, resize_width=64)
        _, _, _, size = ds[0]
        self.assertEqual(size.shape, (2,))

    def test_normalize_false_scales_to_255(self):
        ds = ManifestGenerator(self.manifest, crop=False, normalize=False,
                               resize_height=64, resize_width=64)
        tensor, _, _, _ = ds[0]
        self.assertGreater(tensor.max().item(), 1.0)

    def test_normalize_true_values_in_0_1(self):
        ds = ManifestGenerator(self.manifest, crop=False, normalize=True,
                               resize_height=64, resize_width=64)
        tensor, _, _, _ = ds[0]
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_crop_relative_returns_tensor(self):
        ds = ManifestGenerator(self.manifest_bbox, crop=True, crop_coord='relative',
                               resize_height=64, resize_width=64)
        tensor, _, _, _ = ds[0]
        self.assertIsNotNone(tensor)

    def test_crop_absolute_returns_tensor(self):
        manifest = _make_manifest([self.img_path], with_bbox=True)
        manifest["bbox_x"] = 10
        manifest["bbox_y"] = 10
        manifest["bbox_w"] = 50
        manifest["bbox_h"] = 40
        ds = ManifestGenerator(manifest, crop=True, crop_coord='absolute',
                               resize_height=64, resize_width=64)
        tensor, _, _, _ = ds[0]
        self.assertIsNotNone(tensor)

    def test_nonexistent_image_returns_none_tensor(self):
        manifest = _make_manifest(["/nonexistent/img.jpg"])
        ds = ManifestGenerator(manifest, crop=False,
                               resize_height=64, resize_width=64)
        tensor, path, _, _ = ds[0]
        self.assertIsNone(tensor)
        self.assertEqual(path, "/nonexistent/img.jpg")

    def test_unsupported_extension_returns_none_tensor(self):
        txt_file = Path(self.tmp_dir) / "doc.txt"
        txt_file.write_text("not an image")
        manifest = _make_manifest([str(txt_file)])
        ds = ManifestGenerator(manifest, crop=False,
                               resize_height=64, resize_width=64)
        tensor, _, _, _ = ds[0]
        self.assertIsNone(tensor)

    def test_letterbox_returns_tensor(self):
        ds = ManifestGenerator(self.manifest, crop=False, letterbox=True,
                               resize_height=64, resize_width=64)
        tensor, _, _, _ = ds[0]
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (3, 64, 64))


# ---------------------------------------------------------------------------
# TrainGenerator
# ---------------------------------------------------------------------------

class TestTrainGeneratorInit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "img.jpg")
        cls.classes = ["cat", "dog"]
        cls.manifest = _make_manifest(
            [cls.img_path], label_col="species", labels=["cat"]
        )
        cls.manifest_bbox = _make_manifest(
            [cls.img_path], with_bbox=True, label_col="species", labels=["cat"]
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_creates_dataset(self):
        ds = TrainGenerator(self.manifest, self.classes, crop=False)
        self.assertIsNotNone(ds)

    def test_len(self):
        ds = TrainGenerator(self.manifest, self.classes, crop=False)
        self.assertEqual(len(ds), 1)

    def test_missing_file_col_raises(self):
        with self.assertRaises(ValueError):
            TrainGenerator(self.manifest, self.classes,
                           file_col="nonexistent", crop=False)

    def test_missing_label_col_raises(self):
        with self.assertRaises(ValueError):
            TrainGenerator(self.manifest, self.classes,
                           label_col="nonexistent", crop=False)

    def test_crop_without_bbox_raises(self):
        with self.assertRaises(ValueError):
            TrainGenerator(self.manifest, self.classes, crop=True)

    def test_invalid_crop_coord_raises(self):
        with self.assertRaises(ValueError):
            TrainGenerator(self.manifest_bbox, self.classes,
                           crop=True, crop_coord="diagonal")

    def test_categories_dict_built(self):
        ds = TrainGenerator(self.manifest, self.classes, crop=False)
        self.assertIsInstance(ds.categories, dict)
        self.assertIn("cat", ds.categories)
        self.assertIn("dog", ds.categories)

    def test_cache_dir_created(self):
        cache = Path(self.tmp_dir) / "cache"
        ds = TrainGenerator(self.manifest, self.classes,
                            crop=False, cache_dir=str(cache))
        self.assertTrue(cache.exists())


class TestTrainGeneratorGetItem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "img.jpg", width=100, height=80)
        cls.classes = ["cat", "dog"]
        cls.manifest = _make_manifest(
            [cls.img_path], label_col="species", labels=["cat"]
        )
        cls.manifest_bbox = _make_manifest(
            [cls.img_path], with_bbox=True, label_col="species", labels=["cat"]
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_getitem_returns_tuple_of_three(self):
        ds = TrainGenerator(self.manifest, self.classes, crop=False,
                            resize_height=64, resize_width=64)
        item = ds[0]
        self.assertEqual(len(item), 3)

    def test_tensor_dtype_float32(self):
        ds = TrainGenerator(self.manifest, self.classes, crop=False,
                            resize_height=64, resize_width=64)
        tensor, _, _ = ds[0]
        self.assertEqual(tensor.dtype, torch.float32)

    def test_tensor_shape(self):
        ds = TrainGenerator(self.manifest, self.classes, crop=False,
                            resize_height=64, resize_width=64)
        tensor, _, _ = ds[0]
        self.assertEqual(tensor.shape, (3, 64, 64))

    def test_label_is_int(self):
        ds = TrainGenerator(self.manifest, self.classes, crop=False,
                            resize_height=64, resize_width=64)
        _, label, _ = ds[0]
        self.assertIsInstance(label, int)

    def test_label_correct_index(self):
        ds = TrainGenerator(self.manifest, self.classes, crop=False,
                            resize_height=64, resize_width=64)
        _, label, _ = ds[0]
        self.assertEqual(label, ds.categories["cat"])

    def test_filepath_preserved(self):
        ds = TrainGenerator(self.manifest, self.classes, crop=False,
                            resize_height=64, resize_width=64)
        _, _, path = ds[0]
        self.assertEqual(path, self.img_path)

    def test_crop_relative_returns_tensor(self):
        ds = TrainGenerator(self.manifest_bbox, self.classes,
                            crop=True, crop_coord='relative',
                            resize_height=64, resize_width=64)
        tensor, _, _ = ds[0]
        self.assertIsNotNone(tensor)

    def test_nonexistent_image_returns_none_tensor(self):
        manifest = _make_manifest(["/nonexistent/img.jpg"],
                                  label_col="species", labels=["cat"])
        ds = TrainGenerator(manifest, self.classes, crop=False,
                            resize_height=64, resize_width=64)
        tensor, _, _ = ds[0]
        self.assertIsNone(tensor)

    def test_cache_saves_image(self):
        cache = Path(self.tmp_dir) / "train_cache"
        ds = TrainGenerator(self.manifest, self.classes, crop=False,
                            resize_height=64, resize_width=64,
                            cache_dir=str(cache))
        ds[0]   # first call saves to cache
        cached = list(cache.glob("*.jpg"))
        self.assertGreater(len(cached), 0)

    def test_cache_loads_from_disk(self):
        """Second call should load from cache without error."""
        cache = Path(self.tmp_dir) / "train_cache2"
        ds = TrainGenerator(self.manifest, self.classes, crop=False,
                            resize_height=64, resize_width=64,
                            cache_dir=str(cache))
        ds[0]   # writes cache
        tensor, _, _ = ds[0]   # reads from cache
        self.assertIsNotNone(tensor)


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

class TestCollateFn(unittest.TestCase):

    def _make_item(self, filepath="a.jpg"):
        tensor = torch.zeros(3, 64, 64)
        return tensor, filepath, 0, torch.tensor((80, 100))

    def test_all_good_batch(self):
        batch = [self._make_item("a.jpg"), self._make_item("b.jpg")]
        collated, failed = collate_fn(batch)
        self.assertIsNotNone(collated)
        self.assertEqual(len(failed), 0)

    def test_all_failed_batch(self):
        batch = [(None, "a.jpg", 0, None), (None, "b.jpg", 0, None)]
        collated, failed = collate_fn(batch)
        self.assertIsNone(collated)
        self.assertIn("a.jpg", failed)
        self.assertIn("b.jpg", failed)

    def test_mixed_batch_filters_none(self):
        batch = [self._make_item("a.jpg"), (None, "b.jpg", 0, None)]
        collated, failed = collate_fn(batch)
        self.assertIsNotNone(collated)
        self.assertIn("b.jpg", failed)

    def test_failed_paths_collected(self):
        batch = [(None, "bad1.jpg", 0, None), (None, "bad2.jpg", 0, None)]
        _, failed = collate_fn(batch)
        self.assertEqual(set(failed), {"bad1.jpg", "bad2.jpg"})

    def test_collated_tensor_stacked(self):
        batch = [self._make_item("a.jpg"), self._make_item("b.jpg")]
        collated, _ = collate_fn(batch)
        tensors = collated[0]
        self.assertEqual(tensors.shape[0], 2)   # batch dimension = 2


# ---------------------------------------------------------------------------
# manifest_dataloader
# ---------------------------------------------------------------------------

class TestManifestDataloader(unittest.TestCase):

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
        from torch.utils.data import DataLoader
        dl = manifest_dataloader(self.manifest, crop=False,
                                 resize_height=64, resize_width=64, batch_size=1)
        self.assertIsInstance(dl, DataLoader)

    def test_dataloader_length(self):
        dl = manifest_dataloader(self.manifest, crop=False,
                                 resize_height=64, resize_width=64, batch_size=1)
        self.assertEqual(len(dl), 1)

    def test_iterates_without_error(self):
        dl = manifest_dataloader(self.manifest, crop=False,
                                 resize_height=64, resize_width=64, batch_size=1)
        for batch in dl:
            self.assertIsNotNone(batch)

    def test_crop_true_with_bbox(self):
        dl = manifest_dataloader(self.manifest_bbox, crop=True,
                                 resize_height=64, resize_width=64, batch_size=1)
        for batch in dl:
            collated, failed = batch
            if collated is not None:
                self.assertEqual(collated[0].shape[-1], 64)

    def test_missing_file_col_raises(self):
        with self.assertRaises(ValueError):
            manifest_dataloader(self.manifest, file_col="nonexistent", crop=False)

    def test_batch_contains_tensor_and_failed(self):
        dl = manifest_dataloader(self.manifest, crop=False,
                                 resize_height=64, resize_width=64, batch_size=1)
        batch = next(iter(dl))
        self.assertEqual(len(batch), 2)  # (collated, failed)


# ---------------------------------------------------------------------------
# train_dataloader
# ---------------------------------------------------------------------------

class TestTrainDataloader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.img_path = _make_image(Path(cls.tmp_dir) / "img.jpg")
        cls.classes = ["cat", "dog"]
        cls.manifest = _make_manifest(
            [cls.img_path], label_col="species", labels=["cat"]
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_returns_dataloader(self):
        from torch.utils.data import DataLoader
        dl = train_dataloader(self.manifest, self.classes, crop=False,
                              resize_height=64, resize_width=64, batch_size=1)
        self.assertIsInstance(dl, DataLoader)

    def test_dataloader_length(self):
        dl = train_dataloader(self.manifest, self.classes, crop=False,
                              resize_height=64, resize_width=64, batch_size=1)
        self.assertEqual(len(dl), 1)

    def test_iterates_without_error(self):
        dl = train_dataloader(self.manifest, self.classes, crop=False,
                              resize_height=64, resize_width=64, batch_size=1)
        for batch in dl:
            self.assertIsNotNone(batch)

    def test_missing_label_col_raises(self):
        with self.assertRaises(ValueError):
            train_dataloader(self.manifest, self.classes,
                             label_col="nonexistent", crop=False)

    def test_shuffle_is_true(self):
        dl = train_dataloader(self.manifest, self.classes, crop=False,
                              resize_height=64, resize_width=64, batch_size=1)
        self.assertTrue(dl.dataset is not None)


if __name__ == "__main__":
    unittest.main()