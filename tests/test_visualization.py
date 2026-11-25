"""
Unit tests for animl/utils/visualization.py

"""
import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from animl.utils import visualization
from animl.utils.visualization import MD_COLORS, MD_LABELS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Path to the example images shipped with the repository
_EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "Southwest"

# Pick a couple of real .JPG images so image-loading tests work
_SAMPLE_IMAGES = sorted(_EXAMPLES_DIR.glob("*.JPG"))[:2]


def _make_df(filepath, category=1, conf=0.9, bbox_x=0.1, bbox_y=0.1,
             bbox_w=0.3, bbox_h=0.3):
    """Return a single-row DataFrame with the minimum required columns."""
    return pd.DataFrame([{
        "filepath": str(filepath),
        "category": category,
        "conf": conf,
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
    }])


def _make_series(filepath, **kwargs):
    """Return a single pandas Series (as passed by plot_from_file)."""
    return _make_df(filepath, **kwargs).iloc[0]


# ---------------------------------------------------------------------------
# TestPlotBox
# ---------------------------------------------------------------------------

class TestPlotBox(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Require at least one sample image; skip the class otherwise."""
        if not _SAMPLE_IMAGES:
            raise unittest.SkipTest("No sample images found in examples/Southwest/")
        cls.image_path = _SAMPLE_IMAGES[0]
        if len(_SAMPLE_IMAGES) >= 2:
            cls.image_path2 = _SAMPLE_IMAGES[1]
        else:
            cls.image_path2 = cls.image_path

    # --- positive cases ---

    def test_single_row_series_returns_image(self):
        """plot_box accepts a pandas Series and returns an ndarray when return_img=True."""
        row = _make_series(self.image_path)
        img = visualization.plot_box(row, return_img=True)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.ndim, 3)  # H x W x C

    def test_single_row_dataframe_returns_image(self):
        """plot_box accepts a single-row DataFrame and returns an ndarray."""
        df = _make_df(self.image_path)
        img = visualization.plot_box(df, return_img=True)
        self.assertIsInstance(img, np.ndarray)

    def test_multiple_rows_returns_image(self):
        """plot_box processes a DataFrame with multiple rows (same filepath)."""
        df = pd.DataFrame([
            {"filepath": str(self.image_path), "category": 1, "conf": 0.9,
             "bbox_x": 0.1, "bbox_y": 0.1, "bbox_w": 0.2, "bbox_h": 0.2},
            {"filepath": str(self.image_path), "category": 2, "conf": 0.8,
             "bbox_x": 0.5, "bbox_y": 0.5, "bbox_w": 0.2, "bbox_h": 0.2},
        ])
        img = visualization.plot_box(df, return_img=True)
        self.assertIsInstance(img, np.ndarray)

    def test_image_is_modified_by_drawing(self):
        """The returned image should differ from the original (boxes were drawn)."""
        original = cv2.imread(str(self.image_path))
        df = _make_df(self.image_path)
        result = visualization.plot_box(df, return_img=True)
        # At least some pixels must have changed due to the drawn rectangle
        self.assertFalse(np.array_equal(original, result))

    def test_min_conf_filters_boxes(self):
        """Rows below min_conf should be skipped; image should still be returned."""
        # All rows have conf=0.05, which is below min_conf=0.5 → no boxes drawn
        df = _make_df(self.image_path, conf=0.05)
        img = visualization.plot_box(df, min_conf=0.5, return_img=True)
        self.assertIsInstance(img, np.ndarray)

    def test_classifier_label_col_none(self):
        """classifier_label_col=None should work without printing any label."""
        df = _make_df(self.image_path)
        img = visualization.plot_box(df, classifier_label_col=None, return_img=True)
        self.assertIsInstance(img, np.ndarray)

    def test_classifier_label_col_category(self):
        """classifier_label_col='category' should print the detector label."""
        df = _make_df(self.image_path)
        img = visualization.plot_box(df, classifier_label_col="category", return_img=True)
        self.assertIsInstance(img, np.ndarray)

    def test_classifier_label_col_custom_column(self):
        """A custom label column should be printed when specified."""
        df = _make_df(self.image_path)
        df["prediction"] = "deer"
        img = visualization.plot_box(df, classifier_label_col="prediction", return_img=True)
        self.assertIsInstance(img, np.ndarray)

    def test_show_confidence_true(self):
        """show_confidence=True should still return a valid image."""
        df = _make_df(self.image_path)
        img = visualization.plot_box(df, classifier_label_col="category",
                                     show_confidence=True, return_img=True)
        self.assertIsInstance(img, np.ndarray)

    def test_show_confidence_false(self):
        """show_confidence=False (default) should return a valid image."""
        df = _make_df(self.image_path)
        img = visualization.plot_box(df, classifier_label_col="category",
                                     show_confidence=False, return_img=True)
        self.assertIsInstance(img, np.ndarray)

    def test_nan_bbox_row_is_skipped(self):
        """Rows with NaN bbox coordinates should be silently skipped."""
        df = pd.DataFrame([
            {"filepath": str(self.image_path), "category": 1, "conf": 0.9,
             "bbox_x": float("nan"), "bbox_y": 0.1, "bbox_w": 0.2, "bbox_h": 0.2},
        ])
        img = visualization.plot_box(df, return_img=True)
        self.assertIsInstance(img, np.ndarray)

    def test_custom_colors_and_labels(self):
        """Custom colors and detector_labels dicts should be accepted."""
        colors = {1: (128, 0, 128)}
        labels = {1: "custom_animal"}
        df = _make_df(self.image_path, category=1)
        img = visualization.plot_box(df, colors=colors, detector_labels=labels,
                                     return_img=True)
        self.assertIsInstance(img, np.ndarray)

    def test_colors_none_falls_back_to_defaults(self):
        """Passing colors=None should fall back to MD_COLORS."""
        df = _make_df(self.image_path)
        img = visualization.plot_box(df, colors=None, detector_labels=None,
                                     return_img=True)
        self.assertIsInstance(img, np.ndarray)

    # --- negative / error cases ---

    def test_missing_required_column_raises_value_error(self):
        """A DataFrame missing required columns should raise ValueError."""
        df = pd.DataFrame([{
            "filepath": str(self.image_path),
            "category": 1,
            # 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h' deliberately omitted
        }])
        with self.assertRaises(ValueError):
            visualization.plot_box(df, return_img=True)

    def test_nonexistent_file_raises_file_not_found(self):
        """A path that does not exist should raise FileNotFoundError."""
        df = _make_df("/nonexistent/path/image.jpg")
        with self.assertRaises(FileNotFoundError):
            visualization.plot_box(df, return_img=True)


# ---------------------------------------------------------------------------
# TestPlotAllBoundingBoxes
# ---------------------------------------------------------------------------

class TestPlotAllBoundingBoxes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not _SAMPLE_IMAGES:
            raise unittest.SkipTest("No sample images found in examples/Southwest/")
        cls.image_path = _SAMPLE_IMAGES[0]

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_manifest(self, filepath=None, category=1, conf=0.9):
        fp = filepath or self.image_path
        return pd.DataFrame([{
            "filepath": str(fp),
            "category": category,
            "conf": conf,
            "bbox_x": 0.1,
            "bbox_y": 0.1,
            "bbox_w": 0.3,
            "bbox_h": 0.3,
        }])

    # --- positive cases ---

    def test_creates_output_directory(self):
        """Output directory is created if it does not exist."""
        new_dir = Path(self.temp_dir) / "new_output"
        self.assertFalse(new_dir.exists())
        manifest = self._make_manifest()
        visualization.plot_all_bounding_boxes(manifest, str(new_dir))
        self.assertTrue(new_dir.exists())

    def test_output_file_created_for_image(self):
        """A _box.jpg file should be written for each unique image filepath."""
        manifest = self._make_manifest()
        visualization.plot_all_bounding_boxes(manifest, self.temp_dir)
        stem = Path(self.image_path).stem
        expected = Path(self.temp_dir) / f"{stem}_box.jpg"
        self.assertTrue(expected.exists(), f"Expected output file not found: {expected}")

    def test_output_file_is_valid_image(self):
        """The written _box.jpg should be readable as an image."""
        manifest = self._make_manifest()
        visualization.plot_all_bounding_boxes(manifest, self.temp_dir)
        stem = Path(self.image_path).stem
        out_path = Path(self.temp_dir) / f"{stem}_box.jpg"
        img = cv2.imread(str(out_path))
        self.assertIsNotNone(img)
        self.assertEqual(img.ndim, 3)

    def test_multiple_images_produce_multiple_files(self):
        """Each unique filepath in the manifest gets its own output file."""
        if len(_SAMPLE_IMAGES) < 2:
            self.skipTest("Need at least 2 sample images")
        manifest = pd.DataFrame([
            {"filepath": str(_SAMPLE_IMAGES[0]), "category": 1, "conf": 0.9,
             "bbox_x": 0.1, "bbox_y": 0.1, "bbox_w": 0.3, "bbox_h": 0.3},
            {"filepath": str(_SAMPLE_IMAGES[1]), "category": 1, "conf": 0.8,
             "bbox_x": 0.2, "bbox_y": 0.2, "bbox_w": 0.3, "bbox_h": 0.3},
        ])
        visualization.plot_all_bounding_boxes(manifest, self.temp_dir)
        for img_path in _SAMPLE_IMAGES[:2]:
            expected = Path(self.temp_dir) / f"{img_path.stem}_box.jpg"
            self.assertTrue(expected.exists(), f"Missing: {expected}")

    def test_min_conf_filters_detections(self):
        """Confidence threshold should be respected (function should not crash)."""
        manifest = self._make_manifest(conf=0.05)
        visualization.plot_all_bounding_boxes(manifest, self.temp_dir, min_conf=0.5)
        # File is still written (image with no boxes drawn)
        stem = Path(self.image_path).stem
        out = Path(self.temp_dir) / f"{stem}_box.jpg"
        self.assertTrue(out.exists())

    def test_custom_colors_and_labels(self):
        """Custom colors and detector_labels are accepted without error."""
        colors = {1: (200, 100, 50)}
        labels = {1: "test_animal"}
        manifest = self._make_manifest(category=1)
        visualization.plot_all_bounding_boxes(
            manifest, self.temp_dir,
            colors=colors, detector_labels=labels
        )
        stem = Path(self.image_path).stem
        self.assertTrue((Path(self.temp_dir) / f"{stem}_box.jpg").exists())

    def test_classifier_label_col_category(self):
        """classifier_label_col='category' should work correctly."""
        manifest = self._make_manifest()
        visualization.plot_all_bounding_boxes(
            manifest, self.temp_dir, classifier_label_col="category"
        )
        stem = Path(self.image_path).stem
        self.assertTrue((Path(self.temp_dir) / f"{stem}_box.jpg").exists())

    # --- negative / error cases ---

    def test_missing_file_col_raises_value_error(self):
        """Missing the expected file column raises ValueError."""
        manifest = pd.DataFrame([{
            "path": str(self.image_path),   # wrong column name
            "category": 1, "conf": 0.9,
            "bbox_x": 0.1, "bbox_y": 0.1, "bbox_w": 0.3, "bbox_h": 0.3,
        }])
        with self.assertRaises(ValueError):
            visualization.plot_all_bounding_boxes(manifest, self.temp_dir,
                                                  file_col="filepath")

    def test_mismatched_colors_labels_raises_value_error(self):
        """colors and detector_labels of different lengths raises ValueError."""
        colors = {1: (0, 255, 0), 2: (0, 0, 255)}    # 2 entries
        labels = {1: "animal"}                         # 1 entry
        manifest = self._make_manifest(category=1)
        with self.assertRaises(ValueError):
            visualization.plot_all_bounding_boxes(
                manifest, self.temp_dir,
                colors=colors, detector_labels=labels
            )


if __name__ == "__main__":
    unittest.main()
