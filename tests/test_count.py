"""
Unit tests for animl/count.py

@ Nikita Sharma 2026
"""
import unittest
import pandas as pd
from animl.count import count_detections, deduplicate
from animl.utils.general import get_iou


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_detection_row(filepath='a.jpg', category_label='adult',
                         conf=0.9, bbox_x=0.1, bbox_y=0.1,
                         bbox_w=0.2, bbox_h=0.2):
    """Return a single detection row as a dictionary."""
    return {
        'filepath': filepath,
        'category_label': category_label,
        'conf': conf,
        'bbox_x': bbox_x,
        'bbox_y': bbox_y,
        'bbox_w': bbox_w,
        'bbox_h': bbox_h,
    }


def _make_detections_df(rows):
    """Return a DataFrame from a list of detection row dicts."""
    return pd.DataFrame(rows)


def _make_manifest(filepaths, station='station1',
                   datetime='2024-01-01 12:00:00'):
    """Return a simple manifest DataFrame."""
    return pd.DataFrame({
        'filepath': filepaths,
        'station': [station] * len(filepaths),
        'datetime': [datetime] * len(filepaths),
    })


# ---------------------------------------------------------------------------
# get_iou
# ---------------------------------------------------------------------------

class TestGetIou(unittest.TestCase):

    def test_identical_boxes_returns_one(self):
        """Two identical boxes should have IOU of 1.0."""
        bbox = [0.1, 0.1, 0.2, 0.2]
        self.assertAlmostEqual(get_iou(bbox, bbox), 1.0)

    def test_non_overlapping_boxes_returns_zero(self):
        """Two boxes that don't overlap should have IOU of 0.0."""
        bb1 = [0.0, 0.0, 0.2, 0.2]
        bb2 = [0.5, 0.5, 0.2, 0.2]
        self.assertEqual(get_iou(bb1, bb2), 0.0)

    def test_partial_overlap(self):
        """Two partially overlapping boxes should return IOU between 0 and 1."""
        bb1 = [0.0, 0.0, 0.4, 0.4]
        bb2 = [0.2, 0.2, 0.4, 0.4]
        iou = get_iou(bb1, bb2)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------

class TestDeduplicate(unittest.TestCase):

    def test_no_duplicates(self):
        """Non-overlapping detections should all be kept."""
        rows = [
            _make_detection_row(bbox_x=0.0, bbox_y=0.0),
            _make_detection_row(bbox_x=0.5, bbox_y=0.5),
        ]
        result = deduplicate(_make_detections_df(rows), iou_threshold=0.5)
        self.assertEqual(len(result), 2)

    def test_duplicate_keeps_higher_confidence(self):
        """When two boxes overlap, the one with higher confidence should be kept."""
        rows = [
            _make_detection_row(conf=0.6, category_label='juvenile',
                                bbox_x=0.1, bbox_y=0.1),
            _make_detection_row(conf=0.9, category_label='adult',
                                bbox_x=0.1, bbox_y=0.1),
        ]
        result = deduplicate(_make_detections_df(rows), iou_threshold=0.5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['category_label'], 'adult')
        self.assertAlmostEqual(result.iloc[0]['conf'], 0.9)

    def test_single_detection(self):
        """A single detection should always be kept."""
        result = deduplicate(_make_detections_df([_make_detection_row()]),
                             iou_threshold=0.5)
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# count_detections
# ---------------------------------------------------------------------------

class TestCountDetections(unittest.TestCase):

    def test_basic_count(self):
        """Basic count across a burst of 3 images."""
        filepaths = ['a.jpg', 'b.jpg', 'c.jpg']
        rows = [_make_detection_row(filepath=fp) for fp in filepaths]
        result = count_detections(_make_detections_df(rows),
                                  _make_manifest(filepaths),
                                  station_col='station',
                                  confidence_threshold=0.5,
                                  maxdiff=60)
        self.assertFalse(result.empty)
        self.assertIn('adult', result.columns)
        self.assertIn('juvenile', result.columns)

    def test_low_confidence_filtered_out(self):
        """Detections below confidence threshold should not be counted."""
        filepaths = ['a.jpg', 'b.jpg', 'c.jpg']
        rows = [_make_detection_row(filepath=fp, conf=0.1) for fp in filepaths]
        result = count_detections(_make_detections_df(rows),
                                  _make_manifest(filepaths),
                                  station_col='station',
                                  confidence_threshold=0.5,
                                  maxdiff=60)
        self.assertTrue(result.empty)

    def test_max_n_limits_images(self):
        """max_n should limit the number of images counted per sequence."""
        filepaths = ['a.jpg', 'b.jpg', 'c.jpg']
        rows = [_make_detection_row(filepath=fp) for fp in filepaths]
        result = count_detections(_make_detections_df(rows),
                                  _make_manifest(filepaths),
                                  station_col='station',
                                  confidence_threshold=0.5,
                                  maxdiff=60, max_n=1)
        self.assertFalse(result.empty)


if __name__ == '__main__':
    unittest.main()