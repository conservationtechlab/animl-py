"""
Unit tests for animl/count.py

@ Nikita Sharma 2026
"""
import unittest
import pandas as pd
from animl.count import count_detections, deduplicate, detect_emergence, emergence_density_curve
from animl.utils.general import get_iou


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_detection_row(filepath='a.jpg', category_label='adult',
                        conf=0.9, bbox_x=0.1, bbox_y=0.1,
                        bbox_w=0.2, bbox_h=0.2,
                        station='station1',
                        datetime='2024-01-01 12:00:00'):
    """Return a single detection row as a dictionary."""
    return {
        'filepath': filepath,
        'category_label': category_label,
        'conf': conf,
        'bbox_x': bbox_x,
        'bbox_y': bbox_y,
        'bbox_w': bbox_w,
        'bbox_h': bbox_h,
        'station': station,
        'datetime': datetime,
    }


def _make_detections_df(rows):
    """Return a DataFrame from a list of detection row dicts."""
    return pd.DataFrame(rows)


def _make_counts_df(sequences, owl_juvenile_values, owl_adult_values=None):
    """Return a counts DataFrame with `sequence` and class count columns."""
    n = len(sequences)
    if owl_adult_values is None:
        owl_adult_values = [1.0] * n
    return pd.DataFrame({
        'sequence': sequences,
        'owl_juvenile': owl_juvenile_values,
        'owl_adult': owl_adult_values,
    })


def _make_detections_lookup_df(sequences, station='station1',
                               start='2026-01-01 00:00:00',
                               minutes_apart=10):
    """Return a detections-style DataFrame with one row per sequence,
    used purely for the sequence -> station/datetime/filename lookup
    inside detect_emergence()."""
    base = pd.Timestamp(start)
    rows = []
    for i, seq in enumerate(sequences):
        dt = base + pd.Timedelta(minutes=minutes_apart * i)
        rows.append({
            'sequence': seq,
            'station': station,
            'datetime': dt,
            'filename': f'IMG_{seq}.jpg',
            'filepath': f'/data/{station}/IMG_{seq}.jpg',
        })
    return pd.DataFrame(rows)


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
                                  station_col='station',
                                  confidence_threshold=0.5,
                                  maxdiff=60)
        self.assertFalse(result.empty)
        self.assertIn('adult', result.columns)

    def test_low_confidence_filtered_out(self):
        """Detections below confidence threshold should not be counted."""
        filepaths = ['a.jpg', 'b.jpg', 'c.jpg']
        rows = [_make_detection_row(filepath=fp, conf=0.1) for fp in filepaths]
        result = count_detections(_make_detections_df(rows),
                                  station_col='station',
                                  confidence_threshold=0.5,
                                  maxdiff=60)
        self.assertTrue(result.empty)

    def test_max_n_limits_images(self):
        """max_n should limit the number of images counted per sequence."""
        filepaths = ['a.jpg', 'b.jpg', 'c.jpg']
        rows = [_make_detection_row(filepath=fp) for fp in filepaths]
        result = count_detections(_make_detections_df(rows),
                                  station_col='station',
                                  confidence_threshold=0.5,
                                  maxdiff=60, max_n=1)
        self.assertFalse(result.empty)


# ---------------------------------------------------------------------------
# detect_emergence
# ---------------------------------------------------------------------------

class TestDetectEmergence(unittest.TestCase):
    """Tests for detect_emergence()."""

    def test_raises_on_missing_target_class(self):
        """Should raise ValueError if target_class isn't a counts column."""
        sequences = list(range(5))
        counts = _make_counts_df(sequences, [0, 0, 0, 0, 0])
        detections = _make_detections_lookup_df(sequences)
        with self.assertRaises(ValueError):
            detect_emergence(counts, detections, target_class='nonexistent')

    def test_no_emergence_when_all_zero(self):
        """No emergence should be detected if target_class is always 0."""
        sequences = list(range(30))
        counts = _make_counts_df(sequences, [0] * 30)
        detections = _make_detections_lookup_df(sequences)
        result = detect_emergence(
            counts, detections, target_class='owl_juvenile',
            window_size=10, step_size=5,
            density_threshold=0.3, sustained_windows=2,
        )
        self.assertEqual(len(result), 1)
        self.assertFalse(result.iloc[0]['emerged'])
        self.assertIsNone(result.iloc[0]['emergence_sequence'])

    def test_emergence_detected_after_sustained_presence(self):
        """Emergence should be detected once density crosses threshold
        and stays crossed for the required number of windows."""
        sequences = list(range(40))
        juv_values = [0] * 20 + [1] * 20
        counts = _make_counts_df(sequences, juv_values)
        detections = _make_detections_lookup_df(sequences)

        result = detect_emergence(
            counts, detections, target_class='owl_juvenile',
            window_size=10, step_size=5,
            density_threshold=0.5, sustained_windows=2,
        )
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertTrue(row['emerged'])
        self.assertGreaterEqual(row['emergence_sequence'], 15)
        self.assertIsNotNone(row['emergence_datetime'])
        self.assertIsNotNone(row['emergence_filename'])

    def test_isolated_fluke_does_not_trigger_emergence(self):
        """A single isolated detection surrounded by zeros should not be
        enough to declare emergence, since it won't sustain across
        multiple windows."""
        sequences = list(range(40))
        juv_values = [0] * 40
        juv_values[20] = 1
        counts = _make_counts_df(sequences, juv_values)
        detections = _make_detections_lookup_df(sequences)

        result = detect_emergence(
            counts, detections, target_class='owl_juvenile',
            window_size=10, step_size=5,
            density_threshold=0.3, sustained_windows=3,
        )
        self.assertFalse(result.iloc[0]['emerged'])

    def test_density_metric_sum_vs_fraction(self):
        """'sum' and 'fraction' metrics should both detect emergence but
        can differ in exact timing/threshold scale."""
        sequences = list(range(40))
        juv_values = [0] * 20 + [2] * 20
        counts = _make_counts_df(sequences, juv_values)
        detections = _make_detections_lookup_df(sequences)

        result_fraction = detect_emergence(
            counts, detections, target_class='owl_juvenile',
            window_size=10, step_size=5, density_metric='fraction',
            density_threshold=0.5, sustained_windows=2,
        )
        result_sum = detect_emergence(
            counts, detections, target_class='owl_juvenile',
            window_size=10, step_size=5, density_metric='sum',
            density_threshold=10.0, sustained_windows=2,
        )
        self.assertTrue(result_fraction.iloc[0]['emerged'])
        self.assertTrue(result_sum.iloc[0]['emerged'])

    def test_invalid_density_metric_raises(self):
        """An unrecognized density_metric string should raise ValueError."""
        sequences = list(range(20))
        counts = _make_counts_df(sequences, [1] * 20)
        detections = _make_detections_lookup_df(sequences)
        with self.assertRaises(ValueError):
            detect_emergence(
                counts, detections, target_class='owl_juvenile',
                density_metric='bogus',
            )

    def test_multiple_stations_detected_independently(self):
        """Each station should get its own row and be evaluated on its
        own timeline, independent of other stations."""
        seq_a = list(range(0, 30))
        seq_b = list(range(30, 60))

        counts = pd.concat([
            _make_counts_df(seq_a, [0] * 15 + [1] * 15),
            _make_counts_df(seq_b, [0] * 30),
        ]).reset_index(drop=True)

        detections = pd.concat([
            _make_detections_lookup_df(seq_a, station='stationA'),
            _make_detections_lookup_df(seq_b, station='stationB'),
        ]).reset_index(drop=True)

        result = detect_emergence(
            counts, detections, target_class='owl_juvenile',
            window_size=10, step_size=5,
            density_threshold=0.5, sustained_windows=2,
        )
        self.assertEqual(len(result), 2)
        stations = set(result['station'])
        self.assertEqual(stations, {'stationA', 'stationB'})

        row_a = result[result['station'] == 'stationA'].iloc[0]
        row_b = result[result['station'] == 'stationB'].iloc[0]
        self.assertTrue(row_a['emerged'])
        self.assertFalse(row_b['emerged'])


# ---------------------------------------------------------------------------
# emergence_density_curve
# ---------------------------------------------------------------------------

class TestEmergenceDensityCurve(unittest.TestCase):
    """Tests for emergence_density_curve()."""

    def test_returns_one_row_per_window(self):
        """Should return multiple window rows for a long enough sequence."""
        sequences = list(range(30))
        counts = _make_counts_df(sequences, [0] * 15 + [1] * 15)
        detections = _make_detections_lookup_df(sequences)

        curve = emergence_density_curve(
            counts, detections, target_class='owl_juvenile',
            window_size=10, step_size=5,
        )
        self.assertGreater(len(curve), 1)
        self.assertIn('density', curve.columns)

    def test_density_increases_after_transition(self):
        """Density should be higher in later windows than earlier ones
        when juveniles only appear in the second half."""
        sequences = list(range(40))
        counts = _make_counts_df(sequences, [0] * 20 + [1] * 20)
        detections = _make_detections_lookup_df(sequences)

        curve = emergence_density_curve(
            counts, detections, target_class='owl_juvenile',
            window_size=10, step_size=5,
        )
        first_density = curve.iloc[0]['density']
        last_density = curve.iloc[-1]['density']
        self.assertLess(first_density, last_density)

    def test_raises_on_missing_target_class(self):
        """Should raise ValueError if target_class isn't a counts column."""
        sequences = list(range(10))
        counts = _make_counts_df(sequences, [0] * 10)
        detections = _make_detections_lookup_df(sequences)
        with self.assertRaises(ValueError):
            emergence_density_curve(counts, detections, target_class='nonexistent')


if __name__ == '__main__':
    unittest.main()
