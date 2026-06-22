"""
Unit tests for animl/emergence.py

@ Nikita Sharma 2026
"""
import unittest
import pandas as pd
from animl.emergence import detect_emergence, emergence_density_curve


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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
        # First 20 sequences: no juveniles. Next 20: juveniles present
        # in every sequence (density = 1.0), which should trigger
        # sustained emergence.
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
        # emergence should be detected at or after the transition point (seq 20)
        self.assertGreaterEqual(row['emergence_sequence'], 15)
        self.assertIsNotNone(row['emergence_datetime'])
        self.assertIsNotNone(row['emergence_filename'])

    def test_isolated_fluke_does_not_trigger_emergence(self):
        """A single isolated detection surrounded by zeros should not be
        enough to declare emergence, since it won't sustain across
        multiple windows."""
        sequences = list(range(40))
        juv_values = [0] * 40
        juv_values[20] = 1  # single fluke detection
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
        juv_values = [0] * 20 + [2] * 20  # 2 juveniles per sequence
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
            _make_counts_df(seq_b, [0] * 30),  # never emerges
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
