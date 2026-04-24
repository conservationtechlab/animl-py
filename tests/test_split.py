"""
Unit tests for animl/split.py

@ Kyra Swanson 2023
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from animl.split import get_animals, get_empty, train_val_test


def _make_manifest(categories=(1, 2, 3, 0)):
    """Return a minimal detection manifest with the given categories."""
    n = len(categories)
    return pd.DataFrame({
        'filepath': [f'img{i}.jpg' for i in range(n)],
        'category': list(categories),
        'conf': [0.9] * n,
    })


class TestGetAnimals(unittest.TestCase):

    def test_returns_dataframe(self):
        manifest = _make_manifest()
        result = get_animals(manifest)
        self.assertIsInstance(result, pd.DataFrame)

    def test_only_category_1_returned(self):
        manifest = _make_manifest([1, 2, 1, 0])
        result = get_animals(manifest)
        self.assertTrue((result['category'] == 1).all())

    def test_correct_row_count(self):
        manifest = _make_manifest([1, 2, 1, 0])
        result = get_animals(manifest)
        self.assertEqual(len(result), 2)

    def test_no_animals_returns_empty(self):
        manifest = _make_manifest([2, 0, 3])
        result = get_animals(manifest)
        self.assertEqual(len(result), 0)

    def test_all_animals_returns_all(self):
        manifest = _make_manifest([1, 1, 1])
        result = get_animals(manifest)
        self.assertEqual(len(result), 3)

    def test_nan_category_excluded(self):
        manifest = pd.DataFrame({
            'filepath': ['a.jpg', 'b.jpg', 'c.jpg'],
            'category': [1, np.nan, 1],
            'conf': [0.9, 0.5, 0.8],
        })
        result = get_animals(manifest)
        self.assertEqual(len(result), 2)

    def test_index_is_reset(self):
        manifest = _make_manifest([2, 1, 0, 1])
        result = get_animals(manifest)
        self.assertEqual(list(result.index), list(range(len(result))))

    def test_original_columns_preserved(self):
        manifest = _make_manifest([1, 2])
        result = get_animals(manifest)
        for col in manifest.columns:
            self.assertIn(col, result.columns)

    def test_empty_manifest_returns_empty(self):
        manifest = pd.DataFrame(columns=['filepath', 'category', 'conf'])
        result = get_animals(manifest)
        self.assertEqual(len(result), 0)

    def test_category_float_handled(self):
        manifest = pd.DataFrame({
            'filepath': ['a.jpg', 'b.jpg'],
            'category': [1.0, 2.0],
            'conf': [0.9, 0.8],
        })
        result = get_animals(manifest)
        self.assertEqual(len(result), 1)


class TestGetEmpty(unittest.TestCase):

    def test_returns_dataframe(self):
        manifest = _make_manifest()
        result = get_empty(manifest)
        self.assertIsInstance(result, pd.DataFrame)

    def test_no_category_1_in_result(self):
        manifest = _make_manifest([1, 2, 0, 3])
        result = get_empty(manifest)
        self.assertTrue((result['category'].astype(int) != 1).all())

    def test_correct_row_count(self):
        manifest = _make_manifest([1, 2, 0])
        result = get_empty(manifest)
        self.assertEqual(len(result), 2)

    def test_prediction_column_added(self):
        manifest = _make_manifest([2, 0])
        result = get_empty(manifest)
        self.assertIn('prediction', result.columns)

    def test_confidence_column_added(self):
        manifest = _make_manifest([2, 0])
        result = get_empty(manifest)
        self.assertIn('confidence', result.columns)

    def test_category_2_mapped_to_human(self):
        manifest = _make_manifest([2])
        result = get_empty(manifest)
        self.assertEqual(result.iloc[0]['prediction'], 'human')

    def test_category_3_mapped_to_vehicle(self):
        manifest = _make_manifest([3])
        result = get_empty(manifest)
        self.assertEqual(result.iloc[0]['prediction'], 'vehicle')

    def test_category_0_mapped_to_empty(self):
        manifest = pd.DataFrame({
            'filepath': ['a.jpg'],
            'category': [0],
            'conf': [np.nan],
        })
        result = get_empty(manifest)
        self.assertEqual(result.iloc[0]['prediction'], 'empty')

    def test_empty_conf_replaced_with_one(self):
        manifest = pd.DataFrame({
            'filepath': ['a.jpg'],
            'category': [0],
            'conf': [np.nan],
        })
        result = get_empty(manifest)
        self.assertEqual(result.iloc[0]['confidence'], 1)

    def test_non_nan_conf_preserved(self):
        manifest = pd.DataFrame({
            'filepath': ['a.jpg'],
            'category': [2],
            'conf': [0.75],
        })
        result = get_empty(manifest)
        self.assertAlmostEqual(result.iloc[0]['confidence'], 0.75)

    def test_all_animals_returns_empty_dataframe(self):
        manifest = _make_manifest([1, 1, 1])
        result = get_empty(manifest)
        self.assertEqual(len(result), 0)

    def test_nan_category_treated_as_empty(self):
        """NaN category is filled with 0 (empty) and included in get_empty results."""
        manifest = pd.DataFrame({
            'filepath': ['a.jpg', 'b.jpg', 'c.jpg'],
            'category': [2, np.nan, 0],
            'conf': [0.8, 0.5, 1.0],
        })
        result = get_empty(manifest)
        # All three rows have category != 1: category 2 (human), NaN→0 (empty), 0 (empty)
        self.assertEqual(len(result), 3)

    def test_index_is_reset(self):
        manifest = _make_manifest([1, 2, 1, 0])
        result = get_empty(manifest)
        self.assertEqual(list(result.index), list(range(len(result))))

    def test_empty_manifest_returns_empty(self):
        manifest = pd.DataFrame(columns=['filepath', 'category', 'conf'])
        result = get_empty(manifest)
        self.assertEqual(len(result), 0)

    def test_original_columns_preserved(self):
        manifest = _make_manifest([2, 0])
        result = get_empty(manifest)
        for col in manifest.columns:
            self.assertIn(col, result.columns)

    def test_mixed_categories_split_correctly(self):
        manifest = _make_manifest([1, 2, 1, 0, 3])
        animals = get_animals(manifest)
        empty = get_empty(manifest)
        self.assertEqual(len(animals) + len(empty), len(manifest))


class TestTrainValTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        # Build a manifest large enough to stratify (>=2 per class)
        labels = ['cat'] * 10 + ['dog'] * 10 + ['bird'] * 10
        cls.manifest = pd.DataFrame({
            'filepath': [f'img{i}.jpg' for i in range(30)],
            'class': labels,
            'confidence': np.random.default_rng(0).uniform(0.5, 1.0, 30),
        })

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_returns_three_dataframes(self):
        train, val, test = train_val_test(self.manifest.copy())
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(val, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)

    def test_total_rows_equal_input(self):
        train, val, test = train_val_test(self.manifest.copy())
        self.assertEqual(len(train) + len(val) + len(test), len(self.manifest))

    def test_no_overlap_between_splits(self):
        train, val, test = train_val_test(self.manifest.copy())
        train_paths = set(train['filepath'])
        val_paths = set(val['filepath'])
        test_paths = set(test['filepath'])
        self.assertEqual(len(train_paths & val_paths), 0)
        self.assertEqual(len(train_paths & test_paths), 0)
        self.assertEqual(len(val_paths & test_paths), 0)

    def test_test_size_respected(self):
        train, val, test = train_val_test(self.manifest.copy(), test_size=0.2)
        expected = round(len(self.manifest) * 0.2)
        self.assertAlmostEqual(len(test), expected, delta=1)

    def test_val_size_respected(self):
        train, val, test = train_val_test(self.manifest.copy(), val_size=0.1, test_size=0.1)
        expected = round(len(self.manifest) * 0.1)
        self.assertAlmostEqual(len(val), expected, delta=1)

    def test_seed_produces_reproducible_splits(self):
        train1, val1, test1 = train_val_test(self.manifest.copy(), seed=7)
        train2, val2, test2 = train_val_test(self.manifest.copy(), seed=7)
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_different_seeds_produce_different_splits(self):
        train1, _, _ = train_val_test(self.manifest.copy(), seed=1)
        train2, _, _ = train_val_test(self.manifest.copy(), seed=99)
        # With 30 samples it is very unlikely they produce identical train sets
        self.assertFalse(train1['filepath'].reset_index(drop=True).equals(
            train2['filepath'].reset_index(drop=True)))

    def test_missing_label_col_raises(self):
        with self.assertRaises(ValueError):
            train_val_test(self.manifest.copy(), label_col='nonexistent')

    def test_missing_file_col_raises(self):
        with self.assertRaises(ValueError):
            train_val_test(self.manifest.copy(), file_col='nonexistent')

    def test_test_size_negative_raises(self):
        with self.assertRaises(AssertionError):
            train_val_test(self.manifest.copy(), test_size=-0.1)

    def test_test_size_one_raises(self):
        with self.assertRaises(AssertionError):
            train_val_test(self.manifest.copy(), test_size=1.0)

    def test_val_size_negative_raises(self):
        with self.assertRaises(AssertionError):
            train_val_test(self.manifest.copy(), val_size=-0.1)

    def test_combined_size_too_large_raises(self):
        with self.assertRaises(AssertionError):
            train_val_test(self.manifest.copy(), val_size=0.6, test_size=0.5)

    def test_saves_csv_files_when_out_dir_set(self):
        out_dir = str(Path(self.tmp_dir) / 'splits')
        Path(out_dir).mkdir()
        train_val_test(self.manifest.copy(), out_dir=out_dir)
        self.assertTrue(Path(out_dir, 'train_data.csv').exists())
        self.assertTrue(Path(out_dir, 'validate_data.csv').exists())
        self.assertTrue(Path(out_dir, 'test_data.csv').exists())

    def test_index_is_reset_in_all_splits(self):
        train, val, test = train_val_test(self.manifest.copy())
        for df in (train, val, test):
            self.assertEqual(list(df.index), list(range(len(df))))

    def test_no_conf_col_deduplicates_by_file(self):
        # Each filepath appears twice but with the same label — dedup keeps 1 per file
        filepaths = [f'img{i}.jpg' for i in range(30)]
        labels = ['cat'] * 10 + ['dog'] * 10 + ['bird'] * 10
        manifest = pd.DataFrame({
            'filepath': filepaths + filepaths,  # every file duplicated
            'class': labels + labels,
        })
        # Pass a conf_col that does NOT exist so the dedup-by-file branch runs
        train, val, test = train_val_test(manifest, conf_col='nonexistent_col')
        total = len(train) + len(val) + len(test)
        # dedup by file → 30 unique files
        self.assertEqual(total, 30)

    def test_custom_label_col(self):
        manifest = self.manifest.rename(columns={'class': 'species'})
        train, val, test = train_val_test(manifest, label_col='species')
        self.assertGreater(len(train), 0)

    def test_custom_file_col(self):
        manifest = self.manifest.rename(columns={'filepath': 'path'})
        train, val, test = train_val_test(manifest, file_col='path')
        self.assertGreater(len(train), 0)

    def test_zero_test_size_raises(self):
        # sklearn does not accept test_size=0.0; assert raises
        with self.assertRaises(Exception):
            train_val_test(self.manifest.copy(), test_size=0.0, val_size=0.1)

    def test_zero_val_size_raises(self):
        # sklearn does not accept a computed rel_val_size of 0.0; assert raises
        with self.assertRaises(Exception):
            train_val_test(self.manifest.copy(), test_size=0.1, val_size=0.0)


if __name__ == '__main__':
    unittest.main()
