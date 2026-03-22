"""
Tests for the split module, specifically get_animals and get_empty functions
with NaN handling.

@ Test Suite 2026
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from importlib.util import spec_from_file_location, module_from_spec

import pandas as pd
import numpy as np


# Create a mock for animl.file_management to avoid loading the full animl package
sys.modules['animl'] = MagicMock()
sys.modules['animl.file_management'] = MagicMock()


def mock_save_data(df, path):
    """Mock save_data function"""
    pass


sys.modules['animl.file_management'].save_data = mock_save_data

split_module_path = Path(__file__).parent.parent / "src" / "animl" / "split.py"
spec = spec_from_file_location("split", split_module_path)
split = module_from_spec(spec)
spec.loader.exec_module(split)

get_animals = split.get_animals
get_empty = split.get_empty


class TestGetAnimalsNaNHandling(unittest.TestCase):
    """Test get_animals function with various NaN scenarios"""

    def setUp(self):
        """Create test dataframes"""
        self.manifest_no_nans = pd.DataFrame({
            'category': [1, 1, 0, 2, 3],
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg'],
            'conf': [0.95, 0.87, 0.92, 0.88, 0.91]
        })

        self.manifest_with_nans = pd.DataFrame({
            'category': [1.0, np.nan, 1.0, np.nan, 0.0, 2.0],
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'],
            'conf': [0.95, 0.87, 0.92, 0.88, 0.91, 0.85]
        })

        self.manifest_all_nans = pd.DataFrame({
            'category': [np.nan, np.nan, np.nan],
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
            'conf': [0.95, 0.87, 0.92]
        })

        self.manifest_only_animals = pd.DataFrame({
            'category': [1.0, 1.0, 1.0, np.nan],
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            'conf': [0.95, 0.87, 0.92, 0.88]
        })

    def test_get_animals_no_nans(self):
        """Test get_animals with no NaN values"""
        result = get_animals(self.manifest_no_nans)
        self.assertEqual(len(result), 2)
        self.assertTrue((result['category'] == 1).all())

    def test_get_animals_with_nans(self):
        """Test get_animals removes NaN values from category column"""
        result = get_animals(self.manifest_with_nans)
        # Should have only 2 animals (category 1) and no NaN rows
        self.assertEqual(len(result), 2)
        self.assertTrue((result['category'] == 1).all())
        # Ensure no NaN values in result
        self.assertEqual(result['category'].isna().sum(), 0)

    def test_get_animals_all_nans(self):
        """Test get_animals with all NaN values returns empty dataframe"""
        result = get_animals(self.manifest_all_nans)
        self.assertEqual(len(result), 0)
        self.assertEqual(result['category'].isna().sum(), 0)

    def test_get_animals_only_animals_with_nans(self):
        """Test get_animals when all non-NaN rows are animals"""
        result = get_animals(self.manifest_only_animals)
        self.assertEqual(len(result), 3)
        self.assertTrue((result['category'] == 1).all())

    def test_get_animals_preserves_columns(self):
        """Test that get_animals preserves all columns"""
        result = get_animals(self.manifest_with_nans)
        expected_cols = {'category', 'filepath', 'conf'}
        self.assertTrue(expected_cols.issubset(set(result.columns)))

    def test_get_animals_resets_index(self):
        """Test that get_animals resets index properly"""
        result = get_animals(self.manifest_with_nans)
        self.assertTrue(result.index.equals(pd.RangeIndex(len(result))))


class TestGetEmptyNaNHandling(unittest.TestCase):
    """Test get_empty function with various NaN scenarios"""

    def setUp(self):
        """Create test dataframes"""
        self.manifest_no_nans = pd.DataFrame({
            'category': [1, 0, 2, 3, 0],
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg'],
            'conf': [0.95, 0.87, 0.92, 0.88, 0.91]
        })

        self.manifest_with_nans = pd.DataFrame({
            'category': [1.0, np.nan, 0.0, 2.0, np.nan, 3.0],
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'],
            'conf': [0.95, 0.87, 0.92, 0.88, 0.91, 0.85]
        })

        self.manifest_all_nans = pd.DataFrame({
            'category': [np.nan, np.nan, np.nan],
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
            'conf': [0.95, 0.87, 0.92]
        })

        self.manifest_only_empty_human_vehicle = pd.DataFrame({
            'category': [0.0, 2.0, 3.0, np.nan, 0.0],
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg'],
            'conf': [np.nan, 0.88, 0.91, 0.85, np.nan]
        })

    def test_get_empty_no_nans(self):
        """Test get_empty with no NaN values"""
        result = get_empty(self.manifest_no_nans)
        self.assertEqual(len(result), 4)
        self.assertTrue((result['category'] != 1).all())

    def test_get_empty_with_nans(self):
        """Test get_empty removes NaN values from category column"""
        result = get_empty(self.manifest_with_nans)
        # Should have 3 non-animals (0, 2, 3) and no NaN rows
        self.assertEqual(len(result), 3)
        # Ensure no NaN values in category
        self.assertEqual(result['category'].isna().sum(), 0)
        # Ensure all are non-animals
        self.assertTrue((result['category'] != 1).all())

    def test_get_empty_all_nans(self):
        """Test get_empty with all NaN values returns empty dataframe"""
        result = get_empty(self.manifest_all_nans)
        self.assertEqual(len(result), 0)
        self.assertFalse(result['category'].isna().any())

    def test_get_empty_has_prediction_column(self):
        """Test that get_empty adds prediction column"""
        result = get_empty(self.manifest_no_nans)
        self.assertIn('prediction', result.columns)

    def test_get_empty_prediction_mapping(self):
        """Test that get_empty correctly maps predictions"""
        result = get_empty(self.manifest_with_nans)
        # Check that predictions are correctly mapped
        for idx, row in result.iterrows():
            cat = int(row['category'])
            pred = row['prediction']
            if cat == 0:
                self.assertEqual(pred, 'empty')
            elif cat == 2:
                self.assertEqual(pred, 'human')
            elif cat == 3:
                self.assertEqual(pred, 'vehicle')

    def test_get_empty_confidence_column(self):
        """Test that get_empty adds confidence column"""
        result = get_empty(self.manifest_no_nans)
        self.assertIn('confidence', result.columns)

    def test_get_empty_nan_confidence_handling(self):
        """Test that get_empty replaces NaN confidence with 1"""
        result = get_empty(self.manifest_only_empty_human_vehicle)
        # All confidence values should be numeric (NaN replaced with 1)
        self.assertEqual(result['confidence'].isna().sum(), 0)
        # Check that all values are positive (either 1 from the replacement or original)
        self.assertTrue((result['confidence'] >= 1.0).all() or (result['confidence'] > 0).all())

    def test_get_empty_resets_index(self):
        """Test that get_empty resets index properly"""
        result = get_empty(self.manifest_with_nans)
        self.assertTrue(result.index.equals(pd.RangeIndex(len(result))))

    def test_get_empty_preserves_columns(self):
        """Test that get_empty preserves original columns"""
        result = get_empty(self.manifest_with_nans)
        # Check original columns are present
        self.assertIn('category', result.columns)
        self.assertIn('filepath', result.columns)


class TestGetAnimalsGetEmptyComparison(unittest.TestCase):
    """Test that get_animals and get_empty handle the same dataframe correctly"""

    def test_animals_and_empty_partition_data(self):
        """Test that get_animals and get_empty split data correctly"""
        manifest = pd.DataFrame({
            'category': [1.0, 0.0, 1.0, 2.0, 3.0, np.nan, 1.0],
            'filepath': [f'img{i}.jpg' for i in range(7)],
            'conf': [0.95, 0.87, 0.92, 0.88, 0.91, 0.85, 0.89]
        })

        animals = get_animals(manifest)
        empty_etc = get_empty(manifest)

        # Should have no overlap (except animals should not appear in empty_etc)
        self.assertEqual(len(animals) + len(empty_etc), 6)  # Total non-NaN

        # All animals should have category 1
        self.assertTrue((animals['category'] == 1).all())

        # Empty, etc should not have category 1
        self.assertTrue((empty_etc['category'] != 1).all())


if __name__ == '__main__':
    unittest.main()
