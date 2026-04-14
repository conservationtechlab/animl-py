"""
Unit tests for animl/file_management.py

@ Kyra Swanson 2023
"""
import json
import shutil
import unittest
import tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image


from animl.file_management import (
    build_file_manifest,
    WorkingDirectory,
    save_data,
    load_data,
    save_json,
    load_json,
    check_file,
    save_detection_checkpoint,
    sequence_calculation,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    VALID_EXTENSIONS,
)


class TestConstants(unittest.TestCase):

    def test_image_extensions_are_set(self):
        self.assertIsInstance(IMAGE_EXTENSIONS, set)

    def test_video_extensions_are_set(self):
        self.assertIsInstance(VIDEO_EXTENSIONS, set)

    def test_valid_extensions_union(self):
        self.assertEqual(VALID_EXTENSIONS, IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)

    def test_jpg_in_image_extensions(self):
        self.assertIn('.jpg', IMAGE_EXTENSIONS)

    def test_mp4_in_video_extensions(self):
        self.assertIn('.mp4', VIDEO_EXTENSIONS)


class TestBuildFileManifest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()

        # Create sample images at root level
        for name in ['img1.jpg', 'img2.JPG', 'img3.png']:
            img = Image.fromarray(np.uint8(np.zeros((10, 10, 3))))
            img.save(Path(cls.tmp_dir) / name)

        # Create a non-image file
        (Path(cls.tmp_dir) / 'notes.txt').write_text('ignore me')

        # Create a subdirectory with an image (station1/cam1/file.jpg structure)
        cls.station_dir = Path(cls.tmp_dir) / 'station1'
        cls.camera_dir = cls.station_dir / 'cam1'
        cls.camera_dir.mkdir(parents=True)
        img = Image.fromarray(np.uint8(np.zeros((10, 10, 3))))
        img.save(cls.camera_dir / 'subimg.jpg')

        # Create a second station/camera combo
        cls.station2_dir = Path(cls.tmp_dir) / 'station2'
        cls.camera2_dir = cls.station2_dir / 'cam2'
        cls.camera2_dir.mkdir(parents=True)
        img = Image.fromarray(np.uint8(np.zeros((10, 10, 3))))
        img.save(cls.camera2_dir / 'subimg2.jpg')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_returns_dataframe(self):
        result = build_file_manifest(self.tmp_dir, exif=False)
        self.assertIsInstance(result, pd.DataFrame)

    def test_filters_non_image_files(self):
        result = build_file_manifest(self.tmp_dir, exif=False)
        self.assertFalse(any(result['filepath'].str.endswith('.txt')))

    def test_filepath_column_exists(self):
        result = build_file_manifest(self.tmp_dir, exif=False)
        self.assertIn('filepath', result.columns)

    def test_filename_column_exists(self):
        result = build_file_manifest(self.tmp_dir, exif=False)
        self.assertIn('filename', result.columns)

    def test_extension_column_is_lowercase(self):
        result = build_file_manifest(self.tmp_dir, exif=False)
        for ext in result['extension']:
            self.assertEqual(ext, ext.lower())

    def test_recursive_finds_subdir_images(self):
        result = build_file_manifest(self.tmp_dir, exif=False, recursive=True)
        filenames = result['filename'].tolist()
        self.assertIn('subimg.jpg', filenames)

    def test_non_recursive_ignores_subdir(self):
        result = build_file_manifest(self.tmp_dir, exif=False, recursive=False)
        filenames = result['filename'].tolist()
        self.assertNotIn('subimg.jpg', filenames)

    def test_empty_directory_returns_empty_dataframe(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            result = build_file_manifest(empty_dir, exif=False)
            self.assertTrue(result.empty)

    def test_invalid_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            build_file_manifest('/nonexistent/path', exif=False)

    def test_saves_to_out_file(self):
        with tempfile.TemporaryDirectory() as out_dir:
            out_file = str(Path(out_dir) / 'manifest.csv')
            build_file_manifest(self.tmp_dir, exif=False, out_file=out_file)
            self.assertTrue(Path(out_file).exists())

    def test_loads_existing_out_file(self):
        """If out_file exists and user says 'y', it should load from file."""
        with tempfile.TemporaryDirectory() as out_dir:
            out_file = str(Path(out_dir) / 'manifest.csv')
            result = build_file_manifest(self.tmp_dir, exif=False, out_file=out_file)
            result.to_csv(out_file, index=False)
            from unittest.mock import patch
            with patch('animl.file_management.check_file', return_value=True):
                loaded = build_file_manifest(self.tmp_dir, exif=False, out_file=out_file)
                self.assertIsInstance(loaded, pd.DataFrame)

    def test_exif_adds_datetime_column(self):
        result = build_file_manifest(self.tmp_dir, exif=True)
        self.assertIn('datetime', result.columns)

    def test_exif_adds_width_height(self):
        result = build_file_manifest(self.tmp_dir, exif=True)
        self.assertIn('width', result.columns)
        self.assertIn('height', result.columns)

    # ------------------------------------------------------------------
    # station_depth tests
    # ------------------------------------------------------------------

    def test_station_depth_adds_station_column(self):
        result = build_file_manifest(self.tmp_dir, exif=False, station_depth=0)
        self.assertIn('station', result.columns)

    def test_station_depth_none_does_not_add_station_column(self):
        result = build_file_manifest(self.tmp_dir, exif=False, station_depth=None)
        self.assertNotIn('station', result.columns)

    def test_station_depth_correct_values(self):
        # implementation computes root_depth + station_depth, so first child directory is depth=1
        result = build_file_manifest(self.tmp_dir, exif=False, station_depth=1, recursive=True)
        subdir_rows = result[result['filename'].isin(['subimg.jpg', 'subimg2.jpg'])]
        self.assertTrue(all(subdir_rows['station'].isin(['station1', 'station2'])))

    def test_station_depth_negative_uses_parent_path_part(self):
        result = build_file_manifest(self.tmp_dir, exif=False, station_depth=-1, recursive=True)
        tmp_parts = Path(self.tmp_dir).parts
        expected_station = tmp_parts[-2] if len(tmp_parts) >= 2 else tmp_parts[0]
        self.assertTrue((result['station'] == expected_station).all())

    def test_station_depth_zero_indexed(self):
        # due root_depth offset in implementation, first directory below image_dir is depth=1
        result = build_file_manifest(self.tmp_dir, exif=False, station_depth=1, recursive=True)
        subdir_rows = result[result['filename'] == 'subimg.jpg']
        self.assertFalse(subdir_rows.empty)
        self.assertEqual(subdir_rows.iloc[0]['station'], 'station1')

    # ------------------------------------------------------------------
    # camera_depth tests
    # ------------------------------------------------------------------

    def test_camera_depth_adds_camera_column(self):
        result = build_file_manifest(self.tmp_dir, exif=False, camera_depth=1)
        self.assertIn('camera', result.columns)

    def test_camera_depth_none_does_not_add_camera_column(self):
        result = build_file_manifest(self.tmp_dir, exif=False, camera_depth=None)
        self.assertNotIn('camera', result.columns)

    def test_camera_depth_correct_values(self):
        # structure: tmp_dir/station1/cam1/subimg.jpg -> camera_depth=1 -> 'cam1'
        result = build_file_manifest(self.tmp_dir, exif=False, camera_depth=1, recursive=True)
        subdir_rows = result[result['filename'].isin(['subimg.jpg', 'subimg2.jpg'])]
        self.assertTrue(all(subdir_rows['camera'].isin(['cam1', 'cam2'])))

    def test_camera_depth_negative_uses_parent_path_part(self):
        result = build_file_manifest(self.tmp_dir, exif=False, camera_depth=-1, recursive=True)
        tmp_parts = Path(self.tmp_dir).parts
        expected_camera = tmp_parts[-2] if len(tmp_parts) >= 2 else tmp_parts[0]
        self.assertTrue((result['camera'] == expected_camera).all())

    def test_camera_depth_one_indexed(self):
        # depth 1 should be the second directory below image_dir
        result = build_file_manifest(self.tmp_dir, exif=False, camera_depth=1, recursive=True)
        subdir_rows = result[result['filename'] == 'subimg.jpg']
        self.assertFalse(subdir_rows.empty)
        self.assertEqual(subdir_rows.iloc[0]['camera'], 'cam1')

    def test_station_and_camera_depth_together(self):
        # both columns should be present when both depths are provided
        result = build_file_manifest(self.tmp_dir, exif=False,
                                     station_depth=1, camera_depth=2, recursive=True)
        self.assertIn('station', result.columns)
        self.assertIn('camera', result.columns)

    def test_station_and_camera_values_correct_together(self):
        result = build_file_manifest(self.tmp_dir, exif=False,
                                     station_depth=1, camera_depth=2, recursive=True)
        row = result[result['filename'] == 'subimg.jpg'].iloc[0]
        self.assertEqual(row['station'], 'station1')
        self.assertEqual(row['camera'], 'cam1')

        row2 = result[result['filename'] == 'subimg2.jpg'].iloc[0]
        self.assertEqual(row2['station'], 'station2')
        self.assertEqual(row2['camera'], 'cam2')

    def test_data_timezone_utc(self):
        result = build_file_manifest(self.tmp_dir, exif=True, data_timezone="UTC")
        self.assertIn('datetime', result.columns)
        for value in result['datetime']:
            self.assertIsNotNone(value)
            self.assertIsInstance(value, str)
            datetime.strptime(value, "%Y-%m-%d %H:%M:%S")

        system_tz = datetime.now().astimezone().tzinfo
        first_path = Path(result[result['filename'] == 'img1.jpg'].iloc[0]['filepath'])
        expected = datetime.fromtimestamp(first_path.stat().st_mtime, tz=system_tz).astimezone(
            ZoneInfo("UTC")
        ).strftime("%Y-%m-%d %H:%M:%S")
        self.assertIn(expected, result['datetime'].tolist())

    def test_data_timezone_invalid_falls_back(self):
        result = build_file_manifest(self.tmp_dir, exif=True, data_timezone="NotARealZone")
        self.assertIn('datetime', result.columns)
        for value in result['datetime']:
            self.assertIsNotNone(value)
            datetime.strptime(value, "%Y-%m-%d %H:%M:%S")

        system_tz = datetime.now().astimezone().tzinfo
        first_path = Path(result[result['filename'] == 'img1.jpg'].iloc[0]['filepath'])
        expected = datetime.fromtimestamp(first_path.stat().st_mtime, tz=system_tz).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.assertIn(expected, result['datetime'].tolist())

    def test_station_depth_with_recursive_false_raises(self):
        with self.assertRaises(ValueError):
            build_file_manifest(self.tmp_dir, exif=False, station_depth=1, recursive=False)

    def test_camera_depth_with_recursive_false_raises(self):
        with self.assertRaises(ValueError):
            build_file_manifest(self.tmp_dir, exif=False, camera_depth=1, recursive=False)

    def test_station_depth_returns_none_for_shallow_files(self):
        with tempfile.TemporaryDirectory() as shallow_dir:
            img = Image.fromarray(np.uint8(np.zeros((10, 10, 3))))
            img.save(Path(shallow_dir) / 'root_only.jpg')
            result = build_file_manifest(shallow_dir, exif=False, station_depth=2, recursive=True)
            shallow_row = result[result['filename'] == 'root_only.jpg'].iloc[0]
            self.assertIsNone(shallow_row['station'])

    def test_camera_depth_returns_none_for_shallow_files(self):
        with tempfile.TemporaryDirectory() as shallow_dir:
            img = Image.fromarray(np.uint8(np.zeros((10, 10, 3))))
            img.save(Path(shallow_dir) / 'root_only.jpg')
            result = build_file_manifest(shallow_dir, exif=False, camera_depth=2, recursive=True)
            shallow_row = result[result['filename'] == 'root_only.jpg'].iloc[0]
            self.assertIsNone(shallow_row['camera'])


class TestWorkingDirectory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_creates_base_directory(self):
        wd = WorkingDirectory(self.tmp_dir)
        self.assertTrue(wd.basedir.exists())

    def test_invalid_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            WorkingDirectory('/nonexistent/path')

    def test_file_paths_are_set(self):
        wd = WorkingDirectory(self.tmp_dir)
        self.assertIsNotNone(wd.filemanifest)
        self.assertIsNotNone(wd.imageframes)
        self.assertIsNotNone(wd.results)
        self.assertIsNotNone(wd.predictions)
        self.assertIsNotNone(wd.detections)
        self.assertIsNotNone(wd.mdraw)

    def test_activate_visdir_creates_directory(self):
        wd = WorkingDirectory(self.tmp_dir)
        wd.activate_visdir()
        self.assertTrue(wd.visdir.exists())

    def test_activate_linkdir_creates_directory(self):
        wd = WorkingDirectory(self.tmp_dir)
        wd.activate_linkdir()
        self.assertTrue(wd.linkdir.exists())

    def test_string_path_accepted(self):
        wd = WorkingDirectory(str(self.tmp_dir))
        self.assertTrue(wd.basedir.exists())


class TestSaveLoadData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_save_and_load_roundtrip(self):
        out_file = str(Path(self.tmp_dir) / 'test.csv')
        save_data(self.df, out_file, prompt=False)
        loaded = load_data(out_file)
        pd.testing.assert_frame_equal(self.df, loaded)

    def test_load_non_csv_raises(self):
        with self.assertRaises(AssertionError):
            load_data('somefile.json')

    def test_save_to_nonexistent_directory_raises(self):
        with self.assertRaises(AssertionError):
            save_data(self.df, '/nonexistent/dir/out.csv', prompt=False)

    def test_saved_file_exists(self):
        out_file = str(Path(self.tmp_dir) / 'exists.csv')
        save_data(self.df, out_file, prompt=False)
        self.assertTrue(Path(out_file).exists())


class TestSaveLoadJson(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.data = {'key': 'value', 'numbers': [1, 2, 3]}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_save_and_load_roundtrip(self):
        out_file = str(Path(self.tmp_dir) / 'test.json')
        save_json(self.data, out_file, prompt=False)
        loaded = load_json(out_file)
        self.assertEqual(self.data, loaded)

    def test_load_non_json_raises(self):
        with self.assertRaises(AssertionError):
            load_json('somefile.csv')

    def test_saved_file_is_valid_json(self):
        out_file = str(Path(self.tmp_dir) / 'valid.json')
        save_json(self.data, out_file, prompt=False)
        with open(out_file) as f:
            parsed = json.load(f)
        self.assertEqual(parsed, self.data)


class TestCheckFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_returns_false_for_none(self):
        result = check_file(None)
        self.assertFalse(result)

    def test_returns_false_for_nonexistent_file(self):
        result = check_file('/nonexistent/file.csv')
        self.assertFalse(result)

    def test_existing_file_prompts_user(self):
        f = Path(self.tmp_dir) / 'exists.csv'
        f.write_text('a,b\n1,2')
        from unittest.mock import patch
        with patch('builtins.input', return_value='y'):
            self.assertTrue(check_file(str(f)))
        with patch('builtins.input', return_value='n'):
            self.assertFalse(check_file(str(f)))


class TestSaveDetectionCheckpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_creates_checkpoint_file(self):
        path = str(Path(self.tmp_dir) / 'checkpoint.json')
        save_detection_checkpoint(path, [{'image': 'a.jpg'}])
        self.assertTrue(Path(path).exists())

    def test_checkpoint_contains_images_key(self):
        path = str(Path(self.tmp_dir) / 'checkpoint2.json')
        results = [{'image': 'a.jpg', 'detections': []}]
        save_detection_checkpoint(path, results)
        with open(path) as f:
            data = json.load(f)
        self.assertIn('images', data)
        self.assertEqual(data['images'], results)

    def test_overwrites_existing_checkpoint(self):
        path = str(Path(self.tmp_dir) / 'checkpoint3.json')
        save_detection_checkpoint(path, [{'image': 'a.jpg'}])
        save_detection_checkpoint(path, [{'image': 'b.jpg'}])
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data['images'][0]['image'], 'b.jpg')

    def test_none_path_raises(self):
        with self.assertRaises(AssertionError):
            save_detection_checkpoint(None, [])


class TestSequenceCalculation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest = pd.DataFrame({
            'filepath': ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'],
            'station': ['cam1', 'cam1', 'cam1', 'cam2'],
            'datetime': [
                '2023-01-01 10:00:00',
                '2023-01-01 10:00:30',
                '2023-01-01 10:05:00',
                '2023-01-01 10:00:00',
            ]
        })

    def test_returns_dataframe(self):
        result = sequence_calculation(self.manifest.copy(), station_col='station')
        self.assertIsInstance(result, pd.DataFrame)

    def test_sequence_column_added(self):
        result = sequence_calculation(self.manifest.copy(), station_col='station')
        self.assertIn('sequence', result.columns)

    def test_close_images_same_sequence(self):
        result = sequence_calculation(self.manifest.copy(), station_col='station')
        cam1 = result[result['station'] == 'cam1'].sort_values('datetime')
        self.assertEqual(cam1.iloc[0]['sequence'], cam1.iloc[1]['sequence'])

    def test_far_images_different_sequence(self):
        result = sequence_calculation(self.manifest.copy(), station_col='station')
        cam1 = result[result['station'] == 'cam1'].sort_values('datetime')
        self.assertNotEqual(cam1.iloc[0]['sequence'], cam1.iloc[2]['sequence'])

    def test_invalid_station_col_raises(self):
        with self.assertRaises(Exception):
            sequence_calculation(self.manifest.copy(), station_col='')

    def test_invalid_maxdiff_raises(self):
        with self.assertRaises(Exception):
            sequence_calculation(self.manifest.copy(), station_col='station', maxdiff=-1)

    def test_missing_filepath_col_raises(self):
        df = self.manifest.copy().rename(columns={'filepath': 'path'})
        with self.assertRaises(ValueError):
            sequence_calculation(df, station_col='station', file_col='filepath')

    def test_missing_datetime_col_raises(self):
        df = self.manifest.copy().drop(columns=['datetime'])
        with self.assertRaises(ValueError):
            sequence_calculation(df, station_col='station')

    def test_custom_timestamp_col(self):
        df = self.manifest.copy().rename(columns={'datetime': 'time'})
        result = sequence_calculation(df, station_col='station', timestamp_col='time')
        self.assertIn('sequence', result.columns)

    def test_missing_custom_timestamp_col_raises(self):
        with self.assertRaises(ValueError):
            sequence_calculation(self.manifest.copy(), station_col='station', timestamp_col='time')

    def test_custom_maxdiff(self):
        result = sequence_calculation(self.manifest.copy(), station_col='station', maxdiff=10)
        cam1 = result[result['station'] == 'cam1'].sort_values('datetime').reset_index(drop=True)
        self.assertNotEqual(cam1.iloc[0]['sequence'], cam1.iloc[1]['sequence'])

    def test_different_cameras_different_sequences(self):
        result = sequence_calculation(self.manifest.copy(), station_col='station')
        cam1_seq = set(result[result['station'] == 'cam1']['sequence'])
        cam2_seq = set(result[result['station'] == 'cam2']['sequence'])
        self.assertTrue(cam1_seq.isdisjoint(cam2_seq))

    def test_custom_sort_columns_explicit(self):
        default_result = sequence_calculation(self.manifest.copy(), station_col='station')
        explicit_result = sequence_calculation(
            self.manifest.copy(),
            station_col='station',
            sort_columns=['station', 'datetime'],
        )
        default_sequences = default_result.sort_values('filepath')['sequence'].tolist()
        explicit_sequences = explicit_result.sort_values('filepath')['sequence'].tolist()
        self.assertEqual(default_sequences, explicit_sequences)

    def test_custom_sort_columns(self):
        result = sequence_calculation(
            self.manifest.copy(),
            station_col='station',
            sort_columns=['station', 'datetime'],
        )
        self.assertIn('sequence', result.columns)

    def test_zero_maxdiff(self):
        result = sequence_calculation(self.manifest.copy(), station_col='station', maxdiff=0)
        self.assertEqual(len(set(result['sequence'])), len(result))

    def test_single_row_dataframe(self):
        one_row = self.manifest.head(1).copy()
        result = sequence_calculation(one_row, station_col='station')
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['sequence'], 0)


if __name__ == '__main__':
    unittest.main()
