"""
Unit tests for animl/export.py

@ Kyra Swanson 2023
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from animl.export import (
    export_camptrapdp,
    export_coco,
    export_folders,
    export_megadetector,
    export_yolo,
    remove_link,
    update_labels_from_folders,
)


def _make_coco_manifest(n=2):
    """Return a minimal manifest suitable for export_coco."""
    return pd.DataFrame({
        'filepath': [f'/tmp/img{i}.jpg' for i in range(n)],
        'filename': [f'img{i}.jpg' for i in range(n)],
        'filemodifydate': ['2023-01-01'] * n,
        'frame': [0] * n,
        'max_detection_conf': [0.9] * n,
        'category': [1] * n,
        'conf': [0.9] * n,
        'bbox_x': [0.1] * n,
        'bbox_y': [0.1] * n,
        'bbox_w': [0.2] * n,
        'bbox_h': [0.2] * n,
        'prediction': ['deer'] * n,
        'confidence': [0.85] * n,
        'width': [640] * n,
        'height': [480] * n,
    })


def _make_class_list():
    """Return a minimal class_list DataFrame."""
    return {0: 'empty', 1: 'deer'}


def _make_megadetector_manifest():
    """Return a minimal manifest for export_megadetector."""
    return pd.DataFrame({
        'filepath': ['/tmp/img0.jpg', '/tmp/img0.jpg', '/tmp/img1.jpg'],
        'category': [1, 1, 0],          # last row is empty → should be skipped
        'conf': [0.9, 0.8, 0.0],
        'bbox_x': [0.1, 0.2, 0.0],
        'bbox_y': [0.1, 0.2, 0.0],
        'bbox_w': [0.2, 0.1, 0.0],
        'bbox_h': [0.2, 0.1, 0.0],
        'prediction': ['deer', 'deer', 'empty'],
        'confidence': [0.85, 0.80, 1.0],
    })


def _make_camptrapdp_manifest():
    """Return a minimal manifest for export_camptrapdp."""
    return pd.DataFrame({
        'filepath': ['/tmp/img0.jpg', '/tmp/img0.jpg', '/tmp/img1.jpg'],
        'filename': ['img0.jpg', 'img0.jpg', 'img1.jpg'],
        'extension': ['.jpg', '.jpg', '.jpg'],
        'datetime': ['2023-01-01 12:00:00', '2023-01-01 12:00:00', '2023-01-01 13:00:00'],
        'category': [1, 1, 1],
        'prediction': ['deer', 'elk', 'deer'],
        'confidence': [0.85, 0.80, 0.90],
        'bbox_x': [0.1, 0.2, 0.3],
        'bbox_y': [0.1, 0.2, 0.3],
        'bbox_w': [0.2, 0.1, 0.2],
        'bbox_h': [0.2, 0.1, 0.2],
    })


class TestRemoveLink(unittest.TestCase):

    def test_files_are_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create two real files and put their paths in the manifest
            p1 = Path(tmp) / 'a.jpg'
            p2 = Path(tmp) / 'b.jpg'
            p1.write_bytes(b'dummy')
            p2.write_bytes(b'dummy')

            manifest = pd.DataFrame({'link': [str(p1), str(p2)]})
            remove_link(manifest, link_col='link')

            self.assertFalse(p1.exists())
            self.assertFalse(p2.exists())

    def test_returns_dataframe(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'c.jpg'
            p.write_bytes(b'dummy')
            manifest = pd.DataFrame({'link': [str(p)]})
            result = remove_link(manifest, link_col='link')
            self.assertIsInstance(result, pd.DataFrame)

    def test_missing_file_does_not_raise(self):
        manifest = pd.DataFrame({'link': ['/tmp/nonexistent_animl_test_file.jpg']})
        # Should not raise even if file doesn't exist (missing_ok=True)
        result = remove_link(manifest, link_col='link')
        self.assertIsInstance(result, pd.DataFrame)


class TestUpdateLabelsFromFolders(unittest.TestCase):

    def test_raises_when_uniquename_missing(self):
        manifest = pd.DataFrame({'filepath': ['/tmp/a.jpg'], 'prediction': ['deer']})
        with self.assertRaises(AssertionError):
            update_labels_from_folders(manifest, export_dir='/tmp/fake')

    def test_calls_build_file_manifest(self):
        manifest = pd.DataFrame({
            'filepath': ['/tmp/a.jpg'],
            'prediction': ['deer'],
            'uniquename': ['deer_001.jpg'],
        })
        fake_ground_truth = pd.DataFrame({
            'filepath': ['/tmp/sorted/deer/deer_001.jpg'],
            'filename': ['deer_001.jpg'],
        })
        with patch('animl.file_management.build_file_manifest', return_value=fake_ground_truth):
            result = update_labels_from_folders(manifest, export_dir='/tmp/sorted')
        self.assertIn('label', result.columns)


class TestExportCoco(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest = _make_coco_manifest()
        cls.class_list = _make_class_list()

    def test_output_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'coco.json')
            export_coco(self.manifest.copy(), self.class_list.copy(), out_file)
            self.assertTrue(Path(out_file).exists())

    def test_output_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'coco.json')
            export_coco(self.manifest.copy(), self.class_list.copy(), out_file)
            with open(out_file) as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)

    def test_top_level_keys_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'coco.json')
            result = export_coco(self.manifest.copy(), self.class_list.copy(), out_file)
            for key in ('info', 'licenses', 'images', 'annotations', 'categories'):
                self.assertIn(key, result)

    def test_categories_contain_class_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'coco.json')
            result = export_coco(self.manifest.copy(), self.class_list.copy(), out_file)
            category_names = [c['name'] for c in result['categories']]
            self.assertIn('deer', category_names)

    def test_missing_column_raises_assertion(self):
        bad_manifest = self.manifest.drop(columns=['bbox_x'])
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'coco.json')
            with self.assertRaises(AssertionError):
                export_coco(bad_manifest, self.class_list.copy(), out_file)

    def test_nan_bbox_rows_skipped_in_annotations(self):
        manifest = _make_coco_manifest(n=2)
        # Make first row have NaN bbox
        manifest.loc[0, 'bbox_x'] = float('nan')
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'coco.json')
            result = export_coco(manifest, self.class_list.copy(), out_file)
        # 2 rows but 1 has NaN bbox, so only 1 annotation
        self.assertEqual(len(result['annotations']), 1)

    def test_custom_info_preserved(self):
        custom_info = {'description': 'test', 'version': '1.0', 'date_created': '2023/01/01'}
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'coco.json')
            result = export_coco(self.manifest.copy(), self.class_list.copy(), out_file,
                                 info=custom_info)
        self.assertEqual(result['info']['description'], 'test')

    def test_custom_licenses_preserved(self):
        licenses = [{'id': 1, 'name': 'MIT'}]
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'coco.json')
            result = export_coco(self.manifest.copy(), self.class_list.copy(), out_file,
                                 licenses=licenses)
        self.assertEqual(result['licenses'], licenses)


class TestExportMegadetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest = _make_megadetector_manifest()

    def test_output_file_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'md.json')
            export_megadetector(self.manifest.copy(), out_file=out_file, prompt=False)
            self.assertTrue(Path(out_file).exists())

    def test_output_has_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'md.json')
            export_megadetector(self.manifest.copy(), out_file=out_file, prompt=False)
            with open(out_file) as f:
                data = json.load(f)
        for key in ('info', 'detection_categories', 'classification_categories', 'images'):
            self.assertIn(key, data)

    # TODO: rows with cat=0 should be included but have no detections
    def test_empty_category_rows_skipped(self):
        """Rows where category == 0 should not appear in images."""
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'md.json')
            export_megadetector(self.manifest.copy(), out_file=out_file, prompt=False)
            with open(out_file) as f:
                data = json.load(f)
        # img1.jpg has category=0 and should be excluded
        image_files = [im['file'] for im in data['images']]
        self.assertNotIn('/tmp/img1.jpg', image_files)

    def test_missing_column_raises_value_error(self):
        bad_manifest = self.manifest.drop(columns=['bbox_x'])
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'md.json')
            with self.assertRaises(ValueError):
                export_megadetector(bad_manifest, out_file=out_file, prompt=False)

    def test_detector_name_in_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'md.json')
            export_megadetector(self.manifest.copy(), out_file=out_file,
                                detector='TestDetector', prompt=False)
            with open(out_file) as f:
                data = json.load(f)
        self.assertEqual(data['info']['detector'], 'TestDetector')


class TestExportCamptrapdp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest = _make_camptrapdp_manifest()

    def test_media_csv_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_camptrapdp(self.manifest.copy(), out_dir=tmp)
            self.assertTrue((Path(tmp) / 'media.csv').exists())

    def test_observations_csv_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_camptrapdp(self.manifest.copy(), out_dir=tmp)
            self.assertTrue((Path(tmp) / 'observations.csv').exists())

    def test_datapackage_json_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_camptrapdp(self.manifest.copy(), out_dir=tmp)
            self.assertTrue((Path(tmp) / 'datapackage.json').exists())

    def test_media_csv_has_one_row_per_unique_filepath(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_camptrapdp(self.manifest.copy(), out_dir=tmp)
            media_df = pd.read_csv(Path(tmp) / 'media.csv')
        unique_filepaths = self.manifest['filepath'].nunique()
        self.assertEqual(len(media_df), unique_filepaths)

    def test_observations_csv_has_one_row_per_manifest_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_camptrapdp(self.manifest.copy(), out_dir=tmp)
            obs_df = pd.read_csv(Path(tmp) / 'observations.csv')
        self.assertEqual(len(obs_df), len(self.manifest))

    def test_datapackage_json_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_camptrapdp(self.manifest.copy(), out_dir=tmp)
            with open(Path(tmp) / 'datapackage.json') as f:
                data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_file_public_flag_propagated(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_camptrapdp(self.manifest.copy(), out_dir=tmp, file_public=True)
            media_df = pd.read_csv(Path(tmp) / 'media.csv')
        self.assertTrue(media_df['filePublic'].all())


class TestExportFolders(unittest.TestCase):

    def _make_manifest_with_images(self, tmp_dir, labels=('deer', 'elk')):
        """Create real image files and return a manifest pointing to them."""
        rows = []
        for i, label in enumerate(labels):
            img_path = Path(tmp_dir) / f'img{i}.jpg'
            img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
            img.save(str(img_path))
            rows.append({'filepath': str(img_path), 'prediction': label})
        return pd.DataFrame(rows)

    def test_raises_when_label_col_missing(self):
        manifest = pd.DataFrame({'filepath': ['/tmp/a.jpg']})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError):
                export_folders(manifest, out_dir=tmp)

    def test_species_subdirectories_created(self):
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as out_dir:
                manifest = self._make_manifest_with_images(src_dir, labels=['deer', 'elk'])
                export_folders(manifest, out_dir=out_dir, copy=True)
                self.assertTrue((Path(out_dir) / 'deer').is_dir())
                self.assertTrue((Path(out_dir) / 'elk').is_dir())

    def test_copy_copies_files_to_subdirs(self):
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as out_dir:
                manifest = self._make_manifest_with_images(src_dir, labels=['deer'])
                result = export_folders(manifest, out_dir=out_dir, copy=True)
                # Each link path should exist as a file
                for link in result['link']:
                    self.assertTrue(Path(link).is_file(), f"Expected {link} to exist")

    def test_returns_dataframe(self):
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as out_dir:
                manifest = self._make_manifest_with_images(src_dir, labels=['deer'])
                result = export_folders(manifest, out_dir=out_dir, copy=True)
                self.assertIsInstance(result, pd.DataFrame)

    def test_link_column_added(self):
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as out_dir:
                manifest = self._make_manifest_with_images(src_dir, labels=['deer'])
                result = export_folders(manifest, out_dir=out_dir, copy=True)
                self.assertIn('link', result.columns)


class TestExportYolo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.class_list = pd.DataFrame({'id': [0, 1], 'class': ['empty', 'deer']})

    def _make_yolo_manifest(self, tmp_dir, n=2, prefix='img'):
        rows = []
        for i in range(n):
            img_path = Path(tmp_dir) / f'{prefix}{i}.jpg'
            img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
            img.save(str(img_path))
            rows.append({
                'filepath': str(img_path),
                'prediction': 'deer',
                'bbox_x': 0.1,
                'bbox_y': 0.1,
                'bbox_w': 0.2,
                'bbox_h': 0.2,
            })
        return pd.DataFrame(rows)

    def test_directory_structure_created(self):
        with tempfile.TemporaryDirectory() as src:
            with tempfile.TemporaryDirectory() as out:
                train = self._make_yolo_manifest(src, n=2, prefix='tr')
                val = self._make_yolo_manifest(src, n=2, prefix='va')
                test = self._make_yolo_manifest(src, n=2, prefix='te')
                export_yolo(train, val, test, self.class_list, out_dir=out, hard_copy=True)
                for sub in ('images/train', 'images/val', 'images/test',
                            'labels/train', 'labels/val', 'labels/test'):
                    self.assertTrue((Path(out) / sub).is_dir(), f"Missing dir: {sub}")

    def test_dataset_yaml_created(self):
        with tempfile.TemporaryDirectory() as src:
            with tempfile.TemporaryDirectory() as out:
                manifest = self._make_yolo_manifest(src, prefix='dy')
                export_yolo(manifest, manifest, manifest, self.class_list, out_dir=out,
                            hard_copy=True)
                self.assertTrue((Path(out) / 'dataset.yaml').exists())

    def test_dataset_yaml_has_expected_keys(self):
        import yaml
        with tempfile.TemporaryDirectory() as src:
            with tempfile.TemporaryDirectory() as out:
                manifest = self._make_yolo_manifest(src, prefix='yk')
                export_yolo(manifest, manifest, manifest, self.class_list, out_dir=out,
                            hard_copy=True)
                with open(Path(out) / 'dataset.yaml') as f:
                    data = yaml.safe_load(f)
        for key in ('path', 'train', 'val', 'nc', 'names'):
            self.assertIn(key, data)

    def test_missing_column_raises_assertion(self):
        with tempfile.TemporaryDirectory() as src:
            with tempfile.TemporaryDirectory() as out:
                manifest = self._make_yolo_manifest(src, prefix='mc').drop(columns=['bbox_x'])
                with self.assertRaises(AssertionError):
                    export_yolo(manifest, manifest, manifest, self.class_list, out_dir=out)

    def test_none_test_manifest_accepted(self):
        with tempfile.TemporaryDirectory() as src:
            with tempfile.TemporaryDirectory() as out:
                train = self._make_yolo_manifest(src, n=2, prefix='nt')
                val = self._make_yolo_manifest(src, n=2, prefix='nv')
                # Should not raise with test_manifest=None
                export_yolo(train, val, None, self.class_list, out_dir=out, hard_copy=True)
                self.assertTrue((Path(out) / 'dataset.yaml').exists())

    def test_label_files_written(self):
        with tempfile.TemporaryDirectory() as src:
            with tempfile.TemporaryDirectory() as out:
                train = self._make_yolo_manifest(src, n=1, prefix='lf')
                val = self._make_yolo_manifest(src, n=1, prefix='lv')
                export_yolo(train, val, None, self.class_list, out_dir=out, hard_copy=True)
                label_files = list((Path(out) / 'labels' / 'train').glob('*.txt'))
                self.assertGreater(len(label_files), 0)


if __name__ == '__main__':
    unittest.main()
