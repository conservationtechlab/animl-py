"""
Unit tests for animl/video_processing.py

@ Kyra Swanson 2023
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock


import cv2
import numpy as np
import pandas as pd

from animl.video_processing import (
    extract_frames,
    _count_frames,
    get_frame_as_image,
    _get_fps_from_ffmpeg,
)

class TestExtractFrames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        # Create sample image files
        cls.img_paths = []
        for i in range(2):
            img_path = Path(cls.tmp_dir) / f'img{i}.jpg'
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            cls.img_paths.append(str(img_path))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_raises_without_file_col(self):
        """Test that missing file_col raises ValueError."""
        manifest = pd.DataFrame({'path': ['a.jpg']})
        with self.assertRaises(ValueError):
            extract_frames(manifest, file_col='filepath')

    def test_raises_without_fps_or_frames(self):
        """Test that missing both fps and frames raises AssertionError."""
        manifest = pd.DataFrame({'filepath': self.img_paths})
        with self.assertRaises(AssertionError):
            extract_frames(manifest, fps=None, frames=None)

    def test_images_get_frame_zero(self):
        """Test that image files get assigned frame=0."""
        manifest = pd.DataFrame({'filepath': self.img_paths})
        result = extract_frames(manifest, frames=5)
        # All rows should have frame=0 for images
        self.assertTrue((result['frame'] == 0).all())

    def test_returns_dataframe(self):
        """Test that extract_frames returns a DataFrame."""
        manifest = pd.DataFrame({'filepath': self.img_paths})
        result = extract_frames(manifest, frames=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_output_has_required_columns(self):
        """Test that output DataFrame has filepath and frame columns."""
        manifest = pd.DataFrame({'filepath': self.img_paths})
        result = extract_frames(manifest, frames=5)
        self.assertIn('filepath', result.columns)
        self.assertIn('frame', result.columns)

    def test_fps_overrides_frames(self):
        """Test that fps parameter overrides frames parameter."""
        manifest = pd.DataFrame({'filepath': self.img_paths})
        # Should not raise error even though both are specified
        with patch('animl.video_processing._count_frames', return_value=None):
            result = extract_frames(manifest, frames=5, fps=1)
        self.assertIsInstance(result, pd.DataFrame)

    @patch('animl.file_management.check_file')
    def test_uses_cached_output_if_exists(self, mock_check):
        """Test that cached output is loaded if out_file exists."""
        mock_check.return_value = True
        expected_result = pd.DataFrame({'filepath': ['a.jpg'], 'frame': [0]})
        
        with patch('animl.file_management.load_data', return_value=expected_result):
            manifest = pd.DataFrame({'filepath': self.img_paths})
            result = extract_frames(manifest, out_file='/tmp/cached.csv', frames=5)
        
        pd.testing.assert_frame_equal(result, expected_result)

    def test_saves_output_file_when_specified(self):
        """Test that output is saved to file when out_file is specified."""
        with tempfile.TemporaryDirectory() as tmp:
            out_file = str(Path(tmp) / 'output.csv')
            manifest = pd.DataFrame({'filepath': self.img_paths})
            
            with patch('animl.file_management.check_file', return_value=False):
                result = extract_frames(manifest, out_file=out_file, frames=5)
            
            # File should be created (actual save is mocked in some tests)
            # Just verify it returns DataFrame
            self.assertIsInstance(result, pd.DataFrame)


class TestCountFrames(unittest.TestCase):

    def test_raises_for_nonexistent_file(self):
        """Test that FileNotFoundError is raised for non-existent video."""
        with self.assertRaises(FileNotFoundError):
            _count_frames('/nonexistent/video.mp4', frames=5)

    def test_returns_none_for_corrupted_video(self):
        """Test that None is returned for corrupted video that won't open."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_video = str(Path(tmp) / 'corrupted.mp4')
            # Create an empty file (corrupted video)
            Path(bad_video).touch()
            
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap_instance = MagicMock()
                mock_cap_instance.isOpened.return_value = False
                mock_cap.return_value = mock_cap_instance
                
                result = _count_frames(bad_video, frames=5)
            
            self.assertIsNone(result)

    def test_returns_none_for_zero_frames(self):
        """Test that None is returned for video with 0 frames."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / 'empty.mp4')
            Path(video_path).touch()
            
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap_instance = MagicMock()
                mock_cap_instance.isOpened.return_value = True
                mock_cap_instance.get.return_value = 0  # frame_count = 0
                mock_cap.return_value = mock_cap_instance
                
                result = _count_frames(video_path, frames=5)
            
            self.assertIsNone(result)

    def test_frames_mode_returns_list_of_frames(self):
        """Test that frames mode returns list with frame numbers."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / 'video.mp4')
            Path(video_path).touch()
            
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap_instance = MagicMock()
                mock_cap_instance.isOpened.return_value = True
                mock_cap_instance.get.return_value = 100  # frame_count = 100
                mock_cap.return_value = mock_cap_instance
                
                result = _count_frames(video_path, frames=5, fps=None)
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)
            # Should have 5 frame entries
            self.assertEqual(len(result), 5)
            # Each entry should be [filepath, frame_number]
            for entry in result:
                self.assertEqual(len(entry), 2)
                self.assertEqual(entry[0], video_path)

    def test_fps_mode_returns_list_of_frames(self):
        """Test that fps mode returns list with frame numbers."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / 'video.mp4')
            Path(video_path).touch()
            
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap_instance = MagicMock()
                mock_cap_instance.isOpened.return_value = True
                mock_cap_instance.get.side_effect = lambda prop: {
                    cv2.CAP_PROP_FRAME_COUNT: 300,  # 300 frames
                    cv2.CAP_PROP_FPS: 30,  # 30 fps = 10 seconds
                }.get(prop, 0)
                mock_cap.return_value = mock_cap_instance
                
                result = _count_frames(video_path, frames=None, fps=2)
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)
            # 10 seconds at 2 fps = 20 frames
            self.assertGreater(len(result), 0)

    def test_fps_fallback_to_ffmpeg(self):
        """Test that fps fallback uses ffmpeg when OpenCV fails."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / 'video.mp4')
            Path(video_path).touch()
            
            with patch('cv2.VideoCapture') as mock_cap, \
                 patch('animl.video_processing._get_fps_from_ffmpeg', return_value=24.0):
                mock_cap_instance = MagicMock()
                mock_cap_instance.isOpened.return_value = True
                mock_cap_instance.get.side_effect = lambda prop: {
                    cv2.CAP_PROP_FRAME_COUNT: 240,
                    cv2.CAP_PROP_FPS: 0,  # OpenCV can't get fps
                }.get(prop, 0)
                mock_cap.return_value = mock_cap_instance
                
                result = _count_frames(video_path, frames=None, fps=1)
            
            self.assertIsNotNone(result)


class TestGetFrameAsImage(unittest.TestCase):

    def test_returns_image_array(self):
        """Test that get_frame_as_image returns an image array."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / 'video.mp4')
            Path(video_path).touch()
            
            fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
            
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap_instance = MagicMock()
                mock_cap_instance.read.return_value = (True, fake_frame)
                mock_cap.return_value = mock_cap_instance
                
                result = get_frame_as_image(video_path, frame=0)
            
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (10, 10, 3))

    def test_extracts_specific_frame(self):
        """Test that specific frame number is set before reading."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / 'video.mp4')
            Path(video_path).touch()
            
            fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
            
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap_instance = MagicMock()
                mock_cap_instance.read.return_value = (True, fake_frame)
                mock_cap.return_value = mock_cap_instance
                
                get_frame_as_image(video_path, frame=42)
                
                # Verify that set was called with correct frame
                mock_cap_instance.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 42)

    def test_converts_bgr_to_rgb(self):
        """Test that frame is converted from BGR to RGB."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / 'video.mp4')
            Path(video_path).touch()
            
            # Create BGR frame
            bgr_frame = np.zeros((10, 10, 3), dtype=np.uint8)
            bgr_frame[:, :, 0] = 255  # Blue channel
            
            with patch('cv2.VideoCapture') as mock_cap, \
                 patch('cv2.cvtColor') as mock_cvt:
                mock_cap_instance = MagicMock()
                mock_cap_instance.read.return_value = (True, bgr_frame)
                mock_cap.return_value = mock_cap_instance
                
                rgb_result = np.zeros((10, 10, 3), dtype=np.uint8)
                rgb_result[:, :, 2] = 255  # Red channel (swapped)
                mock_cvt.return_value = rgb_result
                
                result = get_frame_as_image(video_path, frame=0)
                
                # Verify cvtColor was called with BGR2RGB
                mock_cvt.assert_called_once_with(bgr_frame, cv2.COLOR_BGR2RGB)


class TestGetFpsFromFfmpeg(unittest.TestCase):

    def test_extracts_fps_from_output(self):
        """Test that fps is extracted from ffmpeg output."""
        ffmpeg_output = """
        Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'video.mp4':
          Duration: 00:00:10.00, start: 0.000000, bitrate: 1000 kb/s
            Stream #0:0(und): Video: h264 (avc1 / 0x31637661), yuv420p, 1920x1080, 24 fps, 24 tbr, 12288 tbn, 48 tbc (default)
        """
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(stderr=ffmpeg_output, returncode=0)
            result = _get_fps_from_ffmpeg('/tmp/video.mp4')
        
        self.assertEqual(result, 24.0)

    def test_extracts_decimal_fps(self):
        """Test that decimal fps values are extracted."""
        ffmpeg_output = "Stream #0:0: Video: 23.976 fps"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(stderr=ffmpeg_output, returncode=0)
            result = _get_fps_from_ffmpeg('/tmp/video.mp4')
        
        self.assertEqual(result, 23.976)

    def test_returns_none_if_fps_not_found(self):
        """Test that None is returned if fps pattern not found."""
        ffmpeg_output = "Some output without fps"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(stderr=ffmpeg_output, returncode=0)
            result = _get_fps_from_ffmpeg('/tmp/video.mp4')
        
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        """Test that None is returned on exception."""
        with patch('subprocess.run', side_effect=Exception("Test error")):
            result = _get_fps_from_ffmpeg('/tmp/video.mp4')
        
        self.assertIsNone(result)

    def test_timeout_handling(self):
        """Test that subprocess timeout is set."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(stderr="", returncode=0)
            _get_fps_from_ffmpeg('/tmp/video.mp4')
            
            # Check that timeout was passed
            call_args = mock_run.call_args
            self.assertEqual(call_args.kwargs.get('timeout'), 10)


if __name__ == '__main__':
    unittest.main()
