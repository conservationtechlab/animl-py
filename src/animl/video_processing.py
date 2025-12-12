"""
Video Processing Functions

"""
import cv2
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from numpy import vstack
from pathlib import Path
from typing import Optional

from animl import file_management
from animl.utils.general import NUM_THREADS


def extract_frames(files,
                   frames: int = 5,
                   fps: Optional[int] = None,
                   out_file: Optional[str] = None,
                   out_dir: str = None,
                   file_col: str = "filepath",
                   parallel: bool = True,
                   num_workers: int = NUM_THREADS):
    """
    Extract frames from video files in a given DataFrame.
    Can sample frames based on a specified number of frames or frames per second (fps).

    Args:
        files (pd.DataFrame): DataFrame containing file paths to videos and images.
        frames (int): Number of frames to sample from each video (default is 5).
        fps (Optional[int]): Frames per second to sample from each video. If specified, overrides frames.
        out_file (Optional[str]): Path to save the extracted frames manifest as a CSV file.
        out_dir (str): Directory to save extracted frame images. If None, frames are not saved as images.
        file_col (str): Column name in the DataFrame that contains the file paths (default is "filepath").
        parallel (bool): Whether to use multiprocessing for frame extraction (default is True).
        num_workers (int): Number of worker processes to use for parallel processing (default is NUM_THREADS).

    Raises:
        ValueError: If the DataFrame does not contain the specified file column or if both fps and frames are not defined.
        FileNotFoundError: If a video file does not exist.
        AssertionError: If neither fps nor frames are defined.

    Returns:
        pd.DataFrame: A DataFrame containing the file paths and corresponding frame numbers for the extracted frames.
                      The DataFrame will have columns [file_col, "frame"].
    """
    if file_management.check_file(out_file, output_type="ImageFrames"):
        return file_management.load_data(out_file)
    if not {file_col}.issubset(files.columns):
        raise ValueError(f"DataFrame must contain '{file_col}' column.")
    if (fps is not None) and (frames is not None):
        print("If both fps and frames are defined fps will be used.")
    if (fps is None) and (frames is None):
        raise AssertionError("Either fps or frames need to be defined.")

    images = files[files[file_col].apply(
        lambda x: Path(x).suffix.lower()).isin(file_management.IMAGE_EXTENSIONS)]
    images = images.assign(frame=0)

    videos = files[files[file_col].apply(
        lambda x: Path(x).suffix.lower()).isin(file_management.VIDEO_EXTENSIONS)]

    if not videos.empty:
        video_frames = []
        if parallel:
            pool = mp.Pool(num_workers)
            output = [pool.apply(count_frames, args=(video, frames, fps)) for video in tqdm(videos[file_col])]
            output = list(filter(None, output))
            video_frames = vstack(output)
            video_frames = pd.DataFrame(video_frames, columns=[file_col, "frame"])
            video_frames['frame'] = video_frames['frame'].astype(int)
            pool.close()

        else:
            for i, video in tqdm(enumerate(videos[file_col])):
                output = count_frames(video, frames=frames, fps=fps)
                if output is not None:
                    video_frames.extend(output)

            video_frames = pd.DataFrame(video_frames, columns=[file_col, "frame"])
            video_frames['frame'] = video_frames['frame'].astype(int)
        videos = videos.merge(video_frames, on=file_col)

    allframes = pd.concat([images, videos]).reset_index(drop=True)

    if (out_file is not None):
        file_management.save_data(allframes, out_file)

    if out_dir is not None:
        for _, row in videos.iterrows():
            image = get_frame_as_image(row[file_col], frame=row['frame'])
            image_path = Path(out_dir) / f"{Path(row[file_col]).stem}_{row['frame']}.jpg"
            cv2.imwrite(str(image_path), image)

    return allframes


def count_frames(filepath, frames=5, fps=None) -> int:
    """
    Count number of frames in a video

    Args:
        filepath: path to video file
        frames: number of frames to sample
        fps: frames per second to sample

    Returns:
        frames_saved: list of frames to be extracted
    """
    if not Path(filepath).is_file():
        raise FileNotFoundError(f"Video file {filepath} does not exist")

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():  # corrupted video
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        # print(f"Video file {filepath} has 0 frames, skipping.")
        return None

    cap.release()
    cv2.destroyAllWindows()

    frames_saved = []
    frame_capture = 0

    # select by fps
    if fps is not None:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            # try to calculate fps from duration
            duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Sometimes unreliable
            if duration > 0:
                video_fps = frame_count / duration
            else:
                print("Could not determine video FPS, defaulting to set number of frames")
                increment = int(frame_count / frames)
                while len(frames_saved) < frames:
                    frames_saved.append([str(filepath), frame_capture])
                    frame_capture += increment
                return frames_saved

        frames = int(frame_count / video_fps * fps)
        sampled_times = [i / fps for i in range(frames)]
        frames_saved = [min(int(round(t * video_fps)), frame_count-1) for t in sampled_times]

    # select set number of frames
    else:
        increment = int(frame_count / frames)
        while len(frames_saved) < frames:
            frames_saved.append([str(filepath), frame_capture])
            frame_capture += increment

    return frames_saved


# get specific frame of video as QImage
def get_frame_as_image(video_path, frame=0):
    """
    Given a video path, return a specific frame as an RGB image

    Args:
        video_path: path to video file
        frame: frame number to extract  (default is 0)

    Returns:
        rgb_frame: extracted frame as RGB image
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, still = cap.read()  # Read the first frame
    cap.release()

    if ret:
        rgb_frame = cv2.cvtColor(still, cv2.COLOR_BGR2RGB)
    return rgb_frame
