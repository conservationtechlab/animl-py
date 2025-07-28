"""
Video Processing Functions

"""
import cv2
from tqdm import tqdm
from random import randrange
from PIL import Image
import multiprocessing as mp
import pandas as pd
import numpy as np
from numpy import vstack
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (Compose, Resize, ToTensor)


from animl import file_management
from animl.utils.general import NUM_THREADS



def extract_frame_single(file_path: Union[str, pd.DataFrame],
                         out_dir: str,
                         fps: Optional[float] = None,
                         frames: Optional[int] = None) -> pd.DataFrame:
    """
    Extract frames from video for classification

    Args:
        file_path: dataframe of videos
        out_dir: directory to save frames to
        fps: frames per second, otherwise determine mathematically
        frames: number of frames to sample

    Return
        frames_saved: dataframe of still frames for each video
    """
    # Typechecking
    if frames is None and fps is None:
        raise ValueError("Either fps or frames must be specified")

    # File and Directory Validation
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"Video file {file_path} does not exist")
    if not Path(out_dir).is_dir():
        raise NotADirectoryError(f"Output directory {out_dir} does not exist")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():  # corrupted video
        return

    filename = Path(file_path).stem
    uniqueid = '{:05}'.format(randrange(1, 10 ** 5))
    frames_saved = []

    # Typechecking FPS
    if fps == 'None':
        fps = None
    if fps is None:  # select set number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_capture = 0
        increment = int(frame_count / frames)
        while cap.isOpened() and (len(frames_saved) < frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_capture)
            ret, frame = cap.read()
            if not ret:
                break
            frame_name = filename + "-" + uniqueid + "-" + str(frame_capture) + '.jpg'
            out_path = str(out_dir) + '/' + frame_name
            cv2.imwrite(out_path, frame)
            frames_saved.append([out_path, str(file_path), frame_capture])
            frame_capture += increment

    else:  # select by fps
        frame_capture = 0
        while cap.isOpened():
            ret, frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_MSEC, (frame_capture * 1000))
            if not ret:
                break
            frame_name = filename + "-" + uniqueid + "-" + str(frame_capture) + '.jpg'
            out_path = str(out_dir) + '/' + frame_name
            cv2.imwrite(out_path, frame)
            frames_saved.append([out_path, str(file_path), frame_capture])
            frame_capture += fps

    cap.release()
    cv2.destroyAllWindows()
    # corrupted video
    if len(frames_saved) == 0:
        return
    else:
        return frames_saved


def extract_frames(files: Union[str, pd.DataFrame, list[str]],
                   out_dir: str,
                   out_file: Optional[str] = None,
                   fps: Optional[float] = None,
                   frames: Optional[int] = None,
                   file_col: str = "filepath",
                   parallel: bool = False,
                   num_workers: int = NUM_THREADS,
                   checkpoint: int = 1000):
    """
    Extract frames from video for classification

    Args:
        files: dataframe of videos
        out_dir: directory to save frames to
        out_file: file to which results will be saved
        fps: frames per second, otherwise determine mathematically
        frames: number of frames to sample
        file_col: column containing file paths
        parallel: Toggle for parallel processing, defaults to FALSE
        num_workers: number of processors to use if parallel, defaults to NUM_THREADS
        checkpoint: if not parallel, checkpoint ever n files, defaults to 1000

    Returns:
        allframes: dataframe of still frames for each video
    """
    if file_management.check_file(out_file):
        return file_management.load_data(out_file)
    if (fps is not None) and (frames is not None):
        print("If both fps and frames are defined fps will be used.")
    if (fps is None) and (frames is None):
        raise AssertionError("Either fps or frames need to be defined.")

    Path(out_dir).mkdir(exist_ok=True)
    images = files[files[file_col].apply(
        lambda x: Path(x).suffix.lower()).isin(file_management.IMAGE_EXTENSIONS)]
    images = images.assign(frame=images[file_col])
    images = images.assign(framenumber=0)

    videos = files[files[file_col].apply(
        lambda x: Path(x).suffix.lower()).isin(file_management.VIDEO_EXTENSIONS)]

    videos = videos.drop(columns="frame", errors='ignore')

    if not videos.empty:
        video_frames = []
        if parallel:
            pool = mp.Pool(num_workers)
            output = [pool.apply(extract_frame_single, args=(video, out_dir, fps, frames)) for video in tqdm(videos[file_col])]
            output = list(filter(None, output))
            video_frames = vstack(output)
            video_frames = pd.DataFrame(video_frames, columns=["frame", file_col, "framenumber"])
            video_frames['framenumber'] = video_frames['framenumber'].astype(int)
            pool.close()

        else:
            for i, video in tqdm(enumerate(videos[file_col])):
                output = extract_frame_single(video, out_dir=out_dir,
                                              fps=fps, frames=frames)
                if output is not None:
                    video_frames.extend(output)

                if (i % checkpoint == 0) and (out_file is not None):
                    file_management.save_data(images, out_file)

            video_frames = pd.DataFrame(video_frames, columns=["frame", file_col, "framenumber"])
        videos = videos.merge(video_frames, on=file_col)

    allframes = pd.concat([images, videos]).reset_index(drop=True)

    if (out_file is not None):
        file_management.save_data(allframes, out_file)

    return allframes
