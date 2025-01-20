import os
import cv2
from tqdm import tqdm
from random import randrange
import multiprocessing as mp
import pandas as pd
from numpy import vstack
from animl import file_management
import pathlib
from typing import Optional, Union, List


def extract_frame_single(file_path: Union[str, pd.DataFrame],
                         out_dir: str,
                         fps: Optional[float] = None,
                         frames: Optional[int] = None) -> pd.DataFrame:
    """
    Extract frames from video for classification

    Args
        - file_path: dataframe of videos
        - out_dir: directory to save frames to
        - fps: frames per second, otherwise determine mathematically
        - frames: number of frames to sample

    Return
        - frames_saved: dataframe of still frames for each video
    """
    # Typechecking
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be a string, got {type(file_path)}")
    if not isinstance(out_dir, pathlib.PosixPath):
        raise TypeError(f"out_dir must be a string, got {type(out_dir)}")
    if fps is not None and not isinstance(fps, (int, float)):
        raise TypeError(f"fps must be an integer or float, got {type(fps)}")
    if frames is not None and not isinstance(frames, int):
        raise TypeError(f"frames must be an integer, got {type(frames)}")
    if frames is None and fps is None:
        raise ValueError("Either fps or frames must be specified")

    # File and Directory Validation
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Video file {file_path} does not exist")
    if not os.path.isdir(out_dir):
        raise NotADirectoryError(f"Output directory {out_dir} does not exist")

    cap = cv2.VideoCapture(file_path)
    filename = os.path.basename(file_path)
    filename, extension = os.path.splitext(filename)
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
            out_path = os.path.join(str(out_dir), frame_name)
            cv2.imwrite(out_path, frame)
            frames_saved.append([out_path, file_path, frame_capture])
            frame_capture += increment

    else:  # select by fps
        frame_capture = 0
        while cap.isOpened():
            ret, frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_MSEC, (frame_capture * 1000))
            if not ret:
                break
            frame_name = filename + "-" + uniqueid + "-" + str(frame_capture) + '.jpg'
            out_path = os.path.join(str(out_dir), frame_name)
            cv2.imwrite(out_path, frame)
            frames_saved.append([out_path, file_path, frame_capture])
            frame_capture += fps

    cap.release()
    cv2.destroyAllWindows()
    if len(frames_saved) == 0:
        return ["File Error", file_path]

    else:
        return frames_saved


def extract_frames(files: Union[str, pd.DataFrame, List[str]],
                   out_dir: str,
                   out_file: Optional[str] = None,
                   fps: Optional[float] = None,
                   frames: Optional[int] = None,
                   file_col: str = "FilePath",
                   parallel: bool = False,
                   workers: int = mp.cpu_count(),
                   checkpoint: int = 1000):
    """
    Extract frames from video for classification

    Args
        - files: dataframe of videos
        - out_dir: directory to save frames to
        - out_file: file to which results will be saved
        - fps: frames per second, otherwise determine mathematically
        - frames: number of frames to sample
        - file_col: column containing file paths
        - parallel: Toggle for parallel processing, defaults to FALSE
        - workers: number of processors to use if parallel, defaults to 1
        - checkpoint: if not parallel, checkpoint ever n files, defaults to 1000

    Return
        - allframes: dataframe of still frames for each video
    """
    if file_management.check_file(out_file):
        return file_management.load_data(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if (fps is not None) and (frames is not None):
        print("If both fps and frames are defined fps will be used.")
    if (fps is None) and (frames is None):
        raise AssertionError("Either fps or frames need to be defined.")
    # if file_management.check_file(outfile):
    #    temporary = fileManagement.load_data(outfile)
    #    check against checkpoint

    images = files[files[file_col].apply(
        lambda x: os.path.splitext(x)[1].lower()).isin([".jpg", ".jpeg", ".png"])]
    images = images.assign(Frame=images[file_col])

    videos = files[files[file_col].apply(
        lambda x: os.path.splitext(x)[1].lower()).isin([".mp4", ".avi", ".mov", ".wmv",
                                                        ".mpg", ".mpeg", ".asf", ".m4v"])]
    if not videos.empty:
        # TODO add checkpoint to parallel
        if parallel:
            pool = mp.Pool(workers)

            video_frames = vstack([pool.apply(extract_frame_single, args=(video, out_dir, fps, frames))
                                   for video in tqdm(videos[file_col])])

            video_frames = pd.DataFrame(video_frames, columns=["Frame", file_col, "FrameNumber"])

            pool.close()

        else:
            video_frames = []
            for i, video in tqdm(videos.iterrows()):
                video_frames += extract_frame_single(video[file_col], out_dir=out_dir,
                                                     fps=fps, frames=frames)

                if (i % checkpoint == 0) and (out_file is not None):
                    file_management.save_data(images, out_file)

            video_frames = pd.DataFrame(video_frames, columns=["Frame", file_col, "FrameNumber"])

        videos = videos.merge(video_frames, on=file_col)

    allframes = pd.concat([images, videos]).reset_index()

    if (out_file is not None):
        file_management.save_data(allframes, out_file)

    return allframes
