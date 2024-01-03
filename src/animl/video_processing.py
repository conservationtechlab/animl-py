import os
import cv2
from tqdm import tqdm
from random import randrange
import multiprocessing as mp
import pandas as pd
from numpy import vstack
from . import file_management


def extract_images(file_path, out_dir, fps=None, frames=None):
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
    cap = cv2.VideoCapture(file_path)
    filename = os.path.basename(file_path)
    filename, extension = os.path.splitext(filename)
    uniqueid = '{:05}'.format(randrange(1, 10 ** 5))
    frames_saved = []

    if fps is None:  # select set number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_capture = 0
        increment = int(frame_count / frames)
        while cap.isOpened() and (len(frames_saved) < frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_capture)
            ret, frame = cap.read()
            if not ret:
                break

            out_path = (out_dir + filename + "-" +
                        uniqueid + "-" +
                        str(frame_capture) + '.jpg')
            cv2.imwrite(out_path, frame)
            frames_saved.append([out_path, file_path])
            frame_capture += increment

    else:  # select by fps
        frame_capture = 0
        while cap.isOpened():
            ret, frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_MSEC, (frame_capture * 1000))
            if not ret:
                break

            out_path = (out_dir + filename + "-" +
                        '{:05}'.format(randrange(1, 10 ** 5)) + "-" +
                        str(frame_capture) + '.jpg')
            cv2.imwrite(out_path, frame)
            frames_saved.append([out_path, file_path])
            frame_capture += fps

    cap.release()
    cv2.destroyAllWindows()
    if len(frames_saved) == 0:
        return ["File Error", file_path]

    else:
        return frames_saved


def images_from_videos(files, out_dir, out_file=None, format="jpg",
                       fps=None, frames=None, parallel=False,
                       workers=mp.cpu_count(), checkpoint=1000):
    """
    Extract frames from video for classification

    Args
        - files: dataframe of videos
        - out_dir: directory to save frames to
        - out_file: file to which results will be saved
        - format: output format for frames, defaults to jpg
        - fps: frames per second, otherwise determine mathematically
        - frames: number of frames to sample
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

    images = files[files["FileName"].apply(
        lambda x: os.path.splitext(x)[1].lower()).isin([".jpg", ".jpeg", ".png"])]
    images = images.assign(Frame=images["FilePath"])

    videos = files[files["FileName"].apply(
        lambda x: os.path.splitext(x)[1].lower()).isin([".mp4", ".avi", ".mov", ".wmv",
                                                        ".mpg", ".mpeg", ".asf", ".m4v"])]
    if not videos.empty:
        if parallel:
            pool = mp.Pool(workers)

            video_frames = vstack([pool.apply(extract_images, args=(video, out_dir, fps, frames))
                                   for video in tqdm(videos["FilePath"])])

            video_frames = pd.DataFrame(video_frames, columns=["Frame", "FilePath"])

            pool.close()

        else:
            video_frames = []
            for i, video in tqdm(videos.iterrows()):
                video_frames += extract_images(video["FilePath"], out_dir=out_dir,
                                               fps=fps, frames=frames)

                if (i % checkpoint == 0) and (out_file is not None):
                    file_management.save_data(images, out_file)

            video_frames = pd.DataFrame(video_frames, columns=["Frame", "FilePath"])

        videos = videos.merge(video_frames, on="FilePath")

    allframes = pd.concat([images, videos]).reset_index()

    if (out_file is not None):
        file_management.save_data(allframes, out_file)

    return allframes
