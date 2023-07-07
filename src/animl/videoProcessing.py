import os
import cv2
from random import randrange
import multiprocessing as mp
import fileManagement
import pandas as pd
from numpy import vstack

# TODO 
#
# 1. check if video file is corrupt
# 2. implement checkpoint


# Extract frames from video for classification
#
# @param file_path dataframe of videos
# @param out_dir directory to save frames to
# @param fps frames per second, otherwise determine mathematically
# @param frames number of frames to sample
#
# @return dataframe of still frames for each video
def extractImages(file_path, out_dir, fps=None, frames=None):
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
            frames_saved.append([out_path,file_path])
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
            frames_saved.append([out_path,file_path])
            frame_capture += fps

    cap.release()
    cv2.destroyAllWindows()
    
    return frames_saved


# Extract frames from video for classification
#
# @param files dataframe of videos
# @param out_dir directory to save frames to
# @param outfile file to which results will be saved
# @param format output format for frames, defaults to jpg
# @param fps frames per second, otherwise determine mathematically
# @param frames number of frames to sample
# @param parallel Toggle for parallel processing, defaults to FALSE
# @param workers number of processors to use if parallel, defaults to 1
# @param checkpoint if not parallel, checkpoint ever n files, defaults to 1000
#
# @return dataframe of still frames for each video
def images_from_videos(files, out_dir, out_file=None, format="jpg",
                       fps=None, frames=None, parallel=False, 
                       workers=mp.cpu_count(), checkpoint=1000):
    if fileManagement.check_file(out_file):
        return fileManagement.load_data(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if (fps is not None) and (frames is not None):
        print("If both fps and frames are defined fps will be used.")
    if (fps is None) and (frames is None):
        raise AssertionError("Either fps or frames need to be defined.")
    # if fileManagement.check_file(outfile): 
    #    temporary = fileManagement.load_data(outfile)
    #    check against checkpoint   
        
    images = files[files["FileName"].apply(
        lambda x: os.path.splitext(x)[1].lower()).isin([".jpg",".jpeg",".png"])]
    images = images.assign(Frame = images["FilePath"])
    
    videos =  files[files["FileName"].apply(
        lambda x: os.path.splitext(x)[1].lower()).isin([".mp4", ".avi", ".mov", ".wmv",
                                                        ".mpg", ".mpeg", ".asf", ".m4v"])]
    if not videos.empty:
        if parallel:
            pool = mp.Pool(workers)

            video_frames = vstack([pool.apply(extractImages, 
                                               args=(video, out_dir, fps, frames)) for 
                                               video in videos["FilePath"]])

            video_frames = pd.DataFrame(video_frames, columns = ["Frame","FilePath"])

            pool.close()

        else:
            video_frames = []
            for i, video in videos.iterrows():
                video_frames += extractImages(video["FilePath"],out_dir=out_dir, 
                                         fps=fps, frames=frames)

                if (i % checkpoint == 0) and (outfile is not None):
                    fileManagement.save_data(images,outfile)

            video_frames = pd.DataFrame(video_frames, columns = ["Frame","FilePath"])

        videos = videos.merge(video_frames, on="FilePath")

    allframes = pd.concat([images,videos])

    if (out_file is not None):
        fileManagement.save_data(allframes,out_file) 

    return allframes