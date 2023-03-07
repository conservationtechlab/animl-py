import os
import cv2
import random
import multiprocessing as mp


def extractImages(file_path, out_dir, fps=None, frames=None):
    cap = cv2.VideoCapture(file_path)
    filename = os.path.basename(file_path)
    filename, extension = os.path.splitext(filename)
    frames_saved = []

    if fps is None:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_capture = 0
        increment = int(frame_count / frames)
        while cap.isOpened() and (len(frames_saved) < frames):
            cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, frame_capture)
            ret, frame = cap.read()
            if not ret:
                break

            out_path = out_dir + filename + "-" + 
                       '{:05}'.format(random.randrange(1, 10 ** 5)) + "-" + 
                       str(frame_capture) + '.jpg'
            cv2.imwrite(out_path, frame)
            frames_saved.append(out_path)
            frame_capture += increment
    else:
        frame_capture = 0
        while cap.isOpened():
            ret, frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_MSEC, (frame_capture * 1000))
            if not ret:
                break

            out_path = out_dir + filename + "-" + 
                       '{:05}'.format(random.randrange(1, 10 ** 5)) + "-" + 
                       str(frame_capture) + '.jpg'
            cv2.imwrite(out_path, frame)
            frames_saved.append(out_path)
            frame_capture += fps

        cap.release()
        cv2.destroyAllWindows()

    return frames_saved

def imagesFromVideos(image_dir, out_dir, outfile=None, format="jpg", fps=None,
                     frames=None, parallel=False, workers=mp.cpu_count()):
    assert os.path.isdir(image_dir), "image_dir does not exist"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    files = [image_dir + x for x in os.listdir(image_dir)]
    if (fps is not None) and (frames is not None):
        print("If both fps and frames are defined fps will be used.")
    assert (fps is not None) or (frames is not None), "Either fps or frames need to be defined."
    images = []
    videos = []
    for file in files:
        filename, extension = os.path.splitext(file)
        extension = extension.lower()
        if extension == '.jpg' or extension == '.png':
            images.append(file)
        else:
            videos.append(file)

    if parallel:
        pool = mp.Pool(workers)
        videoFrames = [pool.apply(extractImages, args=(video, out_dir, fps, frames)) for video in videos]
        pool.close()
        for x in videoFrames:
            images += x
    else:
        for video in videos:
            images += (extractImages(video, out_dir=out_dir, fps=fps, frames=frames))
    return images
