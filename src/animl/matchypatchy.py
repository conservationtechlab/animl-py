"""
API for MatchyPatchy

"""

import torch
import pandas as pd
from tqdm import tqdm
from animl.megadetector import MegaDetector
from animl.video_processing import extract_frames
from animl.detect import detect_MD_batch, parse_MD
from animl.inference import get_device
from animl.split import get_animals

from animl.generator import reid_dataloader
from animl.reid import viewpoint, miewid


def extract_frames(media, frame_dir):
    frames = extract_frames(media, frame_dir, frames=1)
    return frames


def detect(detector_file, media):
    """
    Function for integration with MatchyPatchy
    """
    detector = MegaDetector(detector_file, device=get_device())
    md_results = detect_MD_batch(detector, media, file_col="filepath", quiet=True)
    detections = parse_MD(md_results, manifest=media)
    detections = get_animals(detections)
    return detections


def viewpoint_estimator(rois, image_paths, viewpoint_filepath):
    device = get_device()
    rois = filter(rois)
    output = []
    if len(rois) > 0:
        viewpoint_dl = reid_dataloader(rois, image_paths, viewpoint.IMAGE_HEIGHT, viewpoint.IMAGE_WIDTH)
        model = viewpoint.load(viewpoint_filepath, device=device)
        with torch.no_grad():
            for _, batch in tqdm(enumerate(viewpoint_dl)):
                img = batch[0]
                roi_id = batch[1].numpy()[0]
                vp = model(img.to(device))
                value = torch.argmax(vp, dim=1).cpu().detach().numpy()[0]
                prob = torch.max(torch.nn.functional.softmax(vp, dim=1), 1)[0]
                prob = prob.cpu().detach().numpy()[0]
                output.append([roi_id,value,prob])
    return output


def miew_embedding(rois, image_paths, miew_filepath):
    device = get_device()
    rois = filter(rois)
    output = []
    if len(rois) > 0:
        dataloader = reid_dataloader(rois, image_paths, miewid.IMAGE_HEIGHT, miewid.IMAGE_WIDTH)
        model = miewid.load(miew_filepath, device=device)
        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader)):
                img = batch[0]
                roi_id = batch[1].numpy()[0]
                emb = model.extract_feat(img.to(device))
                emb = emb.cpu().detach().numpy()[0]
                output.append([roi_id, emb])
    viewpoints = pd.DataFrame(output, columns = ['id', 'value', 'prob'])
    return viewpoint