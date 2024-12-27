"""
API for MatchyPatchy

"""
import torch
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from animl.models.megadetector import MegaDetector
from animl.video_processing import extract_frames
from animl.detect import detect_MD_batch, parse_MD
from animl.split import get_animals
from animl.classification import load_model, predict_species
from animl.utils.torch_utils import get_device

from animl.generator import reid_dataloader
from animl.reid import viewpoint, miewid


def process_videos(media, frame_dir):
    """
    Wrapper to extract frames for MatchyPatchy
    """
    frames = extract_frames(media, frame_dir, frames=1, file_col="filepath")
    return frames


def detect_mp(detector_file, media):
    """
    Wrapper for object detection withinin MatchyPatchy
    """
    detector = MegaDetector(detector_file, device=get_device())
    md_results = detect_MD_batch(detector, media, file_col="filepath", quiet=True)
    detections = parse_MD(md_results, manifest=media)
    detections = get_animals(detections)
    return detections


def classify_mp(animals, config_file):
    """
    Wrapper for classification within MatchyPatchy
    """
    try:
        cfg = yaml.safe_load(open(config_file, 'r'))
    except yaml.YAMLError as exc:
        print(exc)
    classifier_file = config_file.parent / Path(cfg.get('file_name'))
    classlist_file = config_file.parent / Path(cfg.get('class_file'))
    classifier, classes = load_model(classifier_file, classlist_file, device=get_device())
    animals = predict_species(animals, classifier, classes, device=get_device(), file_col="filepath",
                              resize_width=cfg.get('resize_width'), resize_height=cfg.get('resize_height'),
                              normalize=cfg.get('normalize'), batch_size=4)
    return animals


def viewpoint_estimator(rois, image_paths, viewpoint_filepath):
    """
    Wrapper for viewpoint estimation within MatchyPatchy
    """
    device = get_device()
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
                output.append([roi_id, value, prob])
    viewpoints = pd.DataFrame(output, columns=['id', 'value', 'prob'])
    return viewpoints


def miew_embedding(rois, image_paths, miew_filepath):
    """
    Wrapper for MiewID embedding extraction within MatchyPatchy
    """
    device = get_device()
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
    return output
