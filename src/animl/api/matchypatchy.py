"""
API for MatchyPatchy

"""
import yaml
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (Compose, Resize, ToTensor, Normalize)

from animl.classification import load_model, predict_species, single_classification
from animl.utils.torch_utils import get_device

from animl.reid import miewid

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    classes = pd.read_csv(classlist_file)
    classifier, classes = load_model(classifier_file, len(classes), device=get_device())
    predictions = predict_species(animals, classifier, classes, device=get_device(), file_col="filepath",
                                  resize_width=cfg.get('resize_width'), resize_height=cfg.get('resize_height'),
                                  normalize=cfg.get('normalize'), batch_size=4)
    animals = single_classification(animals, predictions)
    return animals


def viewpoint_estimator(model, batch, device=None):
    """
    Wrapper for viewpoint estimation within MatchyPatchy
    """
    if device is None:
        device = get_device()
    img = batch[0]
    roi_id = batch[1].numpy()[0]
    vp = model(img.to(device))
    value = torch.argmax(vp, dim=1).cpu().detach().numpy()[0]
    prob = torch.max(torch.nn.functional.softmax(vp, dim=1), 1)[0]
    prob = prob.cpu().detach().numpy()[0]

    return roi_id, value, prob


def miew_embedding(model, batch, device=None):
    """
    Wrapper for MiewID embedding extraction within MatchyPatchy
    """
    if device is None:
        device = get_device()

    # with torch.no_grad():
    img = batch[0]
    roi_id = batch[1].numpy()[0]
    emb = model.extract_feat(img.to(device))
    emb = emb.cpu().detach().numpy()[0]

    return roi_id, emb


def reid_dataloader(rois, image_path_dict,
                    resize_height=miewid.IMAGE_HEIGHT, resize_width=miewid.IMAGE_WIDTH,
                    batch_size=1, workers=1):
    '''
        Loads a dataset and wraps it in a PyTorch DataLoader object.
        Always dynamically crops

        Args:
            - rois (DataFrame): data to be fed into the model
            - image_path_dict (dict): map of roi_id to filepath
            - resize_width (int): size in pixels for input width
            - resize_height (int): size in pixels for input height
            - batch_size (int): size of each batch
            - workers (int): number of processes to handle the data

        Returns:
            dataloader object


        MIEWIDNET - 440, 440
    '''
    dataset_instance = MiewGenerator(rois, image_path_dict,
                                     resize_height=resize_height,
                                     resize_width=resize_width)

    dataLoader = DataLoader(dataset=dataset_instance,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers)
    return dataLoader


class MiewGenerator(Dataset):
    '''
    Data generator that crops images on the fly, requires relative bbox coordinates,
    ie from MegaDetector

    Options:
        - resize: dynamically resize images to target (square) [W,H]
    '''
    def __init__(self, x, image_path_dict, resize_height=440, resize_width=440):
        self.x = x.reset_index()
        self.image_path_dict = image_path_dict
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.transform = Compose([Resize((self.resize_height, self.resize_width)),
                                  ToTensor(),
                                  Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]), ])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        id = self.x.loc[idx, 'roi_id']
        media_id = self.x.loc[idx, 'media_id']
        image_name = self.image_path_dict[media_id]
        try:
            img = Image.open(image_name).convert('RGB')
        except OSError:
            print("File error", image_name)
            del self.x.iloc[idx]
            return self.__getitem__(idx)

        width, height = img.size

        bbox1 = self.x['bbox_x'].iloc[idx]
        bbox2 = self.x['bbox_y'].iloc[idx]
        bbox3 = self.x['bbox_w'].iloc[idx]
        bbox4 = self.x['bbox_h'].iloc[idx]

        left = width * bbox1
        top = height * bbox2
        right = width * (bbox1 + bbox3)
        bottom = height * (bbox2 + bbox4)

        img = img.crop((left, top, right, bottom))

        img_tensor = self.transform(img)
        img.close()

        return img_tensor, id
