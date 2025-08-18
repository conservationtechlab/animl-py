"""
API for MatchyPatchy

"""
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (Compose, Resize, ToTensor, Normalize)

from animl.utils.general import get_device, NUM_THREADS

from animl.reid import inference

ImageFile.LOAD_TRUNCATED_IMAGES = True


def viewpoint_estimator(model, batch, device=None):
    """
    Wrapper for viewpoint estimation within MatchyPatchy

    Args:
        model: PyTorch model for viewpoint estimation
        batch: a batch of images and their corresponding ROI IDs
        device: the device to run the model on (default is determined by get_device)

    Returns:
        roi_id: the ID of the region of interest
        value: the predicted viewpoint class
        prob: the probability of the predicted class
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

    Args:
        model: PyTorch model for MiewID embedding extraction
        batch: a batch of images and their corresponding ROI IDs
        device: the device to run the model on (default is determined by get_device)

    Returns:
        roi_id: the ID of the region of interest
        emb: the extracted embedding vector for the image
    """
    if device is None:
        device = get_device()

    # with torch.no_grad():
    img = batch[0]
    roi_id = batch[1].numpy()[0]
    emb = model.extract_feat(img.to(device))
    emb = emb.cpu().detach().numpy()[0]

    return roi_id, emb


def reid_dataloader(rois,
                    image_path_dict: dict,
                    resize_width: int = inference.MIEW_WIDTH,
                    resize_height: int = inference.MIEW_HEIGHT,
                    batch_size: int = 1,
                    num_workers: int = NUM_THREADS,
                    normalize: bool = True):
    '''
    Loads a dataset and wraps it in a PyTorch DataLoader object.
    Always dynamically crops

    Args:
        rois (DataFrame): data to be fed into the model
        image_path_dict (dict): map of roi_id to filepath
        resize_width (int): size in pixels for input width
        resize_height (int): size in pixels for input height
        batch_size (int): size of each batch
        num_workers (int): number of processes to handle the data
        normalize (bool): tensors are normalized by default, set to false to un-normalize

    Returns:
        dataloader object

    MIEWIDNET - 440, 440
    '''
    dataset_instance = MiewGenerator(rois, image_path_dict,
                                     resize_height=resize_height,
                                     resize_width=resize_width, normalize=normalize)

    dataLoader = DataLoader(dataset=dataset_instance,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    return dataLoader


class MiewGenerator(Dataset):
    '''
    Data generator that crops images on the fly, requires relative bbox coordinates,
    ie from MegaDetector

    Options:
        - resize_height: height of resized input image
        - resize_width: width of resized input image
        - normalize: normalize images to mean and std of MiewID
    '''
    def __init__(self, x, image_path_dict, resize_height=440, resize_width=440, normalize=True):
        self.x = x.reset_index()
        self.image_path_dict = image_path_dict
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.normalize = normalize
        if self.normalize is True:
            self.transform = Compose([Resize((self.resize_height, self.resize_width)),
                                      ToTensor(),
                                      Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]), ])
        else:
            self.transform = Compose([Resize((self.resize_height, self.resize_width)),
                                      ToTensor(), ])

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

        bbox_x = self.x['bbox_x'].iloc[idx]
        bbox_y = self.x['bbox_y'].iloc[idx]
        bbox_w = self.x['bbox_w'].iloc[idx]
        bbox_h = self.x['bbox_h'].iloc[idx]

        left = width * bbox_x
        top = height * bbox_y
        right = width * (bbox_x + bbox_w)
        bottom = height * (bbox_y + bbox_h)

        img = img.crop((left, top, right, bottom))
        img_tensor = self.transform(img)
        img.close()

        return img_tensor, id
