"""
Generators and Dataloaders

Custom generators for training and inference

@ Kyra Swanson 2023
"""
import hashlib
import os
from typing import Tuple
import pandas as pd
from PIL import Image, ImageOps, ImageFile

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (Compose, Resize, ToTensor, RandomHorizontalFlip,
                                    RandomAffine, RandomGrayscale, RandomApply,
                                    ColorJitter, GaussianBlur)

from animl.utils.general import _setup_size


ImageFile.LOAD_TRUNCATED_IMAGES = True


# TODO: letterboxing for MD
class ResizeWithPadding(torch.nn.Module):
    """
    Pads a crop to given size

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and
    then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as
            (size[0], size[0]).
    """
    def __init__(self, expected_size: Tuple[int, int]) -> None:
        super().__init__()
        self.expected_size = _setup_size(expected_size)

    def forward(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if img.size[0] == 0 or img.size[1] == 0:
            return img
        if img.size[0] > img.size[1]:
            new_size = (self.expected_size[0],
                        int(self.expected_size[1] * img.size[1] / img.size[0]))
        else:
            new_size = (int(self.expected_size[0] * img.size[0] / img.size[1]),
                        self.expected_size[1])
        img = img.resize(new_size, Image.BILINEAR)  # NEAREST BILINEAR
        delta_width = self.expected_size[0] - img.size[0]
        delta_height = self.expected_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height,
                   delta_width - pad_width,
                   delta_height - pad_height)
        return ImageOps.expand(img, padding)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


def image_to_tensor(file_path, resize_width, resize_height):
    '''
    Convert an image to tensor for single detection or classification

    Args:
        file_path (str): path to image
        resize_width (int): resize width in pixels
        resize_height (int): resize height in pixels

    Returns:
        a torch tensor representation of the image
    '''
    try:
        img = Image.open(file_path).convert(mode='RGB')
        img.load()
    except Exception as e:
        print('Image {} cannot be loaded. Exception: {}'.format(file_path, e))
        return None

    tensor_transform = Compose([Resize((resize_height, resize_width)), ToTensor(), ])  # torch.resize order is H,W
    img_tensor = tensor_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)  # add batch dimension
    img.close()
    return img_tensor


class ImageGenerator(Dataset):
    '''
    Data generator that crops images on the fly, requires relative bbox coordinates,
    ie from MegaDetector

    Options:
        file_col: column name containing full file paths
        resize: dynamically resize images to target
        crop: if true, dynamically crop
        normalize: tensors are normalized by default, set to false to un-normalize
    '''
    # TODO: set defaults to 480 after retraining models
    def __init__(self, x: pd.DataFrame,
                 file_col: str = "file",
                 resize_height: int = 299,
                 resize_width: int = 299,
                 crop: bool = True,
                 normalize: bool = True,) -> None:
        self.x = x.reset_index(drop=True)
        self.file_col = file_col
        self.crop = crop
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.buffer = 0
        self.normalize = normalize
        # torch.resize order is H,W
        self.transform = Compose([Resize((self.resize_height, self.resize_width)),
                                  ToTensor(), ])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        image_name = self.x.loc[idx, self.file_col]

        try:
            img = Image.open(image_name).convert('RGB')
        except OSError:
            print("File error", image_name)
            return None

        width, height = img.size

        if self.crop:
            bbox1 = self.x['bbox1'].iloc[idx]
            bbox2 = self.x['bbox2'].iloc[idx]
            bbox3 = self.x['bbox3'].iloc[idx]
            bbox4 = self.x['bbox4'].iloc[idx]

            left = width * bbox1
            top = height * bbox2
            right = width * (bbox1 + bbox3)
            bottom = height * (bbox2 + bbox4)

            left = max(0, int(left) - self.buffer)
            top = max(0, int(top) - self.buffer)
            right = min(width, int(right) + self.buffer)
            bottom = min(height, int(bottom) + self.buffer)
            img = img.crop((left, top, right, bottom))

        img_tensor = self.transform(img)
        img.close()

        if not self.normalize:  # un-normalize
            img_tensor = img_tensor * 255

        return img_tensor, image_name


class TrainGenerator(Dataset):
    '''
    Data generator for training. Requires a list of possible classes

    Options:
        - file_col: column name containing full file paths
        - label_col: column name containing class labels
        - crop: if true, dynamically crop
        - resize: dynamically resize images to target (square)
        - agument: add image augmentations at each batch
    '''
    def __init__(self, x: pd.DataFrame,
                 classes: dict,
                 file_col: str = 'FilePath',
                 label_col: str = 'species',
                 crop: bool = True,
                 augment: bool = False,
                 resize_height: int = 299,
                 resize_width: int = 299,
                 cache_dir: str = None):
        self.x = x
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.file_col = file_col
        self.label_col = label_col
        self.buffer = 0
        self.crop = crop
        self.augment = augment
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        augmentations = Compose([
                                # random horizontal flip
                                RandomHorizontalFlip(p=0.5),
                                # rotate ± 15 degrees and shear ± 7 degrees
                                RandomAffine(degrees=15, shear=(-7, 7)),
                                # convert images to grayscale with 20% probability
                                RandomGrayscale(p=0.2),
                                # apply gaussian blur with 30% probability
                                RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
                                # adjust brightness and contrast for varying lighting conditions
                                ColorJitter(brightness=0.2, contrast=0.2)
                                ])
        if self.augment:
            print("Applying augmentations")
            self.transform = Compose([augmentations,  # augmentations
                                      Resize((self.resize_height, self.resize_width)),
                                      ToTensor(), ])
        else:
            self.transform = Compose([Resize((self.resize_height, self.resize_width)),
                                      ToTensor(), ])
        self.categories = dict([[c, idx] for idx, c in list(enumerate(classes))])

    def __len__(self):
        return len(self.x)

    def _get_cache_path(self, img_path):
        if self.cache_dir is None:
            return ""

        if self.crop:
            identifier = f"{img_path}_{self.x['bbox1']}_{self.x['bbox2']}_{self.x['bbox3']}_{self.x['bbox4']}"
        else:
            identifier = f"{img_path}"
        hash_id = hashlib.md5(identifier.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_id}.jpg")

    def __getitem__(self, idx):
        image_name = self.x.loc[idx, self.file_col]
        label = self.categories[self.x.loc[idx, self.label_col]]
        cache_path = self._get_cache_path(image_name)

        if os.path.exists(cache_path):
            img = Image.open(cache_path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, label, image_name
        else:
            try:
                img = Image.open(image_name).convert('RGB')
            except OSError:
                print("File error", image_name)
                return None

            if self.crop:
                width, height = img.size

                bbox1 = self.x['bbox1'].iloc[idx]
                bbox2 = self.x['bbox2'].iloc[idx]
                bbox3 = self.x['bbox3'].iloc[idx]
                bbox4 = self.x['bbox4'].iloc[idx]

                left = width * bbox1
                top = height * bbox2
                right = width * (bbox1 + bbox3)
                bottom = height * (bbox2 + bbox4)

                left = max(0, int(left) - self.buffer)
                top = max(0, int(top) - self.buffer)
                right = min(width, int(right) + self.buffer)
                bottom = min(height, int(bottom) + self.buffer)
                img = img.crop((left, top, right, bottom))

            img_tensor = self.transform(img)
            if self.cache_dir is not None:
                img.save(cache_path, format="JPEG")
            img.close()

        return img_tensor, label, image_name


def train_dataloader(manifest: pd.DataFrame,
                     classes: dict,
                     file_col: str = "FilePath",
                     label_col: str = "species",
                     crop: bool = False,
                     augment: bool = False,
                     resize_height: int = 480,
                     resize_width: int = 480,
                     batch_size: int = 1,
                     num_workers: int = 1,
                     cache_dir: str = None):
    '''
    Loads a dataset for training and wraps it in a PyTorch DataLoader object.

    Shuffles the data before loading.

    Args:
        manifest (DataFrame): data to be fed into the model
        classes (dict): all possible class labels
        file_col (str): column name containing full file paths
        crop (bool): if true, dynamically crop images
        augment (bool): flag to augment images within loader
        resize_height (int): size in pixels for input height
        resize_width (int): size in pixels for input width
        batch_size (int): size of each batch
        num_workers (int): number of processes to handle the data
        cache_dir (str): if not None, use given cache directory

    Returns:
        dataloader object
    '''
    dataset_instance = TrainGenerator(manifest, classes, file_col, label_col=label_col, crop=crop,
                                      resize_height=resize_height, resize_width=resize_width,
                                      augment=augment, cache_dir=cache_dir)

    dataLoader = DataLoader(dataset=dataset_instance,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    return dataLoader


def manifest_dataloader(manifest: pd.DataFrame,
                        file_col: str = "file",
                        crop: bool = True,
                        normalize: bool = True,
                        resize_height: int = 480,
                        resize_width: int = 480,
                        batch_size: int = 1,
                        num_workers: int = 1):
    '''
    Loads a dataset and wraps it in a PyTorch DataLoader object.

    Always dynamically crops

    Args:
        manifest (DataFrame): data to be fed into the model
        file_col: column name containing full file paths
        crop (bool): if true, dynamically crop images
        normalize (bool): if true, normalize array to values [0,1]
        resize_height (int): size in pixels for input height
        resize_width (int): size in pixels for input width
        batch_size (int): size of each batch
        num_workers (int): number of processes to handle the data
        cache_dir (str): if not None, use given cache directory

    Returns:
        dataloader object
    '''
    if crop is True and not any(manifest.columns.isin(["bbox1"])):
        crop = False

    # default values file_col='file', resize=299
    dataset_instance = ImageGenerator(manifest, file_col=file_col, crop=crop, normalize=normalize,
                                      resize_width=resize_width, resize_height=resize_height)

    dataLoader = DataLoader(dataset=dataset_instance,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    return dataLoader

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
