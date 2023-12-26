"""
Generator Module

This module provides data generators/loaders for Torch, Tensorflow, and for training
PyTorch models.

"""
from tensorflow.keras.utils import Sequence
import numpy as np
from PIL import Image, ImageOps, ImageFile
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (Compose, Resize, ToTensor, Normalize,
                                    RandomHorizontalFlip)
from .utils.torch_utils import _setup_size

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ResizeWithPadding(torch.nn.Module):
    """
        Pads a crop to given size
        If the image is torch Tensor, it is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
        If image size is smaller than output size along any edge, image is padded with 0 and
        then center cropped.
    """
    def __init__(self, expected_size):
        """
            Args
                - expected_size: size in px to pad up to if necessary
        """
        super().__init__()
        self.expected_size = _setup_size(expected_size)

    def forward(self, img):
        """
        Args:
            - img (PIL Image or Tensor): Image to be cropped.

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


class TorchGenerator(Dataset):
    def __init__(self, x, filecol='file', resize=456, buffer=0, batch=1):
        self.x = x
        self.filecol = filecol
        self.resize = resize
        self.buffer = buffer
        self.batch = int(batch)
        self.transform = Compose([
            Resize((self.resize, self.resize)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return int(np.ceil(len(self.x.index) / float(self.batch)))

    def __getitem__(self, idx):
        image_name = self.x[self.filecol].iloc[idx]

        try:
            img = Image.open(image_name).convert('RGB')
        except OSError:
            return

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

        return img_tensor, image_name


class TFGenerator(Sequence):
    def __init__(self, x, filecol='file', resize=299, buffer=0, batch=32):
        self.x = x
        self.filecol = filecol
        self.resize = int(resize)
        self.buffer = buffer
        self.batch = int(batch)

    def __len__(self):
        return int(np.ceil(len(self.x.index) / float(self.batch)))

    def __getitem__(self, idx):
        imgarray = []
        for i in range(min(len(self.x.index), idx * self.batch),
                       min(len(self.x.index), (idx + 1) * self.batch)):
            try:
                file = self.x[self.filecol].iloc[i]
                img = Image.open(file)
            except OSError:
                continue
            width, height = img.size

            bbox1 = self.x['bbox1'].iloc[i]
            bbox2 = self.x['bbox2'].iloc[i]
            bbox3 = self.x['bbox3'].iloc[i]
            bbox4 = self.x['bbox4'].iloc[i]

            left = width * bbox1
            top = height * bbox2
            right = width * (bbox1 + bbox3)
            bottom = height * (bbox2 + bbox4)

            left = max(0, left - self.buffer)
            top = max(0, top - self.buffer)
            right = min(width, right + self.buffer)
            bottom = min(height, bottom + self.buffer)

            img = img.crop((left, top, right, bottom))
            img = img.resize((self.resize, self.resize))
            img = tf.keras.utils.img_to_array(img)
            imgarray.append(img)

        return np.asarray(imgarray)


class TrainGenerator(Dataset):
    def __init__(self, x, categories, filecol='FilePath', resize=299, batch_size=32):
        self.x = x
        self.resize = resize
        self.filecol = filecol
        self.buffer = 0
        self.batch_size = int(batch_size)
        self.transform = Compose([
            Resize((self.resize, self.resize)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
        self.categories = categories

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image_name = self.x[self.filecol].iloc[idx]
        label = self.categories[self.x['label'].iloc[idx]]

        try:
            img = Image.open(image_name).convert('RGB')
        # delete entry in generator if file error, pull next image
        except OSError:
            print("File error", image_name)
            del self.x.iloc[idx]
            return self.__getitem__(idx)

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

        return img_tensor, label, image_name


def train_dataloader(manifest, categories, batch_size=1, workers=1, filecol="FilePath"):
    '''
        Loads a dataset for training and wraps it in a PyTorch DataLoader object.

        Args
            - manifest: dataframe to turn into data loader
            - categories: dictionary of class (label,idx) pairs
            - batch_size: number of images to process at once, defaults to 1
            - workers: number of threads to process on, defaults to 1
            - filecol: dataframe column name to pull file paths from

        Returns
            data_loader object
    '''
    dataset_instance = TrainGenerator(manifest, categories, filecol)

    data_loader = DataLoader(
            dataset=dataset_instance,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers
        )
    return data_loader


def create_dataloader(manifest, batch_size=1, workers=1, filecol="file"):
    '''
        Loads a dataset and wraps it in a PyTorch DataLoader object.

        Args
            - manifest: dataframe to turn into data loader
            - batch_size: number of images to process at once, defaults to 1
            - workers: number of threads to process on, defaults to 1
            - filecol: dataframe column name to pull file paths from

        Returns
            data_loader object
    '''
    dataset_instance = TorchGenerator(manifest, filecol, batch=batch_size)

    data_loader = DataLoader(
            dataset=dataset_instance,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers
        )
    return data_loader
