"""
Generators and Dataloaders

Custom generators for training and inference

"""
import cv2
import hashlib
from typing import Tuple
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import (Compose, Resize, ToImage, ToDtype, Pad, RandomHorizontalFlip,
                                       RandomAffine, RandomGrayscale, RandomApply,
                                       ColorJitter, GaussianBlur)


from animl.model_architecture import SDZWA_CLASSIFIER_SIZE
from animl.file_management import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from animl.utils import get_device

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Letterbox(torch.nn.Module):
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
    def __init__(self, resize_height, resize_width):
        super().__init__()
        self.resize_height = resize_height
        self.resize_width = resize_width

    def forward(self, image):

        width, height = image.size  # PIL image size (width, height)
        ratio_f = self.resize_width / self.resize_height
        ratio_1 = width / height

        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            hp = int(width/ratio_f - height)
            wp = int(ratio_f * height - width)
            if hp > 0 and wp < 0:
                hp = hp // 2
                transform = Compose([Pad((0, hp, 0, hp), 0, "constant"),
                                     Resize([self.resize_height, self.resize_width])])
                return transform(image)

            elif hp < 0 and wp > 0:
                wp = wp // 2
                transform = Compose([Pad((wp, 0, wp, 0), 0, "constant"),
                                     Resize([self.resize_height, self.resize_width])])
                return transform(image)

        else:
            transform = Resize([self.resize_height, self.resize_width])
            return transform(image)


def image_to_tensor(file_path, letterbox, resize_width, resize_height):
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
        print(f'Image {file_path} cannot be loaded. Exception: {e}')
        return None

    width, height = img.size

    if letterbox:
        tensor_transform = Compose([Letterbox(resize_height, resize_width),
                                    ToImage(),
                                    ToDtype(torch.float32, scale=True),])  # torch.resize order is H,W
    else:
        tensor_transform = Compose([Resize((resize_height, resize_width)),
                                    ToImage(),
                                    ToDtype(torch.float32, scale=True),])

    img_tensor = tensor_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)  # add batch dimension
    img.close()
    return img_tensor, [file_path], torch.tensor([(height, width)])


class ManifestGenerator(Dataset):
    '''
    Data generator that crops images on the fly, requires relative bbox coordinates,
    ie from MegaDetector

    Options:
        file_col: column name containing full file paths
        resize_height: size in pixels for input height
        resize_width: size in pixels for input width
        crop: if true, dynamically crop
        crop_coord: if relative, will calculate absolute values based on image size
        normalize: tensors are normalized by default, set to false to un-normalize
        transform: torchvision transforms to apply to images
    '''
    def __init__(self, x: pd.DataFrame,
                 file_col: str = "filepath",
                 resize_height: int = SDZWA_CLASSIFIER_SIZE,
                 resize_width: int = SDZWA_CLASSIFIER_SIZE,
                 crop: bool = True,
                 crop_coord: str = 'relative',
                 normalize: bool = True,
                 letterbox: bool = False,
                 transform: Compose = None) -> None:
        self.x = x.reset_index(drop=True)
        self.file_col = file_col
        if self.file_col not in self.x.columns:
            raise ValueError(f"file_col '{self.file_col}' not found in dataframe columns")
        self.crop = crop
        if self.crop and not {'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'}.issubset(self.x.columns):
            raise ValueError("No bbox columns found for cropping")
        self.crop_coord = crop_coord
        if self.crop_coord not in ['relative', 'absolute']:
            raise ValueError("crop_coord must be either 'relative' or 'absolute'")
        if 'frame' not in self.x.columns:
            self.x['frame'] = 0  # default frame 0 for images

        self.resize_height = resize_height
        self.resize_width = resize_width
        self.buffer = 0
        self.normalize = normalize
        self.letterbox = letterbox

        # letterbox and resize
        if self.letterbox:
            if transform is None:
                self.transform = Compose([Letterbox(self.resize_height, self.resize_width),
                                         ToImage(),
                                         ToDtype(torch.float32, scale=True),])
            else:
                self.transform = Compose([Letterbox(self.resize_height, self.resize_width),
                                          ToImage(),
                                          ToDtype(torch.float32, scale=True),
                                          transform])
        # simply resize - torch.resize order is H,W
        else:
            if transform is None:
                self.transform = Compose([Resize((self.resize_height, self.resize_width)),
                                          ToImage(),
                                          ToDtype(torch.float32, scale=True),])
            else:
                self.transform = Compose([Resize((self.resize_height, self.resize_width)),
                                          ToImage(),
                                          ToDtype(torch.float32, scale=True),
                                          transform,])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str, int, Tensor]:
        filepath = self.x.loc[idx, self.file_col]
        frame = self.x.loc[idx, 'frame']
        ext = Path(filepath).suffix.lower()

        if ext in VIDEO_EXTENSIONS:
            img = self.extract_frames(idx, filepath)
            if img is None:
                return None

        elif ext in IMAGE_EXTENSIONS:
            try:
                img = Image.open(filepath).convert('RGB')
            except OSError:
                print(f"Image {filepath} cannot be opened. Skipping.")
                return None

        else:
            print(f"File {filepath} is not a video or image. Skipping.")
            return None

        width, height = img.size

        # maintain aspect ratio if one dimension is zero
        if self.resize_width > 0 and self.resize_height <= 0:
            self.height = int(width / height * self.resize_width)
        elif self.resize_width <= 0 and self.resize_height > 0:
            self.width = int(height / width * self.height)

        if self.crop:
            bbox_x = self.x['bbox_x'].iloc[idx]
            bbox_y = self.x['bbox_y'].iloc[idx]
            bbox_w = self.x['bbox_w'].iloc[idx]
            bbox_h = self.x['bbox_h'].iloc[idx]

            if self.crop_coord == 'relative':
                left = width * bbox_x
                top = height * bbox_y
                right = width * (bbox_x + bbox_w)
                bottom = height * (bbox_y + bbox_h)

                left = max(0, int(left) - self.buffer)
                top = max(0, int(top) - self.buffer)
                right = min(width, int(right) + self.buffer)
                bottom = min(height, int(bottom) + self.buffer)
                img = img.crop((left, top, right, bottom))

            elif self.crop_coord == 'absolute':
                left = bbox_x
                top = bbox_y
                right = bbox_x + bbox_w
                bottom = bbox_y + bbox_h

                left = max(0, int(left) - self.buffer)
                top = max(0, int(top) - self.buffer)
                right = min(width, int(right) + self.buffer)
                bottom = min(height, int(bottom) + self.buffer)
                img = img.crop((left, top, right, bottom))

        img_tensor = self.transform(img)
        img.close()

        if not self.normalize:  # un-normalize
            img_tensor = img_tensor * 255

        return img_tensor, str(filepath), int(frame), torch.tensor((height, width))

    def extract_frames(self, idx, filepath):
        frame = self.x.loc[idx, 'frame']

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():  # corrupted video
            print(f"Video {filepath} cannot be opened. Skipping.")
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame} in video {filepath} cannot be read. Skipping.")
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        cap.release()
        cv2.destroyAllWindows()
        return img


class TrainGenerator(Dataset):
    '''
    Data generator for training. Requires a list of possible classes

    Options:
        - file_col: column name containing full file paths
        - label_col: column name containing class labels
        - crop: if true, dynamically crop
        - crop_coord: if relative, will calculate absolute values
        - augment: add image augmentations at each batch
        - resize_height: size in pixels for input height
        - resize_width: size in pixels for input width
        - cache_dir: if not None, use given cache directory to store preprocessed images
    '''
    def __init__(self, x: pd.DataFrame,
                 classes: dict,
                 file_col: str = 'filepath',
                 label_col: str = 'species',
                 resize_height: int = SDZWA_CLASSIFIER_SIZE,
                 resize_width: int = SDZWA_CLASSIFIER_SIZE,
                 crop: bool = True,
                 crop_coord: str = 'relative',
                 augment: bool = False,
                 cache_dir: str = None):
        self.x = x
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.file_col = file_col
        if self.file_col not in self.x.columns:
            raise ValueError(f"file_col '{self.file_col}' not found in dataframe columns")
        self.label_col = label_col
        if self.label_col not in self.x.columns:
            raise ValueError(f"label_col '{self.label_col}' not found in dataframe columns")
        self.buffer = 0
        self.crop = crop
        if self.crop and not {'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'}.issubset(self.x.columns):
            raise ValueError("No bbox columns found for cropping")
        self.crop_coord = crop_coord
        if self.crop_coord not in ['relative', 'absolute']:
            raise ValueError("crop_coord must be either 'relative' or 'absolute'")
        # cache directory
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        # augmentations
        self.augment = augment
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
                                      ToImage(),
                                      ToDtype(torch.float32, scale=True),])
        else:
            self.transform = Compose([Resize((self.resize_height, self.resize_width)),
                                      ToImage(),
                                      ToDtype(torch.float32, scale=True),])
        self.categories = dict([[c, idx] for idx, c in list(enumerate(classes))])

    def __len__(self):
        return len(self.x)

    def _get_cache_path(self, img_path):
        if self.cache_dir is None:
            return None

        if self.crop:
            identifier = f"{img_path}_{self.x['bbox_x']}_{self.x['bbox_y']}_{self.x['bbox_w']}_{self.x['bbox_h']}"
        else:
            identifier = f"{img_path}"
        hash_id = hashlib.md5(identifier.encode()).hexdigest()
        return Path(self.cache_dir) / f"{hash_id}.jpg"

    def __getitem__(self, idx):
        image_name = self.x.loc[idx, self.file_col]
        label = self.categories[self.x.loc[idx, self.label_col]]
        cache_path = self._get_cache_path(image_name)

        if cache_path is not None and Path(cache_path).exists():
            img = Image.open(cache_path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, label, image_name
        else:
            try:
                img = Image.open(image_name).convert('RGB')
            except OSError:
                print(f"Image {image_name} cannot be opened. Skipping.")
                return None

            if self.crop:
                width, height = img.size

                bbox_x = self.x['bbox_x'].iloc[idx]
                bbox_y = self.x['bbox_y'].iloc[idx]
                bbox_w = self.x['bbox_w'].iloc[idx]
                bbox_h = self.x['bbox_h'].iloc[idx]

                if self.crop_coord == 'relative':
                    left = width * bbox_x
                    top = height * bbox_y
                    right = width * (bbox_x + bbox_w)
                    bottom = height * (bbox_y + bbox_h)

                    left = max(0, int(left) - self.buffer)
                    top = max(0, int(top) - self.buffer)
                    right = min(width, int(right) + self.buffer)
                    bottom = min(height, int(bottom) + self.buffer)
                    img = img.crop((left, top, right, bottom))
                elif self.crop_coord == 'absolute':
                    left = bbox_x
                    top = bbox_y
                    right = bbox_x + bbox_w
                    bottom = bbox_y + bbox_h

                    left = max(0, int(left) - self.buffer)
                    top = max(0, int(top) - self.buffer)
                    right = min(width, int(right) + self.buffer)
                    bottom = min(height, int(bottom) + self.buffer)
                    img = img.crop((left, top, right, bottom))

            img_tensor = self.transform(img)
            if self.cache_dir is not None:
                img.save(cache_path, format="JPEG")
            img.close()

        return img_tensor, label, str(image_name)


def train_dataloader(manifest: pd.DataFrame,
                     classes: dict,
                     file_col: str = "filepath",
                     label_col: str = "species",
                     resize_height: int = SDZWA_CLASSIFIER_SIZE,
                     resize_width: int = SDZWA_CLASSIFIER_SIZE,
                     crop: bool = False,
                     crop_coord: str = 'relative',
                     augment: bool = False,
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
        label_col (str): column name containing class labels
        crop (bool): if true, dynamically crop images
        crop_coord (str): if relative, will calculate absolute values based on image size
        augment (bool): flag to augment images within loader
        resize_height (int): size in pixels for input height
        resize_width (int): size in pixels for input width
        batch_size (int): size of each batch
        num_workers (int): number of processes to handle the data
        cache_dir (str): if not None, use given cache directory

    Returns:
        dataloader object
    '''
    dataset_instance = TrainGenerator(manifest, classes, file_col, label_col=label_col,
                                      crop=crop, crop_coord=crop_coord,
                                      resize_height=resize_height, resize_width=resize_width,
                                      augment=augment, cache_dir=cache_dir)

    device = get_device(quiet=True)
    pin_memory = False if device == 'cpu' else True

    dataLoader = DataLoader(dataset=dataset_instance,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=pin_memory,
                            collate_fn=collate_fn)
    return dataLoader


def manifest_dataloader(manifest: pd.DataFrame,
                        file_col: str = "filepath",
                        crop: bool = True,
                        crop_coord: str = 'relative',
                        normalize: bool = True,
                        letterbox: bool = False,
                        resize_height: int = SDZWA_CLASSIFIER_SIZE,
                        resize_width: int = SDZWA_CLASSIFIER_SIZE,
                        transform: Compose = None,
                        batch_size: int = 1,
                        num_workers: int = 1) -> DataLoader:
    '''
    Loads a dataset and wraps it in a PyTorch DataLoader object.

    Always dynamically crops

    Args:
        manifest (DataFrame): data to be fed into the model
        file_col: column name containing full file paths
        crop (bool): if true, dynamically crop images
        crop_coord (str): if relative, will calculate absolute values based on image size
        normalize (bool): if true, normalize array to values [0,1]
        resize_height (int): size in pixels for input height
        resize_width (int): size in pixels for input width
        tranform (Compose): torchvision transforms to apply to images
        batch_size (int): size of each batch
        num_workers (int): number of processes to handle the data
        cache_dir (str): if not None, use given cache directory

    Returns:
        dataloader object
    '''
    if crop is True and not any(manifest.columns.isin(["bbox_x"])):
        crop = False

    dataset_instance = ManifestGenerator(manifest, file_col=file_col, crop=crop,
                                         crop_coord=crop_coord, normalize=normalize, letterbox=letterbox,
                                         resize_width=resize_width, resize_height=resize_height, transform=transform)
    # set pin_memory based on device
    device = get_device(quiet=True)
    pin_memory = False if device == 'cpu' else True
    dataLoader = DataLoader(dataset=dataset_instance,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=pin_memory,
                            collate_fn=collate_fn)
    return dataLoader


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
