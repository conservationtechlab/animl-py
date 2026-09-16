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
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import (Compose, Resize, ToImage, ToDtype, Pad, RandomHorizontalFlip,
                                       RandomAffine, RandomGrayscale, RandomApply,
                                       ColorJitter, GaussianBlur, Normalize)

from animl.model_architecture import SDZWA_CLASSIFIER_SIZE, BIOCLIP_CLASSIFIER_SIZE
from animl.reid.miewid import MIEWID_SIZE
from animl.file_management import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Letterbox(torch.nn.Module):
    """
    Pads a PIL image to a given size

    Compares input image dimensions with target aspect ratio.
    If input size is smaller than output size along any edge,
    the image is padded with color and then resized to the desired dimensions

    Args:
        resize_height (int): desired height of the output image
        resize_width (int): desired width of the output image
        color (int): desired color of padding
        interpolation_mode (torchvision.transforms.InterpolationMode): interpolation mode for adding padding
    """
    def __init__(self, resize_height, resize_width, color=0, interpolation_mode=InterpolationMode.BILINEAR):
        super().__init__()
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.color = color
        self.mode = interpolation_mode

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
                transform = Compose([Pad((0, hp, 0, hp), self.color, "constant"),
                                     Resize([self.resize_height, self.resize_width],
                                            interpolation=self.mode)])
                return transform(image)

            elif hp < 0 and wp > 0:
                wp = wp // 2
                transform = Compose([Pad((wp, 0, wp, 0), self.color, "constant"),
                                     Resize([self.resize_height, self.resize_width],
                                            interpolation=self.mode)])
                return transform(image)

        transform = Resize([self.resize_height, self.resize_width], interpolation=self.mode)

        return transform(image)


def image_to_tensor(file_path, resize_height, resize_width, letterbox):
    '''
    Convert an image to tensor for single detection or classification

    Args:
        file_path (str): path to image
        resize_height (int): resize height in pixels
        resize_width (int): resize width in pixels
        letterbox (bool): whether to use letterbox resizing

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

    tensor_transform = _get_model_transforms(resize_height, resize_width, letterbox=letterbox)

    img_tensor = tensor_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)  # add batch dimension
    img.close()
    frame = 0  # default frame 0 for images
    return img_tensor, [file_path], [frame], torch.tensor([(height, width)])


def _get_model_transforms(resize_height,
                          resize_width,
                          architecture: str = None,
                          letterbox: bool = False):
    """
    Generates the image preprocessing pipeline matching the model architecture.

    Args:
        resize_height (int): resize image height
        resize_width (int): resize image width
        architecture (str, optional): Model architecture name. If None, falls
            back to the default transform pipeline.
            ["bioclip_2", "miewid", "efficientnet_v2_m", "convnext_base", "pytorch", "onnx"]
        letterbox (bool): whether to use letterbox resizing for the default pipeline.

    Returns:
        torchvision.transforms.v2.Compose: Complete preprocessing pipeline.
    """
    if not isinstance(resize_height, int) or not isinstance(resize_width, int):
        raise TypeError("resize_height and resize_width must be integers")

    if not isinstance(letterbox, bool):
        raise TypeError("letterbox must be a boolean value")

    # BioClip model preprocessing pipeline
    if architecture == "bioclip_2":
        if resize_width != BIOCLIP_CLASSIFIER_SIZE or resize_height != BIOCLIP_CLASSIFIER_SIZE:
            resize_width, resize_height = (BIOCLIP_CLASSIFIER_SIZE, BIOCLIP_CLASSIFIER_SIZE)
            print(
                "[WARNING] Changing resize_width and resize_height to 224x224 "
                "to satisfy BioClip input requirements."
            )

        return Compose([Letterbox(resize_height=resize_height,
                                  resize_width=resize_width,
                                  color=127,
                                  interpolation_mode=InterpolationMode.BICUBIC),
                        ToImage(),
                        ToDtype(torch.float32, scale=True),
                        Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711]),])

    # MiewID model preprocessing pipeline
    if architecture == "miewid":
        if resize_width != MIEWID_SIZE or resize_height != MIEWID_SIZE:
            resize_width = MIEWID_SIZE
            resize_height = MIEWID_SIZE
            print(
                "[WARNING] Changing resize_width and resize_height to 128x128 "
                "to satisfy MiewID input requirements."
            )

        return Compose([Resize((resize_height, resize_width)),
                        ToImage(),
                        ToDtype(torch.float32, scale=True),
                        Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),])

    # everything else (default) preprocessing pipeline
    else:
        if letterbox:
            return Compose([Letterbox(resize_height, resize_width),
                            ToImage(),
                            ToDtype(torch.float32, scale=True),])

        # default transformations
        return Compose([Resize((resize_height, resize_width)),
                        ToImage(),
                        ToDtype(torch.float32, scale=True),])


def _get_augmentations() -> Compose:
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomAffine(degrees=15, shear=(-7, 7)),
        RandomGrayscale(p=0.2),
        RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        ColorJitter(brightness=0.2, contrast=0.2)
    ])


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
        letterbox: if true, will apply letterbox resizing
        transform: torchvision transforms to apply to images
    '''
    def __init__(self,
                 x: pd.DataFrame,
                 transform: Compose,
                 file_col: str = "filepath",
                 crop: bool = True,
                 crop_coord: str = 'relative',
                 normalize: bool = True) -> None:
        self.x = x.reset_index(drop=True)
        self.file_col = file_col
        if self.file_col not in self.x.columns:
            raise ValueError(f"file_col '{self.file_col}' not found in dataframe columns")
        self.crop = crop
        if not isinstance(self.crop, bool):
            raise TypeError("crop must be a boolean value")
        if self.crop and not {'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'}.issubset(self.x.columns):
            raise ValueError("No bbox columns found for cropping")
        self.crop_coord = crop_coord
        if self.crop_coord not in ['relative', 'absolute']:
            raise ValueError("crop_coord must be either 'relative' or 'absolute'")
        if 'frame' not in self.x.columns:
            self.x['frame'] = 0  # default frame 0 for images

        self.normalize = normalize
        if not isinstance(self.normalize, bool):
            raise TypeError("normalize must be a boolean value")
        # preprocessing transform for the images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str, int, Tensor]:
        try:
            file_row = self.x.iloc[idx]
            filepath = file_row[self.file_col]
            frame = file_row['frame']
            ext = Path(filepath).suffix.lower()

            if ext in VIDEO_EXTENSIONS:
                img = self.extract_frames(idx, filepath)
                if img is None:
                    return None, str(filepath), int(frame), None

            elif ext in IMAGE_EXTENSIONS:
                try:
                    img = Image.open(filepath).convert('RGB')
                except OSError:
                    print(f"Image {filepath} cannot be opened. Skipping.")
                    return None, str(filepath), int(frame), None

            else:
                print(f"File {filepath} is not a video or image. Skipping.")
                return None, str(filepath), int(frame), None

            width, height = img.size

            if self.crop:
                bbox_x = file_row['bbox_x']
                bbox_y = file_row['bbox_y']
                bbox_w = file_row['bbox_w']
                bbox_h = file_row['bbox_h']

                if self.crop_coord == 'relative':
                    left = width * bbox_x
                    top = height * bbox_y
                    right = width * (bbox_x + bbox_w)
                    bottom = height * (bbox_y + bbox_h)

                    left = max(0, int(left))
                    top = max(0, int(top))
                    right = min(width, int(right))
                    bottom = min(height, int(bottom))
                    img = img.crop((left, top, right, bottom))

                elif self.crop_coord == 'absolute':
                    left = bbox_x
                    top = bbox_y
                    right = bbox_x + bbox_w
                    bottom = bbox_y + bbox_h

                    left = max(0, int(left))
                    top = max(0, int(top))
                    right = min(width, int(right))
                    bottom = min(height, int(bottom))
                    img = img.crop((left, top, right, bottom))

            img_tensor = self.transform(img)
            img.close()

            if not self.normalize:  # un-normalize
                img_tensor = img_tensor * 255

            return img_tensor, str(filepath), int(frame), torch.tensor((height, width))

        except Exception as e:
            print(f"Error processing file {filepath}. Exception: {e}")
            return None, str(filepath), int(frame), None

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


def manifest_dataloader(manifest: pd.DataFrame,
                        file_col: str = "filepath",
                        crop: bool = True,
                        crop_coord: str = 'relative',
                        resize_height: int = SDZWA_CLASSIFIER_SIZE,
                        resize_width: int = SDZWA_CLASSIFIER_SIZE,
                        architecture: str = None,
                        letterbox: bool = False,
                        normalize: bool = True,
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
        resize_height (int): size in pixels for input height
        resize_width (int): size in pixels for input width
        architecture (str): model architecture to determine default transforms
        letterbox (bool): if true, maintain aspect ratio with padding
        normalize (bool): if true, normalize array to values [0,1]
        transform (Compose): additional transforms to apply to images beyond resize, etc
        batch_size (int): size of each batch
        num_workers (int): number of processes to handle the data

    Returns:
        dataloader object
    '''
    # get the default model transforms based on the architecture and resizing options
    final_transform = _get_model_transforms(resize_height,
                                            resize_width,
                                            architecture=architecture,
                                            letterbox=letterbox)
    if transform is not None:
        final_transform = Compose([final_transform, transform])

    dataset_instance = ManifestGenerator(manifest,
                                         final_transform,
                                         file_col=file_col,
                                         crop=crop,
                                         crop_coord=crop_coord,
                                         normalize=normalize)

    dataLoader = DataLoader(dataset=dataset_instance,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            collate_fn=collate_fn)
    return dataLoader


def collate_fn(batch):
    good = [x for x in batch if x[0] is not None]
    failed = [x[1] for x in batch if x[0] is None]  # just the filepath strings

    if len(good) == 0:
        return None, failed

    collated = torch.utils.data.dataloader.default_collate(good)
    return collated, failed


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
        - architecture: model architecture
    '''
    def __init__(self,
                 x: pd.DataFrame,
                 classes: dict,
                 transform: Compose,
                 file_col: str = 'filepath',
                 label_col: str = 'species',
                 crop: bool = True,
                 crop_coord: str = 'relative',
                 normalize: bool = True,
                 augment: bool = False,
                 cache_dir: str = None,):
        self.x = x.reset_index(drop=True)
        self.file_col = file_col
        if self.file_col not in self.x.columns:
            raise ValueError(f"file_col '{self.file_col}' not found in dataframe columns")
        self.label_col = label_col
        if self.label_col not in self.x.columns:
            raise ValueError(f"label_col '{self.label_col}' not found in dataframe columns")
        self.crop = crop
        if self.crop and not {'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'}.issubset(self.x.columns):
            raise ValueError("No bbox columns found for cropping")
        self.crop_coord = crop_coord
        if self.crop_coord not in ['relative', 'absolute']:
            raise ValueError("crop_coord must be either 'relative' or 'absolute'")
        # set normalization flag
        self.normalize = normalize
        if not isinstance(self.normalize, bool):
            raise TypeError("normalize must be a boolean value")

        # cache directory
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.transform = transform
        if augment:
            print("Applying augmentations")
            self.transform = Compose(_get_augmentations().transforms + self.transform.transforms)

        self.categories = {c: idx for idx, c in classes.items()}

    def __len__(self):
        return len(self.x)

    def _get_cache_path(self, img_row):
        if self.cache_dir is None:
            return None

        img_path = img_row[self.file_col]

        if self.crop:
            bbox_x = img_row['bbox_x']
            bbox_y = img_row['bbox_y']
            bbox_w = img_row['bbox_w']
            bbox_h = img_row['bbox_h']

            identifier = f"{img_path}_{bbox_x}_{bbox_y}_{bbox_w}_{bbox_h}"
        else:
            identifier = f"{img_path}"
        hash_id = hashlib.md5(identifier.encode()).hexdigest()
        return Path(self.cache_dir) / f"{hash_id}.jpg"

    def __getitem__(self, idx):
        try:
            img_row = self.x.iloc[idx]
            image_name = img_row[self.file_col]
            label = self.categories[img_row[self.label_col]]
            cache_path = self._get_cache_path(img_row)

            if cache_path is not None and Path(cache_path).exists():
                img = Image.open(cache_path).convert("RGB")
                img_tensor = self.transform(img)
                return img_tensor, label, image_name
            else:
                try:
                    img = Image.open(image_name).convert('RGB')
                except OSError:
                    print(f"Image {image_name} cannot be opened. Skipping.")
                    return None, label, str(image_name)

                if self.crop:
                    width, height = img.size

                    bbox_x = img_row['bbox_x']
                    bbox_y = img_row['bbox_y']
                    bbox_w = img_row['bbox_w']
                    bbox_h = img_row['bbox_h']

                    if self.crop_coord == 'relative':
                        left = width * bbox_x
                        top = height * bbox_y
                        right = width * (bbox_x + bbox_w)
                        bottom = height * (bbox_y + bbox_h)

                        left = max(0, int(left))
                        top = max(0, int(top))
                        right = min(width, int(right))
                        bottom = min(height, int(bottom))
                        img = img.crop((left, top, right, bottom))

                    elif self.crop_coord == 'absolute':
                        left = bbox_x
                        top = bbox_y
                        right = bbox_x + bbox_w
                        bottom = bbox_y + bbox_h

                        left = max(0, int(left))
                        top = max(0, int(top))
                        right = min(width, int(right))
                        bottom = min(height, int(bottom))
                        img = img.crop((left, top, right, bottom))

                img_tensor = self.transform(img)

                if not self.normalize:  # un-normalize
                    img_tensor = img_tensor * 255

                if self.cache_dir is not None:
                    img.save(cache_path, format="JPEG")
                img.close()

            return img_tensor, label, str(image_name)

        except Exception as e:
            print(f"Error processing file {image_name}. Exception: {e}")
            return None, label, str(image_name)


def train_dataloader(manifest: pd.DataFrame,
                     classes: dict,
                     file_col: str = "filepath",
                     label_col: str = "species",
                     crop: bool = False,
                     crop_coord: str = 'relative',
                     resize_height: int = SDZWA_CLASSIFIER_SIZE,
                     resize_width: int = SDZWA_CLASSIFIER_SIZE,
                     architecture: str = None,
                     letterbox: bool = False,
                     normalize: bool = True,
                     transform: Compose = None,
                     augment: bool = False,
                     batch_size: int = 1,
                     num_workers: int = 1,
                     cache_dir: str = None,):
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
        resize_height (int): size in pixels for input height
        resize_width (int): size in pixels for input width
        architecture (str): model architecture to determine default transforms
        letterbox (bool): if true, maintain aspect ratio with padding
        normalize (bool): if true, normalize array to values [0,1]
        transform (Compose): additional transformations to apply to the images
        augment (bool): if true, apply data augmentation
        batch_size (int): size of each batch
        num_workers (int): number of processes to handle the data
        cache_dir (str): if not None, use given cache directory

    Returns:
        dataloader object
    '''
    # get transforms for the model
    final_transform = _get_model_transforms(resize_height,
                                            resize_width,
                                            architecture=architecture,
                                            letterbox=letterbox)
    if transform is not None:
        final_transform = Compose([final_transform, transform])

    dataset_instance = TrainGenerator(manifest,
                                      classes,
                                      final_transform,
                                      file_col=file_col,
                                      label_col=label_col,
                                      crop=crop,
                                      crop_coord=crop_coord,
                                      normalize=normalize,
                                      augment=augment,
                                      cache_dir=cache_dir)

    dataLoader = DataLoader(dataset=dataset_instance,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            collate_fn=collate_fn)
    return dataLoader
