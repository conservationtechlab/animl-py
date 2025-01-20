"""
Generators and Dataloaders

"""
from typing import Tuple, Dict
import pandas as pd
from PIL import Image, ImageOps, ImageFile
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (Compose, Resize, ToTensor, RandomHorizontalFlip,
                                    Normalize, RandomAffine, RandomGrayscale, RandomApply,
                                    ColorJitter, GaussianBlur)
from .utils.torch_utils import _setup_size

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ResizeWithPadding(torch.nn.Module):
    """Pads a crop to given size
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


class ImageGenerator(Dataset):
    '''
    Data generator that crops images on the fly, requires relative bbox coordinates,
    ie from MegaDetector

    Options:
        - file_col: column name containing full file paths
        - resize: dynamically resize images to target (square) [W,H]
        - crop: if true, dynamically crop
        - normalize: tensors are normalized by default, set to false to un-normalize
    '''
    def __init__(self, x: pd.DataFrame, file_col: str = "file",
                 resize_height: int = 299, resize_width: int = 299,
                 crop: bool = True, normalize: bool = True,) -> None:
        self.x = x
        self.file_col = file_col
        self.crop = crop
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.buffer = 0
        self.normalize = normalize
        self.transform = Compose([
            # torch.resize order is H,W
            Resize((self.resize_height, self.resize_width)),
            ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        image_name = self.x.loc[idx, self.file_col]

        try:
            img = Image.open(image_name).convert('RGB')
        except OSError:
            print("File error", image_name)
            del self.x.iloc[idx]
            return self.__getitem__(idx)

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


class MiewGenerator(Dataset):
    '''
    Data generator that crops images on the fly, requires relative bbox coordinates,
    ie from MegaDetector

    Options:
        - resize: dynamically resize images to target (square) [W,H]
    '''
    def __init__(self, x: pd.DataFrame, image_path_dict: Dict[str, str],
                 resize_height: int = 440, resize_width: int = 440):
        self.x = x.reset_index()
        self.image_path_dict = image_path_dict
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.transform = Compose([Resize((self.resize_height, self.resize_width)),
                                  ToTensor(),
                                  Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),])

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


# currently working on this class
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
    def __init__(self, x, classes, file_col='FilePath', label_col='species',
                 crop=True, resize_height=299, resize_width=299, augment=False):
        self.x = x
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.file_col = file_col
        self.label_col = label_col
        self.buffer = 0
        self.crop = crop
        self.augment = augment
        augmentations = Compose([
            # rotate ± 15 degrees and shear ± 7 degrees
            RandomAffine(degrees=15, shear=(-7, 7)),
            # convert images to grayscale with 20% probability
            RandomGrayscale(p=0.2),
            # apply gaussian blur with 30% probability
            RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            # adjust brightness and contrast for varying lighting conditions
            ColorJitter(brightness=0.2, contrast=0.2),
        ])
        if self.augment:
            print("Applying augmentations")
            self.transform = Compose([augmentations,  # augmentations
                                      RandomHorizontalFlip(p=0.5),  # random horizontal flip
                                      Resize((self.resize_height, self.resize_width)),
                                      ToTensor(),])
        else:
            self.transform = Compose([RandomHorizontalFlip(p=0.5),  # random horizontal flip
                                      Resize((self.resize_height, self.resize_width)),
                                      ToTensor(),])
        self.categories = dict([[c, idx] for idx, c in list(enumerate(classes))])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image_name = self.x.loc[idx, self.file_col]
        label = self.categories[self.x.loc[idx, self.label_col]]

        try:
            img = Image.open(image_name).convert('RGB')
        except OSError:
            print("File error", image_name)
            del self.x.iloc[idx]
            return self.__getitem__(idx)

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
        img.close()

        return img_tensor, label, image_name


def train_dataloader(manifest, classes, batch_size=1, workers=1, file_col="FilePath",
                     crop=False, resize_height=480, resize_width=480, augment=False):
    '''
        Loads a dataset for training and wraps it in a
        PyTorch DataLoader object. Shuffles the data before loading.

        Args:
            - manifest (DataFrame): data to be fed into the model
            - classes (dict): all possible class labels
            - batch_size (int): size of each batch
            - workers (int): number of processes to handle the data
            - file_col (str): column name containing full file paths
            - crop (bool): if true, dynamically crop images
            - resize_width (int): size in pixels for input width
            - resize_height (int): size in pixels for input height
            - augment (bool): flag to augment images within loader

        Returns:
            dataloader object
    '''
    dataset_instance = TrainGenerator(manifest, classes, file_col, crop=crop,
                                      resize_height=resize_height, resize_width=resize_width,
                                      augment=augment)

    dataLoader = DataLoader(dataset=dataset_instance,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=workers)
    return dataLoader


def manifest_dataloader(manifest, batch_size=1, workers=1, file_col="file",
                        crop=True, normalize=True, resize_width=299, resize_height=299):
    '''
        Loads a dataset and wraps it in a PyTorch DataLoader object.
        Always dynamically crops

        Args:
            - manifest (DataFrame): data to be fed into the model
            - batch_size (int): size of each batch
            - workers (int): number of processes to handle the data
            - file_col: column name containing full file paths
            - crop (bool): if true, dynamically crop images
            - normalize (bool): if true, normalize array to values [0,1]
            - resize_width (int): size in pixels for input width
            - resize_height (int): size in pixels for input height

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
                            shuffle=False,
                            num_workers=workers)
    return dataLoader


def reid_dataloader(rois, image_path_dict, resize_height, resize_width, batch_size=1, workers=1):
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


'''
TENSORFLOW LEGACY

def resize_with_padding(img, expected_size):
    """Pads a crop to given size
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and
    then center cropped.

    Args:
        - img (Img): a loaded Img object
        - expected_size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as
            (size[0], size[0]).
    Returns:
        - the padded Img object
    """
    if img.size[0] == 0 or img.size[1] == 0:
        return img
    if img.size[0] > img.size[1]:
        new_size = (expected_size[0],
                    int(expected_size[1] * img.size[1] / img.size[0]))
    else:
        new_size = (int(expected_size[0] * img.size[0] / img.size[1]),
                    expected_size[1])
    img = img.resize(new_size, Image.BILINEAR)  # NEAREST BILINEAR
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height,
               delta_width - pad_width,
               delta_height - pad_height)
    return ImageOps.expand(img, padding)


class LegacyGenerator(Dataset):

    def __init__(self, x, file_col='Frame', crop=False, resize=456, buffer=0):
        self.x = x
        self.file_col = file_col
        self.crop = crop
        self.resize = int(resize)
        self.buffer = buffer

    def __len__(self):
        return len(self.x.index)

    def __getitem__(self, idx):
        try:
            file = self.x[self.file_col].iloc[idx]
            #img = Image.open(file).convert('RGB')
            img = Image.open(file)
        except OSError:
            print("File error", file)
            del self.x.iloc[idx]
            return self.__getitem__(idx)

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

            left = max(0, left - self.buffer)
            top = max(0, top - self.buffer)
            right = min(width, right + self.buffer)
            bottom = min(height, bottom + self.buffer)
            img = img.crop((left, top, right, bottom))

        img = img.resize((self.resize, self.resize))
        img = np.asarray(img, dtype=np.float32)

        return img, file
'''
