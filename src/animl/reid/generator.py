"""
Image Generators

"""

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (Compose, Resize, Normalize, ToTensor)

class ImageGenerator(Dataset):
    '''
    Data generator that crops images on the fly, requires relative bbox coordinates,
    ie from MegaDetector

    Options:
        - resize: dynamically resize images to target (square) [W,H]
    '''
    def __init__(self, x, image_path_dict, resize_height=440, resize_width=440):
        self.x = x
        self.image_path_dict = image_path_dict
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.transform = Compose([Resize((self.resize_height, self.resize_width)),
                                  ToTensor(),
                                  Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        id = self.x.loc[idx, 'id']
        media_id = self.x.loc[idx, 'media_id']
        image_name = self.image_path_dict[media_id]
        print(image_name)

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


def dataloader(rois, image_path_dict, resize_height, resize_width, batch_size=1, workers=1):
    '''
        Loads a dataset and wraps it in a PyTorch DataLoader object.
        Always dynamically crops

        Args:
            - manifest (DataFrame): data to be fed into the model
            - batch_size (int): size of each batch
            - workers (int): number of processes to handle the data
            - resize_width (int): size in pixels for input width
            - resize_height (int): size in pixels for input height

        Returns:
            dataloader object


        MIEWIDNET - 440, 440
    '''
    dataset_instance = ImageGenerator(rois, image_path_dict, 
                                      resize_height=resize_height, 
                                      resize_width=resize_width)

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers
        )
    return dataLoader