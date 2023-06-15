from tensorflow.keras.utils import Sequence
import numpy as np
from PIL import Image, ImageOps, ImageFile
import tensorflow as tf

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_with_padding(img, expected_size):
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
               delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


class GenerateCropsFromFile(Sequence):
    def __init__(self, x, filecol = 'file', resize=299, buffer=0, batch=32, standardize=True):
        self.x = x
        self.filecol = filecol
        self.resize = int(resize)
        self.buffer = buffer
        self.batch = int(batch)
        self.standardize = standardize

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