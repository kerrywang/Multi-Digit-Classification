import numpy as np
import torch.utils.data as data
from DataPreprocessing.Processor import get_preprocessor
from PIL import Image
import h5py
import cv2

class DataSet(data.Dataset):
    def __init__(self, data_path, transform):
        self.transform = transform
        self.data = h5py.File(data_path, 'r')

    def __len__(self):
        return self.data["labels"].shape[0]

    def __getitem__(self, index):
        image = self.data["cropped_digits"][index]
        digits = self.data["labels"][index]
        length = self.data["length"][index]

        image = Image.fromarray(image)
        image = self.transform(image)

        return image, length, np.array(digits)


if __name__ == "__main__":
    dataset = DataSet("../Data/val.h5", get_preprocessor())

    img, length, digits = dataset[10]
    print(length)
    print(type(digits))
