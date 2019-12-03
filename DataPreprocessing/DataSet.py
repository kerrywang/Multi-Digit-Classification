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

        print ("getting index: {} digit: {} with length: {}".format(index, digits, length))
        length_template = np.zeros((7,), dtype=np.long)
        digit_template = np.zeros((11,), dtype=np.long)

        length_template[length] = 1.0
        new_digits = []
        for digit in digits:
            new_digit = digit_template.copy()
            new_digit[digit] = 1.0
            new_digits.append(new_digit)

        image = Image.fromarray(image)
        image = self.transform(image)

        return image, length_template, np.array(new_digits)


if __name__ == "__main__":
    dataset = DataSet("../Data/val.h5", get_preprocessor())

    img, length, digits = dataset[10]
    print(length)
    print(type(digits))
