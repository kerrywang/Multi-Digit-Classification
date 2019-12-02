import numpy as np
import torch.utils.data as data
from PIL import Image

class DataSet(data.Dataset):
    def __init__(self, data_type, transform):
        self.transfrom = transform
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
