from DataPreprocessing.DataLoader import DataLoader
from Models.VGG16 import VGG16
import torch
from DataPreprocessing.Processor import get_preprocessor

def train():
    batch_size = 32
    learning_rate = 0.01
    early_stopping_threshold = 50

    step = 0
    epoch = 0
    early_stopping = early_stopping_threshold

    model = VGG16(load_weights=True)
    model.cuda()

    transform = get_preprocessor()

    data_set = torch.
