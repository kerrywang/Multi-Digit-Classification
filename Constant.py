import os


def data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")


def train_data():
    return os.path.join(data_dir(), "train")


def test_data():
    return os.path.join(data_dir(), "test")

