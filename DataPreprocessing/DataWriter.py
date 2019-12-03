import h5py
from DataPreprocessing.DataLoader import DataLoader
import numpy as np
import Constant
import os

def convert_meta_to_numpy(metas):
    length = [meta.length for meta in metas]
    labels = [np.array(meta.digit) for meta in metas]
    return np.array(length), np.array(labels)

def write_data(data_dir, write_location):
    data, meta = DataLoader.load_data(data_dir, cropped=True)
    hf_writer = h5py.File(write_location, 'w')
    hf_writer.create_dataset("cropped_digits", data=data)
    hf_writer.create_dataset("length", data=convert_meta_to_numpy(meta)[0])
    hf_writer.create_dataset("labels", data=convert_meta_to_numpy(meta)[1])


def write_train_and_val_data(train_val_split=0.1):
    train_data = Constant.train_data()
    # train_data = os.path.join(Constant.data_dir(), "example")
    data, meta = DataLoader.load_data(train_data, cropped=True)

    print(data)
    total_size = len(data)

    val_sample = np.array([np.random.rand() < train_val_split for _ in range(total_size)])
    train_sample = np.logical_not(val_sample)

    train_portion_data, train_portion_meta = data[train_sample], meta[train_sample]
    val_portion_data, val_portion_meta = data[val_sample], meta[val_sample]

    hf_train = h5py.File('../Data/train.h5', 'w')
    hf_val = h5py.File('../Data/val.h5', 'w')

    print (train_portion_data.shape)
    hf_train.create_dataset("cropped_digits", data=train_portion_data)
    hf_train.create_dataset("length", data=convert_meta_to_numpy(train_portion_meta)[0])
    hf_train.create_dataset("labels", data=convert_meta_to_numpy(train_portion_meta)[1])


    hf_val.create_dataset("cropped_digits", data=val_portion_data)
    hf_val.create_dataset("length", data=convert_meta_to_numpy(val_portion_meta)[0])
    hf_val.create_dataset("labels", data=convert_meta_to_numpy(val_portion_meta)[1])


def write_train_and_val_data_for_detection(train_val_split=0.1):
    train_data = Constant.train_data()
    # train_data = os.path.join(Constant.data_dir(), "example")
    digit_data, _ = DataLoader.load_data(train_data, cropped=True)
    non_digit_data = DataLoader.create_non_digit_data(train_data)

    digit_data_meta = np.ones((digit_data.shape[0],))
    non_digit_meta = np.zeros((non_digit_data.shape[0],))

    print (digit_data.shape)
    print(non_digit_data.shape)

    total_data = np.vstack((digit_data, non_digit_data))
    total_meta =np.append(digit_data_meta, non_digit_meta)
    total_size = total_data.shape[0]
    print (total_data.shape)
    val_sample = np.array([np.random.rand() < train_val_split for _ in range(total_size)])
    train_sample = np.logical_not(val_sample)

    train_portion_data, train_portion_meta = total_data[train_sample], total_meta[train_sample]
    val_portion_data, val_portion_meta = total_data[val_sample], total_meta[val_sample]

    hf_train = h5py.File('../Data/detection-train.h5', 'w')
    hf_val = h5py.File('../Data/detection-val.h5', 'w')

    hf_train.create_dataset("image", data=train_portion_data)
    hf_train.create_dataset("isdigit", data=train_portion_meta)


    hf_val.create_dataset("image", data=val_portion_data)
    hf_val.create_dataset("isdigit", data=val_portion_meta)

if __name__ == "__main__":
    write_train_and_val_data_for_detection()
    # write_train_and_val_data(train_val_split=0.1)
    # write_data(Constant.test_data(), '../Data/test.h5')


