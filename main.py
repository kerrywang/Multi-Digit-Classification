# from DataPreprocessing.DataLoader import Data
from DataPreprocessing.DataLoader import DataLoader
from DataPreprocessing.Processor import Preprocessor
from DataPreprocessing.ProcessMethod import *
import cv2
import os
import Constant
import h5py
import time

def Main():

    train_data, train_metadata = DataLoader.load_data(Constant.train_data())

    train_processor = Preprocessor()
    train_crop_processor = CropMethod([metadata.bbox for metadata in train_metadata])



    # train_path = Constant.train_data()
    # test_path = Constant.test_data()
    #
    # train_meta = os.path.join(train_path, "digitStruct.mat")
    # test_mata = os.path.join(test_path, "digitStruct.mat")
    # for index in range(1, 12):
    #     train_data = [cv2.imread(os.path.join(train_path, "{}.png".format(str(index))))]
    #     test_data = [cv2.imread(os.path.join(test_path, "{}.png".format(str(index))))]
    #     # cv2.imshow("origin", train_data[0])
    #     # cv2.waitKey(0)
    #     with h5py.File(train_meta, 'r') as digit_struct_mat:
    #         train_metadata = [DataLoader.get_attribute(digit_struct_mat, index - 1)]
    #
    #     with h5py.File(test_mata, 'r') as digit_struct_mat:
    #         test_metada = [DataLoader.get_attribute(digit_struct_mat, index - 1)]
    #
    #
    #     train_processor = Preprocessor()
    #     test_processor = Preprocessor()
    #
    #     mean_extractor = MeanExtractorMethod()
    #     resize_processor = ImageResizerMethod((64, 64))
    #
    #     train_crop_processor = CropMethod([metadata.bbox for metadata in train_metadata])
    #     test_crop_processor = CropMethod([metadata.bbox for metadata in test_metada])
    #
    #     train_processor.register_processor([train_crop_processor, resize_processor])
    #     test_processor.register_processor([mean_extractor, test_crop_processor, resize_processor])
    #
    #     train_data = train_processor.Process(train_data)
    #     test_data = test_processor.Process(test_data)
    #     classification = test_data[0]
    #     cv2.imshow("{}.png".format(str(classification)), test_data[0])
    #     cv2.waitKey(0)


if __name__ == "__main__":
    Main()