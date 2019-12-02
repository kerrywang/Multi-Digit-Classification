import abc
import cv2
import numpy as np
from DataPreprocessing.DataLoader import BBox
# from Detection.DetectionAlgo import SlidingWindow

class ProcessMethod(abc.ABC):
    def __init__(self):
        super(ProcessMethod, self).__init__()

    @abc.abstractmethod
    def Process(self, data):
        pass

class ConvertPILToCV2Method(ProcessMethod):
    def __init__(self):
        super(ConvertPILToCV2Method, self).__init__()

    def Process(self, data):
        # print("converting {}".format(data.shape))
        return np.transpose(data, (1, 2, 0))


class GrayMethod(ProcessMethod):
    def __init__(self):
        super(GrayMethod, self).__init__()

    def Process(self, data):
        return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in data]

class ImageResizerMethod(ProcessMethod):
    def __init__(self, resized_shape, inter=cv2.INTER_AREA):
        super(ImageResizerMethod, self).__init__()
        self.shape = resized_shape
        self.inter = inter

    def Process(self, data):
        return [cv2.resize(img, self.shape, interpolation=self.inter) for img in data]


class MeanExtractorMethod(ProcessMethod):
    def __init__(self):
        super(MeanExtractorMethod, self).__init__()

    def Process(self, data):
        return list(map(self.mean_extraction, data))

    def mean_extraction(self, image):
        return image / np.mean(image)

class CropMethod(ProcessMethod):
    def __init__(self, bbox=None, detection_algo=None):
        '''
        if bbox is None, it should do sliding window operation to find the bounding box
        :param bbox:
        '''
        super(CropMethod, self).__init__()
        self.bbox = bbox
        self.detection_algo = detection_algo

    def Process(self, data):
        if not self.bbox:
            self.bbox = self.detection_algo.Process(data)

        # self.crop(data, self.bbox)
        new_data = []
        for bbox, image in zip(self.bbox, data):
            new_data.append(self.crop(image, bbox))
        return new_data

    def crop(self, image: np.array, bbox: BBox)->np.array:
        return bbox.crop(image)




