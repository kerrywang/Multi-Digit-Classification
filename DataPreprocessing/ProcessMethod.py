import abc
import cv2
import numpy as np
from DataPreprocessing.DataLoader import BBox

class ProcessMethod(abc.ABC):
    def __init__(self):
        super(ProcessMethod, self).__init__()

    @abc.abstractmethod
    def Process(self, data):
        pass


class ImageResizerMethod(ProcessMethod):
    def __init__(self, resized_shape, inter=cv2.INTER_AREA):
        super(ImageResizerMethod, self).__init__()
        self.shape = resized_shape
        self.inter = inter

    def Process(self, data):
        return [cv2.resize(single_image, self.shape, interpolation=self.inter) for single_image in data]


class MeanExtractorMethod(ProcessMethod):
    def __init__(self):
        super(MeanExtractorMethod, self).__init__()

    def Process(self, data):
        return list(map(self.mean_extraction, data))

    def mean_extraction(self, image):
        return image / np.mean(image)

class CropMethod(ProcessMethod):
    def __init__(self, bbox=None):
        '''
        if bbox is None, it should do sliding window operation to find the bounding box
        :param bbox:
        '''
        super(CropMethod, self).__init__()
        self.bbox = bbox

    def Process(self, data):
        if not self.bbox:
            self.bbox = self.slidingWindowProcess(data)
        new_data = []
        for bbox, image in zip(self.bbox, data):
            new_data.append(self.crop(image, bbox))
        return new_data

    def crop(self, image: np.array, bbox: BBox)->np.array:
        return bbox.crop(image)


    def slidingWindowProcess(self, data):
        pass



