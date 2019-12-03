from zipfile import ZipFile
import Constant as constant

import numpy as np
import os
import cv2
import h5py

class MetaData:
    def __init__(self, length, digit, bbox):
        self.length = length
        self.digit = digit
        self.bbox = bbox

    def __str__(self):
        res = ""
        print("Meta data: length: {}".format(self.length))
        for i in range(self.length):
            res += str(self.digit[i])
        return res


def build_bounding_box(attrs):
    attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x],
                                                           [attrs['left'], attrs['top'], attrs['width'],
                                                            attrs['height']])
    min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                min(attrs_top),
                                                max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                max(map(lambda x, y: x + y, attrs_top, attrs_height)))
    center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                    (min_top + max_bottom) / 2.0,
                                    max(max_right - min_left, max_bottom - min_top))
    return BBox(center_x - max_side / 2.0,
            center_y - max_side / 2.0,
            max_side,
            max_side)

class BBox(object):
    def __init__(self, left, top, width, height):
        self.left, self.top, self.width, self.height = left, top, width, height

    def width(self):
        return self.width

    def height(self):
        return self.height

    def top_left(self):
        return self.left, self.top

    def bot_right(self):
        return self.left + self.width, self.top + self.height

    def get_area(self):
        return self.width * self.height

    def crop(self, image):
        x_start, y_start, width, height = (max(0, int(round(self.left - 0.15 * self.width))),
                                           max(0, int(round(self.top - 0.15 * self.height))),
                                           int(round(self.width * 1.3)),
                                           int(round(self.height * 1.3)))
        return image[y_start: min(y_start + height, image.shape[0]), x_start: min(x_start + width, image.shape[1])]



    def calculate_iou(self, bbox):
        xA = max(self.top_left()[0], bbox.top_left()[0])
        yA = max(self.top_left()[1], bbox.top_left()[1])
        xB = min(self.bot_right()[0], bbox.bot_right()[0])
        yB = min(self.bot_right()[1], bbox.bot_right()[1])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        return float(interArea) / (self.get_area() + bbox.get_area())



class DataLoader(object):
    @staticmethod
    def GenerateRandomBBox(width, height, bbox_width, bbox_height):
        try:
            x_start = int(round(np.random.randint(0, max(0, width - bbox_width))))
            y_start = int(round(np.random.randint(0, max(0, height - bbox_height))))
            newBox = BBox(x_start, y_start, bbox_width, bbox_height)
            return newBox
        except ValueError:
            return None


    @staticmethod
    def create_non_digit_data(data_path):
        datas, meta_datas = [], []
        index = 0

        meta_data_path = os.path.join(data_path, "digitStruct.mat")
        digit_struct_mat = h5py.File(meta_data_path, 'r')
        for img_file in os.listdir(data_path):
            if img_file.endswith(".png"):
                index = int(img_file.replace(".png", ""))
                meta_data = DataLoader.get_attribute(digit_struct_mat, index - 1)
                if (meta_data.length > 5):
                    print("print index: {} has length larger than 5".format(str(index + 1)))
                    continue  # we ignore this case
                meta_datas.append(meta_data)
                new_image = cv2.imread(os.path.join(data_path, img_file))

                random_bbox = DataLoader.GenerateRandomBBox(new_image.shape[1], new_image.shape[0], 32, 32)
                if random_bbox and random_bbox.calculate_iou(meta_data.bbox) < 0.05:
                    # we can say this is a valid non digit
                    datas.append(cv2.resize(new_image, (64, 64)))

        return np.array(datas)


    @staticmethod
    def load_data(data_path, cropped=True):
        datas, meta_datas = [], []
        index = 0

        meta_data_path = os.path.join(data_path, "digitStruct.mat")
        digit_struct_mat = h5py.File(meta_data_path, 'r')
        for img_file in os.listdir(data_path):
            if img_file.endswith(".png"):
                index = int(img_file.replace(".png", ""))
                meta_data = DataLoader.get_attribute(digit_struct_mat, index - 1)
                if (meta_data.length > 5):
                    print("print index: {} has length larger than 5".format(str(index + 1)))
                    continue # we ignore this case
                meta_datas.append(meta_data)
                new_image = cv2.imread(os.path.join(data_path, img_file))
                if cropped:
                    new_image  = meta_data.bbox.crop(new_image)

                datas.append(cv2.resize(new_image, (64, 64)))

                print (datas[-1].shape)

        return np.array(datas), np.array(meta_datas)

    @staticmethod
    def load_meta_data(data_dir):
        meta_data = []
        meta_data_path = os.path.join(data_dir, "digitStruct.mat")
        with h5py.File(meta_data_path, 'r') as digit_struct_mat:
            name_field = digit_struct_mat['digitStruct']['name']
            meta_data = [DataLoader.get_attribute(digit_struct_mat, index) for index in range(name_field.shape[0])]

        return meta_data

    @staticmethod
    def get_attribute(hdfs_file, index):
        attrs = {}
        item = hdfs_file['digitStruct']['bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = hdfs_file[item][key]
            values = [hdfs_file[attr[i].item()][0][0]
                      for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
            attrs[key] = values

        length = len(attrs['label'])
        label = [10] * max(5, length)
        for i, ch in enumerate(attrs['label']):
            label[i] = int(ch) if int(ch) != 10 else 0

        bbox = build_bounding_box(attrs)
        meta_data = MetaData(length, label, bbox)
        return meta_data


# class Data(object):
#     def __init__(self, data_path):
#         test_path = constant.test_data()
#         train_path = constant.train_data()
#
#         self.test_data = DataLoader.load_data(test_path)
#         self.test_data_metadata = DataLoader.load_meta_data(test_path)
#
#         self.train_data = DataLoader.load_data(train_path)
#         self.train_data_metadata = DataLoader.load_meta_data(train_path)
#
#     def get_data(self):
#         return self.train_data, self.test_data
#
#     def get_meta_data(self):
#         return self.train_data_metadata, self.test_data_metadata


# class DataObject(object):
#     def __init__(self):

if __name__ == "__main__":
    import Constant
    DataLoader.create_non_digit_data(Constant.train_data())