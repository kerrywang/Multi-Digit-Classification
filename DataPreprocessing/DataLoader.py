from zipfile import ZipFile
import Constant as constant
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
        for ch in self.digit:
            if not ch: break
            res += str(ch)
        return res

class BBox(object):
    def __init__(self, attrs):
        self.left, self.top, self.width, self.height = self.build_bounding_box(attrs)

    def width(self):
        return self.width

    def height(self):
        return self.height

    def top_left(self):
        return self.left, self.top

    def crop(self, image):
        x_start, y_start, width, height = (int(round(self.left - 0.15 * self.width)),
                                           int(round(self.top - 0.15 * self.height)),
                                           int(round(self.width * 1.3)),
                                           int(round(self.height * 1.3)))
        return image[y_start: y_start + height, x_start: x_start + width]

    def build_bounding_box(self, attrs):
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
        return (center_x - max_side / 2.0,
                center_y - max_side / 2.0,
                max_side,
                max_side)


class DataLoader(object):
    @staticmethod
    def load_data(data_path):
        data = []
        for img_file in os.listdir(data_path):
            if img_file.endswith(".png"):
                data.append(cv2.imread(os.path.join(data_path, img_file)))
        return data

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
        label = [None] * max(5, length)
        for i, ch in enumerate(attrs['label']):
            label[i] = int(ch)

        bbox = BBox(attrs)
        meta_data = MetaData(length, label, bbox)
        return meta_data


class Data(object):
    def __init__(self):
        test_path = constant.test_data()
        train_path = constant.train_data()

        self.test_data = DataLoader.load_data(test_path)
        self.test_data_metadata = DataLoader.load_meta_data(test_path)

        self.train_data = DataLoader.load_data(train_path)
        self.train_data_metadata = DataLoader.load_meta_data(train_path)

    def get_data(self):
        return self.train_data, self.test_data

    def get_meta_data(self):
        return self.train_data_metadata, self.test_data_metadata

if __name__ == "__main__":
    dt_loader = Data()
    print ("here")
