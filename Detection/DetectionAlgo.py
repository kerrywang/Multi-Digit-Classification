import numpy as np
import torch.nn
import torch
import cv2
import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from Models import Model, VGG16, VGG16Detection
import matplotlib.pyplot as plt


def check_vailid(detection_res, digit_min_prob=0.3, length_min_prob=0.9):
    length_prob, digit1_prob, digit2_prob, digit3_prob, digit4_prob, digit5_prob = detection_res
    N_digit = np.argmax(length_prob)
    digits_prob = np.array([digit1_prob, digit2_prob, digit3_prob, digit4_prob, digit5_prob])
    digits = np.array([np.argmax(digit1_prob),
                       np.argmax(digit2_prob),
                       np.argmax(digit3_prob),
                       np.argmax(digit4_prob),
                       np.argmax(digit5_prob)])
    status = False

    if N_digit > 0:
        # check Number of digit is larger than threshold
        # check that N + 1 : End digits is not valid digit
        #         print (length_prob[N_digit] > length_min_prob)
        #         print (np.all(digits[:N_digit] < 10))
        #         print (np.all(digits[N_digit:] == 10))
        #         print (np.all(digits_prob[:N_digit] >= digit_min_prob))
        status = length_prob[N_digit] > length_min_prob and \
                 np.all(digits[:N_digit] < 10) and \
                 np.all(digits[N_digit:] == 10) and \
                 np.all(np.max(digits_prob[:N_digit]) >= digit_min_prob)

    return status

def to_prob(tensor):
    x = tensor.cpu().detach().numpy()
    return np.exp(x[0]) / sum(np.exp(x[0]))


def inference(model, image):
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.CenterCrop([54, 54]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image)
    image = im_pil.convert('RGB')
    image = transform(image)
    images = image.unsqueeze(dim=0).cuda()

    return model.eval()(images)


def batch_images(model, images):
    N, labels, p_outputs = [], [], []
    for image in list(images):
        length_prob, digit1_prob, digit2_prob, digit3_prob, digit4_prob, digit5_prob = map(to_prob, inference(model, image))
        N_digit = np.argmax(length_prob)
        digits_prob = np.array([np.max(length_prob), np.max(digit1_prob), np.max(digit2_prob), np.max(digit3_prob), np.max(digit4_prob), np.max(digit5_prob)])
        digits = np.array([np.argmax(digit1_prob),
                           np.argmax(digit2_prob),
                           np.argmax(digit3_prob),
                           np.argmax(digit4_prob),
                           np.argmax(digit5_prob)])
        N.append(N_digit)
        labels.append(digits)
        p_outputs.append(digits_prob)

    return np.array(N), np.array(labels), np.array(p_outputs)

def batch_images_detection(model, images):
    isDigitResult = []
    for image in list(images):
        isDigit = inference(model, image)
        print(isDigit.cpu().detach().numpy())
        isdigit = np.argmax(isDigit.cpu().detach().numpy())
        # if isdigit == 1:
        #     cv2.imshow("has digit", image)
        #     cv2.waitKey(0)
        isDigitResult.append(isdigit)
    return np.array(isDigitResult)


def batch_check_valid(probability, labels, N_digits, min_digit_prob=0.5, min_N_prob=0.98):
    valididity = []
    for i in np.arange(labels.shape[0]):
        if N_digits[i] <= 0:
            valididity.append(False)
        else:
            # check N digit probability is significant, i.e. larger than min_N_prob threshold
            # check that for each entry the first N digit is a valid number
            # check that for each entry the N + 1 : <End> digit is not a valid number
            status = probability[i, 0] > min_N_prob and \
                     np.all(labels[i, :N_digits[i]] < 10) and \
                     np.all(labels[i, N_digits[i]:] == 10) and \
                     np.all(probability[i, :N_digits[i]] >= min_digit_prob)
            valididity.append(status)

    return np.array(valididity)

def load_model(checkpoint_path, model_name="VGG16"):
    if model_name == "VGG16":
        model = VGG16()
    elif model_name == "detection":
        model = VGG16Detection()
    else:
        model = Model()

    model.cuda()  # make it an option
    model.restore(checkpoint_path)
    return model

def sliding_window_crops(image, bbox, box_size, resizeTO=(32,32), strides=(4, 4), minDim_size=200, debug=False):
    """stride: in x,y directions. minDim_size: size of minimum dimension"""
    x0,y0,width,height=bbox
    bx, by = box_size
    w,h=max(width,bx+1), max(height,by+1)
    assert h > by and w > bx
    if debug: plt.imsave('debug/crop__bbox_{0}_{1}_{2}_{3}__atSize_{4}_{5}.png'.format(x0,y0,w,h,bx,by), image[y0:y0+h, x0:x0+w][...,[2,1,0]]) # TODO: remove

    def sliding_window_crop_locations(h, w, box_size, strides):
        """height, width, box_size, strides. Returns crop locations"""
        sx, sy = strides
        bx, by = box_size
        xx, yy = np.meshgrid((x0+np.arange(w - bx))[::sx], (y0+np.arange(h - by))[::sy])

        r, c = xx.shape
        x_size = np.ones((r * c, 1)) * bx
        y_size = np.ones((r * c, 1)) * by
        return (np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1), x_size, y_size)))  # [[x,y]] locations

    crop_locs = sliding_window_crop_locations(h, w, box_size, strides)
    crops = [cv2.resize(image[int(y): int(y + by), int(x) : int(x + bx)], resizeTO) for x, y, w, h in crop_locs]

    if debug:
        test_image=image.copy() # TODO: Remove
        for x, y in crop_locs:
            test_image = cv2.rectangle(test_image, (x,y),(x+bx,y+by),(0,0,255),1)
        plt.imsave('debug/slides__bbox_{0}_{1}_{2}_{3}__atSize_{4}_{5}.png'.format(x0,y0,w,h,bx,by), test_image[...,[2,1,0]])

    w,h=resizeTO
    return (np.vstack(crops).reshape(-1, h, w, 3), crop_locs)


class UnionFind(object):
    def __init__(self, N):
        self.cluster = N
        self.parent = list(range(N))
        self.size = [1] * N

    def findParent(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.findParent(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        pi = self.findParent(i)
        pj = self.findParent(j)
        if pi == pj: return

        si, sj = self.size[pi], self.size[pj]

        if si < sj:
            self.parent[pi] = pj
            self.size[pj] += si
        else:
            self.parent[pj] = pi
            self.size[pi] + sj

        self.cluster -= 1

def combine(current_bbox, new_bbox, max_size, box_size):
    if current_bbox is None: return new_bbox
    x_min, y_min = min(current_bbox[0], new_bbox[0]), min(current_bbox[1], new_bbox[1])
    x_max, y_max = max(current_bbox[0], new_bbox[0]), max(current_bbox[1], new_bbox[1])

    w, h = x_max + box_size - x_min, y_max + box_size - y_min
    return (x_min, y_min, w, h)


def combine_bbox_for_class(union, target_parent, max_size, box_size, bboxes_list):
    current_bbox = None
    for i, parent in enumerate(union.parent):
        if parent == target_parent:
            new_bbox = bboxes_list[i]
            current_bbox = combine(current_bbox, new_bbox, max_size, box_size)
    return current_bbox

def combine_bbox_for_top_clusters(union, max_size, max_num_cluster, box_size, bboxes_list):
    unique_parents = set(union.parent)
    union_ranking = sorted([(union.size[parent], parent) for parent in unique_parents])[-max_num_cluster:]
    bboxes = []
    for _, parent in union_ranking:
        bboxes.append(combine_bbox_for_class(union, parent, max_size, box_size, bboxes_list))
    return bboxes




def digit_detection(model, image, debug=True):
    if debug:
        if os.path.exists('./debug'):
            for aFile in os.listdir('./debug'):
                os.remove(os.path.join('./debug', aFile))
        else:
            os.makedirs('./debug')

    window_size = 64
    expansion_ratio = 2
    nrounds = 4
    strides = (4, 4)
    height, width, channel = image.shape
    box_queue = [(0, 0, width, height)]
    max_num_cluster = 3


    def should_connect(first_coor, second_coor):
        x0, y0, w0, h0 = first_coor
        x1, y1, w1, h1 = second_coor
        return (abs(x0 - x1) <= strides[0] * 3) and (abs(y0 - y1) <= strides[1] * 3)

    box_size = window_size

    for i in range(nrounds):
        valid_crop_start_loc = []
        while box_queue:
            bbox = box_queue.pop()
            # crop the given bounding box with box size
            crops, crop_start_loc = sliding_window_crops(image, bbox, [box_size, box_size], strides=strides)
            # n_digit, digit_val, classified_prob = batch_images(model, crops)
            # valid_crop = batch_check_valid(classified_prob, digit_val, n_digit)
            isdigit = batch_images_detection(model, crops)
            valid_crop = np.where(isdigit == 1, True, False)
            # print(valid_crop.shape)
            # print(crop_start_loc.shape)
            valid_crop_start_loc.extend(list(crop_start_loc[valid_crop]))
            print (len(valid_crop_start_loc))

            test_bbox_img = image.copy()
            for  x_min, y_min, w_box, h_box in valid_crop_start_loc:
                test_bbox_img = cv2.rectangle(test_bbox_img, (int(x_min), int(y_min)), (int(x_min + w_box), int(y_min + h_box)),
                                              (0, 0, 255), 1)
            cv2.imshow("tesst", test_bbox_img)
            cv2.waitKey(0)

        num_valid = len(valid_crop_start_loc)
        union = UnionFind(num_valid)
        for i in range(num_valid):
            for j in range(num_valid):
                if i == j: continue
                if should_connect(valid_crop_start_loc[i], valid_crop_start_loc[j]):
                    union.union(i, j)

        bboxes = combine_bbox_for_top_clusters(union, image.shape, max_num_cluster, box_size, valid_crop_start_loc)
        box_queue.extend(bboxes)
        box_size = int(round(box_size * expansion_ratio))

        test_bbox_img = image.copy()
        for bbox in box_queue:
            x_min, y_min, w_box, h_box = bbox
            test_bbox_img = cv2.rectangle(test_bbox_img, (int(x_min), int(y_min)), (int(x_min + w_box), int(y_min + h_box)), (0, 0, 255), 1)
        plt.imsave('debug/test_bbox_at_{}.png'.format(i), test_bbox_img[..., [2, 1, 0]])

if __name__ == "__main__":
    model = load_model("../logs/Detection/BASE_MODEL_5000", "detection")
    image = cv2.imread("../Data/train/1.png")
    digit_detection(model, image)