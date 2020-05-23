# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
# import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image
from random import shuffle
import itertools
import colorsys
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
import skimage.io
import PIL.Image as Image
# yaml.warnings({'YAMLLoadWarning': False})

class_names = ["bg",
               "corn",
               "rice",
               "wheat",
               "rape",
               "soybean",
               "fallow",
               "bareland",
               "wasteland"]
class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]
dest_root_path = "/media/ych/Ubuntu/GG/dataset/crop_detection/GVG/single_kind"
class DrugDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
        #temp = yaml.load(stream=f.read(), Loader=yaml.FullLoader())
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        # print("draw_mask-->",image_id)
        # print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        # print("info-->",info)
        # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "corn")
        self.add_class("shapes", 2, "rice")
        self.add_class("shapes", 3, "wheat")
        self.add_class("shapes", 4, "rape")
        self.add_class("shapes", 5, "soybean")
        self.add_class("shapes", 6, "fallow")
        self.add_class("shapes", 7, "bareland")
        self.add_class("shapes", 8, "wasteland")
        for i in range(count):
            # 获取图片宽和高
            mask_path = os.path.join(dataset_root_path, imglist[i], "label.png")
            yaml_path = os.path.join(dataset_root_path, imglist[i], "info.yaml")
            imgpath = os.path.join(dataset_root_path, imglist[i], "img.png")
            print("Loading dataset{}/{}:{}".format(i+1,count,imgpath))
            cv_img = cv2.imread(imgpath)
            self.add_image("shapes", image_id=i, path=imgpath,
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        #print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("corn") != -1:
                # print "car"
                labels_form.append("corn")
            elif labels[i].find("rice") != -1:
                # print "leg"
                labels_form.append("rice")
            elif labels[i].find("wheat") != -1:
                # print "well"
                labels_form.append("wheat")
            elif labels[i].find("rape") != -1:
                # print "leg"
                labels_form.append("rape")
            elif labels[i].find("soybean") != -1:
                # print "well"
                labels_form.append("soybean")
            elif labels[i].find("fallow") != -1:
                # print "leg"
                labels_form.append("fallow")
            elif labels[i].find("wasteland") != -1:
                # print "well"
                labels_form.append("wasteland")
            elif labels[i].find("bareland") != -1:
                # print "leg"
                labels_form.append("bareland")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def crop_mask(image, mask, class_ids, class_names):
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]

def get_area(image, mask, class_ids, image_id):
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    class_id = top_ids[0]

    h = image.shape[0]
    w = image.shape[1]
    img_array = np.array(image)
    for i in range(112, h-112, 50):
        for j in range(112, w-112, 50):
            #m = mask[i-112:i+112,j-112:j+112,:]
            m = mask[i-112:i+112,j-112:j+112, np.where(class_ids == class_id)[0]]
            m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
            label = is_one(m)
            if label != 0:
                print(image_id)
                print(class_names[class_id])
                class_count[class_id] += 1
                img_out = img_array[i-112:i+112,j-112:j+112,:]
                image = Image.fromarray(img_out)
                save_name = os.path.join(dest_root_path, class_names[class_id], class_names[class_id] + str(class_count[class_id]) + '.png')
                image.save(save_name)



def is_one(twoD_array):
    first = twoD_array[0][0]
    rows = twoD_array.shape[0]
    cols = twoD_array.shape[1]
    cnt = 0
    flag = True
    if first != 0 and class_count[first] < 100000:
        for i in range(rows):
            for j in range(cols):
                cnt +=1
                if first != twoD_array[i][j]:
                    flag = False
                    break
            if not flag :
                break
    else :
        flag = False
    if flag:
        return first
    else:
        return 0

def run():
    ori_root_path = "/media/ych/Ubuntu/GG/dataset/crop_detection/GVG/label/dataset4single_class_train"
    ori_list = os.listdir(ori_root_path)
    ori_count = len(ori_list)
    dataset_ori = DrugDataset()
    dataset_ori.load_shapes(ori_count,  ori_list, ori_root_path)
    dataset_ori.prepare()

    for image_id in dataset_ori.image_ids:
        image = dataset_ori.load_image(image_id)
        mask, class_ids = dataset_ori.load_mask(image_id)
        #visualize.display_top_masks(image, mask, class_ids, dataset_ori.class_names,1)
        get_area(image, mask, class_ids, image_id)



if __name__ == '__main__':
    run()