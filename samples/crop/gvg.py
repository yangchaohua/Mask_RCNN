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
import tensorflow as tf
from mrcnn.config import Config
#import utils
from mrcnn import model as modellib, utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image
from random import shuffle
import skimage.io
yaml.warnings({'YAMLLoadWarning': False})
ROOT_DIR = "/home/tyh/tensorflow/Mask_RCNN"
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num = 0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "pretrained_model", "mask_rcnn_coco.h5")
LAST_TRAIN_MODEL_PATH = "/home/tyh/tensorflow/Mask_RCNN/logs/shapes20200130T1043/mask_rcnn_shapes_0066.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    #IMAGE_MIN_DIM = 448
    #IMAGE_MAX_DIM = 448
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    LEARNING_RATE = 0.001
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 4, 16 * 4, 32 * 4, 64 * 4, 128 * 4)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1280

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 222


config = ShapesConfig()
config.display()


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

def train():
    # 基础设置
    train_root_path = "/home/tyh/tensorflow/dataset/dataset_train"
    val_root_path = "/home/tyh/tensorflow/dataset/dataset_val"
    train_list = os.listdir(train_root_path)
    val_list = os.listdir(val_root_path)
    shuffle(train_list)
    train_count = len(train_list)
    val_count = len(val_list)
    # train与val数据集准备
    dataset_train = DrugDataset()
    dataset_train.load_shapes(train_count,  train_list, train_root_path)
    dataset_train.prepare()

    # print("dataset_train-->",dataset_train._image_ids)

    dataset_val = DrugDataset()
    dataset_val.load_shapes(val_count, val_list, val_root_path)
    dataset_val.prepare()

    # print("dataset_val-->",dataset_val._image_ids)

    # Load and display random samples
    # image_ids = np.random.choice(dataset_train.image_ids, 4)
    # for image_id in image_ids:
    #    image = dataset_train.load_image(image_id)
    #    mask, class_ids = dataset_train.load_mask(image_id)
    #    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        # print(COCO_MODEL_PATH)
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(LAST_TRAIN_MODEL_PATH, by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=200,
                layers="all")
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
def test():
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    model_path = os.path.join("/home/tyh/tensorflow/Mask_RCNN/logs/shapes20200130T1043/mask_rcnn_shapes_0060.h5")
    # model_path = model.find_last()
    class_names = ['BG','corn','rice', 'wheat','rape','soybean','fallow','bareland','wasteland']
    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    test_root_path = "/home/tyh/tensorflow/dataset/dataset_test"
    test_list = os.listdir(test_root_path)
    test_count = len(test_list)
    # train与val数据集准备
    dataset_test= DrugDataset()
    dataset_test.load_shapes(test_count, test_list, test_root_path)
    dataset_test.prepare()
    APs = []
    cnt = 0
    gt_8kind_class = [0., 0., 0., 0., 0., 0., 0., 0.]
    pred_8kind_class = [0., 0., 0., 0., 0., 0., 0., 0.]
    match_8kind_class = [0., 0., 0., 0., 0., 0., 0., 0.]
    for image_id in dataset_test.image_ids:
        # Load image and ground truth data
        image_id = dataset_test.image_ids[2437]
        cnt += 1
        print("testing: {}/{}".format(cnt, len(dataset_test.image_ids)))
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_test, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        #img = skimage.io.imread("/home/tyh/tensorflow/20200222130556.png")
        #results = model.detect([img], verbose=0)
        #r = results[0]
        visualize.display_instances(image,  r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        AP, precisions, recalls, overlaps, gt_match, pred_match = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'],
                             iou_threshold=0.5)
        APs.append(AP)
        for class_id in gt_class_id:
            gt_8kind_class[class_id-1] += 1
        if len(r['class_ids']) > 0:
            for class_id in r['class_ids']:
                pred_8kind_class[class_id-1] += 1
            for pr in pred_match:
                if pr > -1:
                    match_8kind_class[int(pr-1)] += 1

        print("gt:    ", gt_8kind_class)
        print("pred:  ", pred_8kind_class)
        print("match: ", match_8kind_class)
        if cnt >= 500:
            print("precs: ", [round(x / y, 4) for x, y in zip(match_8kind_class, pred_8kind_class)])
            print("avg_precs: ", np.mean([round(x / y, 4) for x, y in zip(match_8kind_class, pred_8kind_class)]))
            print("recs:  ", [round(x / y, 4) for x, y in zip(match_8kind_class, gt_8kind_class)])
            print("avg_recs:  ", np.mean([round(x / y, 4) for x, y in zip(match_8kind_class, gt_8kind_class)]))
    print("mAP: ", np.mean(APs))


if __name__ == '__main__':
    test()
    #train()