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
import skimage.io

import yaml
from PIL import Image

# Root directory of the project
ROOT_DIR = ("/media/ych/Ubuntu/tensorflow/Mask_RCNN")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib

# Directory to save logs and trained models
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num = 0

# Local path to trained weights file
COCO_MODEL_PATH = "/media/ych/Ubuntu/GG/pretrained_weight/mask_rcnn_coco/mask_rcnn_coco.h5"


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
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # background + 1 class

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class DrugDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            #temp = yaml.load(f.read(), None)
            temp = yaml.load(f, Loader=yaml.FullLoader)
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
    def load_shapes(self, count, img_floder,  imglist,  mode):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "corn")
        self.add_class("shapes", 2, "rice")
        self.add_class("shapes", 3, "wheat")
        self.add_class("shapes", 4, "fallow")
        self.add_class("shapes", 5, "wasteland")
        self.add_class("shapes", 6, "bareland")
        self.add_class("shapes", 7, "soybean")
        self.add_class("shapes", 8, "rape")

        if mode == 'train':
            for i in range(count):
                # 获取图片宽和高
                print("{}/{} loading train img: {}\n".format(i, count, imglist[i]))
                mask_path = os.path.join(img_floder, imglist[i], "label.png")
                yaml_path = os.path.join(img_floder, imglist[i], "info.yaml")
                img_path = os.path.join(img_floder, imglist[i], "img.png")
                cv_img = cv2.imread(img_path)

                self.add_image("shapes", image_id=i, path=img_path,
                               width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)
        else:
            for i in range(count):
                # 获取图片宽和高
                print("{}/{} loading val img: {}\n".format(i, count, imglist[i]))
                mask_path = os.path.join(img_floder, imglist[i], "label.png")
                yaml_path = os.path.join(img_floder, imglist[i], "info.yaml")
                img_path = os.path.join(img_floder, imglist[i], "img.png")
                cv_img = cv2.imread(img_path)

                self.add_image("shapes", image_id=i, path=img_path,
                               width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)


    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])

        return image


    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        count = 1
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
                # print "box"
                labels_form.append("corn")
            if labels[i].find("rice") != -1:
                # print "box"
                labels_form.append("rice")
            if labels[i].find("wheat") != -1:
                # print "box"
                labels_form.append("wheat")
            if labels[i].find("fallow") != -1:
                # print "box"
                labels_form.append("fallow")
            if labels[i].find("wasteland") != -1:
                # print "box"
                labels_form.append("wasteland")
            if labels[i].find("bareland") != -1:
                # print "box"
                labels_form.append("bareland")
            if labels[i].find("soybean") != -1:
                # print "box"
                labels_form.append("soybean")
            if labels[i].find("rape") != -1:
                # print "box"
                labels_form.append("rape")

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

def train_model():
    # 基础设置
    dataset_root_path = "/home/ych/Ubuntu/GG/dataset/crop_detection/GVG/label"
    train_img_floder = os.path.join(dataset_root_path, "dataset_train")
    val_img_floder = os.path.join(dataset_root_path, "dataset_val")
    train_imglist = os.listdir(train_img_floder)
    train_count = len(train_imglist)
    val_imglist = os.listdir(val_img_floder)
    val_count = len(val_imglist)

    # train与val数据集准备
    dataset_train = DrugDataset()
    # dataset_train.load_shapes(train_count, train_img_floder,  train_imglist, 'train')
    dataset_train.load_shapes(1000, train_img_floder, train_imglist, 'train')
    dataset_train.prepare()

    # print("dataset_train-->",dataset_train._image_ids)

    dataset_val = DrugDataset()
    # dataset_val.load_shapes(val_count, val_img_floder,  val_imglist, 'val')
    dataset_val.load_shapes(100, val_img_floder,  val_imglist, 'val')
    dataset_val.prepare()

    # Create models in training mode
    config = ShapesConfig()
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # Which weights to start with?
    # 第一次训练时，这里填coco，在产生训练后的模型后，改成last
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last models you trained and continue training
        checkpoint_file = model.find_last()
        model.load_weights(checkpoint_file, by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=300,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=1000,
                layers="all")

class TongueConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def test():
    import skimage.io
    from mrcnn import visualize

    # Create models in training mode
    config = TongueConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    model_path = model.find_last()

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    class_names = ['BG', 'corn', 'rice', 'wheat']

    # Load a random image from the images folder
    file_names = '/media/ych/Ubuntu/GG/dataset/crop_detection/dataset_rotated/corn_586.jpg' # next(os.walk(IMAGE_DIR))[2]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(file_names)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

if __name__ == "__main__":
    train_model()
    #test()