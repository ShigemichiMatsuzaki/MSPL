import os
import sys
import torch
import torch.utils.data as data
from PIL import Image
import cv2
from transforms.segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose, Tensorize
from collections import OrderedDict
import numpy as np
from torchvision.transforms import functional as F
import random
import glob

FOREST_CLASS_LIST = ['road', 'grass', 'tree', 'sky', 'obstacle']

color_encoding = OrderedDict([
    ('road', (170, 170, 170)),
    ('grass', (0, 255, 0)),
    ('tree', (102, 102, 51)),
    ('sky', (0, 120, 255)),
    ('obstacle', (0, 0, 0))
])

color_to_id = {
    (255, 255, 255) : 0,
    (170, 170, 170) : 1,
    (0, 255, 0)     : 2,
    (102, 102, 51)  : 3,
    (0, 60, 0)      : 3,
    (0, 120, 255)   : 4,
    (0, 0, 0)       : 5
}

color_palette = [
    170, 170, 170,
    0, 255, 0,
    102, 102, 51,
    0, 120, 255,
    0, 0, 0
]

class FreiburgForestDataset(data.Dataset):

    def __init__(self, root='/tmp/dataset/freiburg_forest_annotated/', train=True, scale=(0.5, 2.0),
                 size=(480, 256), normalize=True):

        self.root = root
        self.train = train
        self.normalize = normalize

        if self.train:
            data_dir = os.path.join(root, 'train')
        else:
            data_dir = os.path.join(root, 'test')

        self.images = sorted(glob.glob(os.path.join(data_dir, 'rgb/*.jpg')))
        self.masks = sorted(glob.glob(os.path.join(data_dir, 'GT_color/*.png')))

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.train_transforms, self.val_transforms = self.transforms()

    def transforms(self):
        train_transforms = Compose(
            [
                # RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.size),
                RandomFlip(),
                Normalize() if self.normalize else Tensorize()
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.size),
                Normalize() if self.normalize else Tensorize()
            ]
        )
        return train_transforms, val_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(os.path.join(self.root, self.images[index])).convert('RGB')
        label_img = Image.open(os.path.join(self.root, self.masks[index]))

        # Convert color label to label id
        label_img_color_np = np.array(label_img, np.uint8)
        label_img_id_np = np.zeros((label_img_color_np.shape[:2]), dtype=np.uint8)
        for color in color_to_id:
            label_img_id_np[(label_img_color_np == color).all(axis=2)] = color_to_id[color]
#        label_img = Image.fromarray(label_img_id_np)

        # Convert the label values
        label_img_id_np -= 1
        label_img_id_np[label_img_id_np < 0] = 255 # void 
        label_img = Image.fromarray(label_img_id_np)

        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        # Get a file name
        filename = self.images[index].rsplit('/', 1)[1]

        return rgb_img, label_img, filename

#    def __getitem__(self, index):
#        rgb_img = Image.open(self.images[index]).convert('RGB')
#        label_img = Image.open(self.masks[index])
#        '''
#        Open a depth image using OpenCV instead of PIL to deal with int16 format of the image
#        '''
#        if self.use_depth:
#            cv_depth = cv2.imread(self.depths[index], cv2.IMREAD_GRAYSCALE)
#            # cv_depth = cv2.medianBlur(cv_depth, 7)
#            #cv_depth = np.where(cv_depth < 10, cv_depth, 10) * (255 // 10)
#            cv_depth.astype(np.uint8)
#            #print(cv_depth)
#            #print(np.histogram(cv_depth, bins=10))
#            depth_img = Image.fromarray(cv_depth)
##            print(np.asarray(rgb_img))
#
#        if not self.use_traversable:
#            label_np = np.array(label_img, np.uint8)
#            label_np[label_np==0] = 1
#            label_img = Image.fromarray(label_np)
#
#        if self.train:
#            if self.use_depth:
#                rgb_img, label_img, depth_img = self.train_transforms(rgb_img, label_img, depth_img)
#            else:
#                rgb_img, label_img = self.train_transforms(rgb_img, label_img)
#        else:
#            if self.use_depth:
#                rgb_img, label_img, depth_img = self.val_transforms(rgb_img, label_img, depth_img)
#            else:
#                rgb_img, label_img = self.val_transforms(rgb_img, label_img)
#
#        # Get a file name
#        filename = self.images[index]#.rsplit('/', 1)[1]
#
#        if self.use_depth:
#            return rgb_img, label_img, depth_img, filename, self.reg_weight
#        else:
#            return rgb_img, label_img, filename, self.reg_weight