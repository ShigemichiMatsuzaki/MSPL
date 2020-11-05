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

color_to_id_city = {

}

color_to_id_camvid = {
}

color_palette_camvid = [
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    128,64,128,
    0,0,192,
    64,192,128,
    0,0,0,
    128,0,0,
    64,192,0,
    64,64,128,
    0,0,0,
    0,128,64,
    64,0,64,
    192,192,128,
    192,192,128,
    0,64,64,
    192,128,128,
    128,128,0,
    192,192,0,
    128,128,128,
    64,64,0,
    0,128,192,
    64,0,128,
    64,128,192,
    192,128,192,
    64,128,192,
    64,128,192,
    192,64,128,
    192,0,192,
    192,64,0,
    0,0,0
]

color_palette_city = [
      0,  0,  0,
      0,  0,  0,
      0,  0,  0,
      0,  0,  0,
    20,  20,  20,
    111, 74,  0,
     81,  0, 81,
    128, 64,128,
    244, 35,232,
    250,170,160,
    230,150,140,
     70, 70, 70,
    102,102,156,
    190,153,153,
    180,165,180,
    150,100,100,
    150,120, 90,
    153,153,153,
    153,153,153,
    250,170, 30,
    220,220,  0,
    107,142, 35,
    152,251,152,
     70,130,180,
    220, 20, 60,
    255,  0,  0,
      0,  0,142,
      0,  0, 70,
      0, 60,100,
      0,  0, 90,
      0,  0,110,
      0, 80,100,
      0,  0,230,
    119, 11, 32,
      0,  0,142
]

class GTA5(data.Dataset):

    def __init__(self, root, list_name, train=True, scale=(0.5, 2.0), size=(480, 256), normalize=True):

        self.root = root
        self.normalize = normalize
        self.train = train

        self.images = []
        self.masks = []
        data_file = os.path.join(root, list_name)
        with open(data_file, 'r') as lines:
            for line in lines:
                line_split = line.split(',')
#                rgb_img_loc = root + os.sep + line_split[0].rstrip()
                rgb_img_loc = line_split[0].rstrip()
#                rgb_img_loc = root + os.sep + line_split[1].rstrip()
                label_img_loc = line_split[1].rstrip()
                print(rgb_img_loc)
                print(label_img_loc)
                assert os.path.isfile(rgb_img_loc)
                assert os.path.isfile(label_img_loc)
                self.images.append(rgb_img_loc)
                self.masks.append(label_img_loc)

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