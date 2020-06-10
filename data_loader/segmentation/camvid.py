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

CAMVID_CLASS_LIST = [
    'Sky', 
    'Building', 
    'Pole', 
    'Road', 
    'Pavement', 
    'Tree', 
    'SignSymbol', 
    'Fence', 
    'Car', 
    'Pedestrian', 
    'Bicyclist', 
    'Road_marking', 
    'Unlabelled']

color_encoding = OrderedDict([
    ('Sky', (128,128,128)),
    ('Building', (128,0,0)),
    ('Pole', (192,192,128)),
    ('Road', (128,64,128)),
    ('Pavement', (60,40,222)),
    ('Tree', (128,128,0)),
    ('SignSymbol', (192,128,128)),
    ('Fence', (64,64,128)),
    ('Car', (64,0,128)),
    ('Pedestrian', (64,64,0)),
    ('Bicyclist', (0,128,192)),
    ('Road_marking', (255,69,0)),
    ('Unlabelled', (0,0,0))
])

id_camvid_to_greenhouse = np.array([
    4, # Sky
    2, # Building
    2, # Pole
    3, # Road
    3, # Pavement
    1, # Tree
    2, # SignSymbol
    2, # Fence
    2, # Car
    4, # Pedestrian
    4, # Bicyclist
    2, # Road_marking(?)
    4  # Unlabeled
])

class CamVidSegmentation(data.Dataset):

    def __init__(self, root, list_name, train=True, scale=(0.5, 2.0), size=(360, 480), label_conversion=False):

        self.train = train
        data_file = os.path.join(root, list_name)

        self.images = []
        self.masks = []
        with open(data_file, 'r') as lines:
            for line in lines:
                line_split = line.split(',')
#                rgb_img_loc = root + os.sep + line_split[0].rstrip()
                rgb_img_loc = line_split[0].rstrip()
#                rgb_img_loc = root + os.sep + line_split[1].rstrip()
                label_img_loc = line_split[1].rstrip()
                assert os.path.isfile(rgb_img_loc)
                assert os.path.isfile(label_img_loc)
                self.images.append(rgb_img_loc)
                self.masks.append(label_img_loc)

        self.label_conversion = label_conversion
        if self.label_conversion:
            self.ignore_idx = 4
        else:
            self.ignore_idx = 12

        
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
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.size, ignore_idx=self.ignore_idx),
                Resize(size=self.size),
                RandomFlip(),
#                Normalize()
                Tensorize()
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.size),
#                Normalize()
                Tensorize()
            ]
        )
        return train_transforms, val_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert('RGB')
        label_img = Image.open(self.masks[index])

        if self.label_conversion:
            label_np = np.array(label_img, np.int8)
            label_np = id_camvid_to_greenhouse[label_np]
            label_np = np.array(label_np, np.uint8)
        
            label_img = Image.fromarray(label_np)

        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)


        # Get a file name
        filename = self.images[index].rsplit('/', 1)[1]

        return rgb_img, label_img, filename, 0.0

