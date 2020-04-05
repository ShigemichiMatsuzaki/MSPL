import os
import torch
import torch.utils.data as data
from PIL import Image
import cv2
from transforms.segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose, Tensorize
from collections import OrderedDict
import numpy as np
from torchvision.transforms import functional as F

ISHIHARA_RGBD_CLASS_LIST = [
    'Unlabeled',
    'Building',
    'Fence',
    'Others',
    'Pedestrian',
    'Pole',
    'Road line',
    'Road',
    'Sidewalk',
    'Vegetation',
    'Car',
    'Wall',
    'Traffic sign'
]

class IshiharaRGBDSegmentation(data.Dataset):

    def __init__(self, root, list_name, train=True, scale=(0.5, 2.0), size=(400, 304), ignore_idx=255, coarse=True):

        self.train = train
        if self.train:
            data_file = os.path.join(root, list_name)
            if coarse:
                coarse_data_file = os.path.join(root, list_name)
        else:
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

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.train_transforms, self.val_transforms = self.transforms()
        self.ignore_idx = ignore_idx

    def transforms(self):
        train_transforms = Compose(
            [
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.size),
                RandomFlip(),
                Normalize()
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.size),
                Normalize()
            ]
        )
        return train_transforms, val_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert('RGB')
        label_img = Image.open(self.masks[index])

        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        return rgb_img, label_img

class IshiharaRGBDSegmentation(data.Dataset):

    def __init__(self, root, list_name, train=True, scale=(0.5, 2.0), size=(400, 304), ignore_idx=255, use_depth=True):

        self.train = train
        self.use_depth = use_depth
        if self.train:
            data_file = os.path.join(root, list_name)
        else:
            data_file = os.path.join(root, list_name)

        self.images = []
        self.masks = []
        self.depths = []
        with open(data_file, 'r') as lines:
            for line in lines:
                line_split = line.split(',')
#                rgb_img_loc = root + os.sep + line_split[0].rstrip()
                rgb_img_loc = line_split[0].rstrip()
#                rgb_img_loc = root + os.sep + line_split[1].rstrip()
                label_img_loc = line_split[1].rstrip()
                assert os.path.isfile(rgb_img_loc)
                #print(label_img_loc)
                assert os.path.isfile(label_img_loc)
                if self.use_depth:
                    depth_img_loc = line_split[2].rstrip()
                    assert os.path.isfile(depth_img_loc)

                self.images.append(rgb_img_loc)
                self.masks.append(label_img_loc)
                if self.use_depth:
                    self.depths.append(depth_img_loc)

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.train_transforms, self.val_transforms = self.transforms()
        self.ignore_idx = ignore_idx

    def transforms(self):
        train_transforms = Compose(
            [
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.size),
                RandomFlip(),
                Normalize()
#                Tensorize()
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.size),
                Normalize()
#                Tensorize()
            ]
        )
        return train_transforms, val_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert('RGB')
        label_img = Image.open(self.masks[index])
        '''
        Open a depth image using OpenCV instead of PIL to deal with int16 format of the image
        '''
        if self.use_depth:
            cv_depth = cv2.imread(self.depths[index], cv2.IMREAD_GRAYSCALE)
            # cv_depth = cv2.medianBlur(cv_depth, 7)
            #cv_depth = np.where(cv_depth < 10, cv_depth, 10) * (255 // 10)
            cv_depth.astype(np.uint8)
            #print(cv_depth)
            #print(np.histogram(cv_depth, bins=10))
            depth_img = Image.fromarray(cv_depth)
#            print(np.asarray(rgb_img))
           

        if self.train:
            if self.use_depth:
                rgb_img, label_img, depth_img = self.train_transforms(rgb_img, label_img, depth_img)
            else:
                rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            if self.use_depth:
                rgb_img, label_img, depth_img = self.val_transforms(rgb_img, label_img, depth_img)
            else:
                rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        if self.use_depth:
            return rgb_img, label_img, depth_img
        else:
            return rgb_img, label_img

class IshiharaDepth(data.Dataset):

    def __init__(self, root, list_name, train=True, scale=(0.5, 2.0), size=(480, 264), use_filter=True):

        self.train = train
        if self.train:
            data_file = os.path.join(root, list_name)
        else:
            data_file = os.path.join(root, list_name)

        # Whether to apply a median filter to depth images
        self.use_filter = use_filter

        self.images = []
        self.depths = []
        with open(data_file, 'r') as lines:
            for line in lines:
                line_split = line.split(',')
                rgb_img_loc = line_split[0].rstrip()
                depth_img_loc = line_split[1].rstrip()
                assert os.path.isfile(rgb_img_loc)
                assert os.path.isfile(depth_img_loc)

                self.images.append(rgb_img_loc)
                self.depths.append(depth_img_loc)

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
                RandomCrop(crop_size=self.size),
                RandomFlip()
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.size)
            ]
        )
        return train_transforms, val_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert('RGB')
        depth_img = Image.open(self.depths[index])

        '''
        Open a depth image using OpenCV instead of PIL to deal with int16 format of the image
        '''
        cv_depth = cv2.imread(self.depths[index], cv2.IMREAD_GRAYSCALE)
        # Filtering
        if self.use_filter:
            cv_depth = cv2.medianBlur(cv_depth, 7)

        # Limit the maximum value to 10[m]
        cv_depth = np.where(cv_depth < 10, cv_depth, 10) * (255 // 10)
        cv_depth.astype(np.uint8) # just in case
        depth_img = Image.fromarray(cv_depth)
           
        # Apply transformation
        rgb_img, _, depth_img = self.train_transforms(rgb_img, rgb_img, depth_img) # Second arg is a dummy

        return F.to_tensor(rgb_img), F.to_tensor(depth_img)
