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
from data_loader.segmentation.utils import get_label_from_superpixel

GREENHOUSE_CLASS_LIST = ['end_of_plant', 'other_plant', 'artificial', 'ground', 'other']
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
id_cityscapes_to_greenhouse = np.array([
    3, # Road
    3, # Sidewalk
    2, # Building
    2, # Wall
    2, # Fence
    2, # Pole
    2, # Traffic light
    2, # Traffic sign
    1, # Vegetation
    3, # Terrain
    4, # Sky
    4, # Person
    4, # Rider
    2, # Car
    2, # Truck
    2, # Bus
    2, # Train
    2, # Motorcycle
    2, # Bicycle
    4 # Background
])
id_forest_to_greenhouse = np.array([
    3, # road
    1, # grass
    1, # tree
    2, # sky
    2  # obstacle
])

color_encoding = OrderedDict([
    ('end_of_plant', (0, 255, 0)),
    ('other_part_of_plant', (0, 255, 255)),
    ('artificial_objects', (255, 0, 0)),
    ('ground', (255, 255, 0)),
    ('background', (0, 0, 0))
])

color_palette = [
    0, 255, 0,
    0, 255, 255,
    255, 0, 0,
    255, 255, 0,
    0, 0, 0
]

class GreenhouseSegmentation(data.Dataset):

    def __init__(self, root, list_name, train=True, scale=(0.5, 2.0), size=(480, 256), 
                 ignore_idx=255, coarse=True, normalize=True):

        self.train = train
        if self.train:
            data_file = os.path.join(root, list_name)
            if coarse:
                coarse_data_file = os.path.join(root, list_name)
        else:
            data_file = os.path.join(root, list_name)

        self.normalize = normalize

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
        rgb_img = Image.open(self.images[index]).convert('RGB')
        label_img = Image.open(self.masks[index])

        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        # Get a file name
        filename = self.images[index].rsplit('/', 1)[1]

        return rgb_img, label_img, filename

class GreenhouseRGBDSegmentation(data.Dataset):

    def __init__(self, root=None, list_name=None, train=True, scale=(0.5, 2.0), size=(480, 256),
                 use_depth=True, reg_weights=0.0, use_traversable=True, normalize=True):
        self.train = train
        self.use_depth = use_depth
        self.reg_weight = reg_weights
        self.use_traversable = use_traversable
        self.normalize = normalize
#        if self.train:
        if root:
            data_file = os.path.join(root, list_name)
        else:
            data_file = list_name
        #else:
        #    data_file = os.path.join(root, list_name)

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
#                print(rgb_img_loc)
#                print(label_img_loc)
                if not os.path.isfile(rgb_img_loc):
                    print("Not found : " + rgb_img_loc)
                assert os.path.isfile(rgb_img_loc)

                if not os.path.isfile(label_img_loc):
                    print("Not found : " + label_img_loc)
                assert os.path.isfile(label_img_loc)
                if self.use_depth:
                    depth_img_loc = line_split[2].rstrip()
                    if not os.path.isfile(depth_img_loc):
                        print("Not found : " + depth_img_loc)
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

    def transforms(self):
        train_transforms = Compose(
            [
#                RandomScale(scale=self.scale),
#                RandomCrop(crop_size=self.size),
                Resize(size=self.size),
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

        if not self.use_traversable:
            label_np = np.array(label_img, np.uint8)
            label_np[label_np==0] = 1
            label_img = Image.fromarray(label_np)

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

        # Get a file name
        filename = self.images[index]#.rsplit('/', 1)[1]

        if self.use_depth:
            return rgb_img, label_img, depth_img, filename, self.reg_weight
        else:
            return rgb_img, label_img, filename, self.reg_weight

class GreenhouseDepth(data.Dataset):

    def __init__(self, root, list_name, train=True, scale=(0.5, 2.0), size=(480, 256), use_filter=True):

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
#                RandomScale(scale=self.scale),
                Resize(size=self.size),
#                RandomCrop(crop_size=self.size),
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

        # Get a file name
        filename = self.depths[index].rsplit('/', 1)[1]

        return F.to_tensor(rgb_img), F.to_tensor(depth_img), filename

class GreenhouseRGBDSegCls(data.Dataset):

    def __init__(self, root, list_name, train=True, scale=(0.5, 2.0), size=(480, 256), 
                 ignore_idx=4, use_depth=True, use_traversable=False, normalize=True):

        self.train = train
        self.use_depth = use_depth
        if self.train:
            data_file = os.path.join(root, list_name)
        else:
            data_file = os.path.join(root, list_name)

        self.normalize = normalize

        self.images = []
        self.masks = []
        self.depths = []
        self.cls_ids = []
        with open(data_file, 'r') as lines:
            line_num = 0
            for line in lines:
                line_num += 1
                line_split = line.split(',')
                # If the database file is malformed i.e., less than three commas are in one line.
                if len(line_split) != 4:
                    print("Malformed line at line {}".format(line_num))
                    print("One line must have exactly three commas to separate camera image, label image (optional), depth image and class id.")
                    sys.exit(1)

                # Camera
                rgb_img_loc = line_split[0].rstrip()
                assert os.path.isfile(rgb_img_loc)
                self.images.append(rgb_img_loc)

                # Label
                label_img_loc = line_split[1].rstrip()
                if label_img_loc:
                    assert os.path.isfile(label_img_loc)
                    self.masks.append(label_img_loc)
                else:
                    self.masks.append(None)

                # Depth
                depth_img_loc = line_split[2].rstrip()
                assert os.path.isfile(depth_img_loc)
                self.depths.append(depth_img_loc)

                # Class ID
                self.cls_ids.append(int(line_split[3].rstrip()))

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
#                RandomScale(scale=self.scale),
#                RandomCrop(crop_size=self.size),
                Resize(size=self.size),
                RandomFlip(),
                #Normalize()
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
        else:
            depth_img = None
            
        cls_id = self.cls_ids[index]

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

        # Get a file name
        filename = self.images[index].rsplit('/', 1)[1]

        if self.use_depth:
            return rgb_img, label_img, depth_img, cls_id, filename
        else:
            return rgb_img, label_img, cls_id, filename

#
# As a target dataset for training
# 
class GreenhouseRGBDStMineDataSet(data.Dataset):
    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('end_of_plant', (0, 255, 0)),
        ('other_part_of_plant', (0, 255, 255)),
        ('artificial_objects', (255, 0, 0)),
        ('ground', (255, 255, 0)),
        ('background', (0, 0, 0))
    ])

    def __init__(self, list_path, reg_weight = 0.0, rare_id = None,
                 mine_id = None, mine_chance = None, pseudo_root = None, max_iters=None,
                 size=(256, 480), mean=(128, 128, 128), std = (1,1,1), scale=(0.5, 1.5), mirror=True, ignore_label=255,
                 use_traversable=False, use_depth=True, use_label_prop=True, normalize=True):
        self.list_path = list_path
        self.pseudo_root = pseudo_root
        self.crop_h, self.crop_w = size
        self.lscale, self.hscale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.train_data = []
        self.train_labels = []
        self.reg_weight = reg_weight
        self.rare_id = rare_id
        self.mine_id = mine_id
        self.mine_chance = mine_chance
        self.use_depth = use_depth
        self.use_traversable = use_traversable
        self.use_label_prop = use_label_prop
        self.normalize = normalize

        self.images = []
        self.masks = []
        self.depths = []
        with open(list_path, 'r') as lines:
            for line in lines:
                line_split = line.split(',')
#                rgb_img_loc = root + os.sep + line_split[0].rstrip()
                rgb_img_loc = line_split[0].rstrip()
#                rgb_img_loc = root + os.sep + line_split[1].rstrip()
                label_img_loc = line_split[1].rstrip()
                assert os.path.isfile(rgb_img_loc)
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

    def transforms(self):
        train_transforms = Compose(
            [
#                RandomScale(scale=self.scale),
#                RandomCrop(crop_size=(self.size[1], self.size[0])),
                Resize(size=(self.size[1], self.size[0])),
                RandomFlip(),
                Normalize() if self.normalize else Tensorize()
                #
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.size),
                Normalize() if self.normalize else Tensorize()
                #Tensorize()
            ]
        )
        return train_transforms, val_transforms

    def __len__(self):
        return len(self.images)

#    def __len__(self):
#
#        return len(self.files)

#    def generate_scale_label(self, image, label):
#        # f_scale = 0.5 + random.randint(0, 11) / 10.0
#        f_scale = self.lscale + random.randint(0, int((self.hscale-self.lscale)*10)) / 10.0
#        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
#        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
#        return image, label

    def __getitem__(self, index):
#        datafiles = self.files[index]

        img_name = self.images[index].rsplit('/', 1)[1]
        rgb_img = Image.open(self.images[index]).convert('RGB').resize((self.size[1], self.size[0]))
        label_img = Image.open(self.masks[index]).resize((self.size[1], self.size[0]))
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
            depth_img = Image.fromarray(cv_depth).resize((self.size[1], self.size[0]))
#            print(np.asarray(rgb_img))

        if not self.use_traversable:
            label_np = np.array(label_img, np.uint8)
            label_np[label_np==0] = 1
            label_img = Image.fromarray(label_np)

#        if self.train:
#        else:
#            rgb_img, label_img, depth_img = self.val_transforms(rgb_img, label_img)
        #
        image = np.asarray(rgb_img, np.float32).copy()
        label = np.asarray(label_img, np.float32).copy()
        if self.use_depth:
            depth = np.asarray(depth_img, np.float32).copy()

        size = image.shape
        (img_h, img_w) = label.shape
        # image = np.asarray(image, np.float32)
#        image = image/255.0 # scale to [0,1]
#        image -= self.mean # BGR
#        image = image/self.std # np.reshape(self.std,(1,1,3))

        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(
                image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(
                label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
            depth_pad = cv2.copyMakeBorder(
                depth, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0)) if self.use_depth else None
        else:
            img_pad, label_pad = image, label
            depth_pad = depth if self.use_depth else None

        img_h, img_w = label_pad.shape
        # mining or not
        #mine_flag = random.uniform(0, 1) < self.mine_chance
        mine_flag = False
        if mine_flag and len(self.mine_id) > 0:
            label_unique = np.unique(label_pad)
            mine_id_temp = np.array([a for a in self.mine_id if a in label_unique]) # if this image has the mine id
            if mine_id_temp.size != 0:
                # decide the single id to be mined
                mine_id_img = mine_id_temp
                sel_idx = random.randint(0, mine_id_temp.size-1)
                sel_mine_id = mine_id_img[sel_idx]
                # seed for the mined id
                mine_id_loc = np.where(label_pad == sel_mine_id)  # tuple
                mine_id_len = len(mine_id_loc[0])
                seed_loc = random.randint(0, mine_id_len-1)
                hseed = mine_id_loc[0][seed_loc]
                wseed = mine_id_loc[1][seed_loc]
                # patch crop
                half_crop_h = self.crop_h/2
                half_crop_w = self.crop_w/2
                # center crop at the seed
                left_idx = wseed - half_crop_w
                right_idx = wseed + half_crop_w -1
                up_idx = hseed - half_crop_h
                bottom_idx = hseed + half_crop_h - 1
                # shift the left_idx or right_idx if they go beyond the pad margins
                if left_idx < 0:
                    left_idx = 0
                elif right_idx > img_w - 1:
                    left_idx = left_idx - ( ( half_crop_w - 1 ) - (img_w - 1 - wseed) ) # left_idx shifts to the left by the right beyond length
                if up_idx < 0:
                    up_idx = 0
                elif bottom_idx > img_h - 1:
                    up_idx = up_idx - ( ( half_crop_h - 1 ) - (img_h - 1 - hseed) ) # up_idx shifts to the up by the bottom beyond length
                h_off = up_idx
                w_off = left_idx
            else:
                h_off = random.randint(0, img_h - self.crop_h)
                w_off = random.randint(0, img_w - self.crop_w)
        else:
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)

        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w],   np.uint8)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.uint8)
#        image = np.asarray(rgb_img, np.uint8)
#        label = np.asarray(label_np, np.uint8)

        if self.use_depth:
#            depth = np.asarray(depth_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.uint8)
            depth = np.asarray(depth_img, np.uint8).copy()

        if self.use_label_prop:
#            rgb_img_tmp = rgb_img.resize((label.shape[1], label.shape[0]))
            label = get_label_from_superpixel(
                rgb_img_np=image, label_img_np=label, sp_type='watershed')
            label_img = Image.fromarray(label)

        if self.use_depth:
            rgb_img, label_img, depth_img = self.train_transforms(Image.fromarray(image), Image.fromarray(label), Image.fromarray(depth))

            return rgb_img, label_img, depth_img, img_name, self.reg_weight
        else:
            rgb_img, label_img = self.train_transforms(Image.fromarray(image), Image.fromarray(label))

            return rgb_img, label_img, img_name, self.reg_weight

class GreenhouseRGBDConfidenceSegmentation(data.Dataset):

    def __init__(self, root=None, list_name=None, train=True, scale=(0.5, 2.0), size=(480, 256),
                 use_depth=True, reg_weights=0.0, use_traversable=True, conf_root=None, normalize=True):
        self.train = train
        self.use_depth = use_depth
        self.reg_weight = reg_weights
        self.use_traversable = use_traversable
        self.normalize = normalize
#        if self.train:
        if root:
            data_file = os.path.join(root, list_name)
        else:
            data_file = list_name
        #else:
        #    data_file = os.path.join(root, list_name)

        self.images = []
        self.masks = []
        self.depths = []
        self.conf = []
        with open(data_file, 'r') as lines:
            for line in lines:
                line_split = line.split(',')
#                rgb_img_loc = root + os.sep + line_split[0].rstrip()
                rgb_img_loc = line_split[0].rstrip()
#                rgb_img_loc = root + os.sep + line_split[1].rstrip()
                label_img_loc = line_split[1].rstrip()
#                print(rgb_img_loc)
#                print(label_img_loc)
                assert os.path.isfile(rgb_img_loc)
                assert os.path.isfile(label_img_loc)
                if self.use_depth:
                    depth_img_loc = line_split[2].rstrip()
                    assert os.path.isfile(depth_img_loc)

                self.images.append(rgb_img_loc)
                self.masks.append(label_img_loc)
                if self.use_depth:
                    self.depths.append(depth_img_loc)
                
                file_name = rgb_img_loc.rsplit('/', 1)[1]
                assert os.path.isfile(os.path.join(conf_root, file_name))
                self.conf.append(os.path.join(conf_root, file_name))

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
#                RandomScale(scale=self.scale),
#                RandomCrop(crop_size=self.size),
                Resize(size=self.size),
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
        rgb_img = Image.open(self.images[index]).convert('RGB')
        label_img = Image.open(self.masks[index])

        '''
        Open a depth image using OpenCV instead of PIL to deal with int16 format of the image
        '''
        if self.use_depth:
            cv_depth = cv2.imread(self.depths[index], cv2.IMREAD_GRAYSCALE)
            cv_depth.astype(np.uint8)
            depth_img = Image.fromarray(cv_depth)

        if not self.use_traversable:
            label_np = np.array(label_img, np.uint8)
            label_np[label_np==0] = 1
            label_img = Image.fromarray(label_np)

        # Load confidence values
        conf_np = np.load(self.conf[index])

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

        # Get a file name
        filename = self.images[index].rsplit('/', 1)[1]

        if self.use_depth:
            return rgb_img, label_img, depth_img, conf_np, filename, self.reg_weight
        else:
            return rgb_img, label_img, conf_np, filename, self.reg_weight
