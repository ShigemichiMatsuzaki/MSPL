import os
import torch
import torch.utils.data as data
from PIL import Image
import cv2
from transforms.segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose
from collections import OrderedDict
import numpy as np

GREENHOUSE_CLASS_LIST = ['end_of_plant', 'other_plant', 'artificial', 'ground']

class GreenhouseSegmentation(data.Dataset):

    def __init__(self, root, list_name, train=True, scale=(0.5, 2.0), size=(480, 264), ignore_idx=255, coarse=True):

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

class GreenhouseRGBDSegmentation(data.Dataset):

    def __init__(self, root, list_name, train=True, scale=(0.5, 2.0), size=(480, 264), ignore_idx=4, use_depth=True):

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
        '''
        Open a depth image using OpenCV instead of PIL to deal with int16 format of the image
        '''
        if self.use_depth:
            cv_depth = cv2.imread(self.depths[index], cv2.IMREAD_GRAYSCALE)
            cv_depth = cv2.medianBlur(cv_depth, 7)
            cv_depth = np.where(cv_depth < 10, cv_depth, 10) * (255 // 10)
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

#class Greenhouse(data.Dataset):
#    """Greenhouse dataset loader
#
#    Keyword arguments:
#    - root_dir (``string``): Root directory path.
#    - mode (``string``): The type of dataset: 'train' for training set, 'val'
#    for validation set, and 'test' for test set.
#    - transform (``callable``, optional): A function/transform that  takes in
#    an PIL image and returns a transformed version. Default: None.
#    - label_transform (``callable``, optional): A function/transform that takes
#    in the target and transforms it. Default: None.
#    - loader (``callable``, optional): A function to load an image given its
#    path. By default ``default_loader`` is used.
#
#    """
#    # Training dataset root folders
#    train_folder = 'train'
#    train_lbl_folder = 'trainannot'
#
#    # Validation dataset root folders
#    val_folder = 'val'
#    val_lbl_folder = 'valannot'
#
#    # Test dataset root folders
#    test_folder = 'test'
#    test_lbl_folder = 'testannot'
#
#    # Images extension
#    img_extension = '.png'
#
#    # Default encoding for pixel value, class name, and class color
#    color_encoding = OrderedDict([
#        ('end_of_plant', (0, 255, 0)),
#        ('other_part_of_plant', (0, 255, 255)),
#        ('artificial_objects', (255, 0, 0)),
#        ('ground', (255, 255, 0)),
#        ('background', (0, 0, 0))
#    ])
#
#    def __init__(self,
#                 root_dir,
#                 mode='train',
#                 transform=None,
#                 label_transform=None,
#                 loader=utils.pil_loader):
#        self.root_dir = root_dir
#        self.mode = mode
#        self.transform = transform
#        self.label_transform = label_transform
#        self.loader = loader
#
#        if self.mode.lower() == 'train':
#            # Get the training data and labels filepaths
#            self.train_data = utils.get_files(
#                os.path.join(root_dir, self.train_folder),
#                extension_filter=self.img_extension)
#
#            self.train_labels = utils.get_files(
#                os.path.join(root_dir, self.train_lbl_folder),
#                extension_filter=self.img_extension)
#        elif self.mode.lower() == 'val':
#            # Get the validation data and labels filepaths
#            self.val_data = utils.get_files(
#                os.path.join(root_dir, self.val_folder),
#                extension_filter=self.img_extension)
#
#            self.val_labels = utils.get_files(
#                os.path.join(root_dir, self.val_lbl_folder),
#                extension_filter=self.img_extension)
#        elif self.mode.lower() == 'test':
#            # Get the test data and labels filepaths
#            self.test_data = utils.get_files(
#                os.path.join(root_dir, self.test_folder),
#                extension_filter=self.img_extension)
#
#            self.test_labels = utils.get_files(
#                os.path.join(root_dir, self.test_lbl_folder),
#                extension_filter=self.img_extension)
#        else:
#            raise RuntimeError("Unexpected dataset mode. "
#                               "Supported modes are: train, val and test")
#
#    def __getitem__(self, index):
#        """
#        Args:
#        - index (``int``): index of the item in the dataset
#
#        Returns:
#        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
#        of the image.
#
#        """
#        if self.mode.lower() == 'train':
#            data_path, label_path = self.train_data[index], self.train_labels[
#                index]
#        elif self.mode.lower() == 'val':
#            data_path, label_path = self.val_data[index], self.val_labels[
#                index]
#        elif self.mode.lower() == 'test':
#            data_path, label_path = self.test_data[index], self.test_labels[
#                index]
#        else:
#            raise RuntimeError("Unexpected dataset mode. "
#                               "Supported modes are: train, val and test")
#
#        img, label = self.loader(data_path, label_path)
#
#        if self.transform is not None:
#            img = self.transform(img)
#
#        if self.label_transform is not None:
#            label = self.label_transform(label)
#
#        return img, label
#
#    def __len__(self):
#        """Returns the length of the dataset."""
#        if self.mode.lower() == 'train':
#            return len(self.train_data)
#        elif self.mode.lower() == 'val':
#            return len(self.val_data)
#        elif self.mode.lower() == 'test':
#            return len(self.test_data)
#        else:
#            raise RuntimeError("Unexpected dataset mode. "
#                               "Supported modes are: train, val and test")
