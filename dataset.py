from torchvision import transforms
from PIL import Image
import os
import torch
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np

import PIL.Image as PIL_Image

from torchvision.datasets.folder import default_loader
from collections import OrderedDict

from utils.load_dataset import DatasetSplit
from torchvision.transforms.functional import InterpolationMode
from glob import glob
import json
import sys
import cv2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OS_NAME = sys.platform
SEP = '\\' if OS_NAME == "win32" else '/'


def get_data_transforms(size, isize):
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor(),
    ])
    return data_transforms, gt_transforms


# From https://github.com/zhangzilongc/MMR
class AeBADDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            source,
            classname,
            domain_shift_category,
            resize=256,
            imagesize=224,
            split=DatasetSplit.TRAIN,
            **kwargs
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname]
        self.domain_shift_category = domain_shift_category
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        # for test
        self.transform_img = [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor()
        ]
        self.transform_img.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

        self.transform_img = transforms.Compose(self.transform_img)

        # for train
        self.transform_img_MMR = transforms.Compose([
            transforms.RandomResizedCrop(imagesize,
                                         scale=(0.7,
                                                1.),
                                         interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

        transform_mask = [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]

        image = default_loader(image_path)
        if self.split.value == "train":
            image = self.transform_img_MMR(image)
        else:
            image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL_Image.open(mask_path)
            mask = self.transform_mask(mask)
            # avoid erasing the abnormal mask via center crop
            if torch.max(mask) == 0:
                mask = torch.zeros([1, *image.size()[1:]])
                anomaly = "good"
            else:
                mask = mask / torch.max(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        if self.split.value == "train":
            return image, int(anomaly != "good")
        else:
            return image, mask, int(anomaly != "good"), image_path

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = [i for i in os.listdir(classpath)
                             if os.path.isdir(os.path.join(classpath, i))]

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                if os.path.isdir(anomaly_path):
                    anomaly_files = sorted(os.listdir(anomaly_path))
                    imgpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_path, x) for x in anomaly_files
                    ]

                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        anomaly_mask_path = os.path.join(maskpath, anomaly)

                        # use the filename in anomaly file
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, x.split(".")[0] + "_mask.png") for x in anomaly_files
                        ]
                    else:
                        maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


class AeBAD_SDataset(AeBADDataset):
    """
    Demonstration for domain shift setups in AeBAD dataset:
    1. same is without domain shift setups.
    2. The categories include background, illumination and view.
    For more details, please read [Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction](https://arxiv.org/abs/2304.02216).
    """

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = [i for i in os.listdir(classpath)
                             if os.path.isdir(os.path.join(classpath, i))]

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                imgpaths_per_class[classname][anomaly] = []

                if self.split.value == "train" and anomaly == "good":
                    sub_types = [i for i in os.listdir(anomaly_path)
                                 if os.path.isdir(os.path.join(anomaly_path, i))]
                    for sub_good_train in sub_types:
                        imgpaths_per_class = png_load(anomaly_path=anomaly_path,
                                                      sub_good_train=sub_good_train,
                                                      imgpaths_per_class=imgpaths_per_class,
                                                      classname=classname,
                                                      anomaly=anomaly)
                else:
                    # for test mode
                    imgpaths_per_class = png_load(anomaly_path=anomaly_path,
                                                  sub_good_train=self.domain_shift_category,
                                                  imgpaths_per_class=imgpaths_per_class,
                                                  classname=classname,
                                                  anomaly=anomaly)

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    temp_imgpaths_per_class = imgpaths_per_class[classname][anomaly]
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, self.domain_shift_category, x.split("/")[-1]) for x
                        in temp_imgpaths_per_class
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


class AeBAD_VDataset(AeBADDataset):

    def get_image_data(self):
        imgpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            imgpaths_per_class[classname] = {}
            if self.split == DatasetSplit.TRAIN:
                anomaly_types = ["good"]

                for anomaly in anomaly_types:
                    anomaly_path = os.path.join(classpath, anomaly)
                    if os.path.isdir(anomaly_path):
                        crucial_word = "*/*.jpg"
                        anomaly_files = glob(os.path.join(anomaly_path, crucial_word))

                        imgpaths_per_class[classname][anomaly] = anomaly_files
            else:
                video_types = [self.domain_shift_category]

                for video_name in video_types:
                    video_path = os.path.join(classpath, video_name)
                    if os.path.isdir(video_path):
                        anomaly_types = [i for i in os.listdir(video_path)
                                         if os.path.isdir(os.path.join(video_path, i))]
                        for anomaly in anomaly_types:
                            anomaly_path = os.path.join(video_path, anomaly)
                            if os.path.isdir(anomaly_path):
                                crucial_word = "*.jpg"
                                anomaly_files = glob(os.path.join(anomaly_path, crucial_word))

                                imgpaths_per_class[classname][anomaly] = anomaly_files

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


def png_load(anomaly_path,
             sub_good_train,
             imgpaths_per_class,
             classname,
             anomaly):
    specific_anomaly_path = os.path.join(anomaly_path, sub_good_train)
    anomaly_files = glob(os.path.join(specific_anomaly_path, "*.png"))
    imgpaths_per_class[classname][anomaly].extend(anomaly_files)
    return imgpaths_per_class


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob(os.path.join(self.img_path, defect_type) + "/*.jpg") + \
                            glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_paths = list(set(img_paths))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob(os.path.join(self.img_path, defect_type) + "/*.jpg") + \
                            glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths = list(set(img_paths))
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))
                assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, item, transform, gt_transform, phase):
        self.root = root
        self.phase = phase
        if phase == 'test':
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.item = item
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        meta_path = os.path.join(self.root, 'realiad_jsons', 'realiad_jsons', self.item + '.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        meta = meta[self.phase]
        img_tot_paths = [os.path.join(self.root, 'realiad_512', self.item, path['image_path']) for path in meta]
        gt_tot_paths = [
            os.path.join(self.root, 'realiad_512', self.item, path['mask_path']) if path['mask_path'] is not None else 0
            for path
            in
            meta]
        tot_labels = [item['anomaly_class'] != "OK" for item in meta]
        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


def np_load_frame(filename, resize_height, resize_width, c):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    if c == 1:
        image_decoded = np.repeat(cv2.imread(filename, 0)[:, :, None], 3, axis=-1)
    else:
        image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized
