# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
import os.path as osp
from PIL import Image

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import numpy as np
from io import BytesIO
import re
# modified from mmdetection.mmdet.datasets.RepeatDataset
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len
def list_from_file(filename, prefix='', offset=0, max_num=0, encoding='utf-8'):
    cnt = 0
    item_list = []
    with open(filename, 'r', encoding=encoding) as file:
        for _ in range(offset):
            file.readline()
        for line in file:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list

class ExdarkDetection(Dataset):

    CLASSES = ('Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
               'Cup', 'Dog', 'Motorbike', 'People', 'Table')
    def __init__(self,img_prefix, ann_file, transforms,min_size=None, cache_mode=False, local_rank=0, local_size=1,filter_empty_gt=True):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.min_size = min_size
        self.cache_mode = cache_mode
        self.transforms = transforms
        self.filter_empty_gt = filter_empty_gt
        self.data_infos = self.load_annotations(self.ann_file)
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        if cache_mode:
            self.cache = {}
            #self.cache_images()
        valid_inds = self._filter_imgs()
        self.data_infos = [self.data_infos[i] for i in valid_inds]
    
    def load_annotations(self, ann_file):
        #print("suun load_annotations")
        """Load annotation from XML style ann_file.
        Args:
            ann_file (str): Path of XML file. (txt format)
        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = list_from_file(ann_file)
        #print(" mmcv.list_from_file img_ids",img_ids)
        for img_id in img_ids:
            # print('0000', self.img_prefix)
            # print('1111', img_id)
            filename = img_id
            xml_path = osp.join(self.img_prefix.replace('JPEGImages/IMGS','Annotations/LABLE'),
                                f'{img_id}.xml')
            #print("xml_path",xml_path)
            # print(xml_path)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = 0
            height = 0
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, '{}'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))

        return data_infos
    
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:  # True
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix.replace('JPEGImages/IMGS','Annotations/LABLE'),
                                    f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in self.CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds
    
    def get_ann_info(self, idx):
        """Get annotation from XML file by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix.replace('JPEGImages/IMGS','Annotations/LABLE'),
                            f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
    
    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join("", path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join("", path)).convert('RGB')
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        img_info = self.data_infos[idx]
        #print('111', img_info)
        ann_info = self.get_ann_info(idx)
        #print("img_info",img_info,"ann_info",ann_info)
        filename = osp.join(self.img_prefix,
                                img_info['filename'])
        img = self.get_image(filename)
        target = {
            'boxes':torch.from_numpy(ann_info['bboxes']),
            'labels':torch.from_numpy(ann_info['labels']),
            'image_id':torch.tensor(int(''.join(re.findall(r'\d+', img_info['id'])))),
            'orig_size':torch.tensor([img_info['width'],img_info['height']]),
            'size':torch.tensor([img_info['width'],img_info['height']])
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    
    def __len__(self):
        return len(self.data_infos)

    # def __getitem__(self, idx):
    #     img, target = super(Dataset, self).__getitem__(idx)
    #     image_id = self.ids[idx]
    #     target = {'image_id': image_id, 'annotations': target}
    #     img, target = self.prepare(img, target)
    #     if self._transforms is not None:
    #         img, target = self._transforms(img, target)
    #     return img, target

def make_train_transforms():

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

def make_test_transforms():

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])



