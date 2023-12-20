import unittest
import sys
import torch
import torchvision
sys.path.append('../../')
#from datasets.exdark import ExdarkDetection
import json
import os.path as osp
import xml.etree.ElementTree as ET
from PIL import Image
import re
import numpy as np

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
CLASSES = ('Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
               'Cup', 'Dog', 'Motorbike', 'People', 'Table')

dataset = dict()
dataset['info'] = None
dataset['licenses'] = None
dataset['images'] = []
dataset['annotations'] = []
dataset['categories'] = []
class TestTranslateToCOCO(unittest.TestCase):

    def setUp(self):
        #print("\n********setUp")
        self.img_prefix = '/root/autodl-tmp/Exdark/JPEGImages/IMGS'
        self.data_infos = self.load_data_infos('/root/autodl-tmp/Exdark/main/val.txt')
        self.cat2label = {cat: i for i, cat in enumerate(CLASSES)}
        self.min_size = None
        self.filter_empty_gt=True
        valid_inds = self._filter_imgs()
        self.data_infos = [self.data_infos[i] for i in valid_inds]
        self.data_infos = self.data_infos[:2]
    #def test_print(self):
        #print("aa")
        #js = json.load(open('/root/autodl-tmp/COCO/annotations/instances_val2017.json','r'))
        #js = json.load(open('/root/autodl-tmp/Exdark/cocoAnno/instances_exdark_train.json','r'))
        #/root/autodl-tmp/Exdark/cocoAnno
        #print("js",js)
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
                    if name in CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds
    
    def load_data_infos(self, ann_file):
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
            if name not in CLASSES:
                continue
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            # bbox = [
            #     int(float(bnd_box.find('xmin').text)),
            #     int(float(bnd_box.find('ymin').text)),
            #     int(float(bnd_box.find('xmax').text)),
            #     int(float(bnd_box.find('ymax').text))
            # ]
            bbox = [
                float(bnd_box.find('xmin').text),
                float(bnd_box.find('ymin').text),
                float(bnd_box.find('xmax').text),
                float(bnd_box.find('ymax').text)
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
        # if not bboxes:
        #     bboxes = np.zeros((0, 4))
        #     labels = np.zeros((0,))
        # else:
        #     bboxes = np.array(bboxes, ndmin=2) - 1
        #     labels = np.array(labels)
        # if not bboxes_ignore:
        #     bboxes_ignore = np.zeros((0, 4))
        #     labels_ignore = np.zeros((0,))
        # else:
        #     bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        #     labels_ignore = np.array(labels_ignore)
        ann = dict(
            #bboxes=bboxes.astype(np.float32),
            bboxes=bboxes,
            #labels=labels.astype(np.int64),
            labels=labels,
            #bboxes_ignore=bboxes_ignore.astype(np.float32),
            #labels_ignore=labels_ignore.astype(np.int64)
            )
        return ann
    
    def AddCategories(self):
        # init dataset
        #print("\n********test_AddCategories")
        for i,cat in enumerate(CLASSES):
            cat_item  = dict()
            cat_item['supercategory'] = cat
            cat_item['id'] = i
            cat_item['name'] = cat
            dataset['categories'].append(cat_item)
        #print("dataset",dataset)

    def Addimages(self):
        #print("\n********test_Addimages")
        #data_infos = self.load_data_infos('/root/autodl-tmp/Exdark/main/train.txt')
        i = 0
        for data_info in self.data_infos:
            image_item = dict()
            image_item['id'] = int(''.join(re.findall(r'\d+', data_info['id']))) 
            image_item['file_name'] = data_info['id']
            image_item['width'] = data_info['width']
            image_item['height'] = data_info['height']
            dataset['images'].append(image_item)
            i+=1
            # if i > 5:
            #     break
        #print("dataset",dataset)
    def Addannotations(self):
        #print("\n********test_annotations")
        ann_id = 0
        for i,data_info in enumerate(self.data_infos):
            ann = self.get_ann_info(i)
            for box, label in zip(ann['bboxes'], ann['labels']):
                x,y,w,h = float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])

                ann_item = dict()
                ann_item['segmentation'] = [[x,y,x,y+h,x+w,y+h,x+w,y]]
                ann_item['image_id'] = int(''.join(re.findall(r'\d+', data_info['id'])))
                ann_item['iscrowd'] = 0
                ann_item['bbox'] = [x,y,w,h]
                ann_item['category_id'] = label
                ann_item['area'] = w*h
                ann_item['id'] = ann_id

                ann_id += 1
                dataset['annotations'].append(ann_item)
            # if i > 5:
            #     break
        #print("dataset",dataset)
    
    def test_generate_ann(self):
        print("\n********test_generate_ann")
        self.AddCategories()
        self.Addimages()
        self.Addannotations()
        print("\ntotal data",dataset)
        with open('/root/autodl-tmp/projects/DE-DETRs/test/test_dataset/v1.json', 'w', encoding='UTF-8') as file:
            json.dump(dataset,file)

    # def test_generate_ann(self):
    #     print("\n********test_generate_ann")
    #     self.AddCategories()
    #     self.Addimages()
    #     self.Addannotations()
    #     #print("\ntotal data",dataset)
    #     with open('/root/autodl-tmp/Exdark/cocoAnno/instances_exdark_val.json', 'w', encoding='UTF-8') as file:
    #         json.dump(dataset,file)


if __name__ == '__main__':
    unittest.main()