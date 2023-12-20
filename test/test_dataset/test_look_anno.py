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
from pycocotools.coco import COCO
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
class TestLookAnno(unittest.TestCase):

    def test_lookann(self):
        json_path = '/root/autodl-tmp/Exdark/cocoAnno/instances_exdark_val.json'
        img_path = '/root/autodl-tmp/Exdark/JPEGImages/IMGS'
        coco = COCO(json_path)
        ids = list(sorted(coco.imgs.keys()))
        print("number of images:",len(ids))
        coco_classes = dict([(v["id"],v["name"]) for k,v in coco.cats.items()])
        print("coco_classes",coco_classes)
        for img_id in ids[:3]:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            targets = coco.loadAnns(ann_ids)
            path = coco.loadImgs(img_id)[0]['file_name']
            img = Image.open(os.path.join(img_path, path)).convert('RGB')
            draw = ImageDraw.Draw(img)
            for target in targets:
                x, y, w, h = target['bbox']
                x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
                draw.rectangle((x1, y1, x2, y2))
                draw.text((x1, y1),coco_classes[target['category_id']])
            plt.imshow(img)
            plt.savefig('/root/autodl-tmp/Exdark/cocoAnno/'+str(img_id)+'.jpg')  # 保存图像到指定路径
            plt.show()


if __name__ == '__main__':
    unittest.main()