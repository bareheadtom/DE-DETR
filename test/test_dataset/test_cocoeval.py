import unittest
import sys
import torch
import torchvision
sys.path.append('../../')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import random
import numpy as np
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import numpy as np
from terminaltables import AsciiTable
class TestCOCOeval(unittest.TestCase):
    def t1est_cocoeval(self):
        print("\n****************test_cocoeval")
        coco_true = COCO(annotation_file='/root/autodl-tmp/COCO/annotations/instances_val2017.json')
        coco_pre = coco_true.loadRes('/root/autodl-tmp/projects/DE-DETRs/test/test_dataset/predict_results.json')
        coco_eva = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
        coco_eva.evaluate()
        coco_eva.accumulate()
        print(coco_eva.eval['precision'])
        coco_eva.summarize()
        print(coco_eva.eval['precision'])
    
    def test_mycocoeval(self):
        print("\n****************test_mycocoeval")
        coco_true = COCO(annotation_file='/root/autodl-tmp/projects/DE-DETRs/test/test_dataset/v1.json')
        coco_pre = coco_true.loadRes('/root/autodl-tmp/projects/DE-DETRs/test/test_dataset/v2_score.json')

        # coco_true = COCO(annotation_file='/root/autodl-tmp/projects/DE-DETRs/test/test_dataset/v1.json')
        # coco_pre = COCO(annotation_file='/root/autodl-tmp/projects/DE-DETRs/test/test_dataset/v2.json')
        coco_eva = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
        print("params imgIds",coco_eva.params.imgIds)
        print("params catIds",coco_eva.params.catIds)
        print("p.iouTyoe",coco_eva.params.iouType)
        print("p.maxDets",coco_eva.params.maxDets)
        coco_eva.evaluate()
        coco_eva.accumulate()
        #print(coco_eva.eval['precision'])
        coco_eva.summarize()
        #print(coco_eva.eval['precision'])
        precisions = coco_eva.eval['precision']
        #print("precisions",precisions.shape)
        #precisions (10, 101, 12, 4, 3)
        # precision: (iou, recall, cls, area range, max dets)
        self.cat_ids = sorted(coco_true.getCatIds())
        assert len(self.cat_ids) == precisions.shape[2]

        results_per_category = []
        for idx, catId in enumerate(self.cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_true.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (f'{nm["name"]}', f'{float(ap):0.3f}'))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(
            itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(*[
            results_flatten[i::num_columns]
            for i in range(num_columns)
        ])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print('\n' + table.table)
        #print_log('\n' + table.table, logger=logger)
    
    def t1est_addscore(self):
        print("\n****************test_addscore")
        js = json.load(open('/root/autodl-tmp/projects/DE-DETRs/test/test_dataset/v2.json','r'))
        anns = js['annotations']
        print(anns)
        for ann in anns:
            ann['score'] = random.uniform(60, 100)
        with open('/root/autodl-tmp/projects/DE-DETRs/test/test_dataset/v2_score.json', 'w', encoding='UTF-8') as file:
            json.dump(anns,file)

    # def test_lookval(self):
    #     js = json.load(open('/root/autodl-tmp/COCO/annotations/instances_val2017.json','r'))
    #     print("js",js)




if __name__ == '__main__':
    unittest.main()