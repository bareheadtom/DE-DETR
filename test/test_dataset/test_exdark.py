import unittest
import sys
import torch
import torchvision
sys.path.append('../../')
from datasets.exdark import ExdarkDetection

class TestBuildBackbone(unittest.TestCase):
    def test_exdarkgetitem(self):
        print("\n****************test_exdarkgetitem")
        exdark = ExdarkDetection(img_prefix='/root/autodl-tmp/Exdark/JPEGImages/IMGS',
                                 ann_file= '/root/autodl-tmp/Exdark/main/train.txt',
                                 transforms=None)
        img,target = exdark[2]
        print("img",img,"target",target)



if __name__ == '__main__':
    unittest.main()