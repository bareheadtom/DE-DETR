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
import torch
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层
        self.fc1 = nn.Linear(16 * 16 * 16, 128)  # 这里输入维度根据你的输入大小调整
        self.fc2 = nn.Linear(128, 10)  # 输出维度为 10，可以根据需要调整

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)  # 将张量展平以传递给全连接层
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleCNN2(nn.Module):
    def __init__(self):
        super(SimpleCNN2, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层
        self.fc11 = nn.Linear(16 * 16 * 16, 128)  # 这里输入维度根据你的输入大小调整
        self.fc2 = nn.Linear(128, 120)  # 输出维度为 10，可以根据需要调整

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)  # 将张量展平以传递给全连接层
        x = nn.functional.relu(self.fc11(x))
        x = self.fc2(x)
        return x
    
class TestCOCOeval(unittest.TestCase):
    def test_cocoeval(self):
        input_tensor = torch.randn(1, 3, 32, 32)
        simp1 = SimpleCNN()
        simp2 = SimpleCNN2()
        print("\nsimp1",simp1(input_tensor).shape, "simp2",simp2(input_tensor).shape)
        pre_trained = simp1.state_dict()
        model_stat = simp2.state_dict()
        pre_trained_new = {}
        pre_trained_miss = {}
        for k,v in pre_trained.items():
            if k in model_stat and model_stat[k].shape == v.shape:
                pre_trained_new[k] = v
            else:
                pre_trained_miss[k] = v
        model_stat.update(pre_trained_new)
        print("\nloaded",pre_trained_new)
        print("\nmissed", pre_trained_miss)




if __name__ == '__main__':
    unittest.main()