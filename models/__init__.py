# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .ms_poolers import MSROIPooler

def build_model(args):
    return build(args)
