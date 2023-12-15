import torch
import unittest
import sys
from torch import nn, Tensor
sys.path.append('../')
from models.transformer import TransformerEncoderLayer, TransformerEncoder,TransformerDecoderLayer,TransformerDecoder, Transformer
  # 请将"your_module"替换为实际定义TransformerEncoderLayer的模块或文件名
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                    accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from models.detr import MLP
import copy
from models.detr import DETR
from models.backbone import build_backbone,Backbone,FrozenBatchNorm2d,BackboneBase
from util.misc import nested_tensor_from_tensor_list
from models.position_encoding import build_position_encoding
from models.transformer import build_transformer

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TestTransformerEncoderLayer(unittest.TestCase):
    
        

    def test_detr(self):
        print("\n***test_detr")
        samples = NestedTensor(torch.tensor([2, 3, 649, 719]),torch.tensor([2, 649, 719]))
        meta_info = {'size': torch.tensor([[719, 480],
                [608, 649]])}
        backbone = build_backbone(backboneArgs())
        args = transformerArgs()
        transformer =build_transformer(args)
        detr = DETR(
            backbone,
            transformer,
            num_classes=91,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            box_refine=args.box_refine,
            num_feature_levels=args.num_feature_levels,
            init_ref_dim=args.init_ref_dim,
        )
        outputs = detr(samples, meta_info)
        print("outputs",outputs)

        
class backboneArgs:
    hidden_dim =512
    position_embedding = 'sine'
    lr_backbone=1e-05
    masks = False
    num_feature_levels = 3
    dilation = False
    backbone = 'resnet50'

class transformerArgs:    
    num_feature_levels = 3
    dropout=0.1
    nheads=8
    enc_layers=6
    dec_layers=6
    pre_norm=False
    ms_roi=True
    pool_res=4
    num_queries=300
    aux_loss=True
    box_refine=True
    init_ref_dim=2
    hidden_dim =512
    dim_feedforward = 2048
        

       


if __name__ == '__main__':
    unittest.main()
