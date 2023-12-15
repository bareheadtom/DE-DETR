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

class TestTransformer(unittest.TestCase):
    def setUp(self):
        # 在每个测试用例之前设置
        self.d_model = 512
        self.nhead = 8
        self.singleEncoder = TransformerEncoderLayer(512, 8, dim_feedforward=1024)
    
    def test_Mlp(self):
        print("\n***test_Mlp")
        bbox_embed = MLP(512,512,4,3)
        ref_point_head = MLP(512,512,2,2)
        output_norm = torch.randn(34*34, 2, 512)
        query_pos = torch.randn(34*34, 2, 512)

        reference_points_before_sigmoid = ref_point_head(query_pos)
        tmp = bbox_embed(output_norm)
        print("reference_points_before_sigmoid",reference_points_before_sigmoid.shape,"tmp",tmp.shape)
    

    def test_forward_post(self):
        print("\n***test_forward_post")
        src = torch.randn(2, 10, 512)  # 示例输入数据
        output = self.singleEncoder.forward_post(src)
        print(output.shape)
        self.assertEqual(output.shape, src.shape)

    def test_forward_pre(self):
        print("\n***test_forward_pre")
        src = torch.randn(2, 10, 512)  # 示例输入数据
        output = self.singleEncoder.forward_pre(src)
        print(output.shape)
        self.assertEqual(output.shape, src.shape)

    def test_forward(self):
        print("\n***test_forward")
        src = torch.randn(2, 10, 512)  # 示例输入数据
        output = self.singleEncoder.forward(src)
        print(output.shape)
        self.assertEqual(output.shape, src.shape)
    
    def test_nn_MultiheadAttention(self):
        print("\n***test_nn_MultiheadAttention")
        input_dim = 128
        seq_length = 10
        num_heads = 4

        # 创建一个 nn.MultiheadAttention 实例
        multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

        # 假设有一个输入序列input_seq，形状为 (sequence_length, batch_size, input_dim)
        input_seq = torch.randn(seq_length, 2, input_dim)  # 假设 batch_size 为 1

        # 假设query、key、value都使用相同的输入序列input_seq
        # 注意：在实际应用中，query、key、value可以是不同的张量
        query = torch.randn(seq_length//2, 2, input_dim)
        key = torch.randn(seq_length, 2, input_dim)
        value = torch.randn(seq_length, 2, input_dim)

        # 进行多头注意力计算
        # 注意：在实际应用中，你可能需要对输入数据进行更多的预处理和填充操作
        attn_output, attn_weights = multihead_attn(query, key, value)

        # attn_output是注意力加权后的输出结果，形状为 (sequence_length, batch_size, embed_dim)
        # attn_weights包含了注意力权重，形状为 (batch_size, num_heads, sequence_length, sequence_length)

        print("注意力加权输出的形状:", attn_output.shape)
        print("注意力权重的形状:", attn_weights.shape)
    
    def test_transformerEncoder(self):
        print("\n***test_transformerEncoder")
        src = torch.randn(2, 10, 512)
        model = TransformerEncoder(self.singleEncoder, 6)
        output = model(src)
        print(output.shape)
        self.assertEqual(output.shape, src.shape)
    
    def test_TransformerDecoderLayer(self):
        print("\n***test_TransformerDecoderLayer")
        src = torch.randn(34*34, 2, 512) # n(wh) ,bs ,dim(c)
        tgt = torch.randn(34*34, 2, 512)  # 示例输入数据
        encoder = TransformerEncoder(self.singleEncoder, 6)
        memmory = encoder(src)
        mask = torch.randn(2,34*34)
        decoderlayer = TransformerDecoderLayer(512, 8, dim_feedforward=2048,normalize_before=False)
        out = decoderlayer(tgt, memmory,memory_key_padding_mask=mask)
        print(out.shape)
        self.assertEqual(out.shape, src.shape)
    
    def test_update_memory_with_roi(self):
        bbox_embed = MLP(512,512,4,3)
        bbox_embed = _get_clones(bbox_embed, 6)
        ref_point_head = MLP(512,512,2,2)
        print("\n***test_update_memory_with_roi")
        meta_info = {'size': torch.tensor([[719, 480],
                [608, 649]])}
        meta_info['src_size'] = (2,512,17,26)
        memory_full = torch.randn(17*26, 2, 512)
        pos_full = torch.randn(17*26, 2, 512)
        roi_bbox = torch.randn(2,300,4)

        norm = nn.LayerNorm(512)
        decoderlayer = TransformerDecoderLayer(512, 8, dim_feedforward=2048,normalize_before=False)
        decoder = TransformerDecoder(decoderlayer, 6, norm=norm,return_intermediate=True,num_feature_levels=1)
        decoder.bbox_embed = bbox_embed
        decoder.ref_point_head = ref_point_head
        ms_feats = [torch.randn(2,1024,68,102),torch.randn(2,1024,34,51),torch.randn(2,1024,17,26)]
        memory, pos = decoder.update_memory_with_roi(memory_full, pos_full, roi_bbox, meta_info, ms_feats=None)
        print("memory",memory.shape,"pos",pos.shape )
    
    def test_TransformerDecoder(self):
        print("\n***test_TransformerDecoder")
        bbox_embed = MLP(512,512,4,3)
        bbox_embed = _get_clones(bbox_embed, 6)
        ref_point_head = MLP(512,512,2,2)
        meta_info = {'size': torch.tensor([[719, 480],
                [608, 649]])}
        src = torch.randn(2,512,17,26)
        pos_embed = torch.randn(2,512,17,26)
        mask = torch.randn(2, 17,26)

        bs,c,h,w =src.shape
        meta_info['src_size'] = src.shape
        src = src.flatten(2).permute(2,0,1)# hw,bs,c
        pos_embed =pos_embed.flatten(2).permute(2,0,1)#hw, bs,c
        mask = mask.flatten(1) #bs, hw

        query_embed = torch.randn(100,512)# q,c
        query_embed = query_embed.unsqueeze(1).repeat(1,bs,1)# q,bs,c

        tgt = torch.zeros_like(query_embed)# q,bs,c


        ms_feats = [torch.randn(2,1024,68,102),torch.randn(2,1024,34,51),torch.randn(2,1024,17,26)]
        

        norm = nn.LayerNorm(512)
        encoder = TransformerEncoder(self.singleEncoder, 6)
        memory = encoder(src=src,src_key_padding_mask=mask,pos= pos_embed)
        
        decoderlayer = TransformerDecoderLayer(512, 8, dim_feedforward=2048,normalize_before=False)
        decoder = TransformerDecoder(decoderlayer, 6, norm=norm,return_intermediate=True)
        decoder.bbox_embed = bbox_embed
        decoder.ref_point_head = ref_point_head
        
        out = decoder(tgt=tgt,
                      memory=memory,
                      memory_key_padding_mask = mask,
                      pos = pos_embed,
                      query_pos=query_embed, meta_info =meta_info,ms_feats=ms_feats)
        output, output_pos = out
        print("output",output.shape,"output_pos",output_pos.shape)
        #self.assertEqual(out.shape, src.shape)
#python -m unittest test_transformer.TestTransformer.test_Transformer
    def test_Transformer(self):
        bbox_embed = MLP(512,512,4,3)
        bbox_embed = _get_clones(bbox_embed, 6)
        ref_point_head = MLP(512,512,2,2)
        print("\n***test_Transformer")
        srcs = torch.randn(2,256,20,27)
        masks = torch.randn(2,20,27)
        query_embed = torch.randn(300,256)
        pos = torch.randn(2, 256, 20, 27)
        meta_info = {'size': torch.tensor([[719, 480],
                [608, 649]])}
        transformer = Transformer(d_model=256,num_feature_levels=1)
        #transformer.decoder.ref_point_head = ref_point_head
        #transformer.decoder.bbox_embed = bbox_embed
        hs,memory,outputs_coord = transformer(src=srcs,mask=masks,query_embed=query_embed,pos_embed=pos,meta_info=meta_info)
        #print(len(out))
        #print("hs",hs.shape,"outputs_coord",outputs_coord.shape)
        print("hs,memory,outputs_coord",hs.shape,memory.shape,outputs_coord.shape)

    
        


        

       


if __name__ == '__main__':
    unittest.main()

# ***test_Mlp
# reference_points_before_sigmoid torch.Size([1156, 2, 2]) tmp torch.Size([1156, 2, 4])
# .
# ***test_TransformerDecoder
# output torch.Size([6, 100, 2, 512]) output_pos torch.Size([6, 2, 100, 4])
# .
# ***test_TransformerDecoderLayer
# torch.Size([1156, 2, 512])
# .
# ***test_forward
# torch.Size([2, 10, 512])
# .
# ***test_forward_post
# torch.Size([2, 10, 512])
# .
# ***test_forward_pre
# torch.Size([2, 10, 512])
# .
# ***test_nn_MultiheadAttention
# 注意力加权输出的形状: torch.Size([5, 2, 128])
# 注意力权重的形状: torch.Size([2, 5, 10])
# .
# ***test_transformerEncoder
# torch.Size([2, 10, 512])
# .
# ***test_update_memory_with_roi
# memory torch.Size([16, 300, 2, 512]) pos torch.Size([16, 300, 2, 512])

