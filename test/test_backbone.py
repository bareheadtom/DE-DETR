import unittest
import sys
import torch
import torchvision
sys.path.append('../')
from models.backbone import build_backbone,Backbone,FrozenBatchNorm2d,BackboneBase
from util.misc import nested_tensor_from_tensor_list
from models.position_encoding import build_position_encoding


class TestBuildBackbone(unittest.TestCase):

    # def test_build_position_encoding(self):
    #     #编码的是每个token的embeding表示的内容里面的
    #     print("\n***test_build_position_encoding")
    #     class Args:
    #         backbone = 'resnet50'  # 替换为你希望的值
    #         lr_backbone = 1e-5  # 替换为你希望的值
    #         masks = True  # 替换为你希望的值
    #         dilation = False  # 替换为你希望的值
    #         hidden_dim = 256
    #         position_embedding = 'sine'

    #     args = Args()
    #     # dim_t = torch.arange(20, dtype=torch.float32)
    #     # print("dim_t",dim_t)
    #     # dim_t = dim_t // 2
    #     # print("dim_t",dim_t)
    #     # dim_t = 10000 ** (2 * (dim_t // 2) / 20)
    #     # print("dim_t",dim_t)
    #     #input_data = nested_tensor_from_tensor_list([torch.rand( 4, 4), torch.ones( 4, 4)])

    #     not_mask = torch.ones((4,4), dtype=torch.bool)
    #     y_embed = not_mask.cumsum(0, dtype=torch.float32)
    #     x_embed = not_mask.cumsum(1, dtype=torch.float32)
    #     print("print(x_embed,y_embed)",x_embed,y_embed)
    #     print("x_embed[:,:,0::2], x_embed[:,:,1::2]",x_embed[:,0::2], x_embed[:,1::2])

    #     dim_t = torch.arange(10, dtype = torch.float32)
    #     print("dim_t",dim_t)
    #     dim_t =2*(dim_t //2)/10
    #     print("dim_t",dim_t)

    #     pos_x = x_embed[:,:,None] / dim_t
    #     pos_y = y_embed[:,:,None] / dim_t
    #     print("pos_x",pos_x.shape,pos_x)
    #     t = torch.randint(10,(2,6))
    #     print("torch.arange(10)[0::2]",t,t[:,0::2])
    #     pos_x = torch.stack((pos_x[:,:,0::2],pos_x[:,:,1::2]), dim = 3)
    #     pos_y = torch.stack((pos_y[:,:,0::2],pos_y[:,:,1::2]), dim = 3)
    #     print("pisx,posy",pos_x.shape,pos_y.shape,pos_x, pos_y)
    #     pos_x = pos_x.flatten(2)
    #     pos_y = pos_y.flatten(2)
    #     print("pisx,posy",pos_x.shape, pos_y.shape,pos_x, pos_y)
    #     t  =torch.cat((pos_y, pos_x), dim=2)
    #     print("torch.cat((pos_y, pos_x), dim=2)",t.shape, t)
    #     pos = torch.cat((pos_y, pos_x), dim=2).permute(2,0,1)
    #     print("pos",pos.shape,pos)
    #     print("end")
    #     input_data = nested_tensor_from_tensor_list([torch.rand(3, 4, 8), torch.ones(3, 4, 8)])
    #     position_embedding = build_position_encoding(args)
    #     out = position_embedding(input_data)
    #     print("out.shape",out.shape)

    def test_resnet(self):
        print("\n***test_resnet")
        class Args:
            backbone = 'resnet50'  # 替换为你希望的值
            lr_backbone = 1e-5  # 替换为你希望的值
            masks = True  # 替换为你希望的值
            dilation = False  # 替换为你希望的值
            hidden_dim = 256
            position_embedding = 'sine'
            num_feature_levels = 3

        args = Args()
        backboneModule = getattr(torchvision.models, args.backbone)(
            replace_stride_with_dilation=[False,False,args.dilation],
            pretrained = False, norm_layer = FrozenBatchNorm2d)
        #print("backboneModule",backboneModule)
        baseBackbone = BackboneBase(backboneModule, args.lr_backbone > 0, args.masks)
        input_data = nested_tensor_from_tensor_list([torch.rand(3, 768, 1024), torch.rand(3, 768, 1024)])
        out = baseBackbone(input_data)
        #print("out",out)
        for name, x in out.items():
            src, mask = x.decompose()
            print("name",name,"src.shape",src.shape,"mask",mask.shape)
    
    def test_resnetsingle(self):
        print("\n***test_resnetsingle")
        class Args:
            backbone = 'resnet50'  # 替换为你希望的值
            lr_backbone = 1e-5  # 替换为你希望的值
            masks = False  # 替换为你希望的值
            dilation = False  # 替换为你希望的值
            hidden_dim = 256
            position_embedding = 'sine'
            num_feature_levels = 1

        args = Args()
        backboneModule = getattr(torchvision.models, args.backbone)(
            replace_stride_with_dilation=[False,False,args.dilation],
            pretrained = False, norm_layer = FrozenBatchNorm2d)
        #print("backboneModule",backboneModule)
        baseBackbone = BackboneBase(backboneModule, args.lr_backbone > 0, args.masks)
        input_data = nested_tensor_from_tensor_list([torch.rand(3, 768, 1024), torch.rand(3, 768, 1024)])
        out = baseBackbone(input_data)
        #print("out",out)
        for name, x in out.items():
            src, mask = x.decompose()
            print("name",name,"src.shape",src.shape,"mask",mask.shape)

    def test_build_backbone(self):
        print("\n***test_build_backbone")
        device = torch.device('cuda')
        # 模拟参数
        class Args:
            backbone = 'resnet50'  # 替换为你希望的值
            lr_backbone = 1e-5  # 替换为你希望的值
            masks = False  # 替换为你希望的值
            dilation = False  # 替换为你希望的值
            hidden_dim = 256
            position_embedding = 'sine'
            num_feature_levels = 3

        args = Args()

        # 调用函数
        
        backboneAndPositionEmbeding = build_backbone(args)

        #input_data = nested_tensor_from_tensor_list([torch.rand(2, 3, 768, 1024), torch.rand(2, 768, 1024)])
        input_data = nested_tensor_from_tensor_list([torch.rand(3, 768, 1024), torch.rand(3, 768, 1024)])

        #input_data = input_data.to(device)
        #input_data = [torch.rand(2, 3, 768, 1024), torch.rand(2, 768, 1024)]

        #print(model)
        features, pos = backboneAndPositionEmbeding(input_data)
        #print("out",out)
        print("type(features),type(pos)",type(features),len(features),type(pos),len(pos))

        for feature in features:
            src, mask = feature.decompose()
            print("src",src.shape,"mask",mask.shape)
        for po in pos:
            print("pos",po.shape)

        #src, mask = features[-1].decompose()
        #print("src, mask",type(src),type(mask))
        #print("src.shape, mask.shape",src.shape, mask.shape)
        #out = model(input_data)
        #print(out)


if __name__ == '__main__':
    unittest.main()
