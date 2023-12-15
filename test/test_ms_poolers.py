import unittest
import torch
import sys
sys.path.append('../')
from models.ms_poolers import MSROIPooler  # 替换为你的模块名称
from detectron2.structures import Boxes

class TestMSROIPooler(unittest.TestCase):
    def test_forward(self):
        # 构造测试数据
        output_size = [4, 8, 16, 32]
        scales = [0.25, 0.125, 0.0625, 0.03125]
        sampling_ratio = 2
        pooler_type = "ROIAlign"
        canonical_box_size = 224
        canonical_level = 4

        # 创建 MSROIPooler 实例
        pooler = MSROIPooler(
            output_size=output_size,
            scales=scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_box_size=canonical_box_size,
            canonical_level=canonical_level,
        )

        # 构造模拟数据
        batch_size = 2
        num_channels = 3
        dummy_features = [
            torch.randn(batch_size, num_channels, 64, 64),
            torch.randn(batch_size, num_channels, 32, 32),
            torch.randn(batch_size, num_channels, 16, 16),
            torch.randn(batch_size, num_channels, 8, 8),
        ]

        dummy_boxes = [
            Boxes(torch.tensor([[0, 0, 100, 100], [50, 50, 200, 200],[0, 0, 100, 100], [50, 50, 200, 200]])),
            Boxes(torch.tensor([[0, 0, 100, 100], [50, 50, 200, 200],[90, 90, 120, 120]]))
        ]

        # 执行 forward 方法
        output = pooler(dummy_features, dummy_boxes)
        print(len(output))
        for level_output in output:
            print(level_output.shape)
        # 检查输出的类型和形状是否符合预期
        # self.assertIsInstance(output, list)
        # for level_output in output:
        #     self.assertIsInstance(level_output, torch.Tensor)
        #     self.assertEqual(
        #         level_output.shape,
        #         (batch_size * len(dummy_boxes[0]), num_channels, output_size[0], output_size[0]),
        #     )

if __name__ == "__main__":
    unittest.main()
