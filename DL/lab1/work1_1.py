import torch

# 创建不同维度的张量
tensor_1d = torch.tensor([1, 2, 3])
print(tensor_1d.device)  # 输出: cpu
tensor_1d = torch.tensor([1, 2, 3],device='cuda')
tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]],device='cpu')
tensor_3d = torch.tensor([[[1.0, 2], [3, 4]], [[5, 6], [7, 8]]],device='cuda')
# 打印形状和数据类型
print(f"1D张量形状: {tensor_1d.shape}, 数据类型: {tensor_1d.dtype}")
print(f"2D张量形状: {tensor_2d.shape}, 数据类型: {tensor_2d.dtype}")
print(f"3D张量形状: {tensor_3d.shape}, 数据类型: {tensor_3d.dtype}")
