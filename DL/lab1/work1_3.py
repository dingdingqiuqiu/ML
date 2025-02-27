import torch

tensor = torch.tensor([1, 2, 3, 4, 5, 6])

# 改变张量形状
reshaped_tensor = tensor.reshape(2, 3)  # 变为2行3列
print(f"改变形状后的张量: \n{reshaped_tensor}")
