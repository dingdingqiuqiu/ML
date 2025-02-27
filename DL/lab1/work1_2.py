import torch
# 张量加法、减法、乘法、除法
tensor_a = torch.tensor([1, 2, 3])
tensor_b = torch.tensor([4, 5, 6])

# 加法
result_add = tensor_a + tensor_b
# 减法
result_sub = tensor_a - tensor_b
# 乘法
result_mul = tensor_a * tensor_b
# 除法
result_div = tensor_a / tensor_b

print(f"加法结果: {result_add}")
print(f"减法结果: {result_sub}")
print(f"乘法结果: {result_mul}")
print(f"除法结果: {result_div}")
