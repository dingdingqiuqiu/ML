from PIL import Image
from torchvision import transforms

# 读取图像
image = Image.open('example.jpg')

# 转换为张量
transform = transforms.ToTensor()
image_tensor = transform(image)

print(f"图像张量形状: {image_tensor.shape}")

# 调整图像大小
resize_transform = transforms.Resize((256, 256))  # 将图像调整为 256x256
resized_image = resize_transform(image)

# 保存调整后的图像
resized_image.save('new.jpg')

# 转换成张量查看内容
image_tensor_new = transform(resized_image)
print(f"新图像张量: {image_tensor_new}")

# 归一化操作，通常使用 ImageNet 的均值和标准差归一化
normalize_transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和标准差
])

normalized_image = normalize_transform(resized_image)

# 输出归一化后的图像张量
print(f"归一化后的图像张量: {normalized_image}")
