from PIL import Image
from torchvision import transforms

# 读取图像
image = Image.open('example.jpg')
# 组合多个转换操作：调整大小 -> 转换为张量 -> 归一化
transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

processed_image = transform_pipeline(image)

# 输出处理后的图像张量
print(f"处理后的图像张量: {processed_image}")
