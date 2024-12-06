import tensorflow as tf
from tensorflow.keras import layers, models
from keras.datasets import cifar10

# 加载CIFAR-10数据集
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 数据归一化到[0, 1]区间
train_images, test_images = train_images / 255.0, test_images / 255.0

# 定义ResNet块
def resnet_block(X, filters):
    # 保存输入，作为跳跃连接部分
    X_shortcut = X
    
    # 第一个卷积层
    X = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    
    # 第二个卷积层
    X = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    
    # 如果输入和输出通道数不匹配，使用1x1卷积调整
    if X_shortcut.shape[-1] != X.shape[-1]:
        X_shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding='same')(X_shortcut)
    
    # 添加跳跃连接（残差连接）
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    
    return X

# 定义Inception模块
def inception_module(x):
    # 分支1：1x1卷积
    branch1 = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    
    # 分支2：1x1卷积 -> 3x3卷积
    branch2 = layers.Conv2D(16, (1, 1), activation='relu', padding='same')(x)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    
    # 分支3：1x1卷积 -> 5x5卷积
    branch3 = layers.Conv2D(8, (1, 1), activation='relu', padding='same')(x)
    branch3 = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(branch3)
    
    # 分支4：最大池化 -> 1x1卷积
    branch4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch4 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(branch4)
    
    # 将所有分支的输出连接在一起
    x = layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)
    
    return x

# 构建模型
def build_model(input_shape):
    X_input = tf.keras.Input(shape=input_shape)  # 输入层
    
    # 初始卷积层
    X = layers.Conv2D(32, (3, 3), padding='same')(X_input)
    X = layers.BatchNormalization()(X)  # 批归一化
    X = layers.Activation('relu')(X)  # ReLU激活函数
    
    # 添加Inception模块
    X = inception_module(X)
    
    # 添加一个残差块（ResNet）
    X = resnet_block(X, 64)
    
    # 平均池化层
    X = layers.AveragePooling2D((2, 2), padding='same')(X)
    
    # 展平层
    X = layers.Flatten()(X)
    
    # 全连接层，输出10个类别
    X = layers.Dense(10, activation='softmax', name='fc_out')(X)
    
    # 创建模型
    model = models.Model(inputs=X_input, outputs=X)
    return model

# 输入数据形状 (32, 32, 3)
input_shape = (32, 32, 3)

# 构建模型
model = build_model(input_shape)

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # 多分类交叉熵损失函数
              metrics=['accuracy'])  # 评估准确率

# 打印模型概况
model.summary()

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"测试集准确率: {test_acc:.4f}")
