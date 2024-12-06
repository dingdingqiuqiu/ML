import tensorflow as tf
from tensorflow.keras import layers, models
from keras.datasets import cifar10

# 启用内存增长tf.config.experimental.set_memory_growth(physical_devices[0], True)
# 加载 CIFAR-10 数据集
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 数据归一化到 [0, 1] 区间
train_images, test_images = train_images / 255.0, test_images / 255.0

# 定义 Dense Block
def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        # 保存输入，用于连接
        x1 = x
        
        # 1x1卷积
        x1 = layers.Conv2D(4 * growth_rate, (1, 1), padding='same', activation='relu')(x1)
        
        # 3x3卷积
        x1 = layers.Conv2D(growth_rate, (3, 3), padding='same', activation='relu')(x1)
        
        # 将新层的输出与输入 x 拼接
        x = layers.concatenate([x, x1], axis=-1)
        
    return x

# 定义过渡层（用于降低特征图的尺寸和通道数）
def transition_layer(x, reduction=0.5):
    # 使用 1x1 卷积来减少通道数
    num_filters = int(x.shape[-1] * reduction)
    x = layers.Conv2D(num_filters, (1, 1), padding='same', activation='relu')(x)
    
    # 2x2 平均池化
    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    
    return x

# 构建 DenseNet 模型
def build_densenet(input_shape, num_blocks=3, num_layers_per_block=6, growth_rate=32):
    # 输入层
    X_input = tf.keras.Input(shape=input_shape)
    
    # 初始卷积层
    X = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(X_input)
    
    # Dense Blocks 和 Transition Layers
    for _ in range(num_blocks):
        # 添加 Dense Block
        X = dense_block(X, num_layers_per_block, growth_rate)
        
        # 添加 Transition Layer
        X = transition_layer(X)
    
    # 展平层
    X = layers.Flatten()(X)
    
    # 全连接层，输出 10 个类别
    X = layers.Dense(10, activation='softmax')(X)
    
    # 创建模型
    model = models.Model(inputs=X_input, outputs=X)
    return model

# 输入数据形状 (32, 32, 3)
input_shape = (32, 32, 3)

# 构建 DenseNet 模型
model = build_densenet(input_shape)

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # 多分类交叉熵损失函数
              metrics=['accuracy'])  # 评估准确率

# 打印模型概况
model.summary()

# 减少批量大小以避免内存溢出
batch_size = 32

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=batch_size, validation_data=(test_images, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"测试集准确率: {test_acc:.4f}")
