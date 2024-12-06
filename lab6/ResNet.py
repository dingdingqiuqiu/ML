import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

def ResNet_block(X, filters):
    # 第一个卷积层
    X_shortcut = X
    X = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    # 第二个卷积层
    X = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    # 保持与X通道数一致
    X = layers.Conv2D(32, kernel_size=(1, 1))(X)
    # 添加跳跃连接
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    return X

#构建网络
input_shape=(32, 32, 3)#输入形状
# 输入层
X_input = tf.keras.Input(shape=input_shape)
# 初始卷积层
X = layers.Conv2D(32, (3, 3), padding='same')(X_input)
X = layers.BatchNormalization()(X)
X = layers.Activation('relu')(X)
# 添加一个残差块
X = ResNet_block(X, 64)
# 平均池化层
X = layers.AveragePooling2D((2, 2), padding='same')(X)
# 展平层
X = layers.Flatten()(X)
# 全连接层
X = layers.Dense(10, activation='softmax')(X)
# 创建模型
model = Model(inputs=X_input, outputs=X)

#加载数据集
from keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

#train_images = train_images.reshape((60000, 32, 32, 3))
train_images = train_images.astype('float32') / 255

#test_images = test_images.reshape((10000, 32, 32, 3))
test_images = test_images.astype('float32') / 255

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('\nTest accuracy:', test_acc)
