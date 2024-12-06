import tensorflow as tf
from keras import models
from keras import layers

# 检查可用的设备
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置为仅使用第一块GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)  # 动态申请显存
    except RuntimeError as e:
        print(e)

# 构建Keras模型
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 继续训练模型
# model.fit(train_images, train_labels, epochs=5, batch_size=64)
