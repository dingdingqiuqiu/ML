from keras.datasets import mnist
(train_images, train_labels), (test_images,test_labels) = mnist.load_data()

# 构建网络模型
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

# 对数据进行处理，将像素值缩放到0-1之间
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((60000, 28*28))
test_images = test_images.astype('float32') / 255

# 对标签进行one-hot编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 一共三层卷积计算
# 两层池化计算
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2))) # 最大池化层，池化尺寸2*2
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.Flatten()) # 卷积特征拉平
model.add(layers.Dense(64,activation='relu')) # 加入全连接层，节点数64
model.add(layers.Dense(10,activation='softmax')) # 输出层，节点数10,激活函数softmax

model.summary()

# 编译，拟合网络
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5,batch_size=64)

# 在测试机上评估
test_loss, test_acc=model.evaluate(test_images, test_labels)

# 精度
print(test_acc)
