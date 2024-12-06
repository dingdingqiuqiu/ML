# 导入数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
from keras import layers

# 读取数据
data = pd.read_csv('nihe.csv')
data = np.array(data)

# 划分训练集和测试集合
m = data.shape[0]
n = data.shape[1]
test_split = 0.9
train_data = data[0:int(test_split*m), 0:(n-1)]
train_target = data[0:int(test_split*m), (n-1)]
test_data = data[int(test_split*m):, 0:(n-1)]
test_target = data[int(test_split*m):, (n-1)]

# 对数据进行标准化
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

# 构建网络模型
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1))

# 查看模型的参数量
model.summary()

# 编译网络
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# 拟合训练网络
history = model.fit(train_data, train_target, epochs=100, validation_split=0.2, batch_size=32)

# 画出训练集和测试集损失曲线
mae = history.history['mae']
val_mae = history.history['val_mae']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(mae) + 1)
plt.plot(epochs, loss, label="Training loss")
plt.plot(epochs, val_loss, 'r', label="Validation loss")
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("111.png")
plt.close()

# 在测试集上画出预测值和实际值曲线
predictions = model.predict(test_data)

# 反标准化预测值
predictions = predictions * std[-1] + mean[-1]
test_target = test_target * std[-1] + mean[-1]  # 反标准化实际值

samples = range(1, len(test_data) + 1)
plt.plot(samples, test_target, 'b', label='Ground Truth')
plt.plot(samples, predictions, 'r', label="Prediction Value")
plt.title('Ground Truth vs Prediction Value')
plt.xlabel('Samples')
plt.ylabel('Value')
plt.legend()
plt.savefig("222.png")
plt.close()
