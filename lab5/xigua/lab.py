import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

# 设置随机种子
seed = 2020
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
plt.close('all')

# (1) 数据预处理
def preprocess(data):
    # 将非数映射数字
    for title in data.columns:
        if data[title].dtype == 'object':
            encoder = LabelEncoder()
            data[title] = encoder.fit_transform(data[title])
    # 去均值和方差归一化
    ss = StandardScaler()
    X = data.drop('好瓜', axis=1)
    Y = data['好瓜']
    X = ss.fit_transform(X)
    x, y = np.array(X), np.array(Y).reshape(Y.shape[0], 1)
    return x, y

# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 求导
def d_sigmoid(x):
    return x * (1 - x)

# (2) 标准 BP 算法
def standard_BP(x, y, dim=10, eta=0.8, max_iter=500):
    n_samples = x.shape[0]
    w1 = np.random.random((x.shape[1], dim))
    w2 = np.random.random((dim, 1))
    b1 = np.random.random((1, dim))
    b2 = np.random.random((1, 1))
    losslist = []

    for ite in range(max_iter):
        loss_per_ite = []
        for m in range(n_samples):
            xi, yi = x[m, :], y[m, :]
            xi, yi = xi.reshape(1, xi.shape[0]), yi.reshape(1, yi.shape[0])
            
            # 前向传播
            u1 = np.dot(xi, w1) + b1
            out1 = sigmoid(u1)
            u2 = np.dot(out1, w2) + b2
            out2 = sigmoid(u2)
            
            loss = np.square(yi - out2) / 2
            loss_per_ite.append(loss)
            print('iter:%d  loss:%.4f' % (ite, loss[0][0]))

            # 反向传播
            error2 = (yi - out2) * d_sigmoid(out2)
            error1 = np.dot(error2, w2.T) * d_sigmoid(out1)

            # 更新参数
            w2 += eta * np.dot(out1.T, error2)
            b2 += eta * error2
            w1 += eta * np.dot(xi.T, error1)
            b1 += eta * error1

        losslist.append(np.mean(loss_per_ite))

    # Loss 可视化
    plt.figure()
    plt.plot(range(max_iter), losslist)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    # plt.show()
    plt.savefig('111.png')
    plt.close()

    return w1, w2, b1, b2

# (3) 测试
data = pd.read_table('watermelon30.txt', delimiter=',')
data.drop('编号', axis=1, inplace=True)
x, y = preprocess(data)
dim = 10
w1, w2, b1, b2 = standard_BP(x, y, dim)

# 根据当前的 x，预测其类别
u1 = np.dot(x, w1) + b1
out1 = sigmoid(u1)
u2 = np.dot(out1, w2) + b2
out2 = sigmoid(u2)
y_pred = np.round(out2)

result = pd.DataFrame(np.hstack((y, y_pred)), columns=['真值', '预测'])
result.to_excel('result.xlsx', index=False)
