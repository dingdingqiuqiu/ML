# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import accuracy_score

# 加载数据
def loaddata():
    # 从文本文件中加载数据
    data = np.loadtxt('data1.txt', delimiter=',')
    n = data.shape[1] - 1  # 特征数
    X = data[:, 0:n]  # 特征矩阵
    y = data[:, -1].reshape(-1, 1)  # 标签向量
    return X, y

def plot(X, y):
    # 绘制数据点
    pos = np.where(y == 1)  # 正样本索引
    neg = np.where(y == 0)  # 负样本索引
    plt.scatter(X[pos[0], 0], X[pos[0], 1], marker='x', label='Positive')
    plt.scatter(X[neg[0], 0], X[neg[0], 1], marker='o', label='Negative')
    plt.xlabel('Exam 1 score')  # x轴标签
    plt.ylabel('Exam 2 score')  # y轴标签
    plt.legend()  # 显示图例
    # plt.show()
    plt.savefig('dataAndLine.png')
    plt.close()

X, y = loaddata()  # 加载数据并赋值
plot(X, y)  # 绘制数据

def sigmoid(z):
    # 计算Sigmoid函数
    r = 1 / (1 + np.exp(-z))  # Sigmoid公式
    return r

def hypothesis(X, theta):
    # 计算假设函数值
    z = np.dot(X, theta)  # 线性组合
    return sigmoid(z)  # 返回Sigmoid值

def computeCost(X, y, theta):
    m = X.shape[0]  # 样本数
    z = hypothesis(X, theta)  # 计算假设值
    # 计算代价函数
    cost = -np.sum(y * np.log(z) + (1 - y) * np.log(1 - z)) / m  # 代价公式
    return cost

def gradientDescent(X, y, theta, iterations, alpha):
    m = X.shape[0]  # 样本数
    X = np.hstack((np.ones((m, 1)), X))  # 在X前加一列1（偏置项）
    for i in range(iterations):
        # 计算梯度并更新参数
        theta_temp = theta - (alpha / m) * np.dot(X.T, (hypothesis(X, theta) - y))  # 梯度下降更新
        theta = theta_temp  # 更新theta
        # 每10000次迭代输出一次损失值
        if (i % 10000 == 0):
            print('第', i, '次迭代，当前损失为：', computeCost(X, y, theta), 'theta=', theta)
    return theta

def predict(X):
    # 预测函数
    c = np.ones(X.shape[0]).transpose()  # 创建一列全1
    X = np.insert(X, 0, values=c, axis=1)  # 在X前插入全1列
    h = hypothesis(X, theta)  # 计算假设值
    # 根据概率值决定最终的分类
    h[h >= 0.5] = 1  # 大于等于0.5为1类
    h[h < 0.5] = 0   # 小于0.5为0类
    return h

X, y = loaddata()  # 再次加载数据
n = X.shape[1]  # 特征数
theta = np.zeros(n + 1).reshape(n + 1, 1)  # 初始化参数theta
iterations = 250000  # 迭代次数
alpha = 0.008  # 学习率

theta = gradientDescent(X, y, theta, iterations, alpha)  # 执行梯度下降
print('theta=\n', theta)  # 输出theta

def plotDescisionBoundary(X, y, theta):
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])  # 定义颜色
    plt.xlabel('Exam 1 score')  # x轴标签
    plt.ylabel('Exam 2 score')  # y轴标签
    plt.scatter(X[:, 0], X[:, 1], c=np.array(y).squeeze(), cmap=cm_dark, s=30)  # 绘制散点图
    # 计算并绘制决策边界
    x1 = np.array([np.min(X[:, 0]), np.max(X[:, 0])])  # x轴范围
    x2 = -(theta[0] + theta[1] * x1) / theta[2]  # 决策边界方程
    plt.plot(x1, x2, label='Decision Boundary', color='blue')  # 绘制边界线
    plt.legend()  # 显示图例
    # plt.show()
    plt.savefig('side.png')
    plt.close()

plotDescisionBoundary(X, y, theta)  # 绘制决策边界

