import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据文件
data = np.loadtxt('data1.txt', delimiter=',')
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# 添加偏置项
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# 超参数
learning_rate = 0.01
n_iterations = 1000
m = X.shape[0]

# 初始化 theta
theta = np.random.randn(2, 1)

# 梯度下降
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

print(f"训练得到的参数: {theta}")

# 预测函数
def predict(X, theta):
    X_b = np.c_[np.ones((len(X), 1)), X]
    return X_b.dot(theta)

# 预测新样本
X_new = np.array([[0], [2]])
y_predict = predict(X_new, theta)
print(f"预测值: {y_predict}")

plt.plot(X, y, "b.")
plt.plot(X_new, y_predict, "r-")
plt.xlabel("X")
plt.ylabel("y")
plt.title("数据散点图及线性回归直线")
# plt.savefig('dataAndbeeline.png')
# plt.close()
plt.show()

# 记录损失值
loss_history = []

theta = np.random.randn(2, 1)
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
    loss = np.mean((X_b.dot(theta) - y) ** 2)
    loss_history.append(loss)

plt.plot(range(n_iterations), loss_history)
plt.xlabel("迭代次数")
plt.ylabel("损失值")
plt.title("梯度下降过程中损失的变化")
plt.savefig("miss.png")
plt.close()
# 预测函数已经在上面定义
X_new = np.array([[1.5], [3.5]])
y_predict = predict(X_new, theta)
print(f"新的样本数据预测值: {y_predict}")

