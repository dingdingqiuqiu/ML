# title: 使用梯度下降法训练线性回归模型
# purpose: 掌握线性回归的基本原理，以及梯度下降法和最小二乘法

# 导入科学计算库和绘图库
import numpy as np
import matplotlib.pyplot as plt  

# 设置Matplotlib的全局字体为SimHei（黑体），以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决Matplotlib在使用中文字体时，负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 读取数据文件
data = np.loadtxt('data1.txt', delimiter=',')
# print(data)
X = data[:, 0].reshape(-1, 1)
# print(X)
Y = data[:, 1].reshape(-1, 1)

# 添加偏置项
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# 超参数
learning_rate = 0.01
n_iterations = 1000
m = X.shape[0]

# 初始化 theta
theta = np.random.randn(2, 1)
# 记录损失 
loss_history = []

# 梯度下降
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - Y)
    theta = theta - learning_rate * gradients
    loss = np.mean(((X_b.dot(theta) - Y) ** 2))
    loss_history.append(loss)

# 预测直线
w = theta[1][0]
b = theta[0][0]
print(f"训练得到的直线: y = {w:.2f}x + {b:.2f}")


# 绘制散点图及直线
plt.scatter(X, Y)
X_plot = np.array([[x] for x in np.linspace(X.min(), X.max(), 100)])
X_b_plot = np.c_[np.ones((len(X_plot), 1)), X_plot]
Y_plot = X_b_plot.dot(theta)
plt.plot(X_plot, Y_plot, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('散点图及预测直线')
# plt.show()
plt.savefig('dataAndLine.png')
plt.close()


# 绘制损失随迭代次数变化的散点图
iteration_numbers = np.arange(0, n_iterations)
plt.figure()
plt.scatter(iteration_numbers, loss_history)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('均方误差随迭代次数变化的散点图')
# plt.show()
plt.savefig('loss.png')
plt.close()


# 预测函数
def predict():
    x = float(input("请输入变量 x 的值: "))
    X_new = np.array([[x]])
    X_b = np.c_[np.ones((1, 1)), X_new]
    y = X_b.dot(theta)[0][0]
    print(f"预测值: {y}")

predict()


