import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据文件
data = np.loadtxt('data2.txt', delimiter=',')
# x0是偏置项
num = data.shape[0]
# print(num)
x0 = np.ones(num)
x1_original = data[:, 0]
x2_original = data[:, 1]

# x1是商品房面积归一化后的结果
x1_original_min = np.min(x1_original)
x1_original_max = np.max(x1_original)
x1 = (x1_original - x1_original_min) / (x1_original_max - x1_original_min)
# x2是商品房房间数归一化后的结果
x2_original_min = np.min(x2_original)
x2_original_max = np.max(x2_original)
x2 = (x2_original - x2_original_min) / (x2_original_max - x2_original_min)

X = np.stack((x0, x1, x2), axis=1)

Y = data[:, 2].reshape(-1, 1)

# 超参数
learning_rate = 0.01
n_iterations = 5000
m = X.shape[0]

# 初始化theta
theta = np.random.randn(3, 1)
loss_history = []


# 梯度下降
for iteration in range(n_iterations):
    gradients = 2 / m * X.T.dot(X.dot(theta) - Y)
    theta = theta - learning_rate * gradients
    loss = np.mean(((X.dot(theta) - Y) ** 2))
    loss_history.append(loss)


# 反归一化
# print(theta)
x1_max_min = x1_original_max - x1_original_min
x2_max_min = x2_original_max - x2_original_min
theta[0][0] = theta[0][0] - (theta[1][0] * x1_original_min) / (x1_max_min) - (theta[2][0] * x2_original_min) / (x2_max_min)  
theta[1][0] = theta[1][0] / (x1_original_max - x1_original_min)
theta[2][0] = theta[2][0] / (x2_original_max - x2_original_min)

# 打印多元线性回归方程
coeffs = theta.ravel()
equation = f"y = {coeffs[0]:.2f} + {coeffs[1]:.2f}*x1 + {coeffs[2]:.2f}*x2"
print(f"训练得到的多元线性回归方程: {equation}")


# 绘制实际数据x1,x2,y的散点图及预测面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制实际数据的散点图
ax.scatter(x1_original, x2_original, Y.ravel(), c='b', marker='o', label='实际数据')

# 创建网格用于绘制预测面
x1_grid, x2_grid = np.meshgrid(np.linspace(x1_original.min(), x1_original.max(), 10),np.linspace(x2_original.min(), x2_original.max(), 10))

# 准备数据用于预测面
x0_grid = np.ones(x1_grid.size)
X_grid = np.stack((x0_grid, x1_grid.ravel(), x2_grid.ravel()), axis=1)
Y_grid_pred = X_grid.dot(theta).reshape(x1_grid.shape)

# 绘制预测面
ax.plot_surface(x1_grid, x2_grid, Y_grid_pred, alpha=0.5, color='r', label='预测面')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.legend()
# plt.show()
plt.savefig("dataAndPlanes.png")
plt.close()

# 绘制损失随迭代次数变化的散点图
iteration_numbers = np.arange(0, n_iterations)
plt.figure()
plt.scatter(iteration_numbers, loss_history)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('均方误差随迭代次数变化的散点图')
# plt.show()
plt.savefig("loss.png")
plt.close()

# 预测函数
def predict():
    x1_val = float(input("请输入x1的值: "))
    x2_val = float(input("请输入x2的值: "))
    X_new = np.array([[1, x1_val, x2_val]])
    y = X_new.dot(theta)[0][0]
    print(f"预测值: {y}")

predict()
