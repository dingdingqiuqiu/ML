import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io

# 加载数据
def loaddata():
    data = np.loadtxt('watermelon_4.txt', delimiter=',')  # 从文本文件中读取数据
    return data

X = loaddata()  # 调用函数加载数据

# K-means 初始化：随机选择初始的 k 个中心点
def kMeansInitCentroids(X, k):
    m = X.shape[0]  # 数据集的样本数
    centroids = X[np.random.choice(m, k, replace=False), :]  # 随机选择 k 个点作为初始中心点
    print(centroids)
    return centroids


# 选择较近的初始点
# def kMeansInitCentroids(X, k):
#     m = X.shape[0]  # 数据集的样本数
#     centroids = np.zeros((k, X.shape[1]))  # 存储最终的 K 个中心点
#     # 随机选择一个初始点
#     centroid_idx = np.random.choice(m)
#     centroids[0] = X[centroid_idx]  # 将其作为第一个中心点
#     
#     # 计算每个点与第一个中心点的距离
#     distances = np.linalg.norm(X - centroids[0], axis=1)
#     distances[centroid_idx] = np.inf 
#
#     for i in range(1, k):
#         # 选择距离当前已选择中心点最近的点（即最小距离）
#         min_distance_idx = np.argmin(distances)  # 找到距离最小的点的索引
#         centroids[i] = X[min_distance_idx]  # 将该点作为新的中心点
#         distances = np.linalg.norm(X - centroids[i], axis=1)
#         distances[min_distance_idx] = np.inf 
#         
#     
#     print(centroids)
#
#     return centroids

# 选择较远的初始点
# def kMeansInitCentroids(X, k):
#     m = X.shape[0]  # 数据集的样本数
#     centroids = np.zeros((k, X.shape[1]))  # 存储最终的 K 个中心点
#     # 随机选择一个初始点
#     centroid_idx = np.random.choice(m)
#     centroids[0] = X[centroid_idx]  # 将其作为第一个中心点
#     
#     # 计算每个点与第一个中心点的距离
#     distances = np.linalg.norm(X - centroids[0], axis=1)
#     distances[centroid_idx] = 0 
#
#     for i in range(1, k):
#         # 选择距离当前已选择中心点最近的点（即最小距离）
#         min_distance_idx = np.argmax(distances)  # 找到距离最小的点的索引
#         centroids[i] = X[min_distance_idx]  # 将该点作为新的中心点
#         distances = np.linalg.norm(X - centroids[i], axis=1)
#         distances[min_distance_idx] = 0 
#         
#     
#     print(centroids)
#
#     return centroids

# 找到每个数据点属于哪个簇（最近的中心点）
def findClosestCentroids(X, centroids):
    m = X.shape[0]  # 数据集的样本数
    k = centroids.shape[0]  # 聚类的数量（即中心点的个数）
    idx = np.zeros(m)  # 存储每个数据点属于哪个簇（中心点索引）

    for i in range(m):
        distances = np.linalg.norm(X[i] - centroids, axis=1)  # 计算数据点到每个中心点的距离
        idx[i] = np.argmin(distances)  # 将数据点分配给距离最近的中心点
    
    return idx

# 根据每个簇中包含的数据点，重新计算中心点的位置
def computeCentroids(X, idx, k):
    centroids = np.zeros((k, X.shape[1]))  # 存储重新计算的 k 个中心点
    for i in range(k):
        points = X[idx == i]  # 找出属于簇 i 的所有数据点
        centroids[i] = np.mean(points, axis=0)  # 计算簇 i 中所有点的均值，作为新的中心点
    return centroids

# K-means 主函数
def k_means(X, k, max_iters=100):
    centroids = kMeansInitCentroids(X, k)  # 初始化 k 个中心点
    for i in range(max_iters):
        # 步骤 1：为每个数据点分配最近的中心点
        idx = findClosestCentroids(X, centroids)
        
        # 步骤 2：根据每个簇的点重新计算中心点
        centroids = computeCentroids(X, idx, k)
        
    return idx, centroids  # 返回每个点的簇索引和最终的中心点

# 使用 K-means 聚类，设定聚类数量为 3，最大迭代次数为 8
idx, centroids = k_means(X, 3, 8)

# 打印每个数据点的簇索引和最终的中心点
print("簇分配结果：", idx)
print("最终的中心点：", centroids)

# 数据点的散点图
plt.scatter(X[:, 0], X[:, 1], s=20)

# 绘制聚类结果
# cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b', 'y', 'c', 'm'])  # 定义一个7种颜色的颜色映射
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b', 'y', 'c'])  # 定义10种颜色的颜色映射
plt.scatter(X[:, 0], X[:, 1], c=idx, cmap=cm_dark, s=20)  # 绘制数据点，颜色根据簇分配
plt.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(len(centroids)), cmap=cm_dark, marker='*', s=500)  # 绘制中心点，用星形标记，大小为500
plt.savefig('res.png')  # 显示绘制的图形
plt.close()
