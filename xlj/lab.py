# 定义数据文件的路径
data_file_watermelon_3a = "watermelon_3a.csv"

# 导入必要的库
import pandas as pd      # 用于数据处理与分析
import numpy as np       # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘制图形

# 读取 CSV 文件到一个 DataFrame 中（没有表头，所以 header=None）
df = pd.read_csv(data_file_watermelon_3a, header=None)

# 给 DataFrame 添加列名
df.columns = ['id', 'density', 'sugar_content', 'label']  # 'id' 是样本的标识符，'density' 和 'sugar_content' 是特征，'label' 是标签

# 设置 'id' 列为索引（不过在后面的代码中没有使用）
df.set_index(['id'])

# 从数据集中提取特征列 'density' 和 'sugar_content'，并转换为 NumPy 数组（X）
X = df[['density', 'sugar_content']].values

# 提取目标标签 'label' 列（y）
y = df['label'].values

########## SVM 训练与比较
# 使用线性核和高斯核（RBF）两种不同的 SVM 模型进行训练与比较
from sklearn import svm  # 导入支持向量机（SVM）模块

# 遍历两种核函数：线性核（'linear'）和径向基核（RBF，'rbf'）
for fig_num, kernel in enumerate(('linear', 'rbf')):  
    # 补充构建SVM模型及训练代码
    # 创建并训练 SVM 模型，使用当前的核函数
    svc = svm.SVC(kernel=kernel,C=100)  # 初始化一个 SVM 分类器，使用指定的核函数
    svc.fit(X, y)  # 使用特征集 X 和目标标签 y 训练模型
    
    # 对新的测试样本 X_test 进行预测
    X_test = [[0.719, 0.103]]  # 一个新的样本，包含密度=0.719 和糖分含量=0.103

    # 补充训练预测代码
    predicted_label = svc.predict(X_test)  # 使用训练好的模型预测新样本的标签
    print(f"使用 {kernel} 核的预测标签: {predicted_label}")  # 打印预测结果
    
    # 获取训练好的 SVM 模型的支持向量
    sv = svc.support_vectors_  # 获取支持向量的坐标
    
    ##### 绘制决策边界和支持向量
    plt.figure(fig_num)  # 为每个核函数创建一个新的图形
    plt.clf()  # 清空当前图形，准备重新绘制
    
    # 绘制所有数据点，并根据标签对它们进行着色
    plt.scatter(X[:, 0], X[:, 1], edgecolors='k', c=y, cmap=plt.cm.Paired, zorder=10)  
    # 绘制支持向量，用较大的标记区分开来
    plt.scatter(sv[:, 0], sv[:, 1], edgecolors='k', facecolors='none', s=80, linewidths=2, zorder=10)  
    
    # 绘制决策边界和决策区域
    # 根据特征值（密度和糖分含量）确定绘图范围
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    
    # 创建网格，用于绘制决策边界
    XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    # 计算网格中每个点的决策函数值
    Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)  # 将结果重新调整为网格形状
    
    # 使用颜色填充决策区域，根据决策函数值的正负进行区域区分
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)  # 填充决策区域，颜色代表分类结果
    
    # 添加决策边界的等高线，Z=0 表示决策边界，Z=±0.5 为支持向量的边界
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-0.5, 0, 0.5])
    
    # 设置图形标题，显示当前使用的核函数类型（线性核或 RBF 核）
    plt.title(kernel)
    
    # 设置坐标轴为紧凑模式（无多余空间）
    plt.axis('tight')
        
    # 保存绘制的图形
    plt.savefig(kernel)
    plt.close()
