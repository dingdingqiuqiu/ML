import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace

def getDataSet():
    """
    西瓜数据集3.0alpha。 列：[密度，含糖量，好瓜]
    :return: np数组。返回数据集
    """
    dataSet = [
        [0.697, 0.460, '是'],
        [0.774, 0.376, '是'],
        [0.634, 0.264, '是'],
        [0.608, 0.318, '是'],
        [0.556, 0.215, '是'],
        [0.403, 0.237, '是'],
        [0.481, 0.149, '是'],
        [0.437, 0.211, '是'],
        [0.666, 0.091, '否'],
        [0.243, 0.267, '否'],
        [0.245, 0.057, '否'],
        [0.343, 0.099, '否'],
        [0.639, 0.161, '否'],
        [0.657, 0.198, '否'],
        [0.360, 0.370, '否'],
        [0.593, 0.042, '否'],
        [0.719, 0.103, '否']
    ]

    # 将‘是’转为1，‘否’转为-1
    for i in range(len(dataSet)):   
        if dataSet[i][-1] == '是':
            dataSet[i][-1] = 1
        else:
            dataSet[i][-1] = -1

    return np.array(dataSet)

def calErr(dataSet, feature, threshVal, inequal, D):
    """
    计算数据带权值的错误率。
    :param dataSet:     数据集，格式为：[密度，含糖量，好瓜]
    :param feature:     特征索引，[0:密度，1:含糖量]
    :param threshVal:   阈值，用于判断当前特征的分割
    :param inequal:     判断条件，'lt'表示小于，'gt'表示大于
    :param D:           权重矩阵，样本的权重
    :return:            错误率
    """
    DFlatten = D.flatten()   # 展平权重矩阵
    errCnt = 0
    i = 0
    if inequal == 'lt':  # 小于阈值
        for data in dataSet:
            if (data[feature] <= threshVal and data[-1] == -1) or \
               (data[feature] > threshVal and data[-1] == 1):  # 错误分类
                errCnt += 1 * DFlatten[i]  # 累计错误样本的权重
            i += 1
    else:  # 大于阈值
        for data in dataSet:
            if (data[feature] >= threshVal and data[-1] == -1) or \
               (data[feature] < threshVal and data[-1] == 1):
                errCnt += 1 * DFlatten[i]  # 累计错误样本的权重
            i += 1
    return errCnt

def buildStump(dataSet, D):
    """
    通过带权重的数据，建立错误率最小的决策树桩。
    :param dataSet: 数据集
    :param D:       权重
    :return:        最优的决策树桩信息，包括特征、阈值、不等式以及错误率
    """
    m, n = dataSet.shape
    bestErr = np.inf  # 初始化最小错误率为无穷大
    bestStump = {}    # 存储最佳决策树桩
    numSteps = 16.0   # 用于划分阈值的步数

    # 遍历每个特征进行分类
    for i in range(n-1):  # 排除标签列
        rangeMin = dataSet[:, i].min()  # 当前特征的最小值
        rangeMax = dataSet[:, i].max()  # 当前特征的最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 阈值步长
        # 遍历每个阈值
        for j in range(m):  # 遍历所有数据点
            threVal = rangeMin + float(j) * stepSize  # 当前阈值
            # 遍历小于和大于两种不等式情况
            for inequal in ['lt', 'gt']:
                err = calErr(dataSet, i, threVal, inequal, D)  # 计算错误率
                if err < bestErr:  # 如果找到更小的错误率
                    bestErr = err
                    bestStump["feature"] = i  # 最优特征
                    bestStump["threshVal"] = threVal  # 最优阈值
                    bestStump["inequal"] = inequal  # 最优不等式
                    bestStump["err"] = err  # 最优错误率

    return bestStump

def predict(data, bestStump):
    """
    使用决策树桩进行数据预测
    :param data:        待预测数据
    :param bestStump:   最优决策树桩
    :return:            预测结果
    """
    if bestStump["inequal"] == 'lt':
        if data[bestStump["feature"]] <= bestStump["threshVal"]:
            return 1
        else:
            return -1
    else:
        if data[bestStump["feature"]] >= bestStump["threshVal"]:
            return 1
        else:
            return -1

def AdaBoost(dataSet, T):
    """
    AdaBoost算法，迭代T次，训练T个弱分类器
    :param dataSet:  数据集
    :param T:        迭代次数，即训练多少个分类器
    :return:         字典，包含T个分类器
    """
    m, n = dataSet.shape
    D = np.ones((1, m)) / m  # 初始化样本权重，均匀分配
    classLabel = dataSet[:, -1].reshape(1, -1)  # 标签列
    G = {}  # 存储分类器

    for t in range(T):
        stump = buildStump(dataSet, D)  # 构建决策树桩
        err = stump["err"]  # 获取当前分类器的错误率
        alpha = np.log((1 - err) / err) / 2  # 计算分类器的权重
        pre = np.zeros((1, m))  # 存储预测结果
        for i in range(m):
            pre[0][i] = predict(dataSet[i], stump)  # 预测每个样本
        # 更新样本权重，错误分类的样本增加权重
        a = np.exp(-alpha * classLabel * pre)
        D = D * a / np.dot(D, a.T)

        # 存储当前分类器的参数
        G[t] = {}
        G[t]["alpha"] = alpha
        G[t]["stump"] = stump
    return G

def adaPredic(data, G):
    """
    使用AdaBoost的集成分类器进行预测
    :param data:    待预测数据
    :param G:       集成分类器，包含多个决策树桩
    :return:        预测结果
    """
    score = 0
    for key in G.keys():
        pre = predict(data, G[key]["stump"])  # 使用每个树桩进行预测
        score += G[key]["alpha"] * pre  # 加权结果
    flag = 0
    if score > 0:
        flag = 1  # 如果总得分大于0，预测为好瓜（1）
    else:
        flag = -1  # 否则预测为坏瓜（-1）
    return flag

def calcAcc(dataSet, G):
    """
    计算AdaBoost分类器的准确率
    :param dataSet:     数据集
    :param G:           集成分类器
    :return:            准确率
    """
    rightCnt = 0
    for data in dataSet:
        pre = adaPredic(data, G)  # 使用AdaBoost分类器进行预测
        if pre == data[-1]:
            rightCnt += 1  # 如果预测正确，计数
    print(rightCnt)
    print(len(dataSet))
    return rightCnt / float(len(dataSet))  # 准确率 = 正确预测数 / 总数

# 绘制数据集，clf为集成学习器
def plotData(data, clf, filename):
    X1, X2 = [], []
    Y1, Y2 = [], []
    datas = data
    labels = data[:, 2]
    
    # 根据标签分类数据
    for data, label in zip(datas, labels):
        if label > 0:
            X1.append(data[0])
            Y1.append(data[1])
        else:
            X2.append(data[0])
            Y2.append(data[1])

    x = linspace(0, 0.8, 100)
    y = linspace(0, 0.6, 100)

    # 绘制每个决策树桩的分割线
    for key in clf.keys():
        z = [clf[key]["stump"]["threshVal"]] * 100
        if clf[key]["stump"]["feature"] == 0:
            plt.plot(z, y)  # 绘制密度的分割线
        else:
            plt.plot(x, z)  # 绘制含糖量的分割线

    # 绘制数据点
    plt.scatter(X1, Y1, marker='+', label='好瓜', color='b')
    plt.scatter(X2, Y2, marker='_', label='坏瓜', color='r')

    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.xlim(0, 0.8)  # 设置x轴范围
    plt.ylim(0, 0.6)  # 设置y轴范围
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.legend(loc='upper left')
    # plt.show()
    plt.savefig(filename)
    plt.close()

def main():
    dataSet = getDataSet()  # 获取数据集
    for t in [3, 5, 11]:  # 学习器的数量
        G = AdaBoost(dataSet, t)  # 训练AdaBoost模型
        print('集成学习器（字典）：', f"G{t} = {G}")
        print('准确率=', calcAcc(dataSet, G))  # 计算准确率
        plotData(dataSet, G, f'plot_G{t}.png')  # 绘制结果

if __name__ == '__main__':
    main()
