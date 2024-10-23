from math import log  # 导入对数函数，用于计算熵
import numpy as np   # 导入NumPy库，用于数值运算
import operator      # 导入operator模块，用于排序和计数操作

# (1) 加载数据集
def loaddata():
    dataSet = [
        [0, 0, 0, 0, 0, 0, 'yes'],  # 数据行，最后一个元素是标签
        [1, 0, 1, 0, 0, 0, 'yes'],
        [1, 0, 0, 0, 0, 0, 'yes'],
        [0, 0, 1, 0, 0, 0, 'yes'],
        [2, 0, 0, 0, 0, 0, 'yes'],
        [0, 1, 0, 0, 1, 1, 'yes'],
        [1, 1, 0, 1, 1, 1, 'yes'],
        [1, 1, 0, 0, 1, 0, 'yes'],
        [1, 1, 1, 1, 1, 0, 'no'],
        [0, 2, 2, 0, 2, 1, 'no'],
        [2, 2, 2, 2, 2, 0, 'no'],
        [2, 0, 0, 2, 2, 1, 'no'],
        [0, 1, 0, 1, 0, 0, 'no'],
        [2, 1, 1, 1, 0, 0, 'no'],
        [1, 1, 0, 0, 1, 1, 'no'],
        [2, 0, 0, 2, 2, 0, 'no'],
        [0, 0, 1, 1, 1, 0, 'no']
    ]
    feature_name = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']  # 特征名称列表
    return dataSet, feature_name  # 返回数据集和特征名称

# (2) 计算数据集的熵
def entropy(dataSet):
    # 数据集条数
    m = len(dataSet)
    # 保存所有类别以及类别对应的样本数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 获取当前样本的标签
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0  # 初始化类别计数
        labelCounts[currentLabel] += 1  # 增加当前类别的计数
    # 计算数据集的熵
    e = 0.0
    for key in labelCounts:
        prob = labelCounts[key] / m  # 计算当前类别的概率
        e -= prob * log(prob, 2)      # 根据公式计算熵的部分
    return e  # 返回计算得到的熵值

# (3) 划分数据集 
def splitDataSet(dataSet, axis, value):
    # 按给定特征和特征值划分数据集
    # axis对应的是特征的索引
    retDataSet = []  # 初始化划分后的数据集
    # 遍历数据集
    for featVec in dataSet:
        # 检查当前样本在指定特征上的值是否等于给定值
        if featVec[axis] == value:
            # 创建一个新样本，去掉指定特征
            reducedFeatVec = featVec[:axis] + featVec[axis + 1:]
            retDataSet.append(reducedFeatVec)  # 添加到新数据集中
    return retDataSet  # 返回划分后的数据集

# (4) 选择最优特征
def chooseBestFeature(dataSet):
    n = len(dataSet[0]) - 1  # 特征数量（不包括标签）
    # 计算整个数据集的熵
    baseEntropy = entropy(dataSet)
    bestInfoGain = 0.0  # 初始化最佳信息增益
    bestFeature = -1    # 初始化最佳特征的索引

    # 遍历每个特征
    for i in range(n):
        featList = [example[i] for example in dataSet]  # 获取当前特征的所有值
        uniqueVals = set(featList)  # 获取当前特征的唯一值
        newEntropy = 0.0  # 初始化条件熵

        # 遍历当前特征的所有可能取值
        for value in uniqueVals:
            # 按照特征的value值进行数据集划分
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算条件熵
            prob = len(subDataSet) / float(len(dataSet))  # 计算当前子数据集的概率
            newEntropy += prob * entropy(subDataSet)  # 累加条件熵

        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        # 保存最大信息增益及其对应特征
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    # 返回最佳特征        
    return bestFeature

# (5) 类别投票表决
def classVote(classList):
    # 定义字典，保存每个标签对应的个数 
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    # 排序   
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# (6) 递归训练决策树
def trainTree(dataSet, feature_name):
    classList = [example[-1] for example in dataSet]
    # 所有类别均一致
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 数据集中无特征
    if len(dataSet[0]) == 1:
        return classVote(classList)
    # 选择最优特征
    bestFeat = chooseBestFeature(dataSet)
    bestFeatName = feature_name[bestFeat]
    myTree = {bestFeatName: {}}
    
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 遍历uniqueVals中的每个值，生成相应的分支
    for value in uniqueVals:
        sub_feature_name = feature_name[:]
        # 生成在dataSet中bestFeat取值为value的子集；
        sub_dataset = splitDataSet(dataSet, bestFeat, value)
        # 根据得到的子集，生成决策树
        myTree[bestFeatName][value] = trainTree(sub_dataset, sub_feature_name)
        
    return myTree

# (7) 测试代码
myDat, feature_name = loaddata()
myTree = trainTree(myDat, feature_name)
print(myTree)

# 给定新样本，预测类别
def predict(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    
    if isinstance(valueOfFeat, dict):
        classLabel = predict(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
        
    return classLabel

# 测试预测功能
print(predict(myTree, feature_name, [1, 1, 0, 1, 0, 0]))
