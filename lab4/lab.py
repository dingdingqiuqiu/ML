from math import log
import numpy as np
import operator
import csv

# (1) 加载数据集
def loaddata():
    dataSet = [
        [0, 0, 0, 0, 0, 0, 'yes'],
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
    feature_name = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    return dataSet, feature_name

# (2) 计算数据集的熵
def entropy(dataSet):
    # 数据集条数
    m = len(dataSet)
    # 保存所有类别以及类别对应的样本数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 计算数据集的熵
    e = 0.0
    for key in labelCounts:
        prob = labelCounts[key] / m
        e -= prob * log(prob, 2)
    return e

# (3) 划分数据集 
def splitDataSet(dataSet, axis, value):
    # 补充按给定特征和特征值划分好的数据集的代码
    # axis对应的是特征的索引;
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] + featVec[axis + 1:]
            retDataSet.append(reducedFeatVec)
    return retDataSet

# (4) 选择最最优特征
def chooseBestFeature(dataSet):
    n = len(dataSet[0]) - 1
    # 计数整个数据集的熵
    baseEntropy = entropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历每个特征
    for i in range(n):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 遍历当前特征的所有可能取值
        for value in uniqueVals:
            # 按照特征的value值进行数据集划分
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算条件熵
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * entropy(subDataSet)
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
