from math import log
import numpy as np
import operator

# (1) Load dataset
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

# (2) Calculate entropy of the dataset
def entropy(dataSet):
    m = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    e = 0.0
    for key in labelCounts:
        prob = labelCounts[key] / m
        e -= prob * log(prob, 2)
    return e

# (3) Split dataset
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] + featVec[axis + 1:]
            retDataSet.append(reducedFeatVec)
    return retDataSet

# (4) Choose the best feature
def chooseBestFeature(dataSet):
    n = len(dataSet[0]) - 1
    baseEntropy = entropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(n):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * entropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# (5) Class vote
def classVote(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# (6) Train decision tree recursively
def trainTree(dataSet, feature_name):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return classVote(classList)
    bestFeat = chooseBestFeature(dataSet)
    bestFeatName = feature_name[bestFeat]
    myTree = {bestFeatName: {}}
    
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        sub_feature_name = feature_name[:]
        sub_dataset = splitDataSet(dataSet, bestFeat, value)
        myTree[bestFeatName][value] = trainTree(sub_dataset, sub_feature_name)
        
    return myTree

# (7) Print tree structure
def printTree(tree, indent=''):
    if not isinstance(tree, dict):
        print(indent + '-> ' + str(tree))
    else:
        for key in tree:
            print(indent + str(key) + ':')
            printTree(tree[key], indent + '  ')

# (8) Test code
myDat, feature_name = loaddata()
myTree = trainTree(myDat, feature_name)
print("Decision Tree Structure:")
printTree(myTree)

# Given new sample, predict class
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

# Test prediction function
print("Prediction for [0, 1, 0, 1, 0, 0]:", predict(myTree, feature_name, [1, 1, 0, 1, 0, 0]))

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# (8) Prepare the dataset for sklearn
def prepare_data(dataSet):
    X = [example[:-1] for example in dataSet]  # Features
    y = [example[-1] for example in dataSet]   # Labels
    return np.array(X), np.array(y)

# (9) Train decision tree using scikit-learn
X, y = prepare_data(myDat)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# (10) Test prediction function
def predict_with_sklearn(clf, testVec):
    return clf.predict([testVec])[0]

print("Prediction for [0, 1, 0, 1, 0, 0]:", predict_with_sklearn(clf, [0, 1, 0, 1, 0, 0]))
