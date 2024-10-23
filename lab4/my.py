import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import itertools
import operator
from math import log

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
    feature_name = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'label']
    return dataSet, feature_name

# (2) Calculate the entropy of the dataset
def entropy(dataSet):
    m = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    e = 0.0
    for key in labelCounts:
        prob = labelCounts[key] / m
        e -= prob * log(prob, 2)
    return e

# (3) Split the dataset
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
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# (6) Recursive training of the decision tree
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

# (7) Test prediction
def predict(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    
    if key not in secondDict:
        return "no"  # or return a default value, e.g., 'unknown'
    
    valueOfFeat = secondDict[key]
    
    if isinstance(valueOfFeat, dict):
        classLabel = predict(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
        
    return classLabel

# (8) Train the sklearn decision tree with voting
def train_voting_tree(df, n_estimators=5, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    X = df.drop(columns='label')
    y = df['label']
    
    # Create multiple DecisionTree classifiers
    estimators = []
    for _ in range(n_estimators):
        clf = DecisionTreeClassifier(
            random_state=None,  # Random state set to None for different trees
            criterion='entropy',
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        estimators.append((f'decision_tree_{_}', clf))
    
    # Create a VotingClassifier
    voting_clf = VotingClassifier(estimators=estimators, voting='hard')
    voting_clf.fit(X, y)
    return voting_clf

# (9) Generate all possible samples based on feature ranges
def generate_samples():
    ranges = [
        [0, 1, 2],  # for a1
        [0, 1, 2],  # for a2
        [0, 1, 2],  # for a3
        [0, 1, 2],  # for a4
        [0, 1, 2],  # for a5
        [0, 1]      # for a6
    ]
    return list(itertools.product(*ranges))

# (10) Compare predictions and calculate similarity rate
def compare_predictions(manual_tree, library_clf, df):
    samples = generate_samples()
    matches = 0
    total_samples = len(samples)

    for sample in samples:
        manual_prediction = predict(manual_tree, df.columns[:-1].tolist(), list(sample))
        library_prediction = library_clf.predict(pd.DataFrame([sample], columns=df.columns[:-1]))[0]
        print(f"Sample: {sample}, Manual Prediction: {manual_prediction}, Library Prediction: {library_prediction}")

        if manual_prediction == library_prediction:
            matches += 1

    similarity_rate = (matches / total_samples) * 100
    print(f"Similarity Rate: {similarity_rate:.2f}%")

# Main code
data, feature_names = loaddata()  # Load the dataset
manual_tree = trainTree(data, feature_names)  # Train manual decision tree

# Specify parameters for the Voting Classifier
n_estimators = 5  # Number of decision trees in the ensemble
max_depth = 3  # Limit the depth of the trees
min_samples_split = 4  # Minimum samples to split
min_samples_leaf = 2  # Minimum samples at a leaf node

library_clf = train_voting_tree(
    pd.DataFrame(data, columns=feature_names), 
    n_estimators=n_estimators, 
    max_depth=max_depth, 
    min_samples_split=min_samples_split, 
    min_samples_leaf=min_samples_leaf
)  # Train library decision tree with voting

# Compare predictions
compare_predictions(manual_tree, library_clf, pd.DataFrame(data, columns=feature_names))
