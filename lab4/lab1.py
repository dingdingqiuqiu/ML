import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# (1) 加载数据
def loaddata():
    data = [
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
    df = pd.DataFrame(data, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'label'])
    return df

# (2) 训练决策树
def train_decision_tree(df):
    X = df.drop(columns='label')  # 特征
    y = df['label']                # 标签
    clf = DecisionTreeClassifier(random_state=42, criterion='entropy')  # Create a decision tree classifier
    clf.fit(X, y)                   # 训练模型
    return clf

# (3) 预测新样本
def predict(clf, sample):
    return clf.predict([sample])

# (4) Main
df = loaddata()                  # 加载数据集
clf = train_decision_tree(df)   # 训练决策树
print(export_text(clf, feature_names=list(df.columns[:-1])))  # 打印树结构

# 测试预测结果
sample = [1, 1, 0, 1, 0, 0]
prediction = predict(clf, sample)
print(f"Prediction for {sample}: {prediction[0]}")
