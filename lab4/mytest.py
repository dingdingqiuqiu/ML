# 导入所需库
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# (1) 加载数据集
def loaddata():
    # 创建数据集
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
    feature_name = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']  # 特征名称
    return pd.DataFrame(dataSet, columns=feature_name + ['label'])  # 返回DataFrame

# (2) 训练决策树
def train_tree(df):
    X = df.drop(columns='label')  # 特征数据
    y = df['label']                # 标签数据
    clf = DecisionTreeClassifier(random_state=42, criterion='entropy')  # 创建决策树分类器
    clf.fit(X, y)                  # 训练模型
    return clf

# (3) 可视化决策树
def plot_tree(clf, feature_names):
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf, feature_names=feature_names, class_names=clf.classes_, filled=True)
    # plt.show()  # 显示树形结构
    plt.savefig('hello.png')
    plt.close()

# (4) 预测新样本的类别
def predict(clf, sample):
    return clf.predict([sample])[0]  # 返回预测结果

# 主代码
if __name__ == "__main__":
    df = loaddata()  # 加载数据集
    clf = train_tree(df)  # 训练决策树
    print("决策树训练完成。")
    
    # 可视化决策树
    print("决策树结构：")
    plot_tree(clf, df.columns[:-1])  # 绘制决策树
    
    # 测试预测功能
    new_sample = [1, 1, 0, 1, 0, 0]  # 新样本
    prediction = predict(clf, new_sample)  # 预测类别
    print(f"新样本 {new_sample} 的预测类别为: {prediction}")  # 输出预测结果
