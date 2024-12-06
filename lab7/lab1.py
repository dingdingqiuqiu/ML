import numpy as np

# 加载数据函数
def loaddata():
    X = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'],
                  [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'],
                  [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'],
                  [3, 'M'], [3, 'L'], [3, 'L']])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    return X, y

# 训练函数
def Train(trainset, train_labels):
    m = trainset.shape[0]  # 样本数量
    n = trainset.shape[1]  # 特征数量

    # 初始化先验概率和条件概率
    prior_probability = {}  # 先验概率 key是类别值，value是类别的概率值
    conditional_probability = {}  # 条件概率 key的构造：类别,特征索引,特征值

    # 类别的可能取值
    labels = set(train_labels)

    # 计算先验概率（此时没有除以总数据量m）
    for label in labels:
        prior_probability[label] = len(train_labels[train_labels == label]) + 1  # 拉普拉斯平滑

    # 计算条件概率
    for label in labels:
        label_indices = np.where(train_labels == label)[0]  # 取出属于该类别的样本索引
        label_samples = trainset[label_indices]  # 属于该类别的样本

        for j in range(n):  # 遍历每个特征
            feature_values = set(trainset[:, j])  # 当前特征的所有可能取值
            for value in feature_values:
                # 拉普拉斯平滑计算 P(特征值 | 类别)
                count = len(label_samples[label_samples[:, j] == value]) + 1
                total = len(label_samples) + len(feature_values)
                conditional_probability[(label, j, value)] = count / total

    # 最终的先验概率（此时除以总数据量m）
    for label in labels:
        prior_probability[label] = prior_probability[label] / (m + len(labels))

    return prior_probability, conditional_probability, labels

# 预测函数
def predict(data, prior_probability, conditional_probability, train_labels_set):
    result = {}
    for label in train_labels_set:
        temp = prior_probability[label]  # 初始化为先验概率
        for j in range(len(data)):
            key = (label, j, data[j])  # 构造条件概率的键
            if key in conditional_probability:
                temp *= conditional_probability[key]  # 乘以条件概率
            else:
                temp *= 1e-6  # 若某个条件概率不存在，赋予极小值避免零概率
        result[label] = temp  # 保存计算结果

    print('result =', result)
    # 排序返回具有最高概率的类别值
    return sorted(result.items(), key=lambda x: x[1], reverse=True)[0][0]

# 加载数据
X, y = loaddata()

# 训练模型
prior_probability, conditional_probability, train_labels_set = Train(X, y)

# 预测新样本
r_label = predict([2, 'S'], prior_probability, conditional_probability, train_labels_set)

print('r_label =', r_label)
