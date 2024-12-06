import numpy as np

# 数据加载函数
def loaddata():
    # 输入特征矩阵 X 和标签向量 y
    X = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'],
                  [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'],
                  [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'],
                  [3, 'M'], [3, 'L'], [3, 'L']])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    return X, y

# 训练函数
def Train(trainset, train_labels):
    """
    trainset: 输入的训练特征矩阵
    train_labels: 输入的训练标签
    """
    m = trainset.shape[0]  # 样本数量
    n = trainset.shape[1]  # 特征数量

    # 初始化先验概率和条件概率字典
    prior_probability = {}  # 先验概率
    conditional_probability = {}  # 条件概率

    # 获取类别的唯一取值
    labels = set(train_labels)

    # 计算先验概率 (使用拉普拉斯平滑，+1 防止概率为 0)
    for label in labels:
        prior_probability[label] = (len(train_labels[train_labels == label]) + 1) / (m + len(labels))

    # 计算条件概率 (同样使用拉普拉斯平滑)
    for label in labels:
        # 获取属于当前类别的样本索引
        label_indices = np.where(train_labels == label)[0]
        # 当前类别的数据
        label_data = trainset[label_indices]
        # print(label_data)

        # 对每个特征进行条件概率计算
        for j in range(n):
            # 获取该特征的唯一值
            feature_values = set(trainset[:, j])
            # print(feature_values)
            for value in feature_values:
                # 条件概率的 key 为 (类别, 特征索引, 特征值)
                key = (label, j, value)
                # 统计当前特征值在该类别中出现的次数
                count = np.sum(label_data[:, j] == value)
                # 条件概率计算 (加一平滑)
                conditional_probability[key] = (count + 1) / (len(label_indices) + len(feature_values))

    return prior_probability, conditional_probability, labels

# 预测函数
def predict(data, prior_probability, conditional_probability, train_labels_set):
    """
    data: 待预测的数据
    prior_probability: 先验概率
    conditional_probability: 条件概率
    train_labels_set: 标签集合
    """
    result = {}  # 存储每个类别的后验概率

    # 遍历每个类别，计算后验概率
    for label in train_labels_set:
        # 初始概率为先验概率
        prob = prior_probability[label]
        print('初始概率为', prob)
        # 依次乘以各特征对应的条件概率
        for j in range(len(data)):
            # 统一data[j]的数据类型为str
            if isinstance(data[j], (int, float)):
                tmp = str(data[j])
            else:
                tmp = data[j]
            key = (label, j, tmp)
            print('开始计算', key, ':')
            # 使用字典的 get 方法，若不存在该条件概率则取一个很小的值 (避免 0 的出现)
            print(prob, '*', conditional_probability.get(key, 1e-6), ':')
            prob *= conditional_probability.get(key, 1e-6)
        # 保存计算结果
        result[label] = prob

    print('result =', result)
    # 返回后验概率最大的类别
    return sorted(result.items(), key=lambda x: x[1], reverse=True)[0][0]

# 主程序执行部分
X, y = loaddata()  # 加载数据
prior_probability, conditional_probability, train_labels_set = Train(X, y)  # 训练模型
print("先验概率:", prior_probability)
print("\n条件概率:")
for key, value in conditional_probability.items():
    print(f"{key}: {value}")
r_label = predict([2, 'S'], prior_probability, conditional_probability, train_labels_set)  # 预测数据 [2, 'S']
print('r_label =', r_label)  # 输出预测结果

