import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree  # Import tree module
import matplotlib.pyplot as plt

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

# (2) Prepare the dataset for sklearn
def prepare_data(dataSet):
    X = [example[:-1] for example in dataSet]  # Features
    y = [example[-1] for example in dataSet]   # Labels
    return np.array(X), np.array(y)

# (3) Train decision tree using scikit-learn
myDat, feature_name = loaddata()
X, y = prepare_data(myDat)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# (4) Test prediction function
def predict_with_sklearn(clf, testVec):
    return clf.predict([testVec])[0]

print("Prediction for [0, 1, 0, 1, 0, 0]:", predict_with_sklearn(clf, [0, 1, 0, 1, 0, 0]))

# (5) Save the tree visualization as an image
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=feature_name, class_names=clf.classes_)
plt.savefig('decision_tree.png')  # Save the figure as a PNG file
plt.close()  # Close the plot to free up memory
