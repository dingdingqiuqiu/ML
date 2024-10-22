import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# (1) Load dataset
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

# (2) Train the decision tree
def train_decision_tree(df):
    X = df.drop(columns='label')  # Features
    y = df['label']                # Target variable
    clf = DecisionTreeClassifier()  # Create a decision tree classifier
    clf.fit(X, y)                   # Train the model
    return clf

# (3) Predict a new sample
def predict(clf, sample):
    return clf.predict([sample])

# (4) Main code
df = loaddata()                  # Load the dataset
clf = train_decision_tree(df)   # Train the decision tree
print(export_text(clf, feature_names=list(df.columns[:-1])))  # Print the tree structure

# Test prediction
sample = [1, 1, 0, 1, 0, 0]
prediction = predict(clf, sample)
print(f"Prediction for {sample}: {prediction[0]}")
