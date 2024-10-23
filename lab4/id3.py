import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

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
    df = pd.DataFrame(dataSet, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'label'])
    return df

# (2) Train decision tree using ID3
def train_decision_tree(df):
    X = df.drop(columns='label')  # Features
    y = df['label']                # Labels
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)  # ID3
    clf.fit(X, y)                  # Train model
    return clf

# (3) Custom predict function to match your implementation
def predict(clf, sample):
    return clf.predict([sample])[0]

# (4) Main
df = loaddata()                  # Load dataset
clf = train_decision_tree(df)   # Train decision tree
print("Decision Tree Structure:\n", export_text(clf, feature_names=list(df.columns[:-1])))  # Print tree structure

# Test prediction
sample = [1, 1, 0, 1, 0, 0]
prediction = predict(clf, sample)
print(f"Prediction for {sample}: {prediction}")
