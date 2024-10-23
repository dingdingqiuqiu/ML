import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import itertools

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
def train_decision_tree(df, manual=False):
    X = df.drop(columns='label')  # Features
    y = df['label']                # Target variable
    if manual:
        clf = DecisionTreeClassifier()  # Manual implementation
    else:
        clf = DecisionTreeClassifier(random_state=42, criterion='entropy')  # Library implementation
    clf.fit(X, y)                   # Train the model
    return clf

# (3) Predict a new sample
def predict(clf, sample):
    sample_df = pd.DataFrame([sample], columns=df.columns[:-1])
    return clf.predict(sample_df)

# 所有情况
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

# 对比
def compare_predictions(manual_clf, library_clf):
    samples = generate_samples()
    matches = 0
    total = len(samples)
    
    for sample in samples:
        manual_prediction = predict(manual_clf, sample)
        library_prediction = library_clf.predict(pd.DataFrame([sample], columns=df.columns[:-1]))[0]
        if manual_prediction[0] == library_prediction:
            matches += 1
            
    similarity_percentage = (matches / total) * 100
    print(f"Similarity Percentage: {similarity_percentage:.2f}%")

# Main code
df = loaddata()                  # Load the dataset
manual_clf = train_decision_tree(df, manual=True)  # Train manual decision tree
library_clf = train_decision_tree(df, manual=False)  # Train library decision tree

# Compare predictions
compare_predictions(manual_clf, library_clf)
