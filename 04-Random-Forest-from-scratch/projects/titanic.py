# Step 1: Import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Step 2 and 3: Load data and EDA
try:
    df = pd.read_csv("Titanic-Dataset.csv")
    print("Data loaded successfully")
except FileNotFoundError:
    print("File not found check again!")
    exit()
except Exception as e:
    print("Error while loading file :", e)
    exit()

# Step 4: Dropping unwanted columns and data cleaning
unwanted_columns = ["PassengerId", "Name", "Ticket", "Cabin"]
df = df.drop(unwanted_columns, axis=1)

# Fix: fillna with correct inplace handling
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("Data cleaned successfully !")

# Step 5: Manual encoding
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({label: idx for idx, label in enumerate(df["Embarked"].unique())})

# Step 6: Target and feature
X = df.drop(["Survived"], axis=1).values  
y = df["Survived"].values

# Step 7: Train test split from scratch
def train_test_split_scratch(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    
    if n_samples != len(y):
        raise ValueError(f"X has {n_samples} samples but y has {len(y)} samples")
    
    indices = np.random.permutation(n_samples)
    test_count = int(n_samples * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split_scratch(X, y, test_size=0.2, random_state=42)

# Step 8: Gini Impurity
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities**2)

# Step 9: Find the best splitter
def best_split(x, y, n_features):
    n_samples, total_features = x.shape
    feature_indices = np.random.choice(total_features, n_features, replace=False)

    best_gini = 1.0
    best_feature = None
    best_threshold = None

    for feature_index in feature_indices:
        thresholds = np.unique(x[:, feature_index])
        for threshold in thresholds:
            left_indices = x[:, feature_index] <= threshold
            right_indices = x[:, feature_index] > threshold

            if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                continue

            gini_left = gini(y[left_indices])
            gini_right = gini(y[right_indices])

            weighted_gini = (left_indices.sum() * gini_left + right_indices.sum() * gini_right) / n_samples

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_index
                best_threshold = threshold
                
    return best_feature, best_threshold

# Step 10: Decision tree node
class node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Step 11: Decision tree (from Scratch)
class decisionTree:
    def __init__(self, max_depth=5, min_samples_leaf=5, n_features=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf  
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features_total = X.shape[1]
        if self.n_features is None:
            self.n_features = self.n_features_total
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))
        
        # Base cases
        if (depth >= self.max_depth or n_labels == 1 or n_samples < (2 * self.min_samples_leaf)):
            return node(value=self._most_common_label(y))
        feature, threshold = best_split(X, y, self.n_features)
        if feature is None:
            return node(value=self._most_common_label(y))
            
        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices  
        if (left_indices.sum() < self.min_samples_leaf or right_indices.sum() < self.min_samples_leaf):
            return node(value=self._most_common_label(y))
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return node(feature=feature, threshold=threshold, left=left, right=right)

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])  

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)    


clf = decisionTree(max_depth=5, min_samples_leaf=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Decision Tree Accuracy from Scratch: {accuracy * 100:.2f}%")
