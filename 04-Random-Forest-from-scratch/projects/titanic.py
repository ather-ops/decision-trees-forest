# Step 1: Import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Step 2 and 3 load data and EDA
# Step 2
try:
    df=pd.read_csv("Titanic-Dataset.csv")
    print("Data loaded sucessfully")
    print(df.head(10))
except FileNotFoundError:
    print("File not found check again!")
    exit()
except Exception as e:
    print("Erro wile loading file :",e)
    exit()

# Step 3
print("=="*40)
print("Basic statistics:\n",df.describe())
print("=="*40)
print("Basic info:\n",df.info())
print("=="*40)
print("Missing values:\n",df.isnull().sum())
print("=="*40)
print("Duplicated values:\n",df.duplicated().sum())
print("=="*40)
print("Number of rows:",len(df))
print("=="*40)
print("Columns:\n",df.columns.tolist())
print("=="*40)

# step 4: Drpping unwanted columns and data cleaning
# step 4.1 
unwanted_columns=df[["PassengerId","Name","Ticket","Cabin"]]
df=df.drop(unwanted_columns,axis=1)
print("Unwanted columns dropped succesfully")

# step 4.2
df["Age"]=df["Age"].fillna(df["Age"].median(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)

print("Data cleaned sucessfully !")
print("=="*40)
print("After cleaning data :\n",df.head())
print("=="*40)

# Step 5: Manual encoding
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({label: idx for idx, label in enumerate(df["Embarked"].unique())})
print("encoded sucessfully!")
print(df.head())
print("=="*40)
# Step 6: Target and feature
X=df.drop(["Survived"],axis=1)
y=df["Survived"].values

print("Shape of X:",X.shape)
print("=="*40)
print("Shape of Y:",y.shape)
print("=="*40)
# Step 7: Train test split
import numpy as np

def train_test_split_scratch(X, y, test_size=0.2, random_state=42):
    # Convert to numpy arrays if they aren't already
    X = np.array(X)
    y = np.array(y)
    
    np.random.seed(random_state)
    n_samples = X.shape[0]
    
    # Check if X and y have same number of samples
    if n_samples != len(y):
        raise ValueError(f"X has {n_samples} samples but y has {len(y)} samples")
    
    indices = np.random.permutation(n_samples)
    test_count = int(n_samples * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Usage example
X_train, X_test, y_train, y_test = train_test_split_scratch(
    X, y, test_size=0.2, random_state=42
)


print("Train test split done successfully !")
print("=="*40)
print("Shape of X_train:",X_train.shape)
print("Shape of X_test:",X_train.shape)
print("Shape of y_train:",y_train.shape)
print("Shape of y_test:",y_test.shape)
print("=="*40)

# Step 8: Gini Impurity
def gini(y):
    classes,counts=np.unique(y,return_counts=True)
    probabilities=counts/counts.sum()
    return 1 - np.sum(probabilities**2)

print("Gini impurity sucessfuly done!")

# Step 9 Find the best splitter
def best_split(x,y,n_features):
    n_samples,total_features=X.shape
    feature_indices=np.random_choice(total_features,n_features,replace=False)

    best_gini=1.0
    best_feature=None
    best_threshold=None

    for feature_index in feature_indices:
        thresholds=np.unique(x[:,feature_index])
        for threshold in thresholds:
            left_indices=x[:,feature_index]<=threshold
            right_indices=x[:,feature_index]>threshold

            if len(y[left_indices])==0 or len(y[right_indices])==0:
                continue

            gini_left=gini(y[left_indices])
            gini_right=gini(y[right_indices])

            weighted_gini=(left_indices.sum()*gini_left+right_indices*gini_right)/n_samples

            if weighted_gini<best_gini:
                best_gini=weighted_gini
                best_feature=feature_index
                best_threshold=threshold
    return best_feature,best_threshold

# Step 10: Decision tree node
class node:
    def __init__(self,feature=None, threshold=None, left=None , right=None, value=None ):

        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

# Step 11: Decison tree (from Scratch)
class decisionTree:
    def __init__(self,max_depth=5,min_samples_leaf=5,n_features=None):
        self.max_depth=max_depth
        self.min_samples=self.min_samples
        self.n_features=n_features
        self.root=None

        def fit(self,X,Y):
            self.n_features_total=X.shape[1]
            if self.n_features is None:
                self.n_features=self.n_features_total
                self.root=self.grow_tree(X,Y,depth=0)
        def grow_tree(self,X,Y,depth):
            n_samples=X.shape[0]
            n_labels=len(np.unique(Y))

print("Day 1 complete..")

