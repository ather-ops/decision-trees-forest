print("Titanic Survival project")

# Step 1: Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

# Step 2:Load Data with error handling
try:
    df=pd.read_csv("Titanic-Dataset.csv")
    print("Data loaded successfully")
    print("=="*40)
    print(df.head(10))
except FileNotFoundError:
    print("Error . File not found check again!")
    exit()

# Step 3: EDA
print("=="*40)
print("Basic Statistics:\n",df.describe())
print("=="*40)
print("Basic informations:\n",df.info())
print("=="*40)
print("Missing values:\n",df.isnull().sum())
print("=="*40)
print("Duplicated values:\n",df.duplicated().sum())
print("=="*40)
print("No of rows:\n",len(df))
print("=="*40)
print("No of Columns:\n",df.columns.value_counts())
print("=="*40)
print("Columns:\n",df.columns.tolist())
print("=="*40)
df.drop("Cabin",axis=1,inplace=True) # Drop cabin column
# Step 4: Data cleaning
def cleaning(df):
    for col in df.columns:
        if df[col].dtype in ["int64","float64"]:
            df[col]=df[col].fillna(df[col].median())
        else:
            df[col]=df[col].fillna(df[col].mode()[0])
    return df
print("=="*40)
df=cleaning(df)
print("=="*40)
print("Missing values:\n",df.isnull().sum())
print("=="*40)
print("df after filling missing values:\n",df.head())

# Better version of your graphs
plt.figure(figsize=(15,5))

# Graph 1: Sex Distribution (Bar Chart - NOT histogram)
plt.subplot(1,3,1)
df["Sex"].value_counts().plot(kind='bar', color=['blue', 'pink'])
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Sex Distribution")
plt.xticks([0,1], ['Male', 'Female'], rotation=0)
plt.grid(True, alpha=0.3)

# Graph 2: Survived Distribution (Histogram - this is fine)
plt.subplot(1,3,2)
plt.hist(df["Survived"], bins=2, color='green', edgecolor='black')
plt.xlabel("Survived (0=Died, 1=Survived)")
plt.ylabel("Frequency")
plt.title("Survival Distribution")
plt.xticks([0,1], ['Died', 'Survived'])
plt.grid(True, alpha=0.3)

# Graph 3: Embarked Distribution (Bar Chart)
plt.subplot(1,3,3)
df["Embarked"].value_counts().plot(kind='bar', color='orange')
plt.xlabel("Embarked Port")
plt.ylabel("Count")
plt.title("Embarked Distribution")
plt.xticks([0,1,2], ['Cherbourg(C)', 'Southampton(S)', 'Queenstown(Q)'], rotation=0)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Graph 4: Survival Rate by Pclass
plt.figure(figsize=(6,4))
survival_by_class = df.groupby('Pclass')['Survived'].mean()
survival_by_class.plot(kind='bar', color=['gold', 'silver', 'brown'])
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Pclass')
plt.xticks([0,1,2], ['1st Class', '2nd Class', '3rd Class'], rotation=0)
plt.ylim(0, 1)
for i, v in enumerate(survival_by_class):
    plt.text(i, v + 0.02, f'{v:.2%}', ha='center')
plt.grid(True, alpha=0.3)
plt.show()



# Step 5: Manual encode
df["Sex"]=df["Sex"].map({"male":0,"female":1})  # Manual encode Male=0 and Female -1
df["Embarked"]=df["Embarked"].map({"C":0,"S":1,"Q":2}) # C=0, S=1 and Q=2

# Step 6: Feature and Target
X=df.drop(["PassengerId","Survived","Name","Ticket"],axis=1)
y=df["Survived"]

# Step 7: Train test split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42

)
print("=="*40)
print(df.head())
print("=="*40)

# Step 8: Model selection and predictions
model=RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    oob_score=True,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Step 8: Evalution
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("=="*40)

# Step 9 : Checking OOB score
oob_score=model.oob_score_
print("OOB score :",oob_score)
print("=="*40)

# Step 10: Feature Importance
plt.figure(figsize=(8,5))

feature=X.columns
importance=model.feature_importances_
plt.barh(feature, importance, color='teal')
plt.xlabel('Importance Score')
plt.title('Feature Importance - Random Forest')
plt.grid(True, alpha=0.3)
plt.show()


# Step 11: Feature importance values  
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Feature Importance:\n", feature_importance)

# Step 12: New predictions
new_values=[
  {
    "Pclass": 3,
    "Sex": 0,
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 1
  },
  {
    "Pclass": 1,
    "Sex": 1,
    "Age": 38,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 71.2833,
    "Embarked": 0
  },
  {
    "Pclass": 3,
    "Sex": 1,
    "Age": 26,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.925,
    "Embarked": 1
  },
  {
    "Pclass": 1,
    "Sex": 1,
    "Age": 35,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 53.1,
    "Embarked": 1
  },
  {
    "Pclass": 3,
    "Sex": 0,
    "Age": 35,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 8.05,
    "Embarked": 1
  },
  {
    "Pclass": 3,
    "Sex": 0,
    "Age": 42,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 8.4583,
    "Embarked": 1
  },
  {
    "Pclass": 1,
    "Sex": 0,
    "Age": 54,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 51.8625,
    "Embarked": 1
  },
  {
    "Pclass": 3,
    "Sex": 0,
    "Age": 2,
    "SibSp": 3,
    "Parch": 1,
    "Fare": 21.075,
    "Embarked": 1
  },
  {
    "Pclass": 3,
    "Sex": 1,
    "Age": 27,
    "SibSp": 0,
    "Parch": 2,
    "Fare": 11.1333,
    "Embarked": 1
  },
  {
    "Pclass": 2,
    "Sex": 1,
    "Age": 14,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 30.0708,
    "Embarked": 0
  },
  {
    "Pclass": 3,
    "Sex": 1,
    "Age": 4,
    "SibSp": 1,
    "Parch": 1,
    "Fare": 16.7,
    "Embarked": 1
  },
  {
    "Pclass": 1,
    "Sex": 1,
    "Age": 58,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 26.55,
    "Embarked": 1
  },
  {
    "Pclass": 3,
    "Sex": 0,
    "Age": 20,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 8.05,
    "Embarked": 1
  },
  {
    "Pclass": 3,
    "Sex": 0,
    "Age": 39,
    "SibSp": 1,
    "Parch": 5,
    "Fare": 31.275,
    "Embarked": 1
  },
  {
    "Pclass": 3,
    "Sex": 1,
    "Age": 14,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.8542,
    "Embarked": 1
  },
  {
    "Pclass": 2,
    "Sex": 1,
    "Age": 55,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 16,
    "Embarked": 1
  },
  {
    "Pclass": 3,
    "Sex": 0,
    "Age": 2,
    "SibSp": 4,
    "Parch": 1,
    "Fare": 29.125,
    "Embarked": 1
  },
  {
    "Pclass": 2,
    "Sex": 0,
    "Age": 32,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 13,
    "Embarked": 1
  },
  {
    "Pclass": 3,
    "Sex": 1,
    "Age": 31,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 18,
    "Embarked": 1
  },
  {
    "Pclass": 3,
    "Sex": 1,
    "Age": 28,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.225,
    "Embarked": 0
  }
]
new_df=pd.DataFrame(new_values)
new_df=cleaning(new_df)
new_predictions=model.predict(new_df)
for i,new,pred in zip(range(1,21),new_values,new_predictions):
    print(f"New data {i}: {new} => Predicted Survival: {'Yes' if pred==1 else 'No'}")

# Step 13: Save cleaned DataFrame and Model using Joblib
import joblib

# Save cleaned DataFrame
df.to_csv("titanic_cleaned.csv", index=False)
print("Cleaned DataFrame saved as 'titanic_cleaned.csv'")

# Save the trained model
joblib.dump(model, "titanic_model.pkl")
print("Model saved as 'titanic_model.pkl'")

#  Verify files were saved
import os
print("=="*40)
print("Saved Files:")
print("- titanic_cleaned.csv (Size: {} bytes)".format(os.path.getsize("titanic_cleaned.csv")))
print("- titanic_model.pkl (Size: {} bytes)".format(os.path.getsize("titanic_model.pkl")))
print("=="*40)