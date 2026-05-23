print("DRUG CLASSIFIER")
# Step 1: Load libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load data with error handling 
try:
    df=pd.read_csv("Drug200.csv")
    print("Data loaded successfully:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'Drug200.csv' file not found. Please check the file path.")
    exit()
except Exception as e:
    print("An error occurred while loading the data:", e)
    exit()

# Step 3: EDA
print("==" * 40)
print("Exploratory Data Analysis:")
print("==" * 40)
print("Basic information:\n",df.info())
print("==" * 40)
print("Missing values:\n", df.isnull().sum())
print("==" * 40)
print("Drug distribution:\n", df["Drug"].value_counts())
print("==" * 40)
print("columns:",df.columns.tolist())
print("==" * 40)
print("Number of rows:",len(df))

# Step 4: Plots
plt.figure(figsize=(10,4))

# 1:Age Distribution
plt.subplot(1,2,1)
plt.hist(df["Age"],bins=10,color="skyblue",edgecolor="pink")
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True,alpha=0.03)

# 2.NA_to_K Distribution
plt.subplot(1,2,2)
plt.hist(df["Na_to_K"],bins=10, color="green", edgecolor="black")
plt.title("Na_to_K Distribution")
plt.xlabel("Na_to_K")
plt.ylabel("Frequency")
plt.grid(True,alpha=0.3)
plt.show()

# Step 5: Manual encoding
df["Sex"]=df["Sex"].map({"M":0,"F":1})
df["BP"]=df["BP"].map({"LOW":0,"NORMAL":1,"HIGH":2})
df["Cholesterol"]=df["Cholesterol"].map({"NORMAL":0,"HIGH":1})

# Step 6: Feature and Target
X=df.drop("Drug",axis=1)
Y=df["Drug"]

# Step 7: Train test split with shuffle
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size=0.2, 
    random_state=42,
    shuffle=True,  
    stratify=Y 
)    

# Step 8: More regularization
model = DecisionTreeClassifier(
    max_depth=3,           
    min_samples_split=5,   
    min_samples_leaf=4,    
    max_features='sqrt',   
    random_state=42
)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

# Step 9: Evalution
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)
print("=="*40)
print(f"Accuracy score:{accuracy:.2f}")
plt.bar(["Accuracy"], [accuracy*100], color='green')
plt.ylim(0, 100)
plt.ylabel("Percentage")
plt.title(f"Model Accuracy")
plt.text(0, accuracy*100 + 2, f"{accuracy*100:.1f}%", ha='center')
plt.show()


# Step 10: Feature Importance
features=X.columns
importances=model.feature_importances_
plt.bar(features,importances)
plt.title("Feature Importance")
plt.show()

# Step 11: Predict new values
new_patients = [
    [45, 1, 1, 0, 18.5],   # [45, Female, NORMAL BP, NORMAL Cholesterol, Na_to_K=18.5]
    [30, 0, 2, 1, 8.2],     # [30, Male, HIGH BP, HIGH Cholesterol, Na_to_K=8.2]
    [55, 1, 0, 0, 9.5],     # [55, Female, LOW BP, NORMAL Cholesterol, Na_to_K=9.5]
    [25, 0, 1, 1, 12.0],    # [25, Male, NORMAL BP, HIGH Cholesterol, Na_to_K=12.0]
    [60, 1, 2, 0, 7.5],     # [60, Female, HIGH BP, NORMAL Cholesterol, Na_to_K=7.5]
    [35, 0, 0, 1, 20.0],    # [35, Male, LOW BP, HIGH Cholesterol, Na_to_K=20.0]
    [50, 1, 1, 1, 14.0],    # [50, Female, NORMAL BP, HIGH Cholesterol, Na_to_K=14.0]
    [70, 0, 2, 0, 9.0],     # [70, Male, HIGH BP, NORMAL Cholesterol, Na_to_K=9.0]
]
print("=="*40)
new_predictions=model.predict(new_patients)
for i, (patient, pred) in enumerate(zip(new_patients, new_predictions)):
    print(f"Patient {i}: {patient} -> {pred}")
