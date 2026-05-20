# Project 1: Drug Classifier - Decision Tree from Scratch
# Basic Version - Using Actual Data

# Step 1: Import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

print("="*40)
print("DRUG CLASSIFIER")
print("="*40)

# Step 2: Load data
df = pd.read_csv("drug200.csv")
print(df.head())

# Step 3: EDA
print("="*40)
print("Basic statistics:")
print(df.describe())

print("="*40)
print("Missing values:", df.isnull().sum())

print("="*40)
print("Drug Distribution:")
print(df["Drug"].value_counts())

# Step 4: Simple plots
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df["Age"], bins=10, color='skyblue', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(df["Na_to_K"], bins=10, color='lightcoral', edgecolor='black')
plt.title("Na_to_K Distribution")
plt.xlabel("Na_to_K")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Step 5: Encode categorical (keeping original drug names)
df["Sex"] = df["Sex"].map({"M":0, "F":1})
df["BP"] = df["BP"].map({"LOW":0, "NORMAL":1, "HIGH":2})
df["Cholesterol"] = df["Cholesterol"].map({"NORMAL":0, "HIGH":1})

# Step 6: Split features and labels
X = df.drop("Drug", axis=1)
y = df["Drug"]

# Step 7: Train test split
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

print("="*40)
print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# Step 8: Simple decision tree based on Na_to_K (from the data pattern)
def predict_drug(x):
    # x[0]=Age, x[1]=Sex, x[2]=BP, x[3]=Cholesterol, x[4]=Na_to_K
    
    # From data: High Na_to_K (>15) usually drugY
    if x[4] > 15:
        return "drugY"
    # Low Na_to_K with HIGH BP sometimes drugA or drugB
    elif x[4] < 10 and x[2] == 2:
        return "drugA"
    # Low Na_to_K with LOW BP usually drugC
    elif x[4] < 11 and x[2] == 0:
        return "drugC"
    # Normal range
    else:
        return "drugX"

# Step 9: Evaluate
predictions = []
for i in range(len(X_test)):
    pred = predict_drug(X_test.iloc[i].values)
    predictions.append(pred)

# Calculate accuracy
correct = 0
for i in range(len(predictions)):
    if predictions[i] == y_test.iloc[i]:
        correct += 1

accuracy = correct / len(predictions)
print("="*40)
print(f"Accuracy: {accuracy*100:.2f}%")

# Step 10: Show predictions
print("="*40)
print("First 20 predictions:")
print(f"{'Index':<6} {'Actual':<10} {'Predicted':<10} {'Match':<10}")
print("-"*40)
for i in range(20):
    match = "YES" if predictions[i] == y_test.iloc[i] else "NO"
    print(f"{i:<6} {y_test.iloc[i]:<10} {predictions[i]:<10} {match:<10}")

# Step 11: Plot results
plt.figure(figsize=(12, 4))

# Accuracy bar
plt.subplot(1, 3, 1)
plt.bar(["Accuracy"], [accuracy*100], color='green')
plt.ylim(0, 100)
plt.ylabel("Percentage")
plt.title(f"Model Accuracy")
plt.text(0, accuracy*100 + 2, f"{accuracy*100:.1f}%", ha='center')

# Correct vs Incorrect
plt.subplot(1, 3, 2)
correct_count = correct
incorrect_count = len(predictions) - correct
plt.bar(["Correct", "Incorrect"], [correct_count, incorrect_count], color=['green', 'red'])
plt.ylabel("Count")
plt.title("Prediction Results")
plt.text(0, correct_count + 1, str(correct_count), ha='center')
plt.text(1, incorrect_count + 1, str(incorrect_count), ha='center')

# Drug distribution in test set
plt.subplot(1, 3, 3)
drug_counts = y_test.value_counts()
plt.bar(drug_counts.index, drug_counts.values, color='skyblue', edgecolor='black')
plt.xlabel("Drug")
plt.ylabel("Count")
plt.title("Test Set Drug Distribution")

plt.tight_layout()
plt.show()

# Step 12: Show per class accuracy
print("="*40)
print("PER CLASS ACCURACY:")
print("="*40)

for drug in df["Drug"].unique():
    drug_indices = [i for i in range(len(y_test)) if y_test.iloc[i] == drug]
    if len(drug_indices) > 0:
        drug_correct = sum(1 for i in drug_indices if predictions[i] == y_test.iloc[i])
        drug_acc = drug_correct / len(drug_indices) * 100
        print(f"{drug}: {drug_correct}/{len(drug_indices)} ({drug_acc:.1f}%)")

# Step 13: Predict for new patients
print("="*40)
print("PREDICTIONS FOR NEW PATIENTS")
print("="*40)

# New patient data: [Age, Sex, BP, Cholesterol, Na_to_K]
# Sex: M=0, F=1
# BP: LOW=0, NORMAL=1, HIGH=2
# Cholesterol: NORMAL=0, HIGH=1

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

print("\nNew Patient Predictions:")
print("-"*60)
print(f"{'Patient':<10} {'Age':<5} {'Sex':<6} {'BP':<8} {'Chol':<8} {'Na_to_K':<8} {'Predicted Drug':<15}")
print("-"*60)

for i, patient in enumerate(new_patients):
    pred = predict_drug(patient)
    
    # Convert codes back to labels for display
    sex_label = "Female" if patient[1] == 1 else "Male"
    bp_labels = {0: "LOW", 1: "NORMAL", 2: "HIGH"}
    chol_labels = {0: "NORMAL", 1: "HIGH"}
    
    print(f"Patient {i+1:<4} {patient[0]:<5} {sex_label:<6} {bp_labels[patient[2]]:<8} {chol_labels[patient[3]]:<8} {patient[4]:<8.1f} {pred:<15}")

print("="*40)
print("PROJECT COMPLETED")
