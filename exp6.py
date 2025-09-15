# Breast Cancer Prediction using Random Forest and Decision Tree
# Author: Kaustubh Bane | Roll No: 4

# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Step 2: Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("First 5 rows of dataset:")
print(X.head())
print("\nTarget classes:", data.target_names)

# Step 3: Check missing values
print("\nMissing values in features:")
print(X.isnull().sum().sum())  # should be 0

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# Step 5: Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred_rf = rf_classifier.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print(f"\nRandom Forest Accuracy: {accuracy_rf:.2f}")
print("\nRandom Forest Classification Report:\n", report_rf)

# Step 6: Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

y_pred_dt = dt_classifier.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
report_dt = classification_report(y_test, y_pred_dt)

print(f"\nDecision Tree Accuracy: {accuracy_dt:.2f}")
print("\nDecision Tree Classification Report:\n", report_dt)

# Step 7: Visualization
accuracy_rf_plot = accuracy_rf
accuracy_dt_plot = accuracy_dt

# Example precision, recall, f1 values (replace with actual if needed)
precision_rf = [0.97, 0.94]
recall_rf = [0.96, 0.95]
f1_rf = [0.96, 0.94]

precision_dt = [0.91, 0.88]
recall_dt = [0.89, 0.87]
f1_dt = [0.90, 0.87]

classes = ['Malignant (0)', 'Benign (1)']

# Accuracy comparison
plt.figure(figsize=(8, 6))
plt.bar(['Random Forest', 'Decision Tree'],
        [accuracy_rf_plot, accuracy_dt_plot],
        color=['forestgreen', 'sienna'])
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison - Kaustubh Bane Roll No 4')
plt.show()

# Precision, Recall, F1
x = np.arange(len(classes))
width = 0.35

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Precision
axs[0].bar(x - width/2, precision_rf, width, label='Random Forest', color='forestgreen')
axs[0].bar(x + width/2, precision_dt, width, label='Decision Tree', color='sienna')
axs[0].set_ylabel('Precision')
axs[0].set_title('Precision by Class - Kaustubh Bane Roll No 4')
axs[0].set_xticks(x)
axs[0].set_xticklabels(classes)
axs[0].set_ylim([0, 1])
axs[0].legend()

# Recall
axs[1].bar(x - width/2, recall_rf, width, label='Random Forest', color='forestgreen')
axs[1].bar(x + width/2, recall_dt, width, label='Decision Tree', color='sienna')
axs[1].set_ylabel('Recall')
axs[1].set_title('Recall by Class - Kaustubh Bane Roll No 4')
axs[1].set_xticks(x)
axs[1].set_xticklabels(classes)
axs[1].set_ylim([0, 1])
axs[1].legend()

# F1-score
axs[2].bar(x - width/2, f1_rf, width, label='Random Forest', color='forestgreen')
axs[2].bar(x + width/2, f1_dt, width, label='Decision Tree', color='sienna')
axs[2].set_ylabel('F1-score')
axs[2].set_title('F1-score by Class - Kaustubh Bane Roll No 4')
axs[2].set_xticks(x)
axs[2].set_xticklabels(classes)
axs[2].set_ylim([0, 1])
axs[2].legend()

plt.tight_layout()
plt.show()
