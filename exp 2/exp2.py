import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load datasets
train_df = pd.read_csv('training.csv')
test_df = pd.read_csv('testing.csv')

X_train = train_df[['mean radius']].values
y_train = train_df['target'].values

X_test = test_df[['mean radius']].values
y_test = test_df['target'].values

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_probs = model.predict_proba(X_test)[:, 1]
y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_probs]

# Accuracy check
correct = 0
print("\nPredicted vs Actual Comparison:")
print(f"{'Index':<6}{'Mean Radius':<15}{'Predicted':<10}{'Actual':<10}{'Result'}")
for i in range(len(y_test)):
    result = "✅" if y_pred[i] == y_test[i] else "❌ WRONG"
    if y_pred[i] == y_test[i]:
        correct += 1
    print(f"{i:<6}{X_test[i][0]:<15.2f}{y_pred[i]:<10}{y_test[i]:<10}{result}")

accuracy = correct / len(y_test)
print(f"\nManual Test Accuracy: {accuracy:.2f}")

# Plot logistic curve
x_vals = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_vals = model.predict_proba(x_vals)[:, 1]

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, color='blue', label='Logistic Curve')
plt.scatter(X_train, y_train, color='red', label='Training Data')
plt.xlabel('Mean Radius')
plt.ylabel('Probability')
plt.title('Logistic Regression Curve            Kaustubh Bane           Roll no : 4')
plt.legend()
plt.grid(True)
plt.show()
