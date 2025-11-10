import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Add Gaussian noise
np.random.seed(42)
noise = np.random.normal(0, 0.2, X.shape)
X_noisy = X + noise

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y, test_size=0.3, random_state=42
)

# Apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Evaluate accuracy
acc = lda.score(X_test, y_test)
print("LDA Test Accuracy with Noisy Data:", acc)

# Plot LDA projection
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']

for i, color in zip(range(len(iris.target_names)), colors):
    plt.scatter(
        X_train_lda[y_train == i, 0],
        X_train_lda[y_train == i, 1],
        alpha=0.7, c=color, label=iris.target_names[i]
    )

plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("LDA Projection of Noisy Iris Dataset - Kaustubh Bane")
plt.legend()
plt.grid(True)
plt.show()
