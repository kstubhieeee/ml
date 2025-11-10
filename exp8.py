import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# Create synthetic non-linearly separable data
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# Plot the original data
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c='green', edgecolor='k')
plt.title("Original Dataset Kaustubh Bane Roll No 4")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Initialize DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)

# Fit DBSCAN to scaled data
dbscan.fit(X_scaled)

# Get cluster labels
labels = dbscan.labels_
print("Cluster labels:", np.unique(labels))


# Visualize DBSCAN clustering results
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette='tab10', style=labels, s=60)
plt.title("DBSCAN Clustering Results Kaustubh Bane Roll No 4")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend(title='Cluster')
plt.show()


# Trying different parameters
dbscan2 = DBSCAN(eps=0.2, min_samples=10)
dbscan2.fit(X_scaled)

# Plot new results
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=dbscan2.labels_, palette='tab10', s=60)
plt.title("DBSCAN with eps=0.2, min_samples=10 Kaustubh Bane Roll No 4  ")
plt.show()


n_noise = list(labels).count(-1)
print(f"Number of noise points: {n_noise}")