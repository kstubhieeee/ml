import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_moons 
# Generate two moons dataset 
X, y = make_moons(n_samples=100, noise=0.1, random_state=42) 
# Separate classes for plotting 
X0 = X[y == 0] 
X1 = X[y == 1] 
# -------- Plot Original Data -------- 
plt.figure(figsize=(7,5)) 
plt.scatter(X0[:,0], X0[:,1], color='red', marker='o', label='Class 0') 
plt.scatter(X1[:,0], X1[:,1], color='blue', marker='*', label='Class 1') 
plt.title("Original Data - Two Moons  Kaustubh Bane Roll No 4") 
plt.xlabel("x") 
plt.ylabel("y") 
plt.legend() 
plt.show() 
# -------- Feature Transformation (Kernel Trick manually) -------- 
# Add new feature z = x^2 + y^2 
z = (X[:,0]**2 + X[:,1]**2).reshape(-1,1) 
# New dataset (x, z) 
X_new = np.c_[X[:,0], z.ravel()] 
X0_new = X_new[y==0] 
X1_new = X_new[y==1] 
plt.figure(figsize=(7,5)) 
plt.scatter(X0_new[:,0], X0_new[:,1], color='red', marker='o', label='Class 0') 
plt.scatter(X1_new[:,0], X1_new[:,1], color='blue', marker='*', label='Class 1') 
plt.title("Data projected onto new feature z = x² + y²  Kaustubh Bane Roll No 4") 
plt.xlabel("x") 
plt.ylabel("z = x² + y²") 
plt.legend() 
plt.show() 