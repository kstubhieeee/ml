import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset where points are almost on the line y = 2x + 1
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([3.1, 5.0, 6.9, 9.2, 10.8, 13.1, 14.9, 17.0])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Coefficients
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Plot
plt.scatter(X, y, color='green', label='Actual')
plt.plot(X, y_pred, color='blue', label='Fitted Line')
plt.title('Linear Regression — Points Close to Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
