'''
Multi Variable Linear Regression using Scikit-learn's Stochastic Gradient Descent Regressor (SGDRegressor).
'''

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1.0, 2.0, 0.5],
    [2.0, 1.0, 1.5],
    [3.0, 2.5, 2.0],
    [4.0, 3.0, 2.5],
    [5.0, 3.5, 3.0],
    [6.0, 4.0, 3.5],
    [7.0, 4.5, 4.0],
    [8.0, 5.0, 4.5],
    [9.0, 5.5, 5.0],
    [10.0, 6.0, 5.5]
])

y = np.array([
    10.2,
    12.9,
    17.4,
    20.8,
    24.3,
    27.7,
    31.1,
    34.6,
    38.0,
    41.5
])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
model = SGDRegressor(max_iter=10000000, learning_rate='constant')

model.fit(X_norm, y)
y_pred = model.predict(X_norm)

plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()