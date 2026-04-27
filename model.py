import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([35, 42, 50, 56, 63, 71, 78, 85, 91, 96])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Saved model.pkl")
