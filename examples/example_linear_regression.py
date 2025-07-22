import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from algos.linear_regression import LinearRegression
from utils.data_split import train_test_split
from utils.metrics import mean_squared_error, r2_score


# Load data
df = pd.read_csv("data/simple_data.csv")
X = df[["x"]].values
y = df["y"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=42)

# Train model
model = LinearRegression(lr=0.01, epochs=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
