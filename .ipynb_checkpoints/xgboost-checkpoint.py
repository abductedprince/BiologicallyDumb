import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Define NumPy array (Replace with your actual data)
pbd_array = np.array([
    [70, 1.3, 6.8, 0.5, 1, 0.5, 0.10, 19.80],
    [70, 8.0, 2.0, 3.0, 1, 0.5, 0.05, 18.21],
    [30, 8.0, 6.8, 0.5, 2, 0.5, 0.05, 14.05],
    [70, 1.3, 6.8, 3.0, 1, 1.0, 0.05, 19.10],
    [70, 8.0, 2.0, 3.0, 2, 0.5, 0.10, 15.19],
    [70, 8.0, 6.8, 0.5, 2, 1.0, 0.10, 17.62],
    [30, 8.0, 6.8, 3.0, 2, 0.5, 0.10, 16.18],
    [30, 1.3, 6.8, 3.0, 2, 1.0, 0.10, 15.00],
    [30, 8.0, 2.0, 3.0, 2, 1.0, 0.05, 14.94],
    [30, 1.3, 2.0, 0.5, 1, 1.0, 0.10, 14.33],
    [30, 1.3, 2.0, 0.5, 1, 0.5, 0.05, 13.70],
    [30, 1.3, 2.0, 0.5, 1, 0.5, 0.05, 15.42]
])

# Convert to DataFrame
columns = ["Jaggery", "K2HPO4", "Yeast extract", "Ammonium", "NaCl", "MgSO4", "ZnSO4", "YIELD"]
df = pd.DataFrame(pbd_array, columns=columns)

# Split data into train & test
X = df.drop(columns=["YIELD"])
y = df["YIELD"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Regressor
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")