import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Sample Data (Replace with actual values)
pbd_data = np.array([
    # Format: [Jaggery, K2HPO4, Yeast Extract, (NH4)2SO4, NaCl, MgSO4.7H2O, ZnSO4.7H2O, Observed Yield]
    [1, 0.5, 2, 0.8, 1.2, 0.3, 0.01, 16.5],
    [0.8, 0.4, 1.8, 0.9, 1.3, 0.4, 0.02, 15.2],
    # Add more data points...
])

ccd_data = np.array([
    # Format: [Jaggery, Yeast Extract, NaCl, Observed Yield]
    [1, 2, 1.2, 17.5],
    [0.9, 1.8, 1.0, 16.8],
    # Add more data points...
])

# Convert to DataFrame for easier handling
pbd_df = pd.DataFrame(pbd_data, columns=["Jaggery", "K2HPO4", "Yeast Extract", "(NH4)2SO4", "NaCl", "MgSO4.7H2O", "ZnSO4.7H2O", "Observed Yield"])
ccd_df = pd.DataFrame(ccd_data, columns=["Jaggery", "Yeast Extract", "NaCl", "Observed Yield"])

# Handle Missing Data (if any)
pbd_df.dropna(inplace=True)
ccd_df.dropna(inplace=True)

# Splitting into Input (X) and Output (y)
X_pbd = pbd_df.drop(columns=["Observed Yield"])
y_pbd = pbd_df["Observed Yield"]

X_ccd = ccd_df.drop(columns=["Observed Yield"])
y_ccd = ccd_df["Observed Yield"]

# Train-Test Split (80-20)
X_pbd_train, X_pbd_test, y_pbd_train, y_pbd_test = train_test_split(X_pbd, y_pbd, test_size=0.2, random_state=42)
X_ccd_train, X_ccd_test, y_ccd_train, y_ccd_test = train_test_split(X_ccd, y_ccd, test_size=0.2, random_state=42)

# Initialize Regressors
rf_pbd = RandomForestRegressor(n_estimators=100, random_state=42)
rf_ccd = RandomForestRegressor(n_estimators=100, random_state=42)

# Alternative: Decision Tree Regressor (uncomment if needed)
# dt_pbd = DecisionTreeRegressor(random_state=42)
# dt_ccd = DecisionTreeRegressor(random_state=42)

# Train Models
rf_pbd.fit(X_pbd_train, y_pbd_train)
rf_ccd.fit(X_ccd_train, y_ccd_train)

# Predict
y_pbd_pred = rf_pbd.predict(X_pbd_test)
y_ccd_pred = rf_ccd.predict(X_ccd_test)

# Calculate R² Scores
r2_pbd = r2_score(y_pbd_test, y_pbd_pred)
r2_ccd = r2_score(y_ccd_test, y_ccd_pred)

# Print Results
print(f"PBD Model R² Score: {r2_pbd:.4f}")
print(f"CCD Model R² Score: {r2_ccd:.4f}")

# Decision on Experiment Validity
threshold = 0.8  # If R² is below this, experiments may need rechecking

if r2_pbd < threshold:
    print("PBD experimental data may have inconsistencies. Rechecking needed.")
else:
    print("PBD experimental data seems reliable.")

if r2_ccd < threshold:
    print("CCD experimental data may have inconsistencies. Rechecking needed.")
else:
    print("CCD experimental data seems reliable.")