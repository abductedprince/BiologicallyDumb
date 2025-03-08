import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pyDOE2 import ccdesign

class ExperimentalDesignPipeline:
    def __init__(self, data, target_col, quantile_threshold=0.5):
        self.data = data
        self.target_col = target_col
        self.significant_factors = None
        self.quantile_threshold = quantile_threshold  # This is Configurable
    
    def run_xgboost(self):
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        # Significant factors based on dynamic quantile threshold (self.quantile_threshold)
        threshold = importance_df['Importance'].quantile(self.quantile_threshold)  # Configurable quantile selection
        self.significant_factors = importance_df[importance_df['Importance'] > threshold]['Feature'].tolist()
        
        print("Selected Significant Factors:", self.significant_factors)
        return self.significant_factors
    
    def run_ccd(self):
        if not self.significant_factors:
            raise ValueError("Run XGBoost first to determine significant factors.")
        
        num_factors = len(self.significant_factors)
        ccd_matrix = ccdesign(num_factors, center=(2, 2))  # Center points for precision
        
        # Getting high and low values for each factor
        factor_ranges = {}
        for factor in self.significant_factors:
            low, high = self.data[factor].min(), self.data[factor].max()
            factor_ranges[factor] = (low, high)
        
        # Scale CCD matrix to actual factor ranges
        ccd_df = pd.DataFrame(ccd_matrix, columns=self.significant_factors)
        for factor in self.significant_factors:
            low, high = factor_ranges[factor]
            ccd_df[factor] = low + (high - low) * (ccd_df[factor] + 1) / 2  # Scaling to actual range
        
        print("Generated CCD Experimental Data:")
        print(ccd_df.head())
        return ccd_df

# Put in actual PBD dataset
data = pd.DataFrame({
    'Factor1': np.random.uniform(0, 0, 0),
    'Factor2': np.random.uniform(0, 0, 0),
    'Factor3': np.random.uniform(0, 0, 0),
    'Target': np.random.uniform(0, 0, 0),
})

pipeline = ExperimentalDesignPipeline(data, target_col='Target', quantile_threshold=0.5)
significant_factors = pipeline.run_xgboost()
ccd_data = pipeline.run_ccd()