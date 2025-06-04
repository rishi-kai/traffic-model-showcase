# src/models/xgboost_model.py

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Load dataset
df = pd.read_csv("../../data/processed/traffic_features.csv", parse_dates=["DateTime"])
print("✅ Data loaded. Columns:", df.columns)

# Define features and target
features = ['hour', 'day', 'month', 'weekday', 'is_weekend', 'Junction']
target = 'Vehicles'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Train the model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
print("✅ XGBoost model trained.")

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 XGBoost Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save model
os.makedirs('../../models', exist_ok=True)
joblib.dump(xgb_model, '../../models/xgboost_model.pkl')
joblib.dump(features, '../../models/xgboost_columns.pkl')   # ✅ NEW LINE HERE
print("✅ Model saved at ../../models/xgboost_model.pkl")
print("✅ Feature columns saved at ../../models/xgboost_columns.pkl")
