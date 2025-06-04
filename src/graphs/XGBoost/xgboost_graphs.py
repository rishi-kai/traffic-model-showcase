# src/graphs/XGBoost/xgboost_graphs.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../../../data/processed/traffic_features.csv", parse_dates=["DateTime"])
print("✅ Data loaded. Shape:", df.shape)

# Load model and features
model = joblib.load("../../../models/xgboost_model.pkl")
feature_columns = joblib.load("../../../models/xgboost_columns.pkl")

# Predict
X = df[feature_columns]
df["Predicted_Vehicles"] = model.predict(X)

# === 1. Actual vs Predicted Scatter Plot ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Vehicles"], y=df["Predicted_Vehicles"], alpha=0.3)
plt.xlabel("Actual Vehicle Count")
plt.ylabel("Predicted Vehicle Count")
plt.title("XGBoost - Actual vs Predicted Vehicles")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 2. Line Plot: DateTime vs Actual & Predicted Vehicles ===
plt.figure(figsize=(12, 6))
sample_df = df.sort_values("DateTime").iloc[:200]  # optional slicing for clarity
plt.plot(sample_df["DateTime"], sample_df["Vehicles"], label="Actual", linewidth=2)
plt.plot(sample_df["DateTime"], sample_df["Predicted_Vehicles"], label="Predicted", linestyle="--")
plt.xlabel("DateTime")
plt.ylabel("Vehicle Count")
plt.title("XGBoost - Actual vs Predicted Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 3. Residual Plot (Prediction Error) ===
df["Residuals"] = df["Vehicles"] - df["Predicted_Vehicles"]
plt.figure(figsize=(8, 5))
sns.histplot(df["Residuals"], bins=50, kde=True)
plt.title("XGBoost - Residuals Distribution")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
