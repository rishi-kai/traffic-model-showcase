# src/training/analysis.py

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("../../data/processed/traffic_features.csv", parse_dates=["DateTime"])
print("✅ Data loaded.")

# Define consistent features and target
features = ['hour', 'day', 'month', 'weekday', 'is_weekend', 'Junction']
target = 'Vehicles'

X = df[features]
y = df[target]

# Split into test set (same for all models)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model filenames and labels
models_info = [
    ("Linear Regression", "../../models/linear_regression.pkl"),
    ("Random Forest", "../../models/random_forest.pkl"),
    ("XGBoost", "../../models/xgboost_model.pkl")
]

# Store metrics
results = []

# Evaluate each model
for name, path in models_info:
    try:
        model = joblib.load(path)
        # Reorder columns if necessary (to match training order)
        if hasattr(model, "feature_names_in_"):
            X_test_ordered = X_test[model.feature_names_in_]
        else:
            X_test_ordered = X_test  # fallback for models without feature_names_in_

        y_pred = model.predict(X_test_ordered)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"\n🔍 {name} Performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: {mae:.2f}")

        results.append((name, r2, mae))
    except Exception as e:
        print(f"❌ Error evaluating {name}: {e}")

# Plot comparison
if results:
    names = [x[0] for x in results]
    r2_scores = [x[1] for x in results]
    mae_scores = [x[2] for x in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(x - width/2, r2_scores, width, label='R² Score', color='skyblue')
    ax2.bar(x + width/2, mae_scores, width, label='MAE', color='salmon')

    ax1.set_xlabel('Models')
    ax1.set_ylabel('R² Score')
    ax2.set_ylabel('MAE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_title('Model Performance Comparison')

    fig.legend(loc='upper center', ncol=2)
    fig.tight_layout()
    plt.show()
