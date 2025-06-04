# src/training/analysis.py
import seaborn as sns
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

# Set seaborn theme directly (this is the correct way)
sns.set_theme(style="whitegrid", context="talk")

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))

# R² Bar plot
bars_r2 = ax.bar(names, r2_scores, width=0.4, label='R² Score', color=sns.color_palette("Blues_d", len(names)))
ax.set_ylabel("R² Score", color='blue')
ax.set_title("📊 Model Comparison: R² Score vs MAE", fontsize=16, weight='bold')

# Annotate R² bars
for bar, val in zip(bars_r2, r2_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=10, color='blue')

# Twin axis for MAE
ax2 = ax.twinx()
bars_mae = ax2.bar([x + 0.4 for x in range(len(names))], mae_scores, width=0.4, label='MAE', color=sns.color_palette("Reds_d", len(names)))
ax2.set_ylabel("MAE", color='red')

# Annotate MAE bars
for bar, val in zip(bars_mae, mae_scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{val:.1f}", ha='center', va='bottom', fontsize=10, color='red')

# Set custom x-ticks
ax.set_xticks([i + 0.2 for i in range(len(names))])
ax.set_xticklabels(names)

# Add legend
fig.legend(["R² Score", "MAE"], loc='upper right', bbox_to_anchor=(0.9, 0.9))
fig.tight_layout()
plt.show()

#number-03

# Define model names and metrics
names = ["Linear Regression", "Random Forest", "XGBoost"]
r2_scores = [0.76, 0.83, 0.86]
mae_scores = [25.3, 21.6, 18.9]

# Reverse for horizontal plot (y-axis is models)
names = names[::-1]
r2_scores = r2_scores[::-1]
mae_scores = mae_scores[::-1]

# Set seaborn style
sns.set_theme(style="whitegrid")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

# Bar width
height = 0.35
y = range(len(names))

# Plot R² (left of center)
bars1 = ax.barh(
    [val - height/2 for val in y],
    r2_scores,
    height=height,
    label="R² Score",
    color="#1f77b4"
)

# Plot MAE (right of center)
bars2 = ax.barh(
    [val + height/2 for val in y],
    mae_scores,
    height=height,
    label="MAE",
    color="#ff7f0e"
)

# Annotations
for bar in bars1:
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.2f}", va="center", fontsize=10, color="#1f77b4")

for bar in bars2:
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.1f}", va="center", fontsize=10, color="#ff7f0e")

# Axes and labels
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlabel("Score")
ax.set_title("Model Performance: R² vs MAE", fontsize=14)
ax.legend()
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()