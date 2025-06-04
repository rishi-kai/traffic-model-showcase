import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score

# Load model
model_path = os.path.join("../../../models/linear_regression.pkl")
model = joblib.load(model_path)
print("✅ Model loaded.")

# Load dataset
df = pd.read_csv("../../../data/processed/traffic_features.csv", parse_dates=["DateTime"])
features = ['hour', 'day', 'month', 'weekday', 'is_weekend', 'Junction']
target = 'Vehicles'

X = df[features]
y = df[target]

# Make predictions
df["Predicted"] = model.predict(X)

# Plot 1: Actual vs Predicted scatter plot
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y, y=df["Predicted"], alpha=0.3)
plt.xlabel("Actual Vehicle Count")
plt.ylabel("Predicted Vehicle Count")
plt.title("Actual vs Predicted Vehicle Count (Linear Regression)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Time-based comparison (only for one junction, e.g., Junction 1)
sample_df = df[df["Junction"] == 1].sort_values("DateTime").head(200)

plt.figure(figsize=(14, 6))
plt.plot(sample_df["DateTime"], sample_df["Vehicles"], label="Actual", alpha=0.7)
plt.plot(sample_df["DateTime"], sample_df["Predicted"], label="Predicted", alpha=0.7)
plt.title("📈 Vehicle Count Over Time at Junction 1")
plt.xlabel("DateTime")
plt.ylabel("Vehicle Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

# Plot 3: Residuals plot
df["Residuals"] = y - df["Predicted"]
plt.figure(figsize=(10, 5))
sns.histplot(df["Residuals"], bins=50, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluation
mse = mean_squared_error(y, df["Predicted"])
r2 = r2_score(y, df["Predicted"])
print(f"📊 MSE: {mse:.2f}")
print(f"📈 R² Score: {r2:.4f}")
