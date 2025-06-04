import pandas as pd
import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# Model training
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model trained.")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Plotting actual vs predicted
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.xlabel("Actual Vehicle Count")
plt.ylabel("Predicted Vehicle Count")
plt.title("Actual vs Predicted Vehicles")
plt.grid(True)
plt.show()


os.makedirs('../../models', exist_ok=True)
joblib.dump(model,'../../models/linear_regression.pkl')
print("✅ Linear Regression model saved as linear_regression.pkl")