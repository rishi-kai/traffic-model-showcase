import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
rf_model = joblib.load("../../../models/random_forest.pkl")

# Create synthetic input for different hours of the day
hours = list(range(24))
weekday = 2  # Wednesday
is_weekend = 0  # Not a weekend

# Create synthetic dataframe
X_synthetic = pd.DataFrame({
    "hour": hours,
    "day": [15] * 24,
    "month": [6] * 24,
    "weekday": [weekday] * 24,
    "is_weekend": [is_weekend] * 24,
    "Junction": [1] * 24  # Choose a junction
})

# Reorder columns to match training feature order
X_synthetic = X_synthetic[["Junction", "hour", "day", "month", "weekday", "is_weekend"]]

# Predict
predicted_traffic = rf_model.predict(X_synthetic)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(hours, predicted_traffic, marker='o', linestyle='-', color='teal')
plt.xticks(hours)
plt.xlabel("Hour of Day")
plt.ylabel("Predicted Vehicle Count")
plt.title("Predicted Traffic Pattern Over a Day (Weekday)")
plt.grid(True)
plt.tight_layout()
plt.show()
