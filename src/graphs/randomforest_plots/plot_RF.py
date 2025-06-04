import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv(
    "../../../data/processed/traffic_features.csv", parse_dates=["DateTime"])

# Sort by DateTime for time series plot
df = df.sort_values("DateTime")

# Load trained model
rf_model = joblib.load(
    "../../../models/random_forest.pkl")

# Use same features as used in training (order matters!)
X = df.drop(columns=["DateTime", "Vehicles", "ID"])

# Predict
df['Predicted_Vehicles'] = rf_model.predict(X)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df['DateTime'], df['Vehicles'], label='Actual', alpha=0.7)
plt.plot(df['DateTime'], df['Predicted_Vehicles'], label='Predicted', alpha=0.7)
plt.xlabel("DateTime")
plt.ylabel("Vehicle Count")
plt.title("Actual vs Predicted Vehicle Count (Random Forest)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
