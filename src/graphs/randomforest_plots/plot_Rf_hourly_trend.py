import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv( "../../../data/processed/traffic_features.csv", parse_dates=["DateTime"])



# Load model
rf_model = joblib.load( "../../../models/random_forest.pkl")

# Use same features
X = df.drop(columns=["DateTime", "Vehicles", "ID"])

# Predict
df["Predicted_Vehicles"] = rf_model.predict(X)

# Group by hour to observe trend over time of day
hourly_actual = df.groupby("hour")["Vehicles"].mean()
hourly_predicted = df.groupby("hour")["Predicted_Vehicles"].mean()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(hourly_actual.index, hourly_actual.values, label="Actual", marker='o')
plt.plot(hourly_predicted.index, hourly_predicted.values, label="Predicted", marker='x')
plt.title("Average Vehicle Count by Hour of Day (Random Forest)")
plt.xlabel("Hour of Day")
plt.ylabel("Average Vehicle Count")
plt.xticks(range(0, 24))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
