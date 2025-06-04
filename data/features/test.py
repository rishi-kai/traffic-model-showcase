import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


print("Current working dir:", os.getcwd())


print("Looking for CSV at:", csv_path)  # Debug

# Load CSV
df = pd.read_csv(csv_path, parse_dates=["DateTime"])

# Load engineered features
df = pd.read_csv("../../../processed/traffic_features.csv", parse_dates=['DateTime'])

# Set plot style
sns.set(style="whitegrid")

# 1. Traffic volume by hour of day
plt.figure(figsize=(10, 5))
sns.boxplot(x="hour", y="Vehicles", data=df)
plt.title("Vehicle Count by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.show()

# 2. Traffic volume by weekday
plt.figure(figsize=(10, 5))
sns.boxplot(x="weekday", y="Vehicles", data=df)
plt.title("Vehicle Count by Day of Week (0=Monday)")
plt.xlabel("Weekday")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.show()

# 3. Traffic volume by month
plt.figure(figsize=(10, 5))
sns.boxplot(x="month", y="Vehicles", data=df)
plt.title("Vehicle Count by Month")
plt.xlabel("Month")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.show()

# 4. Traffic volume over time (small sample)
plt.figure(figsize=(15, 5))
sample = df[df["Junction"] == 1].sort_values("DateTime").head(500)
plt.plot(sample["DateTime"], sample["Vehicles"])
plt.title("Traffic Volume Over Time (Junction 1)")
plt.xlabel("DateTime")
plt.ylabel("Vehicles")
plt.tight_layout()
plt.show()
