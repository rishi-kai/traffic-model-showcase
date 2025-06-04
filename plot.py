import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/processed/traffic_features.csv", parse_dates=["DateTime"])

# Optional: sanity check
print("Columns in DataFrame:", df.columns.tolist())
print(df.head())

sns.set(style="whitegrid")

# 1. Traffic volume over time
plt.figure(figsize=(12, 5))
sns.lineplot(data=df, x="DateTime", y="Vehicles")
plt.title("Traffic Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.show(block=True)

# 2. Traffic by hour of day
plt.figure(figsize=(10, 5))
sns.boxplot(x="hour", y="Vehicles", data=df)
plt.title("Traffic Volume by Hour of the Day")
plt.xlabel("Hour")
plt.ylabel("Vehicles")
plt.tight_layout()
plt.show(block=True)

# 3. Traffic by weekday
plt.figure(figsize=(10, 5))
sns.barplot(x="weekday", y="Vehicles", data=df)
plt.title("Traffic Volume by Weekday")
plt.xlabel("Weekday (0=Monday)")
plt.ylabel("Vehicles")
plt.tight_layout()
plt.show(block=True)
