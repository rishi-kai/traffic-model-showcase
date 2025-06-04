# explore.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing.load import load_data
from src.preprocessing.clean import clean_data

# Load and clean the data
df = load_data()
df = clean_data(df)

# Show basic structure
print("\n🔍 Data Overview:")
print(df.head())

print("\n📊 Data Types:")
print(df.dtypes)

print("\n📈 Missing Values:")
print(df.isnull().sum())

print("\n📋 Summary Statistics:")
print(df.describe(include='all'))

# Convert DateTime if needed
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Add useful time-based features
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['date'] = df['DateTime'].dt.date

# Total vehicles per junction
plt.figure(figsize=(10, 6))
sns.boxplot(x='Junction', y='Vehicles', data=df)
plt.title('Vehicle Count Distribution per Junction')
plt.ylabel("Vehicle Count")
plt.show()

# Time series plot: traffic trend at each junction
plt.figure(figsize=(14, 6))
for junction in sorted(df['Junction'].unique()):
    junction_df = df[df['Junction'] == junction]
    plt.plot(junction_df['DateTime'], junction_df['Vehicles'], label=f'Junction {junction}')

plt.title("Traffic Volume Over Time at Each Junction")
plt.xlabel("Date")
plt.ylabel("Vehicle Count")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: set plot style
sns.set(style="whitegrid")

# 1️⃣ Vehicle Count Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Vehicles'], kde=True, bins=30, color='skyblue')
plt.title('Vehicle Count Distribution')
plt.xlabel('Number of Vehicles')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('vehicle_count_distribution.png')  # Save to file
plt.show()

# 2️⃣ Traffic Volume at Time per Junction
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='hour', y='Vehicles', hue='Junction', estimator='mean', ci=None)
plt.title('Average Traffic Volume by Hour at Each Junction')
plt.xlabel('Hour of Day')
plt.ylabel('Average Number of Vehicles')
plt.legend(title='Junction')
plt.tight_layout()
plt.savefig('traffic_volume_by_hour_and_junction.png')
plt.show()