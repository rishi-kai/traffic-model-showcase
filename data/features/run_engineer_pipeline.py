import os
import pandas as pd
from engineer import engineer_features

# Load the clean data
df = pd.read_csv("../processed/traffic_clean.csv")

# Apply feature engineering
df = engineer_features(df)

# Ensure the processed directory exists
os.makedirs("data/processed", exist_ok=True)

# Save the output
df.to_csv("../processed/traffic_features.csv", index=False)


print("✅ Feature engineering complete. Saved to 'data/processed/traffic_features.csv'")
