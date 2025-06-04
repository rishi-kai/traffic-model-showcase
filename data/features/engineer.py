import pandas as pd

def engineer_features(df):
    df['DateTime'] = pd.to_datetime(df['DateTime'])  # <-- 🧙‍♂️ Magic line

    df['month'] = df['DateTime'].dt.month
    df['day'] = df['DateTime'].dt.day
    df['hour'] = df['DateTime'].dt.hour
    df['weekday'] = df['DateTime'].dt.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    return df
