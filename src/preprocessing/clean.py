import pandas as pd

def clean_data(df):
    # Convert timestamp if needed
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Create additional features
    df['hour'] = df['DateTime'].dt.hour
    df['day'] = df['DateTime'].dt.dayofweek

    # Handle missing values
    df = df.dropna()

    return df
def save_clean_data(df, path):
    df.to_csv(path, index=False)
    print(f" Saved cleaned data to: {path}")
