from load import load_data
from clean import clean_data, save_clean_data

if __name__ == "__main__":
    raw_path = r"D:\SOURCE_CODE FILES\Python-PROGRAMS\SMTS\data\raw\traffic.csv"
    save_path = r"D:\SOURCE_CODE FILES\Python-PROGRAMS\SMTS\data\processed\traffic_clean.csv"

    df = load_data(raw_path)
    df_clean = clean_data(df)
    save_clean_data(df_clean, save_path)
