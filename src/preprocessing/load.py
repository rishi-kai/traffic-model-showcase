import pandas as pd
import os


def load_data(
        filepath=r"D:\SOURCE_CODE FILES\Python-PROGRAMS\SMTS\data\raw\traffic.csv"):
    if filepath is None:
        filepath = os.path.join(
            "data",
            "raw",
            "Train.csv")

    df = pd.read_csv(
        filepath)
    print(
        "Data Loaded:")
    print(
        df.info())
    print(
        df.head())

    return df
if __name__ == "__main__":
    filepath = r"D:\SOURCE_CODE FILES\Python-PROGRAMS\SMTS\data\raw\traffic.csv"
    load_data(filepath)