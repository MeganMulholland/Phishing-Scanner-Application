import os
import pandas as pd
from preprocess.preprocess import preprocess_email
from bs4 import MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

    all_dfs = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            file_path = os.path.join(DATA_DIR, file)
            print(f"Loading {file}...")
            df_temp = pd.read_csv(file_path)
            df_temp["source"] = file
            all_dfs.append(df_temp)

    df = pd.concat(all_dfs, ignore_index=True)

    print("\nColumns:")
    print(df.columns)

    print("\nLabel values:")
    print(df["label"].unique())
    print("Label dtype:", df["label"].dtype)

    print("\nSample rows:")
    print(df.head())

    df["label_binary"] = df["label"]
    df = df.dropna(subset=["label_binary"])

    df["clean_text"] = df["body"].astype(str).apply(preprocess_email)

    print("\n--- BEFORE ---")
    print(df["body"].iloc[0][:300])

    print("\n--- AFTER ---")
    print(df["clean_text"].iloc[0][:300])

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    OUTPUT_PATH = os.path.join(PROCESSED_DIR, "phishing_emails_clean.csv")

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nProcessed dataset saved to: {OUTPUT_PATH}")
    print("Total usable emails:", len(df))