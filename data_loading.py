import os
import pandas as pd
from preprocessing.preprocess import preprocess_email
from bs4 import MarkupResemblesLocatorWarning
import warnings
#beautiful soup error fix
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
# Load and combine all CSV files


DATA_DIR = "../data/raw"

all_dfs = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        file_path = os.path.join(DATA_DIR, file)
        print(f"Loading {file}...")
        df_temp = pd.read_csv(file_path)
        df_temp["source"] = file  # track dataset source
        all_dfs.append(df_temp)

# combines datasets into one dataframe
df = pd.concat(all_dfs, ignore_index=True)

print("\nColumns:")
print(df.columns)

print("\nLabel values:")
print(df["label"].unique())
print("Label dtype:", df["label"].dtype)

print("\nSample rows:")
print(df.head())


# standardize labels
# (Already numeric 0/1)

df["label_binary"] = df["label"]

# Drop rows with missing labels
df = df.dropna(subset=["label_binary"])

# applies preprocessing
df["clean_text"] = df["body"].astype(str).apply(preprocess_email)

print("\n--- BEFORE ---")
print(df["body"].iloc[0][:300])

print("\n--- AFTER ---")
print(df["clean_text"].iloc[0][:300])


# saves processed dataset


OUTPUT_PATH = "../data/processed/phishing_emails_clean.csv"

# makes sure processed folder exists
os.makedirs("../data/processed", exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nProcessed dataset saved to: {OUTPUT_PATH}")
print("Total usable emails:", len(df))