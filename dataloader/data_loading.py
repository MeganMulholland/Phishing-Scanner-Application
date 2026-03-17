"""
Use this for documentation purposes:
Purpose:
- Load all raw phishing/legitimate email CSVs from data/raw
- Standardize datasets that use different column names
- Combine subject + body into one text field
- Apply shared preprocessing
- Save one clean training file to data/processed/phishing_emails_clean.csv

What changed from the older version:
1. Fixed BASE_DIR so it works with the new folder structure where this script
   lives inside /dataloader instead of the project root.
2. Added schema standardization so datasets with:
      - body or text
      - label_binary or label or target
   can all be merged safely.
3. Preserved subject when available instead of ignoring it.
4. Added a source column to track which CSV each row came from.
5. Used swifter to speed up preprocessing on larger datasets.
"""

from pathlib import Path
import warnings
import pandas as pd
import swifter
from bs4 import MarkupResemblesLocatorWarning

from preprocessing.preprocess import preprocess_email

# Get rid of a common BeautifulSoup warning that appears when text looks like a file path
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
# This script is in: project_root/dataloader/data_loading.py
# parents[1] moves up to the project root.
BASE_DIR = Path(__file__).resolve().parents[1]

# Raw input datasets and processed output folder
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "phishing_emails_clean.csv"

# Ensure processed output directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("OUTPUT_PATH:", OUTPUT_PATH)

# --------------------------------------------------
# HELPER: STANDARDIZE ONE DATAFRAME
# --------------------------------------------------
def standardize_dataset(df_temp: pd.DataFrame, source_name: str) -> pd.DataFrame | None:
    """
    Convert one raw dataset into a shared schema:
    Final columns:
    - subject
    - body
    - label_binary
    - source

    Returns:
        standardized DataFrame, or None if no usable label column exists
    """

    df_temp = df_temp.copy()
    df_temp["source"] = source_name

    # -------------------------------
    # Standardize subject
    # -------------------------------
    # If a dataset does not have subject, create an empty subject column.
    if "subject" not in df_temp.columns:
        df_temp["subject"] = ""

    # -------------------------------
    # Standardize body/text
    # -------------------------------
    # Some datasets use "body", others store the full email in "text".
    if "body" in df_temp.columns:
        df_temp["body"] = df_temp["body"]
    elif "text" in df_temp.columns:
        df_temp["body"] = df_temp["text"]
    else:
        df_temp["body"] = ""

    # -------------------------------
    # Standardize labels
    # -------------------------------
    # Different datasets use different names for the same binary label.
    if "label_binary" in df_temp.columns:
        df_temp["label_binary"] = df_temp["label_binary"]
    elif "label" in df_temp.columns:
        df_temp["label_binary"] = df_temp["label"]
    elif "target" in df_temp.columns:
        df_temp["label_binary"] = df_temp["target"]
    else:
        print(f"Skipping {source_name}: no usable label column found")
        return None

    # Keep only the columns needed by the rest of the pipeline.
    return df_temp[["subject", "body", "label_binary", "source"]]


# --------------------------------------------------
# LOAD AND STANDARDIZE ALL RAW CSV FILES
# --------------------------------------------------
all_dfs = []

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Raw data directory not found: {DATA_DIR}")

for file_path in DATA_DIR.glob("*.csv"):
    print(f"\nLoading {file_path.name}...")

    try:
        df_temp = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Skipping {file_path.name}: could not read file ({e})")
        continue

    standardized = standardize_dataset(df_temp, file_path.name)
    if standardized is not None:
        all_dfs.append(standardized)
        print(f"Added {len(standardized)} rows from {file_path.name}")

if not all_dfs:
    raise ValueError("No usable CSV files were found in data/raw")

# Combine all standardized datasets into one dataframe
df = pd.concat(all_dfs, ignore_index=True)

print("\nCombined raw rows:", len(df))

# --------------------------------------------------
# CLEAN LABELS
# --------------------------------------------------
# Drop rows where label is missing
before = len(df)
df = df.dropna(subset=["label_binary"])
after = len(df)
print(f"Dropped {before - after} rows with missing labels")

# Force labels to integer 0/1 format
# This assumes your label/target columns are already binary-compatible.
df["label_binary"] = df["label_binary"].astype(int)

# --------------------------------------------------
# CLEAN TEXT FIELDS
# --------------------------------------------------
df["subject"] = df["subject"].fillna("").astype(str)
df["body"] = df["body"].fillna("").astype(str)

# Combine subject + body so the model can learn from both.
# This is especially useful for phishing because subjects often contain
# urgency language or account warnings.
df["email_text"] = (df["subject"] + " " + df["body"]).str.strip()

# --------------------------------------------------
# PREPROCESS EMAIL TEXT
# --------------------------------------------------
# CHANGED:
# Older code applied preprocessing row-by-row with plain pandas .apply().
# This version uses swifter, which can speed up large datasets by choosing
# a faster execution strategy automatically.
print("\nPreprocessing email text...")
df["clean_text"] = df["email_text"].swifter.apply(preprocess_email)
df["clean_text"] = df["clean_text"].astype(str).str.strip()

# --------------------------------------------------
# REMOVE EMPTY / DUPLICATE ROWS
# --------------------------------------------------
before = len(df)
df = df[df["clean_text"] != ""]
after = len(df)
print(f"Dropped {before - after} rows with empty clean_text")

before = len(df)
df = df.drop_duplicates(subset=["clean_text"])
after = len(df)
print(f"Dropped {before - after} duplicate emails based on clean_text")

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------
print("\nFinal dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label_binary"].value_counts(dropna=False))

print("\nRows by source:")
print(df["source"].value_counts())

# --------------------------------------------------
# SAVE PROCESSED DATASET
# --------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nProcessed dataset saved to: {OUTPUT_PATH}")
print("Total usable emails:", len(df))