import os
import re
import numpy as np
import pandas as pd

# Scikit-learn imports we need
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import FeatureUnion
from typing import Union

# ******************************
# CONFIGURATION / CONSTANTS
# **********************************

# Base directory (so paths work regardless of where script is run)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to cleaned dataset created by preprocessing script
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "phishing_emails_clean.csv")

# Tier thresholds (empirically chosen)
# SAFE        : p < 0.30
# SUSPICIOUS  : 0.30 <= p < 0.55
# LIKELY_PHISHING : p >= 0.55
T_SUSPICIOUS = 0.30
T_PHISHING = 0.55

# Common urgency / phishing-related terms (used for explanations)
URGENT_WORDS = [
    "urgent", "immediately", "asap", "action required", "verify", "verification",
    "suspend", "suspended", "password", "login", "log in", "account",
    "security alert", "confirm", "locked", "unusual activity",
    "unauthorized", "reset"
]


# *************************************************
# EXPLANATION HELPERS (used for interpretability)
# **************************************************

def has_url(text: str) -> bool:
   # Simple heuristic: does the email have a url?
    return bool(re.search(r"(https?://|www\.)", text.lower()))

def urgent_language_hits(text: str):
    """
    Returns a short list of urgent or threatening phrases
    found in the email text.
    """
    t = text.lower()
    hits = [w for w in URGENT_WORDS if w in t]
    return hits[:5]  # keep explanations concise

### Adjustments/Helper functions to filter explanations further
MONTH_DAY_STOP = {
    "mon","tue","tues","wed","thu","thur","thurs","fri","sat","sun",
    "jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec"
}

def is_good_feature(feat: str) -> bool:
    if "::" in feat:
        kind, token = feat.split("::", 1)
    else:
        kind, token = "word_tfidf", feat

    raw = token.lower()          # keep raw form for whitespace checks
    tok = token.strip().lower()  # normalized form for other checks

    # Drop very short word tokens
    if kind.startswith("word") and len(tok) < 3:
        return False

    # Drop day/month artifacts (like "jul" or "tue jul")
    if kind.startswith("word"):
        parts = tok.split()
        if parts and all(p in MONTH_DAY_STOP for p in parts):
            return False

    # KEEP numbers — only drop obvious hashes / IDs (long mostly-alnum strings)
    alnum = sum(c.isalnum() for c in tok)
    if len(tok) >= 18 and (alnum / max(1, len(tok))) > 0.9:
        return False

    # Clean up char n-grams: reject any that include whitespace anywhere
    if kind.startswith("char"):
        if any(c.isspace() for c in raw):
            return False

        letters = sum(c.isalpha() for c in tok)
        if letters == 0:
            return False

    return True

def top_weighted_phrases(
    text: str,
    vectorizer: Union[TfidfVectorizer, FeatureUnion], #Can choose which one to use
    base_model: LogisticRegression,
    top_k: int = 3
):
    # Identifies the top phrases contributing to a phishing prediction.

    # Contribution is approximated as:
    #   TF-IDF(feature) * model coefficient(feature)
    #This keeps explanations fully interpretable.

    x = vectorizer.transform([text])
    row = x.tocoo()

    # If no features fire, return empty explanation.
    # We can edit our features later, add more words etc
    if row.nnz == 0:
        return []

    coefs = base_model.coef_.ravel()
    contributions = row.data * coefs[row.col]

    # Sort features by descending contribution
    order = np.argsort(contributions)[::-1]

    # Can use character n grams or not
    if isinstance(vectorizer, FeatureUnion):
        feature_names = []
        for name, vec in vectorizer.transformer_list:
            fn = vec.get_feature_names_out()
            feature_names.extend(f"{name}::{f}" for f in fn)
    else:
        feature_names = vectorizer.get_feature_names_out()

## Changed this filter explanations more
    phrases = []
    for idx in order:
        feat = feature_names[row.col[idx]]
        score = float(contributions[idx])

        if score <= 0:
            continue

        if not is_good_feature(feat):
            continue

        phrases.append((feat, score))
        if len(phrases) >= top_k:
            break

    # Fallback if filtering removed everything
    if len(phrases) == 0:
        for idx in order:
            feat = feature_names[row.col[idx]]
            score = float(contributions[idx])
            if score > 0:
                phrases.append((feat, score))
            if len(phrases) >= top_k:
                break

    return phrases

# Helper functions to create new thresholds.
def find_threshold_for_precision(y_true, probs, target_precision=0.99):
    order = np.argsort(probs)[::-1]
    y_sorted = y_true.to_numpy()[order]
    p_sorted = probs[order]

    tp = fp = 0
    best_t = 1.0

    for i in range(len(p_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        if precision >= target_precision:
            best_t = p_sorted[i]

    return float(best_t)


def find_threshold_for_recall(y_true, probs, target_recall=0.99):
    y_arr = y_true.to_numpy()
    total_pos = (y_arr == 1).sum()
    if total_pos == 0:
        return 0.5

    order = np.argsort(probs)[::-1]
    y_sorted = y_arr[order]
    p_sorted = probs[order]

    tp = 0
    for i in range(len(p_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        recall = tp / total_pos
        if recall >= target_recall:
            return float(p_sorted[i])

    return 0.

def tier(prob: float) -> str:
    # Converts the calibrated phishing prob into the 3 tiers.
    if prob >= T_PHISHING:
        return "LIKELY_PHISHING"
    elif prob >= T_SUSPICIOUS:
        return "SUSPICIOUS"
    return "SAFE"


# **********
# Load
# ***********

df = pd.read_csv(DATA_PATH, low_memory=False)
print("Dataset loaded:", df.shape)

# Checking our dataset.
print("Processed CSV path:", DATA_PATH)

# Check empties (should be 0 now that you filtered upstream)
empty_ct = (df["clean_text"].fillna("").astype(str).str.strip() == "").sum()
print("Empty clean_text rows:", empty_ct)

# Label distribution check
print("Label distribution:\n", df["label_binary"].value_counts())

# Duplicate leakage risk check (exact duplicates)
dup_ct = df.duplicated(subset=["clean_text"]).sum()
print("Exact duplicate clean_text rows:", dup_ct)
print("Unique clean_text:", df["clean_text"].nunique(), "out of", len(df))

# Ensure clean text and labels are well-formed
df["clean_text"] = df["clean_text"].fillna("").astype(str)
X = df["clean_text"]
y = df["label_binary"].astype(int)  # force labels to 0/1 ints

# ****************************************
# SPLIT DATA: TRAIN / CALIBRATION / TEST
# ******************************************

# First split: hold out final test set. Same as before
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Second split: reserve part of training data for probability calibration
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print(f"Train size: {len(X_train)} | Calib size: {len(X_calib)} | Test size: {len(X_test)}")


# ******************************
# TEXT VECTORIZATION (WORD + CHAR TF-IDF)
# ********************************
# Added character n grams
vectorizer = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=8000,
            stop_words="english"
        )
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=5,
            max_features=4000
        )
    ),
])

# Fit ONLY on training data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_calib_tfidf = vectorizer.transform(X_calib)
X_test_tfidf = vectorizer.transform(X_test)


# **********************************
# TRAIN BASE MODEL (INTERPRETABLE)
# **********************************

# Logistic Regression is fast, explainable, and works well with TF-IDF
base_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

base_model.fit(X_train_tfidf, y_train)


# *********************************************
# CALIBRATE PROBABILITIES (PLATT SCALING)
# *********************************************

# Calibration makes predicted probabilities more meaningful and stable. Important for our app. Added frozen estimator to fix FutureWarnin

cal_model = CalibratedClassifierCV(
    FrozenEstimator(base_model),
    method="sigmoid",
    cv=5
)
cal_model.fit(X_calib_tfidf, y_calib)

#Tune thresholds using calibration data. Solves the problem of scattered thresholds
cal_probs = cal_model.predict_proba(X_calib_tfidf)[:, 1]

T_PHISHING = find_threshold_for_precision(
    y_calib, cal_probs, target_precision=0.99
)

T_SUSPICIOUS = find_threshold_for_recall(
    y_calib, cal_probs, target_recall=0.99
)

# Safety: enforce ordering
if T_SUSPICIOUS > T_PHISHING:
    T_SUSPICIOUS = max(0.0, T_PHISHING - 0.05)

print(
    f"\nTuned thresholds:"
    f" T_SUSPICIOUS={T_SUSPICIOUS:.3f},"
    f" T_PHISHING={T_PHISHING:.3f}"
)

# ******************************************
# Evaluation with the calibrated probabilities
# *********************************************

# Calibrated probability that an email is phishing
y_probs = cal_model.predict_proba(X_test_tfidf)[:, 1]

# Assign 3 tier labels
tiers = [tier(p) for p in y_probs]

# Binary views for reporting only!!
# Options A and B
# Security mode: suspicious + likely
y_pred_security = (y_probs >= T_SUSPICIOUS).astype(int)

# Strict mode: likely phishing only
y_pred_strict = (y_probs >= T_PHISHING).astype(int)


print("\nSTRICT (only Likely=phishing)")
print(confusion_matrix(y_test, y_pred_strict))
print(classification_report(y_test, y_pred_strict))

print("\nSECURITY (Suspicious OR Likely=phishing)")
print(confusion_matrix(y_test, y_pred_security))
print(classification_report(y_test, y_pred_security))

print("\nTier counts:")
print(pd.Series(tiers).value_counts())


# *********************
# Demo explanations
# ************************

# Create a small df for inspection/test
test_df = pd.DataFrame({
    "text": X_test.values,
    "true_label": y_test.values,
    "probability": y_probs,
    "tier": tiers
})

# Show explanations for a few flagged emails
flagged = test_df[(test_df["tier"] != "SAFE") & (test_df["text"].str.strip() != "")].head(5)
print("\n--- Sample explanations (first 5 flagged emails) ---")

for _, row in flagged.iterrows():
    text = row["text"]
    prob = row["probability"]
    t = row["tier"]
    true_label = row["true_label"]

    phrases = top_weighted_phrases(text, vectorizer, base_model, top_k=3)
    url_flag = has_url(text)
    urgent_hits = urgent_language_hits(text)

    print("\n----------------------------------------")
    print(f"True label: {true_label}")
    print(f"Calibrated score: {prob:.4f}")
    print(f"Tier: {t}")
    print(f"Contains URL: {url_flag}")
    print(f"Urgent language detected: {urgent_hits}")
    print(f"Top contributing phrases: {phrases}")
    print("Email preview:", text[:250].replace("\n", " "))


######################
# Adding a joblib file
# lets app just load one file and work
##########################
import joblib
from pathlib import Path

MODEL_DIR = Path(BASE_DIR) / "models"
MODEL_DIR.mkdir(exist_ok=True)

artifact = {
    "vectorizer": vectorizer,
    "base_model": base_model,   # for top_weighted_phrases()
    "cal_model": cal_model,     # for predict_proba()
    "thresholds": {
        "T_SUSPICIOUS": T_SUSPICIOUS,
        "T_PHISHING": T_PHISHING,
    },
    "urgent_words": URGENT_WORDS,
}

joblib.dump(artifact, MODEL_DIR / "phishing_artifact.joblib")
print("Saved model artifact to:", MODEL_DIR / "phishing_artifact.joblib")