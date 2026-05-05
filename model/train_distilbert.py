"""
DistilBERT Training Script for Phishing Email Detection

This script mirrors the structure of our existing TF-IDF + Logistic Regression
training pipeline, but replaces the model with a transformer-based classifier.

Key goals:
1. Train DistilBERT to classify phishing vs legitimate emails.
2. Preserve our 3-tier output system (SAFE / SUSPICIOUS / LIKELY_PHISHING).
3. Tune thresholds using a calibration dataset.
4. Save the trained model so it can later be used by the application.

Dataset requirements:
CSV must contain:
    clean_text     -> preprocessed email text
    label_binary   -> 0 (legitimate) or 1 (phishing)
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import torch
print("CUDA available:", torch.cuda.is_available())
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Location of the processed phishing dataset used in the current system
DATA_PATH = BASE_DIR / "data" / "processed" / "phishing_emails_clean.csv"

# DistilBERT model checkpoint from HuggingFace
MODEL_NAME = "distilbert-base-uncased"

# Default thresholds for phishing tiers.
# These will be tuned later using the calibration dataset.
T_SUSPICIOUS = 0.30
T_PHISHING = 0.55

# Words commonly used in phishing emails that can help generate
# explanations for the user interface.
URGENT_WORDS = [
    "urgent", "immediately", "asap", "action required", "verify", "verification",
    "suspend", "suspended", "password", "login", "log in", "account",
    "security alert", "confirm", "locked", "unusual activity",
    "unauthorized", "reset"
]

# Directory where the trained model will be saved
MODEL_DIR = BASE_DIR / "models" / "distilbert_phishing"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Recovery / testing mode switch:
# True  -> retrain from HuggingFace base model and save a fresh good model
# False -> load the already-saved trained model from MODEL_DIR and skip retraining
TRAIN_MODE = False

# --------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------

def has_url(text: str) -> bool:
    """
    Detect whether an email contains a URL.
    This can be useful for generating explanations for the UI.
    """
    return bool(re.search(r"(https?://|www\.)", text.lower()))

def urgent_language_hits(text: str):
    """
    Return a list of urgent or threatening phrases detected
    in the email text. Used for explainability.
    """
    t = text.lower()
    hits = [w for w in URGENT_WORDS if w in t]
    return hits[:5]


# Determine a probability threshold that achieves the desired precision.
# This is useful for determining the "LIKELY_PHISHING" threshold
# where we want extremely few false positives. Same exact concept as previous training model.

# Have to change this again. At first it was not strict enough and then it was too strict and caused the recall to collapse.
def find_threshold_for_precision(y_true, probs, target_precision=0.99):
    order = np.argsort(probs)[::-1]
    y_sorted = y_true.to_numpy()[order]
    p_sorted = probs[order]

    total_pos = np.sum(y_sorted == 1)

    tp = fp = 0
    best_t = 1.0
    best_recall = -1.0

    for i in range(len(p_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / total_pos if total_pos > 0 else 0.0

        if precision >= target_precision and recall > best_recall:
            best_recall = recall
            best_t = p_sorted[i]

    return float(best_t)

def find_threshold_for_recall(y_true, probs, target_recall=0.99):
    """
    Determine a probability threshold that achieves the desired recall.
    This threshold defines the lower bound of the SUSPICIOUS category.
    """
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

    return 0.0

# edit the thresholds updated
def find_percentile_thresholds(y_true, probs, legit_pct=99, phishing_pct=10):
    """
    Determine thresholds using both classes:

    T_SUSPICIOUS = high percentile of legitimate-email probabilities
    T_PHISHING   = lower percentile of phishing-email probabilities

    This creates a middle band for uncertain emails.
    """
    y_arr = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)

    legit_probs = probs[y_arr == 0]
    phishing_probs = probs[y_arr == 1]

    if len(legit_probs) == 0 or len(phishing_probs) == 0:
        return 0.30, 0.80

    t_suspicious = float(np.percentile(legit_probs, legit_pct))
    t_phishing = float(np.percentile(phishing_probs, phishing_pct))

    # If overlap is too extreme, fall back to fixed thresholds
    if t_suspicious >= t_phishing:
        return 0.30, 0.80

    return t_suspicious, t_phishing

# Robust threshold tuning:
# Use legit-email probabilities to define the suspicious boundary,
# then search for a likely-phishing threshold that keeps precision high
# without allowing the threshold to become unrealistically strict.
def find_thresholds_robust(y_true, probs, legit_pct=99, min_precision=0.98, max_threshold=0.95):
    """
    Determine thresholds for the 3-tier system in a more stable way.

    T_SUSPICIOUS:
        High percentile of legitimate-email probabilities.
        This keeps most legitimate emails in SAFE.

    T_PHISHING:
        Search across candidate thresholds and choose the one with the
        best recall among thresholds that still satisfy a minimum precision.
        Also cap the threshold so it cannot drift too close to 1.0.
    """
    y_arr = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)

    legit_probs = probs[y_arr == 0]

    # Default lower threshold if class split is somehow broken
    if len(legit_probs) == 0:
        t_suspicious = 0.30
    else:
        t_suspicious = float(np.percentile(legit_probs, legit_pct))

    # Start with a sensible fallback value
    best_t = 0.80
    best_recall = -1.0

    # Search only within a practical range.
    # Lower bound stays above suspicious threshold and not below 0.50.
    search_start = max(t_suspicious + 1e-4, 0.50)
    search_end = max_threshold

    # If suspicious threshold is already too high, fall back safely
    if search_start >= search_end:
        return min(t_suspicious, 0.70), max(0.80, min(0.95, t_suspicious + 0.10))

    candidate_thresholds = np.linspace(search_start, search_end, 200)

    for t in candidate_thresholds:
        preds = (probs >= t).astype(int)

        tp = ((preds == 1) & (y_arr == 1)).sum()
        fp = ((preds == 1) & (y_arr == 0)).sum()
        fn = ((preds == 0) & (y_arr == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Keep the threshold that meets the precision requirement
        # while preserving as much recall as possible.
        if precision >= min_precision and recall > best_recall:
            best_recall = recall
            best_t = float(t)

    # Final safety check so likely-phishing remains above suspicious
    if best_t <= t_suspicious:
        best_t = min(max(t_suspicious + 0.10, 0.80), max_threshold)

    return t_suspicious, best_t

def tier(prob: float) -> str:

    # Convert a phishing probability into the app’s three-tier risk classification.

    if prob >= T_PHISHING:
        return "LIKELY_PHISHING"
    elif prob >= T_SUSPICIOUS:
        return "SUSPICIOUS"
    return "SAFE"


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics during training.
    Metrics tracked:
    - accuracy
    - precision !!! important
    - recall
    - F1 score
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# --------------------------------------------------------
# LOAD DATASET
# --------------------------------------------------------

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset loaded:", df.shape)
print("Processed CSV path:", DATA_PATH)

# Ensure text column exists and contains valid strings
df["clean_text"] = df["clean_text"].fillna("").astype(str)

# Convert labels to integer format
df["label_binary"] = df["label_binary"].astype(int)

X = df["clean_text"]
y = df["label_binary"]

# --------------------------------------------------------
# TRAIN / CALIBRATION / TEST SPLIT
# --------------------------------------------------------

# Hold out a final test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Reserve part of the training data for probability calibration
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print(f"Train size: {len(X_train)} | Calib size: {len(X_calib)} | Test size: {len(X_test)}")

# --------------------------------------------------------
# CONVERT DATA TO HUGGINGFACE DATASETS
# --------------------------------------------------------

train_ds = Dataset.from_pandas(
    pd.DataFrame({"text": X_train.values, "label": y_train.values}),
    preserve_index=False
)

calib_ds = Dataset.from_pandas(
    pd.DataFrame({"text": X_calib.values, "label": y_calib.values}),
    preserve_index=False
)

test_ds = Dataset.from_pandas(
    pd.DataFrame({"text": X_test.values, "label": y_test.values}),
    preserve_index=False
)

# --------------------------------------------------------
# TOKENIZATION
# --------------------------------------------------------

# Load tokenizer corresponding to DistilBERT
# If a trained tokenizer already exists, use it in test mode.
tokenizer_source = MODEL_DIR if (not TRAIN_MODE and (MODEL_DIR / "tokenizer_config.json").exists()) else MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

def tokenize(batch):
    """
    Convert raw text into token IDs used by the model.

    max_length limits sequence size to reduce computation.
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_ds = train_ds.map(tokenize, batched=True)
calib_ds = calib_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Set format for PyTorch training
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
calib_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# --------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------

# Train mode must load from the HuggingFace base checkpoint to recover the good model.
# Test mode loads the already-saved trained model from MODEL_DIR.
if TRAIN_MODE:
    model_source = MODEL_NAME
else:
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"No saved trained model found at {MODEL_DIR}. "
            "Set TRAIN_MODE = True to retrain and save a fresh model."
        )
    model_source = MODEL_DIR

model = AutoModelForSequenceClassification.from_pretrained(
    model_source,
    num_labels=2,
    id2label={0: "LEGITIMATE", 1: "PHISHING"},
    label2id={"LEGITIMATE": 0, "PHISHING": 1}
)

# --------------------------------------------------------
# TRAINING CONFIGURATION
# --------------------------------------------------------

# Remember to adjust after adding other made dataset
# save_strategy="no" avoids filling storage with checkpoint folders
# load_best_model_at_end must be False when save_strategy="no"

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR / "checkpoints"),
    evaluation_strategy="epoch",  # eval_strategy not compatible with this transformers version
    save_strategy="no",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=False,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=calib_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# --------------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------------

if TRAIN_MODE:
    trainer.train()

# --------------------------------------------------------
# THRESHOLD TUNING
# --------------------------------------------------------

calib_preds = trainer.predict(calib_ds)
calib_logits = calib_preds.predictions

calib_probs = torch.softmax(
    torch.tensor(calib_logits), dim=1
).numpy()[:, 1]

# Old threshold methods kept here for reference:
# T_PHISHING = find_threshold_for_precision(y_calib, calib_probs)
# T_SUSPICIOUS = find_threshold_for_recall(y_calib, calib_probs)

# Fixing the thresholds
# Old percentile version kept for reference:
# T_SUSPICIOUS, T_PHISHING = find_percentile_thresholds(
#     y_calib,
#     calib_probs,
#     legit_pct=99,
#     phishing_pct=10
# )

# Robust threshold tuning:
# - suspicious threshold from legit distribution
# - likely-phishing threshold from precision/recall search with guardrails
T_SUSPICIOUS, T_PHISHING = find_thresholds_robust(
    y_calib,
    calib_probs,
    legit_pct=99,
    min_precision=0.98,
    max_threshold=0.95
)

# debugging
y_calib_arr = y_calib.to_numpy() if hasattr(y_calib, "to_numpy") else np.asarray(y_calib)

legit_probs = calib_probs[y_calib_arr == 0]
phishing_probs = calib_probs[y_calib_arr == 1]

print("Legit min/max:", legit_probs.min(), legit_probs.max())
print("Phishing min/max:", phishing_probs.min(), phishing_probs.max())
print("Legit percentiles:", np.percentile(legit_probs, [90, 95, 99, 99.5, 99.9]))
print("Phishing percentiles:", np.percentile(phishing_probs, [0.1, 1, 5, 10, 25, 50]))
print(
    f"\nTuned thresholds:"
    f" T_SUSPICIOUS={T_SUSPICIOUS:.3f},"
    f" T_PHISHING={T_PHISHING:.3f}"
)

# --------------------------------------------------------
# FINAL EVALUATION
# --------------------------------------------------------

test_preds = trainer.predict(test_ds)
test_logits = test_preds.predictions

y_probs = torch.softmax(
    torch.tensor(test_logits), dim=1
).numpy()[:, 1]

tiers = [tier(p) for p in y_probs]

y_pred_strict = (y_probs >= T_PHISHING).astype(int)

print("\nSTRICT (only Likely=phishing)")
print(confusion_matrix(y_test, y_pred_strict))
print(classification_report(y_test, y_pred_strict, zero_division=0))
print("Example probs:", calib_probs[:10])

# Optional: see how many emails fall into each tier
tier_counts = pd.Series(tiers).value_counts()
print("\nTier counts:")
print(tier_counts)

# Optional: evaluate the full risky band too
# This treats both SUSPICIOUS and LIKELY_PHISHING as flagged
y_pred_risky = (y_probs >= T_SUSPICIOUS).astype(int)

print("\nRISKY (Suspicious + Likely Phishing)")
print(confusion_matrix(y_test, y_pred_risky))
print(classification_report(y_test, y_pred_risky, zero_division=0))

# --------------------------------------------------------
# SAVE FINAL TEST RESULTS FOR PRESENTATION
# --------------------------------------------------------

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

results_dir = BASE_DIR / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Save per-email test results
results_df = pd.DataFrame({
    "email_text": X_test.values,
    "true_label": y_test.values,
    "phishing_probability": y_probs,
    "tier": tiers,
    "predicted_strict": y_pred_strict,
    "predicted_risky": y_pred_risky
})

results_df.to_csv(results_dir / "distilbert_test_predictions.csv", index=False)

# Save classification reports
with open(results_dir / "distilbert_test_report.txt", "w") as f:
    f.write("STRICT MODE: LIKELY_PHISHING only\n")
    f.write(str(confusion_matrix(y_test, y_pred_strict)))
    f.write("\n\n")
    f.write(classification_report(y_test, y_pred_strict, zero_division=0))

    f.write("\n\nRISKY MODE: SUSPICIOUS + LIKELY_PHISHING\n")
    f.write(str(confusion_matrix(y_test, y_pred_risky)))
    f.write("\n\n")
    f.write(classification_report(y_test, y_pred_risky, zero_division=0))

# Save confusion matrix image for slides
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_risky,
    display_labels=["Legitimate", "Phishing"]
)
plt.title("DistilBERT Confusion Matrix - Risky Mode")
plt.savefig(results_dir / "distilbert_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved presentation results to:", results_dir)

# --------------------------------------------------------
# SAVE MODEL FOR APPLICATION
# --------------------------------------------------------

# When using in app need a separate "prediction wrapper" since it is not a joblib.
# Only save after an actual training run so we do not overwrite a good model during threshold-only testing.
if TRAIN_MODE:
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    artifact_config = {
        "model_name": MODEL_NAME,
        "thresholds": {
            "T_SUSPICIOUS": float(T_SUSPICIOUS),
            "T_PHISHING": float(T_PHISHING)
        },
        "urgent_words": URGENT_WORDS,
        "max_length": 256
    }

    with open(MODEL_DIR / "artifact_config.json", "w") as f:
        json.dump(artifact_config, f, indent=2)

    print("Saved DistilBERT model to:", MODEL_DIR)
else:
    print("Test mode: model not re-saved.")