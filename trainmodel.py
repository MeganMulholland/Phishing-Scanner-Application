import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

# Load cleaned dataset
#DATA_PATH = "../data/processed/phishing_emails_clean.csv"
# changes this the same as preprocess to match any folder set up
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "phishing_emails_clean.csv")

df = pd.read_csv(DATA_PATH, low_memory=False) # make sure it looks through all the data before deciding a type

print("Dataset loaded:", df.shape)

# Features & Labels

df["clean_text"] = df["clean_text"].fillna("").astype(str) #Fill in nas make sure everything is a string

X = df["clean_text"]
y = df["label_binary"]


# Train/Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify = y # stratify lets both sets keep the same ratio
)


# TF-IDF Vectorization


vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train Model


model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)


# Evaluate
#y_pred = model.predict(X_test_tfidf)

# Evaluate (custom threshold)

# Evaluate (3-tier scoring)

y_probs = model.predict_proba(X_test_tfidf)[:, 1]  # P(phishing)

T_SUSPICIOUS = 0.30
T_PHISHING = 0.55

def tier(prob):
    if prob >= T_PHISHING:
        return "LIKELY_PHISHING"
    elif prob >= T_SUSPICIOUS:
        return "SUSPICIOUS"
    else:
        return "SAFE"

tiers = [tier(p) for p in y_probs]

# Options for the tiers:
# Option A (security): treat SUSPICIOUS and LIKELY as phishing
y_pred_security = (y_probs >= T_SUSPICIOUS).astype(int)

# Option B (strict): only LIKELY_PHISHING counts as phishing
y_pred_strict = (y_probs >= T_PHISHING).astype(int)

# Figuring out what threshold to use
#for t in [0.55, 0.6, 0.65]:
    #y_pred = (y_probs >= t).astype(int)
    #print(f"\nThreshold: {t}")
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))

#print("\nAccuracy:", accuracy_score(y_test, y_pred))
#print("\nClassification Report:\n")
#print(classification_report(y_test, y_pred))

#print("\nConfusion Matrix:\n")
#print(confusion_matrix(y_test, y_pred))


# Tradeoffs of option a vs b
print("\nSTRICT (only Likely=phishing)")
print(confusion_matrix(y_test, y_pred_strict))
print(classification_report(y_test, y_pred_strict))

print("\nSECURITY (Suspicious OR Likely=phishing)")
print(confusion_matrix(y_test, y_pred_security))
print(classification_report(y_test, y_pred_security))

print("\nTier counts:")
print(pd.Series(tiers).value_counts())