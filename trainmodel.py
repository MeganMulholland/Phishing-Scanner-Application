import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    X, y, test_size=0.2, random_state=42
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


model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


# Evaluate


y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))