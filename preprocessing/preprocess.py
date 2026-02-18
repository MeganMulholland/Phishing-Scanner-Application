import re
from bs4 import BeautifulSoup


def preprocess_email(text: str) -> str:

    if not isinstance(text, str):
        return ""

    # makes all text lowercase
    text = text.lower()

    # removes HTML tags but keep visible text
    text = BeautifulSoup(text, "html.parser").get_text()

    # replaces URLs with token
    text = re.sub(r"http[s]?://\S+|www\.\S+", " URL_TOKEN ", text)

    # replaces email addresses with token
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " EMAIL_TOKEN ", text)

    # replaces numbers with token
    text = re.sub(r"\b\d+\b", " NUM_TOKEN ", text)

    # removes special characters (keep basic punctuation)
    text = re.sub(r"[^a-z0-9\s\.\,\!\?]", " ", text)

    # normalizes whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text