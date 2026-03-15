import tkinter as tk
from tkinter import ttk, font
import email
import joblib
from preprocess.preprocess import preprocess_email
from IMAP.imapconnect import decode_str, get_body

# Load model
artifact = joblib.load("models/phishing_artifact.joblib")
vectorizer = artifact["vectorizer"]
cal_model = artifact["cal_model"]
T_SUSPICIOUS = artifact["thresholds"]["T_SUSPICIOUS"]
T_PHISHING = artifact["thresholds"]["T_PHISHING"]

def predict_email(text):
    clean = preprocess_email(text)
    X = vectorizer.transform([clean])
    prob = cal_model.predict_proba(X)[0][1]
    if prob >= T_PHISHING:
        tier = "LIKELY_PHISHING"
    elif prob >= T_SUSPICIOUS:
        tier = "SUSPICIOUS"
    else:
        tier = "SAFE"
    return tier, prob

class DashboardPage:

    def __init__(self, root, mail):
        self.mail = mail

        # -----------------------------
        # Outer frame: light blue, white, gold layered border
        # -----------------------------
        outer_blue = tk.Frame(root, bg="#add8e6", bd=5)
        outer_blue.pack(padx=5, pady=5, fill="both", expand=True)

        outer_white = tk.Frame(outer_blue, bg="white", bd=5)
        outer_white.pack(padx=5, pady=5, fill="both", expand=True)

        outer_gold = tk.Frame(outer_white, bg="#fffacd", bd=5)
        outer_gold.pack(padx=5, pady=5, fill="both", expand=True)

        # -----------------------------
        # Main inner frame
        # -----------------------------
        self.frame = tk.Frame(outer_gold, bg="#f0f8ff")  # light blue background inside
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Fonts
        self.label_font = font.Font(family="Asul", size=12, weight="bold")

        # -----------------------------
        # Scan Emails Button (red text)
        # -----------------------------
        tk.Button(
            self.frame, text="Scan Emails", font=self.label_font,
            fg="red", bg="#add8e6", activebackground="#87ceeb",
            command=self.fetch_emails
        ).pack(pady=10)

        # -----------------------------
        # Treeview with blue row effect
        # -----------------------------
        columns = ("Sender", "Subject", "Tier", "Score")
        self.tree = ttk.Treeview(self.frame, columns=columns, show="headings", height=20)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=200)
        self.tree.pack(fill="both", expand=True)

        # -----------------------------
        # Style the Treeview
        # -----------------------------
        style = ttk.Style()
        style.theme_use("default")

        style.configure(
            "Treeview",
            font=("Asul", 10),
            rowheight=25,
            background="#f0f8ff",
            fieldbackground="#f0f8ff",
            bordercolor="#1E90FF",  # blue border-like effect
            borderwidth=1
        )
        style.map(
            "Treeview",
            background=[("selected", "#add8e6")],
            foreground=[("selected", "black")]
        )

    # -----------------------------
    # Fetch emails function
    # -----------------------------
    def fetch_emails(self):
        self.mail.select("INBOX")
        status, messages = self.mail.search(None, "UNSEEN")
        email_ids = messages[0].split()
        for e_id in email_ids:
            status, msg_data = self.mail.fetch(e_id, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            subject = decode_str(msg.get("Subject"))
            sender = decode_str(msg.get("From"))
            body = get_body(msg)
            tier, prob = predict_email(body)
            self.tree.insert("", "end", values=(sender, subject, tier, f"{prob:.2f}"))