import tkinter as tk
from tkinter import ttk, font
import email
import joblib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

def move_to_spam(mail, e_id):
    try:
        result = mail.copy(e_id, "[Gmail]/Spam")

        if result[0] == "OK":
            mail.store(e_id, "+FLAGS", "\\Deleted")
            mail.expunge()
            print("Moved email to spam")
        else:
            print("Failed to copy to spam")

    except Exception as e:
        print("Spam move error:", e)

class DashboardPage:

    def __init__(self, root, mail):
        self.mail = mail

        self.safe_count = 0
        self.suspicious_count = 0
        self.phishing_count = 0
        self.quarantined_count = 0

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

        tk.Button(
            self.frame,
            text="View Analytics",
            font=self.label_font,
            bg="#add8e6",
            command=self.open_analytics
        ).pack(pady=5)

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

    class AnalyticsPage:

        def __init__(self, root, safe, suspicious, phishing, quarantined):
            root.title("Scan Analytics")
            root.geometry("500x400")
            root.configure(bg="#f0f8ff")

            label_font = font.Font(family="Asul", size=14, weight="bold")

            tk.Label(
                root,
                text="Email Scan Results",
                font=("Asul", 18, "bold"),
                bg="#f0f8ff"
            ).pack(pady=20)

            tk.Label(
                root,
                text=f"Safe Emails: {safe}",
                font=label_font,
                bg="#f0f8ff"
            ).pack(pady=5)

            tk.Label(
                root,
                text=f"Suspicious Emails: {suspicious}",
                font=label_font,
                bg="#f0f8ff"
            ).pack(pady=5)

            tk.Label(
                root,
                text=f"Likely Phishing Emails: {phishing}",
                font=label_font,
                bg="#f0f8ff"
            ).pack(pady=5)

            tk.Label(
                root,
                text=f"Emails Quarantined (Spam): {quarantined}",
                font=label_font,
                bg="#f0f8ff"
            ).pack(pady=20)

            # Final message
            tk.Label(
                root,
                text="You have been cleansed!",
                font=("Asul", 16, "bold"),
                fg="#1E90FF",
                bg="#f0f8ff"
            ).pack(side="bottom", pady=30)

    def open_analytics(self):

        analytics_window = tk.Toplevel()

        AnalyticsPage(
            analytics_window,
            self.safe_count,
            self.suspicious_count,
            self.phishing_count,
            self.quarantined_count
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

            # Check fetch success
            if status != "OK":
                print("Fetch failed")
                continue

            # Check message exists
            if not msg_data or msg_data[0] is None:
                print("Empty message returned by IMAP")
                continue

            try:
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)
            except:
                print("Failed to parse email")
                continue

            subject = decode_str(msg.get("Subject"))
            sender = decode_str(msg.get("From"))
            body = get_body(msg)

            # Combine subject + body
            email_text = subject + " " + body

            tier, prob = predict_email(email_text)

            if tier == "SAFE":
                self.safe_count += 1

            elif tier == "SUSPICIOUS":
                self.suspicious_count += 1

            elif tier == "LIKELY_PHISHING":
                self.phishing_count += 1
                self.quarantined_count += 1
                move_to_spam(self.mail, e_id)

            self.tree.insert("", "end", values=(sender, subject, tier, f"{prob:.2f}"))

class AnalyticsPage:

    def __init__(self, root, safe, suspicious, phishing, quarantined):

        root.title("Scan Analytics")
        root.geometry("600x500")
        root.configure(bg="#f0f8ff")

        label_font = font.Font(family="Asul", size=14, weight="bold")

        if safe + suspicious + phishing == 0:
            safe, suspicious, phishing = 1, 1, 1

        tk.Label(
            root,
            text="Email Scan Results",
            font=("Asul", 18, "bold"),
            bg="#f0f8ff"
        ).pack(pady=10)

        # -----------------------------
        # Email counts
        # -----------------------------
        tk.Label(root, text=f"Safe Emails: {safe}", font=label_font, bg="#f0f8ff").pack()
        tk.Label(root, text=f"Suspicious Emails: {suspicious}", font=label_font, bg="#f0f8ff").pack()
        tk.Label(root, text=f"Likely Phishing Emails: {phishing}", font=label_font, bg="#f0f8ff").pack()
        tk.Label(root, text=f"Emails Quarantined (Spam): {quarantined}", font=label_font, bg="#f0f8ff").pack(pady=10)

        # -----------------------------
        # Create pie chart
        # -----------------------------
        labels = ["Safe", "Suspicious", "Phishing"]
        values = [safe, suspicious, phishing]

        fig, ax = plt.subplots(figsize=(4,4))

        ax.pie(
            values,
            labels=labels,
            autopct="%1.0f%%",
            startangle=90
        )

        ax.set_title("Email Classification Distribution")

        # Embed chart into Tkinter
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

        # -----------------------------
        # Cleansed message
        # -----------------------------
        tk.Label(
            root,
            text="You have been cleansed!",
            font=("Asul", 16, "bold"),
            fg="#1E90FF",
            bg="#f0f8ff"
        ).pack(side="bottom", pady=20)