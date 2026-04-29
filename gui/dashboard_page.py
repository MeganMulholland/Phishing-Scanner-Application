import tkinter as tk
from tkinter import ttk, font
import email
#import joblib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

from preprocessing.preprocess import preprocess_email

from IMAP.imapconnect import decode_str, get_body
from tkinter import messagebox
# Load model
#artifact = joblib.load("models/phishing_artifact.joblib")
#vectorizer = artifact["vectorizer"]
#cal_model = artifact["cal_model"]
#T_SUSPICIOUS = artifact["thresholds"]["T_SUSPICIOUS"]
#T_PHISHING = artifact["thresholds"]["T_PHISHING"]

# Load DistilBERT model
MODEL_PATH = "models/distilbert_phishing"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Thresholds
T_SUSPICIOUS = 0.226
T_PHISHING = 0.831

def predict_email(text):
    clean = preprocess_email(text)

    inputs = tokenizer(
        clean,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    phishing_prob = probs[0][1].item()

    if phishing_prob >= T_PHISHING:
        tier = "LIKELY_PHISHING"
    elif phishing_prob >= T_SUSPICIOUS:
        tier = "SUSPICIOUS"
    else:
        tier = "SAFE"

    return tier, phishing_prob
# Adding explanations
def has_url(text):
    return "http" in text or "www" in text

urgent_words = ["urgent", "immediately", "verify", "suspend", "password", "login", "account"]

def get_explanation(text):
    reasons = []

    text_lower = text.lower()

    if has_url(text_lower):
        reasons.append("Contains link")

    if any(word in text_lower for word in urgent_words):
        reasons.append("Uses urgent language")

    return reasons
def move_to_spam(mail, e_id):
    try:
        result = mail.copy(e_id, "[Gmail]/Spam")

        if result[0] == "OK":
            mail.store(e_id, "+FLAGS", "\\Deleted")
            print("Moved email to spam")
            return True
        else:
            print("Failed to copy to spam:", result)
            return False

    except Exception as e:
        print("Spam move error:", e)
        return False

class DashboardPage:

    def __init__(self, root, mail):
        self.mail = mail

        self.safe_count = 0
        self.suspicious_count = 0
        self.phishing_count = 0
        self.quarantined_count = 0
        self.email_id_map = {}

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
        # Header
        # -----------------------------
        header_frame = tk.Frame(self.frame, bg="#f0f8ff")
        header_frame.pack(fill="x", pady=(10, 15))

        tk.Label(
            header_frame,
            text="Phishy Scanner",
            font=("Asul", 26, "bold"),
            bg="#f0f8ff",
            fg="#1E3A5F"
        ).pack()

        tk.Label(
            header_frame,
            text="AI-powered email phishing detection",
            font=("Asul", 11),
            bg="#f0f8ff",
            fg="#4F6F8F"
        ).pack(pady=(2, 8))

        # -----------------------------
        # Button row
        # -----------------------------
        button_frame = tk.Frame(self.frame, bg="#f0f8ff")
        button_frame.pack(pady=(0, 8))

        tk.Button(
            button_frame,
            text="Scan Emails",
            font=self.label_font,
            fg="white",
            bg="#1E90FF",
            activebackground="#187bcd",
            activeforeground="white",
            width=16,
            command=self.fetch_emails
        ).pack(side="left", padx=8)

        tk.Button(
            button_frame,
            text="View Analytics",
            font=self.label_font,
            fg="white",
            bg="#4F6F8F",
            activebackground="#3f596f",
            activeforeground="white",
            width=16,
            command=self.open_analytics
        ).pack(side="left", padx=8)

        # -----------------------------
        # Status label
        # -----------------------------
        self.status_label = tk.Label(
            self.frame,
            text="Ready to scan emails",
            font=("Asul", 11, "bold"),
            bg="#f0f8ff",
            fg="#4F6F8F"
        )
        self.status_label.pack(pady=(0, 10))
        # -----------------------------
        # Treeview + Scrollbars
        # -----------------------------
        columns = ("Action", "Sender", "Subject", "Tier", "Score", "Reasoning")

        # Frame to hold tree + scrollbars
        tree_frame = tk.Frame(self.frame)
        tree_frame.pack(fill="both", expand=True)

        # Scrollbars
        scroll_y = tk.Scrollbar(tree_frame, orient="vertical")
        scroll_x = tk.Scrollbar(tree_frame, orient="horizontal")

        # Treeview
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=20,
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )

        # Configure scrollbars
        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)

        # Pack scrollbars and tree
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        self.tree.pack(side="left", fill="both", expand=True)

        self.tree.bind("<ButtonRelease-1>", self.handle_tree_click)

        # Column setup
        for col in columns:
            self.tree.heading(col, text=col)

            if col == "Score":
                self.tree.column(col, width=80, anchor="center")
            elif col == "Tier":
                self.tree.column(col, width=150, anchor="center")
            elif col == "Sender":
                self.tree.column(col, width=200)
            elif col == "Subject":
                self.tree.column(col, width=300)
            elif col == "Reasoning":
                self.tree.column(col, width=250)
            elif col == "Action":
                self.tree.column(col, width=120, anchor="center")
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
        self.tree.tag_configure("SAFE", background="#d4edda")
        self.tree.tag_configure("SUSPICIOUS", background="#fff3cd")
        self.tree.tag_configure("LIKELY_PHISHING", background="#f8d7da")
        self.tree.bind("<Double-1>", self.show_email_details)

    def show_email_details(self, event):
        selected = self.tree.selection()
        if not selected:
            return

        values = self.tree.item(selected[0], "values")

        action, sender, subject, tier, score, reason = values

        window = tk.Toplevel()
        window.title("Email Details")

        tk.Label(window, text=f"Sender: {sender}").pack()
        tk.Label(window, text=f"Subject: {subject}").pack()
        tk.Label(window, text=f"Tier: {tier}").pack()
        tk.Label(window, text=f"Score: {score}").pack()
        tk.Label(window, text=f"Reason: {reason}").pack()

    def handle_tree_click(self, event):
        row_id = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)

        if not row_id:
            return

        # Action column is the first column
        if column == "#1":
            values = self.tree.item(row_id, "values")
            subject = values[2]
            tier = values[3]
            score = values[4]
            reason = values[5]

            confirm = messagebox.askyesno(
                "Move to Spam?",
                f"""Move this email to spam?
            
    Subject: {subject}
    Classification: {tier}
    Score: {score}
    Reason: {reason}
    """
            )

            if confirm:
                e_id = self.email_id_map.get(row_id)

                if e_id:
                    success = move_to_spam(self.mail, e_id)

                    if success:
                        self.quarantined_count += 1
                        self.tree.delete(row_id)
                    else:
                        messagebox.showerror("Error", "Could not move this email to spam.")

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

        self.status_label.config(text="Scanning emails... please wait", fg="blue")
        self.frame.update_idletasks()
        self.mail.select("INBOX")
        status, messages = self.mail.search(None, "UNSEEN")
        email_ids = messages[0].split()
        for e_id in email_ids:
            self.frame.update_idletasks()

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

            # Adding in reasons
            reasons = get_explanation(email_text)
            reason_str = ", ".join(reasons) if reasons else "No obvious red flags"

            if tier == "SAFE":
                self.safe_count += 1

            elif tier == "SUSPICIOUS":
                self.suspicious_count += 1

            elif tier == "LIKELY_PHISHING":
                self.phishing_count += 1

            self.frame.update_idletasks()
            item_id = self.tree.insert(
                "", "end",
                values=("Move to spam", sender, subject, tier, f"{prob:.4f}", reason_str),
                tags=(tier,)
            )
            self.email_id_map[item_id] = e_id
        self.status_label.config(text="Scan complete", fg="green")
        self.frame.update_idletasks()


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