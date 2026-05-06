import os
import json
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
import tkinter as tk
from tkinter import ttk, font
import email
# import joblib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from preprocessing.preprocess import preprocess_email
from IMAP.imapconnect import decode_str, get_body
from tkinter import messagebox

import re

TRUSTED_DOMAINS = [
    # Spotify
    "spotify.com",

    # Discord
    "discord.com",
    "m.discord.com",

    # Google
    "google.com",
    "accounts.google.com",
    "mail.google.com",
    "youtube.com",

    # Microsoft
    "microsoft.com",
    "outlook.com",
    "office.com",
    "live.com",

    # Apple
    "apple.com",
    "icloud.com",

    # Amazon
    "amazon.com",

    # Social / networking
    "linkedin.com",
    "facebookmail.com",
    "instagram.com",
    "x.com",
    "twitter.com",

    # Development / school
    "github.com",
    "gitlab.com",

    # Streaming / entertainment
    "netflix.com",
    "hulu.com",
    "disneyplus.com",

    # Shopping
    "target.com",
    "walmart.com",
    "bestbuy.com",
    "ebay.com",

    # Payments
    "paypal.com",
    "venmo.com",
    "cash.app",

    # Food / delivery
    "doordash.com",
    "ubereats.com",
    "grubhub.com",

    # Travel
    "airbnb.com",
    "expedia.com",
    "delta.com",

    # Communication
    "zoom.us",
    "slack.com",

    # Education
    "instructure.com",   # Canvas LMS
    "duolingo.com"
]

def extract_sender_domain(sender):
    match = re.search(r'@([\w\.-]+)', sender.lower())
    return match.group(1) if match else ""

def is_trusted_sender(sender):
    domain = extract_sender_domain(sender)
    return (
        domain in TRUSTED_DOMAINS
        or any(domain.endswith("." + d) for d in TRUSTED_DOMAINS)
    )

def adjust_tier_for_trusted_sender(sender, tier, prob):
    if not is_trusted_sender(sender):
        return tier

    if tier == "LIKELY_PHISHING" and prob < 0.95:
        return "SUSPICIOUS"

    if tier == "SUSPICIOUS" and prob < 0.80:
        return "SAFE"

    return tier

# Color palette
BG_COLOR = "#EEF6F1"          # soft sage
CARD_COLOR = "#F8FBF9"        # near-white sage
PRIMARY_BLUE = "#1E3A5F"      # dark blue
SAGE_GREEN = "#9CAF88"
LIGHT_BLUE = "#D9EAF7"
BUTTON_BLUE = "#1E90FF"
BUTTON_DARK = "#1E3A5F"
SAFE_GREEN = "#D4EDDA"
WARNING_YELLOW = "#FFF3CD"
DANGER_RED = "#F8D7DA"
TEXT_GRAY = "#4F6F8F"

# Load DistilBERT model
model_dir = resource_path("models/distilbert_phishing")

if not os.path.isdir(model_dir):
    raise FileNotFoundError(
        f"Model folder not found: {model_dir}. "
        "Make sure models/distilbert_phishing is included."
    )

try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
except OSError as e:
    raise RuntimeError(f"Could not load DistilBERT model from {model_dir}: {e}")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Thresholds
T_SUSPICIOUS = 0.226
T_PHISHING = 0.831


def predict_email(text):
    try:
        # Handle empty or malformed email text
        if not isinstance(text, str) or not text.strip():
            return "SUSPICIOUS", 0.0

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

    except Exception as e:
        print("Prediction error:", e)
        return "SUSPICIOUS", 0.0

# Adding explanations
def has_url(text):
    return "http" in text or "www" in text

# more urgent words
urgent_words = [
    "urgent", "immediately", "asap", "action required", "act now",
    "verify", "verification", "confirm", "password", "login",
    "account", "suspend", "suspended", "locked", "security alert",
    "unusual activity", "unauthorized", "reset", "expires"
]
# added more reasons!
def get_explanation(text):
    reasons = []
    text_lower = text.lower()

    if has_url(text_lower):
        reasons.append("Contains link")

    if any(word in text_lower for word in urgent_words):
        reasons.append("Uses urgent language")

    if any(word in text_lower for word in ["verify", "confirm", "update", "validate"]):
        reasons.append("Requests account verification")

    if any(word in text_lower for word in ["password", "login", "sign in", "credentials"]):
        reasons.append("Mentions login or password")

    if any(word in text_lower for word in ["suspended", "locked", "disabled", "unusual activity", "unauthorized"]):
        reasons.append("Threatens account restriction")

    if any(word in text_lower for word in ["invoice", "payment", "billing", "refund", "charge"]):
        reasons.append("Mentions payment or billing")

    if any(word in text_lower for word in ["click here", "open attachment", "download", "view document"]):
        reasons.append("Encourages clicking or downloading")

    if any(word in text_lower for word in ["limited time", "act now", "expires", "deadline"]):
        reasons.append("Creates time pressure")

    if any(word in text_lower for word in ["congratulations", "winner", "prize", "reward", "gift card"]):
        reasons.append("Uses reward or prize language")

    return reasons


def move_to_spam(mail, e_id):
    try:
        copy_result = mail.uid("COPY", e_id, "[Gmail]/Spam")

        if copy_result[0] != "OK":
            print("Failed to copy to spam:", e_id, copy_result)
            return False

        delete_result = mail.uid("STORE", e_id, "+FLAGS", "(\\Deleted)")

        if delete_result[0] != "OK":
            print("Failed to mark deleted:", e_id, delete_result)
            return False

        print("Moved email to spam:", e_id)
        return True

    except Exception as e:
        print("Spam move error:", e_id, e)
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
        outer_blue = tk.Frame(root, bg=PRIMARY_BLUE, bd=5)
        outer_blue.pack(padx=5, pady=5, fill="both", expand=True)

        outer_sage = tk.Frame(outer_blue, bg=SAGE_GREEN, bd=5)
        outer_sage.pack(padx=5, pady=5, fill="both", expand=True)

        outer_card = tk.Frame(outer_sage, bg=CARD_COLOR, bd=5)
        outer_card.pack(padx=5, pady=5, fill="both", expand=True)

        # -----------------------------
        # Main inner frame
        # -----------------------------
        self.frame = tk.Frame(outer_card, bg=BG_COLOR)
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Fonts
        self.label_font = font.Font(family="Asul", size=12, weight="bold")

        # -----------------------------
        # Header
        # -----------------------------
        header_frame = tk.Frame(self.frame, bg=BG_COLOR)
        header_frame.pack(fill="x", pady=(10, 15))

        tk.Label(
            header_frame,
            text="PhishyScanner",
            font=("Asul", 26, "bold"),
            bg=BG_COLOR,
            fg=PRIMARY_BLUE
        ).pack()

        tk.Label(
            header_frame,
            text="Machine Learning-Based Email Threat Detection",
            font=("Asul", 11),
            bg=BG_COLOR,
            fg=TEXT_GRAY
        ).pack(pady=(2, 8))

        # -----------------------------
        # Button row
        # -----------------------------
        button_frame = tk.Frame(self.frame, bg=BG_COLOR)
        button_frame.pack(pady=(0, 8))

        tk.Button(
            button_frame,
            text="Scan Emails",
            font=self.label_font,
            fg="white",
            bg=SAGE_GREEN,
            activebackground="#7F936F",
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
        columns = ("Status", "Sender", "Subject", "Tier", "Score", "Reasoning")

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

       # self.tree.bind("<ButtonRelease-1>", self.handle_tree_click)

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
            background=CARD_COLOR,
            fieldbackground=CARD_COLOR,
            bordercolor = PRIMARY_BLUE,  # blue border-like effect
            borderwidth=1
        )
        style.map(
            "Treeview",
            background=[("selected", LIGHT_BLUE)],
            foreground=[("selected", "black")]
        )
        self.tree.tag_configure("SAFE", background=SAFE_GREEN)
        self.tree.tag_configure("SUSPICIOUS", background=WARNING_YELLOW)
        self.tree.tag_configure("LIKELY_PHISHING", background=DANGER_RED)
        self.tree.bind("<Double-1>", self.show_email_details)

        # -----------------------------
        # Spam action buttons
        # -----------------------------
        action_frame = tk.Frame(self.frame, bg=BG_COLOR)
        action_frame.pack(pady=10)

        tk.Label(
            action_frame,
            text="Select one or more emails, then choose an action.",
            bg=BG_COLOR,
            fg=TEXT_GRAY,
            font=("Asul", 10)
        ).pack(pady=(0, 5))

        tk.Button(
            action_frame,
            text="Select All Likely Phishing",
            font=self.label_font,
            bg=SAGE_GREEN,
            fg="white",
            activebackground="#7F936F",
            activeforeground="white",
            width=24,
            command=self.select_all_likely_phishing
        ).pack(side="left", padx=8)

        tk.Button(
            action_frame,
            text="Move Selected to Spam",
            font=self.label_font,
            bg=DANGER_RED,
            fg="black",
            activebackground="#e0aeb5",
            activeforeground="black",
            width=22,
            command=self.move_selected_to_spam
        ).pack(side="left", padx=8)

    def show_email_details(self, event):
        selected = self.tree.selection()
        if not selected:
            return

        values = self.tree.item(selected[0], "values")

        status, sender, subject, tier, score, reason = values

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

    def open_analytics(self):

        analytics_window = tk.Toplevel()

        AnalyticsPage(
            analytics_window,
            self.safe_count,
            self.suspicious_count,
            self.phishing_count,
            self.quarantined_count
        )
# Function for the button
    def select_all_likely_phishing(self):
        self.tree.selection_remove(self.tree.selection())

        selected_count = 0

        for row_id in self.tree.get_children():
            values = self.tree.item(row_id, "values")

            if len(values) >= 4:
                tier = values[3]

                if tier == "LIKELY_PHISHING":
                    self.tree.selection_add(row_id)
                    selected_count += 1

        if selected_count == 0:
            messagebox.showinfo(
                "No Likely Phishing Emails",
                "There are no likely phishing emails to select."
            )
        else:
            self.status_label.config(
                text=f"Selected {selected_count} likely phishing email(s)",
                fg=PRIMARY_BLUE
            )

    def move_selected_to_spam(self):
        selected_rows = list(self.tree.selection())  # important: freeze selection first

        if not selected_rows:
            messagebox.showinfo("No Selection", "Please select at least one email first.")
            return

        confirm = messagebox.askyesno(
            "Move Selected Emails to Spam?",
            f"Move {len(selected_rows)} selected email(s) to spam?"
        )

        if not confirm:
            return

        moved_rows = []
        failed_count = 0

        for row_id in selected_rows:
            e_id = self.email_id_map.get(row_id)

            if not e_id:
                failed_count += 1
                continue

            success = move_to_spam(self.mail, e_id)

            if success:
                moved_rows.append(row_id)
                self.quarantined_count += 1
            else:
                failed_count += 1

        # Actually remove deleted emails from the inbox after all moves
        try:
            self.mail.expunge()
        except Exception as e:
            print("Expunge error:", e)

        # Delete rows from the GUI after IMAP operations finish
        for row_id in moved_rows:
            self.tree.delete(row_id)
            self.email_id_map.pop(row_id, None)

        self.status_label.config(
            text=f"Moved {len(moved_rows)} email(s) to spam. Failed: {failed_count}",
            fg="green" if failed_count == 0 else "red"
        )

        if failed_count > 0:
            messagebox.showwarning(
                "Some Emails Were Not Moved",
                f"{failed_count} selected email(s) could not be moved to spam."
            )
    # -----------------------------
    # Fetch emails function
    # -----------------------------
    def fetch_emails(self):

        self.status_label.config(text="Scanning emails... please wait", fg="blue")
        self.frame.update_idletasks()
        self.mail.select("INBOX")
        try:
            status, _ = self.mail.select("INBOX")
            if status != "OK":
                messagebox.showerror("Mailbox Error", "Could not open INBOX.")
                return

            status, messages = self.mail.uid("search", None, "UNSEEN")

            if status != "OK" or not messages or not messages[0]:
                self.status_label.config(text="No unread emails found", fg="green")
                return

            email_ids = messages[0].split()

        except Exception as e:
            messagebox.showerror("IMAP Error", f"Could not search inbox:\n{e}")
            return

        for e_id in email_ids:
            self.frame.update_idletasks()

            status, msg_data = self.mail.uid("fetch", e_id, "(RFC822)")
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
            except Exception as e:
                print("Failed to parse email: ", e)
                continue

            subject = decode_str(msg.get("Subject"))
            sender = decode_str(msg.get("From"))
            body = get_body(msg)

            # Combine subject + body
            email_text = subject + " " + body

            # Handle prediction failure per email. Won't crash
            try:
                tier, prob = predict_email(email_text)
                tier = adjust_tier_for_trusted_sender(sender, tier, prob)
            except Exception as e:
                print("Prediction failed for email:", e)
                tier, prob = "SUSPICIOUS", 0.0

            # Adding in reasons
            reasons = get_explanation(email_text)

            if is_trusted_sender(sender):
                reasons.append("Trusted sender detected")

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
                values=("Review", sender, subject, tier, f"{prob:.4f}", reason_str),
                tags=(tier,)
            )
            self.email_id_map[item_id] = e_id
        self.status_label.config(text="Scan complete", fg="green")
        self.frame.update_idletasks()

        if self.suspicious_count > 0 or self.phishing_count > 0:
            messagebox.showwarning(
                "Potential Phishing Emails Found",
                f"PhishyScanner found {self.suspicious_count} suspicious email(s) "
                f"and {self.phishing_count} likely phishing email(s).\n\n"
                "Please review these emails carefully. If you do not recognize the sender "
                "or the message seems unsafe, use the 'Move to Spam' option."
            )


#more polished analytics page
class AnalyticsPage:

    def __init__(self, root, safe, suspicious, phishing, quarantined):
        root.title("PhishyScanner Analytics")
        root.geometry("650x550")
        root.configure(bg=BG_COLOR)

        label_font = font.Font(family="Asul", size=13)
        header_font = font.Font(family="Asul", size=20, weight="bold")
        card_font = font.Font(family="Asul", size=12, weight="bold")

        total = safe + suspicious + phishing

        if total == 0:
            chart_values = [1, 1, 1]
        else:
            chart_values = [safe, suspicious, phishing]

        # Header
        tk.Label(
            root,
            text="Scan Analytics",
            font=header_font,
            bg=BG_COLOR,
            fg=PRIMARY_BLUE
        ).pack(pady=(20, 5))

        tk.Label(
            root,
            text="Summary of scanned email classifications",
            font=label_font,
            bg=BG_COLOR,
            fg=TEXT_GRAY
        ).pack(pady=(0, 15))

        # Summary card frame
        card_frame = tk.Frame(root, bg=BG_COLOR)
        card_frame.pack(pady=10)

        def create_card(parent, title, value, color):
            card = tk.Frame(
                parent,
                bg=CARD_COLOR,
                bd=2,
                relief="groove",
                padx=18,
                pady=12
            )
            card.pack(side="left", padx=8)

            tk.Label(
                card,
                text=title,
                font=("Asul", 10, "bold"),
                bg=CARD_COLOR,
                fg=TEXT_GRAY
            ).pack()

            tk.Label(
                card,
                text=str(value),
                font=("Asul", 18, "bold"),
                bg=CARD_COLOR,
                fg=color
            ).pack()

        create_card(card_frame, "Safe", safe, SAGE_GREEN)
        create_card(card_frame, "Suspicious", suspicious, "#B8860B")
        create_card(card_frame, "Phishing", phishing, "#B22222")
        create_card(card_frame, "Moved", quarantined, PRIMARY_BLUE)

        # Chart
        chart_frame = tk.Frame(root, bg=BG_COLOR)
        chart_frame.pack(pady=15)

        labels = ["Safe", "Suspicious", "Phishing"]

        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        fig.patch.set_facecolor("#EEF6F1")
        ax.set_facecolor("#EEF6F1")

        ax.pie(
            chart_values,
            labels=labels,
            autopct="%1.0f%%",
            startangle=90,
            colors=["#D4EDDA", "#FFF3CD", "#F8D7DA"],
            textprops={"fontsize": 9}
        )

        ax.set_title(
            "Email Classification Distribution",
            fontsize=12,
            color="#1E3A5F"
        )

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Footer message
        if phishing > 0:
            footer_text = "Review likely phishing emails and move unsafe messages to spam."
            footer_color = "#B22222"
        elif suspicious > 0:
            footer_text = "Some emails need review before interacting with links or attachments."
            footer_color = "#B8860B"
        else:
            footer_text = "No major phishing threats detected."
            footer_color = SAGE_GREEN

        tk.Label(
            root,
            text=footer_text,
            font=("Asul", 12, "bold"),
            fg=footer_color,
            bg=BG_COLOR,
            wraplength=520,
            justify="center"
        ).pack(pady=(10, 20))
