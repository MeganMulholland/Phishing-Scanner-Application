import tkinter as tk
from tkinter import messagebox, font
from gui.dashboard_page import DashboardPage
from IMAP.imapconnect import connect_to_gmail

# Solely handles the GUI
class LoginPage:

    def __init__(self, root):
        self.root = root
        self.root.configure(bg="#f0f8ff")  # light blue background

        # Core frame for login (centered)
        self.frame = tk.Frame(root, bg="#f0f8ff")
        self.frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Fonts
        self.header_font = font.Font(family="Bouncy", size=24, weight="bold")
        self.label_font = font.Font(family="Asul", size=12)

        # Welcome message
        tk.Label(
            self.frame,
            text="Welcome to PhishyScanner",
            font=self.header_font,
            bg="#f0f8ff"
        ).pack(pady=(0, 20))

        # Email label and entry
        tk.Label(self.frame, text="Email", font=self.label_font, bg="#f0f8ff").pack()
        self.email_entry = tk.Entry(
            self.frame,
            width=40,
            font=self.label_font,
            bd=3,
            relief="groove",
            bg="#add8e6"  # light blue background
        )
        self.email_entry.pack(pady=(0, 10))

        # Password label and entry
        tk.Label(self.frame, text="App Password", font=self.label_font, bg="#f0f8ff").pack()
        self.password_entry = tk.Entry(
            self.frame,
            show="*",
            width=40,
            font=self.label_font,
            bd=3,
            relief="groove",
            bg="#add8e6"
        )
        self.password_entry.pack(pady=(0, 20))

        # Login button
        tk.Button(
            self.frame,
            text="Login",
            font=self.label_font,
            bg="#f9e79f",  # gold button
            command=self.login
        ).pack()

    def login(self):
        email_user = self.email_entry.get()
        password = self.password_entry.get()

        email_user = email_user.strip()

        if not email_user or not password:
            messagebox.showerror("Missing Input", "Please enter both your email and app password.")
            return

        try:
            mail = connect_to_gmail(email_user, password)

            messagebox.showinfo("Success", "Connected to Gmail")

            # Destroy all widgets in the root to remove login page completely
            for widget in self.root.winfo_children():
                widget.destroy()

            # Launch Dashboard
            DashboardPage(self.root, mail)

        #except:
           # messagebox.showerror("Error", "Login failed")

        except Exception as e:
            messagebox.showerror("Error", f"Login failed:\n{e}")
            print("Login failed:", e)