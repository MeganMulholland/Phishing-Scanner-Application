import tkinter as tk
from tkinter import messagebox, font
from gui.dashboard_page import DashboardPage
from IMAP.imapconnect import connect_to_gmail
from PIL import Image, ImageTk

#Color palette
BG_COLOR = "#EEF6F1"        # soft sage
PRIMARY_BLUE = "#1E3A5F"    # dark blue
SAGE_GREEN = "#9CAF88"
INPUT_BG = "#D9EAF7"        # light blue
BUTTON_COLOR = "#1E3A5F"
# Solely handles the GUI
class LoginPage:

    def __init__(self, root):
        self.root = root
        self.root.configure(bg=BG_COLOR)  # light blue background

        # Core frame for login (centered)
        self.frame = tk.Frame(root,bg = BG_COLOR)
        self.frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Fonts
        self.header_font = font.Font(family="Bouncy", size=24, weight="bold")
        self.label_font = font.Font(family="Asul", size=12)
        try:
            logo_path = "assets/logo.png"
            image = Image.open(logo_path)
            image = image.resize((120, 120))  # adjust size if needed
            self.logo = ImageTk.PhotoImage(image)

            tk.Label(self.frame, image=self.logo, bg=BG_COLOR).pack(pady=(0, 10))

        except Exception as e:
            print("Logo failed to load:", e)
        # Welcome message
        tk.Label(
            self.frame,
            text="Welcome to PhishyScanner",
            font=self.header_font,
            fg=PRIMARY_BLUE,
            bg=BG_COLOR
        ).pack(pady=(0, 20))

        # Email label and entry
        tk.Label(self.frame, text="Email", font=self.label_font, bg=BG_COLOR).pack()
        self.email_entry = tk.Entry(
            self.frame,
            width=40,
            font=self.label_font,
            bd=3,
            relief="groove",
            bg=INPUT_BG
        )
        self.email_entry.pack(pady=(0, 10))

        # Password label and entry
        tk.Label(self.frame, text="App Password", font=self.label_font, bg=BG_COLOR).pack()
        self.password_entry = tk.Entry(
            self.frame,
            show="*",
            width=40,
            font=self.label_font,
            bd=3,
            relief="groove",
            bg=INPUT_BG,
        )
        self.password_entry.pack(pady=(0, 20))

        # Login button
        tk.Button(
            self.frame,
            text="Login",
            font=self.label_font,
            bg=BUTTON_COLOR,
            fg="white",# gold button
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