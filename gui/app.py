import tkinter as tk
from gui.login_page import LoginPage

def run_app():

    root = tk.Tk()
    root.title("Email Phishing Scanner")
    root.geometry("900x600")

    LoginPage(root)

    root.mainloop()

if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        print(f"Application failed to start: {e}")