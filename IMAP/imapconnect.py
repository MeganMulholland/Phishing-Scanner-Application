import imaplib
import email
import os
from email.header import decode_header
from dotenv import load_dotenv
from pathlib import Path

# Hanldes the Gmail Connection
# Load credentials from .env
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

IMAP_SERVER = "imap.gmail.com"

EMAIL_ACCOUNT = os.getenv("EMAIL_ACCOUNT")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# Debug check (optional)
#print("EMAIL:", EMAIL_ACCOUNT)
#print("PASSWORD:", APP_PASSWORD)

# -----------------------------
# Helper functions (USED BY GUI)
# -----------------------------

def decode_str(s):
    # Decode email headers. Returns empty string if the header is missing or malformed
    if not s:
        return ""
    try:
        decoded, charset = decode_header(s)[0]
        if isinstance(decoded, bytes):
            return decoded.decode(charset or "utf-8", errors="ignore")
        return str(decoded)
    except Exception as e:
        print ("Header decode error: ", e)
        return ""




def get_body(msg):
    # updated with more bulletproof exceptions
    #Extract the plain-text body from an email message.
    #Returns an empty string if the body cannot be decoded.

    if msg is None:
        return ""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition"))

                if content_type == "text/plain" and "attachment" not in disposition:
                    payload = part.get_payload(decode=True)

                    if payload:
                        return payload.decode(errors="ignore")

        else:
            payload = msg.get_payload(decode=True)

            if payload:
                return payload.decode(errors="ignore")

    except Exception as e:
        print("Email body decode error:", e)
        return ""

    return ""


def connect_to_gmail(email_account=None, app_password=None):
    #Need to update below using this function.
    #Connect to Gmail using IMAP.
    #Used for testing or future reusable login logic.

    email_account = email_account or EMAIL_ACCOUNT
    app_password = app_password or APP_PASSWORD

    if not email_account or not app_password:
        raise ValueError("Missing email account or app password.")

    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(email_account, app_password)
        return mail

    except imaplib.IMAP4.error as e:
        raise ConnectionError(f"IMAP login failed: {e}")

    except Exception as e:
        raise ConnectionError(f"Could not connect to Gmail: {e}")

# -----------------------------
# Only run test code directly
# -----------------------------

if __name__ == "__main__":
    mail = None
    try:
        mail = connect_to_gmail()
        print("Logged into IMAP server")

        status, _ = mail.select("INBOX")

        if status != "OK":
            raise RuntimeError("Could not open INBOX.")

        status, messages = mail.search(None, "UNSEEN")

        if status != "OK":
            raise RuntimeError("Failed to retrieve unread messages.")

        email_ids = messages[0].split() if messages and messages[0] else []

        print(f"Found {len(email_ids)} unread emails\n")

        for e_id in email_ids:
            status, msg_data = mail.fetch(e_id, "(BODY.PEEK[])")

            if status != "OK" or not msg_data or not msg_data[0]:
                print("Failed to fetch email")
                continue

            try:
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)
            except Exception as e:
                print("Failed to parse email:", e)
                continue

            subject = decode_str(msg.get("Subject"))
            from_ = decode_str(msg.get("From"))
            body = get_body(msg)

            print("From:", from_)
            print("Subject:", subject)
            print("Body preview:", body[:200])
            print("-" * 50)

    except Exception as e:
        print("IMAP test failed:", e)

    finally:
        if mail:
            try:
                mail.logout()
                print("Logged out")
            except Exception:
                pass