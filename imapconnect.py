import imaplib
import email
import os
from email.header import decode_header
from dotenv import load_dotenv
from pathlib import Path

# Load credentials from .env

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

IMAP_SERVER = "imap.gmail.com"

EMAIL_ACCOUNT = os.getenv("EMAIL_ACCOUNT")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# Debug check
print("EMAIL:", EMAIL_ACCOUNT)
print("PASSWORD:", APP_PASSWORD)

# decode email headers

def decode_str(s):
    if not s:
        return ""
    decoded, charset = decode_header(s)[0]
    if isinstance(decoded, bytes):
        return decoded.decode(charset or "utf-8", errors="ignore")
    return decoded


# extract body text

def get_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition"))

            if content_type == "text/plain" and "attachment" not in disposition:
                try:
                    return part.get_payload(decode=True).decode(errors="ignore")
                except:
                    return ""
    else:
        try:
            return msg.get_payload(decode=True).decode(errors="ignore")
        except:
            return ""
    return ""


# Connect + login

try:
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, APP_PASSWORD)
    print("Logged into IMAP server")
except imaplib.IMAP4.error as e:
    print("Login failed:", e)
    exit()


# Select inbox

mail.select("INBOX")

# Search unread emails
status, messages = mail.search(None, "UNSEEN")

if status != "OK":
    print("Failed to retrieve messages")
    exit()

email_ids = messages[0].split()

print(f"Found {len(email_ids)} unread emails\n")


# Process emails

for e_id in email_ids:
    status, msg_data = mail.fetch(e_id, "(BODY.PEEK[])")

    if status != "OK":
        print("Failed to fetch email")
        continue

    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)

    subject = decode_str(msg.get("Subject"))
    from_ = decode_str(msg.get("From"))
    body = get_body(msg)

    print("From:", from_)
    print("Subject:", subject)
    print("Body preview:", body[:200])
    print("-" * 50)


# Logout cleanly

mail.logout()
print("Logged out")