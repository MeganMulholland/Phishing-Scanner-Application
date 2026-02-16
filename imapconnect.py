import imaplib
import email

IMAP_SERVER = "imap.gmail.com"
EMAIL_ACCOUNT = "your_email@gmail.com"
APP_PASSWORD = "your_app_password"

mail = imaplib.IMAP4_SSL(IMAP_SERVER)
mail.login(EMAIL_ACCOUNT, APP_PASSWORD)

mail.select("inbox")

status, messages = mail.search(None, "UNSEEN")
email_ids = messages[0].split()

for e_id in email_ids:
    status, msg_data = mail.fetch(e_id, "(RFC822)")
    raw_email = msg_data[0][1]

    msg = email.message_from_bytes(raw_email)

    subject = msg["subject"]
    from_ = msg["from"]

    print("From:", from_)
    print("Subject:", subject)