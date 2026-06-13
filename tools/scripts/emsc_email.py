import smtplib, ssl
import os
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from obspy.core.event import read_events

# --- Configuration Constants ---
# Email settings
SMTP_SERVER = 'vmail2.noa.gr:587'
SMTP_USER = 'monitor_gi'
SMTP_PASS = '%noa07'
SENDER_EMAIL = 'monitor_gi@noa.gr'
EMAIL_SUBJECT  = 'NOAIG'

# Split recipients into BCC and regular recipients
# TO_RECIPIENTS = ['alert@emsc-csem.org', 'seismo@isc.ac.uk']
# BCC_RECIPIENTS = ['fanis.vlahogiannis@gmail.com', 'cpevangelidis@gmail.com']

TO_RECIPIENTS = ['f.vlachogiannis.promracing@gmail.com']
BCC_RECIPIENTS = ['fanis.vlahogiannis@gmail.com']

def emsc_email(output_dir):
    msg = MIMEMultipart("alternative")
    msg["From"] = SENDER_EMAIL
    msg["Subject"] = EMAIL_SUBJECT
    msg["To"] = ", ".join(TO_RECIPIENTS)
    msg["BCC"] = ", ".join(BCC_RECIPIENTS)

    if os.path.exists(os.path.join(output_dir, 'emsc.revise.txt')):
        print(f'No Automatic EMSC email will be sent on revise')
        return

    emsc_file_path = os.path.join(output_dir, 'emsc.txt')
    try:
        with open(emsc_file_path, 'r') as f:
            emsc_content = f.read()
    except FileNotFoundError:
        print(f"Error: {emsc_file_path} not found.")
        return

    msg.attach(MIMEText(emsc_content, "plain"))

    try:
        smtp_host, smtp_port = SMTP_SERVER.split(':')
        with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            
            # Send to all recipients (TO + BCC)
            all_recipients = TO_RECIPIENTS + BCC_RECIPIENTS
            server.sendmail(SENDER_EMAIL, all_recipients, msg.as_string())
            print(f"Email sent to: {', '.join(TO_RECIPIENTS)} (visible)")
            print(f"Email sent to: {', '.join(BCC_RECIPIENTS)} (BCC)")
            
            server.quit()
        print("Email sending process finished.")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == '__main__':
    print(f'=== Running EMSC_EMAIL SCRIPT ==')
    if len(sys.argv) < 2:
        print("Usage: python emsc_email.py <output_directory>")
        sys.exit(1)
