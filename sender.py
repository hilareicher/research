import os
import gzip
import base64
import time
import random
import requests
import shutil
from cryptography.fernet import Fernet

# === CONFIG ===
UPLOAD_URL = "https://bedbf4f0595f.ngrok-free.app/upload"
SOURCE_DIR = "/home/hilareicher.mail.tau.ac.il/research/violence_prediction/results"
ENCRYPTION_KEY = b'nI_99yH47jBByOL0mFvKkBzN2r8H2BgFNU4FQ01vB2c='
fernet = Fernet(ENCRYPTION_KEY)
TMP_ARCHIVE = "archive_dir.tar.gz"

# === FUNCTIONS ===

def create_gzipped_archive(source_dir, output_path):
    shutil.make_archive("archive_dir", 'gztar', root_dir=source_dir)
    return output_path

def encode_file_encrypted(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    encrypted = fernet.encrypt(raw)
    return base64.b64encode(encrypted).decode('utf-8')

def upload_file(filename, encoded_content):
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "python-requests/2.31.0",
        "X-App-Purpose": "metrics-export"
    }
    payload = {
        "filename": filename,
        "data": encoded_content
    }
    response = requests.post(UPLOAD_URL, json=payload, headers=headers)
    print(f"[{response.status_code}] Uploaded {filename}")

# === MAIN EXECUTION ===

def main():
    print("Creating archive...")
    archive_path = create_gzipped_archive(SOURCE_DIR, TMP_ARCHIVE)
    print("Encrypting...")
    encoded = encode_file_encrypted(archive_path)
    print("Uploading...")
    upload_file(os.path.basename(archive_path), encoded)
    os.remove(archive_path)
    print("Done.")

if __name__ == "__main__":
    main()