from flask import Flask, request, jsonify
import os
import base64
import gzip
from cryptography.fernet import Fernet

# Set your encryption key (must match the sender's key)
ENCRYPTION_KEY = b'nI_99yH47jBByOL0mFvKkBzN2r8H2BgFNU4FQ01vB2c='
fernet = Fernet(ENCRYPTION_KEY)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    if not data or 'filename' not in data or 'data' not in data:
        return jsonify({"error": "Invalid format"}), 400

    filename = data['filename'].replace("/", "_")
    path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        # Decode and decrypt
        encrypted = base64.b64decode(data['data'])
        decoded = fernet.decrypt(encrypted)
        with gzip.open(path.replace(".gz", ""), 'wb') as f_out:
            f_out.write(decoded)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok", "file": filename}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)