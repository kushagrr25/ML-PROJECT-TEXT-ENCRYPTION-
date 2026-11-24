# PII Detection & Token Encryption (SVM)

A complete Machine Learning and Cryptography application that detects **Personally Identifiable Information (PII)** words/tokens in arbitrary text strings using a trained **Support Vector Machine (LinearSVC)** and encrypts them using **AES-256 (Fernet)** authenticated encryption.

---

## 📁 Simple Project Structure

```
PII_Detection_Project/
├── pii_dataset.csv      # The PII dataset
├── features.py          # [Code 1] Feature engineering & tokenization
├── train.py             # [Code 2] Model training (SVM with all PII labels marked as IMPORTANT)
├── main.py              # [Code 3] Main script (detection, selection & AES encryption/decryption)
├── ui.py                # [Code 4] Web server to launch & serve the UI
├── index.html           # Simple HTML/CSS/JS Website matching the SaaS card layout
├── model.joblib         # Pre-trained LinearSVC model
├── vectorizer.joblib    # Fitted feature vectorizer
├── secret.key           # AES-256 persistent secret key
├── requirements.txt     # Python dependencies
├── build/               # Self-contained deployment folder
│   ├── index.html       # Production UI
│   ├── style.css        # Clean styling matching the screenshot
│   ├── app.js           # Interactive client logic & API handlers
│   ├── server.py        # Production server entrypoint
│   ├── features.py      # Feature engineering
│   ├── main.py          # Detection & encryption logic
│   ├── model.joblib     # Pre-trained model
│   ├── vectorizer.joblib# Feature vectorizer
│   ├── secret.key       # Encryption key
│   ├── requirements.txt # Minimal deployment dependencies
│   ├── Dockerfile       # Container definition for Cloud deployment
│   └── README.md        # Deployment guide
└── README.md            # Main documentation
```

---

## 🎨 UI Features (Matching the Design)

- **Floating Pill Switcher**: Toggle between `🔒 ENCRYPT PII` and `🔓 DECRYPT TEXT`.
- **Top Input Box**: Dashed container for typing/pasting text, quick sample chips, or uploading `.txt` files.
- **Set Password (Key)**: Custom key / password field (with random key generator 🎲 and copy button 📋).
- **Real-Time Category Badges**: Displays count of detected `Names`, `Phones`, `Emails`, `Addresses`.
- **Coral-Red Action Button**: High-contrast `ENCRYPT PII` / `DECRYPT TEXT` CTA.
- **Right Result Container**:
  - Empty state displays *"Result will appear here"*.
  - When processed: Shows detected PII words with confidence scores, AES-256 encrypted ciphertext with one-click copy, and round-trip decrypted verification.

---

## 🚀 How to Run

### Option 1: Run the UI Locally
```bash
python ui.py
```
*(Automatically opens `http://127.0.0.1:5000` in your web browser)*

### Option 2: CLI Usage (`main.py`)
- **Direct String Processing**:
  ```bash
  python main.py --text "Contact Konstantin Becker at konstantin.becker@gmail.com or 0475 4429797." --key "ABC123"
  ```
- **Decrypt Ciphertext**:
  ```bash
  python main.py --decrypt "Hello [ENC:...]" --key "ABC123"
  ```

---

## 📦 Production Deployment (`build/` folder)

The `build/` folder is completely self-contained with no extra dependencies needed:
```bash
cd build
pip install -r requirements.txt
python server.py
```
Or build and run with Docker:
```bash
cd build
docker build -t pii-app .
docker run -p 5000:5000 pii-app
```
