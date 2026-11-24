import os
import re
import joblib
import argparse
import hashlib
import base64
from cryptography.fernet import Fernet
from features import tokenize_with_spans, extract_token_features, ensure_int32_csr

class PIIEncryptionService:
    """
    Handles PII detection using trained SVM and AES-256 token encryption/decryption.
    Supports persistent keys as well as user-specified custom passwords.
    """
    def __init__(
        self,
        model_path="model.joblib",
        vectorizer_path="vectorizer.joblib",
        key_path="secret.key"
    ):
        # Support fallback paths in outputs/ if in root
        if not os.path.exists(model_path) and os.path.exists(os.path.join("outputs", "pii_svm_model.joblib")):
            model_path = os.path.join("outputs", "pii_svm_model.joblib")
        if not os.path.exists(vectorizer_path) and os.path.exists(os.path.join("outputs", "pii_vectorizer.joblib")):
            vectorizer_path = os.path.join("outputs", "pii_vectorizer.joblib")
        if not os.path.exists(key_path) and os.path.exists(os.path.join("outputs", "secret.key")):
            key_path = os.path.join("outputs", "secret.key")

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Model or vectorizer not found. Please run 'python train.py' first."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        # Load or create persistent encryption key
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                self.key = f.read().strip()
        else:
            self.key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(self.key)

        self.cipher = Fernet(self.key)

    def get_cipher(self, password_or_key: str = None) -> Fernet:
        """Returns a Fernet cipher from a custom password or default key."""
        if not password_or_key:
            return self.cipher
        pw = str(password_or_key).strip()
        if not pw:
            return self.cipher

        # If already a valid 44-char base64 key
        if len(pw) == 44 and pw.endswith('='):
            try:
                return Fernet(pw.encode('utf-8'))
            except Exception:
                pass

        # Derive 32-byte key from password using SHA-256
        derived_key = base64.urlsafe_b64encode(hashlib.sha256(pw.encode('utf-8')).digest())
        return Fernet(derived_key)

    def encrypt_word(self, word: str, cipher: Fernet = None) -> str:
        """Encrypts a single word/token into [ENC:<cipher>]."""
        c = cipher or self.cipher
        encrypted_bytes = c.encrypt(word.encode('utf-8'))
        return f"[ENC:{encrypted_bytes.decode('utf-8')}]"

    def decrypt_word(self, enc_token: str, password_or_key: str = None) -> str:
        """Decrypts [ENC:<cipher>] or raw token back to original plaintext."""
        return self.decrypt_text(enc_token, password_or_key=password_or_key)

    def decrypt_text(self, text: str, password_or_key: str = None) -> str:
        """
        Robustly decrypts text containing [ENC:...] tags or raw Fernet tokens
        using a custom password or default key.
        """
        if not text:
            return ""

        cipher = self.get_cipher(password_or_key)
        cleaned = text.strip()

        # Case 1: The entire input is a bracketed token, e.g. [ENC:gAAAAA...]
        bracket_match = re.match(r'^\[(?:ENC|enc|Enc):\s*([A-Za-z0-9_\-=]+)\s*\]$', cleaned)
        if bracket_match:
            try:
                return cipher.decrypt(bracket_match.group(1).encode('utf-8')).decode('utf-8')
            except Exception:
                try:
                    return self.cipher.decrypt(bracket_match.group(1).encode('utf-8')).decode('utf-8')
                except Exception:
                    pass

        # Case 2: The entire input is just a raw Fernet token without brackets
        if re.match(r'^[A-Za-z0-9_\-=]{50,}\={0,2}$', cleaned):
            try:
                return cipher.decrypt(cleaned.encode('utf-8')).decode('utf-8')
            except Exception:
                try:
                    return self.cipher.decrypt(cleaned.encode('utf-8')).decode('utf-8')
                except Exception:
                    pass

        # Case 3: Embedded [ENC:...] tags in arbitrary sentences
        def replace_tag(match):
            token_str = match.group(1).strip()
            try:
                decrypted_bytes = cipher.decrypt(token_str.encode('utf-8'))
                return decrypted_bytes.decode('utf-8')
            except Exception:
                try:
                    decrypted_bytes = self.cipher.decrypt(token_str.encode('utf-8'))
                    return decrypted_bytes.decode('utf-8')
                except Exception:
                    return match.group(0)

        result = re.sub(r'\[(?:ENC|enc|Enc):\s*([A-Za-z0-9_\-=]+?)\s*\]', replace_tag, text)

        # Case 4: Standalone raw Fernet tokens (starting with gAAAAA)
        def replace_raw(match):
            raw_token = match.group(0)
            try:
                return cipher.decrypt(raw_token.encode('utf-8')).decode('utf-8')
            except Exception:
                try:
                    return self.cipher.decrypt(raw_token.encode('utf-8')).decode('utf-8')
                except Exception:
                    return raw_token

        result = re.sub(r'\bgAAAAA[A-Za-z0-9_\-=]{50,}\={0,2}', replace_raw, result)

        return result

    def process_string(self, text: str, password_or_key: str = None) -> dict:
        """
        Tokenizes text, detects important PII words with SVM, and encrypts them.
        """
        if not text or not text.strip():
            return {
                "original_text": text,
                "encrypted_text": text,
                "important_words": [],
                "decrypted_text": text,
                "count": 0,
                "stats": {"names": 0, "phones": 0, "emails": 0, "addresses": 0, "others": 0}
            }

        cipher = self.get_cipher(password_or_key)

        tokens, spans = tokenize_with_spans(text)
        features = [extract_token_features(tokens, i) for i in range(len(tokens))]
        X = ensure_int32_csr(self.vectorizer.transform(features))

        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)

        important_spans = []
        important_words = []
        stats = {"names": 0, "phones": 0, "emails": 0, "addresses": 0, "others": 0}

        for i, (tok, span, pred, score) in enumerate(zip(tokens, spans, predictions, scores)):
            if pred == 1:
                start, end = span
                important_spans.append((start, end))

                # Simple category inference for stats
                lower = tok.lower()
                if '@' in tok:
                    stats["emails"] += 1
                elif any(c.isdigit() for c in tok) and any(c in tok for c in '+-()'):
                    stats["phones"] += 1
                elif any(addr in lower for addr in ('street', 'st', 'ave', 'road', 'rd', 'blvd', 'dr', 'ln')):
                    stats["addresses"] += 1
                elif tok.istitle():
                    stats["names"] += 1
                else:
                    stats["others"] += 1

                important_words.append({
                    "index": i,
                    "word": tok,
                    "span": (start, end),
                    "confidence": float(score)
                })

        # Replace important words with encrypted versions
        sorted_spans = sorted(important_spans, key=lambda s: s[0])
        result = []
        last_idx = 0
        for start, end in sorted_spans:
            result.append(text[last_idx:start])
            original_word = text[start:end]
            result.append(self.encrypt_word(original_word, cipher=cipher))
            last_idx = end
        result.append(text[last_idx:])
        encrypted_text = "".join(result)

        # Decrypt to verify round-trip fidelity
        decrypted_text = self.decrypt_text(encrypted_text, password_or_key=password_or_key)

        return {
            "original_text": text,
            "encrypted_text": encrypted_text,
            "important_words": important_words,
            "decrypted_text": decrypted_text,
            "count": len(important_words),
            "stats": stats
        }

def display_result(result: dict):
    print("\n" + "="*70)
    print("PII DETECTION & ENCRYPTION RESULT")
    print("="*70)
    print(f"Original Text:\n  {result['original_text']}")
    print("-" * 70)
    print(f"Important Words Detected ({len(result['important_words'])} words):")
    if result['important_words']:
        for item in result['important_words']:
            print(f"  * '{item['word']}' at {item['span']} (confidence: {item['confidence']:.3f})")
    else:
        print("  (None detected)")
    print("-" * 70)
    print(f"Encrypted Text (Important words encrypted with AES-256):\n  {result['encrypted_text']}")
    print("-" * 70)
    print(f"Decrypted Verification:\n  {result['decrypted_text']}")
    print(f"Match Verified: {'YES' if result['decrypted_text'] == result['original_text'] else 'NO'}")
    print("="*70 + "\n")

def interactive_cli(service: PIIEncryptionService):
    print("="*70)
    print("  SVM PII Detection & Encryption System")
    print("  Enter any string to select and encrypt important words.")
    print("  To decrypt text, prefix with 'decrypt:' or paste encrypted text.")
    print("  Type 'exit' to quit.")
    print("="*70)

    while True:
        try:
            user_input = input("\nEnter text > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ('exit', 'quit', 'q'):
                print("Goodbye!")
                break

            if user_input.lower().startswith("decrypt:"):
                raw_to_dec = user_input[len("decrypt:"):].strip()
                decrypted = service.decrypt_text(raw_to_dec)
                print("\n" + "-"*70)
                print(f"Decrypted Output:\n  {decrypted}")
                print("-"*70)
            elif "[ENC:" in user_input or user_input.startswith("gAAAAA"):
                decrypted = service.decrypt_text(user_input)
                print("\n" + "-"*70)
                print("Detected Encrypted String -> Decrypted Output:")
                print(f"  {decrypted}")
                print("-"*70)
            else:
                result = service.process_string(user_input)
                display_result(result)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

def main():
    parser = argparse.ArgumentParser(description="PII Token Detection & Encryption")
    parser.add_argument("--text", "-t", type=str, help="Input string to process")
    parser.add_argument("--key", "-k", type=str, default=None, help="Custom password/key to use")
    parser.add_argument("--decrypt", "-d", type=str, help="Decrypt an encrypted string")
    args = parser.parse_args()

    service = PIIEncryptionService()

    if args.decrypt:
        plain = service.decrypt_text(args.decrypt, password_or_key=args.key)
        print("\nDecrypted Output:\n" + plain)
    elif args.text:
        result = service.process_string(args.text, password_or_key=args.key)
        display_result(result)
    else:
        interactive_cli(service)

if __name__ == '__main__':
    main()
