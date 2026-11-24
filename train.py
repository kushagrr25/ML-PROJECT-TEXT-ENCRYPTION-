import os
import time
import ast
import joblib
import argparse
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import extract_token_features, ensure_int32_csr

def train_model(
    dataset_path="pii_dataset.csv",
    model_output="model.joblib",
    vectorizer_output="vectorizer.joblib",
    key_output="secret.key",
    max_rows=3500,
    test_size=0.2,
    random_state=42
):
    # Locate dataset
    if not os.path.exists(dataset_path):
        alt_path = os.path.join("dataset", "pii_dataset.csv")
        if os.path.exists(alt_path):
            dataset_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at '{dataset_path}' or '{alt_path}'")

    print(f"=== Training SVM on PII Dataset ===")
    print(f"Loading data from: {dataset_path}")
    t0 = time.time()
    df = pd.read_csv(dataset_path, nrows=max_rows) if max_rows else pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} documents in {time.time() - t0:.2f}s")

    print("\n[Step 1/4] Extracting token features & marking PII labels as IMPORTANT...")
    t0 = time.time()
    X_dicts = []
    y_labels = []
    pii_counts = 0

    for _, row in df.iterrows():
        try:
            tokens = ast.literal_eval(row['tokens'])
            labels = ast.literal_eval(row['labels'])
        except Exception:
            continue

        for i, (tok, lbl) in enumerate(zip(tokens, labels)):
            X_dicts.append(extract_token_features(tokens, i))
            # Mark all PII labels as 1 ('IMPORTANT'), and 'O' as 0 ('NOT_IMPORTANT')
            if lbl != 'O':
                y_labels.append(1)
                pii_counts += 1
            else:
                y_labels.append(0)

    total_tokens = len(y_labels)
    print(f"Extracted {total_tokens:,} tokens in {time.time() - t0:.2f}s")
    print(f"  • Important (PII) tokens: {pii_counts:,} ({pii_counts / total_tokens * 100:.2f}%)")
    print(f"  • Neutral (O) tokens:     {total_tokens - pii_counts:,}")

    y_arr = np.array(y_labels, dtype=np.int32)

    print(f"\n[Step 2/4] Splitting train ({int((1-test_size)*100)}%) / test ({int(test_size*100)}%)...")
    X_train_dict, X_test_dict, y_train, y_test = train_test_split(
        X_dicts, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )

    print("\n[Step 3/4] Vectorizing features with DictVectorizer...")
    t0 = time.time()
    vec = DictVectorizer(sparse=True)
    X_train = ensure_int32_csr(vec.fit_transform(X_train_dict))
    X_test = ensure_int32_csr(vec.transform(X_test_dict))
    print(f"Vectorized shape: {X_train.shape} in {time.time() - t0:.2f}s")

    print("\n[Step 4/4] Training Linear Support Vector Machine (LinearSVC)...")
    t0 = time.time()
    svm = LinearSVC(
        dual=False,
        class_weight='balanced',
        max_iter=3000,
        random_state=random_state,
        C=1.0
    )
    svm.fit(X_train, y_train)
    print(f"SVM training completed in {time.time() - t0:.2f}s")

    # Evaluation
    print("\n" + "="*55)
    print("MODEL EVALUATION RESULTS (Test Set)")
    print("="*55)
    y_pred = svm.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['NOT_IMPORTANT (O)', 'IMPORTANT (PII)'], digits=4))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"  True Negatives (O correctly identified):    {tn:,}")
    print(f"  False Positives (O misclassified as PII):   {fp:,}")
    print(f"  False Negatives (PII missed):              {fn:,}")
    print(f"  True Positives (PII correctly identified): {tp:,}")
    print("="*55)

    # Save artifacts
    print(f"\nSaving model to: {model_output}")
    joblib.dump(svm, model_output)

    print(f"Saving vectorizer to: {vectorizer_output}")
    joblib.dump(vec, vectorizer_output)

    # Generate encryption key if not exists
    if not os.path.exists(key_output):
        key = Fernet.generate_key()
        with open(key_output, 'wb') as f:
            f.write(key)
        print(f"Generated secret encryption key at: {key_output}")
    else:
        print(f"Using existing encryption key at: {key_output}")

    print("\nTraining completed successfully!")
    return svm, vec

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SVM on PII dataset")
    parser.add_argument("--data", default="pii_dataset.csv", help="Path to pii_dataset.csv")
    parser.add_argument("--rows", type=int, default=3500, help="Max rows to train on")
    args = parser.parse_args()

    train_model(dataset_path=args.data, max_rows=args.rows)
