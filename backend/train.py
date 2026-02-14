import argparse
import json
import os
import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from joblib import dump

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # remove urls/emails
    s = re.sub(r"(https?://\S+)|(\w+@\w+\.\w+)", " ", s)
    # remove numbers
    s = re.sub(r"\d+", " ", s)
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_dataset(path: str):
    df = pd.read_csv(path, encoding="latin-1")
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    print("📄 Columns detected:", df.columns.tolist())

    # Try to detect possible label/text columns
    possible_label_cols = ["label", "category", "class", "target", "v1"]
    possible_text_cols = ["text", "message", "content", "email", "v2"]

    label_col = None
    text_col = None

    for c in df.columns:
        if any(key in c for key in possible_label_cols):
            label_col = c
        if any(key in c for key in possible_text_cols):
            text_col = c

    # Fallback: assume first two columns are label/text
    if label_col is None or text_col is None:
        label_col, text_col = df.columns[:2]
        print(f"⚠️ Could not detect columns automatically. Using first two: {label_col}, {text_col}")

    # Rename and clean
    df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})
    df = df.dropna()

    # Normalize label values
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"spam": 1, "ham": 0, "non-spam": 0, "nonspam": 0})
    )

    # Drop invalid/missing labels
    df = df[df["label"].isin([0, 1])]
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]

    if len(df) == 0:
        print("❌ Dataset is empty after processing. Check if labels are 'spam'/'ham' or adjust label mapping.")
        print("Unique labels found:", df["label"].unique() if "label" in df else "N/A")
        exit(1)

    print(f"✅ Loaded {len(df)} valid samples.")
    return df


def main(args):
    df = load_dataset(args.data)
    X = df["text"].tolist()
    y = df["label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200, n_jobs=1))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1)

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    dump(pipe, args.model)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "samples": {
            "train": len(X_train),
            "test": len(X_test)
        },
        "vectorizer": {
            "max_features": args.max_features,
            "ngram_range": [1, 2]
        },
        "model": "LogisticRegression(max_iter=200)"
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to:", args.model)
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--model", default=os.path.join("models", "model.joblib"))
    parser.add_argument("--report", default=os.path.join("models", "metrics.json"))
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_features", type=int, default=20000)
    args = parser.parse_args()
    main(args)
