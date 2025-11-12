import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main(args):
    df = pd.read_csv(args.input_csv)  # kolom: teks,label
    X, y = df["teks"], df["label"]

    # Stratify aman untuk data kecil: pastikan test_size menghasilkan n_test >= jumlah kelas
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2)),
        SGDClassifier(loss="log_loss", random_state=42)
    )
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    print("Accuracy:", round(accuracy_score(yte, pred), 3))
    print("\nReport:\n", classification_report(yte, pred))

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(pipe, args.model_path)
    print(f"Model saved to: {args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/train.csv")
    parser.add_argument("--model_path", type=str, default="models/topic_classifier.joblib")
    parser.add_argument("--test_size", type=float, default=0.5)  # aman untuk dataset mini
    args = parser.parse_args()
    main(args)
