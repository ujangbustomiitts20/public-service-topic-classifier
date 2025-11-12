import argparse
import joblib

def main(args):
    pipe = joblib.load(args.model_path)
    texts = args.texts
    preds = pipe.predict(texts)
    for t, p in zip(texts, preds):
        print(f"- '{t}' -> {p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/topic_classifier.joblib")
    parser.add_argument("--texts", nargs="+", required=True, help="Kalimat yang akan diprediksi")
    args = parser.parse_args()
    main(args)
