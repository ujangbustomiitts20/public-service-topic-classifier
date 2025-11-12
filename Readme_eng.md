Perfect 👍 Here’s the **English version of the README.md**, written in a clean and professional GitHub style — suitable for public repositories or portfolio demonstration.

---

```markdown
# 🧠 Topic Classifier — TF-IDF + SGDClassifier

A simple text classification project that categorizes public service requests or complaints into topics such as *administration*, *licensing*, and *infrastructure*.  
This project uses **TF-IDF (Text Feature Extraction)** and **SGDClassifier** to build a lightweight multi-class model that can run efficiently even on Google Colab.

---

## 🚀 Key Features
- **Automatic text preprocessing** using `TfidfVectorizer`
- **Multi-class classification** with `SGDClassifier(loss="log_loss")`
- **Scikit-learn pipeline** (easy to train, save, and reload)
- **Fast accuracy evaluation** even on small datasets
- **Works both on Google Colab and local environments**

---

## 🧩 Project Structure



topic-classifier/
├─ data/
│  └─ train.csv                 # Text & label dataset
├─ models/
│  └─ topic_classifier.joblib   # Saved model (after training)
├─ src/
│  ├─ train.py                  # Model training script
│  └─ predict.py                # Script for making new predictions
├─ requirements.txt
└─ README.md

````

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/username/topic-classifier.git
cd topic-classifier

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

---

## 🧠 Model Training

Prepare your dataset (`data/train.csv`) with two columns:

```csv
teks,label
"Perpanjangan KTP elektronik di kecamatan Ciputat",administrasi
"Pengajuan izin usaha mikro kecil (IUMK) online",perizinan
"Perbaikan lampu jalan dan pelaporan infrastruktur",infrastruktur
...
```

Then run the training script:

```bash
python src/train.py --input_csv data/train.csv --test_size 0.5
```

Expected output:

```
Accuracy: 0.90
Report:
               precision  recall  f1-score  support
administrasi      ...
perizinan         ...
infrastruktur     ...
```

The trained model will be saved in `models/topic_classifier.joblib`.

---

## 🔍 Predict New Sentences

```bash
python src/predict.py \
  --texts "Saya ingin memperpanjang KTP-el" \
          "Laporan trotoar rusak di Pamulang" \
          "Bagaimana cara mengurus izin usaha mikro?"
```

Example output:

```
- 'Saya ingin memperpanjang KTP-el' -> administrasi
- 'Laporan trotoar rusak di Pamulang' -> infrastruktur
- 'Bagaimana cara mengurus izin usaha mikro?' -> perizinan
```

---

## 💡 Quickstart on Google Colab

Simply copy and run the following snippet in a Colab cell:

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd, joblib, os

data = [
    ("Perpanjangan KTP elektronik di kecamatan Ciputat", "administrasi"),
    ("Pengajuan izin usaha mikro kecil (IUMK) online", "perizinan"),
    ("Perbaikan lampu jalan dan pelaporan infrastruktur", "infrastruktur"),
    ("Izin reklame dan pajak reklame tahunan", "perizinan"),
    ("Laporan jalan berlubang di Serpong", "infrastruktur"),
    ("Perekaman KTP-el untuk pemula 17 tahun", "administrasi"),
]
df = pd.DataFrame(data, columns=["teks","label"])

Xtr, Xte, ytr, yte = train_test_split(df["teks"], df["label"], test_size=0.5, stratify=df["label"])
pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2)), SGDClassifier(loss="log_loss"))
pipe.fit(Xtr, ytr)
print("Accuracy:", accuracy_score(yte, pipe.predict(Xte)))
joblib.dump(pipe, "topic_classifier.joblib")
```

---

## 🧰 Dependencies

```text
pandas
scikit-learn
joblib
```

---

## 📚 License

This project is released under the MIT License.
You’re free to use, modify, and share it for educational or research purposes.

---

## 👤 Author

**Aolia Ikhwanudin**
AI & Data Researcher | System Analyst | Lecturer
🌐 [GitHub Profile](https://github.com/ujangbustomiitts20)

```
