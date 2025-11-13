
```markdown
# 🧠 Topic Classifier — TF-IDF + SGDClassifier

Klasifikasi teks sederhana untuk menentukan **topik aduan atau permohonan layanan publik** seperti *administrasi*, *perizinan*, dan *infrastruktur*.  
Proyek ini menggunakan pendekatan **TF-IDF (Text Feature Extraction)** dan **SGDClassifier** untuk membuat model klasifikasi multi-kelas yang ringan dan cepat dijalankan, bahkan di Google Colab.

## 🚀 Fitur Utama
- **Preprocessing otomatis** dengan `TfidfVectorizer`  
- **Klasifikasi multi-kelas** dengan `SGDClassifier(loss="log_loss")`  
- **Pipeline scikit-learn** (mudah disimpan & dimuat ulang)  
- **Akurasi cepat dihitung** meskipun dataset kecil  
- **Dapat dijalankan di Google Colab atau lokal**

## 🧩 Struktur Proyek

topic-classifier/
├─ data/
│  └─ train.csv                 # Dataset teks & label
├─ models/
│  └─ topic_classifier.joblib   # Model tersimpan (setelah training)
├─ src/
│  ├─ train.py                  # Script pelatihan model
│  └─ predict.py                # Script untuk prediksi baru
├─ requirements.txt
└─ README.md

```
## 📦 Instalasi

```bash
# 1. Clone repositori
git clone https://github.com/username/topic-classifier.git
cd topic-classifier

# 2. Buat environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

---

## 🧠 Pelatihan Model

Gunakan dataset CSV (`data/train.csv`) dengan dua kolom:

```csv
teks,label
"Perpanjangan KTP elektronik di kecamatan Ciputat",administrasi
"Pengajuan izin usaha mikro kecil (IUMK) online",perizinan
"Perbaikan lampu jalan dan pelaporan infrastruktur",infrastruktur
...
```

Jalankan:

```bash
python src/train.py --input_csv data/train.csv --test_size 0.5
```

Output terminal:

```
Accuracy: 0.90
Report:
               precision  recall  f1-score  support
administrasi      ...
perizinan         ...
infrastruktur     ...
```

Model tersimpan di `models/topic_classifier.joblib`.

---

## 🔍 Prediksi Kalimat Baru

Cara 1 langsung masukan topik
```bash
python src/predict.py \
  --texts "Saya ingin memperpanjang KTP-el" \
          "Laporan trotoar rusak di Pamulang" \
          "Bagaimana cara mengurus izin usaha mikro?"
```

Output:

```
- 'Saya ingin memperpanjang KTP-el' -> administrasi
- 'Laporan trotoar rusak di Pamulang' -> infrastruktur
- 'Bagaimana cara mengurus izin usaha mikro?' -> perizinan

```
Cara 2 interkatif
```
python src/predict2.py 
```

---


## 🧰 Dependensi

```text
pandas
scikit-learn
joblib
```

---

## 📚 Lisensi

Proyek ini dirilis di bawah lisensi MIT.
Silakan gunakan dan kembangkan ulang untuk keperluan riset atau edukasi.
---

## 👤 Kontributor

**Aolia Ikhwanudin**
AI & Data Researcher | System Analyst | Lecturer
🌐 [https://github.com/ujangbustomiitts20](https://github.com/ujangbustomiitts20)

