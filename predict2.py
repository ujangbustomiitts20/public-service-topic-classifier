import joblib
import os

def main():
    model_path = "models/topic_classifier.joblib"

    if not os.path.exists(model_path):
        print("❌ Model belum ditemukan. Jalankan dulu: python src/train.py")
        return

    # Load model
    pipe = joblib.load(model_path)
    print("✅ Model berhasil dimuat.")
    print("Ketik kalimat untuk diprediksi (atau ketik 'exit' untuk keluar):\n")

    # Loop interaktif
    while True:
        teks = input("🗒️  Masukkan teks: ").strip()
        if teks.lower() in ["exit", "quit", "keluar"]:
            print("👋 Selesai. Terima kasih!")
            break
        if teks == "":
            continue

        pred = pipe.predict([teks])[0]
        print(f"🔹 Topik terdeteksi: {pred}\n")

if __name__ == "__main__":
    main()
