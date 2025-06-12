# Klasifikasi Berita Berbahasa Indonesia

Proyek ini mengklasifikasikan artikel berita berbahasa Indonesia ke dalam 5 kategori (Bola, News, Bisnis, Teknologi, Otomotif) menggunakan model IndoBERT. Dilengkapi aplikasi web interaktif (Streamlit) untuk demonstrasi.

## Ringkasan

* **Tujuan**: Klasifikasi teks berita Indonesia.
* **Model**: IndoBERT (`indobenchmark/indobert-base-p1`) yang di-fine-tune.
* **Dataset**: `SEACrowd/indonesian_news_dataset` dari Hugging Face.
* **Aplikasi (`app.py`)**:
    * Input teks berita.
    * Menampilkan kategori prediksi (Bola ⚽, News 📰, Bisnis 💼, Teknologi 💻, Otomotif 🚗) beserta probabilitasnya.
    * Statistik teks dan contoh artikel.
* **Pelatihan (`code.ipynb`)**:
    * Pra-pemrosesan: Hapus karakter non-alfanumerik, spasi ganda, lowercase.
    * Tokenisasi: `AutoTokenizer` IndoBERT, `max_length=128`.
    * Parameter: AdamW optimizer, lr=2e-5, 3 epochs, batch=16, class weights.
* **Performa (Validasi)**:
    * IndoBERT: Akurasi ~0.9487, F1 ~0.9501.
    * Baseline (XGBoost + TF-IDF): Akurasi ~0.9128, F1 ~0.9127.

## Struktur File

```
.
├── app.py                # Aplikasi Streamlit
├── code.ipynb            # Notebook pelatihan model
├── model_save/           # Direktori model (dihasilkan oleh code.ipynb)
│   ├── config.json
│   ├── model.safetensors (atau pytorch_model.bin)
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
└── README.md             # File ini
```

**Catatan tentang `model_save/`**:
Direktori `model_save/` akan dibuat secara otomatis ketika Anda menjalankan sel-sel pelatihan model di dalam notebook `code.ipynb`, khususnya pada bagian penyimpanan model (`model_to_save.save_pretrained(output_dir)` dan `tokenizer.save_pretrained(output_dir)`). Direktori ini berisi file konfigurasi, bobot model, dan tokenizer yang diperlukan oleh `app.py` untuk melakukan inferensi.

## Setup

1. **Virtual Environment (Opsional tapi direkomendasikan)**:
   ```bash
   python -m venv env
   env\Scripts\activate  # Windows
   # source env/bin/activate  # macOS/Linux
   ```

2. **Install Dependencies**:
   ```bash
   pip install transformers datasets pandas scikit-learn torch numpy seaborn matplotlib streamlit
   ```
   
   Untuk `code.ipynb` (pelatihan & baseline):
   ```bash
   pip install datasets seacrowd xgboost
   ```

## Menjalankan Aplikasi

1. Pastikan direktori `model_save/` berisi model yang telah dilatih (dihasilkan dari `code.ipynb`).
2. Jalankan:
   ```bash
   streamlit run app.py
   ```

## Dikerjakan Oleh
Tiara Agustin (2208107010004)
M. Taris Rizki (2208107010047)
