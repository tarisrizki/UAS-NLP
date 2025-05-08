
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# Load model yang sudah dilatih
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./model_save/")
    model = AutoModelForSequenceClassification.from_pretrained("./model_save/")
    
    # Pindahkan model ke device yang tersedia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Label mapping
    label_mapping = {0: 'bisnis', 1: 'bola', 2: 'news', 3: 'otomotif', 4: 'tekno'}
    
    return tokenizer, model, device, label_mapping

# Fungsi preprocessing
def preprocess_text(text):
    # Hapus karakter non-alfanumerik (kecuali spasi)
    text = re.sub(r'[^\w\s]', '', text)
    # Hapus multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Lowercase
    text = text.lower().strip()
    return text

# Fungsi prediksi
def predict_category(text, tokenizer, model, device, label_mapping):
    # Preprocessing
    text = preprocess_text(text)
    
    # Tokenisasi
    encoded_text = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Pindahkan ke device
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    
    # Evaluation mode
    model.eval()
    
    # Prediksi
    with torch.no_grad():
        outputs = model(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_mask
        )
    
    # Ambil prediksi dengan probabilitas tertinggi
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    # Convert ke label
    predicted_label = label_mapping[prediction]
    
    # Hitung probabilitas untuk semua kelas
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    return {
        'category': predicted_label,
        'probabilities': {label_mapping[i]: float(prob) for i, prob in enumerate(probabilities)}
    }

# UI Aplikasi
st.title("Klasifikasi Berita Berbahasa Indonesia")
st.write("""
Aplikasi ini mengklasifikasikan artikel berita berbahasa Indonesia ke dalam 5 kategori:
- bola (olahraga)
- news (berita umum)
- bisnis
- tekno (teknologi)
- otomotif
""")

# Load model
tokenizer, model, device, label_mapping = load_model()

# Input teks untuk klasifikasi
text_input = st.text_area("Masukkan teks berita:", height=200)

if st.button("Klasifikasikan"):
    if not text_input.strip():
        st.error("Harap masukkan teks berita terlebih dahulu.")
    else:
        with st.spinner("Mengklasifikasikan..."):
            # Prediksi
            prediction = predict_category(text_input, tokenizer, model, device, label_mapping)
            
            # Tampilkan hasil
            st.success(f"Kategori Berita: **{prediction['category'].upper()}**")
            
            # Visualisasi probabilitas
            st.write("Probabilitas per kategori:")
            
            # Buat dataframe untuk visualisasi
            import pandas as pd
            import matplotlib.pyplot as plt
            
            probs_df = pd.DataFrame({
                'Kategori': list(prediction['probabilities'].keys()),
                'Probabilitas': list(prediction['probabilities'].values())
            })
            
            # Sort by probability
            probs_df = probs_df.sort_values('Probabilitas', ascending=False)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(
                probs_df['Kategori'], 
                probs_df['Probabilitas'],
                color=['#1f77b4' if cat == prediction['category'] else '#d3d3d3' for cat in probs_df['Kategori']]
            )
            
            # Tambahkan label
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{height:.2%}',
                    ha='center', 
                    va='bottom'
                )
            
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probabilitas')
            ax.set_title('Probabilitas per Kategori')
            
            # Tampilkan plot
            st.pyplot(fig)

# Contoh artikel per kategori
st.sidebar.header("Contoh Artikel")

example_articles = {
    "Bola": "Timnas Indonesia akan bertanding melawan Malaysia dalam lanjutan kualifikasi Piala Dunia 2026. Pertandingan akan digelar di Stadion Gelora Bung Karno, Jakarta, pada Selasa (17/6/2025) pukul 19.30 WIB.",
    "News": "Presiden Republik Indonesia meresmikan pembangunan jalan tol baru yang menghubungkan Jakarta dan Bandung. Proyek ini diharapkan dapat mengurangi kemacetan dan mempercepat waktu tempuh antar kota.",
    "Bisnis": "Nilai tukar rupiah menguat terhadap dolar AS pada perdagangan hari ini. Penguatan ini didorong oleh aliran modal asing yang masuk ke pasar keuangan domestik seiring dengan perbaikan kondisi ekonomi global.",
    "Tekno": "Apple mengumumkan peluncuran iPhone 16 dengan sejumlah fitur baru, termasuk kemampuan AI generatif yang canggih. Perangkat ini akan dijual mulai pekan depan dengan harga mulai dari Rp 15 juta.",
    "Otomotif": "Produsen mobil listrik Hyundai meluncurkan model terbaru IONIQ 7 di Indonesia. Mobil ini memiliki jangkauan hingga 600 km dengan sekali pengisian daya dan dibanderol mulai dari Rp 800 juta."
}

selected_example = st.sidebar.selectbox("Pilih contoh artikel:", list(example_articles.keys()))

if st.sidebar.button("Gunakan Contoh"):
    st.text_area("Masukkan teks berita:", example_articles[selected_example], height=200)
    
    # Secara otomatis klasifikasikan contoh
    with st.spinner("Mengklasifikasikan..."):
        # Prediksi
        prediction = predict_category(example_articles[selected_example], tokenizer, model, device, label_mapping)
        
        # Tampilkan hasil
        st.success(f"Kategori Berita: **{prediction['category'].upper()}**")
        
        # Visualisasi probabilitas
        st.write("Probabilitas per kategori:")
        
        # Buat dataframe untuk visualisasi
        import pandas as pd
        import matplotlib.pyplot as plt
        
        probs_df = pd.DataFrame({
            'Kategori': list(prediction['probabilities'].keys()),
            'Probabilitas': list(prediction['probabilities'].values())
        })
        
        # Sort by probability
        probs_df = probs_df.sort_values('Probabilitas', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            probs_df['Kategori'], 
            probs_df['Probabilitas'],
            color=['#1f77b4' if cat == prediction['category'] else '#d3d3d3' for cat in probs_df['Kategori']]
        )
        
        # Tambahkan label
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2%}',
                ha='center', 
                va='bottom'
            )
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probabilitas')
        ax.set_title('Probabilitas per Kategori')
        
        # Tampilkan plot
        st.pyplot(fig)

# Info developer
st.sidebar.markdown("---")
st.sidebar.info("""
### Tentang Aplikasi
Aplikasi ini adalah bagian dari proyek Klasifikasi Berita Berbahasa Indonesia menggunakan model IndoBERT.

**Kelas**: SINF6054 - Pemrosesan Bahasa Alami
""")
