import streamlit as st
import torch
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import base64
from PIL import Image
import io

# Check dependencies first
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    st.error("""
    ### Error: `transformers` package tidak lengkap
    
    Silakan install dependency yang diperlukan dengan perintah:
    ```
    pip install "transformers[torch]"
    ```
    """)
    st.stop()

# Try to import optional packages
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Page configuration
st.set_page_config(
    page_title="Indonesian News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gambar base64 untuk background dan assets
background_pattern = """
data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPSc0MCcgaGVpZ2h0PSc0MCcgdmlld0JveD0nMCAwIDQwIDQwJz48ZyBmaWxsLXJ1bGU9J2V2ZW5vZGQnPjxnIGZpbGw9JyNmZmZmZmYnIGZpbGwtb3BhY2l0eT0nMC4wNSc+PHBhdGggZD0nTTAgMGg0MHY0MEgwVjB6TTE1IDE1aDEwdjEwSDE1VjE1eicvPjwvZz48L2c+PC9zdmc+
"""

logo_svg = """
<svg width="180" height="50" viewBox="0 0 180 50" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M13.4 35.4V15H17.9V35.4H13.4ZM26.7953 35.4V20.8H9.9953V16.9H31.2953V35.4H26.7953ZM39.9906 35.4V15H55.9906C58.5239 15 60.6573 15.7 62.3906 17.1C64.1573 18.5 65.0406 20.3333 65.0406 22.6C65.0406 24.8667 64.1573 26.7 62.3906 28.1C60.6573 29.5 58.5239 30.2 55.9906 30.2H44.4906V35.4H39.9906ZM44.4906 26.3H55.5906C56.7906 26.3 57.7239 26 58.3906 25.4C59.0906 24.7667 59.4406 23.8667 59.4406 22.7C59.4406 21.5333 59.0906 20.6333 58.3906 20C57.7239 19.3667 56.7906 19.05 55.5906 19.05H44.4906V26.3ZM68.1859 35.4V15H72.6859V35.4H68.1859ZM91.5813 35.7C89.3146 35.7 87.2813 35.25 85.4813 34.35C83.6813 33.45 82.2646 32.2 81.2313 30.6C80.1979 28.9667 79.6813 27.1333 79.6813 25.1C79.6813 23.0667 80.1979 21.25 81.2313 19.65C82.2646 18.0167 83.6813 16.7667 85.4813 15.9C87.2813 15 89.3146 14.55 91.5813 14.55C93.8479 14.55 95.8979 15 97.7313 15.9C99.5646 16.7667 100.998 18.0167 102.031 19.65C103.065 21.25 103.581 23.0667 103.581 25.1C103.581 27.1333 103.065 28.9667 102.031 30.6C100.998 32.2 99.5646 33.45 97.7313 34.35C95.8979 35.25 93.8479 35.7 91.5813 35.7ZM91.5813 31.5C93.0479 31.5 94.3646 31.2167 95.5313 30.65C96.6979 30.0833 97.6146 29.2833 98.2813 28.25C98.9479 27.2167 99.2813 26.0333 99.2813 24.7C99.2813 23.3667 98.9479 22.1833 98.2813 21.15C97.6146 20.1167 96.6979 19.3167 95.5313 18.75C94.3646 18.1833 93.0479 17.9 91.5813 17.9C90.1146 17.9 88.7979 18.1833 87.6313 18.75C86.4646 19.3167 85.5479 20.1167 84.8813 21.15C84.2146 22.1833 83.8813 23.3667 83.8813 24.7C83.8813 26.0333 84.2146 27.2167 84.8813 28.25C85.5479 29.2833 86.4646 30.0833 87.6313 30.65C88.7979 31.2167 90.1146 31.5 91.5813 31.5ZM111.366 35.4V27.3L111.116 25.1V15H115.616V35.4H111.366ZM142.161 35.4L142.111 22.35L134.811 34.25H132.261L124.961 22.65V35.4H120.761V15H124.461L133.661 29.75L142.661 15H146.311L146.361 35.4H142.161ZM153.356 35.4V15H169.356C171.89 15 174.023 15.7 175.756 17.1C177.523 18.5 178.406 20.3333 178.406 22.6C178.406 24.8667 177.523 26.7 175.756 28.1C174.023 29.5 171.89 30.2 169.356 30.2H157.856V35.4H153.356ZM157.856 26.3H168.956C170.156 26.3 171.09 26 171.756 25.4C172.456 24.7667 172.806 23.8667 172.806 22.7C172.806 21.5333 172.456 20.6333 171.756 20C171.09 19.3667 170.156 19.05 168.956 19.05H157.856V26.3Z" fill="#0068C9"/>
<path d="M4 9V41" stroke="#0068C9" stroke-width="7" stroke-linecap="round"/>
</svg>
"""

illustration_svg = """
<svg width="300" height="200" viewBox="0 0 300 200" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="300" height="200" rx="10" fill="#F0F7FF"/>
<path d="M150 35C94.7715 35 50 79.7715 50 135H250C250 79.7715 205.228 35 150 35Z" fill="#CCDDF5"/>
<rect x="70" y="75" width="160" height="100" rx="5" fill="white" stroke="#0068C9" stroke-width="2"/>
<rect x="85" y="90" width="70" height="10" rx="2" fill="#0068C9"/>
<rect x="85" y="110" width="130" height="6" rx="2" fill="#E1EEFF"/>
<rect x="85" y="124" width="130" height="6" rx="2" fill="#E1EEFF"/>
<rect x="85" y="138" width="130" height="6" rx="2" fill="#E1EEFF"/>
<rect x="85" y="152" width="70" height="6" rx="2" fill="#E1EEFF"/>
<circle cx="215" cy="95" r="15" fill="#F0F7FF" stroke="#0068C9" stroke-width="2"/>
<path d="M208 95H222M215 88V102" stroke="#0068C9" stroke-width="2" stroke-linecap="round"/>
</svg>
"""

wave_svg = """
<svg width="100%" height="50" viewBox="0 0 1200 120" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z" opacity=".25" fill="#0068C9"/>
    <path d="M0,0V15.81C13,36.92,27.64,56.86,47.69,72.05,99.41,111.27,165,111,224.58,91.58c31.15-10.15,60.09-26.07,89.67-39.8,40.92-19,84.73-46,130.83-49.67,36.26-2.85,70.9,9.42,98.6,31.56,31.77,25.39,62.32,62,103.63,73,40.44,10.79,81.35-6.69,119.13-24.28s75.16-39,116.92-43.05c59.73-5.85,113.28,22.88,168.9,38.84,30.2,8.66,59,6.17,87.09-7.5,22.43-10.89,48-26.93,60.65-49.24V0Z" opacity=".5" fill="#0068C9"/>
    <path d="M0,0V5.63C149.93,59,314.09,71.32,475.83,42.57c43-7.64,84.23-20.12,127.61-26.46,59-8.63,112.48,12.24,165.56,35.4C827.93,77.22,886,95.24,951.2,90c86.53-7,172.46-45.71,248.8-84.81V0Z" fill="#0068C9"/>
</svg>
"""

# Custom CSS with Laravel-like styling
st.markdown(f"""
<style>
    /* Base Styles */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@200;400;500;700&family=Roboto+Slab:wght@400;700&display=swap');
    
    * {{
        font-family: 'Poppins', sans-serif;
        box-sizing: border-box;
    }}
    
    /* Framework layout customization */
    .main {{
        background-image: url('{background_pattern}');
        background-color: #f8fafc;
        padding: 0 !important;
        margin: 0;
    }}
    
    .block-container {{
        padding-top: 0 !important;
        max-width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }}
    
    /* Header Section */
    .header {{
        background: linear-gradient(135deg, #0068c9 0%, #1e3a8a 100%);
        padding: 2rem 5%;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
        margin-bottom: 0;
    }}
    
    .header-content {{
        flex: 1;
        z-index: 2;
    }}
    
    .header h1 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Roboto Slab', serif;
        color: white;
    }}
    
    .header p {{
        opacity: 0.9;
        margin-bottom: 1.5rem;
        font-weight: 300;
        font-size: 1.1rem;
        max-width: 70%;
    }}
    
    .header-mask {{
        position: absolute;
        right: 0;
        top: 0;
        width: 40%;
        height: 100%;
        background: rgba(255,255,255,0.05);
        z-index: 1;
        clip-path: polygon(20% 0, 100% 0, 100% 100%, 0% 100%);
    }}
    
    .header-pattern {{
        position: absolute;
        right: 0;
        top: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
        opacity: 0.4;
        background-image: radial-gradient(rgba(255,255,255,0.15) 2px, transparent 2px);
        background-size: 20px 20px;
    }}
    
    /* Content Section */
    .content-section {{
        padding: 2rem 5%;
        max-width: 1400px;
        margin: 0 auto;
        position: relative;
    }}
    
    .wave-divider {{
        width: 100%;
        margin-bottom: -10px;
        display: block;
    }}
    
    /* Cards */
    .card {{
        background: white;
        border-radius: 0.7rem;
        box-shadow: 0 4px 25px rgba(0,0,0,0.05);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e9eef2;
        overflow: hidden;
        position: relative;
    }}
    
    .card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: #0068c9;
        border-radius: 4px 0 0 4px;
    }}
    
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #f0f4f8;
    }}
    
    .card-header h3 {{
        margin: 0;
        font-size: 1.3rem;
        color: #1e293b;
        font-weight: 600;
    }}
    
    .card-body {{
        position: relative;
    }}
    
    /* Badges */
    .badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.75rem;
        border-radius: 50px;
        font-weight: 500;
        font-size: 0.9rem;
        line-height: 1;
    }}
    
    .badge-primary {{
        background-color: #0068c9;
        color: white;
    }}
    
    .badge-outline {{
        background-color: rgba(0,104,201,0.1);
        color: #0068c9;
        border: 1px solid rgba(0,104,201,0.3);
    }}
    
    /* Form Elements */
    textarea {{
        width: 100%;
        min-height: 120px;
        padding: 0.75rem 1rem;
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        font-family: inherit;
        font-size: 1rem;
        transition: border-color 0.2s;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    
    textarea:focus {{
        border-color: #0068c9;
        outline: none;
        box-shadow: 0 0 0 3px rgba(0,104,201,0.1);
    }}
    
    /* Buttons */
    .btn-group {{
        display: flex;
        gap: 0.75rem;
        margin-top: 1rem;
    }}
    
    .btn {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.95rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        gap: 0.5rem;
    }}
    
    .btn-primary {{
        background: #0068c9;
        color: white;
        border: none;
    }}
    
    .btn-primary:hover {{
        background: #0055a5;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,104,201,0.2);
    }}
    
    .btn-outline {{
        background: white;
        color: #0068c9;
        border: 1px solid #0068c9;
    }}
    
    .btn-outline:hover {{
        background: #f0f7ff;
    }}
    
    .btn-secondary {{
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }}
    
    .btn-secondary:hover {{
        background: #e2e8f0;
    }}
    
    /* Icons */
    .icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.5rem;
        height: 1.5rem;
    }}
    
    /* Tables */
    .table-container {{
        overflow-x: auto;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }}
    
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    
    th {{
        text-align: left;
        padding: 1rem;
        font-size: 0.9rem;
        font-weight: 600;
        color: #475569;
        background: #f8fafc;
        border-bottom: 1px solid #e2e8f0;
    }}
    
    td {{
        padding: 1rem;
        font-size: 0.9rem;
        border-bottom: 1px solid #f1f5f9;
        color: #1e293b;
    }}
    
    tr:last-child td {{
        border-bottom: none;
    }}
    
    tr:hover td {{
        background: #f8fafc;
    }}
    
    /* Result Card */
    .result-card {{
        display: flex;
        align-items: center;
        padding: 1.5rem;
        background: #f0f7ff;
        border-radius: 0.7rem;
        border: 1px solid #cfe4ff;
        margin-bottom: 1.5rem;
    }}
    
    .result-icon {{
        width: 3.5rem;
        height: 3.5rem;
        border-radius: 50%;
        background: #0068c9;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
        margin-right: 1rem;
        flex-shrink: 0;
    }}
    
    .result-content {{
        flex: 1;
    }}
    
    .result-content h4 {{
        margin: 0;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        color: #0068c9;
    }}
    
    .result-content p {{
        margin: 0;
        opacity: 0.7;
    }}
    
    /* Summary Stats */
    .stats-row {{
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}
    
    .stat-card {{
        flex: 1;
        background: white;
        border-radius: 0.7rem;
        padding: 1.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border: 1px solid #e9eef2;
        position: relative;
        overflow: hidden;
    }}
    
    .stat-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: #0068c9;
    }}
    
    .stat-card h4 {{
        margin: 0;
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }}
    
    .stat-card h3 {{
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }}
    
    /* Footer */
    .footer {{
        background: #1e293b;
        color: #f1f5f9;
        padding: 2rem 5%;
        text-align: center;
        margin-top: 4rem;
    }}
    
    .footer p {{
        margin: 0.5rem 0;
        opacity: 0.8;
        font-size: 0.9rem;
    }}
    
    /* Sidebar Customization */
    .sidebar .sidebar-content {{
        background: #f8fafc;
        padding-top: 2rem !important;
    }}
    
    /* Example box */
    .example-box {{
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        transition: all 0.2s;
        cursor: pointer;
    }}
    
    .example-box:hover {{
        background: #f0f7ff;
        border-color: #cfe4ff;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,104,201,0.1);
    }}
    
    .example-box h4 {{
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        color: #1e293b;
    }}
    
    .example-box p {{
        margin: 0;
        font-size: 0.9rem;
        color: #64748b;
        line-height: 1.4;
    }}
    
    /* Tooltip */
    .tooltip {{
        position: relative;
        display: inline-block;
    }}
    
    .tooltip .tooltip-text {{
        visibility: hidden;
        width: 200px;
        background-color: #1e293b;
        color: #fff;
        text-align: center;
        border-radius: 0.4rem;
        padding: 0.5rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
    }}
    
    .tooltip:hover .tooltip-text {{
        visibility: visible;
        opacity: 1;
    }}
    
    /* Responsive adjustments */
    @media screen and (max-width: 768px) {{
        .header h1 {{
            font-size: 1.8rem;
        }}
        
        .header p {{
            max-width: 100%;
        }}
        
        .stats-row {{
            flex-direction: column;
        }}
        
        .btn-group {{
            flex-direction: column;
        }}
    }}

    /* Loading animation */
    @keyframes pulse {{
        0% {{ opacity: 0.6; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.6; }}
    }}
    
    .loading-pulse {{
        animation: pulse 1.5s infinite ease-in-out;
    }}

    /* Strealit specific overrides */
    div[data-testid="stVerticalBlock"] {{
        gap: 0 !important;
    }}
    
    div[data-testid="stMarkdownContainer"] p {{
        margin-bottom: 1rem;
    }}
    
    div[data-testid="stForm"] {{
        border: none;
        padding: 0;
    }}
    
    div[data-testid="stExpander"] {{
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        overflow: hidden;
    }}
    
    div[data-testid="stExpanderDetails"] {{
        border-top: 1px solid #e2e8f0;
        padding: 1rem;
    }}
    
    button[kind="primary"] {{
        background-color: #0068c9;
        border-radius: 0.5rem;
    }}
    
    div[data-baseweb="textarea"] {{
        border-radius: 0.5rem;
    }}
    
    div[data-testid="stSidebar"] {{
        background-color: #f8fafc;
    }}
    
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {{
        font-size: 1.3rem;
        color: #1e293b;
        font-weight: 600;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown(f"""
<div class="wave-divider">
    {wave_svg}
</div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header">
    <div class="header-content">
        <h1>Klasifikasi Berita Indonesia</h1>
    </div>  
</div>
""", unsafe_allow_html=True)

st.markdown("""
Aplikasi ini mengklasifikasikan artikel berita ke dalam 5 kategori, yaitu: 
- **Bola (olahraga)**
- **News (Berita Umum)**
- **Bisnis**
- **Teknologi**
- **Otomotif** 

Aplikasi ini juga akan menampilkan hasil prediksi kategori dari teks berita yang dimasukkan, serta nilai probabilitas dan grafik visualisasi untuk masing-masing kategori.""")

st.markdown("---")

# Load model yang sudah dilatih
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("./model_save/")
        model = AutoModelForSequenceClassification.from_pretrained("./model_save/")
        
        # Pindahkan model ke device yang tersedia
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Label mapping dengan icon dan warna
        label_mapping = {
            0: {'name': 'bisnis', 'icon': '💼', 'desc': 'Berita tentang ekonomi, perusahaan, investasi', 'color': '#FF9900'},
            1: {'name': 'bola', 'icon': '⚽', 'desc': 'Berita tentang sepakbola dan olahraga', 'color': '#28a745'},
            2: {'name': 'news', 'icon': '📰', 'desc': 'Berita umum dan politik', 'color': '#0068c9'},
            3: {'name': 'otomotif', 'icon': '🚗', 'desc': 'Berita tentang kendaraan dan teknologi otomotif', 'color': '#dc3545'},
            4: {'name': 'teknologi', 'icon': '💻', 'desc': 'Berita tentang teknologi dan gadget', 'color': '#6f42c1'}
        }
        
        return tokenizer, model, device, label_mapping
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

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
    predicted_category = label_mapping[prediction]
    
    # Hitung probabilitas untuk semua kelas
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    return {
        'category': predicted_category,
        'probabilities': {label_mapping[i]['name']: float(prob) for i, prob in enumerate(probabilities)}
    }

# Fungsi untuk menghasilkan statistik teks
def get_text_stats(text):
    if not text.strip():
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0
        }
    
    # Jumlah karakter (tanpa spasi)
    char_count = len(text.replace(" ", ""))
    
    # Jumlah kata
    words = text.split()
    word_count = len(words)
    
    # Jumlah kalimat (perkiraan sederhana)
    sentence_count = max(1, len(re.split(r'[.!?]+', text)) - 1)
    
    # Rata-rata panjang kata
    avg_word_length = char_count / max(1, word_count)
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 1)
    }

# Fungsi untuk membuat grafik dengan Plotly jika tersedia
def create_probability_chart(probabilities, selected_category, has_plotly=False):
    # Urutkan probabilitas
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    categories = [item[0] for item in sorted_probs]
    values = [item[1] for item in sorted_probs]
    
    if has_plotly:
        # Buat grafik bar dengan Plotly untuk visual yang lebih baik
        colors = ['#0068c9' if cat == selected_category['name'] else '#a6cdf7' for cat in categories]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f"{v:.1%}" for v in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Probabilitas per Kategori',
            xaxis_title='Kategori',
            yaxis_title='Probabilitas',
            yaxis=dict(
                tickformat='.0%',
                range=[0, 1]
            ),
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
        
        return fig
    else:
        # Gunakan matplotlib sebagai fallback
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            categories, 
            values,
            color=['#0068c9' if cat == selected_category['name'] else '#a6cdf7' for cat in categories]
        )
        
        # Tambahkan label
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.1%}',
                ha='center', 
                va='bottom'
            )
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probabilitas')
        ax.set_title('Probabilitas per Kategori')
        plt.tight_layout()
        
        return fig

# Set up sidebar
with st.sidebar:
    # Penjelasan Metode
    with st.expander("📘 Tentang Metode Klasifikasi"):
        st.markdown("""
        #### Metode Klasifikasi yang Digunakan
        
        Aplikasi ini menggunakan model **IndoBERT** yang telah dilatih khusus untuk klasifikasi teks berita berbahasa Indonesia. IndoBERT adalah model bahasa berbasis Transformer yang dilatih pada kumpulan data besar teks bahasa Indonesia.
        
        **Proses klasifikasi meliputi:**
        1. **Preprocessing** - Membersihkan teks dari karakter khusus dan mengubah ke lowercase
        2. **Tokenisasi** - Mengubah teks menjadi token yang dapat diproses model
        3. **Inferensi** - Model menganalisis token dan memprediksi kategori
        4. **Interpretasi** - Hasil berupa probabilitas untuk setiap kategori
        
        Model dilatih menggunakan dataset yang terdiri dari ribuan artikel berita Indonesia dari 5 kategori berbeda dengan akurasi sekitar 95% pada data pengujian.
        """)

    # Cara Menggunakan
    with st.expander("❓ Cara Menggunakan"):
        st.markdown("""
        #### Petunjuk Penggunaan
        
        1. **Masukkan Teks** - Ketik atau tempel artikel berita berbahasa Indonesia pada kolom input
        2. **Klasifikasikan** - Klik tombol "Klasifikasikan" untuk mendapatkan hasil
        3. **Analisis Hasil** - Lihat kategori yang diprediksi beserta probabilitasnya
        4. **Coba Contoh** - Klik pada contoh artikel di panel kanan untuk mencoba aplikasi
        
        **Tips:**
        - Gunakan artikel berita asli untuk hasil terbaik
        - Minimal masukkan satu paragraf teks untuk akurasi yang lebih baik
        - Aplikasi ini dirancang khusus untuk berita berbahasa Indonesia
        """)
    
    st.markdown("---")

    st.markdown("### Tentang Aplikasi")
    st.info("""    
    Aplikasi ini adalah bagian dari proyek Klasifikasi Berita Berbahasa Indonesia menggunakan model IndoBERT.
    
    **Kelas**: SINF6054 - Pemrosesan Bahasa Alami
    """)

# Main content section
st.markdown('<div class="content-section">', unsafe_allow_html=True)

# Card untuk input teks

# Membuat 2 kolom utama
main_col2,blank, main_col1 = st.columns([4, 0.18, 2])


# KOLOM 1: Contoh Artikel
with main_col1:
    st.markdown("### 🔍 Contoh Artikel")
    
    example_articles = {
        "Bisnis": "Jakarta, 13 Mei 2025 – Bank Indonesia menurunkan suku bunga acuan sebesar 25 basis poin menjadi 5,50% untuk mendorong pertumbuhan ekonomi. Keputusan ini diambil setelah melihat inflasi yang tetap terkendali dan stabilitas nilai tukar rupiah.",
        "Bola": "Timnas Indonesia akan bertanding melawan Malaysia dalam lanjutan kualifikasi Piala Dunia 2026. Pertandingan akan digelar di Stadion Gelora Bung Karno, Jakarta, pada Selasa (17/6/2025) pukul 19.30 WIB.",
        "News": "Presiden Republik Indonesia meresmikan pembangunan jalan tol baru yang menghubungkan Jakarta dan Bandung. Proyek ini diharapkan dapat mengurangi kemacetan dan mempercepat waktu tempuh antar kota.",
        "Otomotif": "Produsen mobil listrik Hyundai meluncurkan model terbaru IONIQ 7 di Indonesia. Mobil ini memiliki jangkauan hingga 600 km dengan sekali pengisian daya dan dibanderol mulai dari Rp 800 juta.",
        "Teknologi": "Apple mengumumkan peluncuran iPhone 16 dengan sejumlah fitur baru, termasuk kemampuan AI generatif yang canggih. Perangkat ini akan dijual mulai pekan depan dengan harga mulai dari Rp 15 juta."
    }

    # Styling contoh artikel agar terlihat lebih sebagai kartu
    st.markdown("""
    <style>
    .example-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
        cursor: pointer;
    }
    .example-card:hover {
        background-color: #f0f0f0;
        border-color: #aaa;
    }
    </style>
    """, unsafe_allow_html=True)

    # Contoh artikel dengan clickable box - layout yang lebih rapi
    selected_example = None
    for example_title, example_text in example_articles.items():
        col1, col2 = st.columns([2, 20])
        with col1:
            if st.button("↪", key=f"btn_{example_title}"):
                selected_example = example_text
        with col2:
            st.markdown(f"**{example_title}**")

        
        # Menambahkan sedikit ruang antar contoh
        st.markdown("<div style='margin-bottom: 0px;'></div>", unsafe_allow_html=True)

with blank:
    # Menambahkan sedikit ruang antar contoh
    st.markdown("")

# KOLOM 2: Text Area untuk Input
with main_col2:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <h3>📝 Masukkan Teks Berita</h3>
        </div>
        <div class="card-body">
    """, unsafe_allow_html=True)

    # Check if example was clicked and populate the text area
    placeholder_text = "Ketik atau tempel artikel berita berbahasa Indonesia di sini..."
    initial_value = selected_example or st.session_state.get("text_input_sample", "")

    # Text area for input
    text_input = st.text_area(
        "Teks Berita:",
        value=initial_value,
        height=300,  # Sedikit lebih tinggi karena sekarang di kolom yang lebih sempit
        placeholder=placeholder_text,
        label_visibility="collapsed"
    )

    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Button to classify
    col1, col2 = st.columns([4, 5])
    with col1:
        classify_btn = st.button("🔍 Klasifikasikan", type="primary", use_container_width=True)
    with col2:
        sample_btn = st.button("📋 Gunakan Contoh Acak", use_container_width=True)

# Menambahkan sedikit ruang antar contoh
st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

# Load model
tokenizer, model, device, label_mapping = load_model()

# Random examples
random_example_articles = [
    "Kementerian Perdagangan mencatat peningkatan ekspor kopi sebesar 12% pada kuartal pertama 2025, dengan pasar utama ke Amerika Serikat dan Jepang.",
    "Barcelona memastikan diri lolos ke final Liga Champions setelah menang agregat 4-3 melawan Bayern Munchen dalam laga dramatis di Camp Nou.",
    "BMKG memperingatkan potensi gempa susulan di wilayah Sumatera Barat pasca gempa berkekuatan 6,2 SR yang mengguncang dini hari tadi.",
    "Honda mengumumkan peluncuran model baru SUV hybrid untuk pasar Asia Tenggara dengan fitur hemat bahan bakar dan desain futuristik.",
    "Google meluncurkan fitur baru pada Android 15 yang memungkinkan kontrol aplikasi berbasis gerakan mata untuk pengguna dengan disabilitas.",
    "Presiden Joko Widodo meninjau pembangunan Ibu Kota Nusantara (IKN) di Kalimantan Timur pada Selasa pagi. Ia menyatakan progres pembangunan berjalan sesuai target. Kunjungan ini juga sekaligus untuk memastikan kesiapan infrastruktur dasar menjelang 2025.",
    "Harga beras medium di pasar tradisional naik sebesar Rp1.000 per kilogram dalam sepekan terakhir. Kenaikan ini diduga akibat pasokan dari petani menurun karena musim kemarau. Pemerintah sedang mempertimbangkan operasi pasar untuk menstabilkan harga.",
    "Kementerian Pendidikan resmi menghapus Ujian Nasional (UN) sebagai syarat kelulusan. Sebagai gantinya, evaluasi dilakukan berdasarkan asesmen kompetensi dan proyek belajar. Keputusan ini disambut positif oleh sebagian besar guru dan siswa.",
    "Indonesia meluncurkan satelit komunikasi SATRIA-1 dari Guyana Prancis pada Minggu malam. Satelit ini akan memperkuat jaringan internet di daerah tertinggal, terdepan, dan terluar. Proyek ini merupakan bagian dari upaya pemerataan akses digital nasional.",
    "Timnas Indonesia U-23 berhasil lolos ke final Piala Asia setelah mengalahkan Jepang 2-1. Gol kemenangan dicetak di menit-menit akhir oleh Marselino Ferdinan. Kemenangan ini menjadi sejarah baru bagi sepak bola Indonesia."
]

if sample_btn:
    random_text = np.random.choice(random_example_articles)
    st.session_state["text_input_sample"] = random_text
    st.experimental_rerun()

# Process and predict
if classify_btn or (text_input and text_input.strip() and text_input != placeholder_text):
    if not text_input or not text_input.strip():
        st.error("Harap masukkan teks berita terlebih dahulu.")
    elif not tokenizer or not model or not device or not label_mapping:
        st.error("Model tidak dapat dimuat dengan benar. Silakan periksa instalasi Anda.")
    else:
        with st.spinner("Menganalisis teks..."):
            # Simulate processing time to show the spinner
            time.sleep(0.5)
            
            # Get text statistics first
            text_stats = get_text_stats(text_input)
            
            # Prediksi
            try:
                prediction = predict_category(text_input, tokenizer, model, device, label_mapping)
                
                # Predicted category
                predicted_category = prediction['category']
                
                # Visualize the result
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid {predicted_category['color']}">
                    <div class="result-icon" style="background-color: {predicted_category['color']}">
                        {predicted_category['icon']}
                    </div>
                    <div class="result-content">
                        <h4>Kategori Terdeteksi: {predicted_category['name'].upper()}</h4>
                        <p>{predicted_category['desc']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""---""")
                
                # Show statistics
                st.markdown("""
                <div class="card">
                    <div class="card-header">
                        <h3>📊 Hasil Analisis</h3>
                    </div>
                    <div class="card-body">
                """, unsafe_allow_html=True)
                
                # Stats row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>Jumlah Karakter</h4>
                        <h3>{text_stats['char_count']:,}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>Jumlah Kata</h4>
                        <h3>{text_stats['word_count']:,}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>Jumlah Kalimat</h4>
                        <h3>{text_stats['sentence_count']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>Rata-rata Pjg Kata</h4>
                        <h3>{text_stats['avg_word_length']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""---""")

                # Probability Chart
                st.subheader("Probabilitas per Kategori")
                
                if HAS_PLOTLY:
                    chart = create_probability_chart(prediction['probabilities'], predicted_category, True)
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    chart = create_probability_chart(prediction['probabilities'], predicted_category, False)
                    st.pyplot(chart)
                
                # Show probability table
                st.markdown("<h4>Detail Probabilitas</h4>", unsafe_allow_html=True)
                
                # Create table with color-coded bars
                probs_df = pd.DataFrame({
                    'Kategori': [f"{label_mapping[i]['icon']} {label_mapping[i]['name'].capitalize()}" for i in range(len(label_mapping))],
                    'Probabilitas': [prediction['probabilities'][label_mapping[i]['name']] for i in range(len(label_mapping))]
                })
                
                # Sort by probability
                probs_df = probs_df.sort_values('Probabilitas', ascending=False)
                
                # Format probability as percentage
                probs_df['Persentase'] = probs_df['Probabilitas'].apply(lambda x: f"{x:.2%}")
                
                # Show just kategori, percentage and visual bar
                styled_df = probs_df[['Kategori', 'Persentase']]
                st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                st.markdown("""---""")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat mengklasifikasikan teks: {str(e)}")
                st.exception(e)

st.markdown('</div>', unsafe_allow_html=True)