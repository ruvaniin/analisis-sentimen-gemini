import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Analisis Sentimen Gemini",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with improved visibility
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        color: #333;
        backdrop-filter: blur(10px);
    }
    
    /* Upload section */
    .upload-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 1rem 0;
        color: #333;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .upload-section h4 {
        color: #333 !important;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .upload-section p {
        color: #666 !important;
        margin: 0;
    }
    
    /* Content cards */
    .content-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: #333;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Fix all text colors */
    .stMarkdown, .stText, p, span, div {
        color: light-green !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: 600;
    }
    
    /* Content inside cards should be dark */
    .content-card h1, .content-card h2, .content-card h3, 
    .content-card h4, .content-card h5, .content-card h6,
    .content-card p, .content-card span, .content-card div,
    .content-card li {
        color: #333 !important;
    }
    
    .upload-section * {
        color: #333 !important;
    }
    
    /* List items */
    ul li, ol li {
        color: white !important;
    }
    
    .content-card ul li, .content-card ol li {
        color: #333 !important;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="metric-container"] * {
        color: #333 !important;
    }
    
    /* Alert styling */
    .stAlert {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #333 !important;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .stAlert * {
        color: #333 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stInfo, .stWarning {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #333 !important;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .stSuccess *, .stError *, .stInfo *, .stWarning * {
        color: #333 !important;
    }
    
    /* Code blocks */
    code {
        background: rgba(0, 0, 0, 0.1) !important;
        color: #e91e63 !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #333 !important;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #333 !important;
        border-radius: 0 0 10px 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #333 !important;
        font-weight: 600;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load model dan vectorizer berdasarkan metode
@st.cache_resource
def load_model_and_vectorizer(method):
    try:
        if method == "Lexicon Based":
            model = joblib.load("best_svm_model_lexicon.pkl")
            vectorizer = joblib.load("vectorizer_lexicon.pkl")
        else:
            model = joblib.load("best_svm_model_rating.pkl")
            vectorizer = joblib.load("vectorizer_rating.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("❗ Model files not found. Please check if the model files are in the correct directory.")
        return None, None

# Preprocessing fungsi
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Function to generate word cloud
def generate_wordcloud(text_data, sentiment):
    text = ' '.join(text_data)
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        return wordcloud
    return None

# Function to create sentiment statistics
def create_sentiment_stats(df, sentiment_col):
    stats = df[sentiment_col].value_counts()
    total = len(df)
    
    stats_dict = {}
    for sentiment in stats.index:
        count = stats[sentiment]
        percentage = (count / total) * 100
        stats_dict[sentiment] = {'count': count, 'percentage': percentage}
    
    return stats_dict

# Header
st.markdown("""
<div class="main-header">
    <h1>🔮Analisis Sentimen Ulasan Pengguna Aplikasi Google Gemini</h1>
</div>
""", unsafe_allow_html=True)

# Tentang
st.markdown("## 🧾 Tentang Platform")
st.markdown("""
<div class="content-card">
    <p style='color: #333; margin: 0; font-size: 1.1rem;'>
        Alat ini dirancang untuk memproses dataset ulasan pengguna dan menentukan apakah ulasan dan rating tersebut bersentimen positif atau negatif.
        Analisis sentimen ulasan pengguna aplikasi Google Gemini pada Google Play Store menggunakan algoritma Support Vector Machine (SVM). Hasil ini dapat memberikan gambaran mengenai sentimen pengguna terhadap aplikasi Google Gemini.
    </p>
</div>
""", unsafe_allow_html=True)

# Fitur
st.markdown("## 🔧 Fitur dan Penggunaan")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="content-card">
        <h4 style='color: #333; margin-top: 0; display: flex; align-items: center;'>
            <span style='margin-right: 10px;'>🚀</span> Fitur
        </h4>
        <ul style='color: #333; padding-left: 20px; margin: 0;'>
            <li><strong>Pelabelan Sentimen</strong> positif dan negatif</li>
            <li><strong>Real-time</strong> pre-processing</li>
            <li><strong>Ringkasan</strong> dataset</li>
            <li><strong>Visualisasi</strong> Statistik dan WordCloud</li>
            <li><strong>Pratinjau</strong> ulasan sentimen</li>
            <li><strong>Ekspor</strong> hasil analisis ke file pilihan (csv dan json)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="content-card">
        <h4 style='color: #333; margin-top: 0; display: flex; align-items: center;'>
            <span style='margin-right: 10px;'>📝</span> Cara Penggunaannya
        </h4>
        <ul style='color: #333; padding-left: 20px; margin: 0;'>
            <li><strong>Pilih</strong> Metode Analisis, Opsi Analisis dan Opsi Ekspor</li>
            <li><strong>Upload</strong> file dataset dalam format CSV yang memiliki kolom: 'Review Text' dan 'Rating'</li>
            <li><strong>Lihat</strong> hasil sentimen dan visualisasi</li>
            <li><strong>Ekspor</strong> dataset ke file pilihan (csv dan json)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi")
    
    # Method selection
    metode = st.radio(
        "🔍 Metode Analisis:",
        ["Lexicon Based", "Rating Based"],
        help="Pilih metode analisis sentimen"
    )
    
    st.markdown("---")
    
    # Analysis options
    st.markdown("### 📈 Opsi Analisis")
    show_wordcloud = st.checkbox("🌟 Hasilkan WordCloud", value=True)
    show_detailed_stats = st.checkbox("📊 Tampilkan Statistik Terperinci", value=True)
    show_sample_reviews = st.checkbox("💬 Tampilkan Contoh Ulasan", value=True)
    
    st.markdown("---")
    
    # Export options
    st.markdown("### 📥 Opsi Ekspor")
    export_format = st.selectbox("Format Ekspor:", ["CSV", "JSON"])
    
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; backdrop-filter: blur(10px);'>
        <h4 style='color: #856404; margin: 0 0 10px 0;'>💡 Tips</h4>
        <p style='color: #FFFFFF; margin: 0; font-size: 0.9rem;'>Pilih konfigurasi, opsi analisis dan opsi ekspor terlebih dahulu untuk mendapatkan hasil terbaik.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
        <h2 style='color: white; margin: 0; font-size: 2rem;'>🚀 Mulai Sentimen Sekarang</h2>
        <p style='color: #e8e9ff; margin: 10px 0 0 0; font-size: 1.1rem;'>Ikuti cara penggunaan diatas untuk menganalisis ulasan pengguna</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### 📁 Upload Dataset")
# Main content
col1, col2 = st.columns([2, 1])

with col1:
    
    # File upload section
    st.markdown("""
    <div class="upload-section">
        <h4>📤 Drop file CSV disini</h4>
        <p>Pastikan file Anda berisi kolom 'Review Text' dan 'Rating'</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "",
        type=["csv"],
        help="Unggah file CSV yang berisi 'Review Text' dan 'Rating'"
    )

with col2:
    st.markdown("""
    <div style='background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); color: #333;'>
        <strong>Kolom yang Diperlukan:</strong><br>
        <ul style='margin-top: 5px;'>
        <li>
            <span style="background-color: #f5f5f5; padding: 2px 6px; border-radius: 4px; color: #d63384; font-family: monospace;">Review Text</span>
            <span style="color: #000000;">: Ulasan pengguna</span>
        </li>
        <li>
            <span style="background-color: #f5f5f5; padding: 2px 6px; border-radius: 4px; color: #d63384; font-family: monospace;">Rating</span>
            <span style="color: #000000;">: Peringkat Numerik</span>
        </li>
        </ul>
        <div style='margin-top: 10px; background-color: #1e1e1e; color: #f8f8f2; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 14px;'>
            • CSV files only<br>
            • UTF-8 encoding recommended
        </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        # Load and validate data
        with st.spinner("🔄 Loading and validasi data..."):
            df = pd.read_csv(uploaded_file)
            
        # Data validation
        if 'Review Text' not in df.columns:
            st.error("❗ Kolom 'Review Text' tidak ditemukan dalam file.")
            st.stop()
            
        # Display data info
        st.success(f"✅ Berhasil memuat {len(df)} ulasan!")
        
        # Data overview
        st.markdown("## 📊 Ringkasan Data")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔍Total Ulasan", len(df))
        with col2:
            st.metric("🎯Ulasan Unik", df['Review Text'].nunique())
        with col3:
            avg_length = df['Review Text'].str.len().mean()
            st.metric("📝Rata² Panjang Ulasan", f"{avg_length:.0f} chars")
        with col4:
            if 'Rating' in df.columns:
                avg_rating = df['Rating'].mean()
                st.metric("⭐Rata² Rating", f"{avg_rating:.1f}")
        
        # Sample data
        with st.expander("👀 Preview Data Ulasan", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Data preprocessing
        st.markdown("## 🔄 Pre-Processing")
        
        with st.spinner("Processing teks data..."):
            df['clean_review'] = df['Review Text'].astype(str).apply(clean_text)
            
            # Remove empty reviews
            df = df[df['clean_review'].str.len() > 0]
            
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
        
        st.success("✅ Text pre-processing selesai!")
        
        # Load model and make predictions
        st.markdown("## 🤖 Analisis Sentimen")
        
        model, vectorizer = load_model_and_vectorizer(metode)
        
        if model is not None and vectorizer is not None:
            with st.spinner(f"Menganalisis sentimen menggunakan {metode}..."):
                X = vectorizer.transform(df['clean_review'])
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)
                
                df[f'{metode}_Sentiment'] = predictions
                df[f'{metode}_Confidence'] = probabilities.max(axis=1)
            
            st.success(f"✅ Analisis sentimen diselesaikan menggunakan {metode}!")
            
            # Results visualization
            st.markdown("## 📈 Hasil Analisis")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Ringkasan", "📈 Visualisasi", "💬 Contoh Ulasan", "📋 Statistik Tambahan"])
            
            with tab1:
                # Sentiment distribution
                sentiment_stats = create_sentiment_stats(df, f'{metode}_Sentiment')
                
                cols = st.columns(len(sentiment_stats))
                for i, (sentiment, stats) in enumerate(sentiment_stats.items()):
                    with cols[i]:
                        st.metric(
                            f"{sentiment} Reviews",
                            stats['count'],
                            f"{stats['percentage']:.1f}%"
                        )
                
                # Confidence distribution
                st.markdown("### 🎯 Prediksi Confidence (Keyakinan Model SVM)")

                # Kalikan confidence agar dalam bentuk persen
                df[f'{metode}_Confidence'] = df[f'{metode}_Confidence'] * 100

                # Rata-rata confidence
                avg_confidence = df[f'{metode}_Confidence'].mean()
                st.metric("Rata² Confidence", f"{avg_confidence:.2f}%")

                # Confidence histogram
                fig_conf = px.histogram(
                    df, 
                    x=f'{metode}_Confidence',
                    title="Distribusi Prediksi Confidence Model SVM (%)",
                    nbins=20,
                    labels={f'{metode}_Confidence': 'Confidence (%)'}
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Interactive pie chart
                    st.markdown("#### 📝 Diagram Pie")
                    sentiment_counts = df[f'{metode}_Sentiment'].value_counts()
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Diagram Pie Positif vs Negatif",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart
                    st.markdown("#### 📝 Diagram Batang")
                    fig_bar = px.bar(
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        title="Diagram Batang Positif vs Negatif",
                        color=sentiment_counts.values,
                        color_continuous_scale="viridis"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Word clouds
                if show_wordcloud:
                    st.markdown("### ☁️ WordClouds")
                    sentiments = df[f'{metode}_Sentiment'].unique()
                    
                    cols = st.columns(len(sentiments))
                    for i, sentiment in enumerate(sentiments):
                        sentiment_reviews = df[df[f'{metode}_Sentiment'] == sentiment]['clean_review']
                        wordcloud = generate_wordcloud(sentiment_reviews, sentiment)
                        
                        if wordcloud:
                            with cols[i]:
                                st.markdown(f"**{sentiment}**")
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
            
            with tab3:
                if show_sample_reviews:
                    st.markdown("### 💬 Contoh Ulasan Sentimen")
                    
                    selected_sentiment = st.selectbox(
                        "Pilih Sentimen:",
                        df[f'{metode}_Sentiment'].unique()
                    )
                    
                    sample_reviews = df[df[f'{metode}_Sentiment'] == selected_sentiment].sample(
                        min(5, len(df[df[f'{metode}_Sentiment'] == selected_sentiment]))
                    )
                    
                    for idx, row in sample_reviews.iterrows():
                        with st.expander(f"Review {idx + 1} (Confidence: {row[f'{metode}_Confidence']:.2f})"):
                            st.write(row['Review Text'])
            
            with tab4:
                if show_detailed_stats:
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Rating distribution bar chart
                        st.markdown("#### 📊 Distribusi Rating Pengguna")
                        if 'Rating' in df.columns:
                            rating_counts = df['Rating'].value_counts().sort_index()
                            fig_rating = px.bar(
                                x=rating_counts.index,
                                y=rating_counts.values,
                                title="Jumlah Data per Skor Rating",
                                labels={'x': 'Rating', 'y': 'Jumlah Ulasan'},
                                color=rating_counts.values,
                                color_continuous_scale="Blues"
                            )
                            fig_rating.update_layout(
                                xaxis_title="Skor Rating",
                                yaxis_title="Jumlah Ulasan",
                                showlegend=False
                            )
                            st.plotly_chart(fig_rating, use_container_width=True)
                        else:
                            st.info("Kolom 'Rating' tidak tersedia untuk analisis distribusi rating.")
                    
                    with col2:
                        # Top 10 most frequent words horizontal bar chart
                        st.markdown("#### 📝 10 Kata Terpopuler")
                        from collections import Counter
                        import itertools
                        
                        # Get all words from processed reviews
                        all_words = ' '.join(df['clean_review']).split()
                        # Remove very short words (less than 3 characters)
                        all_words = [word for word in all_words if len(word) >= 3]
                        
                        # Count word frequencies
                        word_counts = Counter(all_words)
                        top_10_words = dict(word_counts.most_common(10))
                        
                        if top_10_words:
                            fig_words = px.bar(
                                x=list(top_10_words.values()),
                                y=list(top_10_words.keys()),
                                orientation='h',
                                title="10 Kata Paling Sering Muncul",
                                labels={'x': 'Frekuensi', 'y': 'Kata'},
                                color=list(top_10_words.values()),
                                color_continuous_scale="Viridis"
                            )
                            fig_words.update_layout(
                                xaxis_title="Frekuensi Kemunculan",
                                yaxis_title="Kata",
                                showlegend=False,
                                height=400
                            )
                            st.plotly_chart(fig_words, use_container_width=True)
                        else:
                            st.info("Tidak dapat menganalisis frekuensi kata dari data yang diproses.")
            
            # Export results
            st.markdown("## 📥 Ekspor Hasil")
            
            # Prepare export data
            export_df = df[['Review Text', f'{metode}_Sentiment', f'{metode}_Confidence']].copy()
            
            if export_format == "CSV":
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil sebagai CSV",
                    data=csv_data,
                    file_name=f'sentiment_analysis_{metode.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
            elif export_format == "JSON":
                json_data = export_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download Hasil sebagai JSON",
                    data=json_data,
                    file_name=f'sentiment_analysis_{metode.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                    mime='application/json'
                )
        
        else:
            st.error("❗ Could not load the model. Please check if model files exist.")
    
    except Exception as e:
        st.error(f"❗ An error occurred: {str(e)}")
        st.info("Please check your file format and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div class="content-card" style='text-align: center; margin-top: 2rem;'>
    <p style='color: #333; font-size: 1.1rem; font-weight: 600; margin: 0;'>📊 Analisis Sentimen Ulasan Aplikasi Google Gemini</p>
    <p style='color: #666; margin: 5px 0 0 0;'>💡 Dibuat dengan Streamlit oleh Ruvani Nuzulha</p>
    <p style='color: #666; margin: 5px 0 0 0;'>📚 Proyek Skripsi Sistem Informasi 2025</p>
</div>
""", unsafe_allow_html=True)
