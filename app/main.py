import json
import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="centered"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── App shell ── */
.stApp {
    background: #f0f4ed;
    font-family: 'DM Sans', sans-serif;
    color: #1c2b1c;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 900px;
    padding: 0 2rem 5rem;
}

/* ── Navbar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0 1.2rem;
    border-bottom: 1px solid #cfdbc8;
    margin-bottom: 3rem;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #1c2b1c;
    letter-spacing: -0.01em;
}
.navbar-tag {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #fff;
    background: #3a7d44;
    padding: 3px 10px;
    border-radius: 20px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 1rem 0 3rem;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3a7d44;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #1c2b1c;
    line-height: 1.15;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}
.hero h1 span {
    color: #3a7d44;
}
.hero-sub {
    color: #4a6741;
    font-size: 1rem;
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto 2rem;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    padding: 1.2rem 2rem;
    background: white;
    border-radius: 14px;
    border: 1px solid #cfdbc8;
    width: fit-content;
    margin: 0 auto;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.stat-item { text-align: center; }
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #1c2b1c;
    display: block;
}
.stat-label {
    font-size: 0.72rem;
    color: #7a9e7a;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
}
.section-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #3a7d44;
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a6741;
}

/* ── Upload zone ── */
.upload-zone {
    background: white;
    border: 2px dashed #b8d4b0;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    transition: border-color 0.2s, background 0.2s;
    cursor: pointer;
}
.upload-zone:hover {
    border-color: #3a7d44;
    background: #f7fbf5;
}
.upload-icon { font-size: 2.2rem; margin-bottom: 0.6rem; }
.upload-title {
    font-weight: 600;
    color: #1c2b1c;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}
.upload-hint { color: #7a9e7a; font-size: 0.82rem; }

/* ── Hide default Streamlit uploader UI ── */
[data-testid="stFileUploaderDropzone"] { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }

/* ── Browse button (Streamlit secondary) ── */
[data-testid="baseButton-secondary"] {
    background: white !important;
    color: #3a7d44 !important;
    border: 1.5px solid #3a7d44 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1.4rem !important;
    margin-top: 0.8rem !important;
    transition: background 0.15s !important;
}
[data-testid="baseButton-secondary"]:hover {
    background: #f0f8f0 !important;
}

/* ── Analysis card ── */
.analysis-card {
    background: white;
    border-radius: 16px;
    border: 1px solid #cfdbc8;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(30,60,30,0.06);
    margin-top: 1rem;
}
.analysis-card-header {
    background: #1c2b1c;
    color: #cfdbc8;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.7rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.analysis-card-body { padding: 1.4rem; }

/* ── Image frame ── */
.img-frame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #dde8d8;
    background: #f7fbf5;
}

/* ── Predict button ── */
.stButton > button {
    background: #3a7d44 !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.02em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
    box-shadow: 0 2px 8px rgba(58,125,68,0.3) !important;
}
.stButton > button:hover {
    background: #2d6235 !important;
    box-shadow: 0 4px 16px rgba(58,125,68,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result box ── */
.result-box {
    background: #f7fbf5;
    border: 1px solid #cfdbc8;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-top: 1rem;
}
.result-row {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.result-field { flex: 1; min-width: 120px; }
.result-field-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7a9e7a;
    margin-bottom: 0.25rem;
}
.result-field-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #1c2b1c;
    line-height: 1.3;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    margin-top: 0.2rem;
}
.status-healthy {
    background: #e2f5e6;
    color: #1a5c2a;
    border: 1px solid #b8d4b0;
}
.status-disease {
    background: #fdf0e8;
    color: #8b3a1a;
    border: 1px solid #e8c4a0;
}

/* ── Info tip ── */
.info-tip {
    background: #eef5ec;
    border-left: 3px solid #3a7d44;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #3a5c3a;
    line-height: 1.5;
    margin-bottom: 1.2rem;
}

/* ── Footer ── */
.footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid #cfdbc8;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.footer-left {
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    color: #1c2b1c;
}
.footer-right {
    font-size: 0.78rem;
    color: #7a9e7a;
}
</style>
""", unsafe_allow_html=True)

# ─── Load model ────────────────────────────────────────────────────────────────
working_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{working_dir}/models/plant_disease_production_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# ─── Helpers ───────────────────────────────────────────────────────────────────
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    raw_name = class_indices[str(predicted_class_index)]
    parts = raw_name.split("___")
    if len(parts) == 2:
        plant = parts[0].replace("_", " ").title()
        disease = parts[1].replace("_", " ").title()
        return plant, disease
    else:
        return "Unknown", raw_name.replace("_", " ").title()

# ─── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">🌿 PlantGuard</div>
    <div class="navbar-tag">AI · Deep Learning</div>
</div>
""", unsafe_allow_html=True)

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Plant Disease Detection System</div>
    <h1>Diagnose your plant<br><span>in seconds.</span></h1>
    <p class="hero-sub">
        Upload a photo of any leaf and our deep learning model will identify
        the disease with high accuracy — no lab required.
    </p>
    <div class="hero-stats">
        <div class="stat-item">
            <span class="stat-num">38</span>
            <span class="stat-label">Disease Classes</span>
        </div>
        <div class="stat-item">
        <span class="stat-num">10,849</span>
        <span class="stat-label">Training images</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">91%</span>
            <span class="stat-label">Accuracy</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">&lt;2s</span>
            <span class="stat-label">Inference Time</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Upload ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-num">1</div>
    <div class="section-title">Upload Leaf Image</div>
</div>
<div class="upload-zone">
    <div class="upload-icon">🍃</div>
    <div class="upload-title">Drop your leaf image here</div>
    <div class="upload-hint">JPEG, JPG or PNG &nbsp;·&nbsp; Max 200MB</div>
</div>
""", unsafe_allow_html=True)

uploaded_image = st.file_uploader(
    "Upload leaf image",
    type=["jpeg", "jpg", "png"],
    label_visibility="collapsed"
)

# ─── Analysis ──────────────────────────────────────────────────────────────────
if uploaded_image is not None:
    image = Image.open(uploaded_image)

    st.markdown("""
    <div class="section-header" style="margin-top:2rem">
        <div class="section-num">2</div>
        <div class="section-title">Review & Predict</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="analysis-card"><div class="analysis-card-header">🔬 Analysis Panel</div><div class="analysis-card-body">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(image.resize((320, 320)), use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-tip">
            For best results, use a clear photo of a single leaf against a plain background in good lighting.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔬 Analyse Leaf"):
            with st.spinner("Running model inference..."):
                plant_name, disease_name = predict_image_class(model, uploaded_image, class_indices)

            is_healthy = "healthy" in disease_name.lower()
            pill_class = "status-healthy" if is_healthy else "status-disease"
            pill_icon = "✅" if is_healthy else "⚠️"
            pill_text = "Healthy" if is_healthy else "Disease Detected"
            advice = "No disease found. Keep up the good care!" if is_healthy else "Consult an agronomist for treatment advice."

            st.markdown(f"""
            <div class="result-box">
                <div class="result-row">
                    <div class="result-field">
                        <div class="result-field-label">Plant</div>
                        <div class="result-field-value">{plant_name}</div>
                    </div>
                    <div class="result-field">
                        <div class="result-field-label">Condition</div>
                        <div class="result-field-value">{disease_name}</div>
                    </div>
                </div>
                <div class="status-pill {pill_class}">{pill_icon} {pill_text}</div>
                <div style="margin-top:0.7rem; font-size:0.8rem; color:#4a6741;">{advice}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-left">🌿 PlantGuard AI</div>
    <div class="footer-right">Arnav Gupta &nbsp;·&nbsp; EEE, MAIT &nbsp;·&nbsp; Built with TensorFlow & Streamlit</div>
</div>
""", unsafe_allow_html=True)