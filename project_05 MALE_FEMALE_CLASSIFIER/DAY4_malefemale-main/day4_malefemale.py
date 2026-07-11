# -*- coding: utf-8 -*-
"""DAY4_malefemale.ipynb"""

import streamlit as st
import numpy as np
from PIL import Image
import joblib

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Female vs Male Classifier",
    page_icon="🚹🚺",
    layout="centered"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(135deg,#fdf2f8,#eff6ff,#f0fdfa);
}

/* Main Container */
.main .block-container{
    max-width:900px;
    padding-top:1rem;
    padding-bottom:2rem;
}

/* Hide Streamlit Menu & Footer */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Hero Banner */
.hero{
    background: linear-gradient(120deg,#7c3aed,#2563eb 60%,#0891b2);
    padding:35px 20px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 8px 25px rgba(37,99,235,0.35);
    margin-bottom:25px;
}
.hero h1{
    color:white;
    font-size:2.3rem;
    font-weight:800;
    margin-bottom:8px;
}
.hero p{
    color:#e0e7ff;
    font-size:17px;
    margin:0;
}

/* Section Heading */
.section-heading{
    display:flex;
    align-items:center;
    gap:10px;
    margin-top:10px;
    margin-bottom:12px;
}
.section-heading h2{
    color:#1e3a8a;
    font-weight:700;
    margin:0;
}
.section-line{
    height:4px;
    width:60px;
    background:linear-gradient(90deg,#2563eb,#06b6d4);
    border-radius:5px;
    margin-bottom:20px;
}

/* File Uploader */
[data-testid="stFileUploader"]{
    background:white;
    border:2px dashed #7c3aed;
    border-radius:15px;
    padding:15px;
}

/* Uploaded Image */
img{
    border-radius:15px;
    border:3px solid #2563eb;
}

/* Success Message */
[data-testid="stAlert"]{
    border-radius:12px;
}

/* Cards */
.card{
    background:white;
    padding:22px;
    border-radius:16px;
    text-align:center;
    box-shadow:0px 6px 20px rgba(0,0,0,0.12);
    transition:transform 0.2s ease;
}
.card:hover{
    transform:translateY(-4px);
}
.card-female{
    border-top:5px solid #ec4899;
}
.card-male{
    border-top:5px solid #2563eb;
}
.card h3{
    margin-bottom:5px;
    color:#334155;
}
.card h1{
    margin:0;
    font-size:2.2rem;
}
.card-female h1{
    color:#db2777;
}
.card-male h1{
    color:#1d4ed8;
}

/* Result Banner */
.result-banner{
    text-align:center;
    padding:18px;
    border-radius:15px;
    font-size:24px;
    font-weight:800;
    color:white;
    margin-bottom:15px;
    box-shadow:0px 6px 18px rgba(0,0,0,0.15);
}
.result-female{
    background:linear-gradient(120deg,#ec4899,#f472b6);
}
.result-male{
    background:linear-gradient(120deg,#2563eb,#0891b2);
}

/* Repository Box */
.repo{
    background:#ffffff;
    border-left:6px solid #7c3aed;
    padding:20px;
    border-radius:14px;
    margin-top:30px;
    box-shadow:0px 3px 12px rgba(0,0,0,0.1);
}
.repo h3{
    color:#1e293b;
    margin-top:0;
}
.repo a{
    color:#2563eb;
    font-weight:600;
    text-decoration:none;
}
.repo a:hover{
    text-decoration:underline;
}

/* Footer */
.footer{
    text-align:center;
    color:gray;
    font-size:14px;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("female_male.pkl")

IMG_SIZE = 64

# -------------------------
# Hero / Title
# -------------------------
st.markdown("""
<div class="hero">
    <h1>🚹🚺 Female vs Male Image Classifier</h1>
    <p>Upload a photo and let our Machine Learning model predict whether the person is <b>Male</b> or <b>Female</b> — instantly.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Upload Section
# -------------------------
st.markdown("""
<div class="section-heading"><h2>📂 Upload Your Image</h2></div>
<div class="section-line"></div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=320
    )

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    resized = np.array(resized)
    resized = resized.flatten()

    prediction = model.predict([resized])[0]
    probability = model.predict_proba([resized])[0]

    st.markdown("""
    <div class="section-heading"><h2>🔮 Prediction Result</h2></div>
    <div class="section-line"></div>
    """, unsafe_allow_html=True)

    if prediction == 0:
        st.markdown("""<div class="result-banner result-female">🚺 Prediction: Female</div>""", unsafe_allow_html=True)
        st.balloons()
        st.toast("🎉 Prediction Completed: Female", icon="🚺")
    else:
        st.markdown("""<div class="result-banner result-male">🚹 Prediction: Male</div>""", unsafe_allow_html=True)
        st.balloons()
        st.toast("🎉 Prediction Completed: Male", icon="🚹")

    st.progress(float(max(probability)))

    st.markdown("""
    <div class="section-heading"><h2>📊 Prediction Confidence</h2></div>
    <div class="section-line"></div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card card-female">
        <h3>🚺 Female</h3>
        <h1>{probability[0]*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card card-male">
        <h3>🚹 Male</h3>
        <h1>{probability[1]*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# GitHub Repository
# -------------------------
st.markdown("---")

st.markdown("""
<div class="repo">

### 💻 Project Source Code

Want to explore the implementation, training process, and project files?

👉 <b>GitHub Repository:</b><br>

<a href="https://github.com/Anjaliy6126/DAY4_malefemale" target="_blank">
https://github.com/Anjaliy6126/DAY4_malefemale
</a>

</div>
""", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class="footer">
<hr>
Made with ❤️ using <b>Python</b>, <b>Scikit-Learn</b> and <b>Streamlit</b><br>
© 2026 Female vs Male Image Classifier
</div>
""", unsafe_allow_html=True)
