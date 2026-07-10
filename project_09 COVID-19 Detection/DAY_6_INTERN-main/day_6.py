import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩺",
    layout="centered"
)

# -------------------- CSS --------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp{
    background: linear-gradient(135deg, #0F172A 0%, #111827 50%, #0B1120 100%);
    background-attachment: fixed;
}

.main{
    padding-top: 10px;
}

/* ---------- Header Card ---------- */
.header-card{
    background: linear-gradient(135deg, #1E3A8A, #1D4ED8, #2563EB);
    padding: 30px 25px;
    border-radius: 20px;
    box-shadow: 0 8px 28px rgba(37,99,235,0.35);
    border: 1px solid #3B82F6;
    margin-bottom: 25px;
    text-align: center;
}

.logo-icon{
    font-size: 70px;
    margin-bottom: 10px;
    filter: drop-shadow(0 4px 10px rgba(0,0,0,0.4));
}

.header-card h1{
    color: #F8FAFC !important;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.header-card p{
    color: #DBEAFE !important;
    font-size: 17px !important;
    margin-top: 10px;
    font-weight: 400;
}

/* ---------- Force all headings and text light ---------- */
h1, h2, h3, h4, h5, h6{
    color:#F1F5F9 !important;
    font-weight: 700 !important;
}

h3{
    text-align: center;
}

p, li, span, label, div{
    color:#E2E8F0;
}

/* Streamlit specific text elements */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span{
    color:#E2E8F0 !important;
    font-size:17px;
}

/* File uploader label + instructions */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span{
    color:#F1F5F9 !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] small{
    color:#CBD5E1 !important;
}

/* ---------- Upload box ---------- */
[data-testid="stFileUploader"]{
    background: #1E293B;
    padding: 25px;
    border-radius: 18px;
    border: 2px dashed #3B82F6;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover{
    border-color: #60A5FA;
    box-shadow: 0 8px 24px rgba(59,130,246,0.3);
}

/* Browse files button inside uploader */
[data-testid="stFileUploader"] button{
    background:#334155 !important;
    color:#F1F5F9 !important;
    border: 1px solid #475569 !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] button:hover{
    background:#475569 !important;
}

/* Uploaded file name text */
[data-testid="stFileUploaderFileName"]{
    color:#F1F5F9 !important;
    font-weight: 600 !important;
}

/* ---------- Image preview card ---------- */
[data-testid="stImage"]{
    display: flex;
    justify-content: center;
    margin: 20px 0;
}

[data-testid="stImage"] img{
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    border: 4px solid #1E293B;
}

/* ---------- Button ---------- */
.stButton>button{
    width:100%;
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color:#F8FAFC !important;
    font-size:20px;
    font-weight:700;
    border-radius:14px;
    padding:14px;
    border:none;
    box-shadow: 0 6px 18px rgba(37,99,235,0.5);
    transition: all 0.25s ease;
    letter-spacing: 0.5px;
}

.stButton>button:hover{
    background: linear-gradient(135deg, #3B82F6, #2563EB);
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(59,130,246,0.6);
    color:#F8FAFC !important;
}

.stButton>button:active{
    transform: translateY(0px);
}

/* ---------- Result boxes ---------- */
.success-box{
    background: linear-gradient(135deg, #14532D, #166534);
    padding:22px;
    border-radius:16px;
    font-size:24px;
    font-weight:800;
    color:#DCFCE7 !important;
    text-align:center;
    box-shadow: 0 8px 24px rgba(20,83,45,0.4);
    border: 2px solid #22C55E;
    margin-top: 15px;
    animation: fadeIn 0.5s ease;
}

.error-box{
    background: linear-gradient(135deg, #7F1D1D, #991B1B);
    padding:22px;
    border-radius:16px;
    font-size:24px;
    font-weight:800;
    color:#FEE2E2 !important;
    text-align:center;
    box-shadow: 0 8px 24px rgba(127,29,29,0.4);
    border: 2px solid #EF4444;
    margin-top: 15px;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn{
    from{ opacity: 0; transform: translateY(8px); }
    to{ opacity: 1; transform: translateY(0px); }
}

/* ---------- Progress bar ---------- */
.stProgress > div > div{
    background: linear-gradient(90deg, #3B82F6, #60A5FA);
    border-radius: 10px;
}

.stProgress{
    margin-top: 10px;
}

/* ---------- Disclaimer ---------- */
.disclaimer{
    background: #422006;
    border-left: 5px solid #EAB308;
    padding: 14px 18px;
    border-radius: 10px;
    font-size: 14px !important;
    color: #FEF3C7 !important;
    font-weight: 600;
    margin-top: 20px;
}

/* ---------- Footer ---------- */
.footer{
    text-align:center;
    font-size:14px;
    color:#94A3B8 !important;
    font-weight: 500;
    padding-top:25px;
    padding-bottom: 10px;
}

hr{
    margin-top: 30px;
    border: none;
    border-top: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# -------------------- Header --------------------
st.markdown("""
<div class="header-card">
    <div class="logo-icon">🩺</div>
    <h1>COVID-19 Detection</h1>
    <p>AI-powered analysis of chest X-ray images to help identify potential COVID-19 indicators</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📤 Upload a Chest X-Ray Image")

uploaded_file = st.file_uploader(
    "Choose an Image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=350)

    img = image.resize((299, 299))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, 0)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing X-ray..."):
            prediction = model.predict(img)
            score = float(prediction[0][0])

        if score > 0.5:
            confidence = score * 100
            st.markdown(
                f"<div class='error-box'>🔴 COVID-19 DETECTED<br><br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            confidence = (1 - score) * 100
            st.markdown(
                f"<div class='success-box'>🟢 NORMAL<br><br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )

        st.progress(confidence / 100)

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> This tool is for educational/demo purposes only and is not a substitute for professional medical diagnosis. Please consult a certified radiologist or physician for accurate diagnosis.
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div class='footer'>Developed with ❤️ using TensorFlow & Streamlit</div>",
    unsafe_allow_html=True
)
