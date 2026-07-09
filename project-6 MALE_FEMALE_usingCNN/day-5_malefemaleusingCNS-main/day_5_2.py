import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="🚹🚺",
    layout="centered"
)

# ----------------------------
# CSS
# ----------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

.stApp{
    background: linear-gradient(135deg, #0F172A 0%, #111827 50%, #0B1120 100%);
    background-attachment: fixed;
}

.block-container{
    background: #1E293B;
    border-radius:20px;
    padding:35px;
    box-shadow:0px 10px 30px rgba(0,0,0,0.5);
    border: 1px solid #334155;
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
    color:#F8FAFC !important;
    font-size: 32px;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.header-card p{
    color:#DBEAFE !important;
    font-size: 16px !important;
    margin-top: 10px;
    font-weight: 400;
}

/* ---------- Force all headings and text light ---------- */
h1,h2,h3,h4,h5,h6{
    color:#F1F5F9 !important;
    font-weight:700 !important;
}

h3{
    text-align:center;
}

p, li, span, label, div{
    color:#E2E8F0;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span{
    color:#E2E8F0 !important;
    font-size:17px;
}

/* File uploader instructions */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span{
    color:#F1F5F9 !important;
    font-weight:600 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] small{
    color:#CBD5E1 !important;
}

/* Upload box */
[data-testid="stFileUploader"]{
    background:#0F172A;
    padding:22px;
    border-radius:16px;
    border:2px dashed #3B82F6;
    box-shadow:0 6px 18px rgba(0,0,0,0.35);
    transition:all 0.3s ease;
}

[data-testid="stFileUploader"]:hover{
    border-color:#60A5FA;
    box-shadow:0 8px 24px rgba(59,130,246,0.3);
}

[data-testid="stFileUploader"] button{
    background:#334155 !important;
    color:#F1F5F9 !important;
    border:1px solid #475569 !important;
    font-weight:600 !important;
}

[data-testid="stFileUploader"] button:hover{
    background:#475569 !important;
}

[data-testid="stFileUploaderFileName"]{
    color:#F1F5F9 !important;
    font-weight:600 !important;
}

/* Image preview */
[data-testid="stImage"]{
    display:flex;
    justify-content:center;
    margin:20px 0;
}

[data-testid="stImage"] img{
    border-radius:16px;
    box-shadow:0 8px 24px rgba(0,0,0,0.5);
    border:4px solid #0F172A;
}

/* Button */
.stButton>button{
    width:100%;
    background:linear-gradient(135deg,#2563EB,#1D4ED8);
    color:#F8FAFC !important;
    border-radius:14px;
    font-size:20px;
    font-weight:700;
    padding:14px;
    border:none;
    box-shadow:0 6px 18px rgba(37,99,235,0.5);
    transition:all 0.25s ease;
    letter-spacing:0.5px;
}

.stButton>button:hover{
    background:linear-gradient(135deg,#3B82F6,#2563EB);
    transform:translateY(-2px);
    box-shadow:0 10px 24px rgba(59,130,246,0.6);
    color:#F8FAFC !important;
}

.stButton>button:active{
    transform:translateY(0px);
}

/* Result boxes */
.success-box{
    background:linear-gradient(135deg,#4C1D95,#6D28D9);
    color:#F3E8FF !important;
    padding:20px;
    border-radius:16px;
    text-align:center;
    font-size:22px;
    font-weight:800;
    box-shadow:0 8px 24px rgba(109,40,217,0.4);
    border:2px solid #A78BFA;
    animation: fadeIn 0.5s ease;
}

.info-box{
    background:linear-gradient(135deg,#1E3A8A,#2563EB);
    color:#DBEAFE !important;
    padding:20px;
    border-radius:16px;
    text-align:center;
    font-size:22px;
    font-weight:800;
    box-shadow:0 8px 24px rgba(37,99,235,0.4);
    border:2px solid #60A5FA;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn{
    from{ opacity:0; transform:translateY(8px); }
    to{ opacity:1; transform:translateY(0px); }
}

/* Score text */
.score-text{
    text-align:center;
    font-size:17px;
    font-weight:600;
    color:#CBD5E1 !important;
    margin-bottom:10px;
}

/* Progress bar */
.stProgress > div > div{
    background:linear-gradient(90deg,#3B82F6,#60A5FA);
    border-radius:10px;
}

/* Footer */
.footer{
    text-align:center;
    margin-top:30px;
    color:#94A3B8 !important;
    font-size:14px;
    font-weight:500;
}

hr{
    border:none;
    border-top:1px solid #334155;
    margin-top:25px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("binary_image_classifier (1).keras")

model = load_model()

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<div class="header-card">
    <div class="logo-icon">🧑‍🤝‍🧑</div>
    <h1>Male vs Female Image Classifier</h1>
    <p>Upload a face image and let AI predict whether it's Male or Female</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📤 Upload a Face Image")
st.write("Click **Predict** after uploading to identify whether the image is classified as **Male** or **Female**.")

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            prediction = model.predict(img)
            score = float(prediction[0][0])

        st.markdown(f"<div class='score-text'>Prediction Score: <b>{score:.4f}</b></div>", unsafe_allow_html=True)

        if score >= 0.5:
            confidence = score * 100
            st.markdown(
                f"""
                <div class='info-box'>
                👨 Prediction : <br><br>
                <b>MALE</b><br><br>
                Confidence : {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
            st.toast("Prediction complete: Male 👨", icon="✅")
        else:
            confidence = (1 - score) * 100
            st.markdown(
                f"""
                <div class='success-box'>
                👩 Prediction : <br><br>
                <b>FEMALE</b><br><br>
                Confidence : {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
            st.toast("Prediction complete: Female 👩", icon="✅")

        st.progress(confidence / 100)

        # ---------------- Lively notification ----------------
        if confidence >= 90:
            st.balloons()
        else:
            st.snow()

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div class='footer'>Developed with ❤️ using TensorFlow & Streamlit</div>",
    unsafe_allow_html=True
)
