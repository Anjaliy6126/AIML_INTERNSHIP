# -*- coding: utf-8 -*-
"""
🎬 Movie Review NLP Analyzer — Streamlit App
End-to-end NLP pipeline using Hugging Face Transformers on movie reviews.

Tasks:
- 😊 Sentiment Analysis (Positive/Negative)
- 🌍 Machine Translation (English → Spanish)
- ❓ Extractive Question Answering
- 📝 Text Summarization
- 📊 Model Evaluation (Accuracy, F1, BLEU)
- 🎭 Sample Review Sentiment Test
"""

import os
import streamlit as st
import pandas as pd
import torch

# ----------------------------------------------------------------------
# ⭐ DEFAULT DATA FILES (already committed to the repo)
# ----------------------------------------------------------------------
# These paths are relative to where app.py lives. Since the CSV and the
# reference-translations file are already checked into the repo, the app
# loads them automatically on startup.
#
# 👉 If your filenames/paths differ, just update the two constants below
#    to match what's actually in your repo.
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "netflix movie dhurandhar 2.csv")
DEFAULT_REFS_PATH = os.path.join(os.path.dirname(__file__), "reference_translations.txt")

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Review NLP Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# DARK THEME CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: linear-gradient(160deg, #0f0c29 0%, #16213e 45%, #1a1a2e 100%);
    color: #eef1ff;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
    border-right: 1px solid rgba(155, 140, 255, 0.25);
}
section[data-testid="stSidebar"] * {
    color: #e8e6ff !important;
}

/* Sidebar nav buttons */
section[data-testid="stSidebar"] .stButton>button {
    width: 100%;
    text-align: left;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e8e6ff !important;
    border-radius: 10px;
    padding: 0.6em 1em;
    margin-bottom: 8px;
    font-weight: 600;
    transition: 0.2s;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background: linear-gradient(90deg, #8e2de2, #4a00e0);
    border-color: #a18cd1;
    transform: translateX(3px);
}

h1, h2, h3 {
    background: linear-gradient(90deg, #a18cd1, #fbc2eb, #8ec5fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}

p, li, span, label, .stMarkdown {
    color: #d9d9f3 !important;
}

.hero-banner {
    padding: 28px 32px;
    border-radius: 20px;
    background: linear-gradient(120deg, rgba(161,140,209,0.18), rgba(142,197,252,0.10));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 22px;
}

.custom-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 18px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.3);
}

.result-box {
    background: linear-gradient(135deg, rgba(161,140,209,0.15), rgba(142,197,252,0.10));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 18px 20px;
    margin-top: 14px;
}

.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #8e2de2, #4a00e0);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.6em 1.4em;
    font-weight: 600;
    transition: 0.25s;
    box-shadow: 0 4px 14px rgba(142, 45, 226, 0.35);
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 22px rgba(142, 45, 226, 0.55);
}

.stTextInput input, .stTextArea textarea {
    background-color: #1e2130 !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
    border: 1px solid #2d3348 !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(161,140,209,0.15), rgba(142,197,252,0.08));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}
div[data-testid="stMetricLabel"] { color: #c9c4ff !important; }
div[data-testid="stMetricValue"] { color: #ffffff !important; }

.stFileUploader > div {
    background: rgba(255,255,255,0.04);
    border: 1.5px dashed rgba(161,140,209,0.5);
    border-radius: 14px;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

hr { border-color: rgba(255,255,255,0.1); }

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display:flex; align-items:center; gap:18px;">
        <div style="font-size:52px;">🎬🤖</div>
        <div>
            <h1 style="margin-bottom:0;">Movie Review NLP Analyzer</h1>
            <p style="margin-top:4px; font-size:15px;">😊 Sentiment • 🌍 Translation • ❓ Q&A • 📝 Summarization — powered by Hugging Face Transformers</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SIDEBAR — NAVIGATION
# ----------------------------------------------------------------------
if "screen" not in st.session_state:
    st.session_state.screen = "🏠 Overview"

with st.sidebar:
    st.markdown("### 🧭 Navigate")

    nav_options = [
        "🏠 Overview",
        "😊 Sentiment Analysis",
        "🌍 Translation (EN → ES)",
        "❓ Question Answering",
        "📝 Summarization",
        "🎭 Sample Review Test",
    ]
    for opt in nav_options:
        if st.button(opt, key=f"nav_{opt}"):
            st.session_state.screen = opt

    st.markdown("---")
    st.markdown("### 💡 Tip")
    st.info("Models load lazily and are cached — the first run of each section may take a minute to download weights.")
    st.markdown("Made with 💜 using **Streamlit + 🤗 Transformers**")

screen = st.session_state.screen

# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def make_demo_reviews():
    return pd.DataFrame({
        "Review": [
            "The movie was fantastic! Great performances all around.",
            "Absolutely loved the action sequences, but the pacing dragged in the middle.",
            "A disappointing sequel — weak plot and forgettable dialogue.",
            "One of the best films I've seen this year, truly emotional.",
            "Not worth the hype, the story felt recycled from the first part.",
        ],
        "Class": ["POSITIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"],
    })


@st.cache_data(show_spinner=False)
def load_reviews_from_path(path):
    return pd.read_csv(path, delimiter=";")


@st.cache_data(show_spinner=False)
def load_reference_translations_from_path(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_reviews():
    """Priority: repo default file > demo data."""
    if os.path.exists(DEFAULT_CSV_PATH):
        return load_reviews_from_path(DEFAULT_CSV_PATH), "repo"
    return make_demo_reviews(), "demo"


def load_reference_translations():
    """Priority: repo default file > none."""
    if os.path.exists(DEFAULT_REFS_PATH):
        return load_reference_translations_from_path(DEFAULT_REFS_PATH), "repo"
    return [], "none"


df, csv_source = load_reviews()
reference_translations, refs_source = load_reference_translations()

source_labels = {"repo": "the repo's default file", "demo": "built-in demo data"}
if csv_source == "demo":
    st.toast(f"⚠️ Couldn't find {os.path.basename(DEFAULT_CSV_PATH)} in the repo — showing demo reviews.", icon="⚠️")
else:
    st.toast(f"✅ Loaded {len(df)} reviews from {source_labels[csv_source]}", icon="✅")

reviews = df["Review"].tolist() if "Review" in df.columns else []
real_labels = df["Class"].tolist() if "Class" in df.columns else []

# ----------------------------------------------------------------------
# CACHED MODEL LOADERS (lazy — only load what's needed)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


@st.cache_resource(show_spinner=False)
def get_translation_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def get_qa_model():
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    model_ckp = "deepset/minilm-uncased-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_ckp)
    model = AutoModelForQuestionAnswering.from_pretrained(model_ckp)
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def get_summarization_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "cnicu/t5-small-booksum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# ----------------------------------------------------------------------
# OVERVIEW
# ----------------------------------------------------------------------
if screen == "🏠 Overview":
    st.markdown("## 🏠 Dataset Overview")

    if reviews:
        c1, c2, c3 = st.columns(3)
        c1.metric("📄 Total Reviews", len(reviews))
        if real_labels:
            pos_count = sum(1 for l in real_labels if str(l).upper() == "POSITIVE")
            c2.metric("😊 Positive", pos_count)
            c3.metric("😞 Negative", len(real_labels) - pos_count)

        st.markdown("### 🔍 Peek at the reviews")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ The reviews file needs a `Review` column (and ideally a `Class` column).")

    if reference_translations:
        st.success(f"✅ Loaded {len(reference_translations)} reference translation line(s) from {source_labels[refs_source]} for BLEU scoring.")

    st.markdown("""
    <div class="custom-card">
    <b>🧠 What this app demonstrates:</b><br><br>
    😊 <b>Sentiment Analysis</b> — DistilBERT fine-tuned on SST-2<br>
    🌍 <b>Machine Translation</b> — English → Spanish via Helsinki-NLP MarianMT<br>
    ❓ <b>Extractive Q&A</b> — MiniLM fine-tuned on SQuAD2<br>
    📝 <b>Summarization</b> — T5-small fine-tuned on BookSum<br>
    🎭 <b>Sample Review Test</b> — quick sentiment check on a custom line
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SENTIMENT ANALYSIS
# ----------------------------------------------------------------------
elif screen == "😊 Sentiment Analysis":
    st.markdown("## 😊 Sentiment Analysis")
    st.markdown("Classify reviews as **Positive** or **Negative** using a DistilBERT model.")

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("#### ✍️ Try a single review")
    custom_review = st.text_area("Enter a review", value="The movie was fantastic!", height=100)
    single_clicked = st.button("🔮 Analyze Sentiment")
    st.markdown('</div>', unsafe_allow_html=True)

    if single_clicked and custom_review.strip():
        with st.spinner("Loading model & analyzing..."):
            classifier = get_sentiment_pipeline()
            result = classifier(custom_review)[0]
        emoji = "😊" if result["label"] == "POSITIVE" else "😞"
        st.markdown(f"""
        <div class="result-box">
        {emoji} <b>Prediction:</b> {result['label']} &nbsp;|&nbsp; <b>Confidence:</b> {result['score']*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 Batch Analysis on Uploaded Reviews")
    run_batch = st.button("🚀 Run Sentiment on All Reviews")

    if run_batch and reviews:
        with st.spinner("Analyzing all reviews..."):
            classifier = get_sentiment_pipeline()
            predicted_labels = classifier(reviews)

        results_df = pd.DataFrame({
            "Review": reviews,
            "Predicted": [p["label"] for p in predicted_labels],
            "Confidence": [f"{p['score']*100:.2f}%" for p in predicted_labels],
        })
        if real_labels:
            results_df.insert(1, "Actual", real_labels)
        st.dataframe(results_df, use_container_width=True)

        if real_labels:
            import evaluate
            accuracy = evaluate.load("accuracy")
            f1 = evaluate.load("f1")

            references = [1 if str(l).upper() == "POSITIVE" else 0 for l in real_labels]
            predictions = [1 if p["label"] == "POSITIVE" else 0 for p in predicted_labels]

            acc_result = accuracy.compute(references=references, predictions=predictions)["accuracy"]
            f1_result = f1.compute(references=references, predictions=predictions)["f1"]

            c1, c2 = st.columns(2)
            c1.metric("🎯 Accuracy", f"{acc_result*100:.2f}%")
            c2.metric("📐 F1 Score", f"{f1_result:.4f}")

# ----------------------------------------------------------------------
# TRANSLATION
# ----------------------------------------------------------------------
elif screen == "🌍 Translation (EN → ES)":
    st.markdown("## 🌍 English → Spanish Translation")
    st.markdown("Powered by the **Helsinki-NLP/opus-mt-en-es** MarianMT model.")

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    default_text = reviews[0] if reviews else "The movie was fantastic!"
    text_to_translate = st.text_area("Text to translate", value=default_text, height=100)
    translate_clicked = st.button("🌐 Translate")
    st.markdown('</div>', unsafe_allow_html=True)

    if translate_clicked and text_to_translate.strip():
        with st.spinner("Loading model & translating..."):
            tokenizer, model = get_translation_model()
            inputs = tokenizer(text_to_translate, return_tensors="pt", truncation=True, max_length=512)
            translated_tokens = model.generate(**inputs)
            translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        st.session_state["last_translation"] = translated_text
        st.markdown(f"""
        <div class="result-box">
        🇬🇧 <b>Original:</b><br>{text_to_translate}<br><br>
        🇪🇸 <b>Translated:</b><br>{translated_text}
        </div>
        """, unsafe_allow_html=True)

    if "last_translation" in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 BLEU Score Evaluation")

        if reference_translations:
            st.caption(f"Using {len(reference_translations)} reference line(s) from {source_labels[refs_source]}.")
            if st.button("📐 Compute BLEU"):
                import evaluate
                bleu = evaluate.load("bleu")
                bleu_score = bleu.compute(
                    predictions=[st.session_state["last_translation"]],
                    references=[reference_translations]
                )
                st.metric("🎯 BLEU Score", f"{bleu_score['bleu']*100:.2f}")
        else:
            st.caption("No reference translations found — paste one below instead.")
            reference_text = st.text_area("Reference translation(s)", height=80, placeholder="La película fue fantástica!")
            if st.button("📐 Compute BLEU"):
                if reference_text.strip():
                    import evaluate
                    bleu = evaluate.load("bleu")
                    references = [line.strip() for line in reference_text.splitlines() if line.strip()]
                    bleu_score = bleu.compute(
                        predictions=[st.session_state["last_translation"]],
                        references=[references]
                    )
                    st.metric("🎯 BLEU Score", f"{bleu_score['bleu']*100:.2f}")
                else:
                    st.warning("⚠️ Please provide at least one reference translation.")

# ----------------------------------------------------------------------
# QUESTION ANSWERING
# ----------------------------------------------------------------------
elif screen == "❓ Question Answering":
    st.markdown("## ❓ Extractive Question Answering")
    st.markdown("Powered by **deepset/minilm-uncased-squad2**, fine-tuned on SQuAD2.")

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    default_context = reviews[1] if len(reviews) > 1 else (reviews[0] if reviews else "")
    context = st.text_area("📄 Context (a review)", value=default_context, height=120)
    question = st.text_input("❔ Question", value="What is the best opinion of the movie?")
    qa_clicked = st.button("🔍 Find Answer")
    st.markdown('</div>', unsafe_allow_html=True)

    if qa_clicked and context.strip() and question.strip():
        with st.spinner("Loading model & extracting answer..."):
            tokenizer, model = get_qa_model()
            inputs = tokenizer(question, context, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits) + 1
            answer_span = inputs["input_ids"][0][start_idx:end_idx]
            answer = tokenizer.decode(answer_span)

        st.markdown(f"""
        <div class="result-box">
        💬 <b>Answer:</b> {answer if answer.strip() else "🤷 No clear answer found in the context."}
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SUMMARIZATION
# ----------------------------------------------------------------------
elif screen == "📝 Summarization":
    st.markdown("## 📝 Text Summarization")
    st.markdown("Powered by **cnicu/t5-small-booksum**, a T5 model fine-tuned for summarization.")

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    default_text = reviews[-1] if reviews else ""
    text_to_summarize = st.text_area("Text to summarize", value=default_text, height=150)
    c1, c2 = st.columns(2)
    with c1:
        max_len = st.slider("Max summary length", 20, 100, 53)
    with c2:
        min_len = st.slider("Min summary length", 5, 50, 20)
    summarize_clicked = st.button("✂️ Summarize")
    st.markdown('</div>', unsafe_allow_html=True)

    if summarize_clicked and text_to_summarize.strip():
        with st.spinner("Loading model & summarizing..."):
            tokenizer, model = get_summarization_model()
            input_text = "summarize: " + text_to_summarize
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            summary_ids = model.generate(
                **inputs,
                max_length=max_len,
                min_length=min_len,
                num_beams=4,
                early_stopping=True,
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.markdown(f"""
        <div class="result-box">
        📝 <b>Summary:</b><br>{summary}
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SAMPLE REVIEW SENTIMENT TEST
# ----------------------------------------------------------------------
elif screen == "🎭 Sample Review Test":
    st.markdown("## 🎭 Sample Review Sentiment Test")
    st.markdown("Quick sentiment check on a standalone sample review — separate from the batch dataset.")

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    sample_review = st.text_area(
        "Sample review",
        value="KGF 2 is an amazing movie with powerful action and excellent performance.",
        height=100
    )
    sample_clicked = st.button("🎬 Analyze Sample Review")
    st.markdown('</div>', unsafe_allow_html=True)

    if sample_clicked and sample_review.strip():
        with st.spinner("Loading model & analyzing..."):
            classifier = get_sentiment_pipeline()
            sentiment_result = classifier(sample_review)
            sentiment = sentiment_result[0]["label"]

        emoji = "😊" if sentiment == "POSITIVE" else "😞"
        st.markdown(f"""
        <div class="result-box">
        📝 <b>Sample Review:</b><br>{sample_review}<br><br>
        {emoji} <b>Sentiment Analysis:</b> {sentiment}
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; opacity:0.7; padding-bottom:10px;">
    Built with 💜 in Streamlit &nbsp;|&nbsp; 🤗 Powered by Hugging Face Transformers
</div>
""", unsafe_allow_html=True)
