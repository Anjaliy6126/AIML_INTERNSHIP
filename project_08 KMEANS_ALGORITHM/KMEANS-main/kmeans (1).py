# -*- coding: utf-8 -*-
# kmeans_exercise.py
#
# Streamlit app version of the K-Means clustering exercise on the Iris dataset.
# Original notebook location:
#     https://colab.research.google.com/drive/1QG_I2NXX4MLCmIdM4WdtXt5WtmkEV_1M
#
# Exercise:
# 1. Use the iris flower dataset and form clusters using petal width & length.
# 2. Check whether scaling helps.
# 3. Draw an elbow plot to find the optimal value of k.
#
# Run with:
#     streamlit run kmeans_exercise.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="K-Means Clustering — Iris Flowers",
    page_icon="🌸",
    layout="wide",
)

# ----------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 15% 10%, #2d2145 0%, #1a1530 45%, #100d20 100%);
        color: #f0f0f0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .top-banner {
        text-align: center;
        padding: 0.3rem 0 0.6rem 0;
        font-size: 0.85rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #f0abfc;
        opacity: 0.85;
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f472b6, #c084fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #d8b4fe;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .sticker {
        display: inline-block;
        font-size: 1.6rem;
        margin: 0 0.3rem;
        animation: float 3s ease-in-out infinite;
    }
    .sticker:nth-child(2) { animation-delay: 0.5s; }
    .sticker:nth-child(3) { animation-delay: 1s; }
    @keyframes float {
        0%   { transform: translateY(0px); }
        50%  { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }
    .card {
        background-color: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    }
    .metric-box {
        text-align: center;
        background-color: rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.8rem;
    }
    .footer-note {
        text-align: center;
        color: #a78bce;
        font-size: 0.8rem;
        margin-top: 2rem;
        opacity: 0.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="top-banner">Unsupervised Learning Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">🌸 K-Means Clustering on Iris Flowers</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Cluster iris flowers by petal length & width, compare scaled vs. unscaled data, '
    'and find the optimal k using the elbow method.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div style="text-align:center; margin-bottom: 1.2rem;">
        <span class="sticker">🌷</span>
        <span class="sticker">📊</span>
        <span class="sticker">🔍</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Instructions
# ----------------------------------------------------------------------
with st.container():
    st.markdown(
        """
        <div class="card">
        <b>📋 How to use</b>
        <ol>
            <li>Choose whether to scale the petal features (recommended for fair distance comparisons).</li>
            <li>Pick a value of <b>k</b> (number of clusters) with the slider.</li>
            <li>View the resulting clusters on the scatter plot.</li>
            <li>Check the elbow plot below to judge whether your chosen k is a good fit.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
@st.cache_data
def get_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["flower"] = iris.target
    df = df.drop(["sepal length (cm)", "sepal width (cm)", "flower"], axis="columns")
    return df

df = get_data()

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    use_scaling = st.checkbox("Apply MinMax scaling", value=True)
    k = st.slider("Number of clusters (k)", min_value=2, max_value=8, value=3)
    max_k_elbow = st.slider("Max k for elbow plot", min_value=4, max_value=15, value=10)

# ----------------------------------------------------------------------
# Preprocess
# ----------------------------------------------------------------------
features = df[["petal length (cm)", "petal width (cm)"]].copy()

if use_scaling:
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(scaled_values, columns=features.columns)
else:
    features_scaled = features.copy()

# ----------------------------------------------------------------------
# K-Means clustering
# ----------------------------------------------------------------------
km = KMeans(n_clusters=k, n_init=10, random_state=42)
yp = km.fit_predict(features_scaled)

plot_df = features.copy()
plot_df["cluster"] = yp

# ----------------------------------------------------------------------
# Layout: scatter plot + metrics
# ----------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌸 Cluster Scatter Plot")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    colors = plt.cm.plasma(np.linspace(0, 1, k))
    for cluster_id in sorted(plot_df["cluster"].unique()):
        subset = plot_df[plot_df["cluster"] == cluster_id]
        ax.scatter(
            subset["petal length (cm)"],
            subset["petal width (cm)"],
            color=colors[cluster_id],
            label=f"Cluster {cluster_id}",
            edgecolors="white",
            linewidths=0.4,
            s=60,
        )

    ax.set_xlabel("Petal length (cm)", color="white")
    ax.set_ylabel("Petal width (cm)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1530", edgecolor="none", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.3)

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Clusters (k)", k)
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Inertia (SSE)", f"{km.inertia_:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Scaling applied", "Yes" if use_scaling else "No")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Elbow plot
# ----------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📉 Elbow Plot")
st.caption("Look for the 'elbow' point where adding more clusters stops meaningfully reducing SSE.")

@st.cache_data
def compute_sse(_features_scaled, max_k):
    sse = []
    k_rng = range(1, max_k + 1)
    for kk in k_rng:
        model = KMeans(n_clusters=kk, n_init=10, random_state=42)
        model.fit(_features_scaled)
        sse.append(model.inertia_)
    return list(k_rng), sse

k_rng, sse = compute_sse(features_scaled, max_k_elbow)

fig2, ax2 = plt.subplots(figsize=(8, 4))
fig2.patch.set_alpha(0)
ax2.set_facecolor("none")
ax2.plot(k_rng, sse, marker="o", color="#c084fc", linewidth=2, markerfacecolor="#f472b6")
ax2.set_xlabel("K", color="white")
ax2.set_ylabel("Sum of squared error (SSE)", color="white")
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_color("white")
    spine.set_alpha(0.3)

st.pyplot(fig2)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Raw data (optional)
# ----------------------------------------------------------------------
with st.expander("🔎 View raw clustered data"):
    st.dataframe(plot_df, use_container_width=True)

st.markdown(
    '<div class="footer-note">Built with Streamlit & scikit-learn · Unsupervised Learning Lab</div>',
    unsafe_allow_html=True,
)
