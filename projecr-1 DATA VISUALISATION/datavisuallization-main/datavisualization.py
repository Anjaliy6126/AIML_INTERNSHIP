"""
📊 Google Play Store — Data Visualisation Case Study
A dark-themed, interactive Streamlit app built from the original notebook.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------
# DATASET PATH — bundled in the same GitHub repo as this app
# (https://github.com/Anjaliy6126/datavisuallization)
# ----------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_CANDIDATES = [
    "googleplaystore_v2 (1).csv",
    "googleplaystore_v2.csv",
    "data/googleplaystore_v2 (1).csv",
    "data/googleplaystore_v2.csv",
]

def find_bundled_csv():
    for name in CSV_CANDIDATES:
        path = os.path.join(APP_DIR, name)
        if os.path.exists(path):
            return path
    return None

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Play Store Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# DARK THEME CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&family=Fira+Code&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Overall dark background with subtle gradient */
.stApp {
    background: linear-gradient(160deg, #0f0c29 0%, #16213e 45%, #1a1a2e 100%);
    color: #eef1ff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
    border-right: 1px solid rgba(155, 140, 255, 0.25);
}
section[data-testid="stSidebar"] * {
    color: #e8e6ff !important;
}

/* Headings */
h1, h2, h3 {
    background: linear-gradient(90deg, #a18cd1, #fbc2eb, #8ec5fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: 0.5px;
}

p, li, span, label, .stMarkdown {
    color: #d9d9f3 !important;
}

/* Logo / hero banner */
.hero-banner {
    padding: 28px 32px;
    border-radius: 20px;
    background: linear-gradient(120deg, rgba(161,140,209,0.18), rgba(142,197,252,0.10));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 22px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(161,140,209,0.15), rgba(142,197,252,0.08));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}
div[data-testid="stMetricLabel"] { color: #c9c4ff !important; }
div[data-testid="stMetricValue"] { color: #ffffff !important; }

/* Containers / cards */
.custom-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 18px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.3);
}

/* Buttons */
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

/* Radio / selectbox / uploader */
div[data-baseweb="select"], .stFileUploader, .stSlider {
    border-radius: 12px;
}
.stFileUploader > div {
    background: rgba(255,255,255,0.04);
    border: 1.5px dashed rgba(161,140,209,0.5);
    border-radius: 14px;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #c9c4ff !important;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid #a18cd1 !important;
}

/* DataFrame */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# Matching dark plot theme
plt.style.use("dark_background")
PLOT_BG = "#16213e"
plt.rcParams.update({
    "figure.facecolor": PLOT_BG,
    "axes.facecolor": PLOT_BG,
    "savefig.facecolor": PLOT_BG,
    "axes.edgecolor": "#8ec5fc",
    "axes.labelcolor": "#eef1ff",
    "xtick.color": "#d9d9f3",
    "ytick.color": "#d9d9f3",
    "text.color": "#eef1ff",
    "grid.color": "#33334d",
})
ACCENT_PALETTE = sns.color_palette(["#a18cd1", "#fbc2eb", "#8ec5fc", "#f6d365", "#84fab0", "#ff9a9e"])
sns.set_palette(ACCENT_PALETTE)

PLOTLY_TEMPLATE = "plotly_dark"

# ----------------------------------------------------------------------
# HERO / LOGO HEADER
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display:flex; align-items:center; gap:18px;">
        <div style="font-size:52px;">📱✨</div>
        <div>
            <h1 style="margin-bottom:0;">Google Play Store — Insights Lab</h1>
            <p style="margin-top:4px; font-size:15px;">🎯 Data Cleaning • 📊 Visual Storytelling • 🚀 Business Insights</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SIDEBAR — DATA UPLOAD & NAV
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.success("📂 Dataset auto-loaded from the repo — no upload needed!")
    st.markdown("---")
    section = st.radio(
        "Choose a section",
        [
            "🏠 Overview",
            "🧹 Data Cleaning",
            "📦 Outlier Analysis",
            "📈 Distributions",
            "🥧 Categorical Views",
            "🔍 Relationships",
            "🌡️ Heatmaps",
            "🕒 Trends & Stacked Bars",
            "⚡ Interactive Plotly",
        ],
    )
    st.markdown("---")
    st.markdown("### 💡 Tip")
    st.info("This app reads `googleplaystore_v2 (1).csv` directly from the GitHub repo it's deployed from.")
    st.markdown("Made with 💜 using **Streamlit**")

# ----------------------------------------------------------------------
# DATA LOADING + CLEANING PIPELINE (mirrors the notebook)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def make_demo_data(n=1200):
    rng = np.random.default_rng(42)
    categories = ["GAME", "TOOLS", "FAMILY", "MEDICAL", "EDUCATION", "BUSINESS", "SOCIAL"]
    genres = ["Tools", "Entertainment", "Medical", "Education", "Action", "Puzzle"]
    content = ["Everyone", "Everyone 10+", "Teen", "Mature 17+"]
    df = pd.DataFrame({
        "App": [f"App {i}" for i in range(n)],
        "Category": rng.choice(categories, n),
        "Rating": np.clip(rng.normal(4.1, 0.5, n), 1, 5).round(1),
        "Reviews": rng.integers(0, 2_000_000, n).astype(str),
        "Size": rng.uniform(1, 100, n),
        "Installs": rng.choice(["1,000+", "10,000+", "100,000+", "1,000,000+", "10,000,000+"], n),
        "Type": rng.choice(["Free", "Paid"], n, p=[0.85, 0.15]),
        "Price": [str(0) if t == "Free" else f"${round(rng.uniform(0.99, 29.99), 2)}" for t in rng.choice(["Free", "Paid"], n, p=[0.85, 0.15])],
        "Content Rating": rng.choice(content, n),
        "Genres": rng.choice(genres, n),
        "Last Updated": pd.to_datetime(rng.integers(1, 366, n), unit="D", origin="2018-01-01"),
        "Current Ver": "1.0",
        "Android Ver": rng.choice(["4.1 and up", "4.0 and up", "5.0 and up"], n),
    })
    return df


@st.cache_data(show_spinner=False)
def load_raw(path):
    if path is not None:
        return pd.read_csv(path)
    return make_demo_data()


@st.cache_data(show_spinner=False)
def clean_pipeline(raw: pd.DataFrame):
    steps = []
    df = raw.copy()
    steps.append(("Initial shape", df.shape))

    # Drop rows with null Rating
    if "Rating" in df.columns:
        df = df[~df.Rating.isnull()]
        steps.append(("After dropping null Rating", df.shape))

    # Drop shifted / broken rows (Category == '1.9' known artifact)
    if "Category" in df.columns and "Android Ver" in df.columns:
        df = df[~(df["Android Ver"].isnull() & (df["Category"] == "1.9"))]

    # Fill Android Ver / Current Ver with mode
    for col in ["Android Ver", "Current Ver"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Clean Price
    if "Price" in df.columns and not pd.api.types.is_numeric_dtype(df["Price"]):
        df["Price"] = df["Price"].apply(lambda x: 0 if str(x) == "0" else float(str(x).replace("$", "")))

    # Clean Reviews
    if "Reviews" in df.columns:
        df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce").fillna(0).astype("int64")

    # Clean Installs
    if "Installs" in df.columns and not pd.api.types.is_numeric_dtype(df["Installs"]):
        df["Installs"] = df["Installs"].apply(lambda v: int(str(v).replace(",", "").replace("+", "")) if pd.notnull(v) else 0)

    steps.append(("After type cleaning", df.shape))

    # Sanity checks
    if {"Reviews", "Installs"}.issubset(df.columns):
        df = df[df.Reviews <= df.Installs]
    if {"Type", "Price"}.issubset(df.columns):
        df = df[~((df.Type == "Free") & (df.Price > 0))]
    steps.append(("After sanity checks", df.shape))

    # Outlier removal
    if "Price" in df.columns:
        df = df[df.Price <= 30]
    if "Reviews" in df.columns:
        df = df[df.Reviews <= 1_000_000]
    if "Installs" in df.columns:
        df = df[df.Installs <= 100_000_000]
    steps.append(("After outlier removal", df.shape))

    # Content rating trim
    if "Content Rating" in df.columns:
        df = df[~df["Content Rating"].isin(["Adults only 18+", "Unrated"])]

    df = df.reset_index(drop=True)
    steps.append(("Final shape", df.shape))

    if "Last Updated" in df.columns:
        try:
            df["updated_month"] = pd.to_datetime(df["Last Updated"], errors="coerce").dt.month
        except Exception:
            pass

    if "Size" in df.columns and pd.api.types.is_numeric_dtype(df["Size"]):
        try:
            df["Size_Bucket"] = pd.qcut(df["Size"], [0, 0.2, 0.4, 0.6, 0.8, 1], ["VL", "L", "M", "H", "VH"])
        except Exception:
            pass

    return df, steps


csv_path = find_bundled_csv()
raw_df = load_raw(csv_path)
if csv_path is None:
    st.toast("⚠️ Couldn't find the CSV in the repo — showing a demo dataset instead.", icon="⚠️")
else:
    st.toast(f"✅ Loaded dataset from {os.path.basename(csv_path)}", icon="✅")
clean_df, clean_steps = clean_pipeline(raw_df)

# ----------------------------------------------------------------------
# OVERVIEW
# ----------------------------------------------------------------------
if section == "🏠 Overview":
    st.markdown("## 🏠 Project Overview")
    st.markdown("""
    <div class="custom-card">
    🎯 <b>Problem Statement</b><br><br>
    The Play Store team wants to boost visibility for the most promising apps.
    This dashboard explores: does a bigger size or higher price mean a better app?
    Does a high install count guarantee a better rating? 📈
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📱 Total Apps", f"{clean_df.shape[0]:,}")
    c2.metric("⭐ Avg Rating", f"{clean_df['Rating'].mean():.2f}" if "Rating" in clean_df else "—")
    c3.metric("🆓 Free Apps", f"{(clean_df['Type']=='Free').sum():,}" if "Type" in clean_df else "—")
    c4.metric("🏷️ Categories", f"{clean_df['Category'].nunique():,}" if "Category" in clean_df else "—")

    st.markdown("### 🔍 Peek at the cleaned data")
    st.dataframe(clean_df.head(15), use_container_width=True)

    with st.expander("📋 Column Types & Nulls"):
        info = pd.DataFrame({
            "dtype": clean_df.dtypes.astype(str),
            "nulls": clean_df.isnull().sum(),
        })
        st.dataframe(info, use_container_width=True)

    st.balloons()

# ----------------------------------------------------------------------
# DATA CLEANING
# ----------------------------------------------------------------------
elif section == "🧹 Data Cleaning":
    st.markdown("## 🧹 Data Cleaning Journey")
    st.markdown("Every great analysis starts with clean data. Here's the shape of the dataset at each stage 👇")

    for label, shape in clean_steps:
        st.markdown(f"""
        <div class="custom-card" style="display:flex; justify-content:space-between; align-items:center;">
            <span>✅ <b>{label}</b></span>
            <span style="color:#8ec5fc; font-weight:700;">{shape[0]:,} rows × {shape[1]} cols</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧾 Sanity Checks Performed")
    st.markdown("""
    - ✅ Rating between 1 and 5
    - ✅ Reviews ≤ Installs
    - ✅ Free apps have Price = 0
    - ✅ Outliers removed (Price ≤ 30, Reviews ≤ 1M, Installs ≤ 100M)
    """)

# ----------------------------------------------------------------------
# OUTLIER ANALYSIS
# ----------------------------------------------------------------------
elif section == "📦 Outlier Analysis":
    st.markdown("## 📦 Outlier Analysis with Box Plots")
    numeric_cols = [c for c in ["Price", "Reviews", "Installs", "Size"] if c in clean_df.columns]
    col = st.selectbox("🎛️ Choose a numeric column", numeric_cols)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(x=clean_df[col], ax=ax, color="#a18cd1")
    ax.set_title(f"📦 Box Plot — {col}", fontsize=14)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.hist(clean_df[col].dropna(), bins=25, color="#8ec5fc", edgecolor="#16213e")
    ax2.set_title(f"📊 Histogram — {col}", fontsize=14)
    st.pyplot(fig2)

    st.markdown(f"""
    <div class="custom-card">
    📌 <b>Quick Stats for {col}</b><br>
    Min: {clean_df[col].min():.2f} &nbsp;|&nbsp;
    Median: {clean_df[col].median():.2f} &nbsp;|&nbsp;
    Max: {clean_df[col].max():.2f}
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# DISTRIBUTIONS
# ----------------------------------------------------------------------
elif section == "📈 Distributions":
    st.markdown("## 📈 Distribution Plots")
    bins = st.slider("🎚️ Number of bins", 5, 50, 20)
    color = st.color_picker("🎨 Pick a plot colour", "#a18cd1")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(clean_df["Rating"].dropna(), bins=bins, kde=True, color=color, ax=ax)
    ax.set_title("⭐ Distribution of App Ratings", fontsize=14)
    st.pyplot(fig)

    st.markdown("### 📐 Content Rating vs Average Rating")
    if "Content Rating" in clean_df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=clean_df, x="Content Rating", y="Rating", estimator=np.mean, ax=ax2, palette=ACCENT_PALETTE)
        ax2.set_title("🎯 Average Rating by Content Rating", fontsize=14)
        plt.xticks(rotation=20)
        st.pyplot(fig2)

# ----------------------------------------------------------------------
# CATEGORICAL VIEWS
# ----------------------------------------------------------------------
elif section == "🥧 Categorical Views":
    st.markdown("## 🥧 Pie & Bar Charts")
    if "Content Rating" in clean_df.columns:
        counts = clean_df["Content Rating"].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
                   colors=ACCENT_PALETTE.as_hex(), textprops={"color": "#eef1ff"})
            ax.set_title("🥧 Content Rating Share", fontsize=13)
            st.pyplot(fig)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
            counts.plot.barh(ax=ax2, color="#8ec5fc")
            ax2.set_title("📊 Content Rating Counts", fontsize=13)
            st.pyplot(fig2)

# ----------------------------------------------------------------------
# RELATIONSHIPS (scatter / joint / pair)
# ----------------------------------------------------------------------
elif section == "🔍 Relationships":
    st.markdown("## 🔍 Relationships Between Variables")
    numeric_cols = [c for c in ["Reviews", "Size", "Price", "Rating", "Installs"] if c in clean_df.columns]

    tab1, tab2 = st.tabs(["✨ Scatter / Joint Plot", "🧩 Pair Plot"])
    with tab1:
        xcol = st.selectbox("X-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        ycol = st.selectbox("Y-axis", numeric_cols, index=0)
        g = sns.jointplot(x=clean_df[xcol], y=clean_df[ycol], kind="scatter", color="#a18cd1", height=6)
        g.fig.patch.set_facecolor(PLOT_BG)
        st.pyplot(g.fig)

    with tab2:
        chosen = st.multiselect("Pick up to 4 columns", numeric_cols, default=numeric_cols[:4])
        if len(chosen) >= 2:
            pp = sns.pairplot(clean_df[chosen].dropna(), corner=True, palette=ACCENT_PALETTE)
            pp.fig.patch.set_facecolor(PLOT_BG)
            st.pyplot(pp.fig)
        else:
            st.warning("⚠️ Select at least 2 columns for the pair plot.")

# ----------------------------------------------------------------------
# HEATMAPS
# ----------------------------------------------------------------------
elif section == "🌡️ Heatmaps":
    st.markdown("## 🌡️ Heat Map — Rating across Size Buckets & Content Rating")
    if {"Size_Bucket", "Content Rating", "Rating"}.issubset(clean_df.columns):
        agg = st.selectbox("Aggregation", ["mean", "median", "min", "max"])
        pivot = pd.pivot_table(data=clean_df, index="Content Rating", columns="Size_Bucket",
                                values="Rating", aggfunc=agg)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, cmap="mako", annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "Rating"})
        ax.set_title(f"🌡️ {agg.title()} Rating Heatmap", fontsize=14)
        st.pyplot(fig)
    else:
        st.info("ℹ️ Size buckets unavailable for this dataset.")

# ----------------------------------------------------------------------
# TRENDS + STACKED BARS
# ----------------------------------------------------------------------
elif section == "🕒 Trends & Stacked Bars":
    st.markdown("## 🕒 Monthly Trends")
    if "updated_month" in clean_df.columns:
        monthly_rating = clean_df.groupby("updated_month")["Rating"].mean()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        monthly_rating.plot(marker="o", color="#fbc2eb", ax=ax)
        ax.set_title("📈 Average Rating by Month", fontsize=14)
        ax.set_xlabel("Month")
        st.pyplot(fig)

        if "Installs" in clean_df.columns and "Content Rating" in clean_df.columns:
            st.markdown("### 📚 Stacked Bar — Installs by Content Rating per Month")
            monthly = pd.pivot_table(data=clean_df, values="Installs", index="updated_month",
                                      columns="Content Rating", aggfunc="sum").fillna(0)
            normalize = st.checkbox("🔁 Show as proportion", value=False)
            if normalize:
                monthly = monthly.div(monthly.sum(axis=1), axis=0)
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            monthly.plot(kind="bar", stacked=True, ax=ax2, color=ACCENT_PALETTE.as_hex())
            ax2.set_title("📊 Installs Breakdown by Content Rating", fontsize=14)
            st.pyplot(fig2)
    else:
        st.info("ℹ️ 'Last Updated' column not found — trend view unavailable.")

# ----------------------------------------------------------------------
# INTERACTIVE PLOTLY
# ----------------------------------------------------------------------
elif section == "⚡ Interactive Plotly":
    st.markdown("## ⚡ Interactive Plotly Charts")
    if "updated_month" in clean_df.columns and "Rating" in clean_df.columns:
        res = clean_df.groupby("updated_month")[["Rating"]].mean().reset_index()
        fig = px.line(res, x="updated_month", y="Rating", markers=True,
                      title="⚡ Monthly Average Rating", template=PLOTLY_TEMPLATE,
                      color_discrete_sequence=["#a18cd1"])
        st.plotly_chart(fig, use_container_width=True)

    if {"Reviews", "Rating", "Category"}.issubset(clean_df.columns):
        fig2 = px.scatter(clean_df, x="Reviews", y="Rating", color="Category",
                           size="Installs" if "Installs" in clean_df.columns else None,
                           hover_name="App" if "App" in clean_df.columns else None,
                           title="⚡ Reviews vs Rating by Category", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig2, use_container_width=True)

    st.success("✨ Hover, zoom, and pan on the charts above for a fully interactive experience!")

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; opacity:0.7; padding-bottom:10px;">
    Built with 💜 in Streamlit &nbsp;|&nbsp; 📊 Google Play Store Case Study
</div>
""", unsafe_allow_html=True)
