# -*- coding: utf-8 -*-
"""
Outliers & Percentile Exercise — Streamlit App
Converted from 1_outliers_percentile_exercise.ipynb

Run with:
    streamlit run outliers_percentile_app.py
"""

import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Outliers & Percentile Exercise", layout="wide")

st.title("📊 Outliers & Percentile Exercise")
st.markdown(
    "This app removes outliers from a dataset using **percentile (quantile) "
    "thresholding** on the `price` column — the same logic as the original notebook."
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
st.sidebar.header("1️⃣ Load Data")

uploaded_file = st.sidebar.file_uploader(
    "Optional: upload a different CSV", type=["csv"]
)

# Look for the CSV that ships with the repo instead of the old Colab-only
# "/content/..." path. We check a few common locations relative to this
# script so it works no matter where the repo is checked out / deployed.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_PATHS = [
    os.path.join(SCRIPT_DIR, "AB_NYC_2019.csv"),
    os.path.join(SCRIPT_DIR, "data", "AB_NYC_2019.csv"),
    "AB_NYC_2019.csv",  # cwd fallback
]


@st.cache_data
def load_csv(source):
    return pd.read_csv(source)


df = None
if uploaded_file is not None:
    df = load_csv(uploaded_file)
else:
    found_path = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
    if found_path:
        st.sidebar.caption(f"Auto-loaded from repo:\n`{found_path}`")
        df = load_csv(found_path)
    else:
        st.warning(
            "Couldn't find `AB_NYC_2019.csv` next to the script (or in a `data/` "
            "subfolder). Either place the CSV alongside this script in the repo, "
            "or upload it using the sidebar."
        )
        st.stop()

st.subheader("Raw Data Preview")
st.dataframe(df.head())

with st.expander("Show `df.describe()`"):
    st.dataframe(df.describe())

# ---------------------------------------------------------------------------
# 2. Choose column + percentile thresholds
# ---------------------------------------------------------------------------
st.sidebar.header("2️⃣ Choose Column & Thresholds")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
default_col = "price" if "price" in numeric_cols else (numeric_cols[0] if numeric_cols else None)

if default_col is None:
    st.error("No numeric columns found in this dataset.")
    st.stop()

col = st.sidebar.selectbox(
    "Column to filter outliers on",
    options=numeric_cols,
    index=numeric_cols.index(default_col),
)

lower_pct, upper_pct = st.sidebar.slider(
    "Percentile range to KEEP (%)",
    min_value=0.0,
    max_value=100.0,
    value=(1.0, 99.9),
    step=0.1,
    help="Rows outside this percentile range on the selected column will be treated as outliers.",
)

min_threshold, max_threshold = df[col].quantile([lower_pct / 100, upper_pct / 100])

st.subheader("Thresholds")
c1, c2 = st.columns(2)
c1.metric(f"Lower threshold ({lower_pct}%)", f"{min_threshold:,.2f}")
c2.metric(f"Upper threshold ({upper_pct}%)", f"{max_threshold:,.2f}")

# ---------------------------------------------------------------------------
# 3. Show outliers below the lower threshold
# ---------------------------------------------------------------------------
st.subheader(f"Rows with `{col}` below the lower threshold")
below_df = df[df[col] < min_threshold]
st.caption(f"{len(below_df):,} row(s) found")
st.dataframe(below_df)

# ---------------------------------------------------------------------------
# 4. Filter dataset (df2) and compare
# ---------------------------------------------------------------------------
df2 = df[(df[col] > min_threshold) & (df[col] < max_threshold)]

st.subheader("Filtered Data (`df2`)")
c1, c2 = st.columns(2)
c1.metric("Original rows", f"{df.shape[0]:,}")
c2.metric("Rows after removing outliers", f"{df2.shape[0]:,}", delta=f"{df2.shape[0] - df.shape[0]:,}")

st.markdown("**Random sample of filtered data:**")
sample_n = min(5, len(df2))
if sample_n > 0:
    st.dataframe(df2.sample(sample_n))
else:
    st.info("No rows remain after filtering with the current thresholds.")

with st.expander(f"Show `df2['{col}'].describe()`"):
    st.dataframe(df2[col].describe())

# ---------------------------------------------------------------------------
# 5. Visual comparison
# ---------------------------------------------------------------------------
st.subheader("Distribution Before vs. After Outlier Removal")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Before** ({df.shape[0]:,} rows)")
    st.bar_chart(df[col].value_counts(bins=30).sort_index())
with c2:
    st.markdown(f"**After** ({df2.shape[0]:,} rows)")
    if len(df2) > 0:
        st.bar_chart(df2[col].value_counts(bins=30).sort_index())

# ---------------------------------------------------------------------------
# 6. Download filtered data
# ---------------------------------------------------------------------------
st.subheader("Download Filtered Data")
csv_bytes = df2.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download df2 as CSV",
    data=csv_bytes,
    file_name="filtered_data.csv",
    mime="text/csv",
)
