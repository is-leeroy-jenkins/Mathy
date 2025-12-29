'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    name.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

# Mathy imports (verified to exist)
from imputers import SimpleImputer
from scalers import StandardScaler, MinMaxScaler, RobustScaler, NormalScaler
from encoders import OneHotEncoder, OrdinalEncoder

try:
    from streamlit_extras.dataframe_explorer import dataframe_explorer
    HAS_EXTRAS = True
except ImportError:
    HAS_EXTRAS = False
    
# -----------------------------------------------------------------------------------------
# Streamlit configuration
# -----------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Mathy",
	page_icon= r'resources\favicon.ico',
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------------------

def initialize_state() -> None:
    defaults = {
        "raw_df": None,
        "df": None,
        "numeric_cols": [],
        "categorical_cols": [],
        "target_col": None,
        "pipeline_log": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()

# -----------------------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------------------

def log_step(message: str) -> None:
    st.session_state.pipeline_log.append(message)


def render_table(df: pd.DataFrame, height: int = 350) -> None:
    st.dataframe(df, use_container_width=True, height=height)


def detect_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


# -----------------------------------------------------------------------------------------
# Sidebar – Data Source
# -----------------------------------------------------------------------------------------

st.sidebar.title("📦 Dataset")

uploaded = st.sidebar.file_uploader(
    "Upload spreadsheet",
    type=["xlsx", "xls", "csv"]
)

use_fallback = st.sidebar.checkbox("Use fallback data", value=False)

if uploaded or use_fallback:
    try:
        if uploaded:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            log_step(f"Loaded uploaded file: {uploaded.name}")
        else:
            df = pd.read_excel("stores/excel/Combined Schedules.xlsx")
            log_step("Loaded fallback dataset: Combined Schedules.xlsx")

        st.session_state.raw_df = df.copy()
        st.session_state.df = df.copy()
        st.session_state.numeric_cols, st.session_state.categorical_cols = detect_column_types(df)

    except Exception as ex:
        st.error(f"Data load failed: {ex}")

# -----------------------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------------------

tabs = st.tabs(["🧹 Data Processing", "📊 Data Analysis"])

# =========================================================================================
# TAB 1 — DATA PROCESSING
# =========================================================================================

with tabs[0]:
    st.header("🧹 Data Processing")

    if st.session_state.raw_df is None:
        st.info("Upload a dataset or enable fallback data to begin.")
        st.stop()

    st.subheader("Raw Data (Interactive)")
    
    if HAS_EXTRAS:
	    filtered_raw = dataframe_explorer( st.session_state.raw_df )
    else:
	    filtered_raw = st.session_state.raw_df
    
    render_table(filtered_raw)

    st.subheader("Column Role Assignment")

    cols = st.session_state.raw_df.columns.tolist()

    target = st.selectbox("Target column (optional)", ["<None>"] + cols)
    st.session_state.target_col = None if target == "<None>" else target

    numeric_cols = st.multiselect(
        "Numeric columns",
        options=cols,
        default=st.session_state.numeric_cols
    )

    categorical_cols = st.multiselect(
        "Categorical columns",
        options=[c for c in cols if c not in numeric_cols],
        default=st.session_state.categorical_cols
    )

    st.session_state.numeric_cols = numeric_cols
    st.session_state.categorical_cols = categorical_cols

    # -------------------------------
    # Imputation
    # -------------------------------

    st.subheader("Missing Value Imputation")

    if st.button("Apply Imputation"):
        df = st.session_state.df.copy()

        if numeric_cols:
            imp = SimpleImputer(strategy="mean")
            df[numeric_cols] = imp.train_transform(df[numeric_cols], None)
            log_step("Imputed numeric columns (mean)")

        if categorical_cols:
            imp = SimpleImputer(strategy="most_frequent")
            df[categorical_cols] = imp.train_transform(df[categorical_cols], None)
            log_step("Imputed categorical columns (most frequent)")

        st.session_state.df = df
        st.success("Imputation completed")

    # -------------------------------
    # Encoding
    # -------------------------------

    st.subheader("Categorical Encoding")

    encoder_type = st.selectbox("Encoding strategy", ["One-Hot", "Ordinal"])

    if st.button("Apply Encoding"):
        df = st.session_state.df.copy()

        if categorical_cols:
            encoder = OneHotEncoder() if encoder_type == "One-Hot" else OrdinalEncoder()
            encoded = encoder.train_transform(df[categorical_cols], None)
            encoded_df = pd.DataFrame(encoded, index=df.index)

            df = df.drop(columns=categorical_cols)
            df = pd.concat([df, encoded_df], axis=1)

            log_step(f"Applied {encoder_type} encoding")

        st.session_state.df = df
        st.success("Encoding completed")

    # -------------------------------
    # Scaling
    # -------------------------------

    st.subheader("Scaling / Normalization")

    scaler_name = st.selectbox("Scaler", ["None", "Standard", "MinMax", "Robust", "Normalize"])

    if st.button("Apply Scaling"):
        df = st.session_state.df.copy()

        if scaler_name != "None" and numeric_cols:
            scaler_map = {
                "Standard": StandardScaler,
                "MinMax": MinMaxScaler,
                "Robust": RobustScaler,
                "Normalize": NormalScaler
            }
            scaler = scaler_map[scaler_name]()
            df[numeric_cols] = scaler.train_transform( df[numeric_cols] )
            log_step(f"Applied {scaler_name} scaling")

        st.session_state.df = df
        st.success("Scaling completed")

    st.subheader("Processed Data")
    render_table(st.session_state.df)

    st.subheader("Pipeline Log")
    for step in st.session_state.pipeline_log:
        st.write(f"• {step}")

# =========================================================================================
# TAB 2 — DATA ANALYSIS (VISUALIZATION-FOCUSED)
# =========================================================================================

with tabs[1]:
    st.header("📊 Data Analysis")

    df = st.session_state.df
    if df is None:
        st.info("No processed data available.")
        st.stop()

    st.subheader("Descriptive Statistics")
    render_table(df.describe(include="all").transpose())

    numeric_df = df.select_dtypes(include=[np.number])

    # -------------------------------
    # Correlation Heatmap
    # -------------------------------

    if not numeric_df.empty:
        st.subheader("Correlation Matrix")

        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(
                    j, i,
                    f"{corr.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

        fig.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Correlation Matrix")

        st.pyplot(fig)
        render_table(corr)

    # -------------------------------
    # Distribution Explorer
    # -------------------------------

    st.subheader("Distribution Explorer")

    col = st.selectbox("Select numeric column", options=numeric_df.columns.tolist())

    if col:
        data = numeric_df[col].dropna()

        fig, ax = plt.subplots(figsize=(8, 5))
        counts, bins, patches = ax.hist(
            data,
            bins=30,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85
        )

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        for count, patch in zip(counts, patches):
            if count > 0:
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    count,
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        st.pyplot(fig)
        render_table(data.to_frame(name=col))
