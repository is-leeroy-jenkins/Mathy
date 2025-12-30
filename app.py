# ******************************************************************************************
# Assembly:                Mathy-Py
# Filename:                app.py
# ******************************************************************************************

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from typing import List

# Mathy (verified)
from imputers import SimpleImputer
from scalers import StandardScaler, MinMaxScaler, RobustScaler, NormalScaler

# sklearn / statsmodels
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.power import TTestPower

# -----------------------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Mathy",
    layout="wide",
    page_icon=r"resources/favicon.ico",
    initial_sidebar_state="expanded"
)

pd.options.display.float_format = "{:.4f}".format

# -----------------------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "raw_df": None,
        "df": None,
        "numeric_cols": [],
        "categorical_cols": [],
        "pipeline_log": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# -----------------------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------------------

def blue_divider() -> None:
    st.markdown(
        "<div style='height:3px;background:#1f77b4;margin:1rem 0;'></div>",
        unsafe_allow_html=True
    )

def log_step(msg: str) -> None:
    st.session_state.pipeline_log.append(msg)

def render_table(df: pd.DataFrame, height: int = 360) -> None:
    disp = df.copy()
    float_cols = disp.select_dtypes(include=[np.floating]).columns
    disp[float_cols] = disp[float_cols].round(4)
    st.dataframe(disp, use_container_width=True, height=height)

def detect_column_types(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    numeric_hints = ("py", "cy", "by", "amount", "total", "value", "balance", "outlay")
    categorical_hints = ("fy", "code", "id", "name", "type", "symbol")

    numeric, categorical = [], []

    for col in df.columns:
        name = col.lower()
        if any(h in name for h in categorical_hints):
            categorical.append(col)
        elif any(h in name for h in numeric_hints):
            numeric.append(col)
        elif pd.api.types.is_float_dtype(df[col]):
            numeric.append(col)
        elif pd.api.types.is_integer_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)

    return numeric, categorical

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(axis=1, how="all")
    out = out.loc[:, out.nunique(dropna=True) > 1]
    return out

def analysis_fillna_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda c: c.fillna(c.mean()) if c.dtype.kind in "fc" else c)

def default_pick(items: List[str], k: int = 2) -> List[str]:
    return items[: min(k, len(items))] if items else []

# -----------------------------------------------------------------------------------------
# Sidebar — Data Source
# -----------------------------------------------------------------------------------------

st.sidebar.title("📦 Dataset")

use_fallback = st.sidebar.checkbox("Use fallback data", value=True)
uploaded = st.sidebar.file_uploader("Upload spreadsheet", type=["xlsx", "xls", "csv"])

if uploaded or use_fallback:
    if uploaded:
        df = pd.read_excel(uploaded) if uploaded.name.endswith("xls") else pd.read_csv(uploaded)
        log_step(f"Loaded uploaded file: {uploaded.name}")
    else:
        df = pd.read_excel("stores/excel/Combined Schedules.xlsx")
        log_step("Loaded fallback dataset")

    st.session_state.raw_df = df.copy()
    st.session_state.df = df.copy()
    st.session_state.numeric_cols, st.session_state.categorical_cols = detect_column_types(df)

# -----------------------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------------------

tabs = st.tabs([
    "🧹 Data Processing",
    "📈 Descriptive Statistics",
    "📐 Inferential Statistics"
])

# =========================================================================================
# TAB 1 — DATA PROCESSING (FULLY RESTORED)
# =========================================================================================

with tabs[0]:
    st.header("🧹 Data Processing")

    if st.session_state.df is None:
        st.info("No data loaded.")
        st.stop()

    df = st.session_state.df

    st.subheader("Current Dataset")
    render_table(df)
    blue_divider()

    # --- Drop / Rename
    c1, c2 = st.columns(2)

    with c1:
        drop_cols = st.multiselect("Columns to drop", df.columns.tolist())
        if st.button("Apply Column Drop"):
            df = df.drop(columns=drop_cols)
            st.session_state.df = df
            st.session_state.numeric_cols, st.session_state.categorical_cols = detect_column_types(df)
            log_step(f"Dropped columns: {drop_cols}")

    with c2:
        rename_col = st.selectbox("Rename column", ["<None>"] + df.columns.tolist())
        new_name = st.text_input("New column name")
        if st.button("Apply Rename") and rename_col != "<None>" and new_name:
            df = df.rename(columns={rename_col: new_name})
            st.session_state.df = df
            st.session_state.numeric_cols, st.session_state.categorical_cols = detect_column_types(df)
            log_step(f"Renamed {rename_col} → {new_name}")

    blue_divider()

    # --- Imputation / Scaling
    c3, c4 = st.columns(2)

    with c3:
        if st.button("Impute Missing Values"):
            if st.session_state.numeric_cols:
                imp = SimpleImputer(strategy="mean")
                df[st.session_state.numeric_cols] = imp.train_transform(
                    df[st.session_state.numeric_cols], None
                )
            if st.session_state.categorical_cols:
                imp = SimpleImputer(strategy="most_frequent")
                df[st.session_state.categorical_cols] = imp.train_transform(
                    df[st.session_state.categorical_cols], None
                )
            st.session_state.df = df
            log_step("Applied imputation")

    with c4:
        scaler_name = st.selectbox(
            "Scaler",
            ["None", "Standard", "MinMax", "Robust", "Normalize"]
        )
        if st.button("Apply Scaling") and scaler_name != "None":
            scaler = {
                "Standard": StandardScaler,
                "MinMax": MinMaxScaler,
                "Robust": RobustScaler,
                "Normalize": NormalScaler
            }[scaler_name]()
            if st.session_state.numeric_cols:
                df[st.session_state.numeric_cols] = scaler.train_transform(
                    df[st.session_state.numeric_cols]
                )
            st.session_state.df = df
            log_step(f"Applied {scaler_name} scaling")

    blue_divider()

    # --- Reset / Export
    c5, c6 = st.columns(2)

    with c5:
        if st.button("Reset to Raw Data"):
            st.session_state.df = st.session_state.raw_df.copy()
            st.session_state.numeric_cols, st.session_state.categorical_cols = detect_column_types(
                st.session_state.df
            )
            log_step("Reset dataset to raw")

    with c6:
        st.download_button(
            "Export Processed Data (CSV)",
            st.session_state.df.to_csv(index=False),
            "processed_data.csv",
            "text/csv"
        )

    st.subheader("Pipeline Log")
    for step in st.session_state.pipeline_log:
        st.write(f"• {step}")

# =========================================================================================
# TAB 2 — DESCRIPTIVE STATISTICS (NO HEIGHT-CONSTRAINED PLOTS)
# =========================================================================================

with tabs[1]:
    st.header("📈 Descriptive Statistics")

    df = st.session_state.df
    num_df = clean_numeric(df.select_dtypes(include=[np.number]))
    if num_df.empty:
        st.stop()

    all_num_cols = num_df.columns.tolist()

    vars_sel = st.multiselect(
        "Select numeric variables",
        all_num_cols,
        default=default_pick(all_num_cols, 3)
    )

    for col in vars_sel:
        s = num_df[col].dropna()

        st.subheader(f"Distribution & Shape — {col}")
        blue_divider()

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(s, bins=30, edgecolor="black", alpha=0.85)
            ax.set_title(f"Histogram — {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(7, 5))
            stats.probplot(s, plot=ax)
            ax.set_title(f"Q–Q Plot — {col}")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.subheader("Correlation Structure")
    blue_divider()

    corr_vars = st.multiselect(
        "Variables for correlation",
        all_num_cols,
        default=default_pick(all_num_cols, 4)
    )

    if len(corr_vars) >= 2:
        corr_df = analysis_fillna_mean(num_df[corr_vars])
        corr = corr_df.corr()

        c3, c4 = st.columns(2)

        with c3:
            render_table(corr)

        with c4:
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(corr_vars)))
            ax.set_yticks(range(len(corr_vars)))
            ax.set_xticklabels(corr_vars, rotation=45, ha="right")
            ax.set_yticklabels(corr_vars)
            ax.set_title("Correlation Heatmap")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.subheader("Principal Component Analysis")
    blue_divider()

    pca_vars = st.multiselect(
        "Variables for PCA",
        all_num_cols,
        default=default_pick(all_num_cols, 4)
    )

    if len(pca_vars) >= 2:
        X = analysis_fillna_mean(num_df[pca_vars])
        n_comp = st.slider("Components", 2, min(6, len(pca_vars)), 3)

        Xs = SKStandardScaler().fit_transform(X)
        pca = PCA(n_components=n_comp).fit(Xs)

        expl = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(n_comp)],
            "Explained Variance (%)": pca.explained_variance_ratio_ * 100
        })

        c5, c6 = st.columns(2)

        with c5:
            render_table(expl)

        with c6:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(expl["Component"], expl["Explained Variance (%)"], edgecolor="black")
            ax.set_ylabel("% Variance Explained")
            ax.set_title("PCA Variance Explained")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# =========================================================================================
# TAB 3 — INFERENTIAL STATISTICS
# =========================================================================================

with tabs[2]:
    st.header("📐 Inferential Statistics")

    df = st.session_state.df
    num_df = clean_numeric(df.select_dtypes(include=[np.number]))
    if num_df.empty:
        st.stop()

    vars_sel = st.multiselect(
        "Variables for inference",
        num_df.columns.tolist(),
        default=default_pick(num_df.columns.tolist(), 2)
    )

    for col in vars_sel:
        s = num_df[col].dropna()

        st.subheader(f"Inference — {col}")
        blue_divider()

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(7, 5))
            stats.probplot(s, plot=ax)
            ax.set_title(f"Q–Q Plot — {col}")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            sh, sp = stats.shapiro(s.sample(min(len(s), 500)))
            render_table(pd.DataFrame({
                "Test": ["Shapiro–Wilk"],
                "Statistic": [sh],
                "p-value": [sp]
            }))

    st.subheader("Power Analysis")
    blue_divider()

    effect_sizes = st.multiselect(
        "Effect sizes (Cohen's d)",
        [0.2, 0.5, 0.8, 1.0],
        default=[0.5]
    )

    base_n = len(num_df[vars_sel[0]].dropna())
    power_model = TTestPower()

    rows = [
        {
            "Effect Size": d,
            "Power": power_model.power(effect_size=d, nobs=base_n, alpha=0.05)
        }
        for d in effect_sizes
    ]

    c7, c8 = st.columns(2)

    with c7:
        render_table(pd.DataFrame(rows))

    with c8:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(
            [r["Effect Size"] for r in rows],
            [r["Power"] for r in rows],
            marker="o"
        )
        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.set_ylabel("Power")
        ax.set_ylim(0, 1.05)
        ax.set_title("Power Curve")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
