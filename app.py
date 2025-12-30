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
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans

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
    "📐 Inferential Statistics",
    "🚨 Anomaly Detection",
	"🧠 Classification Models"
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

# =========================================================================================
# TAB — ANOMALY DETECTION
# =========================================================================================

with tabs[3]:
    st.header("🚨 Anomaly Detection")

    # Local imports to prevent NameError / import-order issues
    from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler as SKStandardScaler

    if st.session_state.df is None:
        st.info("No data loaded.")
        st.stop()

    df = st.session_state.df
    num_df = clean_numeric(df.select_dtypes(include=[np.number]))

    if num_df.empty:
        st.info("No usable numeric data available.")
        st.stop()

    all_cols = num_df.columns.tolist()

    preferred = [c for c in all_cols if c.lower() in ("py", "cy", "by")]
    vars_sel = st.multiselect(
        "Variables to analyze",
        all_cols,
        default=preferred if preferred else default_pick(all_cols, 2)
    )

    if not vars_sel:
        st.info("Select at least one variable.")
        st.stop()

    scale_analysis = st.checkbox(
        "Use analysis-only standardization (recommended for multivariate methods)",
        value=True
    )

    # Analysis-only working frame (never mutate session df)
    work = num_df[vars_sel].copy()

    # IMPORTANT: Use sklearn StandardScaler (aliased) to avoid Mathy wrapper confusion
    if scale_analysis and len(vars_sel) > 1:
        # Fill NA for scaling/modeling only; do not change source df
        mv_scale = work.dropna(axis=0)
        if len(mv_scale) >= 2:
            scaled = SKStandardScaler().fit_transform(mv_scale.values)
            work.loc[mv_scale.index, :] = scaled

    st.subheader("Detection Methods")
    blue_divider()

    c1, c2 = st.columns(2)
    with c1:
        use_z = st.checkbox("Z-Score", True)
        use_mz = st.checkbox("Modified Z-Score (MAD)", True)
        use_iqr = st.checkbox("IQR Fence", True)
        use_knn = st.checkbox("k-NN Distance", False)

    with c2:
        use_lof = st.checkbox("Local Outlier Factor (LOF)", False)
        use_iforest = st.checkbox("Isolation Forest", True)
        use_ocsvm = st.checkbox("One-Class SVM", False)
        use_dbscan = st.checkbox("DBSCAN (Noise)", False)
        use_kmeans = st.checkbox("K-Means Distance", False)
        use_mahal = st.checkbox("Mahalanobis Distance", True)
    
    st.subheader( "Method Parameters" )
    blue_divider( )
    
    # --- Row 1: Statistical thresholds
    c1, c2 = st.columns( 2 )
    
    with c1:
	    with st.container( border=True ):
		    st.markdown( "**Statistical Thresholds**" )
		    z_thresh = st.slider(
			    "Z / Modified Z threshold",
			    2.0, 5.0, 3.0, 0.1,
			    key="z_thresh"
		    )
    
    with c2:
	    with st.container( border=True ):
		    st.markdown( "**IQR Fence**" )
		    iqr_mult = st.slider(
			    "IQR multiplier",
			    1.0, 3.0, 1.5, 0.1,
			    key="iqr_mult"
		    )
    
    # --- Row 2: Proximity methods
    c3, c4 = st.columns( 2 )
    
    with c3:
	    with st.container( border=True ):
		    st.markdown( "**k-NN Distance**" )
		    knn_k = st.slider(
			    "Neighbors (k)",
			    5, 50, 20,
			    key="knn_k"
		    )
		    knn_pct = st.slider(
			    "Distance percentile",
			    90, 99, 95,
			    key="knn_pct"
		    )
    
    with c4:
	    with st.container( border=True ):
		    st.markdown( "**Local Outlier Factor (LOF)**" )
		    lof_k = st.slider(
			    "Neighbors (k)",
			    5, 50, 20,
			    key="lof_k"
		    )
    
    # --- Row 3: Boundary & tree methods
    c5, c6 = st.columns( 2 )
    
    with c5:
	    with st.container( border=True ):
		    st.markdown( "**One-Class SVM**" )
		    oc_kernel = st.selectbox(
			    "Kernel",
			    [ "rbf",
			      "linear",
			      "poly" ],
			    key="oc_kernel"
		    )
		    oc_nu = st.slider(
			    "ν (outlier fraction)",
			    0.01, 0.25, 0.05,
			    key="oc_nu"
		    )
    
    with c6:
	    with st.container( border=True ):
		    st.markdown( "**Isolation Forest**" )
		    st.caption(
			    "Uses adaptive contamination; no exposed hyperparameters."
		    )
    
    # --- Row 4: Clustering-based methods
    c7, c8 = st.columns( 2 )
    
    with c7:
	    with st.container( border=True ):
		    st.markdown( "**DBSCAN (Noise Detection)**" )
		    db_eps = st.slider(
			    "eps",
			    0.1, 5.0, 0.5,
			    key="db_eps"
		    )
		    db_min = st.slider(
			    "min_samples",
			    5, 50, 10,
			    key="db_min"
		    )
    
    with c8:
	    with st.container( border=True ):
		    st.markdown( "**K-Means Distance**" )
		    km_k = st.slider(
			    "Clusters (k)",
			    2, 10, 4,
			    key="km_k"
		    )
		    km_pct = st.slider(
			    "Distance percentile",
			    90, 99, 95,
			    key="km_pct"
		    )
    
    # --- Row 5: Consensus logic
    with st.container( border=True ):
	    st.markdown( "**Consensus Scoring**" )
	    min_methods = st.slider(
		    "Minimum methods flagging a row",
		    1, 9, 1,
		    key="min_methods"
	    )
    
    # Flags table
    flags = pd.DataFrame(index=work.index)

    # -------------------------
    # Univariate methods
    # -------------------------
    for col in vars_sel:
        s = work[col].dropna()
        if s.empty:
            continue

        if use_z:
            std = s.std()
            z = (s - s.mean()) / std if std else pd.Series(0.0, index=s.index)
            flags.loc[s.index, f"{col}_z"] = z.abs() >= z_thresh

        if use_mz:
            med = s.median()
            mad = np.median(np.abs(s - med))
            if mad == 0:
                mz = pd.Series(0.0, index=s.index)
            else:
                mz = 0.6745 * (s - med) / mad
            flags.loc[s.index, f"{col}_mz"] = mz.abs() >= z_thresh

        if use_iqr:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
            flags.loc[s.index, f"{col}_iqr"] = (s < lo) | (s > hi)

    # -------------------------
    # Multivariate methods
    # -------------------------
    mv = work.dropna(axis=0)

    if mv.shape[0] >= 10 and mv.shape[1] >= 2:

        if use_knn and mv.shape[0] > knn_k:
            nn = NearestNeighbors(n_neighbors=knn_k).fit(mv.values)
            d, _ = nn.kneighbors(mv.values)
            kth = d[:, -1]
            cutoff = np.percentile(kth, knn_pct)
            flags.loc[mv.index, "knn"] = kth >= cutoff

        if use_lof:
            lof = LocalOutlierFactor(n_neighbors=lof_k)
            preds = lof.fit_predict(mv.values)
            flags.loc[mv.index, "lof"] = preds == -1

        if use_iforest:
            iso = IsolationForest(random_state=42, contamination="auto")
            preds = iso.fit_predict(mv.values)
            flags.loc[mv.index, "iforest"] = preds == -1

        if use_ocsvm:
            oc = OneClassSVM(kernel=oc_kernel, nu=oc_nu)
            preds = oc.fit_predict(mv.values)
            flags.loc[mv.index, "ocsvm"] = preds == -1

        if use_dbscan:
            db = DBSCAN(eps=db_eps, min_samples=db_min)
            labels = db.fit_predict(mv.values)
            flags.loc[mv.index, "dbscan"] = labels == -1

        if use_kmeans and mv.shape[0] >= km_k:
            km = KMeans(n_clusters=km_k, random_state=42, n_init="auto")
            labels = km.fit_predict(mv.values)
            centers = km.cluster_centers_
            dist = np.linalg.norm(mv.values - centers[labels], axis=1)
            cutoff = np.percentile(dist, km_pct)
            flags.loc[mv.index, "kmeans"] = dist >= cutoff

        if use_mahal:
            cov = np.cov(mv.values, rowvar=False)
            det = np.linalg.det(cov)
            if det != 0 and np.isfinite(det):
                inv = np.linalg.inv(cov)
                mean = mv.mean().values
                diff = mv.values - mean
                md = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
                cutoff = np.sqrt(stats.chi2.ppf(0.975, mv.shape[1]))
                flags.loc[mv.index, "mahal"] = md >= cutoff
            else:
                st.warning("Mahalanobis skipped: covariance matrix is singular or ill-conditioned.")

    # -------------------------
    # Consensus and output
    # -------------------------
    if flags.empty:
        st.info("No methods produced results with current settings.")
        st.stop()

    flags = flags.fillna(False)
    flags["methods_flagged"] = flags.sum(axis=1)

    anomalies = flags[flags["methods_flagged"] >= min_methods].copy()
    anomalies = anomalies.sort_values("methods_flagged", ascending=False)

    st.subheader("Anomaly Summary")
    blue_divider()

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### Flagged Observations")
        render_table(anomalies)

    with c4:
        st.markdown("### Consensus Strength")
        fig, ax = plt.subplots(figsize=(7, 5))
        anomalies["methods_flagged"].value_counts().sort_index().plot(
            kind="bar", edgecolor="black", ax=ax
        )
        ax.set_xlabel("Number of Methods Flagging")
        ax.set_ylabel("Count")
        ax.set_title("Consensus Distribution")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # -------------------------
    # Distributions with anomalies highlighted
    # -------------------------
    st.subheader("Distributions with Anomalies Highlighted")
    blue_divider()

    # Use original (unscaled) series for user interpretation where possible
    interpret = num_df[vars_sel].copy()

    for col in vars_sel:
        if col not in interpret.columns:
            continue

        s = interpret[col].dropna()
        if s.empty:
            continue

        flagged_idx = anomalies.index.intersection(s.index)
        if flagged_idx.empty:
            continue

        c5, c6 = st.columns(2)

        with c5:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(s, bins=30, alpha=0.75, edgecolor="black")
            ax.scatter(
                s.loc[flagged_idx],
                np.zeros(len(flagged_idx)),
                color="red",
                label="Anomalies"
            )
            ax.set_title(f"{col} — Histogram with Anomalies")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c6:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.boxplot(s, vert=False)
            ax.scatter(
                s.loc[flagged_idx],
                np.ones(len(flagged_idx)),
                color="red"
            )
            ax.set_title(f"{col} — Boxplot with Anomalies")
            ax.set_xlabel(col)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.download_button(
        "Export Anomaly Table (CSV)",
        anomalies.to_csv(index=True),
        "anomalies.csv",
        "text/csv"
    )

# =========================================================================================
# TAB — CLASSIFICATION MODELS (NOTEBOOK-ALIGNED, FED-FINANCE TYPING, SAFE DEFAULTS)
# =========================================================================================

with tabs[4]:
    st.header("🧠 Classification Models")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

    # -------------------------------
    # Data
    # -------------------------------
    df = st.session_state.df
    if df is None:
        st.info("No data loaded.")
        st.stop()

    # -------------------------------
    # Helpers (tab-local by request)
    # -------------------------------
    def _is_probably_categorical_numeric(col: str) -> bool:
        """
        Purpose:
            Determines whether a numeric-typed column is likely categorical in federal finance datasets.
        Parameters:
            col: Column name.
        Returns:
            True if the column name suggests categorical coding, otherwise False.
        """
        name = col.lower()
        categorical_hints = (
            "fy", "code", "name", "id", "identifier", "account", "mainaccount", "subaccount",
            "symbol", "tas", "line", "objectclass", "boc", "agency", "bureau", "fund"
        )
        return any(h in name for h in categorical_hints)

    def _true_numeric_candidates(frame: pd.DataFrame) -> list[str]:
        """
        Purpose:
            Returns numeric columns suitable for quantitative modeling (excludes coded identifiers).
        Parameters:
            frame: Input DataFrame.
        Returns:
            List of numeric feature column names.
        """
        numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if not _is_probably_categorical_numeric(c)]

    def _candidate_targets(frame: pd.DataFrame) -> list[str]:
        """
        Purpose:
            Returns classification target candidates aligned with notebook behavior while guarding
            against single-class targets and unsuitable high-cardinality numeric targets.
        Parameters:
            frame: Input DataFrame.
        Returns:
            List of target column names.
        """
        candidates: list[str] = []

        for c in frame.columns:
            nunq = frame[c].nunique(dropna=False)

            # Must have at least 2 classes
            if nunq < 2:
                continue

            # Object / category types are acceptable
            if frame[c].dtype == "object":
                candidates.append(c)
                continue

            # Low-cardinality integer-like targets can be acceptable, but avoid coded identifiers
            if pd.api.types.is_integer_dtype(frame[c]) and nunq <= 50 and not _is_probably_categorical_numeric(c):
                candidates.append(c)

        return candidates

    def _safe_default_multiselect(options: list[str], n: int) -> list[str]:
        """
        Purpose:
            Provides safe defaults for multiselect widgets without triggering rerun loops.
        Parameters:
            options: Available options.
            n: Desired default count.
        Returns:
            A safe default list (possibly empty).
        """
        if not options:
            return []
        return options[: min(n, len(options))]

    # -------------------------------
    # Target selection
    # -------------------------------
    st.subheader("Target Variable")
    blue_divider()

    targets = _candidate_targets(df)

    if not targets:
        st.warning(
            "No valid classification targets found. A target must contain at least two distinct classes."
        )
        st.stop()

    target = st.selectbox(
        "Select target",
        targets,
        index=0,
        key="clf_target"
    )

    y = df[target].copy()
    class_counts = y.value_counts(dropna=False)

    if class_counts.shape[0] < 2:
        # Defensive (should be filtered out already)
        st.error(
            f"Target '{target}' contains only one class ({class_counts.index[0]}). "
            "Classification requires at least two classes."
        )
        st.stop()

    # Show class distribution quickly (useful for Fed data)
    with st.container(border=True):
        st.markdown("**Class Distribution**")
        dist = class_counts.reset_index()
        dist.columns = ["Class", "Count"]
        render_table(dist, height=260)

    # -------------------------------
    # Feature selection (true numeric only)
    # -------------------------------
    st.subheader("Feature Variables")
    blue_divider()

    numeric_features = _true_numeric_candidates(df)

    if not numeric_features:
        st.warning(
            "No true numeric feature columns found after excluding coded identifiers "
            "(e.g., FY, Account, Code, Id)."
        )
        st.stop()

    default_features = _safe_default_multiselect(numeric_features, 2)

    features = st.multiselect(
        "Select numeric features",
        options=numeric_features,
        default=default_features,
        key="clf_features"
    )

    # Prevent rerun loops / “infinite loading”
    if not features:
        st.info("Select at least one numeric feature to proceed.")
        st.stop()

    X = df[features].copy()

    # -------------------------------
    # Train / Test split
    # -------------------------------
    st.subheader("Train / Test Split")
    blue_divider()

    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider(
            "Test size",
            0.1, 0.5, 0.25,
            key="clf_test_size"
        )
    with c2:
        random_state = st.number_input(
            "Random seed",
            value=42,
            step=1,
            key="clf_seed"
        )

    # Stratify only if every class has >= 2 samples
    use_stratify = (class_counts.min() >= 2)

    if not use_stratify:
        st.warning(
            "Stratified splitting disabled because at least one class occurs only once. "
            "Proceeding with a random split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=int(random_state),
        stratify=y if use_stratify else None
    )

    # Standardize numeric features for LR/SVC/kNN (tree models use raw X)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # -------------------------------
    # Model selection (notebook-aligned)
    # -------------------------------
    st.subheader("Models")
    blue_divider()

    model_choices = st.multiselect(
        "Select models",
        [
            "Logistic Regression",
            "k-Nearest Neighbors",
            "Support Vector Classifier",
            "Decision Tree",
            "Random Forest"
        ],
        default=["Logistic Regression", "Random Forest"],
        key="clf_models"
    )

    if not model_choices:
        st.info("Select at least one model.")
        st.stop()

    models: dict[str, object] = {}

    if "Logistic Regression" in model_choices:
        models["Logistic Regression"] = LogisticRegression(max_iter=1000)

    if "k-Nearest Neighbors" in model_choices:
        models["k-Nearest Neighbors"] = KNeighborsClassifier(n_neighbors=5)

    if "Support Vector Classifier" in model_choices:
        models["Support Vector Classifier"] = SVC()

    if "Decision Tree" in model_choices:
        models["Decision Tree"] = DecisionTreeClassifier(random_state=int(random_state))

    if "Random Forest" in model_choices:
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=200,
            random_state=int(random_state)
        )

    # -------------------------------
    # Training & evaluation (train vs test like notebook)
    # -------------------------------
    st.subheader("Model Performance")
    blue_divider()

    results = []

    for name, model in models.items():
        is_tree = name in ("Decision Tree", "Random Forest")

        Xtr = X_train if is_tree else X_train_s
        Xte = X_test if is_tree else X_test_s

        model.fit(Xtr, y_train)

        train_acc = accuracy_score(y_train, model.predict(Xtr))
        test_acc = accuracy_score(y_test, model.predict(Xte))

        results.append(
            {
                "Model": name,
                "Train Accuracy": train_acc,
                "Test Accuracy": test_acc
            }
        )

    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
    render_table(results_df)

    # -------------------------------
    # Confusion matrices
    # -------------------------------
    st.subheader("Confusion Matrices")
    blue_divider()

    # Limit visualizations to 2 per row
    model_names = list(models.keys())
    for i in range(0, len(model_names), 2):
        c_left, c_right = st.columns(2)
        pair = model_names[i:i + 2]

        for col, name in zip((c_left, c_right), pair):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{name}**")

                    model = models[name]
                    is_tree = name in ("Decision Tree", "Random Forest")
                    Xte = X_test if is_tree else X_test_s

                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay.from_estimator(
                        model,
                        Xte,
                        y_test,
                        ax=ax,
                        cmap="Blues",
                        colorbar=False
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

    # -------------------------------
    # Decision regions (your method, correctly gated)
    # -------------------------------
    st.subheader("Decision Regions (Optional)")
    blue_divider()

    if len(features) == 2:
        with st.container(border=True):
            show_regions = st.checkbox(
                "Show decision regions (requires exactly 2 numeric features)",
                value=False,
                key="clf_show_regions"
            )

        if show_regions:
            # Your provided method expects mlxtend.plotting.plot_decision_regions style usage.
            from mlxtend.plotting import plot_decision_regions

            # Only plot for models where regions are meaningful and stable.
            allowed = ("Logistic Regression", "k-Nearest Neighbors", "Support Vector Classifier", "Decision Tree")

            allowed_models = [m for m in model_names if m in allowed]
            if not allowed_models:
                st.info("No compatible models selected for decision regions.")
            else:
                for i in range(0, len(allowed_models), 2):
                    c_left, c_right = st.columns(2)
                    pair = allowed_models[i:i + 2]

                    for col, name in zip((c_left, c_right), pair):
                        with col:
                            with st.container(border=True):
                                st.markdown(f"**{name} — Decision Regions**")

                                model = models[name]

                                # Decision regions are plotted in standardized space.
                                # IMPORTANT: y must be integer-encoded for plot_decision_regions.
                                y_train_enc, uniques = pd.factorize(y_train, sort=True)

                                # Fit on the standardized 2D features.
                                model.fit(X_train_s, y_train_enc)

                                fig, ax = plt.subplots()
                                plot_decision_regions(
                                    X_train_s,
                                    y_train_enc,
                                    clf=model,
                                    ax=ax
                                )
                                ax.set_xlabel(features[0])
                                ax.set_ylabel(features[1])
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
    else:
        st.info("Decision regions require exactly two numeric features.")

    # -------------------------------
    # Export
    # -------------------------------
    st.download_button(
        "Export Classification Metrics (CSV)",
        results_df.to_csv(index=False),
        "classification_metrics.csv",
        "text/csv",
        key="clf_export_metrics"
    )
