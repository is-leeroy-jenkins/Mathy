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
# TAB — CLASSIFICATION MODELS (CORRECTED COLUMN LOGIC, DOMAIN-AWARE)
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
    from sklearn.metrics import accuracy_score, confusion_matrix
    import seaborn as sns

    df = st.session_state.df
    if df is None or df.empty:
        st.info("No dataset loaded.")
        st.stop()

    # ------------------------------------------------------------------
    # Improved column typing logic
    # ------------------------------------------------------------------
    def detect_columns(frame: pd.DataFrame):
        numeric_hints = (
            "amount", "total", "value", "balance", "outlay", "obligation", "expenditure"
        )
        categorical_hints = (
            "fy", "py", "cy", "by", "code", "id", "name", "type", "line",
            "account", "tas", "fund", "bureau", "symbol"
        )

        num_cols, cat_cols = [], []

        for col in frame.columns:
            col_l = col.lower()
            dtype = frame[col].dtype

            if pd.api.types.is_float_dtype(dtype):
                num_cols.append(col)
            elif any(h in col_l for h in numeric_hints):
                num_cols.append(col)
            elif any(h in col_l for h in categorical_hints):
                cat_cols.append(col)
            elif pd.api.types.is_integer_dtype(dtype):
                # integers treated as categorical if code-like
                cat_cols.append(col)
            elif pd.api.types.is_object_dtype(dtype):
                cat_cols.append(col)
            else:
                cat_cols.append(col)

        # Deduplicate just in case
        num_cols = [c for c in num_cols if c not in cat_cols]
        return num_cols, cat_cols

    numeric_cols, categorical_cols = detect_columns(df)

    if not numeric_cols:
        st.warning("No numeric columns detected. Please verify dataset or preprocessing.")
        st.stop()
    if not categorical_cols:
        st.warning("No categorical columns detected. Please verify dataset or preprocessing.")
        st.stop()

    # ------------------------------------------------------------------
    # Target & Features
    # ------------------------------------------------------------------
    st.subheader("Target & Features")
    blue_divider()

    c1, c2 = st.columns(2)
    with c1:
        target = st.selectbox("Target variable (categorical)", categorical_cols, key="clf_target")
    with c2:
        features = st.multiselect(
            "Numeric feature variables",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
            key="clf_features"
        )

    if not features:
        st.info("Select at least one numeric feature to continue.")
        st.stop()

    X = df[features].copy()
    y = df[target].copy()

    if y.nunique() < 2:
        st.error(f"Target '{target}' contains only one unique class.")
        st.stop()

    # ------------------------------------------------------------------
    # Split & Scale
    # ------------------------------------------------------------------
    test_size = st.slider("Test size", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random seed", value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=int(random_state),
        stratify=y if y.value_counts().min() >= 2 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    st.subheader("Model Selection & Training")
    blue_divider()

    models_selected = st.multiselect(
        "Select models to train",
        ["Logistic Regression", "k-Nearest Neighbors", "Support Vector Classifier", "Decision Tree", "Random Forest"],
        default=["Logistic Regression", "Random Forest"]
    )

    if not models_selected:
        st.stop()

    run = st.button("Run Classification Models", type="primary")
    if not run:
        st.stop()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    results, fitted = [], {}
    for name in models_selected:
        if name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
            Xtr, Xte = X_train_s, X_test_s
        elif name == "k-Nearest Neighbors":
            model = KNeighborsClassifier(n_neighbors=5)
            Xtr, Xte = X_train_s, X_test_s
        elif name == "Support Vector Classifier":
            model = SVC()
            Xtr, Xte = X_train_s, X_test_s
        elif name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=int(random_state))
            Xtr, Xte = X_train, X_test
        elif name == "Random Forest":
            model = RandomForestClassifier(n_estimators=200, random_state=int(random_state))
            Xtr, Xte = X_train, X_test
        else:
            continue

        model.fit(Xtr, y_train)
        fitted[name] = model
        results.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, model.predict(Xtr)),
            "Test Accuracy": accuracy_score(y_test, model.predict(Xte))
        })

    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
    render_table(results_df)

    # ------------------------------------------------------------------
    # Confusion Matrices
    # ------------------------------------------------------------------
    st.subheader("Confusion Matrices")
    blue_divider()

    max_classes = 25
    labels = y.unique()[:max_classes]
    short_labels = [str(l)[:20] + ("…" if len(str(l)) > 20 else "") for l in labels]

    cols = st.columns(2)
    for i, (name, model) in enumerate(fitted.items()):
        col = cols[i % 2]
        with col:
            st.markdown(f"**{name}**")
            is_tree = name in ("Decision Tree", "Random Forest")
            Xte = X_test if is_tree else X_test_s
            preds = model.predict(Xte)
            cm = confusion_matrix(y_test, preds, labels=labels)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_norm, ax=ax, cmap="Blues", cbar=False)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{name} (normalized)")
            ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(short_labels, rotation=0, fontsize=7)
            st.pyplot(fig)
            plt.close(fig)

