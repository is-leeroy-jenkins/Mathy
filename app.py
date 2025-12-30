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
import seaborn as sns

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
    st.header("")

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
    st.header("")

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
# TAB — INFERENTIAL STATISTICS (FIXED)
# =========================================================================================
with tabs[2]:
    st.header("📈 Inferential Statistics")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("⚠️ No dataset loaded. Please load or preprocess data first.")
    else:
        numeric_cols = st.session_state.get("numeric_cols", [])
        if not numeric_cols:
            st.info("No numeric columns available.")
        else:
            st.subheader("Correlation Matrix")
            corr = df[numeric_cols].corr()

            fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
            sns.heatmap(corr, cmap="coolwarm", annot=True, ax=ax)
            ax.set_title("Inferential Correlation Matrix", color="black")
            st.pyplot(fig)
    # --- ensure Inferential tab context closes cleanly ---
    st.markdown(
	    "<hr style='border: 1px solid #1f77b4; margin-top: 1rem;'>",
	    unsafe_allow_html=True
    )
    
    # =========================================================================================
    # TAB — INFERENTIAL STATISTICS (FINAL CONSOLIDATED VERSION)
    # =========================================================================================
    with tabs[2]:
	    st.header("📈 Inferential Statistics")
	    
	    import matplotlib.pyplot as plt
	    import seaborn as sns
	    import numpy as np
	    import pandas as pd
	    from scipy import stats
	    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
	    from sklearn.decomposition import PCA
	    
	    # --- Mathy Scaler Integration (Safe Wrapper)
	    try:
		    from scalers import StandardScaler
		    def scale_data(df):
			    return StandardScaler().train_transform(df)
	    except Exception:
		    from sklearn.preprocessing import StandardScaler as SkStandardScaler
		    def scale_data(df):
			    return SkStandardScaler().fit_transform(df)
	    
	    # --- Visualization Style
	    plt.style.use("default")
	    sns.set_theme(style="whitegrid")
	    plt.rcParams.update({"font.size": 9})
	    
	    st.markdown("""
            <style>
            .stDataFrame, .dataframe {
                background-color: #1e1e1e !important;
                color: #ddd !important;
                border: 1px solid #333 !important;
            }
            .dataframe th {
                background-color: #2a2a2a !important;
                color: #eee !important;
            }
            </style>
        """, unsafe_allow_html=True)
	    
	    # --- Data and Column Setup
	    df = st.session_state.get("df", None)
	    if df is None or df.empty:
		    st.warning("⚠️ No dataset loaded. Please load or preprocess data first.")
		    st.stop()
	    
	    numeric_cols = st.session_state.get("numeric_cols", [])
	    categorical_cols = st.session_state.get("categorical_cols", [])
	    
	    # =====================================================================
	    # CORRELATION ANALYSIS
	    # =====================================================================
	    st.subheader("Correlation and Association")
	    if numeric_cols:
		    corr = df[numeric_cols].corr(method="pearson")
		    
		    c1, c2 = st.columns(2)
		    with c1:
			    st.markdown("**Correlation Matrix (Pearson)**")
			    st.dataframe(corr.style.background_gradient(cmap="Greys"), use_container_width=True)
		    
		    with c2:
			    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
			    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
			    ax.set_title("Correlation Heatmap", fontsize=10)
			    st.pyplot(fig)
	    else:
		    st.info("No numeric columns available for correlation analysis.")
	    
	    st.divider()
	    
	    # =====================================================================
	    # NORMALITY TESTS
	    # =====================================================================
	    st.subheader("Normality Tests")
	    if numeric_cols:
		    results = []
		    for col in numeric_cols:
			    x = df[col].dropna()
			    if len(x) > 3:
				    shapiro_p = stats.shapiro(x.sample(min(500, len(x))))[1]
				    dagostino_p = stats.normaltest(x)[1]
				    results.append([col, shapiro_p, dagostino_p])
		    norm_df = pd.DataFrame(results, columns=["Variable", "Shapiro–Wilk p", "D’Agostino p"])
		    st.dataframe(norm_df.style.background_gradient(cmap="Greys"), use_container_width=True)
		    
		    # --- QQ Plot
		    sel_col = st.selectbox("Select variable for Q–Q Plot", numeric_cols)
		    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
		    stats.probplot(df[sel_col].dropna(), plot=ax)
		    ax.set_title(f"Q–Q Plot: {sel_col}")
		    st.pyplot(fig)
	    else:
		    st.info("No numeric variables for normality testing.")
	    
	    st.divider()
	    
	    # =====================================================================
	    # GROUP COMPARISON (ANOVA / KRUSKAL–WALLIS)
	    # =====================================================================
	    st.subheader("Group Comparison (ANOVA / Kruskal–Wallis)")
	    if categorical_cols and numeric_cols:
		    cat_for_anova = st.selectbox("Select categorical grouping variable", categorical_cols)
		    num_for_anova = st.selectbox("Select numeric variable", numeric_cols)
		    if cat_for_anova and num_for_anova:
			    groups = [vals[1].dropna().values for vals in df.groupby(cat_for_anova)[num_for_anova]]
			    if len(groups) > 1:
				    try:
					    f_stat, p_val = stats.f_oneway(*groups)
					    test_type = "One-way ANOVA"
				    except Exception:
					    f_stat, p_val = stats.kruskal(*groups)
					    test_type = "Kruskal–Wallis"
				    st.write(f"**{test_type} p-value:** {p_val:.4f}")
				    
				    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
				    sns.boxplot(data=df, x=cat_for_anova, y=num_for_anova, ax=ax)
				    ax.set_title(f"{test_type} by {cat_for_anova}", fontsize=10)
				    st.pyplot(fig)
			    else:
				    st.info("Selected categorical variable must have at least two groups.")
	    else:
		    st.info("Need both categorical and numeric variables for ANOVA/Kruskal–Wallis tests.")
	    
	    st.divider()
	    
	    # =====================================================================
	    # MUTUAL INFORMATION
	    # =====================================================================
	    st.subheader("Mutual Information (Variable Relevance)")
	    target = st.selectbox("Select target variable", df.columns)
	    features = [f for f in df.columns if f != target]
	    df_valid = df[features + [target]].dropna()
	    
	    if not df_valid.empty:
		    numeric_features = df_valid.select_dtypes(include=[np.number]).columns.tolist()
		    if target in numeric_features:
			    y = df_valid[target]
			    X = df_valid[numeric_features].drop(columns=[target], errors="ignore")
			    mi = mutual_info_regression(X, y)
		    else:
			    y = pd.factorize(df_valid[target])[0]
			    X = df_valid[numeric_features]
			    mi = mutual_info_classif(X, y)
		    
		    mi_df = pd.DataFrame({"Feature": X.columns, "MI Score": mi}).sort_values("MI Score", ascending=False)
		    
		    c1, c2 = st.columns(2)
		    with c1:
			    st.dataframe(mi_df.style.background_gradient(cmap="Greys"), use_container_width=True)
		    with c2:
			    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
			    sns.barplot(data=mi_df, x="MI Score", y="Feature", color="steelblue", ax=ax)
			    ax.set_title("Mutual Information Scores", fontsize=10)
			    st.pyplot(fig)
	    else:
		    st.info("Insufficient valid data for mutual information analysis.")
	    
	    st.divider()
	    
	    # =====================================================================
	    # PRINCIPAL COMPONENT ANALYSIS (PCA)
	    # =====================================================================
	    st.subheader("Principal Component Analysis (PCA)")
	    if len(numeric_cols) >= 2:
		    X = df[numeric_cols].dropna()
		    scaled_X = scale_data(X)
		    
		    pca = PCA(n_components=2)
		    comps = pca.fit_transform(scaled_X)
		    pca_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
		    
		    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
		    sns.scatterplot(data=pca_df, x="PC1", y="PC2", s=30, alpha=0.7, ax=ax)
		    exp = pca.explained_variance_ratio_ * 100
		    ax.set_title(f"PCA Biplot — Var Explained: PC1 {exp[0]:.1f}% · PC2 {exp[1]:.1f}%", fontsize=10)
		    st.pyplot(fig)
	    else:
		    st.info("Need at least two numeric variables for PCA.")
	    
	    st.divider()
	    
	    # =====================================================================
	    # CHI-SQUARE & CRAMÉR'S V
	    # =====================================================================
	    st.subheader("Categorical Association (Chi-square / Cramér’s V)")
	    if len(categorical_cols) >= 2:
		    cat_x = st.selectbox("Select variable X", categorical_cols, key="chi_x")
		    cat_y = st.selectbox("Select variable Y", [c for c in categorical_cols if c != cat_x], key="chi_y")
		    
		    ctab = pd.crosstab(df[cat_x], df[cat_y])
		    chi2, p, _, _ = stats.chi2_contingency(ctab)
		    cramers_v = np.sqrt(chi2 / (ctab.values.sum() * (min(ctab.shape) - 1)))
		    st.write(f"**Chi² p-value:** {p:.4f} | **Cramér’s V:** {cramers_v:.3f}")
		    
		    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
		    sns.heatmap(ctab, cmap="Blues", annot=True, fmt="d", ax=ax)
		    ax.set_title(f"Contingency Table: {cat_x} × {cat_y}", fontsize=10)
		    st.pyplot(fig)
	    else:
		    st.info("Need at least two categorical variables for Chi-square test.")

# =========================================================================================
# TAB 4 — ANOMALY DETECTION
# =========================================================================================

with tabs[3]:
    st.header("🚨 Anomaly Detection")

    if st.session_state.df is None:
        st.info("No data loaded.")
        st.stop()

    df = st.session_state.df

    # -------------------------------------------------------------------------
    # Prepare numeric data (analysis-only)
    # -------------------------------------------------------------------------

    num_df = clean_numeric(df.select_dtypes(include=[np.number]))

    if num_df.empty:
        st.info("No usable numeric columns available for anomaly detection.")
        st.stop()

    all_num_cols = num_df.columns.tolist()

    # Finance-aware default: prefer PY / CY / BY if present
    preferred = [c for c in all_num_cols if c.lower() in ("py", "cy", "by")]
    default_vars = preferred if preferred else default_pick(all_num_cols, 2)

    vars_sel = st.multiselect(
        "Variables to analyze",
        all_num_cols,
        default=default_vars
    )

    if not vars_sel:
        st.info("Select at least one numeric variable to run anomaly detection.")
        st.stop()

    analysis_scale = st.checkbox(
        "Use analysis-only standardization (recommended for multivariate methods)",
        value=True
    )

    # Analysis-only working frame
    work_df = num_df[vars_sel].copy()

    if analysis_scale and len(vars_sel) > 1:
        work_df[:] = SKStandardScaler().fit_transform(work_df.values)

    # -------------------------------------------------------------------------
    # Method Selection
    # -------------------------------------------------------------------------

    st.subheader("Detection Methods")
    blue_divider()

    c_m1, c_m2 = st.columns(2)

    with c_m1:
        use_z = st.checkbox("Z-Score", value=True)
        use_mz = st.checkbox("Modified Z-Score (MAD)", value=True)
        use_iqr = st.checkbox("IQR Fence", value=True)

    with c_m2:
        use_mahal = st.checkbox("Mahalanobis Distance", value=True)
        use_iforest = st.checkbox("Isolation Forest", value=True)
        use_lof = st.checkbox("Local Outlier Factor (LOF)", value=False)

    # -------------------------------------------------------------------------
    # Threshold Controls
    # -------------------------------------------------------------------------

    st.subheader("Thresholds")
    blue_divider()

    c_t1, c_t2 = st.columns(2)

    with c_t1:
        z_thresh = st.slider("Z / Modified Z threshold", 2.0, 5.0, 3.0, 0.1)
        iqr_mult = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)

    with c_t2:
        lof_k = st.slider("LOF neighbors (k)", 5, 50, 20, 1)
        min_methods = st.slider(
            "Consensus: minimum methods flagging a row",
            1, 4, 1, 1
        )

    # -------------------------------------------------------------------------
    # Run Detection
    # -------------------------------------------------------------------------

    anomaly_flags = pd.DataFrame(index=work_df.index)

    # --- Univariate methods
    for col in vars_sel:
        s = work_df[col].dropna()

        if s.empty:
            continue

        if use_z:
            z = (s - s.mean()) / s.std() if s.std() else pd.Series(0, index=s.index)
            anomaly_flags[f"{col}_z"] = z.abs() >= z_thresh

        if use_mz:
            med = s.median()
            mad = np.median(np.abs(s - med))
            if mad == 0:
                mz = pd.Series(0, index=s.index)
            else:
                mz = 0.6745 * (s - med) / mad
            anomaly_flags[f"{col}_mz"] = mz.abs() >= z_thresh

        if use_iqr:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo = q1 - iqr_mult * iqr
            hi = q3 + iqr_mult * iqr
            anomaly_flags[f"{col}_iqr"] = (s < lo) | (s > hi)

    # --- Multivariate methods
    mv_df = work_df.dropna(axis=0)

    if mv_df.shape[0] >= 10 and mv_df.shape[1] >= 2:

        if use_mahal:
            cov = np.cov(mv_df.values, rowvar=False)
            if np.linalg.det(cov) != 0:
                inv_cov = np.linalg.inv(cov)
                mean = mv_df.mean().values
                diffs = mv_df.values - mean
                md = np.sqrt(np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs))
                cutoff = np.sqrt(stats.chi2.ppf(0.975, mv_df.shape[1]))
                anomaly_flags.loc[mv_df.index, "mahal"] = md > cutoff

        if use_iforest:
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination="auto", random_state=42)
            preds = iso.fit_predict(mv_df.values)
            anomaly_flags.loc[mv_df.index, "iforest"] = preds == -1

        if use_lof:
            from sklearn.neighbors import LocalOutlierFactor
            lof = LocalOutlierFactor(n_neighbors=lof_k)
            preds = lof.fit_predict(mv_df.values)
            anomaly_flags.loc[mv_df.index, "lof"] = preds == -1

    # -------------------------------------------------------------------------
    # Consensus & Output
    # -------------------------------------------------------------------------

    st.subheader("Anomaly Summary")
    blue_divider()

    if anomaly_flags.empty:
        st.info("No anomalies detected under the selected methods and thresholds.")
        st.stop()

    anomaly_flags = anomaly_flags.fillna(False)
    anomaly_flags["methods_flagged"] = anomaly_flags.sum(axis=1)

    anomalies = anomaly_flags[anomaly_flags["methods_flagged"] >= min_methods]

    c_o1, c_o2 = st.columns(2)

    with c_o1:
        st.markdown("### Flagged Observations")
        render_table(anomalies.sort_values("methods_flagged", ascending=False))

    with c_o2:
        st.markdown("### Flag Count Distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        anomalies["methods_flagged"].value_counts().sort_index().plot(
            kind="bar", ax=ax, edgecolor="black"
        )
        ax.set_xlabel("Number of Methods Flagging")
        ax.set_ylabel("Observation Count")
        ax.set_title("Consensus Strength")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Visualization — Distribution with Anomalies
    # -------------------------------------------------------------------------

    st.subheader("Distributions with Anomalies Highlighted")
    blue_divider()

    for col in vars_sel:
        if col not in work_df.columns:
            continue

        s = work_df[col]
        flagged_idx = anomalies.index.intersection(s.index)

        if flagged_idx.empty:
            continue

        c_v1, c_v2 = st.columns(2)

        with c_v1:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(s.dropna(), bins=30, alpha=0.7, edgecolor="black")
            ax.scatter(
                s.loc[flagged_idx],
                np.zeros(len(flagged_idx)),
                color="red",
                label="Anomalies"
            )
            ax.set_title(f"{col} — Histogram with Anomalies")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c_v2:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.boxplot(s.dropna(), vert=False)
            ax.scatter(
                s.loc[flagged_idx],
                np.ones(len(flagged_idx)),
                color="red"
            )
            ax.set_title(f"{col} — Boxplot with Anomalies")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    st.download_button(
        "Export Anomaly Table (CSV)",
        anomalies.to_csv(),
        "anomalies.csv",
        "text/csv"
    )


# =========================================================================================
# TAB — CLASSIFICATION MODELS (NOTEBOOK-STYLE, SINGLE-MODEL EXPLORER)
# =========================================================================================
with tabs[4]:
    st.header("🧠 Classification Models")

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc,
        accuracy_score, precision_score, recall_score, f1_score
    )

    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except Exception:
        has_xgb = False

    # -------------------------------------------------------------------------------------
    # STYLING
    # -------------------------------------------------------------------------------------
    st.markdown("""
        <style>
        .stDataFrame, .dataframe {
            background-color: #1e1e1e !important;
            color: #ddd !important;
            border: 1px solid #333 !important;
        }
        .dataframe th {
            background-color: #2a2a2a !important;
            color: #eee !important;
        }
        </style>
    """, unsafe_allow_html=True)

    plt.style.use("default")
    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------------------------------
    # DATA VALIDATION
    # -------------------------------------------------------------------------------------
    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("⚠️ No dataset loaded. Please load or preprocess data first.")
        st.stop()

    numeric_cols = st.session_state.get("numeric_cols", [])
    categorical_cols = st.session_state.get("categorical_cols", [])
    if not numeric_cols or not categorical_cols:
        st.warning("⚠️ No column typing found. Please classify numeric/categorical columns first.")
        st.stop()

    # -------------------------------------------------------------------------------------
    # TARGET & FEATURES
    # -------------------------------------------------------------------------------------
    st.subheader("Target & Features")
    target = st.selectbox("Target (categorical)", categorical_cols)
    features = st.multiselect(
        "Feature columns (numeric)",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )

    if not features or target not in df.columns:
        st.info("Please select a target and at least one feature to continue.")
        st.stop()

    X = df[features].copy()
    y = df[target].copy()

    if y.nunique() < 2:
        st.error("Target must contain at least two classes for classification.")
        st.stop()

    # -------------------------------------------------------------------------------------
    # MODEL SELECTION
    # -------------------------------------------------------------------------------------
    model_options = [
        "Logistic Regression", "k-Nearest Neighbors", "Support Vector Classifier",
        "Decision Tree", "Random Forest", "Naive Bayes", "Gradient Boosting"
    ]
    if has_xgb:
        model_options.append("XGBoost")

    st.subheader("Model Selection")
    model_choice = st.selectbox("Select model", model_options)

    params = {}
    if model_choice == "Logistic Regression":
        params["C"] = st.slider("Inverse regularization strength (C)", 0.01, 10.0, 1.0)
        params["penalty"] = st.selectbox("Penalty", ["l2", "l1", "elasticnet"], index=0)
    elif model_choice == "k-Nearest Neighbors":
        params["n_neighbors"] = st.slider("Number of neighbors", 1, 50, 5)
    elif model_choice == "Support Vector Classifier":
        params["C"] = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
        params["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly"])
    elif model_choice == "Decision Tree":
        params["max_depth"] = st.slider("Max depth", 1, 50, 10)
    elif model_choice == "Random Forest":
        params["n_estimators"] = st.slider("Number of trees", 50, 500, 200, 50)
        params["max_depth"] = st.slider("Max depth", 1, 50, 10)
    elif model_choice == "Gradient Boosting":
        params["n_estimators"] = st.slider("Number of estimators", 50, 500, 100, 50)
        params["learning_rate"] = st.slider("Learning rate", 0.01, 1.0, 0.1)
    elif model_choice == "XGBoost":
        params["n_estimators"] = st.slider("Number of estimators", 50, 500, 100, 50)
        params["learning_rate"] = st.slider("Learning rate", 0.01, 1.0, 0.1)
        params["max_depth"] = st.slider("Max depth", 1, 20, 6)

    # -------------------------------------------------------------------------------------
    # TRAIN / TEST SPLIT
    # -------------------------------------------------------------------------------------
    test_size = st.slider("Test size", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random seed", value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # -------------------------------------------------------------------------------------
    # RUN MODEL
    # -------------------------------------------------------------------------------------
    if st.button("Run Model", type="primary"):
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, C=params["C"])
        elif model_choice == "k-Nearest Neighbors":
            model = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
        elif model_choice == "Support Vector Classifier":
            model = SVC(C=params["C"], kernel=params["kernel"], probability=True)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=params["max_depth"], random_state=int(random_state))
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=int(random_state)
            )
        elif model_choice == "Naive Bayes":
            model = GaussianNB()
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                random_state=int(random_state)
            )
        elif model_choice == "XGBoost":
            model = XGBClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                eval_metric="logloss",
                random_state=int(random_state)
            )

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else None

        # ---------------------------------------------------------------------------------
        # SCORING TABLE
        # ---------------------------------------------------------------------------------
        st.subheader("Model Scores")
        scores = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1 Score": f1_score(y_test, y_pred, average="weighted")
        }
        if y_prob is not None and len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            scores["ROC AUC"] = auc(fpr, tpr)
        else:
            scores["ROC AUC"] = np.nan

        score_df = pd.DataFrame(list(scores.items()), columns=["Metric", "Value"])
        st.dataframe(score_df.style.background_gradient(cmap="Greys"), use_container_width=True)

        # ---------------------------------------------------------------------------------
        # CONFUSION MATRIX
        # ---------------------------------------------------------------------------------
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(facecolor="white")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ---------------------------------------------------------------------------------
        # ROC CURVE (if binary)
        # ---------------------------------------------------------------------------------
        if y_prob is not None and len(np.unique(y)) == 2:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots(facecolor="white")
            ax.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {scores['ROC AUC']:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)

        # ---------------------------------------------------------------------------------
        # FEATURE IMPORTANCE
        # ---------------------------------------------------------------------------------
        st.subheader("Feature Importance / Coefficients")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = None

        if importances is not None:
            fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
            fi_df = fi_df.sort_values("Importance", ascending=False)
            fig, ax = plt.subplots(facecolor="white")
            sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax, color="steelblue")
            ax.set_title("Feature Importance")
            st.pyplot(fig)
        else:
            st.info("Selected model does not provide feature importance coefficients.")

