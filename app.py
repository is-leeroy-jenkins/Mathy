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
# ******************************************************************************************
# Assembly:                Mathy-Py
# Filename:                app.py
# Author:                  Terry D. Eppler (integration)
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

def log_step(msg: str) -> None:
    st.session_state.pipeline_log.append(msg)


def render_table(df: pd.DataFrame, height: int = 360) -> None:
    disp = df.copy()
    float_cols = disp.select_dtypes(include=[np.floating]).columns
    disp[float_cols] = disp[float_cols].round(4)
    st.dataframe(disp, use_container_width=True, height=height)


def blue_divider() -> None:
    st.markdown(
        "<div style='height:3px;background:#1f77b4;margin:1rem 0;'></div>",
        unsafe_allow_html=True
    )

def is_identifier(col: str) -> bool:
    col = col.lower()
    return any(k in col for k in ("id", "key", "index", "symbol"))

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

# -----------------------------------------------------------------------------------------
# Sidebar – Data Source
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
# Tabs (TOP LEVEL)
# -----------------------------------------------------------------------------------------

tabs = st.tabs([
    "🧹 Data Processing",
    "📈 Descriptive Statistics",
    "📐 Inferential Statistics"
])

# =========================================================================================
# TAB 1 — DATA PROCESSING
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

    CRUD_ROW_HEIGHT = 260

    # ---------------------------------------------------------------------
    # Structure
    # ---------------------------------------------------------------------

    st.subheader("Structure")
    c1, c2 = st.columns(2)

    with c1:
        with st.container(height=CRUD_ROW_HEIGHT):
            st.markdown("**Drop Columns**")
            drop_cols = st.multiselect(
                "Columns to drop",
                options=df.columns.tolist(),
                key="drop_cols"
            )
            if st.button("Apply Column Drop", key="btn_drop"):
                df = df.drop(columns=drop_cols)
                st.session_state.df = df
                st.session_state.numeric_cols, st.session_state.categorical_cols = (
                    detect_column_types(df)
                )
                log_step(f"Dropped columns: {drop_cols}")

    with c2:
        with st.container(height=CRUD_ROW_HEIGHT):
            st.markdown("**Rename Column**")
            rename_col = st.selectbox(
                "Column",
                ["<None>"] + df.columns.tolist(),
                key="rename_col"
            )
            new_name = st.text_input("New name", key="rename_name")
            if st.button("Apply Rename", key="btn_rename") and rename_col != "<None>" and new_name:
                df = df.rename(columns={rename_col: new_name})
                st.session_state.df = df
                st.session_state.numeric_cols, st.session_state.categorical_cols = (
                    detect_column_types(df)
                )
                log_step(f"Renamed {rename_col} → {new_name}")

    blue_divider()

    # ---------------------------------------------------------------------
    # Data Quality
    # ---------------------------------------------------------------------

    st.subheader("Data Quality")
    c3, c4 = st.columns(2)

    with c3:
        with st.container(height=CRUD_ROW_HEIGHT):
            st.markdown("**Imputation**")
            if st.button("Impute Missing Values", key="btn_impute"):
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
                st.session_state.numeric_cols, st.session_state.categorical_cols = (
                    detect_column_types(df)
                )
                log_step("Applied imputation")

    with c4:
        with st.container(height=CRUD_ROW_HEIGHT):
            st.markdown("**Scaling**")
            scaler_name = st.selectbox(
                "Scaler",
                ["None", "Standard", "MinMax", "Robust", "Normalize"],
                key="scaler"
            )
            if st.button("Apply Scaling", key="btn_scale") and scaler_name != "None":
                scaler_map = {
                    "Standard": StandardScaler,
                    "MinMax": MinMaxScaler,
                    "Robust": RobustScaler,
                    "Normalize": NormalScaler
                }
                scaler = scaler_map[scaler_name]()
                df[st.session_state.numeric_cols] = scaler.train_transform(
                    df[st.session_state.numeric_cols]
                )
                st.session_state.df = df
                log_step(f"Applied {scaler_name} scaling")

    blue_divider()

    # ---------------------------------------------------------------------
    # State
    # ---------------------------------------------------------------------

    st.subheader("State")
    c5, c6 = st.columns(2)

    with c5:
        with st.container(height=CRUD_ROW_HEIGHT):
            if st.button("Reset to Raw Data", key="btn_reset"):
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.numeric_cols, st.session_state.categorical_cols = (
                    detect_column_types(st.session_state.df)
                )
                log_step("Reset dataset to raw")

    with c6:
        with st.container(height=CRUD_ROW_HEIGHT):
            st.download_button(
                "Export Processed Data (CSV)",
                st.session_state.df.to_csv(index=False),
                "processed_data.csv",
                "text/csv",
                key="btn_export"
            )

    st.subheader("Pipeline Log")
    for step in st.session_state.pipeline_log:
        st.write(f"• {step}")

# =========================================================================================
# TAB 2 — DESCRIPTIVE STATISTICS
# =========================================================================================
    # =========================================================================
    # SECTION — STRUCTURAL RELATIONSHIPS & GEOMETRY (ROBUST)
    # =========================================================================

    st.subheader("Structural Relationships & Geometry")
    blue_divider()

    # -------------------------------------------------------------------------
    # Helpers (local, explicit)
    # -------------------------------------------------------------------------

    def is_identifier(col: str) -> bool:
        name = col.lower()
        return any(k in name for k in ("id", "key", "index", "symbol"))

    def clean_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Coerce numeric data to float and replace non-finite values with NaN."""
        out = frame.copy()
        out = out.replace([np.inf, -np.inf], np.nan)
        # Force float dtype where possible
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    # -------------------------------------------------------------------------
    # Prepare numeric-only frame (safe)
    # -------------------------------------------------------------------------

    num_only_raw = df.select_dtypes(include=[np.number])
    num_only = clean_numeric_frame(num_only_raw)

    # Drop columns that are entirely NaN
    num_only = num_only.dropna(axis=1, how="all")

    # Drop columns that are constant (no variance)
    nunique = num_only.nunique(dropna=True)
    num_only = num_only.loc[:, nunique > 1]

    if num_only.shape[1] < 2:
        st.info("At least two usable numeric columns are required for this section.")
    else:
        # =========================================================================
        # CORRELATION ANALYSIS
        # =========================================================================

        st.markdown("### Correlation Analysis")
        blue_divider()

        corr_method = st.selectbox(
            "Correlation method",
            ["Pearson", "Spearman", "Kendall"],
            key="corr_method"
        )

        corr_map = {
            "Pearson": "pearson",
            "Spearman": "spearman",
            "Kendall": "kendall"
        }

        corr = num_only.corr(method=corr_map[corr_method]).astype(float)

        c1, c2 = st.columns(2)

        with c1:
            with st.container(height=420):
                st.markdown("#### Correlation Matrix")
                render_table(corr)

        with c2:
            with st.container(height=420):
                st.markdown("#### Correlation Heatmap (NaNs masked)")

                corr_vals = corr.values
                mask = np.isnan(corr_vals)

                fig, ax = plt.subplots()
                im = ax.imshow(
                    np.ma.masked_array(corr_vals, mask),
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1
                )

                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticklabels(corr.columns)

                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{corr_method} Correlation Heatmap")
                st.pyplot(fig)

        # =========================================================================
        # MUTUAL INFORMATION (DEPENDENCY)
        # =========================================================================

        st.markdown("### Mutual Information (Dependency Analysis)")
        blue_divider()

        try:
            from sklearn.feature_selection import mutual_info_regression

            valid_targets = [c for c in num_only.columns if not is_identifier(c)]
            if len(valid_targets) < 1:
                st.info("No suitable numeric targets available for mutual information analysis.")
            else:
                target_col = st.selectbox(
                    "Target variable",
                    valid_targets,
                    key="mi_target"
                )

                X_full = num_only.drop(columns=[target_col])
                y_full = num_only[target_col]

                # MI cannot handle NaNs: drop rows where any selected column is missing
                mi_frame = pd.concat([X_full, y_full.rename("__target__")], axis=1).dropna(axis=0)
                if mi_frame.shape[0] < 10 or mi_frame.shape[1] < 3:
                    st.info("Not enough complete observations to compute mutual information reliably.")
                else:
                    X = mi_frame.drop(columns=["__target__"])
                    y = mi_frame["__target__"]

                    mi_scores = mutual_info_regression(X, y, random_state=42)

                    mi_df = (
                        pd.DataFrame({
                            "Feature": X.columns,
                            "Mutual Information": mi_scores
                        })
                        .sort_values("Mutual Information", ascending=False)
                        .reset_index(drop=True)
                    )

                    c3, c4 = st.columns(2)

                    with c3:
                        with st.container(height=420):
                            st.markdown("#### Mutual Information Scores")
                            render_table(mi_df)

                    with c4:
                        with st.container(height=420):
                            fig, ax = plt.subplots()
                            ax.barh(mi_df["Feature"], mi_df["Mutual Information"], edgecolor="black")
                            ax.invert_yaxis()
                            ax.set_xlabel("Mutual Information")
                            ax.set_title("Dependency Strength (Higher = Stronger Dependency)")

                            for i, v in enumerate(mi_df["Mutual Information"]):
                                ax.text(v, i, f"{v:.3f}", va="center")

                            st.pyplot(fig)

        except Exception as ex:
            st.error(f"Mutual information failed: {ex}")

        # =========================================================================
        # PCA (GEOMETRY OF VARIANCE)
        # =========================================================================

        st.markdown("### Principal Component Analysis (PCA)")
        blue_divider()

        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            # PCA cannot handle NaNs: drop rows with any missing values across numeric columns
            pca_frame = num_only.dropna(axis=0)

            if pca_frame.shape[0] < 10:
                st.info("Not enough complete observations for PCA (need more non-missing rows).")
            else:
                max_components = min(10, pca_frame.shape[1])
                if max_components < 2:
                    st.info("Not enough usable numeric variables for PCA.")
                else:
                    n_components = st.slider(
                        "Number of components",
                        min_value=2,
                        max_value=max_components,
                        value=min(5, max_components),
                        key="pca_components"
                    )

                    X_scaled = StandardScaler().fit_transform(pca_frame.values)
                    pca = PCA(n_components=n_components)
                    _ = pca.fit_transform(X_scaled)

                    explained_df = pd.DataFrame({
                        "Component": [f"PC{i+1}" for i in range(n_components)],
                        "Explained Variance (%)": pca.explained_variance_ratio_ * 100
                    })

                    loadings_df = pd.DataFrame(
                        pca.components_.T,
                        index=pca_frame.columns,
                        columns=[f"PC{i+1}" for i in range(n_components)]
                    )

                    c5, c6 = st.columns(2)

                    with c5:
                        with st.container(height=420):
                            st.markdown("#### Variance Explained")
                            render_table(explained_df)

                            fig, ax = plt.subplots()
                            ax.bar(
                                explained_df["Component"],
                                explained_df["Explained Variance (%)"],
                                edgecolor="black"
                            )
                            ax.set_ylabel("% Variance")
                            ax.set_title("PCA Variance Explained")
                            st.pyplot(fig)

                    with c6:
                        with st.container(height=420):
                            st.markdown("#### Component Loadings")
                            render_table(loadings_df)

        except Exception as ex:
            st.error(f"PCA failed: {ex}")


# =========================================================================================
# TAB 3 — INFERENTIAL STATISTICS
# =========================================================================================

with tabs[2]:
    st.header("📐 Inferential Statistics")

    df = st.session_state.df
    num_df = df.select_dtypes(include=[np.number])

    if num_df.empty:
        st.info("No numeric columns available.")
        st.stop()

    col = st.selectbox(
        "Select numeric column",
        num_df.columns.tolist(),
        key="inf_col"
    )

    s = num_df[col].dropna()
    ROW_HEIGHT = 420

    # =========================================================================
    # SECTION 1 — NORMALITY & DISTRIBUTIONAL ASSUMPTIONS
    # =========================================================================

    st.subheader("Distributional Assumptions")
    blue_divider()

    c1, c2 = st.columns(2)

    with c1:
        with st.container(height=ROW_HEIGHT):
            fig, ax = plt.subplots()
            stats.probplot(s, plot=ax)
            ax.set_title("Q–Q Plot (Normal Reference)")
            st.pyplot(fig)

    with c2:
        with st.container(height=ROW_HEIGHT):
            shapiro_stat, shapiro_p = stats.shapiro(s.sample(min(len(s), 500)))
            dag_stat, dag_p = stats.normaltest(s)
            anderson = stats.anderson(s)

            render_table(pd.DataFrame({
                "Test": [
                    "Shapiro–Wilk",
                    "D’Agostino–Pearson",
                    "Anderson–Darling (stat)"
                ],
                "Statistic": [
                    shapiro_stat,
                    dag_stat,
                    anderson.statistic
                ],
                "p-value / Critical": [
                    shapiro_p,
                    dag_p,
                    anderson.critical_values[2]  # 5% level
                ]
            }))

    # =========================================================================
    # SECTION 2 — CONFIDENCE INTERVALS (PARAMETRIC & ROBUST)
    # =========================================================================

    st.subheader("Confidence Intervals")
    blue_divider()

    alpha = st.selectbox(
        "Confidence Level",
        [0.90, 0.95, 0.99],
        index=1,
        key="ci_level"
    )

    c3, c4 = st.columns(2)

    with c3:
        with st.container(height=ROW_HEIGHT):
            mean = s.mean()
            sem = stats.sem(s)
            ci_low, ci_high = stats.t.interval(
                alpha,
                len(s) - 1,
                loc=mean,
                scale=sem
            )

            render_table(pd.DataFrame({
                "Metric": ["Mean", "CI Lower", "CI Upper"],
                "Value": [mean, ci_low, ci_high]
            }))

    with c4:
        with st.container(height=ROW_HEIGHT):
            # Bootstrap CI for median
            rng = np.random.default_rng(42)
            boot = rng.choice(s.values, size=(2000, len(s)), replace=True)
            medians = np.median(boot, axis=1)
            lo, hi = np.percentile(
                medians,
                [(1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100]
            )

            render_table(pd.DataFrame({
                "Metric": ["Median", "Bootstrap CI Lower", "Bootstrap CI Upper"],
                "Value": [s.median(), lo, hi]
            }))

    # =========================================================================
    # SECTION 3 — ONE-SAMPLE HYPOTHESIS TESTING
    # =========================================================================

    st.subheader("One-Sample Hypothesis Tests")
    blue_divider()

    mu0 = st.number_input(
        "Null hypothesis mean (μ₀)",
        value=float(s.mean()),
        key="mu0"
    )

    c5, c6 = st.columns(2)

    with c5:
        with st.container(height=ROW_HEIGHT):
            t_stat, t_p = stats.ttest_1samp(s, mu0)

            render_table(pd.DataFrame({
                "Test": ["One-Sample t-test"],
                "Statistic": [t_stat],
                "p-value": [t_p]
            }))

    with c6:
        with st.container(height=ROW_HEIGHT):
            # Effect size: Cohen's d
            d = (s.mean() - mu0) / s.std()

            render_table(pd.DataFrame({
                "Metric": ["Cohen’s d"],
                "Value": [d]
            }))

    # =========================================================================
    # SECTION 4 — TWO-GROUP COMPARISONS
    # =========================================================================

    st.subheader("Two-Group Comparisons")
    blue_divider()

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if cat_cols:
        group_col = st.selectbox(
            "Grouping column (categorical)",
            cat_cols,
            key="inf_group_col"
        )

        groups = df[[group_col, col]].dropna().groupby(group_col)[col]

        if groups.ngroups == 2:
            g1, g2 = [g.values for _, g in groups]

            c7, c8 = st.columns(2)

            with c7:
                with st.container(height=ROW_HEIGHT):
                    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                    u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")

                    render_table(pd.DataFrame({
                        "Test": [
                            "Welch’s t-test",
                            "Mann–Whitney U"
                        ],
                        "Statistic": [t_stat, u_stat],
                        "p-value": [p_val, u_p]
                    }))

            with c8:
                with st.container(height=ROW_HEIGHT):
                    pooled_sd = np.sqrt((np.var(g1) + np.var(g2)) / 2)
                    d = (np.mean(g1) - np.mean(g2)) / pooled_sd

                    render_table(pd.DataFrame({
                        "Metric": ["Cohen’s d (group diff)"],
                        "Value": [d]
                    }))
        else:
            st.info("Two-group tests require exactly two categories.")
    else:
        st.info("No categorical columns available for grouping.")

    # =========================================================================
    # SECTION 5 — CORRELATION & ASSOCIATION
    # =========================================================================

    st.subheader("Correlation & Association")
    blue_divider()

    c9, c10 = st.columns(2)

    with c9:
        with st.container(height=ROW_HEIGHT):
            corr = df.select_dtypes(include=[np.number]).corr(method="pearson")
            render_table(corr)

    with c10:
        with st.container(height=ROW_HEIGHT):
            spearman = df.select_dtypes(include=[np.number]).corr(method="spearman")
            render_table(spearman)

