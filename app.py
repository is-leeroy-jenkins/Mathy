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

with tabs[1]:
    st.header("📈 Descriptive Statistics")

    df = st.session_state.df
    num_df = df.select_dtypes(include=[np.number])

    if num_df.empty:
        st.info("No numeric columns available.")
        st.stop()

    col = st.selectbox(
        "Select numeric column",
        num_df.columns.tolist(),
        key="desc_col"
    )

    s = num_df[col].dropna()
    ROW_HEIGHT = 420

    # =========================================================================
    # SECTION 1 — DISTRIBUTION & LOCATION
    # =========================================================================

    st.subheader("Distribution & Location")
    blue_divider()

    c1, c2 = st.columns(2)

    with c1:
        with st.container(height=ROW_HEIGHT):
            fig, ax = plt.subplots()
            ax.hist(
                s,
                bins=30,
                edgecolor="black",
                alpha=0.85
            )
            ax.axvline(s.mean(), color="red", linestyle="--", label="Mean")
            ax.axvline(s.median(), color="green", linestyle=":", label="Median")
            ax.set_title("Histogram with Mean & Median")
            ax.legend()
            st.pyplot(fig)

    with c2:
        with st.container(height=ROW_HEIGHT):
            fig, ax = plt.subplots()
            ax.boxplot(
                s,
                vert=True,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", edgecolor="black"),
                medianprops=dict(color="red"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                flierprops=dict(marker="o", markerfacecolor="orange", markersize=4)
            )
            ax.set_title("Boxplot with Outliers")
            st.pyplot(fig)

    render_table(pd.DataFrame({
        "Metric": [
            "Count",
            "Mean",
            "Median",
            "Trimmed Mean (10%)",
            "Min",
            "Max",
            "Range"
        ],
        "Value": [
            s.count(),
            s.mean(),
            s.median(),
            stats.trim_mean(s, 0.10),
            s.min(),
            s.max(),
            s.max() - s.min()
        ]
    }))

    # =========================================================================
    # SECTION 2 — DISPERSION & VARIABILITY
    # =========================================================================

    st.subheader("Dispersion & Variability")
    blue_divider()

    c3, c4 = st.columns(2)

    with c3:
        with st.container(height=ROW_HEIGHT):
            render_table(pd.DataFrame({
                "Metric": [
                    "Variance",
                    "Standard Deviation",
                    "Coefficient of Variation",
                    "Interquartile Range (IQR)",
                    "Median Absolute Deviation (MAD)"
                ],
                "Value": [
                    s.var(),
                    s.std(),
                    s.std() / s.mean() if s.mean() else np.nan,
                    s.quantile(0.75) - s.quantile(0.25),
                    stats.median_abs_deviation(s)
                ]
            }))

    with c4:
        with st.container(height=ROW_HEIGHT):
            fig, ax = plt.subplots()
            ax.violinplot(s, showmeans=True, showmedians=True)
            ax.set_title("Violin Plot (Distribution Density)")
            st.pyplot(fig)

    # =========================================================================
    # SECTION 3 — SHAPE, MOMENTS & DISTRIBUTIONAL DIAGNOSTICS
    # =========================================================================

    st.subheader("Shape, Moments & Distributional Diagnostics")
    blue_divider()

    c5, c6 = st.columns(2)

    with c5:
        with st.container(height=ROW_HEIGHT):
            fig, ax = plt.subplots()
            stats.probplot(s, plot=ax)
            ax.set_title("Q–Q Plot (Normal Reference)")
            st.pyplot(fig)

    with c6:
        with st.container(height=ROW_HEIGHT):
            render_table(pd.DataFrame({
                "Metric": [
                    "Skewness",
                    "Kurtosis (Fisher)",
                    "Kurtosis (Pearson)"
                ],
                "Value": [
                    s.skew(),
                    s.kurtosis(),
                    s.kurtosis() + 3
                ]
            }))

    # =========================================================================
    # SECTION 4 — OUTLIER PREVALENCE (DESCRIPTIVE ONLY)
    # =========================================================================

    st.subheader("Outlier Prevalence (Descriptive)")
    blue_divider()

    z_scores = np.abs(stats.zscore(s, nan_policy="omit"))
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    render_table(pd.DataFrame({
        "Method": [
            "Z-Score > 2",
            "Z-Score > 3",
            "IQR Fence"
        ],
        "Count": [
            int((z_scores > 2).sum()),
            int((z_scores > 3).sum()),
            int(((s < lower) | (s > upper)).sum())
        ],
        "Percent of Observations": [
            float((z_scores > 2).mean() * 100),
            float((z_scores > 3).mean() * 100),
            float(((s < lower) | (s > upper)).mean() * 100)
        ]
    }))

    # =========================================================================
    # SECTION 5 — GROUPED DESCRIPTIVES (FINANCE-CRITICAL)
    # =========================================================================

    st.subheader("Grouped Descriptive Statistics")
    blue_divider()

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if cat_cols:
        group_col = st.selectbox(
            "Group by (categorical)",
            cat_cols,
            key="desc_group_col"
        )

        grouped = (
            df[[group_col, col]]
            .dropna()
            .groupby(group_col)[col]
            .agg(
                count="count",
                mean="mean",
                median="median",
                std="std",
                min="min",
                max="max"
            )
            .reset_index()
        )

        render_table(grouped)

        c7, c8 = st.columns(2)

        with c7:
            with st.container(height=ROW_HEIGHT):
                fig, ax = plt.subplots()
                grouped.sort_values("mean").plot.bar(
                    x=group_col,
                    y="mean",
                    ax=ax,
                    legend=False,
                    edgecolor="black"
                )
                ax.set_title("Group Means")
                st.pyplot(fig)

        with c8:
            with st.container(height=ROW_HEIGHT):
                fig, ax = plt.subplots()
                df.boxplot(
                    column=col,
                    by=group_col,
                    ax=ax,
                    grid=False,
                    rot=45
                )
                ax.set_title("Grouped Boxplots")
                plt.suptitle("")
                st.pyplot(fig)
    else:
        st.info("No categorical columns available for grouping.")


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

