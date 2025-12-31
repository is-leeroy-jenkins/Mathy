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

def init_state( ) -> None:
	defaults = {
			"raw_df": None,
			"df": None,
			"numeric_cols": [ ],
			"categorical_cols": [ ],
			"pipeline_log": [ ]
	}
	for k, v in defaults.items( ):
		if k not in st.session_state:
			st.session_state[ k ] = v

init_state( )

# -----------------------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------------------
def inferential_plot(
    title: str,
    subtitle: str | None = None,
    figsize: tuple[int, int] = (6, 4),
    grid: bool = True,
    ref_line: float | None = None,
    legend: bool = True
):
    """
    Purpose:
        Create a standardized matplotlib figure for inferential plots.

    Parameters:
        title: Main plot title.
        subtitle: Optional subtitle (e.g., test context).
        figsize: Figure size.
        grid: Whether to show background grid.
        ref_line: Optional horizontal reference line.
        legend: Whether to show legend.

    Returns:
        (fig, ax): Matplotlib figure and axis.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Grid (subtle)
    if grid:
        ax.grid(True, alpha=0.25, linewidth=0.8)

    # Titles
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    if subtitle:
        ax.text(
            0.5, 1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            alpha=0.85
        )

    # Reference line
    if ref_line is not None:
        ax.axhline(
            ref_line,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7
        )

    # Legend handling
    if not legend:
        ax.get_legend().remove()
    else:
        ax.legend(frameon=False)

    return fig, ax

def blue_divider( ) -> None:
	st.markdown(
		"<div style='height:3px;background:#1f77b4;margin:1rem 0;'></div>",
		unsafe_allow_html=True
	)

def log_step( msg: str ) -> None:
	st.session_state.pipeline_log.append( msg )

def render_table( df: pd.DataFrame, height: int = 360 ) -> None:
	disp = df.copy( )
	float_cols = disp.select_dtypes( include=[ np.floating ] ).columns
	disp[ float_cols ] = disp[ float_cols ].round( 4 )
	st.dataframe( disp, use_container_width=True, height=height )

def detect_column_types( df: pd.DataFrame ) -> tuple[ List[ str ], List[ str ] ]:
	numeric_hints = ("py", "cy", "by", "amount", "total", "value", "balance", "outlay")
	categorical_hints = ("fy", "code", "id", "name", "type", "symbol")
	
	numeric, categorical = [ ], [ ]
	
	for col in df.columns:
		name = col.lower( )
		if any( h in name for h in categorical_hints ):
			categorical.append( col )
		elif any( h in name for h in numeric_hints ):
			numeric.append( col )
		elif pd.api.types.is_float_dtype( df[ col ] ):
			numeric.append( col )
		elif pd.api.types.is_integer_dtype( df[ col ] ):
			numeric.append( col )
		else:
			categorical.append( col )
	
	return numeric, categorical

def clean_numeric( df: pd.DataFrame ) -> pd.DataFrame:
	out = df.replace( [ np.inf,
	                    -np.inf ], np.nan )
	for c in out.columns:
		out[ c ] = pd.to_numeric( out[ c ], errors="coerce" )
	out = out.dropna( axis=1, how="all" )
	out = out.loc[ :, out.nunique( dropna=True ) > 1 ]
	return out

def analysis_fillna_mean( df: pd.DataFrame ) -> pd.DataFrame:
	return df.apply( lambda c: c.fillna( c.mean( ) ) if c.dtype.kind in "fc" else c )

def default_pick( items: List[ str ], k: int = 2 ) -> List[ str ]:
	return items[ : min( k, len( items ) ) ] if items else [ ]

# -----------------------------------------------------------------------------------------
# Sidebar — Data Source
# -----------------------------------------------------------------------------------------

st.sidebar.title( "📦 Dataset" )

use_fallback = st.sidebar.checkbox( "Use default data", value=True )
uploaded = st.sidebar.file_uploader( "Upload spreadsheet", type=[ "xlsx",
                                                                  "xls",
                                                                  "csv" ] )

if uploaded or use_fallback:
	if uploaded:
		df = pd.read_excel( uploaded ) if uploaded.name.endswith( "xls" ) else pd.read_csv(
			uploaded )
		log_step( f"Loaded uploaded file: {uploaded.name}" )
	else:
		df = pd.read_excel( "stores/excel/Combined Schedules.xlsx" )
		log_step( "Loaded Default Dataset" )
	
	st.session_state.raw_df = df.copy( )
	st.session_state.df = df.copy( )
	st.session_state.numeric_cols, st.session_state.categorical_cols = detect_column_types( df )

# -----------------------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------------------

tabs = st.tabs( [ "💻 Dataset",
                  "🔍 Descriptive Stats",
                  "🧠 Inferential Stats",
                  "🚨 Outlier Analysis",
                  "🏗️ Feature Engineering",
                  "📊 Classifications",
                  "📉 Regressions",
                  "🕸️ Clustering",
                  "⏱️ Time-Series"
                  ] )

# =========================================================================================
# TAB 1 — DATASET (Schema-Aware CRUD + Diagnostics + Two-Column Row Editor)
# =========================================================================================

with tabs[0]:
    st.header("")

    if st.session_state.df is None:
        st.info("No data loaded.")
        st.stop()

    df = st.session_state.df

    # -------------------------------------------------------------------------------------
    # SCHEMA INFERENCE (ROBUST, DATETIME SAFE)
    # -------------------------------------------------------------------------------------
    def infer_schema( df: pd.DataFrame ) -> dict[ str, str ]:
	    schema: dict[ str, str ] = { }
	    n_rows = len( df )
	    
	    for col in df.columns:
		    s = df[ col ]
		    name = col.lower( )
		    
		    # Compute once
		    nunique = s.nunique( dropna=True )
		    unique_ratio = nunique / max( 1, n_rows )
		    
		    # ------------------------------------------------------------------
		    # 1) Datetime: ONLY for object/string columns (prevents PY/CY/BY errors)
		    # ------------------------------------------------------------------
		    if s.dtype == "object":
			    try:
				    parsed_dt = pd.to_datetime( s, errors="coerce" )
				    if parsed_dt.notna( ).sum( ) / max( 1, n_rows ) > 0.9:
					    schema[ col ] = "datetime"
					    continue
			    except Exception:
				    pass
		    
		    # ------------------------------------------------------------------
		    # 2) Numeric detection: ints AND floats
		    # ------------------------------------------------------------------
		    if pd.api.types.is_numeric_dtype( s ):
			    # Identifier heuristics for numeric codes/keys
			    if ("id" in name) or ("code" in name) or ("key" in name) or (unique_ratio > 0.8):
				    schema[ col ] = "identifier"
				    continue
			    
			    # Ordinal: small distinct set and integer-like
			    if pd.api.types.is_integer_dtype( s ) and nunique <= 20:
				    schema[ col ] = "ordinal"
				    continue
			    
			    # Otherwise treat as numeric (includes PY/CY/BY/OY-* int columns)
			    schema[ col ] = "numeric"
			    continue
		    
		    # ------------------------------------------------------------------
		    # 3) Categorical fallback
		    # ------------------------------------------------------------------
		    schema[ col ] = "categorical"
	    
	    return schema
    
    schema = infer_schema(df)
    st.session_state.column_schema = schema

    # Backward-compatible views
    st.session_state.numeric_cols = [c for c, t in schema.items() if t == "numeric"]
    st.session_state.categorical_cols = [c for c, t in schema.items() if t == "categorical"]

    # -------------------------------------------------------------------------------------
    # DATASET DISPLAY
    # -------------------------------------------------------------------------------------

    st.subheader("Current Dataset")
    render_table(df)
    blue_divider()

    # -------------------------------------------------------------------------------------
    # SCHEMA METRICS
    # -------------------------------------------------------------------------------------

    type_counts = pd.Series(schema).value_counts()

    m1, m2, m3, m4, m5 = st.columns(5, border=True)
    m1.metric("Rows", len(df))
    m2.metric("Numeric", type_counts.get("numeric", 0))
    m3.metric("Ordinal / ID",
              type_counts.get("ordinal", 0) + type_counts.get("identifier", 0))
    m4.metric("Categorical", type_counts.get("categorical", 0))
    m5.metric("Datetime", type_counts.get("datetime", 0))

    blue_divider()

    # =====================================================================================
    # ROW EDITOR — TWO-COLUMN, ALTERNATING FIELDS
    # =====================================================================================

    st.subheader("Row Editor")

    row_idx = st.number_input(
        "Select row index",
        min_value=0,
        max_value=len(df) - 1,
        step=1,
        key="row_editor_index"
    )

    row = df.iloc[row_idx]
    updated = {}

    col_left, col_right = st.columns(2)

    with st.form("row_edit_form"):
        for i, (col, dtype) in enumerate(schema.items()):
            target = col_left if i % 2 == 0 else col_right
            val = row[col]

            with target:
                if dtype == "numeric":
                    updated[col] = st.number_input(
                        col, value=float(val) if pd.notna(val) else 0.0
                    )
                elif dtype == "ordinal":
                    updated[col] = st.number_input(
                        col, value=int(val) if pd.notna(val) else 0
                    )
                elif dtype == "datetime":
                    updated[col] = st.date_input(
                        col,
                        value=pd.to_datetime(val).date()
                        if pd.notna(val)
                        else pd.Timestamp.today().date()
                    )
                elif dtype == "categorical":
                    options = df[col].dropna().unique().tolist()
                    updated[col] = st.selectbox(
                        col, options,
                        index=options.index(val) if val in options else 0
                    )
                else:  # identifier
                    updated[col] = st.text_input(
                        col, value=str(val), disabled=True
                    )

        submitted = st.form_submit_button("Apply Row Update")

    if submitted:
        before = df.loc[row_idx].copy()

        for col, value in updated.items():
            if schema[col] == "datetime":
                st.session_state.df.at[row_idx, col] = pd.to_datetime(value)
            else:
                st.session_state.df.at[row_idx, col] = value

        after = st.session_state.df.loc[row_idx]
        log_step(f"Updated row {row_idx}")

        st.success(f"Row {row_idx} updated.")
        st.dataframe(
            pd.DataFrame({"Before": before, "After": after}),
            use_container_width=True
        )
        st.experimental_rerun()

    blue_divider()

    # =====================================================================================
    # DIAGNOSTIC VISUALIZATIONS (TAB-1 APPROPRIATE)
    # =====================================================================================

    st.subheader("Dataset Diagnostics")

    v1, v2 = st.columns(2)

    with v1:
        fig, ax = plt.subplots(figsize=(5, 4))
        type_counts.plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title("Column Type Distribution")
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with v2:
        missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0].head(10)

        if not missing_pct.empty:
            fig, ax = plt.subplots(figsize=(5, 4))
            missing_pct.plot(kind="bar", ax=ax, edgecolor="black")
            ax.set_title("Top Columns by Missing %")
            ax.set_ylabel("Percent Missing")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No missing values detected.")

    blue_divider()

    v3, v4 = st.columns(2)

    with v3:
        cardinality = df.nunique(dropna=True).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        cardinality.plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title("Top Columns by Cardinality")
        ax.set_ylabel("Unique Values")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with v4:
        st.caption("Row edits are confirmed above before commit.")

    blue_divider()

    # -------------------------------------------------------------------------------------
    # COLUMN CRUD
    # -------------------------------------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:
        drop_cols = st.multiselect("Columns to drop", df.columns.tolist())
        if st.button("Apply Column Drop"):
            if len(drop_cols) == len(df.columns):
                st.error("Cannot drop all columns.")
            else:
                st.session_state.df = df.drop(columns=drop_cols)
                log_step(f"Dropped columns: {drop_cols}")
                st.experimental_rerun()

    with c2:
        rename_col = st.selectbox("Rename column", ["<None>"] + df.columns.tolist())
        new_name = st.text_input("New column name")
        if st.button("Apply Rename"):
            if rename_col != "<None>" and new_name:
                if new_name in df.columns:
                    st.error("Column name already exists.")
                else:
                    st.session_state.df = df.rename(columns={rename_col: new_name})
                    log_step(f"Renamed {rename_col} → {new_name}")
                    st.experimental_rerun()

    blue_divider()

    # -------------------------------------------------------------------------------------
    # RESET / EXPORT
    # -------------------------------------------------------------------------------------

    r1, r2 = st.columns(2)

    with r1:
        if st.button("Reset to Raw Data"):
            st.session_state.df = st.session_state.raw_df.copy()
            st.session_state.pipeline_log.clear()
            log_step("Reset dataset to raw")
            st.experimental_rerun()

    with r2:
        st.download_button(
            "Export Dataset (CSV)",
            st.session_state.df.to_csv(index=False),
            "dataset.csv",
            "text/csv"
        )

    st.subheader("Pipeline Log")
    for step in st.session_state.pipeline_log:
        st.write(f"• {step}")



# =========================================================================================
# TAB 2 — DESCRIPTIVE STATISTICS (NO HEIGHT-CONSTRAINED PLOTS)
# =========================================================================================

with tabs[ 1 ]:
	st.header( "" )
	
	df = st.session_state.df
	num_df = clean_numeric( df.select_dtypes( include=[ np.number ] ) )
	if num_df.empty:
		st.stop( )
	
	all_num_cols = num_df.columns.tolist( )
	
	vars_sel = st.multiselect(
		"Select numeric variables",
		all_num_cols,
		default=default_pick( all_num_cols, 3 )
	)
	
	for col in vars_sel:
		s = num_df[ col ].dropna( )
		
		st.subheader( f"Distribution & Shape — {col}" )
		blue_divider( )
		
		c1, c2 = st.columns( 2 )
		
		with c1:
			fig, ax = plt.subplots( figsize=(7, 5) )
			ax.hist( s, bins=30, edgecolor="black", alpha=0.85 )
			ax.set_title( f"Histogram — {col}" )
			ax.set_xlabel( col )
			ax.set_ylabel( "Frequency" )
			fig.tight_layout( )
			st.pyplot( fig, use_container_width=True )
			plt.close( fig )
		
		with c2:
			fig, ax = plt.subplots( figsize=(7, 5) )
			stats.probplot( s, plot=ax )
			ax.set_title( f"Q–Q Plot — {col}" )
			fig.tight_layout( )
			st.pyplot( fig, use_container_width=True )
			plt.close( fig )
	
	st.subheader( "Correlation Structure" )
	blue_divider( )
	
	corr_vars = st.multiselect(
		"Variables for correlation",
		all_num_cols,
		default=default_pick( all_num_cols, 4 )
	)
	
	if len( corr_vars ) >= 2:
		corr_df = analysis_fillna_mean( num_df[ corr_vars ] )
		corr = corr_df.corr( )
		
		c3, c4 = st.columns( 2 )
		
		with c3:
			render_table( corr )
		
		with c4:
			fig, ax = plt.subplots( figsize=(7, 6) )
			im = ax.imshow( corr.values, cmap="coolwarm", vmin=-1, vmax=1 )
			fig.colorbar( im, ax=ax )
			ax.set_xticks( range( len( corr_vars ) ) )
			ax.set_yticks( range( len( corr_vars ) ) )
			ax.set_xticklabels( corr_vars, rotation=45, ha="right" )
			ax.set_yticklabels( corr_vars )
			ax.set_title( "Correlation Heatmap" )
			fig.tight_layout( )
			st.pyplot( fig, use_container_width=True )
			plt.close( fig )
	
	st.subheader( "Principal Component Analysis" )
	blue_divider( )
	
	pca_vars = st.multiselect(
		"Variables for PCA",
		all_num_cols,
		default=default_pick( all_num_cols, 4 )
	)
	
	if len( pca_vars ) >= 2:
		X = analysis_fillna_mean( num_df[ pca_vars ] )
		n_comp = st.slider( "Components", 2, min( 6, len( pca_vars ) ), 3 )
		
		Xs = SKStandardScaler( ).fit_transform( X )
		pca = PCA( n_components=n_comp ).fit( Xs )
		
		expl = pd.DataFrame( {
				"Component": [ f"PC{i + 1}" for i in range( n_comp ) ],
				"Explained Variance (%)": pca.explained_variance_ratio_ * 100
		} )
		
		c5, c6 = st.columns( 2 )
		
		with c5:
			render_table( expl )
		
		with c6:
			fig, ax = plt.subplots( figsize=(7, 5) )
			ax.bar( expl[ "Component" ], expl[ "Explained Variance (%)" ], edgecolor="black" )
			ax.set_ylabel( "% Variance Explained" )
			ax.set_title( "PCA Variance Explained" )
			fig.tight_layout( )
			st.pyplot( fig, use_container_width=True )
			plt.close( fig )

# =========================================================================================
# TAB — INFERENTIAL STATISTICS (Enhanced Visuals Only)
# =========================================================================================

# =========================================================================================
# TAB — INFERENTIAL STATISTICS (RESTORED + VERIFIED + VISUALLY FIXED)
# =========================================================================================

with tabs[2]:

    st.subheader("")

    df = st.session_state.df

    if df is None or df.empty:
        st.info("No data available.")
        st.stop()

    numeric_cols = st.session_state.numeric_cols
    categorical_cols = st.session_state.categorical_cols

    if not numeric_cols:
        st.info("No numeric variables available for inferential analysis.")
        st.stop()

    # -------------------------------------------------------------------------------------
    # VARIABLE SELECTION
    # -------------------------------------------------------------------------------------

    col_y = st.selectbox("Select numeric outcome variable", numeric_cols)

    col_group = None
    if categorical_cols:
        col_group = st.selectbox(
            "Select grouping variable (optional)",
            ["<None>"] + categorical_cols
        )
        if col_group == "<None>":
            col_group = None

    blue_divider()

    # =====================================================================================
    # NORMALITY TEST — SHAPIRO–WILK + Q–Q PLOT (FIXED LAYOUT)
    # =====================================================================================

    st.markdown("### Normality Test")

    y = df[col_y].dropna()

    if len(y) >= 3:
        stat, p_value = stats.shapiro(y)

        fig, ax = plt.subplots(figsize=(5, 5))

        stats.probplot(y, plot=ax)

        ax.set_title(
            f"Q–Q Plot: {col_y} | Shapiro p = {p_value:.3g} | n = {len(y)}",
            fontsize=12,
            fontweight="bold",
            pad=10
        )

        # Prevent scientific-notation collision (e.g., 1e10)
        ax.ticklabel_format(style="plain", axis="both")

        # Visual clarity
        ax.get_lines()[0].set_marker("o")
        ax.get_lines()[0].set_alpha(0.7)
        ax.get_lines()[0].set_markeredgecolor("black")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.write(
            f"Shapiro–Wilk statistic = {stat:.4f}, p-value = {p_value:.4g}"
        )
    else:
        st.info("Not enough observations for normality testing.")

    blue_divider()

    # =====================================================================================
    # GROUP COMPARISON — ANOVA + KRUSKAL–WALLIS (RESTORED)
    # =====================================================================================

    if col_group:
        st.markdown("### Group Comparison")

        grouped = [
            grp[col_y].dropna().values
            for _, grp in df.groupby(col_group)
        ]

        valid_groups = [g for g in grouped if len(g) >= 2]

        if len(valid_groups) >= 2:

            # ----------------------------
            # ANOVA
            # ----------------------------
            f_stat, p_anova = stats.f_oneway(*valid_groups)

            fig, ax = inferential_plot(
                title=f"ANOVA: {col_y} by {col_group}",
                subtitle=f"p = {p_anova:.3g}",
                figsize=(6, 4)
            )

            means = df.groupby(col_group)[col_y].mean()
            means.plot(
                kind="bar",
                ax=ax,
                edgecolor="black",
                linewidth=1.2,
                alpha=0.8
            )

            ax.set_xlabel(col_group)
            ax.set_ylabel(col_y)

            st.pyplot(fig)
            plt.close(fig)

            st.write(
                f"ANOVA F-statistic = {f_stat:.4f}, p-value = {p_anova:.4g}"
            )

            # ----------------------------
            # Kruskal–Wallis
            # ----------------------------
            h_stat, p_kw = stats.kruskal(*valid_groups)

            st.write(
                f"Kruskal–Wallis H-statistic = {h_stat:.4f}, p-value = {p_kw:.4g}"
            )

        else:
            st.info("Not enough valid groups for group comparison.")

        blue_divider()

    # =====================================================================================
    # CORRELATION ANALYSIS — PEARSON + SPEARMAN (RESTORED)
    # =====================================================================================

    st.markdown("### Correlation Analysis")

    col_x2 = st.selectbox(
        "Select second numeric variable",
        [c for c in numeric_cols if c != col_y]
    )

    x = df[col_x2]
    y = df[col_y]

    mask = x.notna() & y.notna()

    if mask.sum() >= 3:

        # Pearson
        r_p, p_p = stats.pearsonr(x[mask], y[mask])

        # Spearman
        r_s, p_s = stats.spearmanr(x[mask], y[mask])

        fig, ax = inferential_plot(
            title=f"Correlation: {col_y} vs {col_x2}",
            subtitle=f"Pearson r = {r_p:.3f} (p={p_p:.3g}) | "
                     f"Spearman ρ = {r_s:.3f} (p={p_s:.3g})",
            figsize=(6, 4),
            ref_line=0.0
        )

        ax.scatter(
            x[mask],
            y[mask],
            alpha=0.7,
            edgecolor="black"
        )

        ax.set_xlabel(col_x2)
        ax.set_ylabel(col_y)

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("Not enough paired observations for correlation.")

    blue_divider()

    # =====================================================================================
    # CATEGORICAL ASSOCIATION — CHI-SQUARE + CRAMÉR’S V (RESTORED)
    # =====================================================================================

    if categorical_cols:
        st.markdown("### Categorical Association")

        col_cat1 = st.selectbox("Select first categorical variable", categorical_cols)
        col_cat2 = st.selectbox(
            "Select second categorical variable",
            [c for c in categorical_cols if c != col_cat1]
        )

        contingency = pd.crosstab(df[col_cat1], df[col_cat2])

        if contingency.size > 0:
            chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)

            n = contingency.values.sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

            st.write(
                f"Chi-square = {chi2:.4f}, "
                f"p-value = {p_chi:.4g}, "
                f"Cramér’s V = {cramers_v:.4f}"
            )

            st.dataframe(contingency, use_container_width=True)
        else:
            st.info("Insufficient data for categorical association.")



# =========================================================================================
# TAB 4 — ANOMALY DETECTION
# =========================================================================================

with tabs[ 3 ]:
	st.header( "" )
	
	if st.session_state.df is None:
		st.info( "No data loaded." )
		st.stop( )
	
	df = st.session_state.df
	
	# -------------------------------------------------------------------------
	# Prepare numeric data (analysis-only)
	# -------------------------------------------------------------------------
	num_df = clean_numeric( df.select_dtypes( include=[ np.number ] ) )
	
	if num_df.empty:
		st.info( "No usable numeric columns available for anomaly detection." )
		st.stop( )
	
	all_num_cols = num_df.columns.tolist( )
	
	# Finance-aware default: prefer PY / CY / BY if present
	preferred = [ c for c in all_num_cols if c.lower( ) in ("py", "cy", "by") ]
	default_vars = preferred if preferred else default_pick( all_num_cols, 2 )
	
	vars_sel = st.multiselect(
		"Variables to analyze",
		all_num_cols,
		default=default_vars
	)
	
	if not vars_sel:
		st.info( "Select at least one numeric variable to run anomaly detection." )
		st.stop( )
	
	analysis_scale = st.checkbox(
		"Use analysis-only standardization (recommended for multivariate methods)",
		value=True
	)
	
	# Analysis-only working frame
	work_df = num_df[ vars_sel ].copy( )
	
	if analysis_scale and len( vars_sel ) > 1:
		work_df[ : ] = SKStandardScaler( ).fit_transform( work_df.values )
	
	# -------------------------------------------------------------------------
	# Method Selection
	# -------------------------------------------------------------------------
	st.subheader( "Detection Methods" )
	blue_divider( )
	
	c_m1, c_m2 = st.columns( 2 )
	
	with c_m1:
		use_z = st.checkbox( "Z-Score", value=True )
		use_mz = st.checkbox( "Modified Z-Score (MAD)", value=True )
		use_iqr = st.checkbox( "IQR Fence", value=True )
	
	with c_m2:
		use_mahal = st.checkbox( "Mahalanobis Distance", value=True )
		use_iforest = st.checkbox( "Isolation Forest", value=True )
		use_lof = st.checkbox( "Local Outlier Factor (LOF)", value=False )
	
	# -------------------------------------------------------------------------
	# Threshold Controls
	# -------------------------------------------------------------------------
	st.subheader( "Thresholds" )
	blue_divider( )
	
	c_t1, c_t2 = st.columns( 2 )
	
	with c_t1:
		z_thresh = st.slider( "Z / Modified Z threshold", 2.0, 5.0, 3.0, 0.1 )
		iqr_mult = st.slider( "IQR multiplier", 1.0, 3.0, 1.5, 0.1 )
	
	with c_t2:
		lof_k = st.slider( "LOF neighbors (k)", 5, 50, 20, 1 )
		min_methods = st.slider(
			"Consensus: minimum methods flagging a row",
			1, 4, 1, 1
		)
	
	# -------------------------------------------------------------------------
	# Run Detection
	# -------------------------------------------------------------------------
	anomaly_flags = pd.DataFrame( index=work_df.index )
	
	# --- Univariate methods
	for col in vars_sel:
		s = work_df[ col ].dropna( )
		
		if s.empty:
			continue
		
		if use_z:
			z = (s - s.mean( )) / s.std( ) if s.std( ) else pd.Series( 0, index=s.index )
			anomaly_flags[ f"{col}_z" ] = z.abs( ) >= z_thresh
		
		if use_mz:
			med = s.median( )
			mad = np.median( np.abs( s - med ) )
			if mad == 0:
				mz = pd.Series( 0, index=s.index )
			else:
				mz = 0.6745 * (s - med) / mad
			anomaly_flags[ f"{col}_mz" ] = mz.abs( ) >= z_thresh
		
		if use_iqr:
			q1, q3 = s.quantile( 0.25 ), s.quantile( 0.75 )
			iqr = q3 - q1
			lo = q1 - iqr_mult * iqr
			hi = q3 + iqr_mult * iqr
			anomaly_flags[ f"{col}_iqr" ] = (s < lo) | (s > hi)
	
	# --- Multivariate methods
	mv_df = work_df.dropna( axis=0 )
	
	if mv_df.shape[ 0 ] >= 10 and mv_df.shape[ 1 ] >= 2:
		if use_mahal:
			cov = np.cov( mv_df.values, rowvar=False )
			if np.linalg.det( cov ) != 0:
				inv_cov = np.linalg.inv( cov )
				mean = mv_df.mean( ).values
				diffs = mv_df.values - mean
				md = np.sqrt( np.einsum( "ij,jk,ik->i", diffs, inv_cov, diffs ) )
				cutoff = np.sqrt( stats.chi2.ppf( 0.975, mv_df.shape[ 1 ] ) )
				anomaly_flags.loc[ mv_df.index, "mahal" ] = md > cutoff
		
		if use_iforest:
			from sklearn.ensemble import IsolationForest
			
			iso = IsolationForest( contamination="auto", random_state=42 )
			preds = iso.fit_predict( mv_df.values )
			anomaly_flags.loc[ mv_df.index, "iforest" ] = preds == -1
		
		if use_lof:
			from sklearn.neighbors import LocalOutlierFactor
			
			lof = LocalOutlierFactor( n_neighbors=lof_k )
			preds = lof.fit_predict( mv_df.values )
			anomaly_flags.loc[ mv_df.index, "lof" ] = preds == -1
	
	# -------------------------------------------------------------------------
	# Consensus & Output
	# -------------------------------------------------------------------------
	st.subheader( "Anomaly Summary" )
	blue_divider( )
	
	if anomaly_flags.empty:
		st.info( "No anomalies detected under the selected methods and thresholds." )
		st.stop( )
	
	anomaly_flags = anomaly_flags.fillna( False )
	anomaly_flags[ "methods_flagged" ] = anomaly_flags.sum( axis=1 )
	
	anomalies = anomaly_flags[ anomaly_flags[ "methods_flagged" ] >= min_methods ]
	
	c_o1, c_o2 = st.columns( 2 )
	
	with c_o1:
		st.markdown( "### Flagged Observations" )
		render_table( anomalies.sort_values( "methods_flagged", ascending=False ) )
	
	with c_o2:
		st.markdown( "### Flag Count Distribution" )
		fig, ax = plt.subplots( figsize=(7, 5) )
		anomalies[ "methods_flagged" ].value_counts( ).sort_index( ).plot(
			kind="bar", ax=ax, edgecolor="black"
		)
		ax.set_xlabel( "Number of Methods Flagging" )
		ax.set_ylabel( "Observation Count" )
		ax.set_title( "Consensus Strength" )
		fig.tight_layout( )
		st.pyplot( fig, use_container_width=True )
		plt.close( fig )
	
	# -------------------------------------------------------------------------
	# Visualization — Distribution with Anomalies
	# -------------------------------------------------------------------------
	st.subheader( "Distributions with Anomalies Highlighted" )
	blue_divider( )
	
	for col in vars_sel:
		if col not in work_df.columns:
			continue
		
		s = work_df[ col ]
		flagged_idx = anomalies.index.intersection( s.index )
		
		if flagged_idx.empty:
			continue
		
		c_v1, c_v2 = st.columns( 2 )
		
		with c_v1:
			fig, ax = plt.subplots( figsize=(7, 5) )
			ax.hist( s.dropna( ), bins=30, alpha=0.7, edgecolor="black" )
			ax.scatter(
				s.loc[ flagged_idx ],
				np.zeros( len( flagged_idx ) ),
				color="red",
				label="Anomalies"
			)
			ax.set_title( f"{col} — Histogram with Anomalies" )
			ax.legend( )
			fig.tight_layout( )
			st.pyplot( fig, use_container_width=True )
			plt.close( fig )
		
		with c_v2:
			fig, ax = plt.subplots( figsize=(7, 5) )
			ax.boxplot( s.dropna( ), vert=False )
			ax.scatter(
				s.loc[ flagged_idx ],
				np.ones( len( flagged_idx ) ),
				color="red"
			)
			ax.set_title( f"{col} — Boxplot with Anomalies" )
			fig.tight_layout( )
			st.pyplot( fig, use_container_width=True )
			plt.close( fig )
	
	# -------------------------------------------------------------------------
	# Export
	# -------------------------------------------------------------------------
	st.download_button(
		"Export Anomaly Table (CSV)",
		anomalies.to_csv( ),
		"anomalies.csv",
		"text/csv"
	)

# =========================================================================================
# TAB 5 — FEATURE ENGINEERING
# =========================================================================================

with tabs[ 4 ]:
	st.caption(
		"Preview and apply feature transformations without modifying the original dataset." )
	
	# ------------------------------------------------------------------
	# Guard: Ensure data exists
	# ------------------------------------------------------------------
	if "df" not in locals( ) or df is None or df.empty:
		st.warning( "No dataset loaded." )
		st.stop( )
	
	df_original = df.copy( )
	
	if "feature_engineered_df" not in st.session_state:
		st.session_state[ "feature_engineered_df" ] = None
	
	# ------------------------------------------------------------------
	# Column classification (reuse existing logic)
	# ------------------------------------------------------------------
	numeric_columns = [
			c for c in df_original.columns
			if df_original[ c ].dtype.kind in { "i",
			                                    "f" }
	]
	
	categorical_columns = [
			c for c in df_original.columns
			if c not in numeric_columns
	]
	
	# ------------------------------------------------------------------
	# Column selection
	# ------------------------------------------------------------------
	st.markdown( "### Column Selection" )
	
	selected_columns = st.multiselect(
		"Select columns for feature engineering",
		options=df_original.columns.tolist( )
	)
	
	if not selected_columns:
		st.info( "Select one or more columns to begin." )
		st.stop( )
	
	df_working = df_original[ selected_columns ].copy( )
	
	# ======================================================================================
	# Missing Value Handling
	# ======================================================================================
	st.markdown( "### Missing Value Handling" )
	
	from imputers import (
		MeanImputer,
		SimpleImputer,
		NearestImputer,
		IterativeImputer
	)
	
	impute_columns = st.multiselect(
		"Columns to impute",
		options=df_working.columns.tolist( )
	)
	
	imputer_type = st.selectbox(
		"Imputation strategy",
		[
				"None",
				"Mean",
				"Median",
				"Most Frequent",
				"Nearest Neighbors",
				"Iterative"
		]
	)
	
	if imputer_type != "None" and impute_columns:
		X_impute = df_working[ impute_columns ].to_numpy( )
		
		if imputer_type == "Mean":
			imputer = MeanImputer( )
		elif imputer_type == "Median":
			imputer = SimpleImputer( strategy="median" )
		elif imputer_type == "Most Frequent":
			imputer = SimpleImputer( strategy="most_frequent" )
		elif imputer_type == "Nearest Neighbors":
			imputer = NearestImputer( )
		elif imputer_type == "Iterative":
			imputer = IterativeImputer( )
		
		X_imputed = imputer.train_transform( X_impute )
		
		df_working[ impute_columns ] = X_imputed
		
		st.caption( "Imputation preview (first 5 rows)" )
		st.dataframe( df_working.head( ) )
	
	# ======================================================================================
	# Encoding
	# ======================================================================================
	st.markdown( "### Encoding" )
	
	from encoders import (
		OneHotEncoder,
		OrdinalEncoder,
		TargetEncoder
	)
	
	encode_columns = st.multiselect(
		"Categorical columns to encode",
		options=[ c for c in df_working.columns if c in categorical_columns ]
	)
	
	encoding_type = st.selectbox(
		"Encoding method",
		[ "None",
		  "One-Hot",
		  "Ordinal",
		  "Target" ]
	)
	
	if encoding_type != "None" and encode_columns:
		X_encode = df_working[ encode_columns ].astype( str ).to_numpy( )
		
		if encoding_type == "One-Hot":
			encoder = OneHotEncoder( sparse=False )
			X_encoded = encoder.train_transform( X_encode )
			
			encoded_df = pd.DataFrame(
				X_encoded,
				index=df_working.index
			)
			
			df_working = df_working.drop( columns=encode_columns )
			df_working = pd.concat( [ df_working,
			                          encoded_df ], axis=1 )
		
		elif encoding_type == "Ordinal":
			encoder = OrdinalEncoder( )
			X_encoded = encoder.train_transform( X_encode )
			
			df_working[ encode_columns ] = X_encoded
		
		elif encoding_type == "Target":
			target_col = st.selectbox(
				"Select target column",
				options=df_original.columns.tolist( )
			)
			
			y = df_original[ target_col ].to_numpy( )
			encoder = TargetEncoder( )
			X_encoded = encoder.train_transform( X_encode, y )
			
			encoded_df = pd.DataFrame(
				X_encoded,
				index=df_working.index
			)
			
			df_working = df_working.drop( columns=encode_columns )
			df_working = pd.concat( [ df_working,
			                          encoded_df ], axis=1 )
		
		st.caption( "Encoding preview (first 5 rows)" )
		st.dataframe( df_working.head( ) )
	
	# ======================================================================================
	# Scaling / Normalization
	# ======================================================================================
	st.markdown( "### Scaling & Normalization" )
	
	from scalers import (
		StandardScaler,
		MinMaxScaler,
		RobustScaler,
		NormalScaler
	)
	
	scale_columns = st.multiselect(
		"Numeric columns to scale",
		options=[ c for c in df_working.columns if c in numeric_columns ]
	)
	
	scaler_type = st.selectbox(
		"Scaler",
		[ "None",
		  "Standard",
		  "Min-Max",
		  "Robust",
		  "Normalize" ]
	)
	
	if scaler_type != "None" and scale_columns:
		X_scale = df_working[ scale_columns ].to_numpy( )
		
		if scaler_type == "Standard":
			scaler = StandardScaler( )
		elif scaler_type == "Min-Max":
			scaler = MinMaxScaler( )
		elif scaler_type == "Robust":
			scaler = RobustScaler( )
		elif scaler_type == "Normalize":
			scaler = NormalScaler( )
			
			X_scaled = scaler.train_transform( X_scale )
			df_working[ scale_columns ] = X_scaled
			
			st.caption( "Scaling preview (first 5 rows)" )
			st.dataframe( df_working.head( ) )
			
	# ======================================================================================
	# Feature Generation
	# ======================================================================================
	st.markdown( "### Feature Generation" )
	
	from encoders import PolynomialFeatures
	
	poly_columns = st.multiselect(
		"Columns for polynomial features",
		options=[ c for c in df_working.columns if c in numeric_columns ]
	)
	
	poly_degree = st.slider(
		"Polynomial degree",
		min_value=2,
		max_value=4,
		value=2
	)
	
	if poly_columns:
		X_poly = df_working[ poly_columns ].to_numpy( )
		poly = PolynomialFeatures( degree=poly_degree )
		X_poly_out = poly.train_transform( X_poly )
		
		poly_df = pd.DataFrame(
			X_poly_out,
			index=df_working.index
		)
		
		df_working = df_working.drop( columns=poly_columns )
		df_working = pd.concat( [ df_working,
		                          poly_df ], axis=1 )
		
		st.caption( "Polynomial feature preview (first 5 rows)" )
		st.dataframe( df_working.head( ) )
	
	# ======================================================================================
	# Apply / Export
	# ======================================================================================
	st.markdown( "### Apply or Export" )
	
	if st.button( "Apply Feature Engineering" ):
		st.session_state[ "feature_engineered_df" ] = df_working.copy( )
		st.success( "Feature-engineered dataset stored in session state." )
	
	st.download_button(
		label="Download Feature-Engineered Dataset (CSV)",
		data=df_working.to_csv( index=False ),
		file_name="feature_engineered_data.csv",
		mime="text/csv"
	)

# =========================================================================================
# TAB 6 — CLASSIFICATION MODELS (NOTEBOOK-STYLE, SINGLE-MODEL EXPLORER)
# =========================================================================================

with tabs[ 5 ]:
	st.header( "" )
	
	# ------------------------------------------------------------------
	# DATA VALIDATION
	# ------------------------------------------------------------------
	df = st.session_state.get( "df", None )
	numeric_cols = st.session_state.get( "numeric_cols", [ ] )
	categorical_cols = st.session_state.get( "categorical_cols", [ ] )
	
	if df is None or df.empty:
		st.warning( "⚠️ No dataset loaded." )
		st.stop( )
	
	if not numeric_cols or not categorical_cols:
		st.warning( "⚠️ Classification requires numeric features and a categorical target." )
		st.stop( )
	
	# ------------------------------------------------------------------
	# TARGET & FEATURES
	# ------------------------------------------------------------------
	st.subheader( "Target & Features" )
	
	target = st.selectbox( "Target (categorical)", categorical_cols )
	features = st.multiselect(
		"Feature columns (numeric)",
		numeric_cols,
		default=numeric_cols[ :3 ]
	)
	
	if not features:
		st.info( "Please select at least one feature." )
		st.stop( )
	
	X = df[ features ].to_numpy( )
	y = df[ target ].to_numpy( )
	
	# ------------------------------------------------------------------
	# MODEL SELECTION
	# ------------------------------------------------------------------
	from classifications import (
		LogisticRegression,
		SupportVector,
		RandomForest,
		NearestNeighbor,
		BaggingModel,
		GradientBoost
	)
	
	model_map = {
			"Logistic Regression": LogisticRegression,
			"Support Vector Machine": SupportVector,
			"Random Forest": RandomForest,
			"k-Nearest Neighbors": NearestNeighbor,
			"Bagging": BaggingModel,
			"Gradient Boosting": GradientBoost
	}
	
	st.subheader( "Model Selection" )
	model_name = st.selectbox( "Select classification model", list( model_map.keys( ) ) )
	
	model = model_map[ model_name ]( )
	
	# ------------------------------------------------------------------
	# TRAIN / TEST SPLIT
	# ------------------------------------------------------------------
	st.subheader( "Training Configuration" )
	
	test_sz = st.slider( "Test set size (%)", 10, 20, 30, key='classifications-1' ) / 100.0
	random_state = st.number_input( "Random state", value=42, step=1, key='classifications-2' )
	
	if st.button( "🚀 Train Classifier" ):
		try:
			X_train, X_test, y_train, y_test = model.split_data(
				X, y, size=test_sz, random=random_state
			)
			
			model.train( X_train, y_train )
			
			# ------------------------------------------------------------------
			# METRICS & ANALYSIS (WRAPPER-PROVIDED)
			# ------------------------------------------------------------------
			st.subheader( "Model Performance" )
			
			analysis_df = model.analyze( X_test, y_test )
			st.dataframe( analysis_df, use_container_width=True )
			
			# ------------------------------------------------------------------
			# CONFUSION MATRIX (BUILT-IN)
			# ------------------------------------------------------------------
			st.subheader( "Confusion Matrix" )
			
			fig_cm = plt.figure( )
			model.confusion_matrix( X_test, y_test )
			st.pyplot( fig_cm )
			
			# ------------------------------------------------------------------
			# ROC CURVE (IF SUPPORTED)
			# ------------------------------------------------------------------
			if hasattr( model, "roc_curve" ):
				st.subheader( "ROC Curve" )
				fig_roc = plt.figure( )
				model.roc_curve( X_test, y_test )
				st.pyplot( fig_roc )
		
		except Exception as e:
			st.error( f"Classification failed: {e}" )

# -------------------------------------------------------------------------------------
# TAB 7 — REGRESSION MODELS TAB
# -------------------------------------------------------------------------------------

with tabs[ 6 ]:
	st.header( "" )
	
	# ------------------------------------------------------------------
	# DATA VALIDATION
	# ------------------------------------------------------------------
	df = st.session_state.get( "df", None )
	numeric_cols = st.session_state.get( "numeric_cols", [ ] )
	
	if df is None or df.empty:
		st.warning( "⚠️ No dataset loaded." )
		st.stop( )
	
	if not numeric_cols:
		st.warning( "⚠️ No numeric columns available for regression." )
		st.stop( )
	
	# ------------------------------------------------------------------
	# TARGET & FEATURES
	# ------------------------------------------------------------------
	st.subheader( "Target & Features" )
	
	target = st.selectbox( "Target (numeric)", numeric_cols )
	features = st.multiselect(
		"Feature columns (numeric)",
		[ c for c in numeric_cols if c != target ],
		default=[ c for c in numeric_cols if c != target ][ :3 ]
	)
	
	if not features:
		st.info( "Please select at least one feature." )
		st.stop( )
	
	X = df[ features ].to_numpy( )
	y = df[ target ].to_numpy( )
	
	# ------------------------------------------------------------------
	# MODEL SELECTION
	# ------------------------------------------------------------------
	from regressions import (
		LeastSquares,
		Ridge,
		LeastAngle,
		ElasticNet,
		BayesianRidge,
		SupportVector,
		GradientDescent,
		BaggingModel,
		GradientBoost,
		RandomForest,
		GaussianProcess
	)
	
	model_map = {
			"Ordinary Least Squares": LeastSquares,
			"Ridge Regression": Ridge,
			"Lasso (Least Angle)": LeastAngle,
			"Elastic Net": ElasticNet,
			"Bayesian Ridge": BayesianRidge,
			"Support Vector": SupportVector,
			"Stochastic Gradient Descent": GradientDescent,
			"Bagging Regressor": BaggingModel,
			"Gradient Boosting": GradientBoost,
			"Random Forest": RandomForest,
			"Gaussian Process": GaussianProcess
	}
	
	st.subheader( "Model Selection" )
	model_name = st.selectbox( "Select regression model", list( model_map.keys( ) ) )
	
	model = model_map[ model_name ]( )
	
	# ------------------------------------------------------------------
	# TRAIN / TEST SPLIT
	# ------------------------------------------------------------------
	st.subheader( "Training Configuration" )
	
	test_size = st.slider( "Test set size (%)", 10, 20, 30, key='regressions-1' ) / 100.0
	random_state = st.number_input( "Random state", value=42, step=1, key='regressions-2' )
	
	if st.button( "🚀 Train Model" ):
		try:
			X_train, X_test, y_train, y_test = model.split_data(
				X, y, size=test_size, random=random_state
			)
			
			model.train( X_train, y_train )
			
			# ------------------------------------------------------------------
			# METRICS
			# ------------------------------------------------------------------
			st.subheader( "Model Performance" )
			
			metrics_df = model.analyze( X_test, y_test )
			st.dataframe( metrics_df, use_container_width=True )
			
			# ------------------------------------------------------------------
			# SCATTER PLOT (BUILT-IN)
			# ------------------------------------------------------------------
			st.subheader( "Observed vs Predicted" )
			
			fig = plt.figure( )
			model.scatter_plot( X_test, y_test )
			st.pyplot( fig )
		
		except Exception as e:
			st.error( f"Regression failed: {e}" )


# -------------------------------------------------------------------------------------
# TAB 8 — CLUSTERING MODELS TAB
# -------------------------------------------------------------------------------------

with tabs[ 7 ]:

    st.subheader(" ")
    st.caption(
        "Explore unsupervised clustering models using numeric features only."
    )

    # ------------------------------------------------------------------
    # Data source resolution
    # ------------------------------------------------------------------
    if "feature_engineered_df" in st.session_state and \
       st.session_state["feature_engineered_df"] is not None:
        df_cluster = st.session_state["feature_engineered_df"].copy()
        st.info("Using feature-engineered dataset.")
    else:
        df_cluster = df.copy()
        st.info("Using original dataset.")

    if df_cluster is None or df_cluster.empty:
        st.warning("No dataset available for clustering.")
        st.stop()

    # ------------------------------------------------------------------
    # Column classification (numeric only)
    # ------------------------------------------------------------------
    numeric_columns = [
        c for c in df_cluster.columns
        if df_cluster[c].dtype.kind in {"i", "f"}
    ]

    if len(numeric_columns) < 2:
        st.warning("At least two numeric columns are required for clustering.")
        st.stop()

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    st.markdown("### Feature Selection")

    feature_columns = st.multiselect(
        "Select features for clustering",
        options=numeric_columns
    )

    if len(feature_columns) < 2:
        st.info("Select at least two features to continue.")
        st.stop()

    X = df_cluster[feature_columns].to_numpy()

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    st.markdown("### Clustering Model")

    from clusters import (
        KMeans,
        DBSCAN,
        Agglomerative
    )

    model_name = st.selectbox(
        "Clustering algorithm",
        [
            "K-Means",
            "DBSCAN",
            "Agglomerative"
        ]
    )

    # ------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------
    st.markdown("### Model Parameters")

    model = None

    if model_name == "K-Means":
        n_clusters = st.number_input("Number of clusters (k)", min_value=2, value=3)
        model = KMeans( clusters=n_clusters )

    elif model_name == "DBSCAN":
        eps = st.number_input("Epsilon (eps)", min_value=0.01, value=0.5)
        min_samples = st.number_input("Min samples", min_value=1, value=5)
        model = DBSCAN(eps=eps, min_samples=min_samples)

    elif model_name == "Agglomerative":
        n_clusters = st.number_input("Number of clusters", min_value=2, value=3)
        linkage = st.selectbox(
            "Linkage",
            ["ward", "complete", "average", "single"]
        )
        model = Agglomerative(
            n_clusters=n_clusters,
            linkage=linkage
        )

    # ------------------------------------------------------------------
    # Fit clustering model
    # ------------------------------------------------------------------
    st.markdown("### Run Clustering")

    if st.button("Run Clustering"):

        labels = model.project(X)

        df_results = df_cluster.copy()
        df_results["Cluster"] = labels

        st.success("Clustering complete.")

        # ------------------------------------------------------------------
        # Cluster summary
        # ------------------------------------------------------------------
        st.markdown("### Cluster Summary")

        cluster_counts = (
            df_results["Cluster"]
            .value_counts()
            .rename("Count")
            .reset_index()
            .rename(columns={"index": "Cluster"})
        )

        st.dataframe(cluster_counts, use_container_width=True)

        # ------------------------------------------------------------------
        # Visualization
        # ------------------------------------------------------------------
        st.markdown("### Cluster Visualization")

        if len(feature_columns) == 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(
                df_results[feature_columns[0]],
                df_results[feature_columns[1]],
                c=df_results["Cluster"],
                alpha=0.7
            )
            ax.set_xlabel(feature_columns[0])
            ax.set_ylabel(feature_columns[1])
            ax.set_title("Cluster Assignments")
            st.pyplot(fig)

        else:
            st.info(
                "Visualization limited to two features. "
                "Reduce feature selection to two columns for scatter plotting."
            )

        # ------------------------------------------------------------------
        # Centroids (if available)
        # ------------------------------------------------------------------
        if hasattr(model, "centroids_"):

            st.markdown("### Cluster Centroids")

            centroid_df = pd.DataFrame(
                model.centroids_,
                columns=feature_columns
            )

            centroid_df.insert(0, "Cluster", range(len(centroid_df)))

            st.dataframe(centroid_df, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 9 — TIME SERIES MODELS TAB
# -------------------------------------------------------------------------------------

with tabs[ 8 ]:

    st.header("")

    # ------------------------------------------------------------------
    # DATA VALIDATION
    # ------------------------------------------------------------------
    df = st.session_state.get("df", None)
    numeric_cols = st.session_state.get("numeric_cols", [])

    if df is None or df.empty:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    if not numeric_cols:
        st.warning("⚠️ No numeric columns available for time-series analysis.")
        st.stop()

    # ------------------------------------------------------------------
    # SERIES SELECTION
    # ------------------------------------------------------------------
    st.subheader("Time-Series Selection")

    series_col = st.selectbox(
        "Select numeric time-series column",
        numeric_cols
    )

    series = df[series_col].dropna().to_numpy()

    if series.ndim != 1 or len(series) < 10:
        st.warning("⚠️ Selected series is too short for modeling.")
        st.stop()

    # ------------------------------------------------------------------
    # MODEL SELECTION
    # ------------------------------------------------------------------
    from forecasting import (
        LaggingSeries,
        ARIMA,
        SARIMA,
        ExpandingWindow
    )

    model_map = {
        "Lagged Linear Regression": "lag",
        "ARIMA": "arima",
        "SARIMA": "sarima"
    }

    st.subheader("Model Selection")
    model_name = st.selectbox("Select time-series model", list(model_map.keys()))

    # ------------------------------------------------------------------
    # MODEL PARAMETERS
    # ------------------------------------------------------------------
    st.subheader("Model Parameters")

    model = None

    if model_name == "Lagged Linear Regression":
        lag = st.number_input("Lag order", min_value=1, value=5)
        model = LaggingSeries(lag=lag)

    elif model_name == "ARIMA":
        p = st.number_input("p (AR)", min_value=0, value=1)
        d = st.number_input("d (I)", min_value=0, value=0)
        q = st.number_input("q (MA)", min_value=0, value=0)
        model = ARIMA(order=(p, d, q))

    elif model_name == "SARIMA":
        p = st.number_input("p (AR)", min_value=0, value=1)
        d = st.number_input("d (I)", min_value=0, value=1)
        q = st.number_input("q (MA)", min_value=0, value=1)
        P = st.number_input("P (Seasonal AR)", min_value=0, value=0)
        D = st.number_input("D (Seasonal I)", min_value=0, value=0)
        Q = st.number_input("Q (Seasonal MA)", min_value=0, value=0)
        s = st.number_input("Season Length", min_value=0, value=0)

        model = SARIMA(
            order=(p, d, q),
            seasonal=(P, D, Q, s)
        )

    # ------------------------------------------------------------------
    # TRAIN / FORECAST
    # ------------------------------------------------------------------
    st.subheader("Train & Forecast")

    forecast_horizon = st.number_input(
        "Forecast horizon (steps)",
        min_value=1,
        value=5
    )

    if st.button("🚀 Run Time-Series Model"):

        try:
            model.train(series)

            forecast = model.project(n_steps=forecast_horizon)

            # ------------------------------------------------------------------
            # METRICS (IN-SAMPLE)
            # ------------------------------------------------------------------
            st.subheader("Model Evaluation")

            metrics = model.analyze()
            st.dataframe(
                pd.DataFrame(metrics, index=["Value"]).T,
                use_container_width=True
            )

            # ------------------------------------------------------------------
            # VISUALIZATION
            # ------------------------------------------------------------------
            st.subheader("Observed vs Forecast")

            fig, ax = plt.subplots()
            ax.plot(series, label="Observed")
            ax.plot(
                range(len(series), len(series) + len(forecast)),
                forecast,
                label="Forecast",
                linestyle="--"
            )
            ax.set_title("Time-Series Forecast")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Time-series modeling failed: {e}")

    # ------------------------------------------------------------------
    # EXPANDING WINDOW CV (OPTIONAL DIAGNOSTIC)
    # ------------------------------------------------------------------
    st.subheader("Expanding Window Cross-Validation")

    with st.expander("Show expanding-window splits"):
        initial = st.number_input("Initial window size", min_value=10, value=30)
        window = st.number_input("Test window size", min_value=1, value=10)

        splitter = ExpandingWindow(initial=initial, windows=window)

        if st.button("Visualize CV Splits"):
            fig = plt.figure()
            splitter.visualize(series)
            st.pyplot(fig)
