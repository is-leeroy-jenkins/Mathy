'''
	******************************************************************************************
	  Assembly:                mathy
	  Filename:                app.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022
	
	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="capp.py" company="Terry D. Eppler">
	
		 mathy app
	
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
		app.py
	</summary>
	******************************************************************************************
'''

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from scipy import stats
from typing import List, Dict

# Mathy
import config as cfg
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


import base64
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.power import TTestPower
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
import seaborn as sns

# ============================================
# Headers/Title
# ============================================
st.logo( image=cfg.LOGO, size='large' )

# ============================================
# Configuration
# ============================================
st.set_page_config( page_title='Mathy', layout='wide',
	page_icon=cfg.FAVICON, initial_sidebar_state='expanded' )

pd.options.display.float_format = '{:.4f}'.format

# ============================================
# Session State
# ============================================

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

# ============================================
# Utilities
# ============================================
def inferential_plot( title: str, subtitle: str | None = None,
    figsize: tuple[int, int] = (6, 4), grid: bool = True,
    ref_line: float | None = None, legend: bool = True ):
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
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

def log_step( msg: str ) -> None:
	st.session_state.pipeline_log.append( msg )

def render_table( df: pd.DataFrame, height: int=360 ) -> None:
	disp = df.copy( )
	float_cols = disp.select_dtypes( include=[ np.floating ] ).columns
	disp[ float_cols ] = disp[ float_cols ].round( 4 )
	st.data_editor( disp, use_container_width=True, height='auto' )

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

def create_visualization( df: pd.DataFrame ):
	st.subheader( 'Visualization Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	categorical_cols = df.select_dtypes( include=[ 'object' ] ).columns.tolist( )
	
	chart = st.selectbox( 'Chart Type', [ 'Histogram', 'Bar', 'Line',
	                                      'Scatter', 'Box', 'Pie', 'Correlation' ] )
	
	if chart == 'Histogram' and numeric_cols:
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.histogram( df, x=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Bar':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.bar( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Line':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.line( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Scatter':
		x = st.selectbox( 'X', numeric_cols )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.scatter( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.box( df, y=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		col = st.selectbox( 'Category Column', categorical_cols )
		fig = px.pie( df, names=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation' and len( numeric_cols ) > 1:
		corr = df[ numeric_cols ].corr( )
		fig = px.imshow( corr, text_auto=True )
		st.plotly_chart( fig, use_container_width=True )

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)
	
if 'mode' not in st.session_state or st.session_state[ 'mode' ] is None:
	st.session_state[ 'mode' ] = 'Data Transformation'

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
	st.title( '📦 Data Source' )

	use_fallback = st.sidebar.checkbox( 'Use default data', value=True )
	uploaded = st.sidebar.file_uploader( label='Upload Spreadsheet', type=[ 'xlsx',  'xls',  'csv' ] )
	
	if uploaded or use_fallback:
		if uploaded:
			df_dataset = pd.read_excel( uploaded ) if uploaded.name.endswith( 'xls' ) else pd.read_csv( uploaded )
			log_step( f'Loaded uploaded file: {uploaded.name}' )
		else:
			df_dataset = pd.read_excel( cfg.DEFAULT_DATA )
			log_step( 'Loaded Default Dataset' )
		
		st.session_state.raw_df = df_dataset.copy( )
		st.session_state.df_dataset = df_dataset.copy( )
		st.session_state.numeric_cols, st.session_state.categorical_cols = detect_column_types( df_dataset )
		
	st.sidebar.divider( )
	
	st.subheader( 'Mode' )
	mode = st.sidebar.radio( 'Select', cfg.MODE.keys( ), index=0 )
	
	style_subheaders( )

# ============================================
# DATA TRANSFORMATION MODE
# ============================================
if mode == 'Data Profile':
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.header( 'Schema' )
		st.divider( )
		
		if st.session_state.df_dataset is None:
			st.info( 'No data loaded.' )
			st.stop( )
		
		df_dataset = st.session_state.df_dataset
		
		# -------------------------------------------------------------------------------------
		# SCHEMA INFERENCE
		# -------------------------------------------------------------------------------------
		def infer_schema( df: pd.DataFrame ) -> Dict[ str, str ]:
			schema: dict[ str, str ] = { }
			n_rows = len( df )
			
			for col in df.columns:
				s = df[ col ]
				name = col.lower( )
				nunique = s.nunique( dropna=True )
				unique_ratio = nunique / max( 1, n_rows )
				
				# ------------------------------------------------------------------
				# 1) Datetime: ONLY for object/string columns (prevents PY/CY/BY errors)
				# ------------------------------------------------------------------
				if s.dtype == 'object':
					try:
						parsed_dt = pd.to_datetime( s, errors='coerce' )
						if parsed_dt.notna( ).sum( ) / max( 1, n_rows ) > 0.9:
							schema[ col ] = 'datetime'
							continue
					except Exception:
						pass
				
				# ------------------------------------------------------------------
				# 2) Numeric detection: ints AND floats
				# ------------------------------------------------------------------
				if pd.api.types.is_numeric_dtype( s ):
					# Identifier heuristics for numeric codes/keys
					if ('id' in name) or ('code' in name) or ('key' in name) or (unique_ratio > 0.8):
						schema[ col ] = 'identifier'
						continue
					if pd.api.types.is_integer_dtype( s ) and nunique <= 20:
						schema[ col ] = 'ordinal'
						continue
					schema[ col ] = 'numeric'
					continue
				
				# ------------------------------------------------------------------
				# 3) Categorical fallback
				# ------------------------------------------------------------------
				schema[ col ] = 'categorical'
			
			return schema
		
		schema = infer_schema( df_dataset )
		st.session_state.column_schema = schema
		st.session_state.numeric_cols = [ c for c, t in schema.items( ) if t == 'numeric' ]
		st.session_state.categorical_cols = [ c for c, t in schema.items( ) if t == 'categorical' ]
		
		# -------------------------------------------------------------------------------------
		# DATASET DISPLAY
		# -------------------------------------------------------------------------------------
	
		st.subheader( 'Data' )
		render_table( df_dataset )
		
		
		# -------------------------------------------------------------------------------------
		# SCHEMA METRICS
		# -------------------------------------------------------------------------------------
		st.subheader( 'Types' )
		st.divider( )
		type_counts = pd.Series( schema ).value_counts( )
		m1, m2, m3, m4, m5 = st.columns( 5, border=True )
		m1.metric( 'Rows', len( df_dataset ) )
		m2.metric( 'Numeric', type_counts.get( 'numeric', 0 ) )
		m3.metric( 'Ordinal / ID', type_counts.get( 'ordinal', 0 ) + type_counts.get( 'identifier', 0 ) )
		m4.metric( 'Categorical', type_counts.get( 'categorical', 0 ) )
		m5.metric( 'Datetime', type_counts.get( 'datetime', 0 ) )
		
		st.divider( )
		st.subheader( 'Records' )
		with st.expander( 'Editor', expanded=True ):
			top_c1, top_c2 = st.columns( [ 0.20, 0.80 ] )
			with top_c1:
				row_idx = st.number_input( 'Select Row Index', min_value=0, max_value=len( df_dataset ) - 1,
					step=1, key='row_editor_index' )
			
			row = df_dataset.iloc[ row_idx ]
			updated = { }
			
			col_left, col_right = st.columns( 2, border=True )
			
			with st.form( 'row_edit_form' ):
				for i, (col, dtype) in enumerate( schema.items( ) ):
					target = col_left if i % 2 == 0 else col_right
					val = row[ col ]
					with target:
						if dtype == 'numeric':
							updated[ col ] = st.number_input(
								col, value=float( val ) if pd.notna( val ) else 0.0 )
						elif dtype == 'ordinal':
							updated[ col ] = st.number_input(
								col, value=int( val ) if pd.notna( val ) else 0 )
						elif dtype == 'datetime':
							updated[ col ] = st.date_input(
								col,
								value=pd.to_datetime( val ).date( )
								if pd.notna( val )
								else pd.Timestamp.today( ).date( ) )
						elif dtype == 'categorical':
							options = df_dataset[ col ].dropna( ).unique( ).tolist( )
							updated[ col ] = st.selectbox(
								col, options,
								index=options.index( val ) if val in options else 0 )
						else:
							updated[ col ] = st.text_input(
								col, value=str( val ), disabled=True )
				
				submitted = st.form_submit_button( 'Apply Row Update' )
			
			if submitted:
				before = df_dataset.loc[ row_idx ].copy( )
				for col, value in updated.items( ):
					if schema[ col ] == 'datetime':
						st.session_state.df_dataset.at[ row_idx, col ] = pd.to_datetime( value )
					else:
						st.session_state.df_dataset.at[ row_idx, col ] = value
				
				after = st.session_state.df_dataset.loc[ row_idx ]
				log_step( f'Updated row {row_idx}' )
				st.success( f'Row {row_idx} updated.' )
				st.data_editor( pd.DataFrame( {
						'Before': before,
						'After': after } ), use_container_width=True )
				st.rerun( )
		
		# =====================================================================================
		# DIAGNOSTIC VISUALIZATIONS (TAB-1 APPROPRIATE)
		# =====================================================================================
		
		st.divider( )
		st.subheader( 'Diagnostics' )
		
		v1, v2 = st.columns( 2, border=True )
		with v1:
			fig, ax = plt.subplots( figsize=(5, 4) )
			type_counts.plot( kind='bar', ax=ax, edgecolor='black' )
			ax.set_title( 'Column Type Distribution' )
			ax.set_ylabel( 'Count' )
			fig.tight_layout( )
			st.pyplot( fig )
			plt.close( fig )
		
		with v2:
			missing_pct = (df_dataset.isna( ).mean( ) * 100).sort_values( ascending=False )
			missing_pct = missing_pct[ missing_pct > 0 ].head( 10 )
			if not missing_pct.empty:
				fig, ax = plt.subplots( figsize=(5, 4) )
				missing_pct.plot( kind='bar', ax=ax, edgecolor='black' )
				ax.set_title( 'Top Columns by Missing %' )
				ax.set_ylabel( 'Percent Missing' )
				fig.tight_layout( )
				st.pyplot( fig )
				plt.close( fig )
			else:
				st.info( 'No Missing Values Detected.' )
		
		st.divider( )
		st.subheader( 'Cardinaltiy' )
		v3, v4 = st.columns( 2, border=True )
		with v3:
			cardinality = df_dataset.nunique( dropna=True ).sort_values( ascending=False ).head( 10 )
			fig, ax = plt.subplots( figsize=(5, 4) )
			cardinality.plot( kind='bar', ax=ax, edgecolor='black' )
			ax.set_title( 'Top Columns by Cardinality' )
			ax.set_ylabel( 'Unique Values' )
			fig.tight_layout( )
			st.pyplot( fig )
			plt.close( fig )
		
		with v4:
			st.caption( 'Row edits are confirmed above before commit.' )
		
		
		# -------------------------------------------------------------------------------------
		# COLUMN CRUD
		# -------------------------------------------------------------------------------------
		st.divider( )
		st.subheader( 'Labels' )
		with st.expander( label='Editor', expanded=True ):
			c1, c2 = st.columns( 2, border=True )
			with c1:
				drop_cols = st.multiselect( 'Columns to Drop', df_dataset.columns.tolist( ) )
				if st.button( 'Apply Column Drop' ):
					if len( drop_cols ) == len( df_dataset.columns ):
						st.error( 'Cannot Drop All Columns.' )
					else:
						st.session_state.df_dataset = df_dataset.drop( columns=drop_cols )
						log_step( f'Dropped Columns: {drop_cols}' )
						st.rerun( )
			
			with c2:
				rename_col = st.selectbox( 'Rename Column', [ '<None>' ] + df_dataset.columns.tolist( ) )
				new_name = st.text_input( 'New Column Name' )
				if st.button( 'Apply Rename' ):
					if rename_col != '<None>' and new_name:
						if new_name in df_dataset.columns:
							st.error( 'Column Name Already Exists.' )
						else:
							st.session_state.df_dataset = df_dataset.rename( columns={ rename_col: new_name } )
							log_step( f'Renamed {rename_col} → {new_name}' )
							st.rerun( )
							
			r1, r2 = st.columns( 2 )
			with r1:
				if st.button( 'Reset to Original' ):
					st.session_state.df_dataset = st.session_state.raw_df.copy( )
					st.session_state.pipeline_log.clear( )
					log_step( 'Reset dataset to original' )
					st.rerun( )
			
			with r2:
				st.download_button( 'Export Dataset (CSV)', st.session_state.df_dataset.to_csv( index=False ),
					'dataset.csv', 'text/csv' )
			
		# -------------------------------------------------------------------------------------
		# RESET / EXPORT
		# -------------------------------------------------------------------------------------
		st.divider( )
		
		for step in st.session_state.pipeline_log:
			st.write( f'• {step}' )

# ============================================
#  DESCRIPTIVE STATISTICS MODE
# ============================================
elif mode == 'Descriptive Statistics':
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.header( cfg.MODE[ 'Descriptive Statistics' ] )
		st.divider( )
		
		df_dataset = st.session_state.df_dataset
		df_numeric = clean_numeric( df_dataset.select_dtypes( include=[ np.number ] ) )
		if df_numeric.empty:
			st.stop( )
			
		num_c1, num_c2c2 = st.columns( [ 0.5, 0.5 ] , border=False )
		with num_c1:
			all_num_cols = df_numeric.columns.tolist( )
			vars_sel = st.multiselect( 'Select Numeric Variables', all_num_cols,
				default=default_pick( all_num_cols, 3 ) )
		
		for col in vars_sel:
			s = df_numeric[ col ].dropna( )
			st.subheader( f'Distribution & Shape — {col}' )
			c1, c2 = st.columns( 2, border=True )
			
			with c1:
				fig, ax = plt.subplots( figsize=(7, 5) )
				ax.hist( s, bins=30, edgecolor='black', alpha=0.85 )
				ax.set_title( f'Histogram — {col}' )
				ax.set_xlabel( col )
				ax.set_ylabel( 'Frequency' )
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
			
			with c2:
				fig, ax = plt.subplots( figsize=(7, 5) )
				stats.probplot( s, plot=ax )
				ax.set_title( f'Q–Q Plot — {col}' )
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
		
		st.subheader( 'Correlation Structure' )
		cor_c1, cor_c2 = st.columns( [ 0.5, 0.5 ] )
		with cor_c1:
			corr_vars = st.multiselect( 'Variables for Correlation',
				all_num_cols, default=default_pick( all_num_cols, 4 ) )
			
		c3, c4 = st.columns( 2, border=True )
		with c3:
			if len( corr_vars ) >= 2:
				df_correlation = analysis_fillna_mean( df_numeric[ corr_vars ] )
				corr = df_correlation.corr( )
				render_table( corr )
				
		with c4:
			fig, ax = plt.subplots( figsize=(7, 6) )
			im = ax.imshow( corr.values, cmap='coolwarm', vmin=-1, vmax=1 )
			fig.colorbar( im, ax=ax )
			ax.set_xticks( range( len( corr_vars ) ) )
			ax.set_yticks( range( len( corr_vars ) ) )
			ax.set_xticklabels( corr_vars, rotation=45, ha='right' )
			ax.set_yticklabels( corr_vars )
			ax.set_title( 'Correlation Heatmap' )
			fig.tight_layout( )
			st.pyplot( fig, use_container_width=True )
			plt.close( fig )
		
		st.subheader( 'Principal Component Analysis' )
		
		pca_c1, pca_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with pca_c1:
			pca_vars = st.multiselect( 'Select Components', all_num_cols,
				default=default_pick( all_num_cols, 4 ) )
		
		with pca_c2:
			n_comp = st.slider( 'Components', 2, min( 6, len( pca_vars ) ), 3 )
			
		if len( pca_vars ) >= 2:
			X = analysis_fillna_mean( df_numeric[ pca_vars ] )
			
			Xs = SKStandardScaler( ).fit_transform( X )
			pca = PCA( n_components=n_comp ).fit( Xs )
			
			df_explained = pd.DataFrame( { 'Component': [ f'PC{i + 1}' for i in range( n_comp ) ],
					'Explained Variance (%)': pca.explained_variance_ratio_ * 100 } )
			
		c5, c6 = st.columns( 2, border=True )
		with c5:
			render_table( df_explained )
		
		with c6:
			fig, ax = plt.subplots( figsize=(7, 5) )
			ax.bar( df_explained[ 'Component' ], df_explained[ 'Explained Variance (%)' ], edgecolor='black' )
			ax.set_ylabel( '% Variance Explained' )
			ax.set_title( 'PCA Variance Explained' )
			fig.tight_layout( )
			st.pyplot( fig, use_container_width=True )
			plt.close( fig )

# ============================================
# INFERENTIAL STATISTICS MODE
# ============================================
elif mode == 'Inferential Statistics':
	st.header( cfg.MODE[ 'Inferential Statistics' ] )
	st.divider( )
	
	df_dataset = st.session_state.df_dataset

	if df_dataset is None or df_dataset.empty:
		st.info( 'No data available.' )
		st.stop( )
	
	numeric_cols = st.session_state.numeric_cols
	categorical_cols = st.session_state.categorical_cols
	
	if not numeric_cols:
		st.info( 'No numeric variables available for inferential analysis.' )
		st.stop( )
	
	# ------------ VARIABLE SELECTION
	nml_c1, nml_c2 = st.columns( [ 0.5, 0.5 ], border=True, gap='medium' )
	col_group = None
	with nml_c1:
		st.markdown( '##### Normality Test' )
		col_y = st.selectbox( 'Select Numeric Outcome Variable', numeric_cols )
		y = df_dataset[ col_y ].dropna( )
		
		# ------------ NORMALITY TEST — SHAPIRO–WILK + Q–Q PLOT
		if len( y ) >= 3:
			stat, p_value = stats.shapiro( y )
			fig, ax = plt.subplots( figsize=(5, 5) )
			stats.probplot( y, plot=ax )
			ax.set_title( f'Q–Q Plot: {col_y} | Shapiro p = {p_value:.3g} | n = {len( y )}',
				fontsize=12, fontweight='bold', pad=10 )
			ax.ticklabel_format( style='plain', axis='both' )
			ax.get_lines( )[ 0 ].set_marker( 'o' )
			ax.get_lines( )[ 0 ].set_alpha( 0.7 )
			ax.get_lines( )[ 0 ].set_markeredgecolor( 'black' )
			fig.tight_layout( )
			st.pyplot( fig )
			plt.close( fig )
			st.text( f'Shapiro–Wilk W = {stat:.4f}, p = {p_value:.4g}' )
		else:
			st.info( 'Not enough observations for normality testing.' )
	
	with nml_c2:
		st.markdown( '##### Group Comparison' )
		col_group = None
		if categorical_cols:
			col_group = st.selectbox( 'Select Grouping Variable (optional)',
				[ '<None>' ] + categorical_cols )
			if col_group == '<None>':
				col_group = None
		
		# ------------ GROUP COMPARISON — ANOVA + KRUSKAL–WALLIS
		if col_group:
			grouped = [ grp[ col_y ].dropna( ).values for _, grp in
			            df_dataset.groupby( col_group ) ]
			valid_groups = [ g for g in grouped if len( g ) >= 2 ]
			if len( valid_groups ) >= 2:
				f_stat, p_anova = stats.f_oneway( *valid_groups )
				fig, ax = inferential_plot( title=f'ANOVA: {col_y} by {col_group}',
					subtitle=f'p = {p_anova:.3g}', figsize=(6, 4) )
				means = df_dataset.groupby( col_group )[ col_y ].mean( )
				means.plot( kind='bar', ax=ax, edgecolor='black', linewidth=1.2, alpha=0.8 )
				ax.set_xlabel( col_group )
				ax.set_ylabel( col_y )
				st.pyplot( fig )
				plt.close( fig )
				h_stat, p_kw = stats.kruskal( *valid_groups )
				A = f'ANOVA: F = {f_stat:.4f}, p = {p_anova:.4f}; '
				B = f'Kruskal–Wallis: H = {h_stat:.4f}, p- = {p_kw:.4g}'
				st.text( A + B )
		else:
			st.info( 'Not enough valid groups for group comparison.' )
			
	
	# ------------ CORRELATION ANALYSIS — PEARSON + SPEARMAN
	st.divider( )
	st.subheader( 'Correlation Analysis' )
	cor_c1, cor_c2 = st.columns( [ 0.5, 0.5 ], border=True )
	with cor_c1:
		col_x2 = st.selectbox( 'Select second numeric variable',
			[ c for c in numeric_cols if c != col_y ] )
		
		x = df_dataset[ col_x2 ]
		y = df_dataset[ col_y ]
	
	with cor_c2:
		mask = x.notna( ) & y.notna( )
		if mask.sum( ) >= 3:
			r_p, p_p = stats.pearsonr( x[ mask ], y[ mask ] )
			r_s, p_s = stats.spearmanr( x[ mask ], y[ mask ] )
		st.text( f'Pearson r = {r_p:.3f} (p={p_p:.3g}) ')
		st.text( f'Spearman ρ = {r_s:.3f} (p= {p_s:.3g})' )
	mask = x.notna( ) & y.notna( )
	if mask.sum( ) >= 3:
		r_p, p_p = stats.pearsonr( x[ mask ], y[ mask ] )
		r_s, p_s = stats.spearmanr( x[ mask ], y[ mask ] )
		fig, ax = inferential_plot( title=f'Correlation: {col_y} vs {col_x2}',
			figsize=(6, 4), ref_line=0.0 )
		ax.scatter( x[ mask ], y[ mask ], alpha=0.7, edgecolor='black' )
		ax.set_xlabel( col_x2 )
		ax.set_ylabel( col_y )
		st.pyplot( fig )
		plt.close( fig )
	else:
		st.info( 'Not enough paired observations for correlation.' )
	
		
	# ------------  CATEGORICAL ASSOCIATION — CHI-SQUARE + CRAMÉR’S V
	st.divider( )
	st.subheader( 'Categorical Association' )
	cat_c1, cat_c2 = st.columns( [ 0.5, 0.5 ], border=True )
	with cat_c1:
		if categorical_cols:
				col_cat1 = st.selectbox( 'Select First Categorical Variable', categorical_cols )
	
	with cat_c2:
		if categorical_cols:
			col_cat2 = st.selectbox( 'Select Second Categorical Variable',
						[ c for c in categorical_cols if c != col_cat1 ] )
			contingency = pd.crosstab( df_dataset[ col_cat1 ], df_dataset[ col_cat2 ] )
	
	st.divider( )
	
	if contingency.size > 0:
			chi2, p_chi, dof, _ = stats.chi2_contingency( contingency )
			n = contingency.values.sum( )
			cramers_v = np.sqrt( chi2 / (n * (min( contingency.shape ) - 1)) )
			st.write( f'Chi-square = {chi2:.4f}, p-value = {p_chi:.4g}, ramér’s V = '
			          f'{cramers_v:.4f}' )
			st.data_editor( contingency, use_container_width=True )
	else:
		st.info( 'Insufficient data for categorical association.' )

# ============================================
# ANOMALY DETECTION MODE
# ============================================
elif mode == 'Anomaly Detection':
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.header( cfg.MODE[ 'Anomaly Detection' ] )
		st.divider( )
		
		if st.session_state.df_dataset is None:
			st.info( 'No data loaded.' )
			st.stop( )
		
		df_dataset = st.session_state.df_dataset
		
		df_numeric = clean_numeric( df_dataset.select_dtypes( include=[ np.number ] ) )
		
		if df_numeric.empty:
			st.info( 'No usable numeric columns available for anomaly detection.' )
			st.stop( )
		
		all_num_cols = df_numeric.columns.tolist( )
		preferred = [ c for c in all_num_cols if c.lower( ) in ( 'py', 'cy', 'by' ) ]
		default_vars = preferred if preferred else default_pick( all_num_cols, 2 )
		vars_sel = st.multiselect( 'Variables to Analyze', all_num_cols, default=default_vars )
		if not vars_sel:
			st.info( 'Select at least one numeric variable to run anomaly detection.' )
			st.stop( )
		
		analysis_scale = st.checkbox( 'Use analysis-only standardization', value=False )
		df_analysis = df_numeric[ vars_sel ].copy( )
		if analysis_scale and len( vars_sel ) > 1:
			df_analysis[ : ] = SKStandardScaler( ).fit_transform( df_analysis.values )
		
		# -------------------------------------------------------------------------
		# Method Selection
		# -------------------------------------------------------------------------
		st.subheader( 'Detection Methods' )
		st.divider( )
		
		c_m1, c_m2 = st.columns( 2, border=True )
		
		with c_m1:
			use_z = st.checkbox( 'Z-Score', value=True )
			use_mz = st.checkbox( 'Modified Z-Score (MAD)', value=True )
			use_iqr = st.checkbox( 'IQR Fence', value=True )
		
		with c_m2:
			use_mahal = st.checkbox( 'Mahalanobis Distance', value=True )
			use_iforest = st.checkbox( 'Isolation Forest', value=True )
			use_lof = st.checkbox( 'Local Outlier Factor (LOF)', value=False )
		
		# -------------------------------------------------------------------------
		# Threshold Controls
		# -------------------------------------------------------------------------
		st.subheader( 'Thresholds' )
		st.divider( )
		
		c_t1, c_t2 = st.columns( 2, border=True )
		
		with c_t1:
			z_thresh = st.slider( 'Z / Modified Z threshold', 2.0, 5.0, 3.0, 0.1 )
			iqr_mult = st.slider( 'IQR multiplier', 1.0, 3.0, 1.5, 0.1 )
		
		with c_t2:
			lof_k = st.slider( 'LOF Neighbors (k)', 5, 50, 20, 1 )
			min_methods = st.slider( 'Consensus: minimum methods flagging a row', 1, 4, 1, 1 )
		
		# -------------------------------------------------------------------------
		# Run Detection
		# -------------------------------------------------------------------------
		df_anamolies = pd.DataFrame( index=df_analysis.index )
		
		# --- Univariate methods
		for col in vars_sel:
			s = df_analysis[ col ].dropna( )
			if s.empty:
				continue
			
			if use_z:
				z = (s - s.mean( ) ) / s.std( ) if s.std( ) else pd.Series( 0, index=s.index )
				df_anamolies[ f'{col}_z' ] = z.abs( ) >= z_thresh
			
			if use_mz:
				med = s.median( )
				mad = np.median( np.abs( s - med ) )
				if mad == 0:
					mz = pd.Series( 0, index=s.index )
				else:
					mz = 0.6745 * (s - med) / mad
					
				df_anamolies[ f'{col}_mz' ] = mz.abs( ) >= z_thresh
			
			if use_iqr:
				q1, q3 = s.quantile( 0.25 ), s.quantile( 0.75 )
				iqr = q3 - q1
				lo = q1 - iqr_mult * iqr
				hi = q3 + iqr_mult * iqr
				df_anamolies[ f'{col}_iqr' ] = (s < lo) | (s > hi)
		
		# --- Multivariate methods
		df_muliti = df_analysis.dropna( axis=0 )
		
		if df_muliti.shape[ 0 ] >= 10 and df_muliti.shape[ 1 ] >= 2:
			if use_mahal:
				cov = np.cov( df_muliti.values, rowvar=False )
				if np.linalg.det( cov ) != 0:
					inv_cov = np.linalg.inv( cov )
					mean = df_muliti.mean( ).values
					diffs = df_muliti.values - mean
					md = np.sqrt( np.einsum( 'ij,jk,ik->i', diffs, inv_cov, diffs ) )
					cutoff = np.sqrt( stats.chi2.ppf( 0.975, df_muliti.shape[ 1 ] ) )
					df_anamolies.loc[ df_muliti.index, 'mahal' ] = md > cutoff
			
			if use_iforest:
				from sklearn.ensemble import IsolationForest
				iso = IsolationForest( contamination='auto', random_state=42 )
				preds = iso.fit_predict( df_muliti.values )
				df_anamolies.loc[ df_muliti.index, 'iforest' ] = preds == -1
			
			if use_lof:
				from sklearn.neighbors import LocalOutlierFactor
				lof = LocalOutlierFactor( n_neighbors=lof_k )
				preds = lof.fit_predict( df_muliti.values )
				df_anamolies.loc[ df_muliti.index, 'lof' ] = preds == -1
		
		# -------------------------------------------------------------------------
		# Consensus & Output
		# -------------------------------------------------------------------------
		st.subheader( 'Outlier Summary' )
		st.divider( )
		if df_anamolies.empty:
			st.info( 'No anomalies detected under the selected methods and thresholds.' )
			st.stop( )
		
		df_anamolies = df_anamolies.fillna( False )
		df_anamolies[ 'methods_flagged' ] = df_anamolies.sum( axis=1 )
		anomalies = df_anamolies[ df_anamolies[ 'methods_flagged' ] >= min_methods ]
		
		c_o1, c_o2 = st.columns( 2, border=True )
		with c_o1:
			st.subheader( 'Flagged Observations' )
			render_table( anomalies.sort_values( 'methods_flagged', ascending=False ) )
		
		with c_o2:
			st.subheader( 'Flag Count Distribution' )
			fig, ax = plt.subplots( figsize=(7, 5) )
			anomalies[ 'methods_flagged' ].value_counts( ).sort_index( ).plot( kind='bar',
				ax=ax, edgecolor='black' )
			ax.set_xlabel( 'Number of Methods Flagging' )
			ax.set_ylabel( 'Observation Count' )
			ax.set_title( 'Consensus Strength' )
			fig.tight_layout( )
			st.pyplot( fig, use_container_width=True )
			plt.close( fig )
		
		# -------------------------------------------------------------------------
		# Visualization — Distribution with Anomalies
		# -------------------------------------------------------------------------
		st.subheader( 'Distributions with Anomalies Highlighted' )
		st.divider( )
		
		for col in vars_sel:
			if col not in df_analysis.columns:
				continue
			
			s = df_analysis[ col ]
			flagged_idx = anomalies.index.intersection( s.index )
			
			if flagged_idx.empty:
				continue
			
			c_v1, c_v2 = st.columns( 2 )
			with c_v1:
				fig, ax = plt.subplots( figsize=(7, 5) )
				ax.hist( s.dropna( ), bins=30, alpha=0.7, edgecolor='black' )
				ax.scatter( s.loc[ flagged_idx ], np.zeros( len( flagged_idx ) ),
					color='red', label='Anomalies' )
				ax.set_title( f'{col} — Histogram with Anomalies' )
				ax.legend( )
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
			
			with c_v2:
				fig, ax = plt.subplots( figsize=(7, 5) )
				ax.boxplot( s.dropna( ), vert=False )
				ax.scatter( s.loc[ flagged_idx ], np.ones( len( flagged_idx ) ), color='red' )
				ax.set_title( f'{col} — Boxplot with Anomalies' )
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
		
		# -------------------------------------------------------------------------
		# Export
		# -------------------------------------------------------------------------
		st.download_button( "Export Anomaly Table (CSV)", anomalies.to_csv( ),
			"anomalies.csv", "text/csv" )

# ============================================
# FEATURE ENGINEERING MODE
# ============================================
elif mode == 'Feature Engineering':
	st.header( cfg.MODE[ 'Feature Engineering' ] )
	st.divider( )
	
	st.caption( 'Apply Feature Transformations' )
	if 'df_dataset' not in locals( ) or df_dataset is None or df_dataset.empty:
		st.warning( 'No dataset loaded.' )
		st.stop( )
	
	df_original = df_dataset.copy( )
	
	if 'df_original' not in st.session_state:
		st.session_state[ 'df_original' ] = None
	
	# ------------------------------------------------------------------
	# Column classification (reuse existing logic)
	# ------------------------------------------------------------------
	numeric_columns = [ c for c in df_original.columns if df_original[ c ].dtype.kind in { 'i',  'f' } ]
	
	categorical_columns = [ c for c in df_original.columns if c not in numeric_columns ]
	
	# ------------------------------------------------------------------
	# Column selection
	# ------------------------------------------------------------------
	st.subheader( 'Column Selection' )	
	selected_columns = st.multiselect( 'Select columns for feature engineering',
		options=df_original.columns.tolist( ) )
	
	if not selected_columns:
		st.info( 'Select one or more columns to begin.' )
		st.stop( )
	
	df_features = df_original[ selected_columns ].copy( )
	
	# ======================================================================================
	# Missing Value Handling
	# ======================================================================================
	st.subheader( 'Missing Value Handling' )
	
	from imputers import ( MeanImputer, SimpleImputer, NearestImputer, IterativeImputer )
	impute_columns = st.multiselect( 'Columns to Impute', options=df_features.columns.tolist( ) )
	imputer_type = st.selectbox( 'Imputation Strategy',
		[ 'None', 'Mean', 'Median', 'Most Frequent', 'Nearest Neighbors', 'Iterative' ] )
	
	if imputer_type != 'None' and impute_columns:
		X_impute = df_features[ impute_columns ].to_numpy( )
		if imputer_type == 'Mean':
			imputer = MeanImputer( )
		elif imputer_type == 'Median':
			imputer = SimpleImputer( strategy='median' )
		elif imputer_type == 'Most Frequent':
			imputer = SimpleImputer( strategy='most_frequent' )
		elif imputer_type == 'Nearest Neighbors':
			imputer = NearestImputer( )
		elif imputer_type == 'Iterative':
			imputer = IterativeImputer( )
		
		X_imputed = imputer.train_transform( X_impute )		
		df_features[ impute_columns ] = X_imputed
		st.caption( 'Imputation Preview (First 5 Rows)' )
		st.data_editor( df_features.head( ) )
	
	# ======================================================================================
	# Encoding
	# ======================================================================================
	st.subheader( 'Encoding' )
	
	from encoders import ( OneHotEncoder, OrdinalEncoder, TargetEncoder )
	
	encode_columns = st.multiselect( 'Categorical Columns to Encode',
		options=[ c for c in df_features.columns if c in categorical_columns ] )
	
	encoding_type = st.selectbox( 'Encoding Method', [ 'None', 'One-Hot', 'Ordinal', 'Target' ] )
	
	if encoding_type != 'None' and encode_columns:
		X_encode = df_features[ encode_columns ].astype( str ).to_numpy( )
		
		if encoding_type == 'One-Hot':
			encoder = OneHotEncoder( sparse=False )
			X_encoded = encoder.train_transform( X_encode )
			df_encoded = pd.DataFrame( X_encoded, index=df_features.index )
			df_features = df_features.drop( columns=encode_columns )
			df_features = pd.concat( [ df_features, df_encoded ], axis=1 )
		elif encoding_type == 'Ordinal':
			encoder = OrdinalEncoder( )
			X_encoded = encoder.train_transform( X_encode )			
			df_features[ encode_columns ] = X_encoded
		elif encoding_type == 'Target':
			target_col = st.selectbox( 'Select target column', options=df_original.columns.tolist( ) )			
			y = df_original[ target_col ].to_numpy( )
			encoder = TargetEncoder( )
			X_encoded = encoder.train_transform( X_encode, y )			
			df_encoded = pd.DataFrame( X_encoded, index=df_features.index )
			df_features = df_features.drop( columns=encode_columns )
			df_features = pd.concat( [ df_features, df_encoded ], axis=1 )
		st.caption( 'Encoding Preview (First 5 Rows)' )
		st.data_editor( df_features.head( ) )
	
	# ======================================================================================
	# Scaling / Normalization
	# ======================================================================================
	st.subheader( 'Scaling & Normalization' )
	from scalers import ( StandardScaler, MinMaxScaler, RobustScaler, NormalScaler )
	
	scale_columns = st.multiselect( 'Numeric Columns to Scale',
		options=[ c for c in df_features.columns if c in numeric_columns ] )
	
	scaler_type = st.selectbox( 'Scaler', [ 'None', 'Standard', 'Min-Max', 'Robust', 'Normalize' ] )
	
	if scaler_type != 'None' and scale_columns:
		X_scale = df_features[ scale_columns ].to_numpy( )
		if scaler_type == 'Standard':
			scaler = StandardScaler( )
		elif scaler_type == 'Min-Max':
			scaler = MinMaxScaler( )
		elif scaler_type == 'Robust':
			scaler = RobustScaler( )
		elif scaler_type == 'Normalize':
			scaler = NormalScaler( )			
			X_scaled = scaler.train_transform( X_scale )
			df_features[ scale_columns ] = X_scaled
			st.caption( 'Scaling preview (First 5 rows)' )
			st.data_editor( df_features.head( ) )
			
	# ======================================================================================
	# Feature Generation
	# ======================================================================================
	st.subheader( 'Feature Generation' )	
	from encoders import PolynomialFeatures
	poly_columns = st.multiselect( 'Columns for Polynomial Features',
		options=[ c for c in df_features.columns if c in numeric_columns ] )
	
	poly_degree = st.slider( 'Polynomial Degree', min_value=2, max_value=4, value=2 )
	
	if poly_columns:
		X_poly = df_features[ poly_columns ].to_numpy( )
		poly = PolynomialFeatures( degree=poly_degree )
		X_poly_out = poly.train_transform( X_poly )
		df_polynomial = pd.DataFrame( X_poly_out, index=df_features.index )
		df_features = df_features.drop( columns=poly_columns )
		df_features = pd.concat( [ df_features, df_polynomial ], axis=1 )
		
		st.caption( 'Polynomial Feature preview (First 5 Rows)' )
		st.data_editor( df_features.head( ) )
	
	# ======================================================================================
	# Apply / Export
	# ======================================================================================
	st.subheader( 'Apply or Export' )
	
	if st.button( 'Apply Feature Engineering' ):
		st.session_state[ 'df_features' ] = df_features.copy( )
		st.success( 'Feature-Engineered Dataset Stored in Session State.' )
	
	st.download_button( label='Download Feature-Engineered Dataset (CSV)',
		data=df_features.to_csv( index=False ), file_name='feature_engineered_data.csv',
		mime='text/csv' )

# ============================================
# CLASSIFICATION MODE
# ============================================
elif mode == 'Classifications':
	st.header( cfg.MODE[ 'Classifications' ] )
	st.divider( )
	df_dataset = st.session_state.get( 'df_dataset', None )
	numeric_cols = st.session_state.get( 'numeric_cols', [ ] )
	categorical_cols = st.session_state.get( 'categorical_cols', [ ] )
	
	if df_dataset is None or df_dataset.empty:
		st.warning( '⚠️ No dataset loaded.' )
		st.stop( )
	
	if not numeric_cols or not categorical_cols:
		st.warning( '⚠️ Classification requires numeric features and a categorical target.' )
		st.stop( )
	
	# ------------------------------------------------------------------
	# TARGET & FEATURES
	# ------------------------------------------------------------------
	st.subheader( 'Target & Features' )	
	target = st.selectbox( 'Target (Categorical)', categorical_cols )
	features = st.multiselect( 'Feature Columns (Numeric)', numeric_cols, default=numeric_cols[ :3 ] )
	
	if not features:
		st.info( 'Please select at least one feature.' )
		st.stop( )
	
	X = df_dataset[ features ].to_numpy( )
	y = df_dataset[ target ].to_numpy( )
	
	# ------------------------------------------------------------------
	# MODEL SELECTION
	# ------------------------------------------------------------------
	from classifications import ( LogisticRegression, SupportVector, RandomForest,
		NearestNeighbor, BaggingModel, GradientBoost )
	
	model_map = \
	{
		'Logistic Regression': LogisticRegression,
		'Support Vector Machine': SupportVector,
		'Random Forest': RandomForest,
		'k-Nearest Neighbors': NearestNeighbor,
		'Bagging': BaggingModel,
		'Gradient Boosting': GradientBoost
	}
	
	st.subheader( 'Model Selection' )
	model_name = st.selectbox( 'Select Classification Model', list( model_map.keys( ) ) )	
	model = model_map[ model_name ]( )
	
	# ------------------------------------------------------------------
	# TRAIN / TEST SPLIT
	# ------------------------------------------------------------------
	st.subheader( 'Training Configuration' )
	test_sz = st.slider( 'Test set size (%)', 10, 20, 30, key='classifications-1' ) / 100.0
	random_state = st.number_input( 'Random state', value=42, step=1, key='classifications-2' )
	if st.button( '🚀 Train Classifier' ):
		try:
			X_train, X_test, y_train, y_test = model.split_data( X, y, size=test_sz, 
				random=random_state )			
			model.train( X_train, y_train )
			
			# ------------------------------------------------------------------
			# METRICS & ANALYSIS 
			# ------------------------------------------------------------------
			st.subheader( 'Model Performance' )			
			df_classifier = model.analyze( X_test, y_test )
			st.data_editor( df_classifier, use_container_width=True )
			
			# ------------------------------------------------------------------
			# CONFUSION MATRIX 
			# ------------------------------------------------------------------
			st.subheader( 'Confusion Matrix' )
			fig_cm = plt.figure( )
			model.confusion_matrix( X_test, y_test )
			st.pyplot( fig_cm )
			
			# ------------------------------------------------------------------
			# ROC CURVE (IF SUPPORTED)
			# ------------------------------------------------------------------
			if hasattr( model, 'roc_curve' ):
				st.subheader( 'ROC Curve' )
				fig_roc = plt.figure( )
				model.roc_curve( X_test, y_test )
				st.pyplot( fig_roc )
		
		except Exception as e:
			st.error( f'Classification failed: {e}' )

# ============================================
# REGRESSION MODE
# ============================================
elif mode == 'Regressions':
	st.header( cfg.MODE[ 'Regressions' ] )
	st.divider( )
	df_dataset = st.session_state.get( 'df_dataset', None )
	numeric_cols = st.session_state.get( 'numeric_cols', [ ] )
	
	if df_dataset is None or df_dataset.empty:
		st.warning( '⚠️ No Dataset Loaded.' )
		st.stop( )
	
	if not numeric_cols:
		st.warning( '⚠️ No numeric columns available for regression.' )
		st.stop( )
	
	# ------------------------------------------------------------------
	# TARGET & FEATURES
	# ------------------------------------------------------------------
	st.subheader( 'Target & Features' )
	
	target = st.selectbox( 'Target (Numeric)', numeric_cols )
	features = st.multiselect( 'Feature Columns (Numeric)',
		[ c for c in numeric_cols if c != target ],
		default=[ c for c in numeric_cols if c != target ][ :3 ] )
	
	if not features:
		st.info( 'Please select at least one feature.' )
		st.stop( )
	
	X = df_dataset[ features ].to_numpy( )
	y = df_dataset[ target ].to_numpy( )
	
	# ------------------------------------------------------------------
	# MODEL SELECTION
	# ------------------------------------------------------------------
	from regressions import ( LeastSquares, Ridge, LeastAngle, ElasticNet, BayesianRidge,
		SupportVector, GradientDescent, BaggingModel, GradientBoost, RandomForest,
		GaussianProcess )
	
	model_map = \
	{
			'Ordinary Least Squares': LeastSquares,
			'Ridge Regression': Ridge,
			'Lasso (Least Angle)': LeastAngle,
			'Elastic Net': ElasticNet,
			'Bayesian Ridge': BayesianRidge,
			'Support Vector': SupportVector,
			'Stochastic Gradient Descent': GradientDescent,
			'Bagging Regressor': BaggingModel,
			'Gradient Boosting': GradientBoost,
			'Random Forest': RandomForest,
			'Gaussian Process': GaussianProcess
	}
	
	st.subheader( 'Model Selection' )
	model_name = st.selectbox( 'Select Regression Model', list( model_map.keys( ) ) )
	model = model_map[ model_name ]( )
	
	# ------------------------------------------------------------------
	# TRAIN / TEST SPLIT
	# ------------------------------------------------------------------
	st.subheader( 'Training Configuration' )
	
	test_size = st.slider( 'Test Set Size (%)', 10, 20, 30, key='regressions-1' ) / 100.0
	random_state = st.number_input( 'Random state', value=42, step=1, key='regressions-2' )
	
	if st.button( '🚀 Train Model' ):
		try:
			X_train, X_test, y_train, y_test = model.split_data( X, y, size=test_size, 
				random=random_state )			
			model.train( X_train, y_train )
			
			# ------------------------------------------------------------------
			# METRICS
			# ------------------------------------------------------------------
			st.subheader( 'Model Performance' )
			
			df_regressor = model.analyze( X_test, y_test )
			st.data_editor( df_regressor, use_container_width=True )
			
			# ------------------------------------------------------------------
			# SCATTER PLOT (BUILT-IN)
			# ------------------------------------------------------------------
			st.subheader( 'Observed vs Predicted' )			
			fig = plt.figure( )
			model.scatter_plot( X_test, y_test )
			st.pyplot( fig )		
		except Exception as e:
			st.error( f'Regression failed: {e}' )

# ============================================
# CLUSTERING MODELS MODE
# ============================================
elif mode == 'Clustering':
	st.header( cfg.MODE[ 'Clustering' ] )
	st.divider( )
	st.caption( 'Explore Clustering Models' )
	
	# ------------------------------------------------------------------
	# Data source resolution
	# ------------------------------------------------------------------
	if 'df_features' in st.session_state and \
			st.session_state[ 'df_features' ] is not None:
		df_cluster = st.session_state[ 'df_features' ].copy( )
		st.info( 'Using Feature-Engineered Dataset.' )
	else:
		df_cluster = df_dataset.copy( )
		st.info( 'Using Original Dataset.' )
	
	if df_cluster is None or df_cluster.empty:
		st.warning( 'No Dataset Available for Clustering.' )
		st.stop( )
	
	# ------------------------------------------------------------------
	# Column classification (numeric only)
	# ------------------------------------------------------------------
	numeric_columns = [ c for c in df_cluster.columns
                    if df_cluster[ c ].dtype.kind in { 'i', 'f' } ]

	if len( numeric_columns ) < 2:
		st.warning( 'At least two numeric columns are required for clustering.' )
		st.stop( )
	
	# ------------------------------------------------------------------
	# Feature selection
	# ------------------------------------------------------------------
	st.subheader( 'Feature Selection' )
	feature_columns = st.multiselect( 'Select Features for Clustering', options=numeric_columns )
	
	if len( feature_columns ) < 2:
		st.info( 'Select at least two features to continue.' )
		st.stop( )
	
	X = df_cluster[ feature_columns ].to_numpy( )

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
	st.subheader( 'Clustering Model' )

	from clusters import (KMeans, DBSCAN, Agglomerative)
	
	model_name = st.selectbox( 'Clustering Algorithm',
		[ 'K-Means', 'DBSCAN', 'Agglomerative' ] )
	
	# ------------------------------------------------------------------
	# Model parameters
	# ------------------------------------------------------------------
	st.subheader( 'Model Parameters' )
	
	model = None

	if model_name == 'K-Means':
		n_clusters = st.number_input( 'Number of Clusters (K)', min_value=2, value=3 )
		model = KMeans( clusters=n_clusters )
	elif model_name == 'DBSCAN':
		eps = st.number_input( 'Epsilon (eps)', min_value=0.01, value=0.5 )
		min_samples = st.number_input( 'Min Samples', min_value=1, value=5 )
		model = DBSCAN( samples=min_samples )
	elif model_name == 'Agglomerative':
		n_clusters = st.number_input( 'Number of clusters', min_value=2, value=3 )
		linkage = st.selectbox( 'Linkage', [ 'ward', 'complete', 'average', 'single' ] )
		model = Agglomerative( n_clusters=n_clusters, linkage=linkage )
        
    # ------------------------------------------------------------------
    # Fit clustering model
    # ------------------------------------------------------------------
	st.subheader( 'Run Clustering' )
	
	if st.button( 'Run Clustering' ):
		labels = model.project( X )
		df_results = df_cluster.copy( )
		df_results[ 'Cluster' ] = labels		
		st.success( 'Clustering complete.' )
	
	# ------------------------------------------------------------------
	# Cluster summary
	# ------------------------------------------------------------------
		st.subheader( 'Cluster Summary' )
	
		cluster_counts = (df_results[ 'Cluster' ]
		.value_counts( )
		.rename( 'Count' )
		.reset_index( )
		.rename( columns={ 'index': 'Cluster' } ))
		
		st.data_editor( cluster_counts, use_container_width=True )
	
	# ------------------------------------------------------------------
	# Visualization
	# ------------------------------------------------------------------
	st.subheader( 'Cluster Visualization' )
	
	if len( feature_columns ) == 2:
		fig, ax = plt.subplots( )
		scatter = ax.scatter( df_results[ feature_columns[ 0 ] ],
			df_results[ feature_columns[ 1 ] ], c=df_results[ 'Cluster' ], alpha=0.7 )
		ax.set_xlabel( feature_columns[ 0 ] )
		ax.set_ylabel( feature_columns[ 1 ] )
		ax.set_title( 'Cluster Assignments' )
		st.pyplot( fig )
	else:
		st.info( 'Visualization limited to two features.' )
	
	# ------------------------------------------------------------------
	# Centroids (if available)
	# ------------------------------------------------------------------
	if hasattr( model, 'centroids_' ):
		st.subheader( 'Cluster Centroids' )
		df_centroid = pd.DataFrame( model.centroids_, columns=feature_columns )
		df_centroid.insert( 0, 'Cluster', range( len( df_centroid ) ) )
		st.data_editor( df_centroid, use_container_width=True )

# ============================================
# TIME SERIES MODE
# ============================================
elif mode == 'Time-Series':
	st.header( cfg.MODE[ 'Time-Series' ] )
	st.divider( )
	
	# ------------------------------------------------------------------
	# DATA VALIDATION
	# ------------------------------------------------------------------
	df_dataset = st.session_state.get( 'df_infer', None )
	numeric_cols = st.session_state.get( 'numeric_cols', [ ] )
	
	if df_dataset is None or df_dataset.empty:
		st.warning( '⚠️ No dataset loaded.' )
		st.stop( )
	
	if not numeric_cols:
		st.warning( '⚠️ No numeric columns available for time-series analysis.' )
		st.stop( )
	
	# ------------------------------------------------------------------
	# SERIES SELECTION
	# ------------------------------------------------------------------
	st.subheader( 'Time-Series Selection' )
	series_col = st.selectbox( 'Select Numeric Time-Series Column', numeric_cols )
	series = df_dataset[ series_col ].dropna( ).to_numpy( )
	if series.ndim != 1 or len( series ) < 10:
		st.warning( '⚠️ Selected series is too short for modeling.' )
		st.stop( )
	
	# ------------------------------------------------------------------
	# MODEL SELECTION
	# ------------------------------------------------------------------
	from forecasting import (LaggingSeries, ARIMA, SARIMA, ExpandingWindow)
	
	model_map = \
		{
				'Lagged Linear Regression': 'lag',
				'ARIMA': 'arima',
				'SARIMA': 'sarima'
		}
	
	st.subheader( 'Model Selection' )
	model_name = st.selectbox( 'Select time-series model', list( model_map.keys( ) ) )
	
	# ------------------------------------------------------------------
	# MODEL PARAMETERS
	# ------------------------------------------------------------------
	st.subheader( 'Model Parameters' )
	
	model = None
	
	if model_name == 'Lagged Linear Regression':
		lag = st.number_input( 'Lag order', min_value=1, value=5 )
		model = LaggingSeries( lag=lag )
	
	elif model_name == 'ARIMA':
		p = st.number_input( 'p (AR)', min_value=0, value=1 )
		d = st.number_input( 'd (I)', min_value=0, value=0 )
		q = st.number_input( 'q (MA)', min_value=0, value=0 )
		model = ARIMA( order=(p, d, q) )
	
	elif model_name == 'SARIMA':
		p = st.number_input( 'p (AR)', min_value=0, value=1 )
		d = st.number_input( 'd (I)', min_value=0, value=1 )
		q = st.number_input( 'q (MA)', min_value=0, value=1 )
		P = st.number_input( 'P (Seasonal AR)', min_value=0, value=0 )
		D = st.number_input( 'D (Seasonal I)', min_value=0, value=0 )
		Q = st.number_input( 'Q (Seasonal MA)', min_value=0, value=0 )
		s = st.number_input( 'Season Length', min_value=0, value=0 )
	
	model = SARIMA( order=(p, d, q), seasonal=(P, D, Q, s) )
	
	# ------------------------------------------------------------------
	# TRAIN / FORECAST
	# ------------------------------------------------------------------
	st.subheader( 'Train & Forecast' )
	forecast_horizon = st.number_input( 'Forecast Horizon (Steps)', min_value=1, value=5 )
	
	if st.button( '🚀 Run Time-Series Model' ):
		try:
			model.train( series )
			forecast = model.project( n_steps=forecast_horizon )
			st.subheader( 'Model Evaluation' )
			metrics = model.analyze( )
			st.data_editor( pd.DataFrame( metrics, index=[ 'Value' ] ).T, use_container_width=True )
			st.subheader( 'Observed vs Forecast' )
			fig, ax = plt.subplots( )
			ax.plot( series, label='Observed' )
			ax.plot( range( len( series ), len( series ) + len( forecast ) ), forecast,
				label='Forecast', linestyle='--' )
			ax.set_title( 'Time-Series Forecast' )
			ax.legend( )
			st.pyplot( fig )
		except Exception as e:
			st.error( f'Time-Series Modeling failed: {e}' )
	
	# ------------------------------------------------------------------
	# EXPANDING WINDOW CV (OPTIONAL DIAGNOSTIC)
	# ------------------------------------------------------------------
	st.subheader( 'Expanding Window Cross-Validation' )
	
	with st.expander( 'Show expanding-window splits' ):
		initial = st.number_input( 'Initial window size', min_value=10, value=30 )
		window = st.number_input( 'Test window size', min_value=1, value=10 )
		splitter = ExpandingWindow( initial=initial, windows=window )
		
		if st.button( 'Visualize CV Splits' ):
			fig = plt.figure( )
			splitter.visualize( series )
			st.pyplot( fig )
