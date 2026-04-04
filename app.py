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
from typing import List, Dict, Optional

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

pd.options.display.float_format = '{:,.4f}'.format

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
	num_cols = disp.select_dtypes( include=[ np.number ] ).columns
	disp[ float_cols ] = disp[ float_cols ].round( 4 )
	disp[ num_cols ] = df[ num_cols ].map( '{:,.2f}'.format )
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

def styled_scatter( ax: plt.Axes, x: np.ndarray, y: np.ndarray, series_index: int = 0,
		label: Optional[ str ] = None, size: int = 30, ) -> None:
	"""
	
		Purpose:
		________
		Draw a consistently styled scatter plot with clear point boundaries and
		visually distinct series.
	
		Parameters:
		___________
		ax : plt.Axes
			Matplotlib axes to draw on.
		x : np.ndarray
			X-coordinates of the points.
		y : np.ndarray
			Y-coordinates of the points.
		series_index : int, optional
			Index used to pick color and marker from predefined palettes.
		label : Optional[str], optional
			Legend label for the series, if any.
		size : int, optional
			Marker size for the scatter plot.
	
		Returns:
		________
		None
			This function draws on the provided axes in-place.
		
	"""
	color = cfg.PALETTE[ series_index % len( cfg.PALETTE ) ]
	marker = cfg.MARKERS[ series_index % len( cfg.MARKERS ) ]
	ax.scatter( x, y, s=size, alpha=0.9, edgecolors="#020617",
		linewidths=0.6, c=[ color ], marker=marker, label=label, )
	ax.grid( True, alpha=0.25 )

def auto_float_format( series: pd.Series, max_decimals: int = 4 ) -> str:
	"""
	
		Purpose:
		________
		Infer a reasonable float formatting pattern for a numeric series based on
		its scale, so large values are readable and decimals are not excessive.
	
		Parameters:
		___________
		series : pd.Series
			Series whose numeric magnitude is used to pick the format.
		max_decimals : int, optional
			Maximum number of decimal places allowed in the format string.
	
		Returns:
		________
		str
			A Python format string such as '{:,.2f}' appropriate for the series.
			
	"""
	s = pd.to_numeric( series, errors="coerce" )
	s = s.replace( [ np.inf, -np.inf ], np.nan ).dropna( )
	if s.empty:
		return "{:,.2f}"
	
	mag = float( np.nanpercentile( np.abs( s.values ), 95 ) )
	
	if mag >= 1e9:
		decimals = 0
	elif mag >= 1e6:
		decimals = 1
	elif mag >= 1e3:
		decimals = 2
	elif mag >= 1:
		decimals = 3
	else:
		decimals = 4
	
	decimals = min( decimals, max_decimals )
	return f"{{:,.{decimals}f}}"

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
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stMarkdownContainer"] h4,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h4 {
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
				# 1) Datetime: ONLY for object/string columns
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
								col, value=float( val) if pd.notna( val ) else 0.0 )
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
			fig, ax = plt.subplots( figsize=(6, 4.5) )
			type_counts.sort_values( ascending=False ).plot(
				kind='bar',
				ax=ax,
				width=0.75,
				edgecolor='#0f172a',
				linewidth=0.9
			)
			ax.set_title( 'Column Type Distribution', fontsize=12, fontweight='bold' )
			ax.set_xlabel( '' )
			ax.set_ylabel( 'Count' )
			ax.grid( axis='y', alpha=0.25, linestyle='--' )
			ax.spines[ 'top' ].set_visible( False )
			ax.spines[ 'right' ].set_visible( False )
			for container in ax.containers:
				ax.bar_label( container, padding=3, fontsize=9 )
			fig.tight_layout( )
			st.pyplot( fig )
			plt.close( fig )
		
		with v2:
			missing_pct = (df_dataset.isna( ).mean( ) * 100).sort_values( ascending=False )
			missing_pct = missing_pct[ missing_pct > 0 ].head( 10 )
			if not missing_pct.empty:
				fig, ax = plt.subplots( figsize=(6, 4.5) )
				missing_pct.sort_values( ascending=True ).plot(
					kind='barh',
					ax=ax,
					width=0.75,
					edgecolor='#0f172a',
					linewidth=0.9
				)
				ax.set_title( 'Top Columns by Missing %', fontsize=12, fontweight='bold' )
				ax.set_xlabel( 'Percent Missing' )
				ax.set_ylabel( '' )
				ax.grid( axis='x', alpha=0.25, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				for container in ax.containers:
					labels = [ f'{v:.1f}%' for v in
					           missing_pct.sort_values( ascending=True ).values ]
					ax.bar_label( container, labels=labels, padding=3, fontsize=9 )
				fig.tight_layout( )
				st.pyplot( fig )
				plt.close( fig )
			else:
				st.info( 'No Missing Values Detected.' )
		
		st.divider( )
		st.subheader( 'Cardinality' )
		v3, v4 = st.columns( 2, border=True )
		with v3:
			cardinality = df_dataset.nunique( dropna=True ).sort_values( ascending=False ).head( 10 )
			fig, ax = plt.subplots( figsize=(6, 4.5) )
			cardinality.sort_values( ascending=True ).plot(
				kind='barh',
				ax=ax,
				width=0.75,
				edgecolor='#0f172a',
				linewidth=0.9
			)
			ax.set_title( 'Top Columns by Cardinality', fontsize=12, fontweight='bold' )
			ax.set_xlabel( 'Unique Values' )
			ax.set_ylabel( '' )
			ax.grid( axis='x', alpha=0.25, linestyle='--' )
			ax.spines[ 'top' ].set_visible( False )
			ax.spines[ 'right' ].set_visible( False )
			for container in ax.containers:
				ax.bar_label( container, padding=3, fontsize=9 )
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
		# Probability Distributions
		# -------------------------------------------------------------------------------------
		st.divider( )
		st.subheader( 'Numeric Distributions' )
		
		numeric_dist_cols = [
				c for c in df_dataset.columns
				if pd.api.types.is_numeric_dtype( df_dataset[ c ] )
				   and not pd.api.types.is_bool_dtype( df_dataset[ c ] )
		]
		
		if not numeric_dist_cols:
			st.info( 'No numeric columns detected.' )
		else:
			st.caption( f'{len( numeric_dist_cols )} numeric column(s) detected.' )
			
			ctrl1, ctrl2, ctrl3 = st.columns( 3, border=True )
			with ctrl1:
				dist_bins = st.slider(
					'Bins',
					min_value=10,
					max_value=60,
					value=30,
					step=5,
					key='profile_numeric_dist_bins'
				)
			
			with ctrl2:
				show_kde = st.checkbox(
					'Show KDE Overlay',
					value=True,
					key='profile_numeric_dist_kde'
				)
			
			with ctrl3:
				dist_mode = st.radio(
					'Display',
					options=[ 'Density', 'Frequency' ],
					horizontal=True,
					key='profile_numeric_dist_mode'
				)
			
			st.markdown(
				"""
				<style>
				[data-testid="stMetricLabel"] p {
					font-size: 0.72rem;
				}
				
				[data-testid="stMetricValue"] {
					font-size: 0.95rem;
				}
				
				[data-testid="stMetric"] {
					padding-top: 0.10rem;
					padding-bottom: 0.10rem;
				}
				</style>
				""",
				unsafe_allow_html=True
			)
			
			stat_mode = 'density' if dist_mode == 'Density' else 'count'
			grid_cols = st.columns( 2, border=True )
			
			for i, col in enumerate( numeric_dist_cols ):
				with grid_cols[ i % 2 ]:
					s = pd.to_numeric( df_dataset[ col ], errors='coerce' )
					s = s.replace( [ np.inf, -np.inf ], np.nan ).dropna( )
					
					if s.empty:
						st.warning( f'{col}: no plottable numeric values.' )
						continue
					
					fig, ax = plt.subplots( figsize=(7, 4.5) )
					sns.histplot(
						s,
						bins=dist_bins,
						kde=show_kde,
						stat=stat_mode,
						ax=ax,
						edgecolor='#0f172a',
						line_kws={ 'linewidth': 2.0 } if show_kde else None
					)
					
					mean_val = float( s.mean( ) )
					median_val = float( s.median( ) )
					
					ax.axvline(
						mean_val,
						linestyle='--',
						linewidth=1.5,
						label=f'Mean: {mean_val:,.3f}'
					)
					ax.axvline(
						median_val,
						linestyle=':',
						linewidth=1.5,
						label=f'Median: {median_val:,.3f}'
					)
					
					ax.set_title( f'Distribution — {col}', fontsize=12, fontweight='bold' )
					ax.set_xlabel( col )
					ax.set_ylabel( 'Density' if stat_mode == 'density' else 'Frequency' )
					ax.grid( True, alpha=0.25, linestyle='--' )
					ax.spines[ 'top' ].set_visible( False )
					ax.spines[ 'right' ].set_visible( False )
					ax.legend( frameon=False, fontsize=9 )
					
					fig.tight_layout( )
					st.pyplot( fig )
					plt.close( fig )
					
					m1, m2, m3, m4 = st.columns( 4 )
					m1.metric( 'Count', f'{len( s ):,}' )
					m2.metric( 'Mean', f'{mean_val:,.3f}' )
					m3.metric( 'Median', f'{median_val:,.3f}' )
					m4.metric(
						'Std',
						f'{float( s.std( ddof=1 ) ):,.3f}' if len( s ) > 1 else '0.000'
					)
		
		
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
		st.header( cfg.MODE[ 'Descriptive Statistics' ], help=cfg.DESCRIPTIVE_STATISTICS )
		st.divider( )
		
		df_dataset = st.session_state.df_dataset
		df_numeric = clean_numeric( df_dataset.select_dtypes( include=[ np.number ] ) )
		
		if df_numeric.empty:
			st.info( 'No numeric variables available for descriptive analysis.' )
			st.stop( )
		
		all_num_cols = df_numeric.columns.tolist( )
		
		st.markdown(
			"""
			<style>
			[data-testid="stMetricLabel"] p {
				font-size: 0.72rem;
			}
			
			[data-testid="stMetricValue"] {
				font-size: 0.95rem;
			}
			
			[data-testid="stMetric"] {
				padding-top: 0.10rem;
				padding-bottom: 0.10rem;
			}
			</style>
			""",
			unsafe_allow_html=True
		)
		
		num_c1, num_c2 = st.columns( [ 0.5, 0.5 ], border=False )
		with num_c1:
			vars_sel = st.multiselect(
				'Select Numeric Variables',
				all_num_cols,
				default=default_pick( all_num_cols, 3 )
			)
			
		st.divider( )
		
		st.subheader( 'Descriptive Summary' )
		
		sum_c1, sum_c2 = st.columns( [ 0.55, 0.45 ], border=False )
		with sum_c1:
			summary_vars = st.multiselect(
				'Variables for Summary Table',
				all_num_cols,
				default=all_num_cols[ : min( 8, len( all_num_cols ) ) ],
				key='desc_summary_vars'
			)
		
		with sum_c2:
			show_percentiles = st.checkbox(
				'Include Percentiles',
				value=True,
				key='desc_summary_percentiles'
			)
		
		if summary_vars:
			df_summary_source = df_numeric[ summary_vars ].copy( )
			percentiles = [ 0.05, 0.25, 0.50, 0.75, 0.95 ] if show_percentiles else None
			
			df_descriptive = df_summary_source.describe( percentiles=percentiles ).T.reset_index( )
			df_descriptive = df_descriptive.rename( columns={ 'index': 'Variable' } )
			
			df_descriptive[ 'Variance' ] = df_summary_source.var( ddof=1 ).values
			df_descriptive[ 'Missing' ] = df_dataset[ summary_vars ].isna( ).sum( ).values
			df_descriptive[ 'Missing %' ] = (
					df_dataset[ summary_vars ].isna( ).mean( ).values * 100.0
			)
			df_descriptive[ 'Skew' ] = df_summary_source.skew( ).values
			df_descriptive[ 'Kurtosis' ] = df_summary_source.kurtosis( ).values
			df_descriptive[ 'Zeros' ] = (df_summary_source == 0).sum( ).values
			df_descriptive[ 'Zeros %' ] = (
					(df_summary_source == 0).mean( ).values * 100.0
			)
			
			ordered_cols = [ 'Variable', 'count', 'mean', 'std', 'Variance', 'min' ]
			if show_percentiles:
				for pcol in [ '5%', '25%', '50%', '75%', '95%' ]:
					if pcol in df_descriptive.columns:
						ordered_cols.append( pcol )
			ordered_cols += [ 'max', 'Missing', 'Missing %', 'Zeros', 'Zeros %', 'Skew', 'Kurtosis' ]
			ordered_cols = [ c for c in ordered_cols if c in df_descriptive.columns ]
			df_descriptive = df_descriptive[ ordered_cols ]
			
			for c in df_descriptive.columns:
				if c != 'Variable':
					df_descriptive[ c ] = pd.to_numeric( df_descriptive[ c ], errors='coerce' )
			
			column_config = {
					'Variable': st.column_config.TextColumn( 'Variable', width='medium' ),
					'count': st.column_config.NumberColumn( 'Count', format='%.0f' ),
					'mean': st.column_config.NumberColumn( 'Mean', format='%.4f' ),
					'std': st.column_config.NumberColumn( 'Std', format='%.4f' ),
					'Variance': st.column_config.NumberColumn( 'Variance', format='%.4f' ),
					'min': st.column_config.NumberColumn( 'Min', format='%.4f' ),
					'5%': st.column_config.NumberColumn( 'P5', format='%.4f' ),
					'25%': st.column_config.NumberColumn( 'P25', format='%.4f' ),
					'50%': st.column_config.NumberColumn( 'Median', format='%.4f' ),
					'75%': st.column_config.NumberColumn( 'P75', format='%.4f' ),
					'95%': st.column_config.NumberColumn( 'P95', format='%.4f' ),
					'max': st.column_config.NumberColumn( 'Max', format='%.4f' ),
					'Missing': st.column_config.NumberColumn( 'Missing', format='%.0f' ),
					'Missing %': st.column_config.NumberColumn( 'Missing %', format='%.2f' ),
					'Zeros': st.column_config.NumberColumn( 'Zeros', format='%.0f' ),
					'Zeros %': st.column_config.NumberColumn( 'Zeros %', format='%.2f' ),
					'Skew': st.column_config.NumberColumn( 'Skew', format='%.4f' ),
					'Kurtosis': st.column_config.NumberColumn( 'Kurtosis', format='%.4f' )
			}
			
			column_config = { k: v for k, v in column_config.items( ) if k in df_descriptive.columns }
			
			st.data_editor(
				df_descriptive,
				use_container_width=True,
				hide_index=True,
				disabled=True,
				column_config=column_config,
				key='desc_summary_editor'
			)
		else:
			st.info( 'Select one or more numeric variables to display descriptive statistics.' )
		
		st.divider( )
	
		with num_c2:
			dist_bins = st.slider(
				'Distribution Bins',
				min_value=10,
				max_value=60,
				value=30,
				step=5,
				key='desc_dist_bins'
			)
		
		for col in vars_sel:
			s = pd.to_numeric( df_numeric[ col ], errors='coerce' )
			s = s.replace( [ np.inf, -np.inf ], np.nan ).dropna( )
			
			if s.empty:
				st.warning( f'{col}: no plottable numeric values.' )
				continue
			
			st.subheader( f'Distribution & Shape — {col}' )
			c1, c2 = st.columns( 2, border=True )
			
			with c1:
				fig, ax = plt.subplots( figsize=(7, 4.75) )
				sns.histplot(
					s,
					bins=dist_bins,
					kde=True,
					stat='count',
					ax=ax,
					edgecolor='#0f172a',
					line_kws={ 'linewidth': 2.0 }
				)
				
				mean_val = float( s.mean( ) )
				median_val = float( s.median( ) )
				
				ax.axvline(
					mean_val,
					linestyle='--',
					linewidth=1.5,
					label=f'Mean: {mean_val:,.3f}'
				)
				ax.axvline(
					median_val,
					linestyle=':',
					linewidth=1.5,
					label=f'Median: {median_val:,.3f}'
				)
				
				ax.set_title( f'Histogram — {col}', fontsize=12, fontweight='bold' )
				ax.set_xlabel( col )
				ax.set_ylabel( 'Frequency' )
				ax.grid( True, alpha=0.25, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				ax.legend( frameon=False, fontsize=9 )
				
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
				
				m1, m2, m3, m4 = st.columns( 4 )
				m1.metric( 'Count', f'{len( s ):,}' )
				m2.metric( 'Mean', f'{mean_val:,.3f}' )
				m3.metric( 'Median', f'{median_val:,.3f}' )
				m4.metric( 'Std', f'{float( s.std( ddof=1 ) ):,.3f}' if len( s ) > 1 else '0.000' )
			
			with c2:
				fig, ax = plt.subplots( figsize=(7, 4.75) )
				stats.probplot( s, plot=ax )
				ax.set_title( f'Q–Q Plot — {col}', fontsize=12, fontweight='bold' )
				ax.grid( True, alpha=0.20, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				
				if len( ax.get_lines( ) ) >= 1:
					ax.get_lines( )[ 0 ].set_marker( 'o' )
					ax.get_lines( )[ 0 ].set_alpha( 0.72 )
					ax.get_lines( )[ 0 ].set_markeredgecolor( 'black' )
				
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
				
				try:
					if 3 <= len( s ) <= 5000:
						shapiro_stat, shapiro_p = stats.shapiro( s )
						q1, q2, q3 = st.columns( 3 )
						q1.metric( 'Skew', f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
						q2.metric( 'Kurtosis', f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
						q3.metric( 'Shapiro p', f'{shapiro_p:,.4f}' )
					else:
						q1, q2, q3 = st.columns( 3 )
						q1.metric( 'Skew', f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
						q2.metric( 'Kurtosis', f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
						q3.metric( 'Shapiro p', 'n/a' )
				except Exception:
					q1, q2, q3 = st.columns( 3 )
					q1.metric( 'Skew', f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
					q2.metric( 'Kurtosis', f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
					q3.metric( 'Shapiro p', 'n/a' )
		
		st.divider( )
		st.subheader( 'Correlation Structure', help=cfg.CORRELATION_STRUCTURE )
		
		cor_c1, cor_c2 = st.columns( [ 0.5, 0.5 ], border=False )
		with cor_c1:
			corr_vars = st.multiselect(
				'Variables for Correlation',
				all_num_cols,
				default=default_pick( all_num_cols, 4 )
			)
		
		with cor_c2:
			corr_method = st.radio(
				'Method',
				options=[ 'Pearson', 'Spearman' ],
				horizontal=True,
				key='desc_corr_method'
			)
		
		c3, c4 = st.columns( 2, border=True )
		if len( corr_vars ) >= 2:
			df_correlation = analysis_fillna_mean( df_numeric[ corr_vars ] )
			corr = df_correlation.corr( method=corr_method.lower( ) )
			
			with c3:
				render_table( corr )
			
			with c4:
				fig, ax = plt.subplots( figsize=(7, 6) )
				sns.heatmap(
					corr,
					ax=ax,
					cmap='coolwarm',
					vmin=-1,
					vmax=1,
					center=0,
					annot=True,
					fmt='.2f',
					square=False,
					linewidths=0.5,
					cbar_kws={ 'shrink': 0.85, 'label': 'Correlation' }
				)
				ax.set_title(
					f'Correlation Heatmap — {corr_method}',
					fontsize=12,
					fontweight='bold',
					pad=10
				)
				ax.set_xticklabels( ax.get_xticklabels( ), rotation=45, ha='right' )
				ax.set_yticklabels( ax.get_yticklabels( ), rotation=0 )
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
		else:
			with c3:
				st.info( 'Select at least two numeric variables.' )
			with c4:
				st.caption( 'Heatmap will appear here once at least two variables are selected.' )
		
		st.divider( )
		st.subheader( 'Principal Component Analysis', help=cfg.PCA )
		
		pca_c1, pca_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with pca_c1:
			pca_vars = st.multiselect(
				'Select Components',
				all_num_cols,
				default=default_pick( all_num_cols, 4 )
			)
		
		with pca_c2:
			max_components = max( 2, min( 6, len( pca_vars ) ) ) if pca_vars else 2
			n_comp = st.slider( 'Components', 2, max_components, min( 3, max_components ) )
		
		c5, c6 = st.columns( 2, border=True )
		if len( pca_vars ) >= 2:
			X = analysis_fillna_mean( df_numeric[ pca_vars ] )
			Xs = SKStandardScaler( ).fit_transform( X )
			pca = PCA( n_components=n_comp ).fit( Xs )
			
			df_explained = pd.DataFrame(
				{
						'Component': [ f'PC{i + 1}' for i in range( n_comp ) ],
						'Explained Variance (%)': pca.explained_variance_ratio_ * 100
				}
			)
			
			with c5:
				render_table( df_explained )
			
			with c6:
				fig, ax = plt.subplots( figsize=(7, 5) )
				bars = ax.bar(
					df_explained[ 'Component' ],
					df_explained[ 'Explained Variance (%)' ],
					edgecolor='#0f172a',
					linewidth=0.9
				)
				ax.set_ylabel( '% Variance Explained' )
				ax.set_title( 'PCA Variance Explained', fontsize=12, fontweight='bold' )
				ax.grid( axis='y', alpha=0.25, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				ax.bar_label( bars, fmt='%.1f', padding=3, fontsize=9 )
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
		else:
			with c5:
				st.info( 'Select at least two numeric variables for PCA.' )
			with c6:
				st.caption( 'Explained variance chart will appear here once at least two variables are selected.' )

# ============================================
# INFERENTIAL STATISTICS MODE
# ============================================
elif mode == 'Inferential Statistics':
	st.header( cfg.MODE[ 'Inferential Statistics' ], help=cfg.INFERENTIAL_STATISTICS )
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
	
	st.markdown(
		"""
		<style>
		[data-testid="stMetricLabel"] p {
			font-size: 0.72rem;
		}
		
		[data-testid="stMetricValue"] {
			font-size: 0.95rem;
		}
		
		[data-testid="stMetric"] {
			padding-top: 0.10rem;
			padding-bottom: 0.10rem;
		}
		</style>
		""",
		unsafe_allow_html=True
	)
	# -------------------------------------------------------------------------------------
	# INFERENTIAL SUMMARY
	# -------------------------------------------------------------------------------------
	st.subheader( 'Inferential Summary' )
	
	sum_r1c1, sum_r1c2, sum_r1c3 = st.columns( 3, border=False )
	with sum_r1c1:
		summary_y = st.selectbox(
			'Summary Outcome Variable',
			numeric_cols,
			key='infer_summary_y'
		)
	
	with sum_r1c2:
		summary_x = st.selectbox(
			'Summary Second Numeric Variable',
			[ '<None>' ] + [ c for c in numeric_cols if c != summary_y ],
			key='infer_summary_x'
		)
		if summary_x == '<None>':
			summary_x = None
	
	with sum_r1c3:
		if categorical_cols:
			summary_group = st.selectbox(
				'Summary Grouping Variable',
				[ '<None>' ] + categorical_cols,
				key='infer_summary_group'
			)
			if summary_group == '<None>':
				summary_group = None
		else:
			summary_group = None
			st.caption( 'No categorical grouping variables available.' )
	
	sum_r2c1, sum_r2c2 = st.columns( 2, border=False )
	with sum_r2c1:
		if len( categorical_cols ) >= 2:
			summary_cat1 = st.selectbox(
				'Summary First Categorical Variable',
				categorical_cols,
				key='infer_summary_cat1'
			)
		else:
			summary_cat1 = None
			st.caption( 'At least two categorical variables are required.' )
	
	with sum_r2c2:
		if summary_cat1 and len( categorical_cols ) >= 2:
			summary_cat2 = st.selectbox(
				'Summary Second Categorical Variable',
				[ c for c in categorical_cols if c != summary_cat1 ],
				key='infer_summary_cat2'
			)
		else:
			summary_cat2 = None
	
	infer_rows = [ ]
	
	# -----------------------------------------------------------------
	# Normality Summary
	# -----------------------------------------------------------------
	summary_series = pd.to_numeric( df_dataset[ summary_y ], errors='coerce' ).dropna( )
	if len( summary_series ) >= 3:
		try:
			shapiro_stat, shapiro_p = stats.shapiro( summary_series )
			infer_rows.append(
				{
						'Analysis': 'Outcome Distribution',
						'Test': 'Shapiro-Wilk',
						'Statistic': shapiro_stat,
						'P-Value': shapiro_p,
						'DoF': np.nan,
						'Effect Size': np.nan,
						'N': float( len( summary_series ) ),
						'Notes': 'Normality assessment'
				}
			)
		except Exception:
			pass
	
	# -----------------------------------------------------------------
	# Group Comparison Summary
	# -----------------------------------------------------------------
	if summary_group:
		df_group_summary = df_dataset[ [ summary_group, summary_y ] ].copy( )
		df_group_summary[ summary_y ] = pd.to_numeric(
			df_group_summary[ summary_y ], errors='coerce' )
		df_group_summary = df_group_summary.dropna( subset=[ summary_group, summary_y ] )
		
		group_arrays = [ grp[ summary_y ].values for _, grp in
		                 df_group_summary.groupby( summary_group ) ]
		valid_group_arrays = [ g for g in group_arrays if len( g ) >= 2 ]
		
		if len( valid_group_arrays ) >= 2:
			try:
				f_stat, p_anova = stats.f_oneway( *valid_group_arrays )
				infer_rows.append(
					{
							'Analysis': 'Group Comparison',
							'Test': 'One-Way ANOVA',
							'Statistic': f_stat,
							'P-Value': p_anova,
							'DoF': float( len( valid_group_arrays ) - 1 ),
							'Effect Size': np.nan,
							'N': float( sum( len( g ) for g in valid_group_arrays ) ),
							'Notes': f'{summary_y} by {summary_group}'
					}
				)
			except Exception:
				pass
			
			try:
				h_stat, p_kw = stats.kruskal( *valid_group_arrays )
				infer_rows.append(
					{
							'Analysis': 'Group Comparison',
							'Test': 'Kruskal-Wallis',
							'Statistic': h_stat,
							'P-Value': p_kw,
							'DoF': float( len( valid_group_arrays ) - 1 ),
							'Effect Size': np.nan,
							'N': float( sum( len( g ) for g in valid_group_arrays ) ),
							'Notes': f'{summary_y} by {summary_group}'
					}
				)
			except Exception:
				pass
	
	# -----------------------------------------------------------------
	# Correlation Summary
	# -----------------------------------------------------------------
	if summary_x:
		x_summary = pd.to_numeric( df_dataset[ summary_x ], errors='coerce' )
		y_summary = pd.to_numeric( df_dataset[ summary_y ], errors='coerce' )
		pair_mask = x_summary.notna( ) & y_summary.notna( )
		
		if pair_mask.sum( ) >= 3:
			try:
				pearson_r, pearson_p = stats.pearsonr(
					x_summary[ pair_mask ], y_summary[ pair_mask ] )
				infer_rows.append(
					{
							'Analysis': 'Association',
							'Test': 'Pearson Correlation',
							'Statistic': pearson_r,
							'P-Value': pearson_p,
							'DoF': float( pair_mask.sum( ) - 2 ),
							'Effect Size': abs( pearson_r ),
							'N': float( pair_mask.sum( ) ),
							'Notes': f'{summary_y} vs {summary_x}'
					}
				)
			except Exception:
				pass
			
			try:
				spearman_rho, spearman_p = stats.spearmanr(
					x_summary[ pair_mask ],
					y_summary[ pair_mask ]
				)
				infer_rows.append(
					{
							'Analysis': 'Association',
							'Test': 'Spearman Correlation',
							'Statistic': spearman_rho,
							'P-Value': spearman_p,
							'DoF': np.nan,
							'Effect Size': abs( spearman_rho ),
							'N': float( pair_mask.sum( ) ),
							'Notes': f'{summary_y} vs {summary_x}'
					}
				)
			except Exception:
				pass
	
	# -----------------------------------------------------------------
	# Categorical Association Summary
	# -----------------------------------------------------------------
	if summary_cat1 and summary_cat2:
		contingency_summary = pd.crosstab( df_dataset[ summary_cat1 ], df_dataset[ summary_cat2 ] )
		
		if not contingency_summary.empty and contingency_summary.shape[ 0 ] >= 2 and \
				contingency_summary.shape[ 1 ] >= 2:
			try:
				chi2_stat, chi2_p, chi2_dof, expected = stats.chi2_contingency( contingency_summary )
				n_total = contingency_summary.to_numpy( ).sum( )
				phi2 = chi2_stat / n_total if n_total > 0 else np.nan
				r_dim, c_dim = contingency_summary.shape
				cramers_v = (
						np.sqrt( phi2 / min( c_dim - 1, r_dim - 1 ) )
						if min( c_dim - 1, r_dim - 1 ) > 0
						else np.nan
				)
				
				infer_rows.append(
					{
							'Analysis': 'Categorical Association',
							'Test': 'Chi-Square',
							'Statistic': chi2_stat,
							'P-Value': chi2_p,
							'DoF': float( chi2_dof ),
							'Effect Size': cramers_v,
							'N': float( n_total ),
							'Notes': f'{summary_cat1} vs {summary_cat2}'
					}
				)
			except Exception:
				pass
	
	if infer_rows:
		df_infer_summary = pd.DataFrame( infer_rows )
		
		for c in [ 'Statistic', 'P-Value', 'DoF', 'Effect Size', 'N' ]:
			if c in df_infer_summary.columns:
				df_infer_summary[ c ] = pd.to_numeric( df_infer_summary[ c ], errors='coerce' )
		
		infer_column_config = {
				'Analysis': st.column_config.TextColumn( 'Analysis', width='medium' ),
				'Test': st.column_config.TextColumn( 'Test', width='medium' ),
				'Statistic': st.column_config.NumberColumn( 'Statistic', format='%.4f' ),
				'P-Value': st.column_config.NumberColumn( 'P-Value', format='%.4g' ),
				'DoF': st.column_config.NumberColumn( 'DoF', format='%.0f' ),
				'Effect Size': st.column_config.NumberColumn( 'Effect Size', format='%.4f' ),
				'N': st.column_config.NumberColumn( 'N', format='%.0f' ),
				'Notes': st.column_config.TextColumn( 'Notes', width='large' )
		}
		
		st.data_editor(
			df_infer_summary,
			use_container_width=True,
			hide_index=True,
			disabled=True,
			column_config=infer_column_config,
			key='infer_summary_editor'
		)
	else:
		st.info( 'Unable to compute inferential summary for the current selections.' )
	
	st.divider( )
	
	# -------------------------------------------------------------------------------------
	# NORMALITY + GROUP COMPARISON
	# -------------------------------------------------------------------------------------
	nml_c1, nml_c2 = st.columns( [ 0.5, 0.5 ], border=True, gap='medium' )
	col_group = None
	
	with nml_c1:
		st.subheader( 'Normality Test', help=cfg.NORMALITY_TESTING )
		col_y = st.selectbox( 'Select Numeric Outcome Variable', numeric_cols )
		y = pd.to_numeric( df_dataset[ col_y ], errors='coerce' ).dropna( )
		
		if len( y ) >= 3:
			stat, p_value = stats.shapiro( y )
			
			fig, ax = plt.subplots( figsize=(6.25, 5.25) )
			stats.probplot( y, plot=ax )
			ax.set_title( f'Q–Q Plot — {col_y}', fontsize=12, fontweight='bold', pad=10 )
			ax.grid( True, alpha=0.20, linestyle='--' )
			ax.spines[ 'top' ].set_visible( False )
			ax.spines[ 'right' ].set_visible( False )
			ax.ticklabel_format( style='plain', axis='both' )
			
			if len( ax.get_lines( ) ) >= 1:
				ax.get_lines( )[ 0 ].set_marker( 'o' )
				ax.get_lines( )[ 0 ].set_alpha( 0.72 )
				ax.get_lines( )[ 0 ].set_markeredgecolor( 'black' )
			
			fig.tight_layout( )
			st.pyplot( fig )
			plt.close( fig )
			
			m1, m2, m3 = st.columns( 3 )
			m1.metric( 'Count', f'{len( y ):,}' )
			m2.metric( 'Shapiro W', f'{stat:,.4f}' )
			m3.metric( 'Shapiro p', f'{p_value:,.4g}' )
			
			if p_value < 0.05:
				st.caption( 'Distribution departs from normality at α = 0.05.' )
			else:
				st.caption( 'Distribution does not significantly depart from normality at α = 0.05.' )
		else:
			st.info( 'Not enough observations for normality testing.' )
	
	with nml_c2:
		st.subheader( 'Group Comparison' )
		
		if categorical_cols:
			col_group = st.selectbox( 'Select Grouping Variable (optional)',
				[ '<None>' ] + categorical_cols )
			if col_group == '<None>':
				col_group = None
		
		if col_group:
			df_group = df_dataset[ [ col_group, col_y ] ].copy( )
			df_group[ col_y ] = pd.to_numeric( df_group[ col_y ], errors='coerce' )
			df_group = df_group.dropna( subset=[ col_group, col_y ] )
			
			grouped = [ grp[ col_y ].values for _, grp in df_group.groupby( col_group ) ]
			valid_groups = [ g for g in grouped if len( g ) >= 2 ]
			
			if len( valid_groups ) >= 2:
				f_stat, p_anova = stats.f_oneway( *valid_groups )
				h_stat, p_kw = stats.kruskal( *valid_groups )
				
				fig, ax = plt.subplots( figsize=(6.5, 5.25) )
				sns.boxplot(
					data=df_group,
					x=col_group,
					y=col_y,
					ax=ax
				)
				sns.stripplot(
					data=df_group,
					x=col_group,
					y=col_y,
					ax=ax,
					color='black',
					alpha=0.45,
					size=4
				)
				ax.set_title( f'Group Comparison — {col_y} by {col_group}',
					fontsize=12, fontweight='bold', pad=10 )
				ax.set_xlabel( col_group )
				ax.set_ylabel( col_y )
				ax.grid( axis='y', alpha=0.20, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				ax.tick_params( axis='x', rotation=30 )
				fig.tight_layout( )
				st.pyplot( fig )
				plt.close( fig )
				
				g1, g2, g3, g4 = st.columns( 4 )
				g1.metric( 'Groups', f'{len( valid_groups ):,}' )
				g2.metric( 'ANOVA F', f'{f_stat:,.4f}' )
				g3.metric( 'ANOVA p', f'{p_anova:,.4g}' )
				g4.metric( 'Kruskal p', f'{p_kw:,.4g}' )
				
				st.caption(
					f'Kruskal–Wallis H = {h_stat:.4f}. '
					f'Use the nonparametric result when normality or homoscedasticity is doubtful.'
				)
			else:
				st.info( 'Not enough valid groups for group comparison.' )
		else:
			st.info( 'Select a grouping variable to compare groups.' )
	
	# -------------------------------------------------------------------------------------
	# CORRELATION ANALYSIS
	# -------------------------------------------------------------------------------------
	st.divider( )
	st.subheader( 'Correlation Analysis' )
	
	cor_c1, cor_c2 = st.columns( [ 0.5, 0.5 ], border=True )
	with cor_c1:
		candidate_x = [ c for c in numeric_cols if c != col_y ]
		if not candidate_x:
			st.info( 'A second numeric variable is required for correlation analysis.' )
			col_x2 = None
		else:
			col_x2 = st.selectbox( 'Select Second Numeric Variable', candidate_x )
	
	with cor_c2:
		if col_x2:
			x = pd.to_numeric( df_dataset[ col_x2 ], errors='coerce' )
			y2 = pd.to_numeric( df_dataset[ col_y ], errors='coerce' )
			mask = x.notna( ) & y2.notna( )
			
			if mask.sum( ) >= 3:
				r_p, p_p = stats.pearsonr( x[ mask ], y2[ mask ] )
				r_s, p_s = stats.spearmanr( x[ mask ], y2[ mask ] )
				
				r1, r2, r3, r4 = st.columns( 4 )
				r1.metric( 'Pairs', f'{int( mask.sum( ) ):,}' )
				r2.metric( 'Pearson r', f'{r_p:,.3f}' )
				r3.metric( 'Pearson p', f'{p_p:,.4g}' )
				r4.metric( 'Spearman ρ', f'{r_s:,.3f}' )
				
				st.caption( f'Spearman p = {p_s:.4g}' )
			else:
				st.info( 'Not enough paired observations for correlation.' )
	
	if col_x2:
		mask = x.notna( ) & y2.notna( )
		if mask.sum( ) >= 3:
			fig, ax = plt.subplots( figsize=(7, 5.25) )
			ax.scatter( x[ mask ], y2[ mask ], alpha=0.70, edgecolor='black' )
			
			if mask.sum( ) >= 2:
				try:
					m, b = np.polyfit( x[ mask ], y2[ mask ], 1 )
					xline = np.linspace( float( x[ mask ].min( ) ), float( x[ mask ].max( ) ), 100 )
					ax.plot( xline, m * xline + b, linewidth=2.0, linestyle='--' )
				except Exception:
					pass
			
			ax.set_title( f'Correlation — {col_y} vs {col_x2}',
				fontsize=12, fontweight='bold', pad=10 )
			ax.set_xlabel( col_x2 )
			ax.set_ylabel( col_y )
			ax.grid( True, alpha=0.20, linestyle='--' )
			ax.spines[ 'top' ].set_visible( False )
			ax.spines[ 'right' ].set_visible( False )
			fig.tight_layout( )
			st.pyplot( fig )
			plt.close( fig )
	
	# -------------------------------------------------------------------------------------
	# CATEGORICAL ASSOCIATION
	# -------------------------------------------------------------------------------------
	st.divider( )
	st.subheader( 'Categorical Association' )
	
	if not categorical_cols or len( categorical_cols ) < 2:
		st.info( 'At least two categorical variables are required for categorical association.' )
	else:
		cat_c1, cat_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with cat_c1:
			col_cat1 = st.selectbox( 'Select First Categorical Variable', categorical_cols )
		
		with cat_c2:
			col_cat2 = st.selectbox(
				'Select Second Categorical Variable',
				[ c for c in categorical_cols if c != col_cat1 ]
			)
		
		contingency = pd.crosstab( df_dataset[ col_cat1 ], df_dataset[ col_cat2 ] )
		
		if contingency.empty or contingency.shape[ 0 ] < 2 or contingency.shape[ 1 ] < 2:
			st.info( 'Not enough categorical variation for chi-square analysis.' )
		else:
			chi2, p_chi, dof, expected = stats.chi2_contingency( contingency )
			n = contingency.to_numpy( ).sum( )
			phi2 = chi2 / n if n > 0 else np.nan
			r, k = contingency.shape
			cramers_v = np.sqrt( phi2 / min( k - 1, r - 1 ) ) if min( k - 1, r - 1 ) > 0 else np.nan
			
			ca1, ca2 = st.columns( 2, border=True )
			with ca1:
				render_table( contingency.reset_index( ) )
			
			with ca2:
				fig, ax = plt.subplots( figsize=(7, 5.5) )
				sns.heatmap(
					contingency,
					annot=True,
					fmt='d',
					cmap='Blues',
					linewidths=0.5,
					ax=ax,
					cbar_kws={ 'shrink': 0.85, 'label': 'Count' }
				)
				ax.set_title(
					f'Contingency Heatmap — {col_cat1} vs {col_cat2}',
					fontsize=12,
					fontweight='bold',
					pad=10
				)
				ax.set_xlabel( col_cat2 )
				ax.set_ylabel( col_cat1 )
				fig.tight_layout( )
				st.pyplot( fig )
				plt.close( fig )
			
			cm1, cm2, cm3, cm4 = st.columns( 4 )
			cm1.metric( 'Chi-square', f'{chi2:,.4f}' )
			cm2.metric( 'p-value', f'{p_chi:,.4g}' )
			cm3.metric( 'DoF', f'{dof:,}' )
			cm4.metric( "Cramér's V", f'{cramers_v:,.4f}' if np.isfinite( cramers_v ) else 'n/a' )

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
