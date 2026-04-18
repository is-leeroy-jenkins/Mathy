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

import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from scipy import stats
from typing import List, Dict, Optional, Tuple, Any

# Mathy
import config as cfg

# sklearn / statsmodels
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
	confusion_matrix, roc_curve, auc, r2_score,
	accuracy_score, precision_score, recall_score, f1_score
)

try:
	from xgboost import XGBClassifier
	
	has_xgb = True
except Exception:
	has_xgb = False

import time
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from statsmodels.stats.power import TTestPower
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
import seaborn as sns
import sklearn.feature_selection as sf
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split as split
from scalers import ( StandardScaler, MinMaxScaler, RobustScaler, NormalScaler, MaxAbsScaler )

from imputers import ( MeanImputer, NearestImputer, IterativeImputer, SimpleImputer )

from encoders import ( OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder,
                       PolynomialFeatures )

from transformers import ( Binarizer, LabelBinarizer, MultiLabelBinarizer, TfidfTransformer,
	ColumnTransformer, TfidfVectorizer, CountVectorizer, HashVectorizer, DictVectorizer,
	FeatureHasher )

from clusters import ( KMeans, DBSCAN, Agglomerative, Spectral, OPTICS, MeanShift,
	AffinityPropagation, Birch )

from features import ( VarianceThreshold, CCA, PCA, SelectBest, SelectPercent, SBS, RFE )

import classifications as classification_model
import regressions as regression_model

from classifications import ( Perceptron, LogisticRegression, DecisionTree,
	SupportVector, RandomForest, NearestNeighbor, BaggingModel, AdaptiveBoost, GradientBoost )

from encoders import (OneHotEncoder, OrdinalEncoder, TargetEncoder)



from imputers import (MeanImputer, SimpleImputer, NearestImputer, IterativeImputer)
from forecasting import ( LaggingSeries, LagBoostingSeries, ARIMA, SARIMA, TimeSeriesSpliter )

# ============================================
# Session State
# ============================================

if 'mode' not in st.session_state or st.session_state[ 'mode' ] is None:
	st.session_state[ 'mode' ] = 'Data Transformation'

if 'df_dataset' not in st.session_state or st.session_state[ 'df_dataset' ] is None:
	st.session_state[ 'df_dataset'] = pd.DataFrame( )
	
# ------ Data Processing Members

if 'df_original' not in st.session_state or st.session_state[ 'df_original' ] is None:
	st.session_state[ 'df_original' ] = pd.DataFrame( )

if 'df_features' not in st.session_state:
	st.session_state[ 'df_features' ] = pd.DataFrame( )

if 'df_targets' not in st.session_state:
	st.session_state[ 'df_targets' ] = pd.DataFrame( )

if 'df_processed' not in st.session_state or st.session_state[ 'df_processed' ] is None:
	st.session_state[ 'df_processed' ] = pd.DataFrame( )

if 'df_dataset' not in st.session_state or st.session_state[ 'df_dataset' ] is None:
	st.session_state[ 'df_dataset' ] = pd.DataFrame( )

if 'df_working' not in st.session_state or st.session_state[ 'df_working' ] is None:
	st.session_state[ 'df_working' ] = pd.DataFrame( )

if 'df_scores' not in st.session_state:
	st.session_state[ 'df_scores' ] = pd.DataFrame( )

if 'df_model' not in st.session_state or st.session_state[ 'df_model' ] is None:
	st.session_state[ 'df_model' ] = pd.DataFrame( )

if 'df_evaluationuation' not in st.session_state or st.session_state[ 'df_evaluationuation' ] is None:
	st.session_state[ 'df_evaluationuation' ] = pd.DataFrame( )

if 'numeric_columns' not in st.session_state:
	st.session_state[ 'numeric_columns' ] = [ ]

if 'categorical_columns' not in st.session_state:
	st.session_state[ 'categorical_columns' ] = [ ]

if 'active_features' not in st.session_state:
	st.session_state[ 'active_features' ] = [ ]

if 'active_targets' not in st.session_state:
	st.session_state[ 'active_targets' ] = [ ]

if 'features' not in st.session_state:
	st.session_state[ 'features' ] = [ ]

if 'targets' not in st.session_state:
	st.session_state[ 'targets' ] = [ ]

if 'X_data' not in st.session_state:
	st.session_state[ 'X_data' ] = None
	
if 'X_train' not in st.session_state:
	st.session_state[ 'X_train' ] = None

if 'y_train' not in st.session_state:
	st.session_state[ 'y_train' ] = None

if 'X_test' not in st.session_state:
	st.session_state[ 'X_test' ] = None

if 'y_test' not in st.session_state:
	st.session_state[ 'y_test' ] = None

if 'y_prediction' not in st.session_state:
	st.session_state[ 'y_prediction' ] = None
	
if 'y_series' not in st.session_state:
	st.session_state[ 'y_series' ] = None

if 'target_count' not in st.session_state:
	st.session_state[ 'target_count' ] = 0.0
	
if 'elapsed_seconds' not in st.session_state:
	st.session_state[ 'elapsed_seconds' ] = 0.0

# ----------- Classfication Members

if 'df_classification' not in st.session_state or st.session_state[ 'df_classification' ] is None:
	st.session_state[ 'df_classification' ] = pd.DataFrame( )

if 'df_classification_scores' not in st.session_state or st.session_state[ 'df_classification_scores' ] is None:
	st.session_state[ 'df_classification_scores' ] = pd.DataFrame( )

# ----------- Regression Members

if 'df_regression' not in st.session_state or st.session_state[ 'df_regression' ] is None:
	st.session_state[ 'df_regression' ] = pd.DataFrame( )

if 'df_regression_scores' not in st.session_state or st.session_state[ 'df_regression_scores' ] is None:
	st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )

# ----------- Clustering Members

if 'df_cluster' not in st.session_state or st.session_state[ 'df_cluster' ] is None:
	st.session_state[ 'df_cluster' ] = pd.DataFrame( )

if 'df_cluster_results' not in st.session_state:
	st.session_state[ 'df_cluster_results' ] = pd.DataFrame( )

if 'df_cluster_counts' not in st.session_state:
	st.session_state[ 'df_cluster_counts' ] = pd.DataFrame( )

if 'df_cluster_metrics' not in st.session_state:
	st.session_state[ 'df_cluster_metrics' ] = pd.DataFrame( )

if 'df_cluster_centroids' not in st.session_state:
	st.session_state[ 'df_cluster_centroids' ] = pd.DataFrame( )

if 'df_cluster_details' not in st.session_state:
	st.session_state[ 'df_cluster_details' ] = pd.DataFrame( )

if 'cluster_plot_features' not in st.session_state:
	st.session_state[ 'cluster_plot_features' ] = [ ]

if 'cluster_signature' not in st.session_state:
	st.session_state[ 'cluster_signature' ] = None

# ============================================
# Session State
# ============================================

def init_state( ) -> None:
	defaults = {
			'df_dataset': None,
			'df_original': None,
			'df_processed': None,
			'numeric_columns': [ ],
			'categorical_columns': [ ],
			'features': [ ],
			'targets': [ ],
			'pipeline_log': [ ]
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
	numeric_hints = ('py', 'cy', 'by', 'amount', 'total', 'value', 'balance', 'outlay')
	categorical_hints = ('fy', 'code', 'id', 'name', 'type', 'symbol')
	
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
	s = pd.to_numeric( series, errors='coerce' )
	s = s.replace( [ np.inf, -np.inf ], np.nan ).dropna( )
	if s.empty:
		return '{:,.2f}'
	
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
	categorical_columns = df.select_dtypes( include=[ 'object' ] ).columns.tolist( )
	
	chart = st.selectbox( 'Chart Type', [ 'Histogram', 'Bar', 'Line', 'Scatter',
	                                      'Box', 'Pie', 'Correlation' ] )
	
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
		col = st.selectbox( 'Category Column', categorical_columns )
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
		div[data-testid="stMarkdownContainer"] h6,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h4 {
			color: rgb(2, 98, 201) !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

# ----------- Data Plumbing Utilities

def get_numeric_columns( df_frame: pd.DataFrame ) -> list[ str ]:
	"""
		Purpose:
		--------
		Return numeric columns from the specified dataframe.

		Parameters:
		-----------
		df_frame ( pd.DataFrame ): Input dataframe.

		Returns:
		--------
		list[ str ]:
			Numeric column names.
	"""
	return [ c for c in df_frame.columns
	         if pd.api.types.is_numeric_dtype( df_frame[ c ] ) ]

def get_categorical_columns( df_frame: pd.DataFrame ) -> list[ str ]:
	"""
		Purpose:
		--------
		Return non-numeric columns from the specified dataframe.

		Parameters:
		-----------
		df_frame ( pd.DataFrame ): Input dataframe.

		Returns:
		--------
		list[ str ]:
			Non-numeric column names.
	"""
	return [ c for c in df_frame.columns
	         if not pd.api.types.is_numeric_dtype( df_frame[ c ] ) ]

def get_working_frame( ) -> pd.DataFrame:
	"""
		Purpose:
		--------
		Return the current working dataframe used by Data Plumbing.

		Parameters:
		-----------
		None

		Returns:
		--------
		pd.DataFrame:
			Current working dataframe.
	"""
	df_working = st.session_state.get( 'df_processed' )
	if df_working is None or df_working.empty:
		return st.session_state[ 'df_original' ].copy( )
	return df_working.copy( )

def get_feature_columns( df_frame: pd.DataFrame ) -> list[ str ]:
	"""
		Purpose:
		--------
		Return active feature columns that still exist in the specified dataframe.

		Parameters:
		-----------
		df_frame ( pd.DataFrame ): Input dataframe.

		Returns:
		--------
		list[ str ]:
			Active feature columns.
	"""
	return [ c for c in st.session_state.get( 'features', [ ] )
	         if c in df_frame.columns ]

def get_target_columns( df_frame: pd.DataFrame ) -> list[ str ]:
	"""
		Purpose:
		--------
		Return active target columns that still exist in the specified dataframe.

		Parameters:
		-----------
		df_frame ( pd.DataFrame ): Input dataframe.

		Returns:
		--------
		list[ str ]:
			Active target columns.
	"""
	return [ c for c in st.session_state.get( 'targets', [ ] )
	         if c in df_frame.columns ]

def commit_frame( df_frame: pd.DataFrame ) -> None:
	"""
		Purpose:
		--------
		Commit the updated working dataframe to session state.

		Parameters:
		-----------
		df_frame ( pd.DataFrame ): Updated dataframe.

		Returns:
		--------
		None
	"""
	feature_columns = get_feature_columns( df_frame )
	target_columns = get_target_columns( df_frame )
	
	if feature_columns:
		st.session_state[ 'df_features' ] = df_frame[ feature_columns ].copy( )
	else:
		st.session_state[ 'df_features' ] = pd.DataFrame( index=df_frame.index )
	
	if target_columns:
		st.session_state[ 'df_targets' ] = df_frame[ target_columns ].copy( )
	else:
		st.session_state[ 'df_targets' ] = pd.DataFrame( index=df_frame.index )

def working_to_original( ) -> None:
	"""
		Purpose:
		--------
		Reset Data Plumbing session state back to df_original.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	"""
	df_reset = st.session_state[ 'df_original' ].copy( )
	st.session_state[ 'df_working' ] = df_reset.copy( )
	commit_frame( df_reset )

def processed_to_working( ) -> None:
	"""
		Purpose:
		--------
		Reset Data Plumbing session state back to df_original.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	"""
	df_reset = st.session_state[ 'df_working' ].copy( )
	st.session_state[ 'df_processed' ] = df_reset.copy( )
	commit_frame( df_reset )
	
def normalize_result_frame( result: object, index: pd.Index,
		prefix: str, columns: list[ str ] ) -> pd.DataFrame:
	"""
		Purpose:
		--------
		Normalize transformation output into a dataframe.

		Parameters:
		-----------
		result ( object ): Transformation result.
		index ( pd.Index ): Output index.
		prefix ( str ): Prefix used for generated column names.
		columns ( list[ str ] | None ): Optional output column names.

		Returns:
		--------
		pd.DataFrame:
			Normalized dataframe result.
	"""
	if isinstance( result, pd.DataFrame ):
		df_result = result.copy( )
		df_result.index = index
		return df_result
	
	if isinstance( result, tuple ):
		parts = [ ]
		for i, item in enumerate( result ):
			df_part = normalize_result_frame(
				item,
				index=index,
				prefix=f'{prefix}_{i + 1}',
				columns=None
			)
			parts.append( df_part )
		return pd.concat( parts, axis=1 )
	
	if hasattr( result, 'toarray' ):
		result = result.toarray( )
	
	arr = np.asarray( result )
	
	if arr.ndim == 1:
		col_name = columns[ 0 ] if columns and len( columns ) == 1 else prefix
		return pd.DataFrame( arr, index=index, columns=[ col_name ] )
	
	if columns and len( columns ) == arr.shape[ 1 ]:
		return pd.DataFrame( arr, index=index, columns=columns )
	
	generated = [ f'{prefix}_{i + 1}' for i in range( arr.shape[ 1 ] ) ]
	return pd.DataFrame( arr, index=index, columns=generated )

def replace_columns( df_frame: pd.DataFrame, column_names: list[ str ],
		result: object, prefix: str ) -> pd.DataFrame:
	"""
		Purpose:
		--------
		Replace selected columns in a dataframe with transformed output.

		Parameters:
		-----------
		df_frame ( pd.DataFrame ): Input dataframe.
		column_names ( list[ str ] ): Columns being replaced.
		result ( object ): Transformation output.
		prefix ( str ): Prefix used for generated columns.

		Returns:
		--------
		pd.DataFrame:
			Updated dataframe.
	"""
	df_result = normalize_result_frame( result=result, index=df_frame.index,
		prefix=prefix, columns=column_names )
	
	df_updated = df_frame.drop( columns=column_names, errors='ignore' )
	df_updated = pd.concat( [ df_updated, df_result ], axis=1 )
	return df_updated

def apply_text_vectorizer( df_frame: pd.DataFrame, column_names: list[ str ],
		vectorizer: object, prefix: str ) -> pd.DataFrame:
	"""
		Purpose:
		--------
		Apply a text vectorizer to selected columns joined row-wise.

		Parameters:
		-----------
		df_frame ( pd.DataFrame ): Input dataframe.
		column_names ( list[ str ] ): Text columns.
		vectorizer ( object ): Vectorizer wrapper.
		prefix ( str ): Prefix used for generated columns.

		Returns:
		--------
		pd.DataFrame:
			Updated dataframe.
	"""
	text = df_frame[ column_names ].fillna( '' ).astype( str ).agg( ' '.join, axis=1 ).tolist( )
	
	result = vectorizer.train_transform( text )
	df_result = normalize_result_frame( result=result, index=df_frame.index,
		prefix=prefix, columns=None )
	
	df_updated = df_frame.drop( columns=column_names, errors='ignore' )
	df_updated = pd.concat( [ df_updated, df_result ], axis=1 )
	return df_updated

def apply_dict_transform( df_frame: pd.DataFrame, column_names: list[ str ],
		transformer: object, prefix: str ) -> pd.DataFrame:
	"""
		Purpose:
		--------
		Apply a dictionary-based transformer to selected columns.

		Parameters:
		-----------
		df_frame ( pd.DataFrame ): Input dataframe.
		column_names ( list[ str ] ): Columns converted to row dictionaries.
		transformer ( object ): Dict-like transformer wrapper.
		prefix ( str ): Prefix used for generated columns.

		Returns:
		--------
		pd.DataFrame:
			Updated dataframe.
	"""
	records = df_frame[ column_names ].fillna( '' ).to_dict( orient='records' )
	result = transformer.train_transform( records )
	df_result = normalize_result_frame(
		result=result,
		index=df_frame.index,
		prefix=prefix,
		columns=None
	)
	
	df_updated = df_frame.drop( columns=column_names, errors='ignore' )
	df_updated = pd.concat( [ df_updated, df_result ], axis=1 )
	return df_updated

def parse_multilabel_series( series: pd.Series, delimiter: str ) -> np.ndarray:
	"""
		Purpose:
		--------
		Parse a delimited series into an array of label lists.

		Parameters:
		-----------
		series ( pd.Series ): Input label series.
		delimiter ( str ): Delimiter separating labels.

		Returns:
		--------
		np.ndarray:
			Array of parsed label collections.
	"""
	values = series.fillna( '' ).astype( str ).apply(
		lambda s: [ item.strip( ) for item in s.split( delimiter ) if item.strip( ) ]
	)
	return values.to_numpy( )

def score_function_from_name( name: str ) -> object:
	"""
		Purpose:
		--------
		Map a display name to a sklearn scoring function.

		Parameters:
		-----------
		name ( str ): Display name.

		Returns:
		--------
		object:
			Scoring function.
	"""
	mapper = {
			'chi2': sf.chi2,
			'f_classif': sf.f_classif,
			'f_regression': sf.f_regression,
			'mutual_info_classif': sf.mutual_info_classif,
			'mutual_info_regression': sf.mutual_info_regression
	}
	return mapper[ name ]

# ----------  Database Utilities ----------

def initialize_database( ) -> None:
	"""
		Purpose:
		--------
		Ensure required SQLite tables exist and that the Prompts table contains the
		columns required by the prompt utilities and Prompt Engineering mode.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	"""
	Path( 'stores/sqlite' ).mkdir( parents=True, exist_ok=True )
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS chat_history
            (
                id
                INTEGER
                PRIMARY
                KEY
                AUTOINCREMENT,
                role
                TEXT,
                content
                TEXT
            )
			"""
		)
		
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS embeddings
            (
                id
                INTEGER
                PRIMARY
                KEY
                AUTOINCREMENT,
                chunk
                TEXT,
                vector
                BLOB
            )
			"""
		)
		
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS Prompts
            (
                PromptsId
                INTEGER
                NOT
                NULL
                PRIMARY
                KEY
                AUTOINCREMENT,
                Caption
                TEXT,
                Name
                TEXT
            (
                80
            ),
                Text TEXT,
                Version TEXT
            (
                80
            ),
                ID TEXT
            (
                80
            )
                )
			"""
		)
		
		prompt_columns = [ row[ 1 ] for row in
		                   conn.execute( 'PRAGMA table_info("Prompts");' ).fetchall( ) ]
		
		if 'Caption' not in prompt_columns:
			conn.execute( 'ALTER TABLE "Prompts" ADD COLUMN "Caption" TEXT;' )
		
		conn.commit( )

def create_connection( ) -> sqlite3.Connection:
	return sqlite3.connect( cfg.DB_PATH )

def list_tables( ) -> List[ str ]:
	with create_connection( ) as conn:
		_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
		rows = conn.execute( _query ).fetchall( )
		return [ r[ 0 ] for r in rows ]

def create_schema( table: str ) -> List[ Tuple ]:
	with create_connection( ) as conn:
		return conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )

def read_table( table: str, limit: int = None, offset: int = 0 ) -> pd.DataFrame:
	"""
	
		Purpose:
		--------
		Read a SQLite table into a pandas DataFrame using a normalized scalar-only path.
	
		Parameters:
		-----------
		table : str
			Table name.
		limit : int = None
			Optional row limit.
		offset : int = 0
			Optional row offset.
	
		Returns:
		--------
		pd.DataFrame
			DataFrame of plain Python scalar values.
	
	"""
	if not table:
		return pd.DataFrame( )
	
	query = f'SELECT * FROM "{table}"'
	if limit:
		query += f' LIMIT {int( limit )} OFFSET {int( offset )}'
	
	with create_connection( ) as conn:
		cur = conn.cursor( )
		cur.execute( query )
		
		raw_columns = [ d[ 0 ] for d in (cur.description or [ ]) ]
		rows = cur.fetchall( )
	
	seen: Dict[ str, int ] = { }
	columns: List[ str ] = [ ]
	
	for col in raw_columns:
		name = str( col )
		if name not in seen:
			seen[ name ] = 0
			columns.append( name )
		else:
			seen[ name ] += 1
			columns.append( f'{name}_{seen[ name ]}' )
	
	def _scalarize( value: Any ) -> Any:
		if value is None or isinstance( value, (str, int, float, bool) ):
			return value
		
		if isinstance( value, bytes ):
			try:
				return value.decode( 'utf-8' )
			except Exception:
				return value.hex( )
		
		if isinstance( value, (list, tuple, set, dict) ):
			try:
				return str( normalize( value ) )
			except Exception:
				return str( value )
		
		if hasattr( value, 'model_dump' ):
			try:
				return str( value.model_dump( ) )
			except Exception:
				return str( value )
		
		return str( value )
	
	normalized_rows: List[ Dict[ str, Any ] ] = [ ]
	for row in rows:
		record: Dict[ str, Any ] = { }
		for idx, col in enumerate( columns ):
			record[ col ] = _scalarize( row[ idx ] )
		normalized_rows.append( record )
	
	return pd.DataFrame( normalized_rows, columns=columns )

def render_table( df: pd.DataFrame ) -> None:
	"""
	
		Purpose:
		--------
		Render a DataFrame safely in Streamlit. Use the normal interactive dataframe
		first, and fall back to HTML rendering if Streamlit/PyArrow serialization fails.
	
		Parameters:
		-----------
		df : pd.DataFrame
			The DataFrame to render.
	
		Returns:
		--------
		None
	
	"""
	if df is None:
		st.info( 'No data available.' )
		return
	
	try:
		st.data_editor( df, use_container_width=True )
		return
	except Exception:
		pass
	
	fallback_df = df.copy( )
	fallback_df = fallback_df.where( pd.notnull( fallback_df ), '' )
	
	for col in fallback_df.columns:
		fallback_df[ col ] = fallback_df[ col ].map(
			lambda x: x if isinstance( x, (str, int, float, bool) ) or x == '' else str( x ) )
	
	st.markdown( fallback_df.to_html( index=False, escape=True ), unsafe_allow_html=True )

def make_display_safe( df: pd.DataFrame ) -> pd.DataFrame:
	display_df = df.copy( )
	
	for col in display_df.columns:
		display_df[ col ] = display_df[ col ].map(
			lambda x: '' if x is None else str( x )
		)
	
	return display_df

def drop_table( table: str ) -> None:
	"""
		Purpose:
		--------
		Safely drop a table if it exists.
	
		Parameters:
		-----------
		table : str
			Table name.
	"""
	if not table:
		return
	
	with create_connection( ) as conn:
		conn.execute( f'DROP TABLE IF EXISTS "{table}";' )
		conn.commit( )

def create_index( table: str, column: str ) -> None:
	"""
		Purpose:
		--------
		Create a safe SQLite index on a specified table column.
	
		Handles:
			- Spaces in column names
			- Special characters
			- Reserved words
			- Duplicate index names
			- Validation against actual table schema
	
		Parameters:
		-----------
		table : str
			Table name.
		column : str
			Column name to index.
	"""
	if not table or not column:
		return
	
	# ------------------------------------------------------------------
	# Validate table exists
	# ------------------------------------------------------------------
	tables = list_tables( )
	if table not in tables:
		raise ValueError( 'Invalid table name.' )
	
	# ------------------------------------------------------------------
	# Validate column exists
	# ------------------------------------------------------------------
	schema = create_schema( table )
	valid_columns = [ col[ 1 ] for col in schema ]
	
	if column not in valid_columns:
		raise ValueError( 'Invalid column name.' )
	
	# ------------------------------------------------------------------
	# Sanitize index name (identifier only)
	# ------------------------------------------------------------------
	safe_index_name = re.sub( r"[^0-9a-zA-Z_]+", "_", f"idx_{table}_{column}" )
	
	# ------------------------------------------------------------------
	# Create index safely (quote identifiers)
	# ------------------------------------------------------------------
	sql = f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" ON "{table}"("{column}");'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def apply_filters( df: pd.DataFrame ) -> pd.DataFrame:
	st.subheader( 'Advanced Filters' )
	conditions = [ ]
	col1, col2, col3 = st.columns( 3 )
	column = col1.selectbox( 'Column', df.columns )
	operator = col2.selectbox( 'Operator', [ '=', '!=', '>', '<', '>=', '<=', 'contains' ] )
	value = col3.text_input( 'Value' )
	if value:
		if operator == '=':
			df = df[ df[ column ] == value ]
		elif operator == '!=':
			df = df[ df[ column ] != value ]
		elif operator == '>':
			df = df[ df[ column ].astype( float ) > float( value ) ]
		elif operator == '<':
			df = df[ df[ column ].astype( float ) < float( value ) ]
		elif operator == '>=':
			df = df[ df[ column ].astype( float ) >= float( value ) ]
		elif operator == '<=':
			df = df[ df[ column ].astype( float ) <= float( value ) ]
		elif operator == 'contains':
			df = df[ df[ column ].astype( str ).str.contains( value ) ]
	
	return df

def create_aggregation( df: pd.DataFrame ):
	st.subheader( 'Aggregation Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	
	if not numeric_cols:
		st.info( 'No numeric columns available.' )
		return
	
	col = st.selectbox( 'Column', numeric_cols )
	agg = st.selectbox( 'Aggregation', [ 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'MEDIAN' ] )
	
	if agg == 'COUNT':
		result = df[ col ].count( )
	elif agg == 'SUM':
		result = df[ col ].sum( )
	elif agg == 'AVG':
		result = df[ col ].mean( )
	elif agg == 'MIN':
		result = df[ col ].min( )
	elif agg == 'MAX':
		result = df[ col ].max( )
	elif agg == 'MEDIAN':
		result = df[ col ].median( )
	
	st.metric( 'Result', result )

def create_visualization( df: pd.DataFrame ) -> None:
	"""
	
		Purpose:
		--------
		Render data visualizations without passing pandas objects directly into
		Plotly/Narwhals.
		
		Parameters:
		-----------
		df : pd.DataFrame
			The input DataFrame.
		
		Returns:
		--------
		None
		
	"""
	st.subheader( 'Visualization Engine' )
	
	if df is None or df.empty:
		st.info( 'No data available.' )
		return
	
	df_plot = df.copy( )
	
	for col in df_plot.columns:
		if df_plot[ col ].dtype == object:
			df_plot[ col ] = df_plot[ col ].map(
				lambda x: '' if x is None else str( x )
			)
	
	numeric_cols: List[ str ] = [ ]
	for col in df_plot.columns:
		series_num = pd.to_numeric( df_plot[ col ], errors='coerce' )
		if series_num.notna( ).any( ):
			numeric_cols.append( col )
	
	categorical_columns: List[ str ] = [ col for col in df_plot.columns if col not in numeric_cols ]
	
	chart = st.selectbox(
		'Chart Type',
		[ 'Histogram', 'Bar', 'Line', 'Scatter', 'Box', 'Pie', 'Correlation' ] )
	
	if chart == 'Histogram':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = go.Figure( data=[ go.Histogram( x=values ) ] )
		fig.update_layout( xaxis_title=col, yaxis_title='Count' )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Bar':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = go.Figure( data=[ go.Bar( x=x_values, y=y_values ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Line':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='lines' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Scatter':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		x = st.selectbox( 'X', numeric_cols, key='viz_scatter_x' )
		y = st.selectbox( 'Y', numeric_cols, key='viz_scatter_y' )
		
		x_series = pd.to_numeric( df_plot[ x ], errors='coerce' )
		y_series = pd.to_numeric( df_plot[ y ], errors='coerce' )
		mask = x_series.notna( ) & y_series.notna( )
		
		x_values = x_series[ mask ].tolist( )
		y_values = y_series[ mask ].tolist( )
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='markers' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols, key='viz_box_col' )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = go.Figure( data=[ go.Box( y=values, name=col ) ] )
		fig.update_layout( yaxis_title=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		if not categorical_columns:
			st.info( 'No categorical columns available.' )
			return
		
		col = st.selectbox( 'Category Column', categorical_columns )
		counts = df_plot[ col ].astype( str ).value_counts( )
		
		fig = go.Figure(
			data=[ go.Pie( labels=counts.index.tolist( ), values=counts.values.tolist( ) ) ] )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		corr_df = pd.DataFrame( )
		for col in numeric_cols:
			corr_df[ col ] = pd.to_numeric( df_plot[ col ], errors='coerce' )
		
		corr = corr_df.corr( )
		
		fig = go.Figure(
			data=[ go.Heatmap(
				z=corr.values.tolist( ),
				x=corr.columns.tolist( ),
				y=corr.index.tolist( ) ) ] )
		st.plotly_chart( fig, use_container_width=True )

def convert_dataframe( table_name: str, df: pd.DataFrame ):
	columns = [ ]
	for col in df.columns:
		sql_type = get_sqlite_type( df[ col ].dtype )
		safe_col = col.replace( ' ', '_' )
		columns.append( f'{safe_col} {sql_type}' )
	
	create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join( columns )});'
	
	with create_connection( ) as conn:
		conn.execute( create_stmt )
		conn.commit( )

def insert_data( table_name: str, df: pd.DataFrame ):
	df = df.copy( )
	df.columns = [ c.replace( ' ', '_' ) for c in df.columns ]
	
	placeholders = ', '.join( [ '?' ] * len( df.columns ) )
	stmt = f'INSERT INTO {table_name} VALUES ({placeholders});'
	
	with create_connection( ) as conn:
		conn.executemany( stmt, df.values.tolist( ) )
		conn.commit( )

def get_sqlite_type( dtype ) -> str:
	"""
		Purpose:
		--------
		Map a pandas dtype to an appropriate SQLite column type.
	
		Parameters:
		-----------
		dtype : pandas dtype
			The dtype of a pandas Series.
	
		Returns:
		--------
		str
			SQLite column type.
	"""
	dtype_str = str( dtype ).lower( )
	
	# ------------------------------------------------------------------
	# Integer Types (including nullable Int64)
	# ------------------------------------------------------------------
	if 'int' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Float Types
	# ------------------------------------------------------------------
	if 'float' in dtype_str:
		return 'REAL'
	
	# ------------------------------------------------------------------
	# Boolean
	# ------------------------------------------------------------------
	if 'bool' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Datetime
	# ------------------------------------------------------------------
	if 'datetime' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Categorical
	# ------------------------------------------------------------------
	if 'category' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Default fallback
	# ------------------------------------------------------------------
	return 'TEXT'

def create_custom_table( table_name: str, columns: list ) -> None:
	"""
		Purpose:
		--------
		Create a custom SQLite table from column definitions.
	
		Parameters:
		-----------
		table_name : str
			Name of table.
	
		columns : list of dict
			[
				{
					"name": str,
					"type": str,
					"not_null": bool,
					"primary_key": bool,
					"auto_increment": bool
				}
			]
	"""
	if not table_name:
		raise ValueError( 'Table name required.' )
	
	# Validate identifier
	if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", table_name ):
		raise ValueError( 'Invalid table name.' )
	
	col_defs = [ ]
	
	for col in columns:
		col_name = col[ 'name' ]
		col_type = col[ 'type' ].upper( )
		
		if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", col_name ):
			raise ValueError( f"Invalid column name: {col_name}" )
		
		definition = f'"{col_name}" {col_type}'
		
		if col[ 'primary_key' ]:
			definition += ' PRIMARY KEY'
			if col[ 'auto_increment' ] and col_type == 'INTEGER':
				definition += ' AUTOINCREMENT'
		
		if col[ "not_null" ]:
			definition += " NOT NULL"
		
		col_defs.append( definition )
	
	sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join( col_defs )});'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def is_safe_query( query: str ) -> bool:
	"""
	
		Purpose:
		--------
		Determine whether a SQL query is read-only and safe to execute.
	
		Allows:
			SELECT
			WITH (CTE returning SELECT)
			EXPLAIN SELECT
			PRAGMA (read-only)
	
		Blocks:
			INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH,
			DETACH, VACUUM, REPLACE, TRIGGER, and multiple statements.
			
	"""
	if not query or not isinstance( query, str ):
		return False
	
	q = query.strip( ).lower( )
	
	# ------------------------------------------------------------------
	# Block multiple statements
	# ------------------------------------------------------------------
	if ';' in q[ :-1 ]:
		return False
	
	# ------------------------------------------------------------------
	# Remove SQL comments
	# ------------------------------------------------------------------
	q = re.sub( r"--.*?$", "", q, flags=re.MULTILINE )
	q = re.sub( r"/\*.*?\*/", "", q, flags=re.DOTALL )
	q = q.strip( )
	
	# ------------------------------------------------------------------
	# Allowed starting keywords
	# ------------------------------------------------------------------
	allowed_starts = ('select', 'with', 'explain', 'pragma')
	if not q.startswith( allowed_starts ):
		return False
	
	# ------------------------------------------------------------------
	# Block dangerous keywords anywhere
	# ------------------------------------------------------------------
	blocked_keywords = ('insert ', 'update ', 'delete ', 'drop ', 'alter ',
	                    'create ', 'attach ', 'detach ', 'vacuum ', 'replace ', 'trigger ')
	
	for keyword in blocked_keywords:
		if keyword in q:
			return False
	
	return True

def create_identifier( name: str ) -> str:
	"""
	
		Purpose:
		--------
		Sanitize a string into a safe SQLite identifier.
	
		- Replaces invalid characters with underscores
		- Ensures it starts with a letter or underscore
		- Prevents empty names
		
	"""
	if not name or not isinstance( name, str ):
		raise ValueError( 'Invalid Identifier.' )
	
	safe = re.sub( r'[^0-9a-zA-Z_]', '_', name.strip( ) )
	if not re.match( r'^[A-Za-z_]', safe ):
		safe = f'_{safe}'
	
	if not safe:
		raise ValueError( 'Invalid identifier after sanitization.' )
	
	return safe

def get_indexes( table: str ):
	with create_connection( ) as conn:
		rows = conn.execute( f'PRAGMA index_list("{table}");' ).fetchall( )
		return rows

def add_column( table: str, column: str, col_type: str ):
	column = create_identifier( column )
	col_type = col_type.upper( )
	
	with create_connection( ) as conn:
		conn.execute(
			f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};' )
		conn.commit( )

def rename_column( table_name: str, old_name: str, new_name: str ) -> None:
	"""
	
		Purpose:
		--------
		Rename a column within an existing SQLite table. Attempts native ALTER TABLE rename
		first; if it fails, falls back to a schema-safe rebuild preserving column order, data,
		and indexes.

		Parameters:
		-----------
		table_name : str
			Table containing the column.

		old_name : str
			Existing column name.

		new_name : str
			New column name.

		Returns:
		--------
		None
		
	"""
	if not table_name or not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute(
				f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}";'
			)
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table_name,)
		).fetchall( )
		
		schema = conn.execute( f'PRAGMA table_info("{table_name}");' ).fetchall( )
		cols = [ r[ 1 ] for r in schema ]
		if old_name not in cols:
			raise ValueError( "Column not found." )
		
		mapped_cols = [ (new_name if c == old_name else c) for c in cols ]
		
		temp_table = f"{table_name}__rebuild_temp"
		
		col_defs: List[ str ] = [ ]
		pk_cols = [ r for r in schema if int( r[ 5 ] or 0 ) > 0 ]
		single_pk = len( pk_cols ) == 1
		
		for row in schema:
			col_name = row[ 1 ]
			col_type = row[ 2 ] or ''
			not_null = int( row[ 3 ] or 0 )
			default_value = row[ 4 ]
			pk = int( row[ 5 ] or 0 )
			
			out_name = new_name if col_name == old_name else col_name
			col_def = f'"{out_name}" {col_type}'.strip( )
			
			if not_null:
				col_def += ' NOT NULL'
			
			if default_value is not None:
				col_def += f' DEFAULT {default_value}'
			
			if single_pk and pk == 1:
				col_def += ' PRIMARY KEY'
			
			col_defs.append( col_def )
		
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( col_defs )});'
		
		old_select = ", ".join( [ f'"{c}"' for c in cols ] )
		new_insert = ", ".join( [ f'"{c}"' for c in mapped_cols ] )
		
		conn.execute( "BEGIN" )
		conn.execute( new_create_sql )
		conn.execute(
			f'INSERT INTO "{temp_table}" ({new_insert}) SELECT {old_select} FROM "{table_name}";'
		)
		
		conn.execute( f'DROP TABLE "{table_name}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'"{old_name}"', f'"{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def create_profile_table( table: str ):
	df = read_table( table )
	profile_rows = [ ]
	total_rows = len( df )
	for col in df.columns:
		series = df[ col ]
		null_count = series.isna( ).sum( )
		distinct_count = series.nunique( dropna=True )
		row = \
			{
					'column': col, 'dtype': str( series.dtype ),
					'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
					'distinct_%': round( ( distinct_count / total_rows) * 100, 2 ) if total_rows else 0,
			}
		
		if pd.api.types.is_numeric_dtype( series ):
			row[ 'min' ] = series.min( )
			row[ 'max' ] = series.max( )
			row[ 'mean' ] = series.mean( )
		else:
			row[ 'min' ] = None
			row[ 'max' ] = None
			row[ 'mean' ] = None
		
		profile_rows.append( row )
	
	return pd.DataFrame( profile_rows )

def drop_column( table: str, column: str ):
	if not table or not column:
		raise ValueError( 'Table and column required.' )
	
	with create_connection( ) as conn:
		# ------------------------------------------------------------
		# Fetch original CREATE TABLE statement
		# ------------------------------------------------------------
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( 'Table definition not found.' )
		
		create_sql = row[ 0 ]
		
		# ------------------------------------------------------------
		# Extract column definitions
		# ------------------------------------------------------------
		open_paren = create_sql.find( "(" )
		close_paren = create_sql.rfind( ")" )
		
		if open_paren == -1 or close_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		inner = create_sql[ open_paren + 1: close_paren ]
		
		column_defs = [ c.strip( ) for c in inner.split( "," ) ]
		
		# Remove target column
		new_defs = [ ]
		for col_def in column_defs:
			col_name = col_def.split( )[ 0 ].strip( '"' )
			if col_name != column:
				new_defs.append( col_def )
		
		if len( new_defs ) == len( column_defs ):
			raise ValueError( "Column not found." )
		
		# ------------------------------------------------------------
		# Build new CREATE TABLE statement
		# ------------------------------------------------------------
		temp_table = f"{table}_rebuild_temp"
		
		new_create_sql = (
				f'CREATE TABLE "{temp_table}" ('
				+ ", ".join( new_defs )
				+ ");"
		)
		
		# ------------------------------------------------------------
		# Begin transaction
		# ------------------------------------------------------------
		conn.execute( "BEGIN" )
		
		conn.execute( new_create_sql )
		
		remaining_cols = [
				c.split( )[ 0 ].strip( '"' )
				for c in new_defs
		]
		
		col_list = ", ".join( [ f'"{c}"' for c in remaining_cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_table}" ({col_list}) '
			f'SELECT {col_list} FROM "{table}";'
		)
		
		# Preserve indexes
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table,)
		).fetchall( )
		
		conn.execute( f'DROP TABLE "{table}";' )
		conn.execute(
			f'ALTER TABLE "{temp_table}" RENAME TO "{table}";'
		)
		
		# Recreate indexes
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if column not in idx_sql:
				conn.execute( idx_sql )
		
		conn.commit( )

def rename_table( old_name: str, new_name: str ) -> None:
	"""
	
		Purpose:
		--------
		Rename an existing SQLite table. Attempts native ALTER TABLE rename first; if it fails,
		falls back to a schema-safe rebuild using the original CREATE TABLE statement and
		preserves indexes.

		Parameters:
		-----------
		old_name : str
			Existing table name.

		new_name : str
			New table name.

		Returns:
		--------
		None
		
	"""
	if not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute( f'ALTER TABLE "{old_name}" RENAME TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(old_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(old_name,)
		).fetchall( )
		
		open_paren = create_sql.find( "(" )
		if open_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		temp_name = f"{new_name}__rebuild_temp"
		
		conn.execute( "BEGIN" )
		conn.execute( f'CREATE TABLE "{temp_name}" {create_sql[ open_paren: ]}' )
		
		cols = [ r[ 1 ] for r in conn.execute( f'PRAGMA table_info("{old_name}");' ).fetchall( ) ]
		col_list = ", ".join( [ f'"{c}"' for c in cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_name}" ({col_list}) SELECT {col_list} FROM "{old_name}";'
		)
		
		conn.execute( f'DROP TABLE "{old_name}";' )
		conn.execute( f'ALTER TABLE "{temp_name}" RENAME TO "{new_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'ON "{old_name}"', f'ON "{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

# ============================================
# Page Configuration
# ============================================
st.set_page_config( page_title='Mathy', layout='wide',
	page_icon=cfg.FAVICON, initial_sidebar_state='expanded' )

st.logo( image=cfg.LOGO, size='large' )
pd.options.display.float_format = '{:,.2f}'.format

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
	st.sidebar.divider( )
	st.subheader( 'Source' )
	use_fallback = st.sidebar.checkbox( 'Use Default Data', value=True )
	uploaded = st.sidebar.file_uploader( label='Upload Spreadsheet', type=[ 'xlsx',  'xls',  'csv' ] )
	if uploaded or use_fallback:
		if uploaded:
			df_dataset = pd.read_excel( uploaded ) if uploaded.name.endswith( 'xls' ) else pd.read_csv( uploaded )
			df_original = df_dataset.copy( )
			log_step( f'Loaded uploaded file: {uploaded.name}' )
		else:
			df_dataset = pd.read_excel( cfg.DEFAULT_DATA )
			df_original = df_dataset.copy( )
			log_step( 'Loaded Default Dataset' )
		
		st.session_state.raw_df = df_dataset.copy( )
		st.session_state[ 'df_original' ] = df_original.copy( )
		st.session_state[ 'df_dataset' ] = df_dataset.copy( )
		
	st.sidebar.divider( )
	st.subheader( 'Mode' )
	mode = st.sidebar.radio( 'Select', cfg.MODE.keys( ), index=0 )
	style_subheaders( )

# ============================================
# DATA PROFILING MODE
# ============================================
if mode == 'Data Profile':
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Data Profile' ] )
		st.divider( )
		
		if st.session_state.df_dataset is None:
			st.info( 'No data loaded.' )
			st.stop( )
		
		df_dataset = st.session_state[ 'df_dataset' ]
		df_original = df_dataset.copy( )
		
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
		numeric_columns = [ c for c, t in schema.items( ) if t == 'numeric' ]
		categorical_columns = [ c for c, t in schema.items( ) if t == 'categorical' ]
		st.session_state[ 'numeric_columns' ] = numeric_columns
		st.session_state[ 'categorical_columns' ] = categorical_columns
		
		# -------------------------------------------------------------------------------------
		# DATASET DISPLAY
		# -------------------------------------------------------------------------------------
		st.markdown( '##### Data' )
		render_table( df_dataset )
		
		# -------------------------------------------------------------------------------------
		# SCHEMA METRICS
		# -------------------------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Types' )
		type_counts = pd.Series( schema ).value_counts( )
		m1, m2, m3, m4, m5 = st.columns( 5, border=True )
		m1.metric( 'Rows', len( df_dataset ) )
		m2.metric( 'Numeric', type_counts.get( 'numeric', 0 ) )
		m3.metric( 'Ordinal / ID', type_counts.get( 'ordinal', 0 ) + type_counts.get( 'identifier', 0 ) )
		m4.metric( 'Categorical', type_counts.get( 'categorical', 0 ) )
		m5.metric( 'Datetime', type_counts.get( 'datetime', 0 ) )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Records' )
		
		with st.expander( label='Edit', icon='✏️', expanded=True ):
			top_c1, top_c2 = st.columns( [ 0.20, 0.80 ] )
			with top_c1:
				row_idx = st.number_input( 'Select Index', min_value=0,
					max_value=len( df_dataset ) - 1, step=1, key='row_editor_index' )
			
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
				st.data_editor( pd.DataFrame( { 'Before': before, 'After': after } ),
					use_container_width=True )
				
				st.rerun( )
				
		# =====================================================================================
		# DIAGNOSTIC VISUALIZATIONS (TAB-1 APPROPRIATE)
		# =====================================================================================
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Diagnostics' )
		
		v1, v2 = st.columns( 2, border=True )
		with v1:
			fig, ax = plt.subplots( figsize=(6, 4.5) )
			type_counts.sort_values( ascending=False ).plot(
				kind='bar', ax=ax, width=0.75, edgecolor='#0f172a', linewidth=0.9 )
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
				missing_pct.sort_values( ascending=True ).plot( kind='barh', ax=ax,
					width=0.75, edgecolor='#0f172a', linewidth=0.9 )
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
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Cardinality', help=cfg.DATA_CARDINALITY )
		
		v3, v4 = st.columns( 2, border=True )
		with v3:
			cardinality = df_dataset.nunique( dropna=True ).sort_values( ascending=False ).head( 10 )
			fig, ax = plt.subplots( figsize=(6, 4.5) )
			cardinality.sort_values( ascending=True ).plot( kind='barh', ax=ax, width=0.75,
				edgecolor='#0f172a', linewidth=0.9 )
			
			ax.set_title( 'Top Columns by Cardinality', fontsize=10, fontweight='bold' )
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
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Labels' )
		
		with st.expander( label='Edit', icon='✏️', expanded=True ):
			c1, c2 = st.columns( 2, border=True )
			with c1:
				drop_cols = st.multiselect( 'Columns to Drop', df_dataset.columns.tolist( ) )
				if st.button( 'Drop Column' ):
					if len( drop_cols ) == len( df_dataset.columns ):
						st.error( 'Cannot Drop All Columns.' )
					else:
						st.session_state.df_dataset = df_dataset.drop( columns=drop_cols )
						log_step( f'Dropped Columns: {drop_cols}' )
						st.rerun( )
			
			with c2:
				rename_col = st.selectbox( 'Rename Column', [ '<None>' ] + df_dataset.columns.tolist( ) )
				new_name = st.text_input( 'New Column Name' )
				if st.button( 'Rename' ):
					if rename_col != '<None>' and new_name:
						if new_name in df_dataset.columns:
							st.error( 'Column Name Already Exists.' )
						else:
							st.session_state.df_dataset = df_dataset.rename( columns={ rename_col: new_name } )
							log_step( f'Renamed {rename_col} → {new_name}' )
							st.rerun( )
							
			r1, r2 = st.columns( 2 )
			with r1:
				if st.button( label='Reset to Original', icon='🔄' ):
					st.session_state.df_dataset = st.session_state.raw_df.copy( )
					st.session_state.pipeline_log.clear( )
					log_step( 'Reset dataset to original' )
					st.rerun( )
			
			with r2:
				st.download_button( 'Export Dataset (CSV)',
					st.session_state.df_dataset.to_csv( index=False ),
					'dataset.csv', 'text/csv', icon='📥', )
				
		# -------------------------------------------------------------------------------------
		# Probability Distributions
		# -------------------------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Numeric Distributions' )
		
		numeric_dist_cols = [ c for c in df_dataset.columns
				if pd.api.types.is_numeric_dtype( df_dataset[ c ] )
				   and not pd.api.types.is_bool_dtype( df_dataset[ c ] ) ]
		
		if not numeric_dist_cols:
			st.info( 'No numeric columns detected.' )
		else:
			st.caption( f'{len( numeric_dist_cols )} numeric column(s) detected.' )
			
			ctrl1, ctrl2, ctrl3 = st.columns( 3, border=True )
			with ctrl1:
				dist_bins = st.slider( 'Bins', min_value=10, max_value=60,
					value=30, step=5, key='profile_numeric_dist_bins' )
			
			with ctrl2:
				show_kde = st.checkbox( 'Show KDE Overlay', value=True,
					key='profile_numeric_dist_kde' )
			
			with ctrl3:
				dist_mode = st.radio( 'Display', options=[ 'Density', 'Frequency' ],
					horizontal=True, key='profile_numeric_dist_mode' )
			
			st.markdown(
				"""
				<style>
				[data-testid="stMetricLabel"] p {
					font-size: 0.80rem;
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
				unsafe_allow_html=True )
			
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
					sns.histplot( s, bins=dist_bins, kde=show_kde, stat=stat_mode,
						ax=ax, edgecolor='#0f172a',
						line_kws={ 'linewidth': 2.0 } if show_kde else None )
					
					mean_val = float( s.mean( ) )
					median_val = float( s.median( ) )
					
					ax.axvline( mean_val, linestyle='--', linewidth=1.5,
						label=f'Mean: {mean_val:,.2f}' )
					
					ax.axvline( median_val, linestyle=':', linewidth=1.5,
						label=f'Median: {median_val:,.2f}' )
					
					ax.set_title( f'Distribution — {col}', fontsize=10, fontweight='bold' )
					ax.set_xlabel( col )
					ax.set_ylabel( 'Density' if stat_mode == 'density' else 'Frequency' )
					ax.grid( True, alpha=0.25, linestyle='--' )
					ax.spines[ 'top' ].set_visible( False )
					ax.spines[ 'right' ].set_visible( False )
					ax.legend( frameon=False, fontsize=9 )
					
					fig.tight_layout( )
					st.pyplot( fig )
					plt.close( fig )
					
					m1, m2, m3, m4 = st.columns( 4, border=True )
					m1.metric( 'Count', f'{len( s ):,}' )
					m2.metric( 'Mean', f'{mean_val:,.2f}' )
					m3.metric( 'Median', f'{median_val:,.2f}' )
					m4.metric( 'Std', f'{float( s.std( ddof=1 ) ):,.2f}' if len( s ) > 1 else '0.00' )
		
		
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
		st.subheader( cfg.MODE[ 'Descriptive Statistics' ], help=cfg.DESCRIPTIVE_STATISTICS )
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
				font-size: 0.80rem;
			}
			
			[data-testid="stMetricValue"] {
				font-size: 0.95rem;
			}
			
			[data-testid="stMetric"] {
				padding-top: 0.10rem;
				padding-bottom: 0.10rem;
			}
			</style>
			""", unsafe_allow_html=True )
		
		st.markdown( '##### Summary' )
		
		num_c1, num_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with num_c1:
			vars_sel = st.multiselect( 'Select Numeric Variables', all_num_cols,
				default=default_pick( all_num_cols, 3 ) )
			
		sum_c1, sum_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with sum_c1:
			summary_vars = st.multiselect( 'Variables for Summary Table',
				all_num_cols, default=all_num_cols[ : min( 8, len( all_num_cols ) ) ],
				key='desc_summary_vars' )
		
		with sum_c2:
			show_percentiles = st.checkbox( 'Include Percentiles', value=True,
				key='desc_summary_percentiles' )
		
		if summary_vars:
			df_summary_source = df_numeric[ summary_vars ].copy( )
			percentiles = [ 0.05, 0.25, 0.50, 0.75, 0.95 ] if show_percentiles else None
			df_descriptive = df_summary_source.describe( percentiles=percentiles ).T.reset_index( )
			df_descriptive = df_descriptive.rename( columns={ 'index': 'Variable' } )
			df_descriptive[ 'Variance' ] = df_summary_source.var( ddof=1 ).values
			df_descriptive[ 'Missing' ] = df_dataset[ summary_vars ].isna( ).sum( ).values
			df_descriptive[ 'Missing %' ] = (
					df_dataset[ summary_vars ].isna( ).mean( ).values * 100.0 )
			
			df_descriptive[ 'Skew' ] = df_summary_source.skew( ).values
			df_descriptive[ 'Kurtosis' ] = df_summary_source.kurtosis( ).values
			df_descriptive[ 'Zeros' ] = (df_summary_source == 0).sum( ).values
			df_descriptive[ 'Zeros %' ] = (
					(df_summary_source == 0).mean( ).values * 100.0 )
			
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
					'mean': st.column_config.NumberColumn( 'Mean', format='%.2f' ),
					'std': st.column_config.NumberColumn( 'Std', format='%.2f' ),
					'Variance': st.column_config.NumberColumn( 'Variance', format='%.2f' ),
					'min': st.column_config.NumberColumn( 'Min', format='%.2f' ),
					'5%': st.column_config.NumberColumn( 'P5', format='%.2f' ),
					'25%': st.column_config.NumberColumn( 'P25', format='%.2f' ),
					'50%': st.column_config.NumberColumn( 'Median', format='%.2f' ),
					'75%': st.column_config.NumberColumn( 'P75', format='%.2f' ),
					'95%': st.column_config.NumberColumn( 'P95', format='%.2f' ),
					'max': st.column_config.NumberColumn( 'Max', format='%.2f' ),
					'Missing': st.column_config.NumberColumn( 'Missing', format='%.0f' ),
					'Missing %': st.column_config.NumberColumn( 'Missing %', format='%.2f' ),
					'Zeros': st.column_config.NumberColumn( 'Zeros', format='%.0f' ),
					'Zeros %': st.column_config.NumberColumn( 'Zeros %', format='%.2f' ),
					'Skew': st.column_config.NumberColumn( 'Skew', format='%.4f' ),
					'Kurtosis': st.column_config.NumberColumn( 'Kurtosis', format='%.4f' )
			}
			
			column_config = { k: v for k, v in column_config.items( ) if k in df_descriptive.columns }
			
			st.data_editor( df_descriptive, use_container_width=True, hide_index=True,
				disabled=True, column_config=column_config, key='desc_summary_editor' )
		else:
			st.info( 'Select one or more numeric variables to display descriptive statistics.' )
		
		with num_c2:
			dist_bins = st.slider( 'Distribution Bins', min_value=10, max_value=60, value=30,
				step=5, key='desc_dist_bins' )
		
		for col in vars_sel:
			s = pd.to_numeric( df_numeric[ col ], errors='coerce' )
			s = s.replace( [ np.inf, -np.inf ], np.nan ).dropna( )
			
			if s.empty:
				st.warning( f'{col}: no plottable numeric values.' )
				continue
		
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
			st.markdown( f'##### Distribution & Shape — {col}' )
			
			c1, c2 = st.columns( 2, border=True )
			with c1:
				fig, ax = plt.subplots( figsize=(7, 4.75) )
				sns.histplot( s, bins=dist_bins, kde=True, stat='count', ax=ax,
					edgecolor='#0f172a', line_kws={ 'linewidth': 2.0 } )
				
				mean_val = float( s.mean( ) )
				median_val = float( s.median( ) )
				ax.axvline( mean_val, linestyle='--',
					linewidth=1.5, label=f'Mean: {mean_val:,.2f}' )
				
				ax.axvline( median_val, linestyle=':', linewidth=1.5,
					label=f'Median: {median_val:,.2f}' )
				
				ax.set_title( f'Histogram — {col}', fontsize=10, fontweight='bold' )
				ax.set_xlabel( col )
				ax.set_ylabel( 'Frequency' )
				ax.grid( True, alpha=0.25, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				ax.legend( frameon=False, fontsize=9 )
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
				
				m1, m2, m3, m4 = st.columns( 4, border=True )
				m1.metric( 'Count', f'{len( s ):,}' )
				m2.metric( 'Mean', f'{mean_val:,.2f}' )
				m3.metric( 'Median', f'{median_val:,.2f}' )
				m4.metric( 'Deviation', f'{float( s.std( ddof=1 ) ):,.2f}' if len( s ) > 1 else '0.000' )
			
			with c2:
				fig, ax = plt.subplots( figsize=(7, 4.75) )
				stats.probplot( s, plot=ax )
				ax.set_title( f'Q–Q Plot — {col}', fontsize=10, fontweight='bold' )
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
						q1, q2, q3 = st.columns( 3, border=True )
						q1.metric( 'Skew', f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
						q2.metric( 'Kurtosis', f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
						q3.metric( 'Shapiro P', f'{shapiro_p:,.3f}' )
					else:
						q1, q2, q3 = st.columns( 3, border=True )
						q1.metric( 'Skew', f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
						q2.metric( 'Kurtosis', f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
						q3.metric( 'Shapiro P', 'n/a' )
				except Exception:
					q1, q2, q3 = st.columns( 3, border=True )
					q1.metric( 'Skew', f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
					q2.metric( 'Kurtosis', f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
					q3.metric( 'Shapiro P', 'n/a' )

		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Correlation Structure', help=cfg.CORRELATION_STRUCTURE )
		
		cor_c1, cor_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with cor_c1:
			corr_vars = st.multiselect( 'Variables for Correlation', all_num_cols,
				default=default_pick( all_num_cols, 4 ) )
		
		with cor_c2:
			corr_method = st.radio( 'Correlation Method', options=[ 'Pearson', 'Spearman' ],
				horizontal=True, key='desc_corr_method', help=cfg.CORRELATION_HEATMAP )
		
		c3, c4 = st.columns( 2, border=True )
		if len( corr_vars ) >= 2:
			df_correlation = analysis_fillna_mean( df_numeric[ corr_vars ] )
			corr = df_correlation.corr( method=corr_method.lower( ) )
			
			with c3:
				render_table( corr )
			
			with c4:
				fig, ax = plt.subplots( figsize=(7, 6) )
				sns.heatmap( corr, ax=ax, cmap='coolwarm', vmin=-1, vmax=1,
					center=0, annot=True, fmt='.2f', square=False, linewidths=0.5,
					cbar_kws={ 'shrink': 0.85, 'label': 'Correlation' } )
				
				ax.set_title( f'Correlation Heatmap — {corr_method}', fontsize=12,
					fontweight='bold', pad=10 )
				
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
			
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Principal Component Analysis', help=cfg.PCA )
		
		pca_c1, pca_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with pca_c1:
			pca_vars = st.multiselect( 'Select Components', all_num_cols,
				default=default_pick( all_num_cols, 4 ) )
		
		with pca_c2:
			max_components = max( 2, min( 6, len( pca_vars ) ) ) if pca_vars else 2
			n_comp = st.slider( 'Components', 2, max_components, min( 3, max_components ) )
		
		c5, c6 = st.columns( 2, border=True )
		if len( pca_vars ) >= 2:
			X = analysis_fillna_mean( df_numeric[ pca_vars ] )
			Xs = SKStandardScaler( ).fit_transform( X )
			pca = PCA( num=n_comp ).train( Xs )
			
			df_explained = pd.DataFrame(
				{
					'Component': [ f'PC{i + 1}' for i in range( n_comp ) ],
					'Explained Variance (%)': pca.explained_variance_ratio * 100
				} )
			
			with c5:
				render_table( df_explained )
			
			with c6:
				fig, ax = plt.subplots( figsize=(7, 5) )
				bars = ax.bar( df_explained[ 'Component' ], df_explained[ 'Explained Variance (%)' ],
					edgecolor='#0f172a', linewidth=0.9 )
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
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Inferential Statistics' ], help=cfg.INFERENTIAL_STATISTICS )
		st.divider( )
		
		df_dataset = st.session_state.df_dataset
		if df_dataset is None or df_dataset.empty:
			st.info( 'No data available.' )
			st.stop( )
		
		numeric_columns = st.session_state.numeric_columns
		categorical_columns = st.session_state.categorical_columns
		
		if not numeric_columns:
			st.info( 'No numeric variables available for inferential analysis.' )
			st.stop( )
		
		st.markdown(
			"""
			<style>
			[data-testid="stMetricLabel"] p {
				font-size: 0.80rem;
			}
			
			[data-testid="stMetricValue"] {
				font-size: 0.95rem;
			}
			
			[data-testid="stMetric"] {
				padding-top: 0.10rem;
				padding-bottom: 0.10rem;
			}
			</style>
			""", unsafe_allow_html=True )
		
		# -------------------------------------------------------------------------------------
		# INFERENTIAL SUMMARY
		# -------------------------------------------------------------------------------------
		st.markdown( '##### Summary' )
		
		sum_r1c1, sum_r1c2, sum_r1c3 = st.columns( 3, border=True )
		with sum_r1c1:
			summary_y = st.selectbox( 'Summary Outcome Variable', numeric_columns,
				key='infer_summary_y' )
		
		with sum_r1c2:
			summary_x = st.selectbox( 'Summary Second Numeric Variable',
				[ '<None>' ] + [ c for c in numeric_columns if c != summary_y ],
				key='infer_summary_x' )
			if summary_x == '<None>':
				summary_x = None
		
		with sum_r1c3:
			if categorical_columns:
				summary_group = st.selectbox( 'Summary Grouping Variable',
					[ '<None>' ] + categorical_columns, key='infer_summary_group' )
				if summary_group == '<None>':
					summary_group = None
			else:
				summary_group = None
				st.caption( 'No categorical grouping variables available.' )
		
		sum_r2c1, sum_r2c2 = st.columns( 2, border=True )
		with sum_r2c1:
			if len( categorical_columns ) >= 2:
				summary_cat1 = st.selectbox( 'Summary First Categorical Variable',
					categorical_columns, key='infer_summary_cat1' )
			else:
				summary_cat1 = None
				st.caption( 'At least two categorical variables are required.' )
		
		with sum_r2c2:
			if summary_cat1 and len( categorical_columns ) >= 2:
				summary_cat2 = st.selectbox( 'Summary Second Categorical Variable',
					[ c for c in categorical_columns if c != summary_cat1 ],
					key='infer_summary_cat2' )
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
					'Notes': 'Normality Assessment'
				})
			except Exception:
				pass
		
		# -----------------------------------------------------------------
		# Group Comparison Summary
		# -----------------------------------------------------------------
		if summary_group:
			df_group_summary = df_dataset[ [ summary_group, summary_y ] ].copy( )
			df_group_summary[ summary_y ] = pd.to_numeric( df_group_summary[ summary_y ],
				errors='coerce' )
			
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
					} )
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
					} )
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
					pearson_r, pearson_p = stats.pearsonr( x_summary[ pair_mask ],
						y_summary[ pair_mask ] )
					
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
					} )
				except Exception:
					pass
				
				try:
					spearman_rho, spearman_p = stats.spearmanr( x_summary[ pair_mask ],
						y_summary[ pair_mask ] )
					
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
					})
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
					chi2_stat, chi2_p, chi2_dof, expected = stats.chi2_contingency(
						contingency_summary )
					
					n_total = contingency_summary.to_numpy( ).sum( )
					phi2 = chi2_stat / n_total if n_total > 0 else np.nan
					r_dim, c_dim = contingency_summary.shape
					cramers_v = ( np.sqrt( phi2 / min( c_dim - 1, r_dim - 1 ) )
							if min( c_dim - 1, r_dim - 1 ) > 0
							else np.nan )
					
					infer_rows.append(
					{
						'Analysis': 'Categorical Association',
						'Test': 'Chi-Square',
						'Statistic': chi2_stat,
						'P Value': chi2_p,
						'DoF': float( chi2_dof ),
						'Effect Size': cramers_v,
						'N': float( n_total ),
						'Notes': f'{summary_cat1} vs {summary_cat2}'
					} )
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
					'Statistic': st.column_config.NumberColumn( 'Statistic', format='%.2f' ),
					'P-Value': st.column_config.NumberColumn( 'P Value', format='%.2g' ),
					'DoF': st.column_config.NumberColumn( 'DoF', format='%.0f' ),
					'Effect Size': st.column_config.NumberColumn( 'Effect Size', format='%.2f' ),
					'N': st.column_config.NumberColumn( 'N', format='%.0f' ),
					'Notes': st.column_config.TextColumn( 'Notes', width='large' )
			}
			
			st.data_editor( df_infer_summary, use_container_width=True, hide_index=True,
				disabled=True, column_config=infer_column_config, key='infer_summary_editor' )
		else:
			st.info( 'Unable to compute inferential summary for the current selections.' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# -------------------------------------------------------------------------------------
		# NORMALITY + GROUP COMPARISON
		# -------------------------------------------------------------------------------------
		nml_c1, nml_c2 = st.columns( [ 0.5, 0.5 ], border=True, gap='medium' )
		col_group = None
		
		with nml_c1:
			st.markdown( '##### Normality Test', help=cfg.NORMALITY_TESTING )
			col_y = st.selectbox( 'Select Numeric Outcome Variable', numeric_columns )
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
				m2.metric( 'Shapiro-W', f'{stat:,.2f}' )
				m3.metric( 'Shapiro-P', f'{p_value:,.2g}' )
				
				if p_value < 0.05:
					st.caption( 'Distribution departs from normality at α = 0.05.' )
				else:
					st.caption('Distribution does not significantly depart from normality at α=0.05')
			else:
				st.info( 'Not enough observations for normality testing.' )
		
		with nml_c2:
			st.markdown( '##### Group Comparison' )
			if categorical_columns:
				col_group = st.selectbox( 'Select Grouping Variable (optional)',
					[ '<None>' ] + categorical_columns )
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
					sns.boxplot( data=df_group, x=col_group, y=col_y, ax=ax )
					sns.stripplot( data=df_group, x=col_group, y=col_y, ax=ax,
						color='black', alpha=0.45, size=4 )
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
					g2.metric( 'ANOVA F', f'{f_stat:,.2f}' )
					g3.metric( 'ANOVA P', f'{p_anova:,.2g}' )
					g4.metric( 'Kruskal P', f'{p_kw:,.2g}' )
					st.caption( f'Kruskal–Wallis H = {h_stat:.4f}. '
						f'Use the nonparametric result when normality or homoscedasticity is doubtful.' )
				else:
					st.info( 'Not enough valid groups for group comparison.' )
			else:
				st.info( 'Select a grouping variable to compare groups.' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# -------------------------------------------------------------------------------------
		# CORRELATION ANALYSIS
		# -------------------------------------------------------------------------------------
		st.markdown( '##### Correlation Analysis' )
		cor_c1, cor_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with cor_c1:
			candidate_x = [ c for c in numeric_columns if c != col_y ]
			if not candidate_x:
				st.info( 'A second numeric variable is required for Correlation Analysis.' )
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
					r2.metric( 'Pearson R', f'{r_p:,.2f}' )
					r3.metric( 'Pearson P', f'{p_p:,.2g}' )
					r4.metric( 'Spearman R', f'{r_s:,.2f}' )
					st.caption( f'Spearman P = {p_s:.2g}' )
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
						xline = np.linspace( float( x[ mask ].min( ) ),
							float( x[ mask ].max( ) ), 100 )
						ax.plot( xline, m * xline + b, linewidth=2.0, linestyle='--' )
					except Exception:
						pass
				
				ax.set_title( f'Correlation — {col_y} vs {col_x2}', fontsize=12,
					fontweight='bold', pad=10 )
				
				ax.set_xlabel( col_x2 )
				ax.set_ylabel( col_y )
				ax.grid( True, alpha=0.20, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				fig.tight_layout( )
				st.pyplot( fig )
				plt.close( fig )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# -------------------------------------------------------------------------------------
		# CATEGORICAL ASSOCIATION
		# -------------------------------------------------------------------------------------
		st.markdown( '##### Categorical Association' )
		if not categorical_columns or len( categorical_columns ) < 2:
			st.info( 'At least two categorical variables are required for categorical association.' )
		else:
			cat_c1, cat_c2 = st.columns( [ 0.5, 0.5 ] )
			with cat_c1:
				col_cat1 = st.selectbox( 'Select First Categorical Variable', categorical_columns )
				
			with cat_c2:
				col_cat2 = st.selectbox( 'Select Second Categorical Variable',
					[ c for c in categorical_columns if c != col_cat1 ] )
			
			contingency = pd.crosstab( df_dataset[ col_cat1 ], df_dataset[ col_cat2 ] )
			
			if contingency.empty or contingency.shape[ 0 ] < 2 or contingency.shape[ 1 ] < 2:
				st.info( 'Not enough categorical variation for Chi-Square Analysis.' )
			else:
				chi2, p_chi, dof, expected = stats.chi2_contingency( contingency )
				n = contingency.to_numpy( ).sum( )
				phi2 = chi2 / n if n > 0 else np.nan
				r, k = contingency.shape
				cramers_v = np.sqrt( phi2 / min( k - 1, r - 1 ) ) if min( k - 1, r - 1 ) > 0 else np.nan
				
				ca1, ca2 = st.columns( 2, border=True )
				with ca1:
					st.data_editor( contingency, key='inference_data',
						height='stretch', num_rows='dynamic')
				
				with ca2:
					fig, ax = plt.subplots( figsize=( 7, 5.5 ) )
					sns.heatmap( contingency, annot=True, fmt='d', cmap='Blues',
						linewidths=0.5, ax=ax, cbar_kws={ 'shrink': 0.85, 'label': 'Count' } )
					
					ax.set_title( f'Contingency Heatmap — {col_cat1} vs {col_cat2}',
						fontsize=12, fontweight='bold', pad=10 )
					
					ax.set_xlabel( col_cat2 )
					ax.set_ylabel( col_cat1 )
					fig.tight_layout( )
					st.pyplot( fig )
					plt.close( fig )
			
				cm1, cm2, cm3, cm4 = st.columns( 4, border=True )
				cm1.metric( 'Chi-Square', f'{chi2:,.2f}' )
				cm2.metric( 'P Value', f'{p_chi:,.2g}' )
				cm3.metric( 'DoF', f'{dof:,}' )
				cm4.metric( "Cramér's V", f'{cramers_v:,.2f}' if np.isfinite( cramers_v ) else 'n/a' )

# ============================================
# ANOMALY DETECTION MODE
# ============================================
elif mode == 'Anomaly Detection':
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Anomaly Detection' ] )
		st.divider( )
		if st.session_state.df_dataset is None:
			st.info( 'No data loaded.' )
			st.stop( )
		
		df_dataset = st.session_state.df_dataset
		df_numeric = clean_numeric( df_dataset.select_dtypes( include=[ np.number ] ) )
		if df_numeric.empty:
			st.info( 'No usable numeric columns available for anomaly detection.' )
			st.stop( )
		
		st.markdown(
			"""
			<style>
			[data-testid="stMetricLabel"] p {
				font-size: 0.80rem;
			}
			
			[data-testid="stMetricValue"] {
				font-size: 0.95rem;
			}
			
			[data-testid="stMetric"] {
				padding-top: 0.10rem;
				padding-bottom: 0.10rem;
			}
			</style>
			""", unsafe_allow_html=True )
		
		msg = 'Select fields/columns containing numerical data to analyze outliers'
		st.markdown( '##### Feature Selection', help=msg )
		aml_c1, aml_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with aml_c1:
			all_num_cols = df_numeric.columns.tolist( )
			preferred = [ c for c in all_num_cols if c.lower( ) in ('py', 'cy', 'by') ]
			default_vars = preferred if preferred else default_pick( all_num_cols, 2 )
			vars_sel = st.multiselect( 'Variables to Analyze', all_num_cols, default=default_vars )
			
			if not vars_sel:
				st.info( 'Select at least one numeric variable to run anomaly detection.' )
				st.stop( )
		
		with aml_c2:
			analysis_scale = st.checkbox( 'Use Analysis-Only Standardization', value=False )
			df_analysis = df_numeric[ vars_sel ].astype( float ).copy( )
	
			if analysis_scale and len( vars_sel ) > 1:
				df_analysis = pd.DataFrame( SKStandardScaler( ).fit_transform( df_analysis.values ),
					columns=df_analysis.columns, index=df_analysis.index )
				
			if analysis_scale and len( vars_sel ) > 1:
				df_analysis[ : ] = SKStandardScaler( ).fit_transform( df_analysis.values )
		
		# -------------------------------------------------------------------------
		# Method Selection
		# -------------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Detection Methods' )
		
		c_m1, c_m2 = st.columns( 2, border=True )
		with c_m1:
			use_z = st.checkbox( 'Z-Score', value=True, help=cfg.Z_SCORE )
			use_mz = st.checkbox( 'Modified Z-Score (MAD)', value=True, help=cfg.MODIFIED_Z )
			use_iqr = st.checkbox( 'IQR Fence', value=True, help=cfg.IQR )
		
		with c_m2:
			use_mahal = st.checkbox( 'Mahalanobis Distance', value=True, help=cfg.MAHALANOBIS )
			use_iforest = st.checkbox( 'Isolation Forest', value=True, help=cfg.ISOLATION_FOREST )
			use_lof = st.checkbox( 'Local Outlier Factor (LOF)', value=False, help=cfg.LOF )
		
		# -------------------------------------------------------------------------
		# Threshold Controls
		# -------------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Thresholds' )
		
		c_t1, c_t2 = st.columns( 2, border=True )
		with c_t1:
			z_thresh = st.slider( 'Z / Modified Z threshold', 2.0, 5.0, 3.0, 0.1, help=cfg.MODIFIED_Z )
			iqr_mult = st.slider( 'IQR Multiplier', 1.0, 3.0, 1.5, 0.1, help=cfg.IQR_MULTIPLIER )
		
		with c_t2:
			lof_k = st.slider( 'LOF Neighbors (k)', 5, 50, 20, 1, help=cfg.LOF_K )
			min_methods = st.slider( 'Consensus: minimum methods flagging a row', 1, 4, 1, 1,
			help=cfg.MIN_METHODS)
		
		# -------------------------------------------------------------------------
		# Run Detection
		# -------------------------------------------------------------------------
		df_anamolies = pd.DataFrame( index=df_analysis.index )
		for col in vars_sel:
			s = df_analysis[ col ].dropna( )
			if s.empty:
				continue
			
			if use_z:
				z = (s - s.mean( )) / s.std( ) if s.std( ) else pd.Series( 0.0, index=s.index )
				df_anamolies[ f'{col}_z' ] = z.abs( ) >= z_thresh
			
			if use_mz:
				med = s.median( )
				mad = np.median( np.abs( s - med ) )
				if mad == 0:
					mz = pd.Series( 0.0, index=s.index )
				else:
					mz = 0.6745 * (s - med) / mad
				df_anamolies[ f'{col}_mz' ] = mz.abs( ) >= z_thresh
			
			if use_iqr:
				q1, q3 = s.quantile( 0.25 ), s.quantile( 0.75 )
				iqr = q3 - q1
				lo = q1 - iqr_mult * iqr
				hi = q3 + iqr_mult * iqr
				df_anamolies[ f'{col}_iqr' ] = (s < lo) | (s > hi)
		
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
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Outlier Summary' )
		
		if df_anamolies.empty:
			st.info( 'No anomalies detected under the selected methods and thresholds.' )
			st.stop( )
		
		df_anamolies = df_anamolies.fillna( False )
		df_anamolies[ 'methods_flagged' ] = df_anamolies.sum( axis=1 )
		anomalies = df_anamolies[ df_anamolies[ 'methods_flagged' ] >= min_methods ].copy( )
		
		m1, m2, m3, m4 = st.columns( 4, border=True )
		m1.metric( 'Rows Analyzed', f'{len( df_analysis ):,}' )
		m2.metric( 'Flagged Rows', f'{len( anomalies ):,}' )
		m3.metric( 'Flag Rate %',
			f'{(100.0 * len( anomalies ) / max( 1, len( df_analysis ) )):,.2f}' )
		m4.metric( 'Min Methods', f'{min_methods:,}' )
		
		c_o1, c_o2 = st.columns( 2, border=True )
		
		with c_o1:
			st.markdown( '##### Flagged Observations' )
			render_table( anomalies.sort_values( 'methods_flagged', ascending=False ) )
		
		with c_o2:
			st.markdown( '##### Flag Count Distribution' )
			if anomalies.empty:
				st.info( 'No rows met the current consensus threshold.' )
			else:
				fig, ax = plt.subplots( figsize=(7, 5) )
				vc = anomalies[ 'methods_flagged' ].value_counts( ).sort_index( )
				bars = ax.bar( vc.index.astype( str ), vc.values, width=0.75,
					edgecolor='black', linewidth=0.9 )
				ax.set_xlabel( 'Number of Methods Flagging' )
				ax.set_ylabel( 'Observation Count' )
				ax.set_title( 'Consensus Strength', fontsize=10, fontweight='bold' )
				ax.grid( axis='y', alpha=0.25, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				ax.bar_label( bars, padding=3, fontsize=9 )
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
		
		# -------------------------------------------------------------------------
		# Visualization — Distribution with Anomalies
		# -------------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Empirical Cumulative Distribution Function (ECDF)', help=cfg.ECDF )
		
		for col in vars_sel:
			if col not in df_analysis.columns:
				continue
			
			s = pd.to_numeric( df_analysis[ col ], errors='coerce' ).replace(
				[ np.inf, -np.inf ], np.nan )
			
			s_clean = s.dropna( )
			if s_clean.empty:
				continue
			
			flagged_idx = anomalies.index.intersection( s_clean.index )
			flagged_vals = s_clean.loc[ flagged_idx ] if not flagged_idx.empty else pd.Series( dtype=float )
		
			c_v1, c_v2 = st.columns( 2, border=True )
			with c_v1:
				fig, ax = plt.subplots( figsize=(7, 5) )
				
				s_sorted = np.sort( s_clean.values.astype( float ) )
				n_vals = len( s_sorted )
				y_ecdf = np.arange( 1, n_vals + 1 ) / n_vals
				
				ax.step( s_sorted, y_ecdf, where='post', linewidth=2.0, label='ECDF' )
				
				if not flagged_vals.empty:
					flagged_array = flagged_vals.values.astype( float )
					flagged_y = np.searchsorted( s_sorted, flagged_array, side='right' ) / n_vals
					ax.scatter( flagged_array, flagged_y, color='crimson', alpha=0.90, s=42,
						edgecolors='black', linewidths=0.5, label='Flagged' )
				
				mean_val = float( s_clean.mean( ) )
				median_val = float( s_clean.median( ) )
				
				ax.axvline( mean_val, linestyle='--', linewidth=1.4,
					label=f'Mean: {mean_val:,.2f}' )
				
				ax.axvline( median_val, linestyle=':', linewidth=1.4,
					label=f'Median: {median_val:,.2f}' )
				
				ax.set_title( f'{col} — ECDF with Anomalies', fontsize=10, fontweight='bold' )
				ax.set_xlabel( col )
				ax.set_ylabel( 'Cumulative Probability' )
				ax.set_ylim( 0.0, 1.02 )
				ax.grid( True, alpha=0.25, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				ax.legend( frameon=False, fontsize=9 )
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
			
			with c_v2:
				fig, ax = plt.subplots( figsize=(7, 5) )
				sns.violinplot( x=s_clean.values, ax=ax, inner=None, cut=0, linewidth=0.9 )
				ax.boxplot( s_clean.values, vert=False, widths=0.20, patch_artist=True,
					boxprops=dict( facecolor='white', edgecolor='black', linewidth=1.0 ),
					medianprops=dict( color='black', linewidth=1.4 ),
					whiskerprops=dict( color='black', linewidth=0.9 ),
					capprops=dict( color='black', linewidth=0.9 ),
					flierprops=dict( marker='o', markerfacecolor='#475569', markeredgecolor='black',
						markersize=4, alpha=0.7 ) )
				
				if not flagged_vals.empty:
					ax.scatter( flagged_vals.values, np.ones( len( flagged_vals ) ), color='crimson',
						alpha=0.85, s=34, edgecolors='black', linewidths=0.4, label='Flagged',
						zorder=3 )
				
				ax.axvline( float( s_clean.mean( ) ), linestyle='--', linewidth=1.4 )
				ax.axvline( float( s_clean.median( ) ), linestyle=':', linewidth=1.4 )
				ax.set_title( f'{col} — Violin / Box Summary', fontsize=10, fontweight='bold' )
				ax.set_xlabel( col )
				ax.set_yticks( [ ] )
				ax.grid( axis='x', alpha=0.25, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				if not flagged_vals.empty:
					ax.legend( frameon=False, fontsize=9 )
				
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# -------------------------------------------------------------------------
		# Bivariate View
		# -------------------------------------------------------------------------
		if len( vars_sel ) >= 2:
			st.markdown( '##### Bivariate View' )
			
			x_col = vars_sel[ 0 ]
			y_col = vars_sel[ 1 ]
			df_scatter = df_analysis[ [ x_col, y_col ] ].copy( ).dropna( )
			
			if not df_scatter.empty:
				flag_mask = df_scatter.index.isin( anomalies.index )
				
				fig, ax = plt.subplots( figsize=(8, 5.5) )
				ax.scatter( df_scatter.loc[ ~flag_mask, x_col ].values,
					df_scatter.loc[ ~flag_mask, y_col ].values, s=34, alpha=0.70, edgecolors='black',
					linewidths=0.5, label='Inliers' )
				
				if flag_mask.any( ):
					ax.scatter( df_scatter.loc[ flag_mask, x_col ].values,
						df_scatter.loc[ flag_mask, y_col ].values, s=52, alpha=0.92, edgecolors='black',
						linewidths=0.7, c='crimson', marker='X', label='Flagged' )
				
				ax.set_title( f'Anomaly Scatter — {x_col} vs {y_col}', fontsize=12,
					fontweight='bold', pad=10 )
				ax.set_xlabel( x_col )
				ax.set_ylabel( y_col )
				ax.grid( True, alpha=0.20, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				ax.legend( frameon=False )
				fig.tight_layout( )
				st.pyplot( fig, use_container_width=True )
				plt.close( fig )
		
		# -------------------------------------------------------------------------
		# Export
		# -------------------------------------------------------------------------
		st.download_button( "Export Anomaly Table (CSV)", anomalies.to_csv( ), "anomalies.csv",
			"text/csv" )

# ============================================
# CLASSIFICATION MODE
# ============================================
elif mode == 'Classification Models':
	df_original = st.session_state.get( 'df_dataset', None )
	df_dataset = st.session_state.get( 'df_dataset', None )
	df_working = st.session_state.get( 'df_working', None )
	df_processed = st.session_state.get( 'df_processed', None )
	df_classification = st.session_state.get( 'df_classification', None )
	df_model = st.session_state.get( 'df_model', None )
	df_scores = st.session_state.get( 'df_scores', None )
	df_predictions = st.session_state.get( 'df_predictions', None )
	numeric_columns = st.session_state.get( 'numeric_columns', [ ] )
	categorical_columns = st.session_state.get( 'categorical_columns', [ ] )
	features = st.session_state.get( 'features', [ ] )
	targets = st.session_state.get( 'targets', [ ] )
	active_features = st.session_state.get( 'active_features', [ ] )
	active_targets = st.session_state.get( 'active_targets', [ ] )
	X_data = st.session_state.get( 'X_data', None )
	X_train = st.session_state.get( 'X_train', None )
	X_test = st.session_state.get( 'X_test', None )
	y_train = st.session_state.get( 'y_train', None )
	y_test = st.session_state.get( 'y_test', None )
	y_series = st.session_state.get( 'y_series', None )
	y_predictions = st.session_state.get( 'y_predictions', None )
	elapsed_seconds = st.session_state.get( 'elapsed_seconds', 0.0 )
	
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Classification Models' ] )
		st.caption( 'Predictive Models for Categorical, Discrete-Values' )
		st.divider( )
		
		if df_dataset is None or df_dataset.empty:
			st.warning( '⚠️ No dataset loaded.' )
			st.stop( )
		
		df_original = df_dataset.copy( )
		st.session_state[ 'df_original' ] = df_original.copy( )
		numeric_columns = [ c for c in df_original.columns
		                    if pd.api.types.is_numeric_dtype( df_original[ c ] ) ]
		
		categorical_columns = [ c for c in df_original.columns if c not in numeric_columns ]
		st.session_state[ 'numeric_columns' ] = numeric_columns
		st.session_state[ 'categorical_columns' ] = categorical_columns
		if not numeric_columns or not categorical_columns:
			st.warning( '⚠️ Classifications requires numeric features and a float target.' )
			st.stop( )
		
		# ======================================================================================
		# Data Selection
		# ======================================================================================
		st.markdown( '##### Feature Selection' )
		st.caption( f'Samples: {len( df_original ):,} | Features: {len( df_original.columns ):,}' )
		
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			features = st.multiselect( 'Select Features', options=df_original.columns,
				key='classification_features' )
		
		with col_c2:
			target_options = [ t for t in df_original.columns if t not in features  ]
			targets = st.selectbox( 'Select Target', options=target_options,
				key='classification_target' )
		
		# Create Button
		sel_b1, sel_b2 = st.columns( [ 0.5, 0.5 ] )
		with sel_b1:
			if st.button( 'Create Working Dataset', icon='➕', key='classification_create_dataset',
					use_container_width=True ):
				
				selected_all = features.copy( )
				if targets not in selected_all:
					selected_all.append( targets )
				
				if selected_all:
					df_working = df_original[ selected_all ].copy( )
				else:
					df_working = df_original.copy( )
				
				st.session_state[ 'features' ] = features.copy( )
				st.session_state[ 'targets' ] = [ targets ] if targets else [ ]
				st.session_state[ 'df_working' ] = df_working.copy( )
				commit_frame( df_working )
				st.success( 'Working Dataset Created!' )
		
		# Reset Button
		with sel_b2:
			if st.button( 'Reset Working Dataset', icon='🔁', key='classification_reset_to_original',
					use_container_width=True ):
				
				st.session_state[ 'features' ] = [ ]
				st.session_state[ 'targets' ] = [ ]
				st.session_state[ 'df_working' ] = pd.DataFrame( )
				df_working = None
				df_processed = None
				st.success( 'Reset to Original' )
				st.rerun( )
		
			
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		if df_working is None:
			st.stop( )
		st.markdown( '##### Working Data' )
		
		st.caption( f'Samples: {len( df_working ):,} | Feautres: {len( df_working.columns ):,}' )
		st.data_editor( df_working, key='classification_working_data' )
		
		
		# -----------------------------------------------------------------
		# Data Processing
		# -----------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Feature-Engineering' )
		
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			
			with st.expander( label='Data Scaling', icon='⚖️', key='classification_scalers' ):
				
				with st.expander( 'Standard Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.STANDARD_SCALER )
					
					columns = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_standard_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button(  label='Apply', icon='✔️', use_container_width=True,
								key='classification_standard_scaler_apply'  ):
							
							if columns:
								scaler = StandardScaler( )
								df_processed = df_working.copy( )
								result = scaler.train_transform( df_processed[ columns ].to_numpy( ))
								df_processed[ columns ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Standard Scaler applied.' )
							
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_standard_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
					
				with st.expander( 'Min-Max Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.MINMAX_SCALER )
					
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_minmax_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_minmax_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MinMaxScaler( )
								result = scaler.train_transform( df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Min-Max Scaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔄',
								key='classification_minmax_scaler_reset', use_container_width=True ):
							
							st.session_state[ 'df_processed' ] = None
							df_processed = None
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.ROBUST_SCALER )
					
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_robust_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_robust_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = RobustScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ scale_cols ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Robust Scaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_robust_scaler_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_processed' ] = None
							df_processed = None
							st.success( 'Reset to Working.' )
							
				with st.expander( 'Normal Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left', help=cfg.NORMAL_SCALER )
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ],
						index=1, key='classification_normal_scaler_norm' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_normal_scaler_apply',
								use_container_width=True ):
							
							if scale_cols:
								scaler = NormalScaler( norm=norm )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ columns ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'NormalScaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_normal_scaler_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.MAXABS_SCALER )
					
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_maxabs_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_maxabs_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MaxAbsScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ columns ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'MaxAbsScaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_maxabs_scaler_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
							
			with st.expander( label='Data Imputation', icon='🧹', key='classification_imputers' ):
				
				with st.expander( 'Mean Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.MEAN_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='classification_mean_imputer_indicator' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_mean_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = MeanImputer( strategy='mean', add_indicator=add_indicator )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'mean_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'MeanImputer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_mean_imputer_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.NEAREST_NEIGHBOR_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1,
						value=5, step=1, key='classification_nearest_imputer_neighbors' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_nearest_imputer_apply',
								use_container_width=True ):
							
							if impute_cols:
								imputer = NearestImputer( neighbors=int( neighbors ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'nearest_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Nearest Imputer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_nearest_imputer_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.ITERATIVE_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_iterative_imputer_cols' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1,
						value=10, step=1, key='classification_iterative_imputer_max_iter' )
					
					random_state = st.number_input( 'Random State', min_value=0,
						value=0, step=1, key='classification_iterative_imputer_random_state' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Iterative Imputer', icon='✔️',
								key='classification_iterative_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = IterativeImputer( max_iter=int( max_iter ),
									random_state=int( random_state ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'iterative_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Iterative Imputer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_iterative_imputer_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.SIMPLE_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_simple_imputer_cols' )
					
					strategy = st.selectbox( 'Strategy',
						options=[ 'mean', 'median', 'most_frequent', 'constant' ],
						key='classification_simple_imputer_strategy' )
					
					fill_value = st.text_input( 'Fill Value', value='0.0',
						key='classification_simple_imputer_fill_value' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='classification_simple_imputer_indicator' )
					
					keep_empty_features = st.checkbox( 'Keep Empty Features', value=False,
						key='classification_simple_imputer_keep_empty' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SimpleImputer', icon='✔️',
								key='classification_simpleimputer_apply',
								use_container_width=True ):
							
							if impute_cols:
								if strategy in [ 'mean', 'median' ]:
									df_input = df_processed[ impute_cols ].apply(
										pd.to_numeric, errors='coerce' )
									fill_object: object = 0.0
								elif strategy == 'constant':
									df_input = df_processed[ impute_cols ].copy( )
									fill_object = fill_value
								else:
									df_input = df_processed[ impute_cols ].copy( )
									fill_object = fill_value
								
								imputer = SimpleImputer( strategy=strategy, fill_value=fill_object,
									add_indicator=add_indicator,
									keep_empty_features=keep_empty_features )
								
								result = imputer.train_transform( df_input.to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'simple_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Simple Imputer Applied' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_simple_imputer_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
			
			with st.expander( label='Data Encoding', icon='🔣', key='classification_encoders' ):
				
				with st.expander( 'One-Hot Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.ONEHOT_ENCODER )
					
					encode_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='classification_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0,
						key='classification_onehot_unknown' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_onehot_apply',
								use_container_width=True ):
							
							if encode_cols:
								encoder = OneHotEncoder( sparse=bool( sparse ), unknown=unknown )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, encode_cols,
									result, 'onehot' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'One-Hot Encoder applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_onehot_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.ORDINAL_ENCODER )
					
					encode_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_ordinal_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_ordinal_apply',
								use_container_width=True ):
							
							if encode_cols:
								encoder = OrdinalEncoder( )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								df_processed[ encode_cols ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Ordinal Encoder Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_ordinal_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Label Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_ENCODER )
					
					target_col = st.selectbox( 'Column', options=df_working.columns,
						key='classification_label_encoder_col' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_label_encoder_apply',
								use_container_width=True ):
							
							if target_col:
								encoder = LabelEncoder( )
								result = encoder.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed[ target_col ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Label Encoder Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_label_encoder_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
					
					st.session_state[ 'df_processed' ] = df_processed
				
				with st.expander( 'Target Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.TARGET_ENCODER )
					
					encode_cols = st.multiselect( 'Categorical Feature Columns',
						options=df_working.columns, key='classification_target_encoder_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='classification_target_encoder_target_col' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_target_encoder_apply',
								use_container_width=True ):
							
							if encode_cols and target_col:
								df_processed = df_working.copy( )
								encoder = TargetEncoder( )
								X_enc = df_processed[ encode_cols ].astype( str ).to_numpy( )
								y_enc = df_processed[ target_col ].to_numpy( )
								result = encoder.train_transform( X_enc, y_enc )
								
								df_processed = replace_columns( df_processed, encode_cols, result,
									'target_encoder' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Target Encoder Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_target_encoder_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.POLYNOMIAL_FEATURES )
					
					poly_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4,
						value=2, key='classification_polynomial_degree' )
					
					interaction = st.checkbox( 'Interaction Only', value=True,
						key='classification_polynomial_interaction' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_polynomial_apply',
								use_container_width=True ):
							
							if poly_cols:
								df_processed = df_working.copy( )
								encoder = PolynomialFeatures( degree=int( degree ),
									interaction=bool( interaction ) )
								
								result = encoder.train_transform(
									df_processed[ poly_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, poly_cols, result,
									'polynomial' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'PolynomialFeatures applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_polynomial_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
							
		with feature_c2:
			
			with st.expander( label='Data Transformation', icon='⚡', key='classification_transformers' ):
				
				with st.expander( 'Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.BINARIZER )
					
					transform_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_binarizer_cols' )
					
					threshold = st.number_input( 'Threshold', value=0.0, step=0.1,
						key='classification_binarizer_threshold' )
					
					copy = st.checkbox( 'Copy', value=True, key='classification_binarizer_copy' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Binarizer', key='classification_binarizer_apply',
								use_container_width=True ):
							
							if transform_cols:
								df_processed = df_working.copy( )
								transformer = Binarizer( threshold=float( threshold ),
									copy=bool( copy ) )
								
								result = transformer.train_transform(
									df_processed[ transform_cols ].to_numpy( ) )
								
								df_processed[ transform_cols ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Binarizer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_binarizer_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_BINARIZER )
					
					target_col = st.selectbox( 'Column', options=df_working.columns,
						key='classification_label_binarizer_col' )
					
					pos_label = st.number_input( 'Positive Label', value=1, step=1,
						key='classification_label_binarizer_pos' )
					
					neg_label = st.number_input( 'Negative Label', value=0, step=1,
						key='classification_label_binarizer_neg' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='classification_label_binarizer_sparse' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply LabelBinarizer', key='classification_lblbinarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ), sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, [ target_col ], result,
									'label_binarizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Label Binarizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_lblbinarizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.MULTILABEL_BINARIZER )
					
					target_col = st.selectbox( 'Column', options=df_working.columns,
						key='classification_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='classification_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='classification_multilabel_binarizer_sparse' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_multilabel_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								y_multi = parse_multilabel_series( df_processed[ target_col ],
									delimiter=delimiter )
								
								transformer = MultiLabelBinarizer( classes=None,
									sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform( y_multi )
								df_processed = replace_columns( df_processed, [ target_col ],
									result, 'multilabel_binarizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Multi-Label Binarizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_multilabel_binarizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'TF-IDF Transformer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.TDIDF_TRANSFORMER )
					
					text_count_cols = st.multiselect( 'Count Matrix Columns',  options=df_working.columns,
						key='classification_tfidf_transformer_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', None ],
						index=1, key='classification_tfidf_transformer_norm' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='classification_tfidf_transformer_use_idf' )
					
					smooth_idf = st.checkbox( 'Smooth IDF', value=True,
						key='classification_tfidf_transformer_smooth_idf' )
					
					sublinear_tf = st.checkbox( 'Sublinear TF', value=False,
						key='classification_tfidf_transformer_sublinear' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_tfidf_transformer_apply',
								use_container_width=True ):
							if text_count_cols:
								df_processed = df_working.copy( )
								transformer = TfidfTransformer( norm=norm, use_idf=bool( use_idf ),
									smooth_idf=bool( smooth_idf ), sublinear_tf=bool( sublinear_tf ) )
								
								result = transformer.train_transform(
									df_processed[ text_count_cols ].apply(
										pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, text_count_cols,
									result, 'tfidf_transformer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'TFIDF Transformer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_tfidf_transformer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Column Transformer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.COLUMN_TRANSFORMER )
					
					numeric_columns = st.multiselect( 'Numeric Columns', options=df_working.columns,
						key='classification_column_transformer_numeric_columns' )
					
					categorical_columns = st.multiselect( 'Categorical Columns',
						options=df_working.columns,
						key='classification_column_transformer_categorical_columns' )
					
					numeric_transform = st.selectbox( 'Numeric Transformer',
						options=[ 'StandardScaler', 'MinMaxScaler', 'RobustScaler',
								'MaxAbsScaler', 'Binarizer', 'None' ],
						key='classification_column_transformer_numeric_transform' )
					
					categorical_transform = st.selectbox( 'Categorical Transformer',
						options=[ 'OneHotEncoder', 'OrdinalEncoder', 'None' ],
						key='classification_column_transformer_categorical_transform' )
					
					remainder = st.selectbox( 'Remainder', options=[ 'drop', 'passthrough' ],
						key='classification_column_transformer_remainder' )
					
					sparse_threshold = st.slider( 'Sparse Threshold', min_value=0.0,
						max_value=1.0, value=0.3,
						key='classification_column_transformer_sparse_threshold' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Column Transformer',
								key='classification_column_transformer_apply',
								use_container_width=True ):
							df_processed = df_working.copy( )
							transformers = [ ]
							
							if numeric_columns and numeric_transform != 'None':
								if numeric_transform == 'StandardScaler':
									numeric_model = StandardScaler( ).model
								elif numeric_transform == 'MinMaxScaler':
									numeric_model = MinMaxScaler( ).model
								elif numeric_transform == 'RobustScaler':
									numeric_model = RobustScaler( ).model
								elif numeric_transform == 'MaxAbsScaler':
									numeric_model = MaxAbsScaler( ).model
								else:
									numeric_model = Binarizer( ).model
								
								transformers.append( ('numeric', numeric_model, numeric_columns) )
							
							if categorical_columns and categorical_transform != 'None':
								if categorical_transform == 'OneHotEncoder':
									categorical_model = OneHotEncoder( sparse=False,
										unknown='ignore' ).model
								else:
									categorical_model = OrdinalEncoder( ).model
								
								transformers.append( ('categorical', categorical_model,
								                      categorical_columns) )
							
							if transformers:
								transformer = ColumnTransformer( transformers=transformers,
									remainder=remainder, sparse_threshold=float( sparse_threshold ),
									n_jobs=None, transformer_weights=None, verbose=False )
								
								result = transformer.train_transform( df_processed )
								df_processed = normalize_result_frame( result=result,
									index=df_processed.index, prefix='column_transformer',
									columns=None )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'ColumnTransformer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_column_transformer_reset',
								use_container_width=True ):
							
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
			
			with st.expander( label='Feature Extration', icon='⛏️', key='classification_extractors' ):
				
				with st.expander( 'TF-IDF Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.TDIDF_VECTORIZER )
					
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns,
						key='classification_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='classification_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='classification_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='classification_tfidf_vectorizer_use_idf' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_tfidf_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = TfidfVectorizer( ngram_range=( 1, int( ngram_max ) ),
									max_features=None if int( max_features ) == 0 else int( max_features ),
									use_idf=bool( use_idf ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'tfidf_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'TFIDF Vectorizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_tfidf_vectorizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.COUNT_VECTORIZER )
					
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns,
						key='classification_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='classification_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0,
						step=1, key='classification_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='classification_count_vectorizer_binary' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_count_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = CountVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int( max_features ),
									binary=bool( binary ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'count_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Count Vectorizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_count_vectorizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.HASH_VECTORIZER )
					
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns,
						key='classification_hash_vectorizer_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='classification_hash_vectorizer_n_features' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3,
						value=1, key='classification_hash_vectorizer_ngram_max' )
					
					binary = st.checkbox( 'Binary', value=False,
						key='classification_hash_vectorizer_binary' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='classification_hash_vectorizer_alternate_sign' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_hash_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = HashVectorizer( num=int( n_features ),
									ngram_range=(1, int( ngram_max )), binary=bool( binary ),
									alternate_sign=bool( alternate_sign ) )
								df_processed = apply_text_vectorizer( df_processed,
									text_cols, transformer, 'hash_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'HashVectorizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_hash_vectorizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.DICT_VECTORIZER )
					
					dict_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='classification_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='classification_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='classification_dict_vectorizer_sort' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_dict_vectorizer_apply',
								use_container_width=True ):
							
							if dict_cols:
								df_processed = df_working.copy( )
								transformer = DictVectorizer( dtype=np.float64, separator=separator,
									sparse=bool( sparse ), sort=bool( sort ) )
								
								df_processed = apply_dict_transform( df_processed, dict_cols,
									transformer, 'dict_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'DictVectorizer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_dict_vectorizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.FEATURE_HASHER )
					
					hash_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_feature_hasher_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='classification_feature_hasher_n_features' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='classification_feature_hasher_alternate_sign' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_feature_hasher_apply',
								use_container_width=True ):
							
							if hash_cols:
								df_processed = df_working.copy( )
								transformer = FeatureHasher( n_features=int( n_features ),
									input_type='dict', dtype=np.float64,
									alternate_sign=bool( alternate_sign ) )
								
								df_processed = apply_dict_transform( df_processed, hash_cols,
									transformer, 'feature_hasher' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'FeatureHasher applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_feature_hasher_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
			
			with st.expander( label='Dimensionality Reduction', icon='🎚️', key='classification_selectors' ):
				
				with st.expander( 'Variance Threshold', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.VARIANCE_THRESHOLD )
					
					select_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0,
						step=0.01, key='classification_variance_threshold_value' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_variance_threshold_apply',
								use_container_width=True ):
							
							if select_cols:
								df_processed = df_working.copy( )
								selector = VarianceThreshold( thresh=float( threshold ) )
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result,
									'variance_threshold' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'VarianceThreshold applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_variance_threshold_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Canonical Correlation Analysis (CCA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.CCA )
					
					X_cols = st.multiselect( 'Predictor Columns', options=df_working.columns,
						key='classification_cca_x_cols' )
					
					y_cols = st.multiselect( 'Target Columns', options=df_working.columns,
						key='classification_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2,
						step=1, key='classification_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True,
						key='classification_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500,
						step=1, key='classification_cca_max_iter' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_cca_apply',
								use_container_width=True ):
							if X_cols and y_cols:
								df_processed = df_working.copy( )
								selector = CCA( num=int( n_components ), scale=bool( scale ),
									size=int( max_iter ) )
								
								result = selector.train_transform( df_processed[ X_cols ].to_numpy( ),
									df_processed[ y_cols ].to_numpy( ) )
								
								df_result = normalize_result_frame( result=result,
									index=df_processed.index, prefix='cca', columns=None )
								
								df_processed = pd.concat(
									[ df_processed.drop( columns=X_cols + y_cols, errors='ignore' ),
									  df_result ], axis=1 )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'CCA Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_cca_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Principle Component Analysis (PCA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.PCA)
					
					select_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='classification_pca_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2,
						step=1, key='classification_pca_components' )
					
					solver = st.selectbox( 'SVD Solver',
						options=[ 'auto', 'full', 'randomized', 'covariance_eigh', 'arpack' ],
						key='classification_pca_solver' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_pca_apply',
								use_container_width=True ):
							
							if select_cols:
								df_processed = df_working.copy( )
								selector = PCA( num=int( n_components ), solver=solver )
								
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result, 'pca' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'PCA applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_pca_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Select-Best', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SELECT_BEST )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='classification_selectbest_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='classification_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
								'mutual_info_regression' ],
						key='classification_selectbest_score_name' )
					
					k_best = st.number_input( 'K', min_value=1, value=5, step=1,
						key='classification_selectbest_k' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_selectbest_apply',
								use_container_width=True ):
							
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectBest(
									score_func=score_function_from_name( score_name ),
									num=int( k_best ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'select_best' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Select Best Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_selectbest_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Select-Percent', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SELECT_PERCENT )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='classification_selectpercent_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='classification_selectpercent_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
								'mutual_info_regression' ],
						key='classification_selectpercent_score_name' )
					
					percentile = st.slider( 'Percentile', min_value=1, max_value=100, value=10,
						key='classification_selectpercent_percentile' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SelectPercent',
								key='classification_selectpercent_apply', use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectPercent(
									score_func=score_function_from_name( score_name ),
									pct=int( percentile ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'select_percent' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'SelectPercent applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_selectpercent_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Sequential Back Selection (SBS)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SBS )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='classification_sbs_x_cols' )
					
					target_col = st.selectbox( 'Target Column',
						options=df_working.columns, key='classification_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='classification_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='classification_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1,
						step=1, key='classification_sbs_random_state' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_sbs_apply',
								use_container_width=True ):
							
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SBS( classifier=None, k_features=int( k_features ),
									test_size=float( test_size ), random_state=int( random_state ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'sbs' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'SBS applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_sbs_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Recursive Feature Elimination (RFA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.RFE )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='classification_rfe_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='classification_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain',
						min_value=1, value=1, step=1, key='classification_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0,
						step=1, key='classification_rfe_verbose' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_rfe_apply',
								use_container_width=True ):
							
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = RFE( k_features=int( k_features ), verbose=int( verbose ) )
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'rfe' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'RFE applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_rfe_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		if df_processed is None:
			st.stop( )
		st.markdown( '##### Processed Data' )
		
		st.caption( f'Samples: {len( df_processed ):,} | Features: {len( df_processed.columns ):,}' )
		st.data_editor( df_processed, key='classification_processed_data' )
			
		# ------------------------------------------------------------------
		# MODEL TRAINING
		# ------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Model Training', help=cfg.CLASSIFICATION_MODELS )
		
		active_features = [ ftr for ftr in st.session_state.get( 'features', [ ] )
		                    if ftr in df_processed.columns ]
		
		active_targets = [ tgt for tgt in st.session_state.get( 'targets', [ ] )
		                   if tgt in df_processed.columns ]
		
		st.session_state[ 'active_features' ] = active_features
		st.session_state[ 'active_targets' ] = active_targets
		if len( active_targets ) != 1:
			st.warning( '⚠️ Classification models requires exactly one processed target column.' )
			st.stop( )
		
		target_name = active_targets[ 0 ]
		
		df_model = df_processed[ active_features + [ target_name ] ].copy( )
		df_model = df_model.dropna( subset=active_features + [ target_name ] ).copy( )
		st.session_state[ 'df_model' ] = df_model.copy( )
		X_data = df_model[ active_features ].copy( )
		st.session_state[ 'X_data' ] = X_data.copy( )
		for col in X_data.columns:
			X_data[ col ] = pd.to_numeric( X_data[ col ], errors='coerce' )
		
		if X_data.isna( ).any( ).any( ):
			st.warning( '⚠️ One or more feature columns are still non-numeric after preprocessing.'
				'Apply the appropriate encoder/transformer before training.' )
			st.stop( )
		
		# Create training matrix and target vector
		X = X_data.to_numpy( dtype=float )
		y_series = df_model[ target_name ].copy( )
		st.session_state[ 'y_series' ] = y_series.copy( )
		if isinstance( y_series, pd.DataFrame ):
			st.warning( '⚠️ The processed target must resolve to a single column.' )
			st.stop( )
		
		y = y_series.to_numpy( )
		if y.ndim != 1:
			y = np.ravel( y )
		
		if len( y ) != len( X ):
			st.warning( '⚠️ Feature and target row counts do not match.' )
			st.stop( )
		
		if pd.api.types.is_numeric_dtype( y_series ):
			y_num = pd.to_numeric( y_series, errors='coerce' ).dropna( )
			if len( y_num ) == 0:
				st.warning( '⚠️ The processed target contains no valid values.' )
				st.stop( )
			
			unique_count = int( y_num.nunique( ) )
			unique_ratio = float( unique_count / max( 1, len( y_num ) ) )
			
			if unique_count > 20 and unique_ratio > 0.20:
				st.warning( '⚠️ The processed target appears continuous. '
					'Use a label encoder/binarizer for classification targets, not a scaler.' )
				
				st.stop( )
		
		class_counts = pd.Series( y ).value_counts( dropna=False )
		if len( class_counts ) < 2:
			st.warning( '⚠️ Classification requires at least two classes.' )
			st.stop( )
		
		df_classifiction = df_model.copy( )
		st.session_state[ 'df_classification' ] = df_classification.copy( )
	
		# ------------------------------------------------------------------
		# Classification Models
		# ------------------------------------------------------------------
		with st.expander( 'Linear Models', expanded=True ):
			
			with st.expander( 'Perceptron', expanded=True ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.PERCEPTRON_CLASSIFIER )
				
				perceptron_defaults = {
						'classification_perceptron_alpha': 0.001000,
						'classification_perceptron_eta': 1.000000,
						'classification_perceptron_iters': 1000,
						'classification_perceptron_shuffle': False,
						'classification_perceptron_penalty': None,
						'classification_perceptron_test_size': 20,
						'classification_perceptron_random_state': 1
				}
				
				for key, value in perceptron_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				per_c1, per_c2, per_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with per_c1:
					st.markdown( '###### Model Parameters' )
					perceptron_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_perceptron_alpha' ] ),
						step=0.000100, format='%.6f', key='classification_perceptron_alpha' )
					
					perceptron_eta = st.number_input( 'Eta', min_value=0.000001,
						value=float( st.session_state[ 'classification_perceptron_eta' ] ),
						step=0.100000, format='%.6f',
						key='classification_perceptron_eta' )
					
					perceptron_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_perceptron_iters' ] ),
						step=1, key='classification_perceptron_iters' )
				
				with per_c2:
					st.markdown( '###### Regularization / Split' )
					perceptron_shuffle = st.checkbox( 'Shuffle',
						value=bool( st.session_state[ 'classification_perceptron_shuffle' ] ),
						key='classification_perceptron_shuffle' )
					
					perceptron_penalty = st.selectbox( 'Penalty',
						options=[ None, 'l2', 'l1', 'elasticnet' ],
						index=[ None, 'l2', 'l1', 'elasticnet' ].index(
							st.session_state[ 'classification_perceptron_penalty' ] ),
						format_func=lambda v: 'None' if v is None else str( v ),
						key='classification_perceptron_penalty' )
					
					perceptron_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_perceptron_test_size' ] ),
						step=1, key='classification_perceptron_test_size' ) / 100.0
				
				with per_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					perceptron_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_perceptron_random_state' ] ),
						step=1, key='classification_perceptron_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				per_btn_1, per_btn_2 = st.columns( 2 )
				with per_btn_1:
					train_perceptron = st.button( '🚆 Train Perceptron',
						key='classification_perceptron_train', use_container_width=True )
				
				with per_btn_2:
					reset_perceptron = st.button( '🔁 Reset Perceptron',
						key='classification_perceptron_reset', use_container_width=True )
				
				if reset_perceptron:
					for key, value in perceptron_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_perceptron:
					try:
						start_time = time.perf_counter( )
						model = classification_model.Perceptron( alpha=float( perceptron_alpha ),
							eta=float( perceptron_eta ), iters=int( perceptron_iters ),
							shuffle=bool( perceptron_shuffle ), penalty=perceptron_penalty,
							random=int( perceptron_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=perceptron_test_size, random=int( perceptron_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Perceptron training failed: {ex}' )
					
			with st.expander( 'Ordinary Least Squares', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.LEAST_SQUARES_CLASSIFIER )
				
				least_squares_defaults = {
						'classification_least_squares_alpha': 0.000100,
						'classification_least_squares_eta': 0.010000,
						'classification_least_squares_iters': 1000,
						'classification_least_squares_shuffle': False,
						'classification_least_squares_penalty': 'l2',
						'classification_least_squares_test_size': 20,
						'classification_least_squares_random_state': 42
				}
				
				for key, value in least_squares_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				ls_c1, ls_c2, ls_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ls_c1:
					st.markdown( '###### Model Parameters' )
					least_squares_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_least_squares_alpha' ] ),
						step=0.000100, format='%.6f', key='classification_leastsquares_alpha' )
					
					least_squares_eta = st.number_input( 'Eta', min_value=0.000001,
						value=float( st.session_state[ 'classification_least_squares_eta' ] ),
						step=0.010000, format='%.6f', key='classification_leastsquares_eta' )
					
					least_squares_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_least_squares_iters' ] ),
						step=1, key='classification_leastsquares_iters' )
				
				with ls_c2:
					st.markdown( '###### Regularization / Split' )
					least_squares_shuffle = st.checkbox( 'Shuffle',
						value=bool( st.session_state[ 'classification_least_squares_shuffle' ] ),
						key='classification_leastsquares_shuffle' )
					
					least_squares_penalty = st.selectbox( 'Penalty',
						options=[ None, 'l2', 'l1', 'elasticnet' ],
						index=[ None, 'l2', 'l1', 'elasticnet' ].index(
							st.session_state[ 'classification_least_squares_penalty' ] ),
						format_func=lambda v: 'None' if v is None else str( v ),
						key='classification_leastsquares_penalty' )
					
					least_squares_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30, step=1,
						value=int( st.session_state[ 'classification_least_squares_test_size' ] ),
						key='classification_leastsquares_test_size' ) / 100.0
				
				with ls_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					least_squares_random_state = st.number_input( 'Random State', step=1,
						value=int( st.session_state[ 'classification_least_squares_random_state' ]),
						key='classification_leastsquares_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				ls_btn_1, ls_btn_2 = st.columns( 2 )
				with ls_btn_1:
					train_least_squares = st.button( '🚆 Train Least Squares',
						key='classification_leastsquares_train', use_container_width=True )
				
				with ls_btn_2:
					reset_least_squares = st.button( '🔁 Reset Least Squares',
						key='classification_leastsquares_reset', use_container_width=True )
				
				if reset_least_squares:
					for key, value in least_squares_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_least_squares:
					try:
						start_time = time.perf_counter( )
						model = classification_model.LeastSquares( alpha=float( least_squares_alpha ),
							eta=float( least_squares_eta ), iters=int( least_squares_iters ),
							shuffle=bool( least_squares_shuffle ), penalty=least_squares_penalty,
							random=int( least_squares_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( least_squares_test_size ),
							random=int( least_squares_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ),
							'Processing Time (Seconds)', round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ),
							'Testing Rows', int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {'Actual': y_test, 'Predicted': y_prediction })
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Least Squares training failed: {ex}' )
					
			with st.expander( 'Logistic Regression', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.LOGISTIC_REGRESSION )
				
				logistic_defaults = {
						'classification_logistic_c': 1.000000,
						'classification_logistic_penalty': 'l2',
						'classification_logistic_iters': 1000,
						'classification_logistic_multiclass': 'multinomial',
						'classification_logistic_solver': 'lbfgs',
						'classification_logistic_test_size': 20,
						'classification_logistic_random_state': 42
				}
				
				for key, value in logistic_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				log_c1, log_c2, log_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with log_c1:
					st.markdown( '###### Model Parameters' )
					logistic_c = st.number_input( 'C', min_value=0.000001,
						value=float( st.session_state[ 'classification_logistic_c' ] ),
						step=0.100000, format='%.6f', key='classification_logistic_c' )
					
					logistic_penalty = st.selectbox( 'Penalty',
						options=[ 'l2', 'l1', 'elasticnet', 'none', None ],
						index=[ 'l2', 'l1', 'elasticnet', 'none', None ].index(
							st.session_state[ 'classification_logistic_penalty' ]
						), format_func=lambda v: 'None' if v is None else str( v ),
						key='classification_logistic_penalty' )
					
					logistic_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_logistic_iters' ] ),
						step=1, key='classification_logistic_iters' )
				
				with log_c2:
					st.markdown( '###### Strategy / Solver' )
					logistic_multiclass = st.selectbox( 'Multiclass',
						options=[ 'multinomial', 'ovr', 'auto' ],
						index=[ 'multinomial', 'ovr', 'auto' ].index(
							st.session_state[ 'classification_logistic_multiclass' ] ),
						key='classification_logistic_multiclass' )
					
					logistic_solver = st.selectbox( 'Solver',
						options=[ 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga' ],
						index=[ 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag',
						        'saga' ].index(
							st.session_state[ 'classification_logistic_solver' ] ),
						key='classification_logistic_solver' )
					
					logistic_test_size = st.slider( 'Test Set Size (%)',
						min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_logistic_test_size' ] ),
						step=1, key='classification_logistic_test_size' ) / 100.0
				
				with log_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					logistic_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_logistic_random_state' ] ),
						step=1, key='classification_logistic_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
						f'Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				log_btn_1, log_btn_2 = st.columns( 2 )
				with log_btn_1:
					train_logistic = st.button( '🚆 Train Logistic Regression',
						key='classification_logistic_train', use_container_width=True )
				
				with log_btn_2:
					reset_logistic = st.button( '🔁 Reset Logistic Regression',
						key='classification_logistic_reset', use_container_width=True )
				
				if reset_logistic:
					for key, value in logistic_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_logistic:
					try:
						start_time = time.perf_counter( )
						model = classification_model.LogisticRegression( C=float( logistic_c ),
							penalty=logistic_penalty, iters=int( logistic_iters ),
							multiclass=str( logistic_multiclass ), solver=str( logistic_solver ),
							random=int( logistic_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( logistic_test_size ), random=int( logistic_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Logistic Regression training failed: {ex}' )
					
			with st.expander( 'Ridge Classification', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.RIDGE_CLASSIFIER )
				
				ridge_defaults = { 'classification_ridge_alpha': 1.000000,
						'classification_ridge_solver': 'auto', 'classification_ridge_iters': 1000,
						'classification_ridge_test_size': 20, 'classification_ridge_random_state': 42 }
				
				for key, value in ridge_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				ridge_c1, ridge_c2, ridge_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ridge_c1:
					st.markdown( '###### Model Parameters' )
					ridge_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_ridge_alpha' ] ),
						step=0.100000, format='%.6f',
						key='classification_ridge_alpha' )
					
					ridge_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_ridge_iters' ] ), step=1,
						key='classification_ridge_iters' )
				
				with ridge_c2:
					st.markdown( '###### Solver / Split' )
					ridge_solver = st.selectbox( 'Solver',
						options=[ 'auto', 'svd', 'cholesky', 'lsqr',
						          'sparse_cg', 'sag', 'saga', 'lbfgs' ],
						index=[ 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg',
						        'sag', 'saga', 'lbfgs' ].index(
							st.session_state[ 'classification_ridge_solver' ] ),
						key='classification_ridge_solver' )
					
					ridge_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_ridge_test_size' ] ),
						step=1, key='classification_ridge_test_size' ) / 100.0
				
				with ridge_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					ridge_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_ridge_random_state' ] ),
						step=1, key='classification_ridge_random_state' )
					
					st.caption( f'Samples: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				ridge_btn_1, ridge_btn_2 = st.columns( 2 )
				with ridge_btn_1:
					train_ridge = st.button( '🚆 Train Ridge', key='classification_ridge_train',
						use_container_width=True )
				
				with ridge_btn_2:
					reset_ridge = st.button( '🔁 Reset Ridge', key='classification_ridge_reset',
						use_container_width=True )
				
				if reset_ridge:
					for key, value in ridge_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_ridge:
					try:
						start_time = time.perf_counter( )
						model = classification_model.Ridge( alpha=float( ridge_alpha ),
							solver=str( ridge_solver ), iters=int( ridge_iters ),
							rando=int( ridge_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( ridge_test_size ), random=int( ridge_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Ridge training failed: {ex}' )
					
			with st.expander( 'Lasso Classification', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.LASSO_CLASSIFIER )
				
				lasso_defaults = {
						'classification_lasso_alpha': 1.000000,
						'classification_lasso_iters': 500,
						'classification_lasso_threshold': 0.500000,
						'classification_lasso_selection': 'random',
						'classification_lasso_test_size': 20,
						'classification_lasso_random_state': 42
				}
				
				for key, value in lasso_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				lasso_c1, lasso_c2, lasso_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with lasso_c1:
					st.markdown( '###### Model Parameters' )
					lasso_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_lasso_alpha' ] ),
						step=0.100000, format='%.6f',
						key='classification_lasso_alpha' )
					
					lasso_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_lasso_iters' ] ),
						step=1, key='classification_lasso_iters' )
					
					lasso_threshold = st.number_input( 'Threshold', min_value=0.000000,
						max_value=1.000000,
						value=float( st.session_state[ 'classification_lasso_threshold' ] ),
						step=0.050000, format='%.6f',
						key='classification_lasso_threshold' )
				
				with lasso_c2:
					st.markdown( '###### Selection / Split' )
					lasso_selection = st.selectbox( 'Selection',
						options=[ 'cyclic', 'random' ],
						index=[ 'cyclic', 'random' ].index(
							st.session_state[ 'classification_lasso_selection' ] ),
						key='classification_lasso_selection' )
					
					lasso_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_lasso_test_size' ] ),
						step=1, key='classification_lasso_test_size' ) / 100.0
				
				with lasso_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					lasso_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_lasso_random_state' ] ),
						step=1, key='classification_lasso_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				lasso_btn_1, lasso_btn_2 = st.columns( 2 )
				with lasso_btn_1:
					train_lasso = st.button( '🚆 Train Lasso', key='classification_lasso_train',
						use_container_width=True )
				
				with lasso_btn_2:
					reset_lasso = st.button( '🔁 Reset Lasso', key='classification_lasso_reset',
						use_container_width=True )
				
				if reset_lasso:
					for key, value in lasso_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_lasso:
					try:
						start_time = time.perf_counter( )
						
						model = classification_model.Lasso( alpha=float( lasso_alpha ),
							iters=int( lasso_iters ), rando=int( lasso_random_state ),
							threshold=float( lasso_threshold ), selection=str( lasso_selection ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( lasso_test_size ), random=int( lasso_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.score( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Lasso training failed: {ex}' )
					
			with st.expander( 'Gradient Descent', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.GRADIENT_DESCENT_CLASSIFIER )
				
				gradient_defaults = {
						'classification_gradient_loss': 'hinge',
						'classification_gradient_penalty': 'l2',
						'classification_gradient_alpha': 0.000100,
						'classification_gradient_iters': 1000,
						'classification_gradient_shuffle': True,
						'classification_gradient_eta': 0.010000,
						'classification_gradient_learning': 'optimal',
						'classification_gradient_power': 0.500000,
						'classification_gradient_epsilon': 0.100000,
						'classification_gradient_test_size': 20,
						'classification_gradient_random_state': 42
				}
				
				for key, value in gradient_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				gd_c1, gd_c2, gd_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with gd_c1:
					st.markdown( '###### Model Parameters' )
					gradient_loss = st.selectbox( 'Loss',
						options=[
								'hinge',
								'log_loss',
								'modified_huber',
								'squared_hinge',
								'perceptron',
								'huber',
								'epsilon_insensitive',
								'squared_error'
						],
						index=[
								'hinge',
								'log_loss',
								'modified_huber',
								'squared_hinge',
								'perceptron',
								'huber',
								'epsilon_insensitive',
								'squared_error'
						].index( st.session_state[ 'classification_gradient_loss' ] ),
						key='classification_gradient_loss' )
					
					gradient_penalty = st.selectbox( 'Penalty',
						options=[ None, 'l2', 'l1', 'elasticnet' ],
						index=[ None, 'l2', 'l1', 'elasticnet' ].index(
							st.session_state[ 'classification_gradient_penalty' ] ),
						format_func=lambda v: 'None' if v is None else str( v ),
						key='classification_gradient_penalty' )
					
					gradient_alpha = st.number_input( 'Alpha',
						min_value=0.000001,
						value=float( st.session_state[ 'classification_gradient_alpha' ] ),
						step=0.000100, format='%.6f',
						key='classification_gradient_alpha' )
					
					gradient_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_gradient_iters' ] ),
						step=1, key='classification_gradient_iters' )
				
				with gd_c2:
					st.markdown( '###### Learning Controls' )
					gradient_shuffle = st.checkbox( 'Shuffle',
						value=bool( st.session_state[ 'classification_gradient_shuffle' ] ),
						key='classification_gradient_shuffle' )
					
					gradient_eta = st.number_input( 'Eta', min_value=0.000001,
						value=float( st.session_state[ 'classification_gradient_eta' ] ),
						step=0.010000, format='%.6f', key='classification_gradient_eta' )
					
					gradient_learning = st.selectbox( 'Learning Rate Schedule',
						options=[ 'constant', 'optimal', 'invscaling', 'adaptive' ],
						index=[ 'constant', 'optimal', 'invscaling', 'adaptive' ].index(
							st.session_state[ 'classification_gradient_learning' ] ),
						key='classification_gradient_learning' )
					
					gradient_power = st.number_input( 'Power T', min_value=0.000000,
						value=float( st.session_state[ 'classification_gradient_power' ] ),
						step=0.100000, format='%.6f',
						key='classification_gradient_power' )
					
					gradient_epsilon = st.number_input( 'Epsilon', min_value=0.000000,
						value=float( st.session_state[ 'classification_gradient_epsilon' ] ),
						step=0.010000, format='%.6f',
						key='classification_gradient_epsilon' )
				
				with gd_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					gradient_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30,
						value=int( st.session_state[ 'classification_gradient_test_size' ] ),
						step=1, key='classification_gradient_test_size' ) / 100.0
					
					gradient_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_gradient_random_state' ] ),
						step=1, key='classification_gradient_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				gd_btn_1, gd_btn_2 = st.columns( 2 )
				with gd_btn_1:
					train_gradient = st.button( '🚆 Train Gradient Descent',
						key='classification_gradient_train', use_container_width=True )
				
				with gd_btn_2:
					reset_gradient = st.button( '🔁 Reset Gradient Descent',
						key='classification_gradient_reset', use_container_width=True )
				
				if reset_gradient:
					for key, value in gradient_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_gradient:
					try:
						start_time = time.perf_counter( )
						model = classification_model.GradientDescent( loss=str( gradient_loss ),
							penalty=gradient_penalty, alpha=float( gradient_alpha ),
							iters=int( gradient_iters ), shuffle=bool( gradient_shuffle ),
							eta=float( gradient_eta ), learning=str( gradient_learning ),
							power=float( gradient_power ), epsilon=float( gradient_epsilon ),
							rando=int( gradient_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( gradient_test_size ), random=int( gradient_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.score( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Gradient Descent training failed: {ex}' )
		
		with st.expander( 'Instance Models', expanded=True ):
			
			with st.expander( 'Nearest Neighbor', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.NEAREST_NEIGHBOR_CLASSFIER )
				
				nearest_defaults = {
						'classification_nearest_num': 5,
						'classification_nearest_algorithm': 'auto',
						'classification_nearest_power': 2,
						'classification_nearest_metric': 'minkowski',
						'classification_nearest_leafs': 30,
						'classification_nearest_test_size': 20,
						'classification_nearest_random_state': 42
				}
				
				for key, value in nearest_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				nn_c1, nn_c2, nn_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with nn_c1:
					st.markdown( '###### Model Parameters' )
					nearest_num = st.number_input( 'Neighbors', min_value=1,
						value=int( st.session_state[ 'classification_nearest_num' ] ),
						step=1, key='classification_nearest_num' )
					
					nearest_power = st.number_input( 'Power', min_value=1,
						value=int( st.session_state[ 'classification_nearest_power' ] ),
						step=1, key='classification_nearest_power' )
					
					nearest_leafs = st.number_input( 'Leaf Size', min_value=1,
						value=int( st.session_state[ 'classification_nearest_leafs' ] ),
						step=1, key='classification_nearest_leafs' )
				
				with nn_c2:
					st.markdown( '###### Distance / Search' )
					nearest_algorithm = st.selectbox( 'Algorithm',
						options=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ],
						index=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ].index(
							st.session_state[ 'classification_nearest_algorithm' ]
						), key='classification_nearest_algorithm' )
					
					nearest_metric = st.selectbox( 'Metric',
						options=[
								'minkowski',
								'euclidean',
								'manhattan',
								'chebyshev',
								'hamming',
								'canberra',
								'braycurtis',
								'cityblock',
								'cosine',
								'l1',
								'l2',
								'nan_euclidean',
								'mahalanobis',
								'seuclidean'
						],
						index=[
								'minkowski',
								'euclidean',
								'manhattan',
								'chebyshev',
								'hamming',
								'canberra',
								'braycurtis',
								'cityblock',
								'cosine',
								'l1',
								'l2',
								'nan_euclidean',
								'mahalanobis',
								'seuclidean'
						].index( st.session_state[ 'classification_nearest_metric' ] ),
						key='classification_nearest_metric' )
					
					nearest_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_nearest_test_size' ] ),
						step=1, key='classification_nearest_test_size' ) / 100.0
				
				with nn_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					nearest_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_nearest_random_state' ] ),
						step=1, key='classification_nearest_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				nn_btn_1, nn_btn_2 = st.columns( 2, border=True )
				with nn_btn_1:
					train_nearest = st.button( '🚆 Train Nearest Neighbor',
						key='classification_nearest_train', use_container_width=True )
				
				with nn_btn_2:
					reset_nearest = st.button( '🔁 Reset Nearest Neighbor',
						key='classification_nearest_reset', use_container_width=True )
				
				if reset_nearest:
					for key, value in nearest_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_nearest:
					try:
						start_time = time.perf_counter( )
						model = classification_model.NearestNeighbor( num=int( nearest_num ),
							algorithm=str( nearest_algorithm ), power=int( nearest_power ),
							metric=str( nearest_metric ), leafs=int( nearest_leafs ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( nearest_test_size ), random=int( nearest_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Nearest Neighbor training failed: {ex}' )
			
			with st.expander( 'Support Vector Machine', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.SUPPORT_VECTOR_CLASSIFIER )
				
				svm_defaults = {
						'classification_svm_c': 1.000000,
						'classification_svm_kernel': 'rbf',
						'classification_svm_degree': 3,
						'classification_svm_test_size': 20,
						'classification_svm_random_state': 42
				}
				
				for key, value in svm_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				svm_c1, svm_c2, svm_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with svm_c1:
					st.markdown( '###### SVM Parameters' )
					svm_c = st.number_input( 'C', min_value=0.000001,
						value=float( st.session_state[ 'classification_svm_c' ] ),
						step=0.100000, format='%.6f', key='classification_svm_c' )
					
					svm_kernel = st.selectbox( 'Kernel',
						options=[ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ],
						index=[ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ].index(
							st.session_state[ 'classification_svm_kernel' ] ),
						key='classification_svm_kernel' )
					
					svm_degree = st.number_input( 'Degree', min_value=1,
						value=int( st.session_state[ 'classification_svm_degree' ] ),
						step=1, key='classification_svm_degree' )
				
				with svm_c2:
					st.markdown( '###### Split' )
					svm_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30,
						value=int( st.session_state[ 'classification_svm_test_size' ] ),
						step=1, key='classification_svm_test_size' ) / 100.0
					
					if svm_kernel != 'poly':
						st.caption( 'Degree is only used when kernel = poly.' )
				
				with svm_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					svm_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_svm_random_state' ] ),
						step=1, key='classification_svm_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				svm_btn_1, svm_btn_2 = st.columns( 2 )
				with svm_btn_1:
					train_svm = st.button( '🚆 Train Support Vector',
						key='classification_svm_train', use_container_width=True )
				
				with svm_btn_2:
					reset_svm = st.button( '🔁 Reset Support Vector',
						key='classification_svm_reset', use_container_width=True )
				
				if reset_svm:
					for key, value in svm_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_svm:
					try:
						start_time = time.perf_counter( )
						model = classification_model.SupportVector( C=float( svm_c ),
							kernel=str( svm_kernel ), degree=int( svm_degree ),
							random=int( svm_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( svm_test_size ), random=int( svm_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Support Vector training failed: {ex}' )
		
		with st.expander( 'Tree Models', expanded=True ):
			
			with st.expander( 'Decision Tree', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.DESICION_TREE_CLASSIFIER )
				
				tree_defaults = {
						'classification_tree_criterion': 'gini',
						'classification_tree_splitter': 'best',
						'classification_tree_depth': 0,
						'classification_tree_min_split': 2,
						'classification_tree_min_leaf': 1,
						'classification_tree_test_size': 20,
						'classification_tree_random_state': 42
				}
				
				for key, value in tree_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				tree_c1, tree_c2, tree_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with tree_c1:
					st.markdown( '###### Hyper-Parameters' )
					tree_criterion = st.selectbox(
						'Criterion',
						options=[ 'gini', 'entropy', 'log_loss' ],
						index=[ 'gini', 'entropy', 'log_loss' ].index(
							st.session_state[ 'classification_tree_criterion' ] ),
						key='classification_tree_criterion' )
					
					tree_splitter = st.selectbox( 'Splitter', options=[ 'best', 'random' ],
						index=[ 'best', 'random' ].index(
							st.session_state[ 'classification_tree_splitter' ] ),
						key='classification_tree_splitter' )
					
					tree_depth = st.number_input( 'Max Depth (0 = None)', min_value=0,
						value=int( st.session_state[ 'classification_tree_depth' ] ),
						step=1, key='classification_tree_depth' )
				
				with tree_c2:
					st.markdown( '###### Node Constraints' )
					tree_min_split = st.number_input( 'Min Samples Split', min_value=2,
						value=int( st.session_state[ 'classification_tree_min_split' ] ),
						step=1, key='classification_tree_min_split' )
					
					tree_min_leaf = st.number_input( 'Min Samples Leaf', min_value=1,
						value=int( st.session_state[ 'classification_tree_min_leaf' ] ),
						step=1, key='classification_tree_min_leaf' )
					
					tree_test_size = st.slider( 'Test Set Size (%)',
						min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_tree_test_size' ] ),
						step=1, key='classification_tree_test_size' ) / 100.0
				
				with tree_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					tree_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_tree_random_state' ] ),
						step=1, key='classification_tree_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				tree_btn_1, tree_btn_2 = st.columns( 2 )
				with tree_btn_1:
					train_tree = st.button( '🚆 Train Decision Tree',
						key='classification_tree_train', use_container_width=True )
				
				with tree_btn_2:
					reset_tree = st.button( '🔁 Reset Decision Tree',
						key='classification_tree_reset', use_container_width=True )
				
				if reset_tree:
					for key, value in tree_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_tree:
					try:
						start_time = time.perf_counter( )
						model = classification_model.DecisionTree( criterion=str( tree_criterion ),
							splitter=str( tree_splitter ),
							depth=None if int( tree_depth ) == 0 else int( tree_depth ),
							min_split=int( tree_min_split ), min_leaf=int( tree_min_leaf ),
							random=int( tree_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( tree_test_size ), random=int( tree_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
								'Actual': y_test,
								'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Decision Tree training failed: {ex}' )
				
			with st.expander( 'Random Forest', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.RANDOM_FOREST_CLASSIFIER )
				
				forest_defaults = {
						'classification_forest_estimators': 100,
						'classification_forest_criterion': 'gini',
						'classification_forest_depth': 0,
						'classification_forest_min_split': 2,
						'classification_forest_min_leaf': 1,
						'classification_forest_test_size': 20,
						'classification_forest_random_state': 42
				}
				
				for key, value in forest_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				forest_c1, forest_c2, forest_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with forest_c1:
					st.markdown( '###### Hyper-Parameters' )
					forest_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_forest_estimators' ] ),
						step=1, key='classification_forest_estimators' )
					
					forest_criterion = st.selectbox( 'Criterion',
						options=[ 'gini', 'entropy', 'log_loss' ],
						index=[ 'gini', 'entropy', 'log_loss' ].index(
							st.session_state[ 'classification_forest_criterion' ] ),
						key='classification_forest_criterion' )
					
					forest_depth = st.number_input( 'Max Depth (0 = None)', min_value=0,
						value=int( st.session_state[ 'classification_forest_depth' ] ),
						step=1, key='classification_forest_depth' )
				
				with forest_c2:
					st.markdown( '###### Node Constraints' )
					forest_min_split = st.number_input( 'Min Samples Split', min_value=2,
						value=int( st.session_state[ 'classification_forest_min_split' ] ),
						step=1, key='classification_forest_min_split' )
					
					forest_min_leaf = st.number_input( 'Min Samples Leaf', min_value=1,
						value=int( st.session_state[ 'classification_forest_min_leaf' ] ),
						step=1, key='classification_forest_min_leaf' )
					
					forest_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30,
						value=int( st.session_state[ 'classification_forest_test_size' ] ),
						step=1, key='classification_forest_test_size' ) / 100.0
				
				with forest_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					forest_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_forest_random_state' ] ),
						step=1, key='classification_forest_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				forest_btn_1, forest_btn_2 = st.columns( 2 )
				with forest_btn_1:
					train_forest = st.button( '🚆 Train Random Forest',
						key='classification_forest_train', use_container_width=True )
				
				with forest_btn_2:
					reset_forest = st.button( '🔁 Reset Random Forest',
						key='classification_forest_reset', use_container_width=True )
				
				if reset_forest:
					for key, value in forest_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_forest:
					try:
						start_time = time.perf_counter( )
						model = classification_model.RandomForest(
							n_estimators=int( forest_estimators ), criterion=str( forest_criterion ),
							depth=None if int( forest_depth ) == 0 else int( forest_depth ),
							min_split=int( forest_min_split ), min_leaf=int( forest_min_leaf ),
							random=int( forest_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( forest_test_size ), random=int( forest_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ),
							'Processing Time (Seconds)', round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
								'Actual': y_test,
								'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Random Forest training failed: {ex}' )
		
		with st.expander( 'Ensemble Models', expanded=True ):
			
			with st.expander( 'Gradient Boost', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.GRADIENT_BOOST_CLASSIFIER )
				
				gb_defaults = {
						'classification_gb_estimators': 100,
						'classification_gb_rate': 0.100000,
						'classification_gb_depth': 3,
						'classification_gb_criterion': 'friedman_mse',
						'classification_gb_test_size': 20,
						'classification_gb_random_state': 42
				}
				
				for key, value in gb_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				gb_c1, gb_c2, gb_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with gb_c1:
					st.markdown( '###### Hyper-Parameters' )
					gb_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_gb_estimators' ] ),
						step=1, key='classification_gb_estimators' )
					
					gb_rate = st.number_input( 'Learning Rate', min_value=0.000001,
						value=float( st.session_state[ 'classification_gb_rate' ] ),
						step=0.010000, format='%.6f', key='classification_gb_rate' )
					
					gb_depth = st.number_input( 'Max Depth', min_value=1,
						value=int( st.session_state[ 'classification_gb_depth' ] ), step=1,
						key='classification_gb_depth' )
				
				with gb_c2:
					st.markdown( '###### Criterion / Split' )
					gb_criterion = st.selectbox( 'Criterion',
						options=[ 'friedman_mse', 'squared_error' ],
						index=[ 'friedman_mse', 'squared_error' ].index(
							st.session_state[ 'classification_gb_criterion' ] ),
						key='classification_gb_criterion' )
					
					gb_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_gb_test_size' ] ), step=1,
						key='classification_gb_test_size' ) / 100.0
				
				with gb_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					gb_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_gb_random_state' ] ),
						step=1, key='classification_gb_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				gb_btn_1, gb_btn_2 = st.columns( 2 )
				with gb_btn_1:
					train_gb = st.button( '🚆 Train Gradient Boost', key='classification_gb_train',
						use_container_width=True )
				
				with gb_btn_2:
					reset_gb = st.button( '🔁 Reset Gradient Boost', key='classification_gb_reset',
						use_container_width=True )
				
				if reset_gb:
					for key, value in gb_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_gb:
					try:
						start_time = time.perf_counter( )
						model = classification_model.GradientBoost( estimators=int( gb_estimators ),
							rate=float( gb_rate ), depth=int( gb_depth ),
							criterion=str( gb_criterion ), random=int( gb_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( gb_test_size ), random=int( gb_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Gradient Boost training failed: {ex}' )
					
			with st.expander( 'Adaptive Boost', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.ADAPTIVE_BOOST_CLASSIFIER )
				
				ab_defaults = {
						'classification_ab_estimators': 50,
						'classification_ab_rate': 1.000000,
						'classification_ab_algorithm': 'SAMME',
						'classification_ab_test_size': 20,
						'classification_ab_random_state': 42
				}
				
				for key, value in ab_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				ab_c1, ab_c2, ab_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ab_c1:
					st.markdown( '###### Hyper-Parameters' )
					ab_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_ab_estimators' ] ),
						step=1, key='classification_ab_estimators' )
					
					ab_rate = st.number_input( 'Learning Rate', min_value=0.000001,
						value=float( st.session_state[ 'classification_ab_rate' ] ),
						step=0.010000, format='%.6f', key='classification_ab_rate' )
				
				with ab_c2:
					st.markdown( '###### Algorithm / Split' )
					ab_algorithm = st.selectbox( 'Algorithm', options=[ 'SAMME', 'deprecated', None ],
						index=[ 'SAMME', 'deprecated', None ].index(
							st.session_state[ 'classification_ab_algorithm' ] ),
						format_func=lambda v: 'None' if v is None else str( v ),
						key='classification_ab_algorithm' )
					
					ab_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_ab_test_size' ] ),
						step=1, key='classification_ab_test_size' ) / 100.0
				
				with ab_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					ab_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_ab_random_state' ] ),
						step=1, key='classification_ab_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
						f'Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				ab_btn_1, ab_btn_2 = st.columns( 2 )
				with ab_btn_1:
					train_ab = st.button( '🚆 Train Adaptive Boost', key='classification_ab_train',
						use_container_width=True )
				
				with ab_btn_2:
					reset_ab = st.button( '🔁 Reset Adaptive Boost', key='classification_ab_reset',
						use_container_width=True )
				
				if reset_ab:
					for key, value in ab_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_ab:
					try:
						start_time = time.perf_counter( )
						model = classification_model.AdaptiveBoost( base=None,
							estimators=int( ab_estimators ), rate=float( ab_rate ),
							algorithm=ab_algorithm, random=int( ab_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( ab_test_size ), random=int( ab_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Adaptive Boost training failed: {ex}' )
					
			with st.expander( 'Bagging Model', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.BAGGING_CLASSIFIER )
				
				bag_defaults = {
						'classification_bag_estimators': 50,
						'classification_bag_test_size': 20,
						'classification_bag_random_state': 42
				}
				
				for key, value in bag_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				bag_c1, bag_c2, bag_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with bag_c1:
					st.markdown( '###### Hyper-Parameters' )
					bag_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_bag_estimators' ] ),
						step=1, key='classification_bag_estimators' )
				
				with bag_c2:
					st.markdown( '###### Split' )
					bag_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_bag_test_size' ] ),
						step=1, key='classification_bag_test_size' ) / 100.0
				
				with bag_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					bag_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_bag_random_state' ] ),
						step=1, key='classification_bag_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
						f'Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				bag_btn_1, bag_btn_2 = st.columns( 2 )
				with bag_btn_1:
					train_bag = st.button( '🚆 Train Bagging Model', key='classification_bag_train',
						use_container_width=True )
				
				with bag_btn_2:
					reset_bag = st.button( '🔁 Reset Bagging Model', key='classification_bag_reset',
						use_container_width=True )
				
				if reset_bag:
					for key, value in bag_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_bag:
					try:
						start_time = time.perf_counter( )
						
						model = classification_model.BaggingModel( estimators=int( bag_estimators ),
							random=int( bag_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( bag_test_size ), random=int( bag_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Bagging Model training failed: {ex}' )
					
			with st.expander( 'Voting Model', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.VOTING_CLASSFIER )
				
				vote_defaults = {
						'classification_vote_mode': 'hard',
						'classification_vote_include_logistic': True,
						'classification_vote_include_tree': True,
						'classification_vote_include_knn': True,
						'classification_vote_include_forest': False,
						'classification_vote_include_nb': False,
						'classification_vote_test_size': 20,
						'classification_vote_random_state': 42
				}
				
				for key, value in vote_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				vote_c1, vote_c2, vote_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with vote_c1:
					st.markdown( '###### Voting Strategy' )
					vote_mode = st.selectbox( 'Vote', options=[ 'hard', 'soft' ],
						index=[ 'hard', 'soft' ].index(
							st.session_state[ 'classification_vote_mode' ] ),
						key='classification_vote_mode' )
					
					st.caption( 'Select at least two base estimators.' )
				
				with vote_c2:
					st.markdown( '###### Base Estimators' )
					vote_include_logistic = st.checkbox( 'Logistic Regression',
						value=bool( st.session_state[ 'classification_vote_include_logistic' ] ),
						key='classification_vote_include_logistic' )
					
					vote_include_tree = st.checkbox( 'Decision Tree',
						value=bool( st.session_state[ 'classification_vote_include_tree' ] ),
						key='classification_vote_include_tree' )
					
					vote_include_knn = st.checkbox( 'k-Nearest Neighbors',
						value=bool( st.session_state[ 'classification_vote_include_knn' ] ),
						key='classification_vote_include_knn' )
					
					vote_include_forest = st.checkbox( 'Random Forest',
						value=bool( st.session_state[ 'classification_vote_include_forest' ] ),
						key='classification_vote_include_forest' )
					
					vote_include_nb = st.checkbox( 'Gaussian Naive Bayes',
						value=bool( st.session_state[ 'classification_vote_include_nb' ] ),
						key='classification_vote_include_nb' )
				
				with vote_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					vote_test_size = st.slider(
						'Test Set Size (%)',
						min_value=10,
						max_value=30,
						value=int( st.session_state[ 'classification_vote_test_size' ] ),
						step=1,
						key='classification_vote_test_size'
					) / 100.0
					
					vote_random_state = st.number_input(
						'Random State',
						value=int( st.session_state[ 'classification_vote_random_state' ] ),
						step=1,
						key='classification_vote_random_state'
					)
					
					st.caption(
						f'Rows: {len( df_model ):,} | '
						f'Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}'
					)
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				vote_btn_1, vote_btn_2 = st.columns( 2 )
				with vote_btn_1:
					train_vote = st.button( '🚆 Train Voting Model', key='classification_vote_train',
						use_container_width=True )
				
				with vote_btn_2:
					reset_vote = st.button( '🔁 Reset Voting Model', key='classification_vote_reset',
						use_container_width=True )
				
				if reset_vote:
					for key, value in vote_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_vote:
					try:
						estimators = [ ]
						if vote_include_logistic:
							estimators.append( 'logistic', LogisticRegression( max_iter=1000,
											random_state=int( vote_random_state ) ) )
						
						if vote_include_tree:
							estimators.append( ( 'tree', DecisionTreeClassifier(
											random_state=int( vote_random_state ) ) ) )
						
						if vote_include_knn:
							estimators.append( ( 'knn', KNeighborsClassifier( ) ) )
						
						if vote_include_forest:
							estimators.append( ( 'forest', RandomForestClassifier(
											random_state=int( vote_random_state ) ) ) )
						
						if vote_include_nb:
							estimators.append( ( 'naive_bayes', GaussianNB( ) ) )
						
						if len( estimators ) < 2:
							st.warning( '⚠️ Voting Model requires at least two base estimators.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.VotingModel( estimators=estimators,
							vote=str( vote_mode ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( vote_test_size ), random=int( vote_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.score( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Estimator Count',
							int( len( estimators ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Vote Mode',
							str( vote_mode ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Voting Model training failed: {ex}' )
					
			with st.expander( 'Stacking Model', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.STACKING_CLASSIFIER )
				
				stack_defaults = {
						'classification_stack_include_logistic': True,
						'classification_stack_include_tree': True,
						'classification_stack_include_knn': True,
						'classification_stack_include_forest': False,
						'classification_stack_include_nb': False,
						'classification_stack_final': 'logistic',
						'classification_stack_test_size': 20,
						'classification_stack_random_state': 42
				}
				
				for key, value in stack_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				stack_c1, stack_c2, stack_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with stack_c1:
					st.markdown( '###### Final Estimator' )
					stack_final = st.selectbox( 'Final Estimator', options=[ 'logistic', 'tree' ],
						index=[ 'logistic', 'tree' ].index(
							st.session_state[ 'classification_stack_final' ] ),
						key='classification_stack_final' )
					
					st.caption( 'Select at least two base estimators.' )
				
				with stack_c2:
					st.markdown( '###### Base Estimators' )
					stack_include_logistic = st.checkbox( 'Logistic Regression',
						value=bool( st.session_state[ 'classification_stack_include_logistic' ] ),
						key='classification_stack_include_logistic' )
					
					stack_include_tree = st.checkbox( 'Decision Tree',
						value=bool( st.session_state[ 'classification_stack_include_tree' ] ),
						key='classification_stack_include_tree' )
					
					stack_include_knn = st.checkbox( 'k-Nearest Neighbors',
						value=bool( st.session_state[ 'classification_stack_include_knn' ] ),
						key='classification_stack_include_knn' )
					
					stack_include_forest = st.checkbox( 'Random Forest',
						value=bool( st.session_state[ 'classification_stack_include_forest' ] ),
						key='classification_stack_include_forest' )
					
					stack_include_nb = st.checkbox( 'Gaussian Naive Bayes',
						value=bool( st.session_state[ 'classification_stack_include_nb' ] ),
						key='classification_stack_include_nb' )
				
				with stack_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					stack_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_stack_test_size' ] ), step=1,
						key='classification_stack_test_size' ) / 100.0
					
					stack_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_stack_random_state' ] ),
						step=1, key='classification_stack_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
						f'Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				stack_btn_1, stack_btn_2 = st.columns( 2 )
				with stack_btn_1:
					train_stack = st.button( '🚆 Train Stacking Model',
						key='classification_stack_train', use_container_width=True )
				
				with stack_btn_2:
					reset_stack = st.button( '🔁 Reset Stacking Model',
						key='classification_stack_reset',
						use_container_width=True )
				
				if reset_stack:
					for key, value in stack_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_model' ] = None
					st.session_state[ 'df_scores' ] = None
					st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_stack:
					try:
						estimators = [ ]
						
						if stack_include_logistic:
							estimators.append(
								( 'logistic', LogisticRegression( max_iter=1000,
											random_state=int( stack_random_state ) ) ) )
						
						if stack_include_tree:
							estimators.append( 'tree', DecisionTreeClassifier(
								random_state=int( stack_random_state ) ) )
						
						if stack_include_knn:
							estimators.append( ( 'knn', KNeighborsClassifier( ) ) )
						
						if stack_include_forest:
							estimators.append( ( 'forest', RandomForestClassifier(
								random_state=int( stack_random_state ) ) ) )
						
						if stack_include_nb:
							estimators.append( 'naive_bayes', GaussianNB( ) )
						
						if len( estimators ) < 2:
							st.warning( '⚠️ Stacking Model requires at least two base estimators.' )
							st.stop( )
						
						if stack_final == 'logistic':
							final_estimator = LogisticRegression( max_iter=1000,
								random_state=int( stack_random_state ) )
						else:
							final_estimator = DecisionTreeClassifier(
								random_state=int( stack_random_state ) )
						
						start_time = time.perf_counter( )
						model = classification_model.StackingModel( est=estimators,
							final=final_estimator )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( stack_test_size ), random=int( stack_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.score( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Estimator Count',
							int( len( estimators ) ) )
						
						df_scores.insert(
							len( df_scores.columns ), 'Final Estimator', str( stack_final ) )
						
						df_predictions = pd.DataFrame( {'Actual': y_test,
							 'Predicted': y_prediction } )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Stacking Model training failed: {ex}' )
					
		with st.expander( 'Neural Models', expanded=True ):
			
			with st.expander( 'Multi-Layer Perceptron', expanded=False ):
				mlp_defaults = {
						'classification_mlp_hidden_1': 100,
						'classification_mlp_hidden_2': 0,
						'classification_mlp_activation': 'logistic',
						'classification_mlp_solver': 'lbfgs',
						'classification_mlp_alpha': 0.000100,
						'classification_mlp_learning': 'constant',
						'classification_mlp_test_size': 20,
						'classification_mlp_random_state': 42
				}
				
				for key, value in mlp_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				mlp_c1, mlp_c2, mlp_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with mlp_c1:
					st.markdown( '###### Network Structure' )
					mlp_hidden_1 = st.number_input( 'Hidden Layer 1', min_value=1,
						value=int( st.session_state[ 'classification_mlp_hidden_1' ] ),
						step=1, key='classification_mlp_hidden_1' )
					
					mlp_hidden_2 = st.number_input( 'Hidden Layer 2 (0 = none)', min_value=0,
						value=int( st.session_state[ 'classification_mlp_hidden_2' ] ),
						step=1, key='classification_mlp_hidden_2' )
					
					mlp_activation = st.selectbox( 'Activation',
						options=[ 'identity', 'logistic', 'tanh', 'relu' ],
						index=[ 'identity', 'logistic', 'tanh', 'relu' ].index(
							st.session_state[ 'classification_mlp_activation' ] ),
						key='classification_mlp_activation' )
				
				with mlp_c2:
					st.markdown( '###### Optimization' )
					mlp_solver = st.selectbox( 'Solver', options=[ 'lbfgs', 'sgd', 'adam' ],
						index=[ 'lbfgs', 'sgd', 'adam' ].index(
							st.session_state[ 'classification_mlp_solver' ] ),
						key='classification_mlp_solver' )
					
					mlp_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_mlp_alpha' ] ),
						step=0.000100, format='%.6f', key='classification_mlp_alpha' )
					
					mlp_learning = st.selectbox( 'Learning Rate Schedule',
						options=[ 'constant', 'invscaling', 'adaptive' ],
						index=[ 'constant', 'invscaling', 'adaptive' ].index(
							st.session_state[ 'classification_mlp_learning' ]
						),
						key='classification_mlp_learning' )
					
					mlp_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_mlp_test_size' ] ),
						step=1, key='classification_mlp_test_size' ) / 100.0
				
				with mlp_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					mlp_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_mlp_random_state' ] ), step=1,
						key='classification_mlp_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				mlp_btn_1, mlp_btn_2 = st.columns( 2 )
				with mlp_btn_1:
					train_mlp = st.button( '🚆 Train Multi-Layer Perceptron',
						key='classification_mlp_train', use_container_width=True )
				
				with mlp_btn_2:
					reset_mlp = st.button( '🔁 Reset Multi-Layer Perceptron',
						key='classification_mlp_reset', use_container_width=True )
				
				if reset_mlp:
					for key, value in mlp_defaults.items( ):
						st.session_state[ key ] = value
						st.session_state[ 'elapsed_seconds' ] = 0.0
						st.session_state[ 'model' ] = None
						st.session_state[ 'y_predictions' ] = None
						st.session_state[ 'X_train' ] = None
						st.session_state[ 'X_test' ] = None
						st.session_state[ 'y_train' ] = None
						st.session_state[ 'y_test' ] = None
						st.session_state[ 'df_model' ] = None
						st.session_state[ 'df_scores' ] = None
						st.session_state[ 'df_predictions' ] = None
					st.rerun( )
				
				if train_mlp:
					try:
						if int( mlp_hidden_2 ) > 0:
							hidden_layers = (int( mlp_hidden_1 ), int( mlp_hidden_2 ))
						else:
							hidden_layers = (int( mlp_hidden_1 ),)
						
						start_time = time.perf_counter( )
						
						model = classification_model.MultiLayerPerceptron( hidden=hidden_layers,
							activation=str( mlp_activation ), solver=str( mlp_solver ),
							alpha=float( mlp_alpha ), learning=str( mlp_learning ),
							rando=int( mlp_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( mlp_test_size ), random=int( mlp_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.score( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Hidden Layers',
							str( hidden_layers ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_predictions' ] = y_predictions.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'MultiLayerPerceptron training failed: {ex}' )
		
		if model is None:
			st.stop( )
		# ------------------------------------------------------------------
		# Performance Metrics & Visualizations
		# ------------------------------------------------------------------
		
		target_count = int( st.session_state.get( 'target_count', 0 ) )
		
		has_metric_frame = ( isinstance( df_scores, pd.DataFrame )
				and not df_scores.empty
				and 'Accuracy Score' in df_scores.columns
				and 'Mis-Classifications' in df_scores.columns )
		
		has_prediction_frame = ( isinstance( df_predictions, pd.DataFrame )
				and not df_predictions.empty
				and 'Actual' in df_predictions.columns
				and 'Predicted' in df_predictions.columns )
		
		has_visual_context = ( model is not None and X_test is not None and y_test is not None
				and y_prediction is not None )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Model Performance' )
		
		if has_metric_frame:
			m1, m2, m3 = st.columns( 3, border=True )
			with m1:
				st.metric( 'Accuracy', f"{float( df_scores.at[ 0, 'Accuracy Score' ] ):0.2f}" )
			
			with m2:
				st.metric( 'Mis-Classifications',
					f"{int( df_scores.at[ 0, 'Mis-Classifications' ] ):,}" )
			
			with m3:
				st.metric( 'Processing Time', f'{elapsed_seconds:0.2f} sec' )
			
			st.data_editor( df_scores, use_container_width=True,
				key='classification_performance_scores' )
		else:
			st.info( 'No classification performance metrics are available yet. '
				'Train a classification model to populate this section.' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Predictions' )
		
		if has_prediction_frame:
			st.data_editor( df_predictions, use_container_width=True,
				key='classification_performance_predictions' )
		else:
			st.info( 'No predictions are available for the current classification result.' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Confusion Matrix', help=cfg.CONFUSION_MATRIX )
		
		if has_visual_context:
			try:
				plt.close( 'all' )
				model.confusion_matrix( X_test, y_test )
				fig_cm = plt.gcf( )
				fig_cm.set_size_inches( 6, 4 )
				
				for ax_cm in fig_cm.axes:
					ax_cm.tick_params( axis='x', labelrotation=45, labelsize=8 )
					ax_cm.tick_params( axis='y', labelsize=8 )
					for label in ax_cm.get_xticklabels( ):
						label.set_ha( 'right' )
				
				fig_cm.tight_layout( )
				st.pyplot( fig_cm, use_container_width=True )
				plt.close( fig_cm )
			except Exception as e:
				st.info( f'Confusion Matrix skipped: {e}' )
				plt.close( 'all' )
		else:
			st.info( 'Confusion Matrix is unavailable until a classification model is trained.' )
		
		# ------------------------------------------------------------------
		# ACTUAL VS PREDICTED CLASS COUNTS
		# ------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Actual vs Predicted Counts' )
		
		if has_visual_context:
			try:
				actual_counts = pd.Series( y_test ).value_counts( ).sort_index( )
				pred_counts = pd.Series( y_prediction ).value_counts( ).sort_index( )
				df_counts = pd.DataFrame( {
							'Actual': actual_counts,
							'Predicted': pred_counts
					} ).fillna( 0 )
				
				fig_counts, ax_counts = plt.subplots( figsize=(6, 4) )
				df_counts.plot( kind='bar', ax=ax_counts )
				ax_counts.set_xlabel( 'Class' )
				ax_counts.set_ylabel( 'Count' )
				ax_counts.set_title( 'Actual vs Predicted Class Counts' )
				ax_counts.grid( axis='y', alpha=0.3 )
				fig_counts.tight_layout( )
				st.pyplot( fig_counts )
				plt.close( fig_counts )
			except Exception as e:
				st.info( f'Actual vs Predicted Counts skipped: {e}' )
				plt.close( 'all' )
		else:
			st.info( 'Actual vs Predicted Counts are unavailable until a model is trained.' )
		
		# ------------------------------------------------------------------
		# PER-CLASS ACCURACY
		# ------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Per-Class Accuracy', help=cfg.PERCLASS_ACCURACY )
		
		if has_visual_context:
			try:
				df_evaluation = pd.DataFrame( {
							'Actual': y_test,
							'Predicted': y_prediction
					} )
				df_evaluation[ 'Correct' ] = (
						df_evaluation[ 'Actual' ] == df_evaluation[ 'Predicted' ] ).astype( int )
				
				df_class_acc = df_evaluation.groupby( 'Actual',
					dropna=False )[ 'Correct' ].mean( ).sort_index( )
				
				fig_acc, ax_acc = plt.subplots( figsize=(6, 4) )
				ax_acc.bar( df_class_acc.index.astype( str ), df_class_acc.values )
				ax_acc.set_xlabel( 'Class' )
				ax_acc.set_ylabel( 'Accuracy' )
				ax_acc.set_ylim( 0.0, 1.05 )
				ax_acc.set_title( 'Per-Class Accuracy' )
				ax_acc.grid( axis='y', alpha=0.3 )
				fig_acc.tight_layout( )
				st.pyplot( fig_acc )
				plt.close( fig_acc )
			except Exception as e:
				st.info( f'Per-Class Accuracy skipped: {e}' )
				plt.close( 'all' )
		else:
			st.info( 'Per-Class Accuracy is unavailable until a model is trained.' )
		
		# ------------------------------------------------------------------
		# PREDICTION CONFIDENCE
		# ------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Prediction Confidence', help=cfg.PREDICTION_CONFIDENCE )
		
		if has_visual_context and hasattr( model, 'predict_probability' ):
			try:
				proba = model.predict_probability( X_test )
				if ( isinstance( proba, np.ndarray ) and proba.ndim == 2
						and proba.shape[ 1 ] > 1 ):
					max_conf = np.max( proba, axis=1 )
					fig_conf, ax_conf = plt.subplots( figsize=(6, 4) )
					ax_conf.hist( max_conf, bins=20 )
					ax_conf.set_xlabel( 'Maximum Predicted Probability' )
					ax_conf.set_ylabel( 'Frequency' )
					ax_conf.set_title( 'Prediction Confidence Distribution' )
					ax_conf.grid( axis='y', alpha=0.3 )
					fig_conf.tight_layout( )
					st.pyplot( fig_conf )
					plt.close( fig_conf )
			except Exception as e:
				st.info( f'Prediction Confidence skipped: {e}' )
				plt.close( 'all' )
		else:
			st.info( 'Prediction Confidence is unavailable until a model is trained.' )
		
		# ------------------------------------------------------------------
		# OBSERVED VS PREDICTED
		# ------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Observed vs Predicted' )
		
		if has_visual_context and target_count == 2 and hasattr( model, 'scatter_plot' ):
			try:
				plt.close( 'all' )
				model.scatter_plot( X_test, y_test )
				st.pyplot( plt.gcf( ) )
				plt.close( 'all' )
			except Exception as e:
				st.info( f'Observed vs Predicted plot skipped: {e}' )
				plt.close( 'all' )
		else:
			st.info( 'Observed vs Predicted is shown only for binary classification targets.' )
		
		# ------------------------------------------------------------------
		# ROC CURVE
		# ------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### ROC Curve', help=cfg.ROC_CURVE )
		
		if has_visual_context and target_count == 2 and hasattr( model, 'roc_curve' ):
			try:
				plt.close( 'all' )
				model.roc_curve( X_test, y_test )
				st.pyplot( plt.gcf( ) )
				plt.close( 'all' )
			except Exception as e:
				st.info( f'ROC Curve Skipped: {e}' )
				plt.close( 'all' )
		else:
			st.info( 'ROC curve is available only for binary classification targets.' )

# ============================================
# REGRESSION MODE
# ============================================
elif mode == 'Regression Models':
	df_original = st.session_state.get( 'df_dataset', None )
	df_dataset = st.session_state.get( 'df_dataset', None )
	df_working = st.session_state.get( 'df_working', None )
	df_processed = st.session_state.get( 'df_processed', None )
	df_regression = st.session_state.get( 'df_regression', None )
	df_model = st.session_state.get( 'df_model', None )
	df_scores = st.session_state.get( 'df_scores', None )
	df_predictions = st.session_state.get( 'df_predictions', None )
	numeric_columns = st.session_state.get( 'numeric_columns', [ ] )
	categorical_columns = st.session_state.get( 'categorical_columns', [ ] )
	features = st.session_state.get( 'features', [ ] )
	targets = st.session_state.get( 'targets', [ ] )
	active_features = st.session_state.get( 'active_features', [ ] )
	active_targets = st.session_state.get( 'active_targets', [ ] )
	X_data = st.session_state.get( 'X_data', None )
	X_train = st.session_state.get( 'X_train', None )
	X_test = st.session_state.get( 'X_test', None )
	y_train = st.session_state.get( 'y_train', None )
	y_test = st.session_state.get( 'y_test', None )
	y_series = st.session_state.get( 'y_series', None )
	elapsed_seconds = st.session_state.get( 'elapsed_seconds', 0.0 )
	
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )		
	with center:
		st.subheader( cfg.MODE[ 'Regression Models' ] )
		st.caption( 'Predictive Models for Continuous-Values' )
		st.divider( )
		if df_dataset is None or df_dataset.empty:
			st.warning( '⚠️ No dataset loaded.' )
			st.stop( )
		
		df_original = df_dataset.copy( )
		st.session_state[ 'df_original' ] = df_original
		numeric_columns = [ c for c in df_original.columns if pd.api.types.is_numeric_dtype( df_original[ c ] ) ]
		
		categorical_columns = [ c for c in df_original.columns if c not in numeric_columns ]
		st.session_state[ 'categorical_columns' ] = categorical_columns
		st.session_state[ 'numeric_columns' ] = numeric_columns
		if not numeric_columns or not categorical_columns:
			st.warning( '⚠️ Regression requires numeric targets and a categorical features.' )
			st.stop( )
		
		# ======================================================================================
		# Features Selection
		# ======================================================================================
		st.markdown( '##### Feature Selection' )
		st.caption( f'Samples: {len( df_original ):,} | Features: {len( df_original.columns ):,}' )
		
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			features = st.multiselect( 'Select Features', options=df_working.columns  )
			st.session_state[ 'features' ] = features
		
		with col_c2:
			target_options = [ t for t in numeric_columns if t not in features ]
			targets = st.selectbox( 'Select Target', options=target_options,
				key='regression_target' )
			
			st.session_state[ 'targets' ] = targets
		
		sel_b1, sel_b2 = st.columns( [ 0.5, 0.5 ] )
		with sel_b1:
			if st.button( 'Create Working Dataset', icon='➕', key='regression_create_dataset',
					use_container_width=True ):
				
				selected_all = features.copy( )
				target = [ tgt in targets if tgt not in selected_all else [ ] ]
				selected_all.append( target )
				
				if selected_all:
					df_working = df_original[ selected_all ].copy( )
				else:
					df_working = df_original.copy( )
				
				st.session_state[ 'features' ] = features.copy( )
				st.session_state[ 'targets' ] = [ target ] if target else [ ]
				st.session_state[ 'df_working' ] = df_working.copy( )
				st.session_state[ 'df_processed' ] = df_working.copy( )
				commit_frame( df_working )
				st.success( 'Working Dataset Created!' )
		
		with sel_b2:
			if st.button( 'Reset Working Dataset', icon='🔁', key='regression_reset_to_original',
					use_container_width=True ):
				
				df_working = df_original.copy( )
				st.session_state[ 'features' ] = [ ]
				st.session_state[ 'targets' ] = [ ]
				st.session_state[ 'df_working' ] = df_working.copy( )
				st.session_state[ 'df_processed' ] = df_working.copy( )
				commit_frame( df_working )
				st.success( 'Reset to Original' )
		
		df_working = st.session_state.get( 'df_working', pd.DataFrame( ) ).copy( )
		df_processed = st.session_state.get( 'df_processed', pd.DataFrame( ) ).copy( )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Working Data' )
		
		st.caption( f'Samples: {len( df_working ):,} | Fields: {len( df_working.columns ):,}' )
		st.data_editor( df_working, key='regression_working_data' )
		
		# -----------------------------------------------------------------
		# Data Processing
		# -----------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Feature-Engineering' )
		
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			
			with st.expander( label='Data Scaling', icon='⚖️', key='regression_scalers' ):
				
				with st.expander( 'Standard Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.STANDARD_SCALER )
					
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_standard_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_standard_scaler_apply',
								use_container_width=True ):
							
							if scale_cols:
								scaler = StandardScaler( )
								result = scaler.train_transform( df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'Standard Scaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_standard_scaler_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Min-Max Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.MINMAX_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_minmax_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_minmax_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MinMaxScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'Min-Max Scaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_minmax_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.ROBUST_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_robust_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_robust_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = RobustScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'RobustScaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_robust_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Normal Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.NORMAL_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ],
						index=1, key='regression_normal_scaler_norm' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_normal_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = NormalScaler( norm=norm )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'NormalScaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_normal_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.MAXABS_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_maxabs_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_maxabs_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MaxAbsScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'MaxAbsScaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_maxabs_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Data Imputation', icon='🧹', key='regression_imputers' ):
				
				with st.expander( 'Mean Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.MEAN_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='regression_mean_imputer_indicator' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_mean_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = MeanImputer( strategy='mean', add_indicator=add_indicator )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'mean_imputer' )
								
								commit_frame( df_processed )
								st.success( 'MeanImputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_mean_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.NEAREST_NEIGHBOR_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1,
						value=5, step=1, key='regression_nearest_imputer_neighbors' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_nearest_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = NearestImputer( neighbors=int( neighbors ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'nearest_imputer' )
								
								commit_frame( df_processed )
								st.success( 'Nearest Imputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_nearest_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.ITERATIVE_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_iterative_imputer_cols' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1,
						value=10, step=1, key='regression_iterative_imputer_max_iter' )
					
					random_state = st.number_input( 'Random State', min_value=0,
						value=0, step=1, key='regression_iterative_imputer_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Iterative Imputer',
								key='regression_iterative_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = IterativeImputer( max_iter=int( max_iter ),
									random_state=int( random_state ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'iterative_imputer' )
								commit_frame( df_processed )
								st.success( 'Iterative Imputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_iterative_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.SIMPLE_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_simple_imputer_cols' )
					
					strategy = st.selectbox( 'Strategy',
						options=[ 'mean', 'median', 'most_frequent', 'constant' ],
						key='regression_simple_imputer_strategy' )
					
					fill_value = st.text_input( 'Fill Value', value='0.0',
						key='regression_simple_imputer_fill_value' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='regression_simple_imputer_indicator' )
					
					keep_empty_features = st.checkbox( 'Keep Empty Features', value=False,
						key='regression_simple_imputer_keep_empty' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SimpleImputer', key='regression_simpleimputer_apply',
								use_container_width=True ):
							if impute_cols:
								if strategy in [ 'mean', 'median' ]:
									df_input = df_processed[ impute_cols ].apply(
										pd.to_numeric, errors='coerce' )
									fill_object: object = 0.0
								elif strategy == 'constant':
									df_input = df_processed[ impute_cols ].copy( )
									fill_object = fill_value
								else:
									df_input = df_processed[ impute_cols ].copy( )
									fill_object = fill_value
								
								imputer = SimpleImputer( strategy=strategy, fill_value=fill_object,
									add_indicator=add_indicator,
									keep_empty_features=keep_empty_features )
								
								result = imputer.train_transform( df_input.to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'simple_imputer' )
								
								commit_frame( df_processed )
								st.success( 'Simple Imputer Applied' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_simple_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Data Encoding', icon='🔣', key='regression_encoders' ):
				
				with st.expander( 'One-Hot Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.ONEHOT_ENCODER )
					encode_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='regression_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='regression_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0,
						key='regression_onehot_unknown' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_onehot_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OneHotEncoder( sparse=bool( sparse ), unknown=unknown )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, encode_cols,
									result, 'onehot' )
								commit_frame( df_processed )
								st.success( 'OneHotEncoder applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_onehot_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.ORDINAL_ENCODER )
					encode_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='regression_ordinal_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_ordinal_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OrdinalEncoder( )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								df_processed[ encode_cols ] = result
								commit_frame( df_processed )
								st.success( 'Ordinal Encoder Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_ordinal_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Label Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.LABEL_ENCODER )
					target_col = st.selectbox( 'Column', options=categorical_columns,
						key='regression_label_encoder_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_label_encoder_apply',
								use_container_width=True ):
							if target_col:
								encoder = LabelEncoder( )
								result = encoder.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed[ target_col ] = result
								commit_frame( df_processed )
								st.success( 'Label Encoder Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_label_encoder_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Target Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.TARGET_ENCODER )
					encode_cols = st.multiselect( 'Categorical Feature Columns',
						options=categorical_columns, key='regression_target_encoder_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='regression_target_encoder_target_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_target_encoder_apply',
								use_container_width=True ):
							if encode_cols and target_col:
								df_processed = df_working.copy( )
								encoder = TargetEncoder( )
								X_enc = df_processed[ encode_cols ].astype( str ).to_numpy( )
								y_enc = df_processed[ target_col ].to_numpy( )
								result = encoder.train_transform( X_enc, y_enc )
								
								df_processed = replace_columns( df_processed, encode_cols, result,
									'target_encoder' )
								
								commit_frame( df_processed )
								st.success( 'Target Encoder Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_target_encoder_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.POLYNOMIAL_FEATURES )
					poly_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4,
						value=2, key='regression_polynomial_degree' )
					
					interaction = st.checkbox( 'Interaction Only', value=True,
						key='regression_polynomial_interaction' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_polynomial_apply',
								use_container_width=True ):
							if poly_cols:
								df_processed = df_working.copy( )
								
								encoder = PolynomialFeatures( degree=int( degree ),
									interaction=bool( interaction ) )
								
								result = encoder.train_transform(
									df_processed[ poly_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, poly_cols, result,
									'polynomial' )
								
								commit_frame( df_processed )
								st.success( 'PolynomialFeatures applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_polynomial_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
		
		with feature_c2:
			
			with st.expander( label='Data Transformation', icon='⚡', key='regression_transformers' ):
				
				with st.expander( 'Binarizer', expanded=False ):
					transform_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_binarizer_cols' )
					
					threshold = st.number_input( 'Threshold', value=0.0, step=0.1,
						key='regression_binarizer_threshold' )
					
					copy = st.checkbox( 'Copy', value=True, key='regression_binarizer_copy' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Binarizer',
								key='regression_binarizer_apply',
								use_container_width=True ):
							if transform_cols:
								df_processed = df_working.copy( )
								transformer = Binarizer(
									threshold=float( threshold ),
									copy=bool( copy ) )
								result = transformer.train_transform(
									df_processed[ transform_cols ].to_numpy( ) )
								
								df_processed[ transform_cols ] = result
								commit_frame( df_processed )
								st.success( 'Binarizer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_binarizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column',
						options=categorical_columns,
						key='regression_label_binarizer_col' )
					
					pos_label = st.number_input( 'Positive Label', value=1, step=1,
						key='regression_label_binarizer_pos' )
					
					neg_label = st.number_input( 'Negative Label', value=0, step=1,
						key='regression_label_binarizer_neg' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='regression_label_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply LabelBinarizer',
								key='regression_label_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ), sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, [
										target_col ], result,
									'label_binarizer' )
								commit_frame( df_processed )
								st.success( 'Label Binarizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_label_binarizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column',
						options=categorical_columns,
						key='regression_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='regression_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='regression_multilabel_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_multilabel_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								y_multi = parse_multilabel_series( df_processed[ target_col ],
									delimiter=delimiter )
								
								transformer = MultiLabelBinarizer( classes=None,
									sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform( y_multi )
								df_processed = replace_columns( df_processed, [ target_col ],
									result, 'multilabel_binarizer' )
								
								commit_frame( df_processed )
								st.success( 'Multi-Label Binarizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_multilabel_binarizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'TF-IDF Transformer', expanded=False ):
					text_count_cols = st.multiselect( 'Count Matrix Columns',
						options=numeric_columns,
						key='regression_tfidf_transformer_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', None ],
						index=1, key='regression_tfidf_transformer_norm' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='regression_tfidf_transformer_use_idf' )
					
					smooth_idf = st.checkbox( 'Smooth IDF', value=True,
						key='regression_tfidf_transformer_smooth_idf' )
					
					sublinear_tf = st.checkbox( 'Sublinear TF', value=False,
						key='regression_tfidf_transformer_sublinear' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_tfidf_transformer_apply',
								use_container_width=True ):
							if text_count_cols:
								df_processed = df_working.copy( )
								transformer = TfidfTransformer( norm=norm, use_idf=bool( use_idf ),
									smooth_idf=bool( smooth_idf ), sublinear_tf=bool( sublinear_tf ) )
								
								result = transformer.train_transform(
									df_processed[ text_count_cols ].apply(
										pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, text_count_cols,
									result, 'tfidf_transformer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Transformer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_tfidf_transformer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Column Transformer', expanded=False ):
					numeric_columns = st.multiselect( 'Numeric Columns', options=numeric_columns,
						key='regression_column_transformer_numeric_columns' )
					
					categorical_columns = st.multiselect( 'Categorical Columns',
						options=categorical_columns,
						key='regression_column_transformer_categorical_columns' )
					
					numeric_transform = st.selectbox( 'Numeric Transformer',
						options=[ 'StandardScaler', 'MinMaxScaler', 'RobustScaler',
						          'MaxAbsScaler', 'Binarizer', 'None' ],
						key='regression_column_transformer_numeric_transform' )
					
					categorical_transform = st.selectbox( 'Categorical Transformer',
						options=[ 'OneHotEncoder', 'OrdinalEncoder', 'None' ],
						key='regression_column_transformer_categorical_transform' )
					
					remainder = st.selectbox( 'Remainder', options=[ 'drop', 'passthrough' ],
						key='regression_column_transformer_remainder' )
					
					sparse_threshold = st.slider( 'Sparse Threshold', min_value=0.0,
						max_value=1.0, value=0.3,
						key='regression_column_transformer_sparse_threshold' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Column Transformer',
								key='regression_column_transformer_apply',
								use_container_width=True ):
							df_processed = df_working.copy( )
							transformers = [ ]
							
							if numeric_columns and numeric_transform != 'None':
								if numeric_transform == 'StandardScaler':
									numeric_model = StandardScaler( ).model
								elif numeric_transform == 'MinMaxScaler':
									numeric_model = MinMaxScaler( ).model
								elif numeric_transform == 'RobustScaler':
									numeric_model = RobustScaler( ).model
								elif numeric_transform == 'MaxAbsScaler':
									numeric_model = MaxAbsScaler( ).model
								else:
									numeric_model = Binarizer( ).model
								
								transformers.append( ('numeric', numeric_model, numeric_columns) )
							
							if categorical_columns and categorical_transform != 'None':
								if categorical_transform == 'OneHotEncoder':
									categorical_model = OneHotEncoder( sparse=False,
										unknown='ignore' ).model
								else:
									categorical_model = OrdinalEncoder( ).model
								
								transformers.append( ('categorical', categorical_model,
								                      categorical_columns) )
							
							if transformers:
								transformer = ColumnTransformer( transformers=transformers,
									remainder=remainder, sparse_threshold=float( sparse_threshold ),
									n_jobs=None, transformer_weights=None, verbose=False )
								
								result = transformer.train_transform( df_processed )
								df_processed = normalize_result_frame( result=result,
									index=df_processed.index, prefix='column_transformer',
									columns=None )
								
								commit_frame( df_processed )
								st.success( 'ColumnTransformer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_column_transformer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Feature Extration', icon='⛏️', key='regression_extractors' ):
				
				with st.expander( 'TF-IDF Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns',
						options=categorical_columns,
						key='regression_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='regression_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='regression_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='regression_tfidf_vectorizer_use_idf' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_tfidf_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = TfidfVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int( max_features ),
									use_idf=bool( use_idf ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'tfidf_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Vectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_tfidf_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns',
						options=categorical_columns,
						key='regression_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='regression_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0,
						step=1, key='regression_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='regression_count_vectorizer_binary' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_count_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = CountVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int( max_features ),
									binary=bool( binary ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'count_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'Count Vectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_count_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns',
						options=categorical_columns,
						key='regression_hash_vectorizer_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='regression_hash_vectorizer_n_features' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3,
						value=1, key='regression_hash_vectorizer_ngram_max' )
					
					binary = st.checkbox( 'Binary', value=False,
						key='regression_hash_vectorizer_binary' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='regression_hash_vectorizer_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_hash_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = HashVectorizer( num=int( n_features ),
									ngram_range=(1, int( ngram_max )), binary=bool( binary ),
									alternate_sign=bool( alternate_sign ) )
								df_processed = apply_text_vectorizer( df_processed,
									text_cols, transformer, 'hash_vectorizer' )
								commit_frame( df_processed )
								st.success( 'HashVectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_hash_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					dict_cols = st.multiselect( 'Columns',
						options=categorical_columns,
						key='regression_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='regression_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='regression_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='regression_dict_vectorizer_sort' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_dict_vectorizer_apply',
								use_container_width=True ):
							if dict_cols:
								df_processed = df_working.copy( )
								transformer = DictVectorizer( dtype=np.float64, separator=separator,
									sparse=bool( sparse ), sort=bool( sort ) )
								
								df_processed = apply_dict_transform( df_processed, dict_cols,
									transformer, 'dict_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'DictVectorizer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_dict_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					hash_cols = st.multiselect( 'Columns',
						options=categorical_columns,
						key='regression_feature_hasher_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='regression_feature_hasher_n_features' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='regression_feature_hasher_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_feature_hasher_apply',
								use_container_width=True ):
							if hash_cols:
								df_processed = df_working.copy( )
								transformer = FeatureHasher( n_features=int( n_features ),
									input_type='dict', dtype=np.float64,
									alternate_sign=bool( alternate_sign ) )
								
								df_processed = apply_dict_transform( df_processed, hash_cols,
									transformer, 'feature_hasher' )
								commit_frame( df_processed )
								st.success( 'FeatureHasher applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_feature_hasher_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Dimensionality Reduction', icon='🎚️', key='regression_selectors' ):
				
				with st.expander( 'Variance Threshold', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0,
						step=0.01, key='regression_variance_threshold_value' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_variance_threshold_apply',
								use_container_width=True ):
							if select_cols:
								df_processed = df_working.copy( )
								selector = VarianceThreshold( thresh=float( threshold ) )
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result,
									'variance_threshold' )
								
								commit_frame( df_processed )
								st.success( 'VarianceThreshold applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_variance_threshold_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Canonical Correlation Analysis', expanded=False ):
					X_cols = st.multiselect( 'Predictor Columns', options=numeric_columns,
						key='regression_cca_x_cols' )
					
					y_cols = st.multiselect( 'Target Columns', options=numeric_columns,
						key='regression_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2,
						step=1, key='regression_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True,
						key='regression_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500,
						step=1, key='regression_cca_max_iter' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_cca_apply',
								use_container_width=True ):
							if X_cols and y_cols:
								df_processed = df_working.copy( )
								selector = CCA( num=int( n_components ), scale=bool( scale ),
									size=int( max_iter ) )
								
								result = selector.train_transform(
									df_processed[ X_cols ].to_numpy( ),
									df_processed[ y_cols ].to_numpy( ) )
								
								df_result = normalize_result_frame( result=result,
									index=df_processed.index, prefix='cca', columns=None )
								
								df_processed = pd.concat(
									[ df_processed.drop( columns=X_cols + y_cols, errors='ignore' ),
									  df_result ], axis=1 )
								
								commit_frame( df_processed )
								st.success( 'CCA Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_cca_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Principle Component Analysis', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_pca_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2,
						step=1, key='regression_pca_components' )
					
					solver = st.selectbox( 'SVD Solver',
						options=[ 'auto', 'full', 'randomized', 'covariance_eigh', 'arpack' ],
						key='regression_pca_solver' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_pca_apply',
								use_container_width=True ):
							if select_cols:
								df_processed = df_working.copy( )
								selector = PCA( num=int( n_components ), solver=solver )
								
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result, 'pca' )
								commit_frame( df_processed )
								st.success( 'PCA applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_pca_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Select-Best', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='regression_selectbest_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='regression_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
						          'mutual_info_regression' ],
						key='regression_selectbest_score_name' )
					
					k_best = st.number_input( 'K', min_value=1, value=5, step=1,
						key='regression_selectbest_k' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_selectbest_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectBest(
									score_func=score_function_from_name( score_name ),
									num=int( k_best ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'select_best' )
								commit_frame( df_processed )
								st.success( 'Select Best Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_selectbest_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Select-Percent', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='regression_selectpercent_x_cols' )
					
					target_col = st.selectbox( 'Target Column',
						options=categorical_columns,
						key='regression_selectpercent_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
						          'mutual_info_regression' ],
						key='regression_selectpercent_score_name' )
					
					percentile = st.slider( 'Percentile', min_value=1, max_value=100, value=10,
						key='regression_selectpercent_percentile' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SelectPercent',
								key='regression_selectpercent_apply', use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectPercent(
									score_func=score_function_from_name( score_name ),
									pct=int( percentile ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'select_percent' )
								commit_frame( df_processed )
								st.success( 'SelectPercent applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_selectpercent_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Sequential Back Selection', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='regression_sbs_x_cols' )
					
					target_col = st.selectbox( 'Target Column',
						options=categorical_columns,
						key='regression_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='regression_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='regression_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1,
						step=1, key='regression_sbs_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_sbs_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SBS( classifier=None, k_features=int( k_features ),
									test_size=float( test_size ), random_state=int( random_state ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'sbs' )
								
								commit_frame( df_processed )
								st.success( 'SBS applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_sbs_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Recursive Feature Elimination', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='regression_rfe_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='regression_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain',
						min_value=1, value=1, step=1, key='regression_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0,
						step=1, key='regression_rfe_verbose' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_rfe_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = RFE( k_features=int( k_features ), verbose=int( verbose ) )
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'rfe' )
								commit_frame( df_processed )
								st.success( 'RFE applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_rfe_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Processed Data' )
		st.caption( f'Samples: {len( df_processed ):,} | Features: {len( df_processed.columns ):,}' )
		st.data_editor( df_processed, key='regression_processed_data' )
		
		# ------------------------------------------------------------------
		# MODEL TRAINING
		# ------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Model Training' )
		
		active_features = [ c for c in st.session_state.get( 'features', [ ] )
		                    if c in df_processed.columns ]
		
		active_targets = [ c for c in st.session_state.get( 'targets', [ ] )
		                   if c in df_processed.columns ]
		
		if not active_features:
			st.warning( '⚠️ No valid feature columns remain after preprocessing.' )
			st.stop( )
		
		if not active_targets:
			st.warning( '⚠️ No valid target columns remain after preprocessing.' )
			st.stop( )
		
		target_name = active_targets[ 0 ]
		df_model = df_processed.copy( )
		
		# ------------------------------------------------------------------
		# REGRESSION MODELS
		# ------------------------------------------------------------------
		with st.expander( 'Linear Models', expanded=False ):
			
			with st.expander( 'Ordinary Least Squares', expanded=False ):
				ols_defaults = {
						'regression_ols_test_size': 0.20,
						'regression_ols_random_state': 42,
						'regression_ols_fit_intercept': True,
						'regression_ols_copy_x': True,
						'regression_ols_tol': 1e-6,
						'regression_ols_n_jobs': 1,
						'regression_ols_positive': False
				}
				
				for key, value in ols_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Linear regression for continuous targets.' )
				
				ols_c1, ols_c2, ols_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ols_c1:
					st.markdown( '###### Data Split' )
					ols_test_size = st.slider(
						'Test Set Size (%)',
						min_value=10,
						max_value=40,
						value=int( st.session_state[ 'regression_ols_test_size' ] * 100 ),
						step=5,
						key='regression_ols_test_size_slider'
					) / 100.0
					
					ols_random_state = int( st.number_input( 'Random State', min_value=0,
							value=int( st.session_state[ 'regression_ols_random_state' ] ),
							step=1, key='regression_ols_random_state_input' ) )
				
				with ols_c2:
					st.markdown( '###### Model Parameters' )
					ols_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_ols_fit_intercept' ] ),
						key='regression_ols_fit_intercept_check' )
					
					ols_copy_x = st.checkbox( 'Copy X',
						value=bool( st.session_state[ 'regression_ols_copy_x' ] ),
						key='regression_ols_copy_x_check' )
					
					ols_positive = st.checkbox( 'Positive Coefficients',
						value=bool( st.session_state[ 'regression_ols_positive' ] ),
						key='regression_ols_positive_check' )
				
				with ols_c3:
					st.markdown( '###### Solver Settings' )
					ols_tol = float( st.number_input( 'Tolerance', min_value=0.0,
							value=float( st.session_state[ 'regression_ols_tol' ] ),
							step=0.000001, format='%.6f', key='regression_ols_tol_input' ) )
					
					ols_n_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
							value=int( st.session_state[ 'regression_ols_n_jobs' ] ),
							step=1, key='regression_ols_n_jobs_input' ) )
				
				ols_btn_1, ols_btn_2 = st.columns( 2 )
				with ols_btn_1:
					train_ols = st.button( '🚆 Train Ordinary Least Squares',
						key='regression_ols_train', use_container_width=True )
				
				with ols_btn_2:
					reset_ols = st.button( '🔄 Reset Ordinary Least Squares',
						key='regression_ols_reset', use_container_width=True )
				
				if reset_ols:
					for key, value in ols_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_ols_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_ols:
					try:
						st.session_state[ 'regression_ols_test_size' ] = float( ols_test_size )
						st.session_state[ 'regression_ols_random_state' ] = int( ols_random_state )
						st.session_state[ 'regression_ols_fit_intercept' ] = bool( ols_fit_intercept )
						st.session_state[ 'regression_ols_copy_x' ] = bool( ols_copy_x )
						st.session_state[ 'regression_ols_tol' ] = float( ols_tol )
						st.session_state[ 'regression_ols_n_jobs' ] = int( ols_n_jobs )
						st.session_state[ 'regression_ols_positive' ] = bool( ols_positive )
						
						df_training = df_model.copy( )
						X = df_training[ active_features ].apply(
							pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric(
							df_training[ target_name ], errors='coerce' ).fillna(
							0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct values.'
							)
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = regression_model.LeastSquares( fit=bool( ols_fit_intercept ),
							copy=bool( ols_copy_x ), tol=float( ols_tol ), jobs=int( ols_n_jobs ),
							positive=bool( ols_positive ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( ols_test_size ), random=int( ols_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_ols_elapsed_seconds' ] = elapsed_seconds
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as e:
						st.error( f'Error training Ordinary Least Squares: {e}' )
					
			with st.expander( 'Ridge Regression', expanded=False ):
				ridge_defaults = {
						'regression_ridge_alpha': 1.0,
						'regression_ridge_fit_intercept': True,
						'regression_ridge_copy_x': True,
						'regression_ridge_max_iter': 0,
						'regression_ridge_tol': 0.0001,
						'regression_ridge_solver': 'auto',
						'regression_ridge_positive': False,
						'regression_ridge_test_size': 0.20,
						'regression_ridge_random_state': 42
				}
				
				for key, value in ridge_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'L2-regularized linear regression for continuous targets.' )
				
				ridge_c1, ridge_c2, ridge_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ridge_c1:
					st.markdown( '###### Model Parameters' )
					
					ridge_alpha = float( st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'regression_ridge_alpha' ] ),
						step=0.100000, format='%.6f', key='regression_ridge_alpha_input' ) )
					
					ridge_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_ridge_fit_intercept' ] ),
						key='regression_ridge_fit_intercept_check' )
					
					ridge_copy_x = st.checkbox( 'Copy X',
						value=bool( st.session_state[ 'regression_ridge_copy_x' ] ),
						key='regression_ridge_copy_x_check' )
				
				with ridge_c2:
					st.markdown( '###### Solver / Iteration' )
					
					ridge_solver = st.selectbox( 'Solver',
						options=[
								'auto',
								'svd',
								'cholesky',
								'lsqr',
								'sparse_cg',
								'sag',
								'saga',
								'lbfgs'
						],
						index=[
								'auto',
								'svd',
								'cholesky',
								'lsqr',
								'sparse_cg',
								'sag',
								'saga',
								'lbfgs'
						].index( st.session_state[ 'regression_ridge_solver' ] ),
						key='regression_ridge_solver_select' )
					
					ridge_max_iter_raw = int( st.number_input( 'Max Iterations (0 = Auto)', min_value=0,
							value=int( st.session_state[ 'regression_ridge_max_iter' ] ),
							step=1, key='regression_ridge_max_iter_input' ) )
					
					ridge_tol = float( st.number_input( 'Tolerance', min_value=0.0,
							value=float( st.session_state[ 'regression_ridge_tol' ] ),
							step=0.000100, format='%.6f', key='regression_ridge_tol_input' ) )
					
					ridge_positive = st.checkbox( 'Positive Coefficients',
						value=bool( st.session_state[ 'regression_ridge_positive' ] ),
						key='regression_ridge_positive_check' )
					
					if ridge_positive and ridge_solver != 'lbfgs':
						st.info( "Positive coefficients require the 'lbfgs' solver." )
				
				with ridge_c3:
					st.markdown( '###### Data Split' )
					
					ridge_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_ridge_test_size' ] * 100 ),
						step=5, key='regression_ridge_test_size_slider' ) / 100.0
					
					ridge_random_state = int( st.number_input( 'Random State', min_value=0,
							value=int( st.session_state[ 'regression_ridge_random_state' ] ),
							step=1, key='regression_ridge_random_state_input' ) )
				
				ridge_btn_1, ridge_btn_2 = st.columns( 2 )
				with ridge_btn_1: train_ridge = st.button( '🚆 Train Ridge Regression',
					key='regression_ridge_train', use_container_width=True )
				
				with ridge_btn_2:
					reset_ridge = st.button( '🔄 Reset Ridge Regression', key='regression_ridge_reset',
						use_container_width=True )
				
				if reset_ridge:
					for key, value in ridge_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.session_state[ 'regression_ridge_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_ridge:
					try:
						st.session_state[ 'regression_ridge_alpha' ] = float( ridge_alpha )
						st.session_state[ 'regression_ridge_fit_intercept' ] = bool( ridge_fit_intercept )
						st.session_state[ 'regression_ridge_copy_x' ] = bool( ridge_copy_x )
						st.session_state[ 'regression_ridge_max_iter' ] = int( ridge_max_iter_raw )
						st.session_state[ 'regression_ridge_tol' ] = float( ridge_tol )
						st.session_state[ 'regression_ridge_solver' ] = str( ridge_solver )
						st.session_state[ 'regression_ridge_positive' ] = bool( ridge_positive )
						st.session_state[ 'regression_ridge_test_size' ] = float( ridge_test_size )
						st.session_state[ 'regression_ridge_random_state' ] = int( ridge_random_state )
						
						df_training = df_model.copy( )
						X = df_training[ active_features ].apply(
							pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric( df_training[ target_name ],
							errors='coerce' ).fillna( 0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							m = ('⚠️ The selected numeric target must contain at '
							     'least two distinct values.')
							
							st.warning( m )
							st.stop( )
						
						effective_solver = 'lbfgs' if ridge_positive else ridge_solver
						effective_max_iter = None if ridge_max_iter_raw == 0 else int( ridge_max_iter_raw )
						
						start_time = time.perf_counter( )
						model = regression_model.Ridge( alpha=float( ridge_alpha ),
							fit=bool( ridge_fit_intercept ), copy=bool( ridge_copy_x ),
							iters=effective_max_iter, tol=float( ridge_tol ),
							solver=str( effective_solver ), positive=bool( ridge_positive ),
							rando=int( ridge_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( ridge_test_size ), random=int( ridge_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_ridge_elapsed_seconds' ] = elapsed_seconds
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						if df_scores is not None and not df_scores.empty:
							if df_scores.shape[ 1 ] == 1:
								df_scores.columns = [ 'Value' ]
							
							df_scores.loc[ 'Training Score', 'Value' ] = float( model.training_score )
							df_scores.loc[ 'Testing Score', 'Value' ] = float( model.testing_score )
							df_scores.loc[ 'R-Squared Score', 'Value' ] = float( r2_score( y_test, y_prediction ) )
							df_scores.loc[ 'Processing Time (Seconds)', 'Value' ] = round( elapsed_seconds, 4 )
							df_scores.loc[ 'Training Rows', 'Value' ] = int( len( X_train ) )
							df_scores.loc[ 'Testing Rows', 'Value' ] = int( len( X_test ) )
							df_scores.loc[ 'Alpha', 'Value' ] = float( ridge_alpha )
							df_scores.loc[ 'Solver', 'Value' ] = str( effective_solver )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Ridge Regression training failed: {ex}' )
					
			with st.expander( 'Lasso Regression', expanded=False ):
				lasso_defaults = {
						'regression_lasso_alpha': 1.0,
						'regression_lasso_fit_intercept': True,
						'regression_lasso_precompute': False,
						'regression_lasso_copy_x': True,
						'regression_lasso_max_iter': 1000,
						'regression_lasso_tol': 0.0001,
						'regression_lasso_warm_start': False,
						'regression_lasso_positive': False,
						'regression_lasso_random_state': 42,
						'regression_lasso_selection': 'cyclic',
						'regression_lasso_test_size': 0.20
				}
				
				for key, value in lasso_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'L1-regularized linear regression for continuous targets.' )
				
				lasso_c1, lasso_c2, lasso_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with lasso_c1:
					st.markdown( '###### Model Parameters' )
					
					lasso_alpha = float(
						st.number_input(
							'Alpha',
							min_value=0.000001,
							value=float( st.session_state[ 'regression_lasso_alpha' ] ),
							step=0.100000,
							format='%.6f',
							key='regression_lasso_alpha_input'
						)
					)
					
					lasso_fit_intercept = st.checkbox(
						'Fit Intercept',
						value=bool( st.session_state[ 'regression_lasso_fit_intercept' ] ),
						key='regression_lasso_fit_intercept_check'
					)
					
					lasso_precompute = st.checkbox(
						'Precompute',
						value=bool( st.session_state[ 'regression_lasso_precompute' ] ),
						key='regression_lasso_precompute_check'
					)
					
					lasso_copy_x = st.checkbox(
						'Copy X',
						value=bool( st.session_state[ 'regression_lasso_copy_x' ] ),
						key='regression_lasso_copy_x_check'
					)
				
				with lasso_c2:
					st.markdown( '###### Solver / Iteration' )
					
					lasso_max_iter = int(
						st.number_input(
							'Max Iterations',
							min_value=1,
							value=int( st.session_state[ 'regression_lasso_max_iter' ] ),
							step=1,
							key='regression_lasso_max_iter_input'
						)
					)
					
					lasso_tol = float(
						st.number_input(
							'Tolerance',
							min_value=0.0,
							value=float( st.session_state[ 'regression_lasso_tol' ] ),
							step=0.000100,
							format='%.6f',
							key='regression_lasso_tol_input'
						)
					)
					
					lasso_warm_start = st.checkbox(
						'Warm Start',
						value=bool( st.session_state[ 'regression_lasso_warm_start' ] ),
						key='regression_lasso_warm_start_check'
					)
					
					lasso_positive = st.checkbox(
						'Positive Coefficients',
						value=bool( st.session_state[ 'regression_lasso_positive' ] ),
						key='regression_lasso_positive_check'
					)
					
					lasso_selection = st.selectbox(
						'Selection',
						options=[ 'cyclic', 'random' ],
						index=[ 'cyclic', 'random' ].index(
							st.session_state[ 'regression_lasso_selection' ]
						),
						key='regression_lasso_selection_select'
					)
				
				with lasso_c3:
					st.markdown( '###### Data Split' )
					
					lasso_test_size = st.slider(
						'Test Set Size (%)',
						min_value=10,
						max_value=40,
						value=int( st.session_state[ 'regression_lasso_test_size' ] * 100 ),
						step=5,
						key='regression_lasso_test_size_slider'
					) / 100.0
					
					lasso_random_state = int(
						st.number_input(
							'Random State',
							min_value=0,
							value=int( st.session_state[ 'regression_lasso_random_state' ] ),
							step=1,
							key='regression_lasso_random_state_input'
						)
					)
				
				lasso_btn_1, lasso_btn_2 = st.columns( 2 )
				with lasso_btn_1:
					train_lasso = st.button(
						'🚆 Train Lasso Regression',
						key='regression_lasso_train',
						use_container_width=True
					)
				
				with lasso_btn_2:
					reset_lasso = st.button(
						'🔄 Reset Lasso Regression',
						key='regression_lasso_reset',
						use_container_width=True
					)
				
				if reset_lasso:
					for key, value in lasso_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.session_state[ 'regression_lasso_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_lasso:
					try:
						st.session_state[ 'regression_lasso_alpha' ] = float( lasso_alpha )
						st.session_state[ 'regression_lasso_fit_intercept' ] = bool(
							lasso_fit_intercept
						)
						st.session_state[ 'regression_lasso_precompute' ] = bool(
							lasso_precompute
						)
						st.session_state[ 'regression_lasso_copy_x' ] = bool( lasso_copy_x )
						st.session_state[ 'regression_lasso_max_iter' ] = int( lasso_max_iter )
						st.session_state[ 'regression_lasso_tol' ] = float( lasso_tol )
						st.session_state[ 'regression_lasso_warm_start' ] = bool(
							lasso_warm_start
						)
						st.session_state[ 'regression_lasso_positive' ] = bool( lasso_positive )
						st.session_state[ 'regression_lasso_random_state' ] = int(
							lasso_random_state
						)
						st.session_state[ 'regression_lasso_selection' ] = str(
							lasso_selection
						)
						st.session_state[ 'regression_lasso_test_size' ] = float(
							lasso_test_size
						)
						
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply(
							pd.to_numeric,
							errors='coerce'
						).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric(
							df_training[ target_name ],
							errors='coerce'
						).fillna( 0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct values.'
							)
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = regression_model.Lasso(
							alpha=float( lasso_alpha ),
							fit=bool( lasso_fit_intercept ),
							precompute=bool( lasso_precompute ),
							copy=bool( lasso_copy_x ),
							iters=int( lasso_max_iter ),
							tol=float( lasso_tol ),
							warm=bool( lasso_warm_start ),
							positive=bool( lasso_positive ),
							rando=int( lasso_random_state ),
							selection=str( lasso_selection )
						)
						
						X_train, X_test, y_train, y_test = model.split_data(
							X,
							y,
							size=float( lasso_test_size ),
							random=int( lasso_random_state )
						)
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_lasso_elapsed_seconds' ] = elapsed_seconds
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if df_scores.shape[ 1 ] == 1:
								df_scores.columns = [ 'Value' ]
							
							df_scores.loc[ 'Training Score', 'Value' ] = float(
								model.training_score
							)
							df_scores.loc[ 'Testing Score', 'Value' ] = float(
								model.testing_score
							)
							df_scores.loc[ 'R-Squared Score', 'Value' ] = float( r2_score(
								y_test, y_prediction ) )
							
							df_scores.loc[ 'Processing Time (Seconds)', 'Value' ] = round(
								elapsed_seconds,
								4
							)
							df_scores.loc[ 'Training Rows', 'Value' ] = int( len( X_train ) )
							df_scores.loc[ 'Testing Rows', 'Value' ] = int( len( X_test ) )
							df_scores.loc[ 'Alpha', 'Value' ] = float( lasso_alpha )
							df_scores.loc[ 'Selection', 'Value' ] = str( lasso_selection )
						
						df_predictions = pd.DataFrame(
							{
									'Actual': y_test,
									'Predicted': y_prediction
							}
						)
						
						df_coefficients = pd.DataFrame(
							{
									'Feature': active_features,
									'Coefficient': np.asarray( model.weights ).reshape( -1 )
							}
						)
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Lasso Regression training failed: {ex}' )
					
			with st.expander( 'Elastic Net', expanded=False ):
				elastic_defaults = {
						'regression_elastic_alpha': 1.0,
						'regression_elastic_ratio': 0.5,
						'regression_elastic_fit_intercept': True,
						'regression_elastic_precompute': False,
						'regression_elastic_copy_x': True,
						'regression_elastic_max_iter': 1000,
						'regression_elastic_tol': 0.0001,
						'regression_elastic_warm_start': False,
						'regression_elastic_positive': False,
						'regression_elastic_random_state': 42,
						'regression_elastic_selection': 'cyclic',
						'regression_elastic_test_size': 0.20
				}
				
				for key, value in elastic_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Combined L1/L2-regularized linear regression for continuous targets.' )
				
				elastic_c1, elastic_c2, elastic_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with elastic_c1:
					st.markdown( '###### Model Parameters' )
					elastic_alpha = float(
						st.number_input(
							'Alpha',
							min_value=0.000001,
							value=float( st.session_state[ 'regression_elastic_alpha' ] ),
							step=0.100000,
							format='%.6f',
							key='regression_elastic_alpha_input'
						)
					)
					
					elastic_ratio = float(
						st.slider(
							'L1 Ratio',
							min_value=0.0,
							max_value=1.0,
							value=float( st.session_state[ 'regression_elastic_ratio' ] ),
							step=0.05,
							key='regression_elastic_ratio_slider'
						)
					)
					
					elastic_fit_intercept = st.checkbox(
						'Fit Intercept',
						value=bool( st.session_state[ 'regression_elastic_fit_intercept' ] ),
						key='regression_elastic_fit_intercept_check'
					)
					
					elastic_precompute = st.checkbox(
						'Precompute',
						value=bool( st.session_state[ 'regression_elastic_precompute' ] ),
						key='regression_elastic_precompute_check'
					)
				
				with elastic_c2:
					st.markdown( '###### Solver / Iteration' )
					
					elastic_copy_x = st.checkbox(
						'Copy X',
						value=bool( st.session_state[ 'regression_elastic_copy_x' ] ),
						key='regression_elastic_copy_x_check'
					)
					
					elastic_max_iter = int(
						st.number_input(
							'Max Iterations',
							min_value=1,
							value=int( st.session_state[ 'regression_elastic_max_iter' ] ),
							step=1,
							key='regression_elastic_max_iter_input'
						)
					)
					
					elastic_tol = float(
						st.number_input(
							'Tolerance',
							min_value=0.0,
							value=float( st.session_state[ 'regression_elastic_tol' ] ),
							step=0.000100,
							format='%.6f',
							key='regression_elastic_tol_input'
						)
					)
					
					elastic_warm_start = st.checkbox(
						'Warm Start',
						value=bool( st.session_state[ 'regression_elastic_warm_start' ] ),
						key='regression_elastic_warm_start_check'
					)
					
					elastic_positive = st.checkbox(
						'Positive Coefficients',
						value=bool( st.session_state[ 'regression_elastic_positive' ] ),
						key='regression_elastic_positive_check'
					)
					
					elastic_selection = st.selectbox(
						'Selection',
						options=[ 'cyclic', 'random' ],
						index=[ 'cyclic', 'random' ].index(
							st.session_state[ 'regression_elastic_selection' ]
						),
						key='regression_elastic_selection_select'
					)
				
				with elastic_c3:
					st.markdown( '###### Data Split' )
					elastic_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_elastic_test_size' ] * 100 ),
						step=5, key='regression_elastic_test_size_slider' ) / 100.0
					
					elastic_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_elastic_random_state' ] ), step=1,
						key='regression_elastic_random_state_input' ) )
					
					if elastic_ratio == 1.0:
						st.info( 'l1_ratio = 1.0 is equivalent to Lasso.' )
				
				elastic_btn_1, elastic_btn_2 = st.columns( 2 )
				with elastic_btn_1:
					train_elastic = st.button(
						'🚆 Train Elastic Net',
						key='regression_elastic_train',
						use_container_width=True
					)
				
				with elastic_btn_2:
					reset_elastic = st.button(
						'🔄 Reset Elastic Net',
						key='regression_elastic_reset',
						use_container_width=True
					)
				
				if reset_elastic:
					for key, value in elastic_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.session_state[ 'regression_elastic_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_elastic:
					try:
						st.session_state[ 'regression_elastic_alpha' ] = float( elastic_alpha )
						st.session_state[ 'regression_elastic_ratio' ] = float( elastic_ratio )
						st.session_state[ 'regression_elastic_fit_intercept' ] = bool(
							elastic_fit_intercept
						)
						st.session_state[ 'regression_elastic_precompute' ] = bool(
							elastic_precompute
						)
						st.session_state[ 'regression_elastic_copy_x' ] = bool( elastic_copy_x )
						st.session_state[ 'regression_elastic_max_iter' ] = int( elastic_max_iter )
						st.session_state[ 'regression_elastic_tol' ] = float( elastic_tol )
						st.session_state[ 'regression_elastic_warm_start' ] = bool(
							elastic_warm_start
						)
						st.session_state[ 'regression_elastic_positive' ] = bool(
							elastic_positive
						)
						st.session_state[ 'regression_elastic_random_state' ] = int(
							elastic_random_state
						)
						st.session_state[ 'regression_elastic_selection' ] = str(
							elastic_selection
						)
						st.session_state[ 'regression_elastic_test_size' ] = float(
							elastic_test_size
						)
						
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply(
							pd.to_numeric,
							errors='coerce'
						).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric(
							df_training[ target_name ],
							errors='coerce'
						).fillna( 0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct values.'
							)
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = regression_model.ElasticNet(
							alpha=float( elastic_alpha ),
							ratio=float( elastic_ratio ),
							fit=bool( elastic_fit_intercept ),
							precompute=bool( elastic_precompute ),
							iters=int( elastic_max_iter ),
							copy=bool( elastic_copy_x ),
							tol=float( elastic_tol ),
							warm=bool( elastic_warm_start ),
							positive=bool( elastic_positive ),
							rando=int( elastic_random_state ),
							select=str( elastic_selection )
						)
						
						X_train, X_test, y_train, y_test = model.split_data(
							X,
							y,
							size=float( elastic_test_size ),
							random=int( elastic_random_state )
						)
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_elastic_elapsed_seconds' ] = elapsed_seconds
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if df_scores.shape[ 1 ] == 1:
								df_scores.columns = [ 'Value' ]
							
							df_scores.loc[ 'Training Score', 'Value' ] = float(
								model.training_score
							)
							df_scores.loc[ 'Testing Score', 'Value' ] = float(
								model.testing_score
							)
							df_scores.loc[ 'R-Squared Score', 'Value' ] = float(
								r2_score( y_test, y_prediction )
							)
							df_scores.loc[ 'Processing Time (Seconds)', 'Value' ] = round(
								elapsed_seconds,
								4
							)
							df_scores.loc[ 'Training Rows', 'Value' ] = int( len( X_train ) )
							df_scores.loc[ 'Testing Rows', 'Value' ] = int( len( X_test ) )
							df_scores.loc[ 'Alpha', 'Value' ] = float( elastic_alpha )
							df_scores.loc[ 'L1 Ratio', 'Value' ] = float( elastic_ratio )
							df_scores.loc[ 'Selection', 'Value' ] = str( elastic_selection )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						
						df_coefficients = pd.DataFrame( {
									'Feature': active_features,
									'Coefficient': np.asarray( model.weights ).reshape( -1 )
							} )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Elastic Net training failed: {ex}' )
					
			with st.expander( 'Bayesian Ridge', expanded=False ):
				bayes_defaults = {
						'regression_bayes_max_iter': 300,
						'regression_bayes_shape_alpha': 0.000001,
						'regression_bayes_scale_alpha': 0.000001,
						'regression_bayes_shape_lambda': 0.000001,
						'regression_bayes_scale_lambda': 0.000001,
						'regression_bayes_tol': 0.001000,
						'regression_bayes_alpha_init': 0.0,
						'regression_bayes_lambda_init': 0.0,
						'regression_bayes_compute_score': False,
						'regression_bayes_fit_intercept': True,
						'regression_bayes_copy_x': True,
						'regression_bayes_verbose': False,
						'regression_bayes_test_size': 0.20,
						'regression_bayes_random_state': 42
				}
				
				for key, value in bayes_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Bayesian linear regression with automatic regularization estimation.' )
				
				bayes_c1, bayes_c2, bayes_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with bayes_c1:
					st.markdown( '###### Prior / Precision Parameters' )
					
					bayes_shape_alpha = float(
						st.number_input(
							'Alpha Shape',
							min_value=0.0,
							value=float( st.session_state[ 'regression_bayes_shape_alpha' ] ),
							step=0.000001,
							format='%.6f',
							key='regression_bayes_shape_alpha_input'
						)
					)
					
					bayes_scale_alpha = float(
						st.number_input(
							'Alpha Scale',
							min_value=0.0,
							value=float( st.session_state[ 'regression_bayes_scale_alpha' ] ),
							step=0.000001,
							format='%.6f',
							key='regression_bayes_scale_alpha_input'
						)
					)
					
					bayes_shape_lambda = float(
						st.number_input(
							'Lambda Shape',
							min_value=0.0,
							value=float( st.session_state[ 'regression_bayes_shape_lambda' ] ),
							step=0.000001,
							format='%.6f',
							key='regression_bayes_shape_lambda_input'
						)
					)
					
					bayes_scale_lambda = float(
						st.number_input(
							'Lambda Scale',
							min_value=0.0,
							value=float( st.session_state[ 'regression_bayes_scale_lambda' ] ),
							step=0.000001,
							format='%.6f',
							key='regression_bayes_scale_lambda_input'
						)
					)
				
				with bayes_c2:
					st.markdown( '###### Model Parameters' )
					
					bayes_max_iter = int(
						st.number_input(
							'Max Iterations',
							min_value=1,
							value=int( st.session_state[ 'regression_bayes_max_iter' ] ),
							step=1,
							key='regression_bayes_max_iter_input'
						)
					)
					
					bayes_tol = float(
						st.number_input(
							'Tolerance',
							min_value=0.0,
							value=float( st.session_state[ 'regression_bayes_tol' ] ),
							step=0.000100,
							format='%.6f',
							key='regression_bayes_tol_input'
						)
					)
					
					bayes_alpha_init_raw = float(
						st.number_input( 'Alpha Init (0 = None)', min_value=0.0,
							value=float( st.session_state[ 'regression_bayes_alpha_init' ] ),
							step=0.000100, format='%.6f',
							key='regression_bayes_alpha_init_input' ) )
					
					bayes_lambda_init_raw = float(
						st.number_input(
							'Lambda Init (0 = None)',
							min_value=0.0,
							value=float( st.session_state[ 'regression_bayes_lambda_init' ] ),
							step=0.000100,
							format='%.6f',
							key='regression_bayes_lambda_init_input'
						)
					)
					
					bayes_compute_score = st.checkbox(
						'Compute Marginal Log Likelihood',
						value=bool( st.session_state[ 'regression_bayes_compute_score' ] ),
						key='regression_bayes_compute_score_check'
					)
					
					bayes_fit_intercept = st.checkbox(
						'Fit Intercept',
						value=bool( st.session_state[ 'regression_bayes_fit_intercept' ] ),
						key='regression_bayes_fit_intercept_check'
					)
					
					bayes_copy_x = st.checkbox(
						'Copy X',
						value=bool( st.session_state[ 'regression_bayes_copy_x' ] ),
						key='regression_bayes_copy_x_check'
					)
					
					bayes_verbose = st.checkbox(
						'Verbose',
						value=bool( st.session_state[ 'regression_bayes_verbose' ] ),
						key='regression_bayes_verbose_check'
					)
				
				with bayes_c3:
					st.markdown( '###### Data Split' )
					
					bayes_test_size = st.slider(
						'Test Set Size (%)',
						min_value=10,
						max_value=40,
						value=int( st.session_state[ 'regression_bayes_test_size' ] * 100 ),
						step=5,
						key='regression_bayes_test_size_slider'
					) / 100.0
					
					bayes_random_state = int(
						st.number_input(
							'Random State',
							min_value=0,
							value=int( st.session_state[ 'regression_bayes_random_state' ] ),
							step=1,
							key='regression_bayes_random_state_input'
						)
					)
				
				bayes_btn_1, bayes_btn_2 = st.columns( 2 )
				with bayes_btn_1:
					train_bayes = st.button(
						'🚆 Train Bayesian Ridge',
						key='regression_bayes_train',
						use_container_width=True
					)
				
				with bayes_btn_2:
					reset_bayes = st.button(
						'🔄 Reset Bayesian Ridge',
						key='regression_bayes_reset',
						use_container_width=True
					)
				
				if reset_bayes:
					for key, value in bayes_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.session_state[ 'regression_bayes_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_bayes:
					try:
						st.session_state[ 'regression_bayes_max_iter' ] = int( bayes_max_iter )
						st.session_state[ 'regression_bayes_shape_alpha' ] = float(
							bayes_shape_alpha
						)
						st.session_state[ 'regression_bayes_scale_alpha' ] = float(
							bayes_scale_alpha
						)
						st.session_state[ 'regression_bayes_shape_lambda' ] = float(
							bayes_shape_lambda
						)
						st.session_state[ 'regression_bayes_scale_lambda' ] = float(
							bayes_scale_lambda
						)
						st.session_state[ 'regression_bayes_tol' ] = float( bayes_tol )
						st.session_state[ 'regression_bayes_alpha_init' ] = float(
							bayes_alpha_init_raw
						)
						st.session_state[ 'regression_bayes_lambda_init' ] = float(
							bayes_lambda_init_raw
						)
						st.session_state[ 'regression_bayes_compute_score' ] = bool(
							bayes_compute_score
						)
						st.session_state[ 'regression_bayes_fit_intercept' ] = bool(
							bayes_fit_intercept
						)
						st.session_state[ 'regression_bayes_copy_x' ] = bool( bayes_copy_x )
						st.session_state[ 'regression_bayes_verbose' ] = bool( bayes_verbose )
						st.session_state[ 'regression_bayes_test_size' ] = float(
							bayes_test_size
						)
						st.session_state[ 'regression_bayes_random_state' ] = int(
							bayes_random_state
						)
						
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply(
							pd.to_numeric,
							errors='coerce'
						).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric(
							df_training[ target_name ],
							errors='coerce'
						).fillna( 0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct values.'
							)
							st.stop( )
						
						bayes_alpha_init = None if bayes_alpha_init_raw == 0.0 else float(
							bayes_alpha_init_raw
						)
						
						bayes_lambda_init = None if bayes_lambda_init_raw == 0.0 else float(
							bayes_lambda_init_raw
						)
						
						start_time = time.perf_counter( )
						
						model = regression_model.BayesianRidge(
							max=int( bayes_max_iter ),
							shape_alpha=float( bayes_shape_alpha ),
							scale_alpha=float( bayes_scale_alpha ),
							shape_lambda=float( bayes_shape_lambda ),
							scale_lambda=float( bayes_scale_lambda ),
							tol=float( bayes_tol ),
							alpha_init=bayes_alpha_init,
							lambda_init=bayes_lambda_init,
							compute_score=bool( bayes_compute_score ),
							fit=bool( bayes_fit_intercept ),
							copy=bool( bayes_copy_x ),
							verbose=bool( bayes_verbose )
						)
						
						X_train, X_test, y_train, y_test = model.split_data(
							X,
							y,
							size=float( bayes_test_size ),
							random=int( bayes_random_state )
						)
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_bayes_elapsed_seconds' ] = elapsed_seconds
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							elif df_scores.shape[ 1 ] == 1:
								df_scores = df_scores.reset_index( )
								df_scores.columns = [ 'Metric', 'Value' ]
							
							df_extra = pd.DataFrame(
								{
										'Metric': [
												'Processing Time (Seconds)',
												'Training Rows',
												'Testing Rows',
												'Max Iterations'
										],
										'Value': [
												round( elapsed_seconds, 4 ),
												int( len( X_train ) ),
												int( len( X_test ) ),
												int( bayes_max_iter )
										]
								}
							)
							
							df_scores = pd.concat(
								[ df_scores, df_extra ],
								ignore_index=True
							)
						
						df_predictions = pd.DataFrame(
							{
									'Actual': y_test,
									'Predicted': y_prediction
							}
						)
						
						df_coefficients = pd.DataFrame(
							{
									'Feature': active_features,
									'Coefficient': np.asarray( model.model.coef_ ).reshape( -1 )
							}
						)
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Bayesian Ridge training failed: {ex}' )
			
			with st.expander( 'Stochastic Gradient Descent', expanded=False ):
				sgd_defaults = {
						'regression_sgd_loss': 'squared_error',
						'regression_sgd_penalty': 'l2',
						'regression_sgd_alpha': 0.0001,
						'regression_sgd_iters': 1000,
						'regression_sgd_shuffle': True,
						'regression_sgd_learning_rate': 'invscaling',
						'regression_sgd_l1_ratio': 0.15,
						'regression_sgd_fit_intercept': True,
						'regression_sgd_tol': 0.001000,
						'regression_sgd_verbose': 0,
						'regression_sgd_epsilon': 0.1,
						'regression_sgd_eta0': 0.01,
						'regression_sgd_power_t': 0.25,
						'regression_sgd_early_stopping': False,
						'regression_sgd_validation_fraction': 0.1,
						'regression_sgd_n_iter_no_change': 5,
						'regression_sgd_warm_start': False,
						'regression_sgd_average': False,
						'regression_sgd_test_size': 0.20,
						'regression_sgd_random_state': 42
				}
				
				for key, value in sgd_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Linear regression trained with SGD for large-scale continuous targets.' )
				
				sgd_c1, sgd_c2, sgd_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with sgd_c1:
					st.markdown( '###### Loss / Penalty' )
					
					sgd_loss = st.selectbox(
						'Loss',
						options=[
								'squared_error',
								'huber',
								'epsilon_insensitive',
								'squared_epsilon_insensitive'
						],
						index=[
								'squared_error',
								'huber',
								'epsilon_insensitive',
								'squared_epsilon_insensitive'
						].index( st.session_state[ 'regression_sgd_loss' ] ),
						key='regression_sgd_loss_select'
					)
					
					sgd_penalty = st.selectbox(
						'Penalty',
						options=[ None, 'l2', 'l1', 'elasticnet' ],
						index=[ None, 'l2', 'l1', 'elasticnet' ].index(
							st.session_state[ 'regression_sgd_penalty' ]
						),
						format_func=lambda v: 'None' if v is None else str( v ),
						key='regression_sgd_penalty_select'
					)
					
					sgd_alpha = float(
						st.number_input(
							'Alpha',
							min_value=0.000001,
							value=float( st.session_state[ 'regression_sgd_alpha' ] ),
							step=0.000100,
							format='%.6f',
							key='regression_sgd_alpha_input'
						)
					)
					
					sgd_iters = int(
						st.number_input(
							'Iterations',
							min_value=1,
							value=int( st.session_state[ 'regression_sgd_iters' ] ),
							step=1,
							key='regression_sgd_iters_input'
						)
					)
					
					sgd_l1_ratio = float(
						st.slider(
							'L1 Ratio',
							min_value=0.0,
							max_value=1.0,
							value=float( st.session_state[ 'regression_sgd_l1_ratio' ] ),
							step=0.05,
							key='regression_sgd_l1_ratio_slider'
						)
					)
				
				with sgd_c2:
					st.markdown( '###### Learning Controls' )
					
					sgd_shuffle = st.checkbox(
						'Shuffle',
						value=bool( st.session_state[ 'regression_sgd_shuffle' ] ),
						key='regression_sgd_shuffle_check'
					)
					
					sgd_learning_rate = st.selectbox(
						'Learning Rate Schedule',
						options=[ 'constant', 'optimal', 'invscaling', 'adaptive' ],
						index=[ 'constant', 'optimal', 'invscaling', 'adaptive' ].index(
							st.session_state[ 'regression_sgd_learning_rate' ]
						),
						key='regression_sgd_learning_rate_select'
					)
					
					sgd_eta0 = float(
						st.number_input(
							'Eta0',
							min_value=0.000001,
							value=float( st.session_state[ 'regression_sgd_eta0' ] ),
							step=0.010000,
							format='%.6f',
							key='regression_sgd_eta0_input'
						)
					)
					
					sgd_power_t = float(
						st.number_input(
							'Power T',
							min_value=0.0,
							value=float( st.session_state[ 'regression_sgd_power_t' ] ),
							step=0.100000,
							format='%.6f',
							key='regression_sgd_power_t_input'
						)
					)
					
					sgd_epsilon = float(
						st.number_input(
							'Epsilon',
							min_value=0.0,
							value=float( st.session_state[ 'regression_sgd_epsilon' ] ),
							step=0.010000,
							format='%.6f',
							key='regression_sgd_epsilon_input'
						)
					)
					
					sgd_tol = float(
						st.number_input(
							'Tolerance',
							min_value=0.0,
							value=float( st.session_state[ 'regression_sgd_tol' ] ),
							step=0.000100,
							format='%.6f',
							key='regression_sgd_tol_input'
						)
					)
					
					sgd_fit_intercept = st.checkbox(
						'Fit Intercept',
						value=bool( st.session_state[ 'regression_sgd_fit_intercept' ] ),
						key='regression_sgd_fit_intercept_check'
					)
				
				with sgd_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					
					sgd_early_stopping = st.checkbox(
						'Early Stopping',
						value=bool( st.session_state[ 'regression_sgd_early_stopping' ] ),
						key='regression_sgd_early_stopping_check'
					)
					
					sgd_validation_fraction = float(
						st.slider(
							'Validation Fraction',
							min_value=0.05,
							max_value=0.40,
							value=float( st.session_state[ 'regression_sgd_validation_fraction' ] ),
							step=0.05,
							key='regression_sgd_validation_fraction_slider'
						)
					)
					
					sgd_n_iter_no_change = int(
						st.number_input(
							'N Iter No Change',
							min_value=1,
							value=int( st.session_state[ 'regression_sgd_n_iter_no_change' ] ),
							step=1,
							key='regression_sgd_n_iter_no_change_input'
						)
					)
					
					sgd_warm_start = st.checkbox(
						'Warm Start',
						value=bool( st.session_state[ 'regression_sgd_warm_start' ] ),
						key='regression_sgd_warm_start_check'
					)
					
					sgd_average = st.checkbox(
						'Average Weights',
						value=bool( st.session_state[ 'regression_sgd_average' ] ),
						key='regression_sgd_average_check'
					)
					
					sgd_verbose = int(
						st.number_input(
							'Verbose',
							min_value=0,
							value=int( st.session_state[ 'regression_sgd_verbose' ] ),
							step=1,
							key='regression_sgd_verbose_input'
						)
					)
					
					sgd_test_size = st.slider(
						'Test Set Size (%)',
						min_value=10,
						max_value=40,
						value=int( st.session_state[ 'regression_sgd_test_size' ] * 100 ),
						step=5,
						key='regression_sgd_test_size_slider'
					) / 100.0
					
					sgd_random_state = int(
						st.number_input(
							'Random State',
							min_value=0,
							value=int( st.session_state[ 'regression_sgd_random_state' ] ),
							step=1,
							key='regression_sgd_random_state_input'
						)
					)
					
					if sgd_penalty != 'elasticnet':
						st.caption( 'L1 Ratio is only used when Penalty = elasticnet.' )
					
					if sgd_loss not in [ 'huber', 'epsilon_insensitive',
					                     'squared_epsilon_insensitive' ]:
						st.caption( 'Epsilon is only used by Huber and epsilon-insensitive losses.' )
				
				sgd_btn_1, sgd_btn_2 = st.columns( 2 )
				
				with sgd_btn_1:
					train_sgd = st.button(
						'🚆 Train Stochastic Gradient Descent',
						key='regression_sgd_train',
						use_container_width=True
					)
				
				with sgd_btn_2:
					reset_sgd = st.button(
						'🔄 Reset Stochastic Gradient Descent',
						key='regression_sgd_reset',
						use_container_width=True
					)
				
				if reset_sgd:
					for key, value in sgd_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.session_state[ 'regression_sgd_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_sgd:
					try:
						st.session_state[ 'regression_sgd_loss' ] = str( sgd_loss )
						st.session_state[ 'regression_sgd_penalty' ] = sgd_penalty
						st.session_state[ 'regression_sgd_alpha' ] = float( sgd_alpha )
						st.session_state[ 'regression_sgd_iters' ] = int( sgd_iters )
						st.session_state[ 'regression_sgd_shuffle' ] = bool( sgd_shuffle )
						st.session_state[
							'regression_sgd_learning_rate' ] = str( sgd_learning_rate )
						st.session_state[ 'regression_sgd_l1_ratio' ] = float( sgd_l1_ratio )
						st.session_state[ 'regression_sgd_fit_intercept' ] = bool(
							sgd_fit_intercept
						)
						st.session_state[ 'regression_sgd_tol' ] = float( sgd_tol )
						st.session_state[ 'regression_sgd_verbose' ] = int( sgd_verbose )
						st.session_state[ 'regression_sgd_epsilon' ] = float( sgd_epsilon )
						st.session_state[ 'regression_sgd_eta0' ] = float( sgd_eta0 )
						st.session_state[ 'regression_sgd_power_t' ] = float( sgd_power_t )
						st.session_state[ 'regression_sgd_early_stopping' ] = bool(
							sgd_early_stopping
						)
						st.session_state[ 'regression_sgd_validation_fraction' ] = float(
							sgd_validation_fraction
						)
						st.session_state[ 'regression_sgd_n_iter_no_change' ] = int(
							sgd_n_iter_no_change
						)
						st.session_state[ 'regression_sgd_warm_start' ] = bool( sgd_warm_start )
						st.session_state[ 'regression_sgd_average' ] = bool( sgd_average )
						st.session_state[ 'regression_sgd_test_size' ] = float( sgd_test_size )
						st.session_state[ 'regression_sgd_random_state' ] = int(
							sgd_random_state
						)
						
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply(
							pd.to_numeric,
							errors='coerce'
						).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric(
							df_training[ target_name ],
							errors='coerce'
						).fillna( 0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct values.'
							)
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = regression_model.GradientDescent(
							loss=str( sgd_loss ),
							iters=int( sgd_iters ),
							penalty=sgd_penalty,
							alpha=float( sgd_alpha ),
							rando=int( sgd_random_state ),
							learning_rate=str( sgd_learning_rate ),
							l1_ratio=float( sgd_l1_ratio ),
							fit=bool( sgd_fit_intercept ),
							tol=float( sgd_tol ),
							shuffle=bool( sgd_shuffle ),
							verbose=int( sgd_verbose ),
							epsilon=float( sgd_epsilon ),
							eta0=float( sgd_eta0 ),
							power_t=float( sgd_power_t ),
							early_stopping=bool( sgd_early_stopping ),
							validation_fraction=float( sgd_validation_fraction ),
							n_iter_no_change=int( sgd_n_iter_no_change ),
							warm=bool( sgd_warm_start ),
							average=bool( sgd_average )
						)
						
						X_train, X_test, y_train, y_test = model.split_data(
							X,
							y,
							size=float( sgd_test_size ),
							random=int( sgd_random_state )
						)
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_sgd_elapsed_seconds' ] = elapsed_seconds
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							elif df_scores.shape[ 1 ] == 1:
								df_scores = df_scores.reset_index( )
								df_scores.columns = [ 'Metric', 'Value' ]
							
							df_extra = pd.DataFrame(
								{
										'Metric': [
												'Processing Time (Seconds)',
												'Training Rows',
												'Testing Rows',
												'Loss',
												'Penalty',
												'Learning Rate'
										],
										'Value': [
												round( elapsed_seconds, 4 ),
												int( len( X_train ) ),
												int( len( X_test ) ),
												str( sgd_loss ),
												'None' if sgd_penalty is None else str( sgd_penalty ),
												str( sgd_learning_rate )
										]
								}
							)
							
							df_scores = pd.concat(
								[ df_scores, df_extra ],
								ignore_index=True
							)
						
						df_predictions = pd.DataFrame(
							{
									'Actual': y_test,
									'Predicted': y_prediction
							}
						)
						
						df_coefficients = pd.DataFrame(
							{
									'Feature': active_features,
									'Coefficient': np.asarray( model.weights ).reshape( -1 )
							}
						)
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Stochastic Gradient Descent training failed: {ex}' )
		
		with st.expander( 'Instance Models', expanded=False ):
			
			with st.expander( 'k-Nearest Neighbors', expanded=False ):
				knn_defaults = {
						'regression_knn_neighbors': 5,
						'regression_knn_weights': 'uniform',
						'regression_knn_algorithm': 'auto',
						'regression_knn_leaf_size': 30,
						'regression_knn_power': 2.0,
						'regression_knn_metric': 'minkowski',
						'regression_knn_jobs': 1,
						'regression_knn_test_size': 0.20,
						'regression_knn_random_state': 42
				}
				for key, value in knn_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Instance-based regression using nearby observations.' )
				
				knn_c1, knn_c2, knn_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with knn_c1:
					st.markdown( '###### Neighbor Parameters' )
					knn_neighbors = int( st.number_input( 'Neighbors', min_value=1,
						value=int( st.session_state[ 'regression_knn_neighbors' ] ), step=1,
						key='regression_knn_neighbors_input' ) )
					
					knn_weights = st.selectbox( 'Weights', options=[ 'uniform', 'distance' ],
						index=[ 'uniform', 'distance' ].index(
							st.session_state[ 'regression_knn_weights' ] ),
						key='regression_knn_weights_select' )
					
					knn_power = float( st.number_input( 'Power', min_value=1.0,
						value=float( st.session_state[ 'regression_knn_power' ] ), step=1.0,
						format='%.1f', key='regression_knn_power_input' ) )
					
					knn_leaf_size = int( st.number_input( 'Leaf Size', min_value=1,
						value=int( st.session_state[ 'regression_knn_leaf_size' ] ), step=1,
						key='regression_knn_leaf_size_input' ) )
				
				with knn_c2:
					st.markdown( '###### Distance / Search' )
					knn_algorithm = st.selectbox( 'Algorithm', options=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ],
						index=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ].index(
							st.session_state[ 'regression_knn_algorithm' ] ),
						key='regression_knn_algorithm_select' )
					
					knn_metric = st.selectbox( 'Metric',
						options=[
								'minkowski',
								'euclidean',
								'manhattan',
								'chebyshev',
								'canberra',
								'braycurtis',
								'cityblock',
								'cosine',
								'l1',
								'l2',
								'nan_euclidean',
								'hamming'
						],
						index=[
								'minkowski',
								'euclidean',
								'manhattan',
								'chebyshev',
								'canberra',
								'braycurtis',
								'cityblock',
								'cosine',
								'l1',
								'l2',
								'nan_euclidean',
								'hamming'
						].index( st.session_state[ 'regression_knn_metric' ] ),
						key='regression_knn_metric_select' )
					
					knn_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
						value=int( st.session_state[ 'regression_knn_jobs' ] ), step=1,
						key='regression_knn_jobs_input' ) )
					
					if knn_metric != 'minkowski':
						st.caption( 'Power is primarily used with the Minkowski metric.' )
				
				with knn_c3:
					st.markdown( '###### Data Split' )
					knn_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_knn_test_size' ] * 100 ),
						step=5,
						key='regression_knn_test_size_slider' ) / 100.0
					
					knn_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_knn_random_state' ] ), step=1,
						key='regression_knn_random_state_input' ) )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Target: {target_name}' )
				
				knn_btn_1, knn_btn_2 = st.columns( 2 )
				with knn_btn_1:
					train_knn = st.button( '🚆 Train k-Nearest Neighbors',
						key='regression_knn_train', use_container_width=True )
				
				with knn_btn_2:
					reset_knn = st.button( '🔄 Reset k-Nearest Neighbors',
						key='regression_knn_reset', use_container_width=True )
				
				if reset_knn:
					for key, value in knn_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_knn_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_knn:
					try:
						st.session_state[ 'regression_knn_neighbors' ] = int( knn_neighbors )
						st.session_state[ 'regression_knn_weights' ] = str( knn_weights )
						st.session_state[ 'regression_knn_algorithm' ] = str( knn_algorithm )
						st.session_state[ 'regression_knn_leaf_size' ] = int( knn_leaf_size )
						st.session_state[ 'regression_knn_power' ] = float( knn_power )
						st.session_state[ 'regression_knn_metric' ] = str( knn_metric )
						st.session_state[ 'regression_knn_jobs' ] = int( knn_jobs )
						st.session_state[ 'regression_knn_test_size' ] = float( knn_test_size )
						st.session_state[ 'regression_knn_random_state' ] = int( knn_random_state )
						
						df_training = df_model.copy( )
						X = df_training[ active_features ].apply(
							pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric(
							df_training[ target_name ], errors='coerce' ).fillna( 0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ The target must contain at least two distinct values.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						model = regression_model.NearestNeighbor( num=int( knn_neighbors ),
							weight=str( knn_weights ), algo=str( knn_algorithm ),
							leaf=int( knn_leaf_size ), power=float( knn_power ),
							metric=str( knn_metric ), metric_params=None,
							jobs=int( knn_jobs ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( knn_test_size ), random=int( knn_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_knn_elapsed_seconds' ] = elapsed_seconds
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							elif df_scores.shape[ 1 ] == 1:
								df_scores = df_scores.reset_index( )
								df_scores.columns = [ 'Metric', 'Value' ]
							
							df_extra = pd.DataFrame( {
										'Metric': [
												'Processing Time (Seconds)',
												'Training Rows',
												'Testing Rows',
												'Neighbors',
												'Weights',
												'Metric'
										],
										'Value': [
												round( elapsed_seconds, 4 ),
												int( len( X_train ) ),
												int( len( X_test ) ),
												int( knn_neighbors ),
												str( knn_weights ),
												str( knn_metric )
										]
								} )
							
							df_scores = pd.concat( [ df_scores, df_extra ], ignore_index=True )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'k-Nearest Neighbors training failed: {ex}' )
			
			with st.expander( 'Support Vector Machine', expanded=False ):
				svr_defaults = {
						'regression_svr_kernel': 'rbf',
						'regression_svr_degree': 3,
						'regression_svr_gamma_mode': 'scale',
						'regression_svr_gamma_value': 0.100000,
						'regression_svr_coef0': 0.0,
						'regression_svr_tol': 0.001000,
						'regression_svr_c': 1.0,
						'regression_svr_epsilon': 0.1,
						'regression_svr_shrinking': True,
						'regression_svr_cache_size': 200.0,
						'regression_svr_verbose': False,
						'regression_svr_max_iter': -1,
						'regression_svr_test_size': 0.20,
						'regression_svr_random_state': 42
				}
				
				for key, value in svr_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Support Vector Regression for continuous targets.' )
				
				svr_c1, svr_c2, svr_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with svr_c1:
					st.markdown( '###### Kernel Parameters' )
					svr_kernel = st.selectbox( 'Kernel',
						options=[ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ],
						index=[ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ].index(
							st.session_state[ 'regression_svr_kernel' ] ),
						key='regression_svr_kernel_select' )
					
					svr_degree = int( st.number_input( 'Degree', min_value=1,
						value=int( st.session_state[ 'regression_svr_degree' ] ),
						step=1, key='regression_svr_degree_input' ) )
					
					svr_gamma_mode = st.selectbox( 'Gamma',
						options=[ 'scale', 'auto', 'custom' ],
						index=[ 'scale', 'auto', 'custom' ].index(
							st.session_state[ 'regression_svr_gamma_mode' ]
						),
						key='regression_svr_gamma_mode_select' )
					
					svr_gamma_value = float( st.number_input( 'Gamma Value', min_value=0.000001,
						value=float( st.session_state[ 'regression_svr_gamma_value' ] ),
						step=0.010000, format='%.6f', key='regression_svr_gamma_value_input' ) )
					
					svr_coef0 = float( st.number_input( 'Coef0',
						value=float( st.session_state[ 'regression_svr_coef0' ] ),
						step=0.100000, format='%.6f', key='regression_svr_coef0_input' ) )
				
				with svr_c2:
					st.markdown( '###### Regularization / Solver' )
					
					svr_c = float( st.number_input( 'C', min_value=0.000001,
						value=float( st.session_state[ 'regression_svr_c' ] ),
						step=0.100000, format='%.6f', key='regression_svr_c_input' ) )
					
					svr_epsilon = float( st.number_input( 'Epsilon', min_value=0.000001,
						value=float( st.session_state[ 'regression_svr_epsilon' ] ),
						step=0.010000, format='%.6f', key='regression_svr_epsilon_input' ) )
					
					svr_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_svr_tol' ] ),
						step=0.000100, format='%.6f', key='regression_svr_tol_input' ) )
					
					svr_shrinking = st.checkbox( 'Shrinking Heuristic',
						value=bool( st.session_state[ 'regression_svr_shrinking' ] ),
						key='regression_svr_shrinking_check' )
					
					svr_cache_size = float( st.number_input( 'Cache Size (MB)', min_value=1.0,
							value=float( st.session_state[ 'regression_svr_cache_size' ] ),
							step=10.0, format='%.1f',
							key='regression_svr_cache_size_input' ) )
					
					svr_verbose = st.checkbox( 'Verbose',
						value=bool( st.session_state[ 'regression_svr_verbose' ] ),
						key='regression_svr_verbose_check' )
					
					svr_max_iter = int( st.number_input( 'Max Iterations (-1 = No Limit)',
							value=int( st.session_state[ 'regression_svr_max_iter' ] ),
							step=1, key='regression_svr_max_iter_input' ) )
				
				with svr_c3:
					st.markdown( '###### Data Split' )
					
					svr_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_svr_test_size' ] * 100 ),
						step=5, key='regression_svr_test_size_slider' ) / 100.0
					
					svr_random_state = int( st.number_input( 'Random State', min_value=0,
							value=int( st.session_state[ 'regression_svr_random_state' ] ),
							step=1, key='regression_svr_random_state_input' ) )
					
					if svr_kernel != 'poly':
						st.caption( 'Degree is only used when kernel = poly.' )
					
					if svr_kernel not in [ 'poly', 'sigmoid' ]:
						st.caption( 'Coef0 is mainly used by poly and sigmoid kernels.' )
					
					if svr_gamma_mode != 'custom':
						st.caption( 'Gamma Value is only used when Gamma = custom.' )
				
				svr_btn_1, svr_btn_2 = st.columns( 2 )
				with svr_btn_1:
					train_svr = st.button( '🚆 Train Support Vector', key='regression_svr_train',
						use_container_width=True )
				
				with svr_btn_2:
					reset_svr = st.button( '🔄 Reset Support Vector',
						key='regression_svr_reset', use_container_width=True )
				
				if reset_svr:
					for key, value in svr_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_svr_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_svr:
					try:
						st.session_state[ 'regression_svr_kernel' ] = str( svr_kernel )
						st.session_state[ 'regression_svr_degree' ] = int( svr_degree )
						st.session_state[ 'regression_svr_gamma_mode' ] = str( svr_gamma_mode )
						st.session_state[ 'regression_svr_gamma_value' ] = float( svr_gamma_value )
						st.session_state[ 'regression_svr_coef0' ] = float( svr_coef0 )
						st.session_state[ 'regression_svr_tol' ] = float( svr_tol )
						st.session_state[ 'regression_svr_c' ] = float( svr_c )
						st.session_state[ 'regression_svr_epsilon' ] = float( svr_epsilon )
						st.session_state[ 'regression_svr_shrinking' ] = bool( svr_shrinking )
						st.session_state[ 'regression_svr_cache_size' ] = float( svr_cache_size )
						st.session_state[ 'regression_svr_verbose' ] = bool( svr_verbose )
						st.session_state[ 'regression_svr_max_iter' ] = int( svr_max_iter )
						st.session_state[ 'regression_svr_test_size' ] = float( svr_test_size )
						st.session_state[ 'regression_svr_random_state' ] = int( svr_random_state )
						
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply(
							pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric(
							df_training[ target_name ], errors='coerce' ).fillna(
							0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct values.'
							)
							st.stop( )
						
						effective_gamma = ( float( svr_gamma_value )
								if svr_gamma_mode == 'custom'
								else str( svr_gamma_mode ) )
						
						start_time = time.perf_counter( )
						
						model = regression_model.SupportVector( kernel=str( svr_kernel ),
							degree=int( svr_degree ), gamma=effective_gamma,
							coef0=float( svr_coef0 ), tol=float( svr_tol ),
							penalty=float( svr_c ), epsilon=float( svr_epsilon ),
							shrinking=bool( svr_shrinking ), cache=float( svr_cache_size ),
							verbose=bool( svr_verbose ), iters=int( svr_max_iter ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( svr_test_size ), random=int( svr_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_svr_elapsed_seconds' ] = elapsed_seconds
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							elif df_scores.shape[ 1 ] == 1:
								df_scores = df_scores.reset_index( )
								df_scores.columns = [ 'Metric', 'Value' ]
							
							df_extra = pd.DataFrame( {
										'Metric': [
												'Processing Time (Seconds)',
												'Training Rows',
												'Testing Rows',
												'Kernel',
												'C',
												'Epsilon'
										],
										'Value': [
												round( elapsed_seconds, 4 ),
												int( len( X_train ) ),
												int( len( X_test ) ),
												str( svr_kernel ),
												float( svr_c ),
												float( svr_epsilon )
										]
								} )
							
							df_scores = pd.concat( [ df_scores, df_extra ], ignore_index=True )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Support Vector training failed: {ex}' )
		
		with st.expander( 'Tree Models', expander=True ):
			
			with st.expander( 'Extra Trees Regressor', expanded=False ):
				extra_defaults = {
						'regression_extra_estimators': 100,
						'regression_extra_criterion': 'squared_error',
						'regression_extra_max_depth_mode': 'none',
						'regression_extra_max_depth_value': 10,
						'regression_extra_max_features_mode': 'all',
						'regression_extra_max_features_value': 1.0,
						'regression_extra_bootstrap': False,
						'regression_extra_oob_score': False,
						'regression_extra_warm_start': False,
						'regression_extra_jobs': 1,
						'regression_extra_verbose': 0,
						'regression_extra_test_size': 0.20,
						'regression_extra_random_state': 42 }
				
				for key, value in extra_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Highly randomized tree ensemble for continuous targets.' )
				
				extra_c1, extra_c2, extra_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with extra_c1:
					st.markdown( '###### Forest Parameters' )
					extra_estimators = int( st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'regression_extra_estimators' ] ),
						step=1, key='regression_extra_estimators_input' ) )
					
					extra_criterion = st.selectbox( 'Criterion',
						options=[ 'squared_error', 'absolute_error', 'friedman_mse', 'poisson' ],
						index=[ 'squared_error', 'absolute_error', 'friedman_mse', 'poisson' ].index(
							st.session_state[ 'regression_extra_criterion' ] ),
						key='regression_extra_criterion_select' )
					
					extra_max_depth_mode = st.selectbox( 'Max Depth', options=[ 'none', 'custom' ],
						index=[ 'none', 'custom' ].index(
							st.session_state[ 'regression_extra_max_depth_mode' ] ),
						key='regression_extra_max_depth_mode_select' )
					
					extra_max_depth_value = int( st.number_input( 'Max Depth Value', min_value=1,
						value=int( st.session_state[ 'regression_extra_max_depth_value' ] ),
						step=1, key='regression_extra_max_depth_value_input' ) )
				
				with extra_c2:
					st.markdown( '###### Feature / Run Settings' )
					extra_max_features_mode = st.selectbox( 'Max Features',
						options=[ 'all', 'sqrt', 'log2', 'fraction' ],
						index=[ 'all', 'sqrt', 'log2', 'fraction' ].index(
							st.session_state[ 'regression_extra_max_features_mode' ]
						), key='regression_extra_max_features_mode_select' )
					
					extra_max_features_value = float( st.slider( 'Max Features Fraction',
						min_value=0.10, max_value=1.00,
						value=float( st.session_state[ 'regression_extra_max_features_value' ] ),
						step=0.05, key='regression_extra_max_features_value_slider' ) )
					
					extra_bootstrap = st.checkbox( 'Bootstrap Samples',
						value=bool( st.session_state[ 'regression_extra_bootstrap' ] ),
						key='regression_extra_bootstrap_check' )
					
					extra_oob_score = st.checkbox( 'Out-of-Bag Score',
						value=bool( st.session_state[ 'regression_extra_oob_score' ] ),
						key='regression_extra_oob_score_check' )
					
					extra_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_extra_warm_start' ] ),
						key='regression_extra_warm_start_check' )
					
					extra_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
							value=int( st.session_state[ 'regression_extra_jobs' ] ), step=1,
							key='regression_extra_jobs_input' ) )
					
					extra_verbose = int( st.number_input( 'Verbose', min_value=0,
							value=int( st.session_state[ 'regression_extra_verbose' ] ), step=1,
							key='regression_extra_verbose_input' ) )
				
				with extra_c3:
					st.markdown( '###### Data Split' )
					extra_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_extra_test_size' ] * 100 ),
						step=5, key='regression_extra_test_size_slider' ) / 100.0
					
					extra_random_state = int( st.number_input( 'Random State', min_value=0,
							value=int( st.session_state[ 'regression_extra_random_state' ] ),
							step=1, key='regression_extra_random_state_input' ) )
					
					if extra_oob_score and not extra_bootstrap:
						st.info( 'Out-of-bag scoring requires Bootstrap Samples.' )
					
					if extra_max_features_mode != 'fraction':
						st.caption( 'Max Features Fraction is only used when Max Features = fraction.' )
					
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Target: {target_name}' )
				
				extra_btn_1, extra_btn_2 = st.columns( 2 )
				
				with extra_btn_1:
					train_extra = st.button( '🚆 Train Extra Trees Regressor',
						key='regression_extra_train', use_container_width=True )
				
				with extra_btn_2:
					reset_extra = st.button( '🔄 Reset Extra Trees Regressor',
						key='regression_extra_reset', use_container_width=True )
				
				if reset_extra:
					for key, value in extra_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_extra_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_extra:
					try:
						st.session_state[ 'regression_extra_estimators' ] = int( extra_estimators )
						st.session_state[ 'regression_extra_criterion' ] = str( extra_criterion )
						st.session_state[ 'regression_extra_max_depth_mode' ] = str(
							extra_max_depth_mode )
						
						st.session_state[ 'regression_extra_max_depth_value' ] = int(
							extra_max_depth_value )
						
						st.session_state[ 'regression_extra_max_features_mode' ] = str(
							extra_max_features_mode )
						
						st.session_state[ 'regression_extra_max_features_value' ] = float(
							extra_max_features_value )
						
						st.session_state[ 'regression_extra_bootstrap' ] = bool( extra_bootstrap )
						st.session_state[ 'regression_extra_oob_score' ] = bool( extra_oob_score )
						st.session_state[ 'regression_extra_warm_start' ] = bool( extra_warm_start )
						st.session_state[ 'regression_extra_jobs' ] = int( extra_jobs )
						st.session_state[ 'regression_extra_verbose' ] = int( extra_verbose )
						st.session_state[ 'regression_extra_test_size' ] = float( extra_test_size )
						st.session_state[ 'regression_extra_random_state' ] = int( extra_random_state )
						
						df_training = df_model.copy( )
						X = df_training[ active_features ].apply(
							pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric(
							df_training[ target_name ], errors='coerce' ).fillna(
							0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ The target must contain at least two distinct values.' )
							st.stop( )
						
						effective_depth = ( None if extra_max_depth_mode == 'none'
								else int( extra_max_depth_value ) )
						
						if extra_max_features_mode == 'all':
							effective_features = 1.0
						elif extra_max_features_mode == 'sqrt':
							effective_features = 'sqrt'
						elif extra_max_features_mode == 'log2':
							effective_features = 'log2'
						else:
							effective_features = float( extra_max_features_value )
						
						effective_oob = bool( extra_oob_score ) and bool( extra_bootstrap )
						start_time = time.perf_counter( )
						
						model = regression_model.ExtraTreesModel( estimators=int( extra_estimators ),
							criterion=str( extra_criterion ), depth=effective_depth,
							features=effective_features, bootstrap=bool( extra_bootstrap ),
							oob_score=bool( effective_oob ), warm=bool( extra_warm_start ),
							jobs=int( extra_jobs ), rando=int( extra_random_state ),
							verbose=int( extra_verbose ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( extra_test_size ), random=int( extra_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_extra_elapsed_seconds' ] = elapsed_seconds
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							
							df_extra = pd.DataFrame( {
										'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
												'Testing Rows', 'Estimators', 'Criterion',
												'Max Depth', 'Max Features', 'Bootstrap' ],
										'Value': [ round( elapsed_seconds, 4 ),
												int( len( X_train ) ), int( len( X_test ) ),
												int( extra_estimators ), str( extra_criterion ),
												'None' if effective_depth is None else int( effective_depth ),
												str( effective_features ), bool( extra_bootstrap ) ]
								} )
							
							if effective_oob and hasattr( model.model, 'oob_score_' ):
								df_extra = pd.concat( [ df_extra, pd.DataFrame( {
														'Metric': [ 'OOB Score' ],
														'Value': [ float( model.model.oob_score_ ) ]
												} ) ],
									ignore_index=True )
							
							df_scores = pd.concat( [ df_scores, df_extra ], ignore_index=True )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Extra Trees Regressor training failed: {ex}' )
					
			with st.expander( 'Random Forest', expanded=False ):
				rf_defaults = {
						'regression_rf_estimators': 100,
						'regression_rf_criterion': 'squared_error',
						'regression_rf_max_depth_mode': 'none',
						'regression_rf_max_depth_value': 10,
						'regression_rf_min_samples_split': 2,
						'regression_rf_min_samples_leaf': 1,
						'regression_rf_min_weight_fraction_leaf': 0.0,
						'regression_rf_max_features_mode': 'all',
						'regression_rf_max_features_value': 1.0,
						'regression_rf_max_leaf_nodes_mode': 'none',
						'regression_rf_max_leaf_nodes_value': 31,
						'regression_rf_min_impurity_decrease': 0.0,
						'regression_rf_bootstrap': True,
						'regression_rf_oob_score': False,
						'regression_rf_jobs': 1,
						'regression_rf_verbose': 0,
						'regression_rf_warm_start': False,
						'regression_rf_ccp_alpha': 0.0,
						'regression_rf_max_samples_mode': 'all',
						'regression_rf_max_samples_value': 1.0,
						'regression_rf_test_size': 0.20,
						'regression_rf_random_state': 42
				}
				
				for key, value in rf_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Bootstrap random forest for continuous targets.' )
				rf_c1, rf_c2, rf_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with rf_c1:
					st.markdown( '###### Forest Parameters' )
					rf_estimators = int( st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'regression_rf_estimators' ] ),
						step=1, key='regression_rf_estimators_input' ) )
					
					rf_criterion = st.selectbox( 'Criterion',
						options=[ 'squared_error', 'absolute_error', 'friedman_mse', 'poisson' ],
						index=[ 'squared_error', 'absolute_error', 'friedman_mse', 'poisson' ].index(
							st.session_state[ 'regression_rf_criterion' ] ),
						key='regression_rf_criterion_select' )
					
					rf_max_depth_mode = st.selectbox( 'Max Depth', options=[ 'none', 'custom' ],
						index=[ 'none', 'custom' ].index(
							st.session_state[ 'regression_rf_max_depth_mode' ] ),
						key='regression_rf_max_depth_mode_select' )
					
					rf_max_depth_value = int( st.number_input( 'Max Depth Value', min_value=1,
						value=int( st.session_state[ 'regression_rf_max_depth_value' ] ),
						step=1, key='regression_rf_max_depth_value_input' ) )
					
					rf_min_samples_split = int( st.number_input( 'Min Samples Split', min_value=2,
						value=int( st.session_state[ 'regression_rf_min_samples_split' ] ),
						step=1, key='regression_rf_min_samples_split_input' ) )
					
					rf_min_samples_leaf = int( st.number_input( 'Min Samples Leaf', min_value=1,
						value=int( st.session_state[ 'regression_rf_min_samples_leaf' ] ),
						step=1, key='regression_rf_min_samples_leaf_input' ) )
				
				with rf_c2:
					st.markdown( '###### Node / Feature Controls' )
					rf_min_weight_fraction_leaf = float( st.number_input( 'Min Weight Fraction Leaf',
						min_value=0.0,
						value=float(st.session_state[ 'regression_rf_min_weight_fraction_leaf']),
						step=0.010000, format='%.6f',
						key='regression_rf_min_weight_fraction_leaf_input' ) )
					
					rf_max_features_mode = st.selectbox( 'Max Features',
						options=[ 'all', 'sqrt', 'log2', 'fraction' ],
						index=[ 'all', 'sqrt', 'log2', 'fraction' ].index(
							st.session_state[ 'regression_rf_max_features_mode' ] ),
						key='regression_rf_max_features_mode_select' )
					
					rf_max_features_value = float( st.slider( 'Max Features Fraction',
						min_value=0.10, max_value=1.00,
						value=float( st.session_state[ 'regression_rf_max_features_value' ] ),
						step=0.05, key='regression_rf_max_features_value_slider' ) )
					
					rf_max_leaf_nodes_mode = st.selectbox( 'Max Leaf Nodes',
						options=[ 'none', 'custom' ],
						index=[ 'none', 'custom' ].index(
							st.session_state[ 'regression_rf_max_leaf_nodes_mode' ] ),
						key='regression_rf_max_leaf_nodes_mode_select' )
					
					rf_max_leaf_nodes_value = int( st.number_input( 'Max Leaf Nodes Value',
						min_value=2, step=1,
						value=int( st.session_state[ 'regression_rf_max_leaf_nodes_value' ] ),
						key='regression_rf_max_leaf_nodes_value_input' ) )
					
					rf_min_impurity_decrease = float( st.number_input( 'Min Impurity Decrease',
						value=float( st.session_state[ 'regression_rf_min_impurity_decrease' ] ),
						step=0.000100, format='%.6f', min_value=0.0,
						key='regression_rf_min_impurity_decrease_input' ) )
				
				with rf_c3:
					st.markdown( '###### Sampling / Run Configuration' )
					rf_bootstrap = st.checkbox( 'Bootstrap Samples',
						value=bool( st.session_state[ 'regression_rf_bootstrap' ] ),
						key='regression_rf_bootstrap_check' )
					
					rf_oob_score = st.checkbox( 'Out-of-Bag Score',
						value=bool( st.session_state[ 'regression_rf_oob_score' ] ),
						key='regression_rf_oob_score_check' )
					
					rf_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
							value=int( st.session_state[ 'regression_rf_jobs' ] ), step=1,
							key='regression_rf_jobs_input' ) )
					
					rf_verbose = int( st.number_input( 'Verbose', min_value=0,
						value=int( st.session_state[ 'regression_rf_verbose' ] ), step=1,
						key='regression_rf_verbose_input' ) )
					
					rf_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_rf_warm_start' ] ),
						key='regression_rf_warm_start_check' )
					
					rf_ccp_alpha = float( st.number_input( 'CCP Alpha', min_value=0.0,
						value=float( st.session_state[ 'regression_rf_ccp_alpha' ] ),
						step=0.000100, format='%.6f',
						key='regression_rf_ccp_alpha_input' ) )
					
					rf_max_samples_mode = st.selectbox( 'Max Samples',
						options=[ 'all', 'fraction' ],
						index=[ 'all', 'fraction' ].index(
							st.session_state[ 'regression_rf_max_samples_mode' ] ),
						key='regression_rf_max_samples_mode_select' )
					
					rf_max_samples_value = float( st.slider( 'Max Samples Fraction', min_value=0.10,
						max_value=1.00,
						value=float( st.session_state[ 'regression_rf_max_samples_value' ] ),
						step=0.05, key='regression_rf_max_samples_value_slider' ) )
					
					rf_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_rf_test_size' ] * 100 ),
						step=5, key='regression_rf_test_size_slider' ) / 100.0
					
					rf_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_rf_random_state' ] ), step=1,
						key='regression_rf_random_state_input' ) )
					
					if rf_oob_score and not rf_bootstrap:
						st.info( 'Out-of-bag scoring requires Bootstrap Samples.' )
					
					if rf_max_features_mode != 'fraction':
						st.caption( 'Max Features Fraction is only used when Max Features = fraction.' )
					
					if rf_max_samples_mode != 'fraction':
						st.caption( 'Max Samples Fraction is only used when Max Samples = fraction.' )
				
				rf_btn_1, rf_btn_2 = st.columns( 2 )
				with rf_btn_1:
					train_rf = st.button( '🚆 Train Random Forest', key='regression_rf_train',
						use_container_width=True )
				
				with rf_btn_2:
					reset_rf = st.button( '🔄 Reset Random Forest', key='regression_rf_reset',
						use_container_width=True )
				
				if reset_rf:
					for key, value in rf_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_rf_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_rf:
					try:
						st.session_state[ 'regression_rf_estimators' ] = int( rf_estimators )
						st.session_state[ 'regression_rf_criterion' ] = str( rf_criterion )
						st.session_state[ 'regression_rf_max_depth_mode' ] = str(
							rf_max_depth_mode )
						
						st.session_state[ 'regression_rf_max_depth_value' ] = int( rf_max_depth_value )
						
						st.session_state[ 'regression_rf_min_samples_split' ] = int(
							rf_min_samples_split )
						
						st.session_state[ 'regression_rf_min_samples_leaf' ] = int(
							rf_min_samples_leaf )
						
						st.session_state[ 'regression_rf_min_weight_fraction_leaf' ] = float(
							rf_min_weight_fraction_leaf )
						
						st.session_state[ 'regression_rf_max_features_mode' ] = str(
							rf_max_features_mode )
						
						st.session_state[ 'regression_rf_max_features_value' ] = float(
							rf_max_features_value )
						
						st.session_state[ 'regression_rf_max_leaf_nodes_mode' ] = str(
							rf_max_leaf_nodes_mode )
						
						st.session_state[ 'regression_rf_max_leaf_nodes_value' ] = int(
							rf_max_leaf_nodes_value )
						
						st.session_state[ 'regression_rf_min_impurity_decrease' ] = float(
							rf_min_impurity_decrease )
						
						st.session_state[ 'regression_rf_bootstrap' ] = bool( rf_bootstrap )
						st.session_state[ 'regression_rf_oob_score' ] = bool( rf_oob_score )
						st.session_state[ 'regression_rf_jobs' ] = int( rf_jobs )
						st.session_state[ 'regression_rf_verbose' ] = int( rf_verbose )
						st.session_state[ 'regression_rf_warm_start' ] = bool( rf_warm_start )
						st.session_state[ 'regression_rf_ccp_alpha' ] = float( rf_ccp_alpha )
						st.session_state[ 'regression_rf_max_samples_mode' ] = str(
							rf_max_samples_mode )
						
						st.session_state[ 'regression_rf_max_samples_value' ] = float(
							rf_max_samples_value )
						
						st.session_state[ 'regression_rf_test_size' ] = float( rf_test_size )
						st.session_state[ 'regression_rf_random_state' ] = int(
							rf_random_state )
						
						
						df_training = df_model.copy( )
						X = df_training[ active_features ].apply(
							pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric( df_training[ target_name ],
							errors='coerce' ).fillna( 0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ The target must contain at least two distinct values.' )
							st.stop( )
						
						effective_depth = ( None if rf_max_depth_mode == 'none'
						                    else int( rf_max_depth_value ) )
						
						if rf_max_features_mode == 'all':
							effective_features = 1.0
						elif rf_max_features_mode == 'sqrt':
							effective_features = 'sqrt'
						elif rf_max_features_mode == 'log2':
							effective_features = 'log2'
						else:
							effective_features = float( rf_max_features_value )
						
						effective_leaf_nodes = ( None if rf_max_leaf_nodes_mode == 'none'
								else int( rf_max_leaf_nodes_value ) )
						
						effective_samples = ( None if rf_max_samples_mode == 'all'
								else float( rf_max_samples_value ) )
						
						effective_oob = bool( rf_oob_score ) and bool( rf_bootstrap )
						start_time = time.perf_counter( )
						model = regression_model.RandomForest( estimators=int( rf_estimators ),
							criterion=str( rf_criterion ), depth=effective_depth,
							split=int( rf_min_samples_split ), leaf=int( rf_min_samples_leaf ),
							weight_fraction=float( rf_min_weight_fraction_leaf ),
							features=effective_features, leaf_nodes=effective_leaf_nodes,
							impurity=float( rf_min_impurity_decrease ),
							bootstrap=bool( rf_bootstrap ), oob_score=bool( effective_oob ),
							jobs=int( rf_jobs ), rando=int( rf_random_state ),
							verbose=int( rf_verbose ), warm=bool( rf_warm_start ),
							ccp_alpha=float( rf_ccp_alpha ), samples=effective_samples,
							monotonic=None )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( rf_test_size ), random=int( rf_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_rf_elapsed_seconds' ] = elapsed_seconds
						df_scores = model.analyze( X_test, y_test ).copy( )
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							
							df_extra = pd.DataFrame( {
										'Metric': [
												'Processing Time (Seconds)',
												'Training Rows',
												'Testing Rows',
												'Estimators',
												'Criterion',
												'Max Depth',
												'Max Features',
												'Bootstrap'
										],
										'Value': [
												round( elapsed_seconds, 4 ),
												int( len( X_train ) ),
												int( len( X_test ) ),
												int( rf_estimators ),
												str( rf_criterion ),
												'None' if effective_depth is None else int( effective_depth ),
												str( effective_features ),
												bool( rf_bootstrap )
										]
								} )
							
							if effective_oob and hasattr( model.model, 'oob_score_' ):
								df_extra = pd.concat( [ df_extra, pd.DataFrame( {
														'Metric': [ 'OOB Score' ],
														'Value': [ float( model.model.oob_score_ ) ]
												} ) ],
									ignore_index=True )
							
							df_scores = pd.concat( [ df_scores, df_extra ], ignore_index=True )
						
						df_predictions = pd.DataFrame( {
									'Actual': y_test,
									'Predicted': y_prediction
							} )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Random Forest training failed: {ex}' )
		
		with st.expander( 'Ensemble Models', expanded=False ):
			
			with st.expander( 'Adaptive Boosting', expanded=False ):
				ada_defaults = {
						'regression_ada_estimators': 50,
						'regression_ada_learning_rate': 1.0,
						'regression_ada_loss': 'linear',
						'regression_ada_test_size': 0.20,
						'regression_ada_random_state': 42
				}
				
				for key, value in ada_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'AdaBoost ensemble for continuous targets.' )
				
				ada_c1, ada_c2, ada_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ada_c1:
					st.markdown( '###### Ensemble Parameters' )
					
					ada_estimators = int( st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'regression_ada_estimators' ] ),
						step=1, key='regression_ada_estimators_input' ) )
					
					ada_learning_rate = float( st.number_input( 'Learning Rate', min_value=0.000001,
						value=float( st.session_state[ 'regression_ada_learning_rate' ] ),
						step=0.100000, format='%.6f', key='regression_ada_learning_rate_input' ) )
					
					ada_loss = st.selectbox( 'Loss', options=[ 'linear', 'square', 'exponential' ],
						index=[ 'linear', 'square', 'exponential' ].index(
							st.session_state[ 'regression_ada_loss' ] ),
						key='regression_ada_loss_select' )
				
				with ada_c2:
					st.markdown( '###### Data Split' )
					
					ada_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_ada_test_size' ] * 100 ),
						step=5, key='regression_ada_test_size_slider' ) / 100.0
					
					ada_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_ada_random_state' ] ), step=1,
						key='regression_ada_random_state_input' ) )
				
				with ada_c3:
					st.markdown( '###### Context' )
					st.caption( f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Target: {target_name}' )
				
				ada_btn_1, ada_btn_2 = st.columns( 2 )
				with ada_btn_1:
					train_ada = st.button( '🚆 Train Adaptive Boosting',
						key='regression_ada_train', use_container_width=True )
				
				with ada_btn_2:
					reset_ada = st.button( '🔄 Reset Adaptive Boosting', key='regression_ada_reset',
						use_container_width=True )
				
				if reset_ada:
					for key, value in ada_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_ada_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_ada:
					try:
						st.session_state[ 'regression_ada_estimators' ] = int( ada_estimators )
						st.session_state[ 'regression_ada_learning_rate' ] = float(
							ada_learning_rate
						)
						st.session_state[ 'regression_ada_loss' ] = str( ada_loss )
						st.session_state[ 'regression_ada_test_size' ] = float( ada_test_size )
						st.session_state[ 'regression_ada_random_state' ] = int(
							ada_random_state
						)
						
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply(
							pd.to_numeric,
							errors='coerce'
						).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric(
							df_training[ target_name ],
							errors='coerce'
						).fillna( 0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct values.'
							)
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = regression_model.AdaptiveBoost(
							estimators=int( ada_estimators ),
							rate=float( ada_learning_rate ),
							loss=str( ada_loss ),
							rando=int( ada_random_state )
						)
						
						X_train, X_test, y_train, y_test = model.split_data(
							X,
							y,
							size=float( ada_test_size ),
							random=int( ada_random_state )
						)
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_ada_elapsed_seconds' ] = elapsed_seconds
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							elif df_scores.shape[ 1 ] == 1:
								df_scores = df_scores.reset_index( )
								df_scores.columns = [ 'Metric', 'Value' ]
							
							df_extra = pd.DataFrame(
								{
										'Metric': [
												'Processing Time (Seconds)',
												'Training Rows',
												'Testing Rows',
												'Estimators',
												'Learning Rate',
												'Loss'
										],
										'Value': [
												round( elapsed_seconds, 4 ),
												int( len( X_train ) ),
												int( len( X_test ) ),
												int( ada_estimators ),
												float( ada_learning_rate ),
												str( ada_loss )
										]
								}
							)
							
							df_scores = pd.concat(
								[ df_scores, df_extra ],
								ignore_index=True
							)
						
						df_predictions = pd.DataFrame(
							{
									'Actual': y_test,
									'Predicted': y_prediction
							}
						)
						
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Adaptive Boosting training failed: {ex}' )
					
			with st.expander( 'Gradient Boosting', expanded=False ):
				gb_defaults = {
						'regression_gb_loss': 'squared_error',
						'regression_gb_learning_rate': 0.100000,
						'regression_gb_estimators': 100,
						'regression_gb_subsample': 1.0,
						'regression_gb_criterion': 'friedman_mse',
						'regression_gb_min_samples_split': 2,
						'regression_gb_min_samples_leaf': 1,
						'regression_gb_min_weight_fraction_leaf': 0.0,
						'regression_gb_max_depth_mode': 'custom',
						'regression_gb_max_depth_value': 3,
						'regression_gb_min_impurity_decrease': 0.0,
						'regression_gb_max_features_mode': 'none',
						'regression_gb_max_features_value': 1.0,
						'regression_gb_alpha': 0.9,
						'regression_gb_verbose': 0,
						'regression_gb_max_leaf_nodes_mode': 'none',
						'regression_gb_max_leaf_nodes_value': 31,
						'regression_gb_warm_start': False,
						'regression_gb_validation_fraction': 0.1,
						'regression_gb_n_iter_no_change_mode': 'none',
						'regression_gb_n_iter_no_change_value': 5,
						'regression_gb_tol': 0.000100,
						'regression_gb_ccp_alpha': 0.0,
						'regression_gb_test_size': 0.20,
						'regression_gb_random_state': 42
				}
				
				for key, value in gb_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Sequential tree boosting for continuous targets.' )
				
				gb_c1, gb_c2, gb_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with gb_c1:
					st.markdown( '###### Boosting Parameters' )
					
					gb_loss = st.selectbox(
						'Loss',
						options=[ 'squared_error', 'absolute_error', 'huber', 'quantile' ],
						index=[ 'squared_error', 'absolute_error', 'huber', 'quantile' ].index(
							st.session_state[ 'regression_gb_loss' ]
						),
						key='regression_gb_loss_select'
					)
					
					gb_learning_rate = float(
						st.number_input(
							'Learning Rate',
							min_value=0.000001,
							value=float( st.session_state[ 'regression_gb_learning_rate' ] ),
							step=0.010000,
							format='%.6f',
							key='regression_gb_learning_rate_input'
						)
					)
					
					gb_estimators = int(
						st.number_input(
							'Estimators',
							min_value=1,
							value=int( st.session_state[ 'regression_gb_estimators' ] ),
							step=1,
							key='regression_gb_estimators_input'
						)
					)
					
					gb_subsample = float(
						st.slider(
							'Subsample',
							min_value=0.10,
							max_value=1.00,
							value=float( st.session_state[ 'regression_gb_subsample' ] ),
							step=0.05,
							key='regression_gb_subsample_slider'
						)
					)
					
					gb_alpha = float(
						st.slider(
							'Alpha',
							min_value=0.01,
							max_value=0.99,
							value=float( st.session_state[ 'regression_gb_alpha' ] ),
							step=0.01,
							key='regression_gb_alpha_slider'
						)
					)
				
				with gb_c2:
					st.markdown( '###### Tree Parameters' )
					
					gb_criterion = st.selectbox(
						'Criterion',
						options=[ 'friedman_mse', 'squared_error' ],
						index=[ 'friedman_mse', 'squared_error' ].index(
							st.session_state[ 'regression_gb_criterion' ]
						),
						key='regression_gb_criterion_select'
					)
					
					gb_min_samples_split = int(
						st.number_input(
							'Min Samples Split',
							min_value=2,
							value=int( st.session_state[ 'regression_gb_min_samples_split' ] ),
							step=1,
							key='regression_gb_min_samples_split_input'
						)
					)
					
					gb_min_samples_leaf = int(
						st.number_input(
							'Min Samples Leaf',
							min_value=1,
							value=int( st.session_state[ 'regression_gb_min_samples_leaf' ] ),
							step=1,
							key='regression_gb_min_samples_leaf_input'
						)
					)
					
					gb_min_weight_fraction_leaf = float(
						st.number_input(
							'Min Weight Fraction Leaf',
							min_value=0.0,
							value=float(
								st.session_state[ 'regression_gb_min_weight_fraction_leaf' ]
							),
							step=0.010000,
							format='%.6f',
							key='regression_gb_min_weight_fraction_leaf_input'
						)
					)
					
					gb_max_depth_mode = st.selectbox(
						'Max Depth',
						options=[ 'none', 'custom' ],
						index=[ 'none', 'custom' ].index(
							st.session_state[ 'regression_gb_max_depth_mode' ]
						),
						key='regression_gb_max_depth_mode_select'
					)
					
					gb_max_depth_value = int(
						st.number_input(
							'Max Depth Value',
							min_value=1,
							value=int( st.session_state[ 'regression_gb_max_depth_value' ] ),
							step=1,
							key='regression_gb_max_depth_value_input'
						)
					)
					
					gb_min_impurity_decrease = float(
						st.number_input(
							'Min Impurity Decrease',
							min_value=0.0,
							value=float(
								st.session_state[ 'regression_gb_min_impurity_decrease' ]
							),
							step=0.000100,
							format='%.6f',
							key='regression_gb_min_impurity_decrease_input'
						)
					)
				
				with gb_c3:
					st.markdown( '###### Feature / Run Controls' )
					
					gb_max_features_mode = st.selectbox(
						'Max Features',
						options=[ 'none', 'sqrt', 'log2', 'fraction' ],
						index=[ 'none', 'sqrt', 'log2', 'fraction' ].index(
							st.session_state[ 'regression_gb_max_features_mode' ]
						),
						key='regression_gb_max_features_mode_select'
					)
					
					gb_max_features_value = float(
						st.slider(
							'Max Features Fraction',
							min_value=0.10,
							max_value=1.00,
							value=float( st.session_state[ 'regression_gb_max_features_value' ] ),
							step=0.05,
							key='regression_gb_max_features_value_slider'
						)
					)
					
					gb_max_leaf_nodes_mode = st.selectbox(
						'Max Leaf Nodes',
						options=[ 'none', 'custom' ],
						index=[ 'none', 'custom' ].index(
							st.session_state[ 'regression_gb_max_leaf_nodes_mode' ]
						),
						key='regression_gb_max_leaf_nodes_mode_select'
					)
					
					gb_max_leaf_nodes_value = int(
						st.number_input(
							'Max Leaf Nodes Value',
							min_value=2,
							value=int( st.session_state[ 'regression_gb_max_leaf_nodes_value' ] ),
							step=1,
							key='regression_gb_max_leaf_nodes_value_input'
						)
					)
					
					gb_warm_start = st.checkbox(
						'Warm Start',
						value=bool( st.session_state[ 'regression_gb_warm_start' ] ),
						key='regression_gb_warm_start_check'
					)
					
					gb_validation_fraction = float(
						st.slider(
							'Validation Fraction',
							min_value=0.05,
							max_value=0.40,
							value=float(
								st.session_state[ 'regression_gb_validation_fraction' ]
							),
							step=0.05,
							key='regression_gb_validation_fraction_slider'
						)
					)
					
					gb_n_iter_no_change_mode = st.selectbox(
						'N Iter No Change',
						options=[ 'none', 'custom' ],
						index=[ 'none', 'custom' ].index(
							st.session_state[ 'regression_gb_n_iter_no_change_mode' ]
						),
						key='regression_gb_n_iter_no_change_mode_select'
					)
					
					gb_n_iter_no_change_value = int(
						st.number_input(
							'N Iter No Change Value',
							min_value=1,
							value=int(
								st.session_state[ 'regression_gb_n_iter_no_change_value' ]
							),
							step=1,
							key='regression_gb_n_iter_no_change_value_input'
						)
					)
					
					gb_tol = float(
						st.number_input(
							'Tolerance',
							min_value=0.0,
							value=float( st.session_state[ 'regression_gb_tol' ] ),
							step=0.000100,
							format='%.6f',
							key='regression_gb_tol_input'
						)
					)
					
					gb_ccp_alpha = float(
						st.number_input(
							'CCP Alpha',
							min_value=0.0,
							value=float( st.session_state[ 'regression_gb_ccp_alpha' ] ),
							step=0.000100,
							format='%.6f',
							key='regression_gb_ccp_alpha_input'
						)
					)
					
					gb_verbose = int(
						st.number_input(
							'Verbose',
							min_value=0,
							value=int( st.session_state[ 'regression_gb_verbose' ] ),
							step=1,
							key='regression_gb_verbose_input'
						)
					)
					
					gb_test_size = st.slider(
						'Test Set Size (%)',
						min_value=10,
						max_value=40,
						value=int( st.session_state[ 'regression_gb_test_size' ] * 100 ),
						step=5,
						key='regression_gb_test_size_slider'
					) / 100.0
					
					gb_random_state = int(
						st.number_input(
							'Random State',
							min_value=0,
							value=int( st.session_state[ 'regression_gb_random_state' ] ),
							step=1,
							key='regression_gb_random_state_input'
						)
					)
					
					if gb_loss not in [ 'huber', 'quantile' ]:
						st.caption( 'Alpha is only used by huber and quantile losses.' )
					
					if gb_max_features_mode != 'fraction':
						st.caption( 'Max Features Fraction is only used when Max Features = fraction.' )
				
				gb_btn_1, gb_btn_2 = st.columns( 2 )
				
				with gb_btn_1:
					train_gb = st.button(
						'🚆 Train Gradient Boosting',
						key='regression_gb_train',
						use_container_width=True
					)
				
				with gb_btn_2:
					reset_gb = st.button(
						'🔄 Reset Gradient Boosting',
						key='regression_gb_reset',
						use_container_width=True
					)
				
				if reset_gb:
					for key, value in gb_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_gb_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_gb:
					try:
						st.session_state[ 'regression_gb_loss' ] = str( gb_loss )
						st.session_state[ 'regression_gb_learning_rate' ] = float(
							gb_learning_rate
						)
						st.session_state[ 'regression_gb_estimators' ] = int( gb_estimators )
						st.session_state[ 'regression_gb_subsample' ] = float( gb_subsample )
						st.session_state[ 'regression_gb_criterion' ] = str( gb_criterion )
						st.session_state[ 'regression_gb_min_samples_split' ] = int(
							gb_min_samples_split
						)
						st.session_state[ 'regression_gb_min_samples_leaf' ] = int(
							gb_min_samples_leaf
						)
						st.session_state[ 'regression_gb_min_weight_fraction_leaf' ] = float(
							gb_min_weight_fraction_leaf
						)
						st.session_state[ 'regression_gb_max_depth_mode' ] = str(
							gb_max_depth_mode
						)
						st.session_state[ 'regression_gb_max_depth_value' ] = int(
							gb_max_depth_value
						)
						st.session_state[ 'regression_gb_min_impurity_decrease' ] = float(
							gb_min_impurity_decrease
						)
						st.session_state[ 'regression_gb_max_features_mode' ] = str(
							gb_max_features_mode
						)
						st.session_state[ 'regression_gb_max_features_value' ] = float(
							gb_max_features_value
						)
						st.session_state[ 'regression_gb_alpha' ] = float( gb_alpha )
						st.session_state[ 'regression_gb_verbose' ] = int( gb_verbose )
						st.session_state[ 'regression_gb_max_leaf_nodes_mode' ] = str(
							gb_max_leaf_nodes_mode
						)
						st.session_state[ 'regression_gb_max_leaf_nodes_value' ] = int(
							gb_max_leaf_nodes_value
						)
						st.session_state[ 'regression_gb_warm_start' ] = bool( gb_warm_start )
						st.session_state[ 'regression_gb_validation_fraction' ] = float(
							gb_validation_fraction
						)
						st.session_state[ 'regression_gb_n_iter_no_change_mode' ] = str(
							gb_n_iter_no_change_mode
						)
						st.session_state[ 'regression_gb_n_iter_no_change_value' ] = int(
							gb_n_iter_no_change_value
						)
						st.session_state[ 'regression_gb_tol' ] = float( gb_tol )
						st.session_state[ 'regression_gb_ccp_alpha' ] = float( gb_ccp_alpha )
						st.session_state[ 'regression_gb_test_size' ] = float( gb_test_size )
						st.session_state[ 'regression_gb_random_state' ] = int(
							gb_random_state
						)
						
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply(
							pd.to_numeric,
							errors='coerce'
						).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric(
							df_training[ target_name ],
							errors='coerce'
						).fillna( 0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct values.'
							)
							st.stop( )
						
						effective_depth = (
								None
								if gb_max_depth_mode == 'none'
								else int( gb_max_depth_value )
						)
						
						if gb_max_features_mode == 'none':
							effective_features = None
						elif gb_max_features_mode == 'sqrt':
							effective_features = 'sqrt'
						elif gb_max_features_mode == 'log2':
							effective_features = 'log2'
						else:
							effective_features = float( gb_max_features_value )
						
						effective_leaf_nodes = (
								None
								if gb_max_leaf_nodes_mode == 'none'
								else int( gb_max_leaf_nodes_value )
						)
						
						effective_no_change = (
								None
								if gb_n_iter_no_change_mode == 'none'
								else int( gb_n_iter_no_change_value )
						)
						
						start_time = time.perf_counter( )
						
						model = regression_model.GradientBoost(
							loss=str( gb_loss ),
							rate=float( gb_learning_rate ),
							estimators=int( gb_estimators ),
							subsample=float( gb_subsample ),
							criterion=str( gb_criterion ),
							split=int( gb_min_samples_split ),
							leaf=int( gb_min_samples_leaf ),
							weight_fraction=float( gb_min_weight_fraction_leaf ),
							depth=effective_depth,
							impurity=float( gb_min_impurity_decrease ),
							init=None,
							rando=int( gb_random_state ),
							features=effective_features,
							alpha=float( gb_alpha ),
							verbose=int( gb_verbose ),
							leaf_nodes=effective_leaf_nodes,
							warm=bool( gb_warm_start ),
							validation_fraction=float( gb_validation_fraction ),
							no_change=effective_no_change,
							tol=float( gb_tol ),
							ccp_alpha=float( gb_ccp_alpha )
						)
						
						X_train, X_test, y_train, y_test = model.split_data(
							X,
							y,
							size=float( gb_test_size ),
							random=int( gb_random_state )
						)
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_gb_elapsed_seconds' ] = elapsed_seconds
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							
							df_extra = pd.DataFrame(
								{
										'Metric': [
												'Processing Time (Seconds)',
												'Training Rows',
												'Testing Rows',
												'Estimators',
												'Learning Rate',
												'Loss',
												'Subsample',
												'Criterion'
										],
										'Value': [
												round( elapsed_seconds, 4 ),
												int( len( X_train ) ),
												int( len( X_test ) ),
												int( gb_estimators ),
												float( gb_learning_rate ),
												str( gb_loss ),
												float( gb_subsample ),
												str( gb_criterion )
										]
								}
							)
							
							df_scores = pd.concat(
								[ df_scores, df_extra ],
								ignore_index=True
							)
						
						df_predictions = pd.DataFrame(
							{
									'Actual': y_test,
									'Predicted': y_prediction
							}
						)
						
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Gradient Boosting training failed: {ex}' )
					
			with st.expander( 'Voting Regressor', expanded=False ):
				vote_defaults = {
						'regression_vote_include_ols': True,
						'regression_vote_include_ridge': True,
						'regression_vote_include_knn': True,
						'regression_vote_use_weights': False,
						'regression_vote_weight_ols': 1.0,
						'regression_vote_weight_ridge': 1.0,
						'regression_vote_weight_knn': 1.0,
						'regression_vote_jobs': 1,
						'regression_vote_verbose': False,
						'regression_vote_test_size': 0.20,
						'regression_vote_random_state': 42
				}
				
				for key, value in vote_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Average predictions from multiple regressors fit on the full dataset.' )
				
				vote_c1, vote_c2, vote_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with vote_c1:
					st.markdown( '###### Base Estimators' )
					
					vote_include_ols = st.checkbox( 'Ordinary Least Squares',
						value=bool( st.session_state[ 'regression_vote_include_ols' ] ),
						key='regression_vote_include_ols'
					)
					
					vote_include_ridge = st.checkbox(
						'Ridge Regression',
						value=bool( st.session_state[ 'regression_vote_include_ridge' ] ),
						key='regression_vote_include_ridge'
					)
					
					vote_include_knn = st.checkbox(
						'k-Nearest Neighbors',
						value=bool( st.session_state[ 'regression_vote_include_knn' ] ),
						key='regression_vote_include_knn'
					)
					
					st.caption( 'Select at least two base estimators.' )
				
				with vote_c2:
					st.markdown( '###### Weighting' )
					
					vote_use_weights = st.checkbox(
						'Use Custom Weights',
						value=bool( st.session_state[ 'regression_vote_use_weights' ] ),
						key='regression_vote_use_weights'
					)
					
					vote_weight_ols = float(
						st.number_input(
							'OLS Weight',
							min_value=0.0,
							value=float( st.session_state[ 'regression_vote_weight_ols' ] ),
							step=0.10,
							format='%.2f',
							key='regression_vote_weight_ols'
						)
					)
					
					vote_weight_ridge = float(
						st.number_input(
							'Ridge Weight',
							min_value=0.0,
							value=float( st.session_state[ 'regression_vote_weight_ridge' ] ),
							step=0.10,
							format='%.2f',
							key='regression_vote_weight_ridge'
						)
					)
					
					vote_weight_knn = float(
						st.number_input(
							'kNN Weight',
							min_value=0.0,
							value=float( st.session_state[ 'regression_vote_weight_knn' ] ),
							step=0.10,
							format='%.2f',
							key='regression_vote_weight_knn'
						)
					)
					
					st.caption( 'Weights are only applied when custom weighting is enabled.' )
				
				with vote_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					
					vote_jobs = int(
						st.number_input(
							'Parallel Jobs',
							min_value=1,
							value=int( st.session_state[ 'regression_vote_jobs' ] ),
							step=1,
							key='regression_vote_jobs'
						)
					)
					
					vote_verbose = st.checkbox(
						'Verbose',
						value=bool( st.session_state[ 'regression_vote_verbose' ] ),
						key='regression_vote_verbose'
					)
					
					vote_test_size = st.slider(
						'Test Set Size (%)',
						min_value=10,
						max_value=30,
						value=int( st.session_state[ 'regression_vote_test_size' ] * 100 ),
						step=1,
						key='regression_vote_test_size'
					) / 100.0
					
					vote_random_state = int(
						st.number_input(
							'Random State',
							min_value=0,
							value=int( st.session_state[ 'regression_vote_random_state' ] ),
							step=1,
							key='regression_vote_random_state'
						)
					)
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Target: {target_name}'
					)
				
				vote_btn_1, vote_btn_2 = st.columns( 2 )
				with vote_btn_1:
					train_vote = st.button(
						'🚆 Train Voting Regressor',
						key='regression_vote_train',
						use_container_width=True
					)
				
				with vote_btn_2:
					reset_vote = st.button(
						'🔄 Reset Voting Regressor',
						key='regression_vote_reset',
						use_container_width=True
					)
				
				if reset_vote:
					for key, value in vote_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_vote_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_vote:
					try:
						st.session_state[ 'regression_vote_include_ols' ] = bool(
							vote_include_ols
						)
						st.session_state[ 'regression_vote_include_ridge' ] = bool(
							vote_include_ridge
						)
						st.session_state[ 'regression_vote_include_knn' ] = bool(
							vote_include_knn
						)
						st.session_state[ 'regression_vote_use_weights' ] = bool(
							vote_use_weights
						)
						st.session_state[ 'regression_vote_weight_ols' ] = float(
							vote_weight_ols
						)
						st.session_state[ 'regression_vote_weight_ridge' ] = float(
							vote_weight_ridge
						)
						st.session_state[ 'regression_vote_weight_knn' ] = float(
							vote_weight_knn
						)
						st.session_state[ 'regression_vote_jobs' ] = int( vote_jobs )
						st.session_state[ 'regression_vote_verbose' ] = bool( vote_verbose )
						st.session_state[ 'regression_vote_test_size' ] = float(
							vote_test_size
						)
						st.session_state[ 'regression_vote_random_state' ] = int(
							vote_random_state
						)
						
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply(
							pd.to_numeric,
							errors='coerce'
						).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric(
							df_training[ target_name ],
							errors='coerce'
						).fillna( 0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct values.'
							)
							st.stop( )
						
						estimators = [ ]
						weights = [ ]
						
						if vote_include_ols:
							estimators.append(
								(
										'least_squares',
										regression_model.skl.LinearRegression( )
								)
							)
							weights.append( float( vote_weight_ols ) )
						
						if vote_include_ridge:
							estimators.append(
								(
										'ridge',
										regression_model.skl.Ridge(
											random_state=int( vote_random_state )
										)
								)
							)
							weights.append( float( vote_weight_ridge ) )
						
						if vote_include_knn:
							estimators.append(
								(
										'nearest_neighbor',
										regression_model.skn.KNeighborsRegressor( )
								)
							)
							weights.append( float( vote_weight_knn ) )
						
						if len( estimators ) < 2:
							st.warning( '⚠️ Voting Regressor requires at least two base estimators.' )
							st.stop( )
						
						if vote_use_weights and all( w == 0.0 for w in weights ):
							st.warning( '⚠️ At least one voting weight must be greater than zero.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = regression_model.VotingModel(
							est=estimators,
							weights=weights if vote_use_weights else None,
							jobs=int( vote_jobs ),
							verbose=bool( vote_verbose )
						)
						
						X_train, X_test, y_train, y_test = model.split_data(
							X,
							y,
							size=float( vote_test_size ),
							random=int( vote_random_state )
						)
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_vote_elapsed_seconds' ] = elapsed_seconds
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							df_extra = pd.DataFrame(
								{
										'Metric': [
												'Processing Time (Seconds)',
												'Training Rows',
												'Testing Rows',
												'Estimator Count',
												'Weighted Voting'
										],
										'Value': [
												round( elapsed_seconds, 4 ),
												int( len( X_train ) ),
												int( len( X_test ) ),
												int( len( estimators ) ),
												bool( vote_use_weights )
										]
								}
							)
							
							df_scores = pd.concat(
								[ df_scores, df_extra ],
								ignore_index=True
							)
						
						df_predictions = pd.DataFrame(
							{
									'Actual': y_test,
									'Predicted': y_prediction
							}
						)
						
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Voting Regressor training failed: {ex}' )
					
			with st.expander( 'Stacking Regressor', expanded=False ):
				stack_defaults = {
						'regression_stack_include_ols': True,
						'regression_stack_include_ridge': True,
						'regression_stack_include_knn': True,
						'regression_stack_final_estimator': 'ridge',
						'regression_stack_cv_mode': 'default',
						'regression_stack_cv_value': 5,
						'regression_stack_jobs': 1,
						'regression_stack_passthrough': False,
						'regression_stack_verbose': 0,
						'regression_stack_test_size': 0.20,
						'regression_stack_random_state': 42
				}
			
				for key, value in stack_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Stacked generalization ensemble for continuous targets.' )
				stack_c1, stack_c2, stack_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with stack_c1:
					st.markdown( '###### Base Estimators' )
					
					stack_include_ols = st.checkbox( 'Ordinary Least Squares',
						value=bool( st.session_state[ 'regression_stack_include_ols' ] ),
						key='regression_stack_include_ols' )
					
					stack_include_ridge = st.checkbox( 'Ridge Regression',
						value=bool( st.session_state[ 'regression_stack_include_ridge' ] ),
						key='regression_stack_include_ridge' )
					
					stack_include_knn = st.checkbox( 'k-Nearest Neighbors',
						value=bool( st.session_state[ 'regression_stack_include_knn' ] ),
						key='regression_stack_include_knn' )
					
					st.caption( 'Select at least two base estimators.' )
				
				with stack_c2:
					st.markdown( '###### Meta-Estimator' )
					
					stack_final_estimator = st.selectbox(
						'Final Estimator',
						options=[ 'linear_regression', 'ridge', 'knn' ],
						index=[ 'linear_regression', 'ridge', 'knn' ].index(
							st.session_state[ 'regression_stack_final_estimator' ]
						),
						key='regression_stack_final_estimator'
					)
					
					stack_cv_mode = st.selectbox(
						'Cross-Validation',
						options=[ 'default', 'custom' ],
						index=[ 'default', 'custom' ].index(
							st.session_state[ 'regression_stack_cv_mode' ]
						),
						key='regression_stack_cv_mode'
					)
					
					stack_cv_value = int(
						st.number_input(
							'CV Folds',
							min_value=2,
							value=int( st.session_state[ 'regression_stack_cv_value' ] ),
							step=1,
							key='regression_stack_cv_value'
						)
					)
					
					stack_passthrough = st.checkbox(
						'Passthrough Original Features',
						value=bool( st.session_state[ 'regression_stack_passthrough' ] ),
						key='regression_stack_passthrough'
					)
				
				with stack_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					
					stack_jobs = int(
						st.number_input( 'Parallel Jobs', min_value=1,
							value=int( st.session_state[ 'regression_stack_jobs' ] ),
							step=1, key='regression_stack_jobs' ) )
					
					stack_verbose = int(
						st.number_input( 'Verbose', min_value=0,
							value=int( st.session_state[ 'regression_stack_verbose' ] ),
							step=1, key='regression_stack_verbose' ) )
					
					stack_test_size = st.slider(
						'Test Set Size (%)',
						min_value=10,
						max_value=30,
						value=int( st.session_state[ 'regression_stack_test_size' ] * 100 ),
						step=1,
						key='regression_stack_test_size'
					) / 100.0
					
					stack_random_state = int(
						st.number_input(
							'Random State',
							min_value=0,
							value=int( st.session_state[ 'regression_stack_random_state' ] ),
							step=1,
							key='regression_stack_random_state'
						)
					)
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Target: {target_name}'
					)
				
				stack_btn_1, stack_btn_2 = st.columns( 2 )
				
				with stack_btn_1:
					train_stack = st.button(
						'🚆 Train Stacking Regressor',
						key='regression_stack_train',
						use_container_width=True
					)
				
				with stack_btn_2:
					reset_stack = st.button(
						'🔄 Reset Stacking Regressor',
						key='regression_stack_reset',
						use_container_width=True
					)
				
				if reset_stack:
					for key, value in stack_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_stack_elapsed_seconds' ] = None
					st.rerun( )
				
				if train_stack:
					try:
						st.session_state[ 'regression_stack_include_ols' ] = bool(
							stack_include_ols
						)
						st.session_state[ 'regression_stack_include_ridge' ] = bool(
							stack_include_ridge
						)
						st.session_state[ 'regression_stack_include_knn' ] = bool(
							stack_include_knn
						)
						st.session_state[ 'regression_stack_final_estimator' ] = str(
							stack_final_estimator
						)
						st.session_state[ 'regression_stack_cv_mode' ] = str( stack_cv_mode )
						st.session_state[ 'regression_stack_cv_value' ] = int( stack_cv_value )
						st.session_state[ 'regression_stack_jobs' ] = int( stack_jobs )
						st.session_state[ 'regression_stack_passthrough' ] = bool(
							stack_passthrough
						)
						st.session_state[ 'regression_stack_verbose' ] = int( stack_verbose )
						st.session_state[ 'regression_stack_test_size' ] = float(
							stack_test_size
						)
						st.session_state[ 'regression_stack_random_state' ] = int(
							stack_random_state
						)
						
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply(
							pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric( df_training[ target_name ],
							errors='coerce' ).fillna( 0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct values.'
							)
							st.stop( )
						
						estimators = [ ]
						
						if stack_include_ols:
							estimators.append(
								(
										'least_squares',
										regression_model.skl.LinearRegression( )
								)
							)
						
						if stack_include_ridge:
							estimators.append(
								(
										'ridge',
										regression_model.skl.Ridge(
											random_state=int( stack_random_state )
										)
								)
							)
						
						if stack_include_knn:
							estimators.append(
								(
										'nearest_neighbor',
										regression_model.skn.KNeighborsRegressor( )
								)
							)
						
						if len( estimators ) < 2:
							st.warning(
								'⚠️ Stacking Regressor requires at least two base estimators.'
							)
							st.stop( )
						
						if stack_final_estimator == 'linear_regression':
							final_estimator = regression_model.skl.LinearRegression( )
						elif stack_final_estimator == 'ridge':
							final_estimator = regression_model.skl.Ridge(
								random_state=int( stack_random_state )
							)
						else:
							final_estimator = regression_model.skn.KNeighborsRegressor( )
						
						effective_cv = None if stack_cv_mode == 'default' else int( stack_cv_value )
						
						start_time = time.perf_counter( )
						
						model = regression_model.StackingModel( est=estimators,
							final=final_estimator, cv=effective_cv, jobs=int( stack_jobs ),
							passthrough=bool( stack_passthrough ), verbose=int( stack_verbose ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( stack_test_size ), random=int( stack_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_stack_elapsed_seconds' ] = elapsed_seconds
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							elif df_scores.shape[ 1 ] == 1:
								df_scores = df_scores.reset_index( )
								df_scores.columns = [ 'Metric', 'Value' ]
							
							df_extra = pd.DataFrame( {
										'Metric': [
												'Training Score',
												'Testing Score',
												'Processing Time (Seconds)',
												'Training Rows',
												'Testing Rows',
												'Estimator Count',
												'Passthrough'
										],
										'Value': [
												float( model.training_score ),
												float( model.testing_score ),
												round( elapsed_seconds, 4 ),
												int( len( X_train ) ),
												int( len( X_test ) ),
												int( len( estimators ) ),
												bool( stack_passthrough )
										]
								} )
							
							df_scores = pd.concat(
								[ df_scores, df_extra ],
								ignore_index=True
							)
						
						df_predictions = pd.DataFrame(
							{
									'Actual': y_test,
									'Predicted': y_prediction
							}
						)
						
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Stacking Regressor training failed: {ex}' )
				
		# ------------------------------------------------------------------
		# PREDICTIONS
		# ------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Predictions' )
		y_prediction = model.project( X_test )
		df_predictions = pd.DataFrame( {
					'Observed': y_test,
					'Predicted': y_prediction,
					'Residual': y_test - y_prediction
			} )
		
		st.data_editor( df_predictions, use_container_width=True )
		
		# ------------------------------------------------------------------
		# MODEL DETAILS
		# ------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Model Details' )
		detail_rows = [ ]
		if hasattr( model, 'features' ):
			try:
				detail_rows.append( { 'Property': 'Features', 'Value': model.features } )
			except Exception:
				pass
		
		if hasattr( model, 'training_score' ):
			try:
				detail_rows.append( { 'Property': 'Training Score',
				                      'Value': model.training_score } )
			except Exception:
				pass
		
		if hasattr( model, 'testing_score' ):
			try:
				detail_rows.append( { 'Property': 'Testing Score',
				                      'Value': model.testing_score } )
			except Exception:
				pass
		
		if hasattr( model, 'weights' ):
			try:
				weights = model.weights
				if weights is not None:
					df_weights = pd.DataFrame(
						{
								'Feature': features,
								'Weight': np.asarray( weights ).reshape( -1 )
						} )
					st.caption( 'Coefficients' )
					st.data_editor( df_weights, use_container_width=True )
			except Exception:
				pass
		
		if hasattr( model, 'intercept' ):
			try:
				intercept = model.intercept
				if intercept is not None:
					detail_rows.append( { 'Property': 'Intercept', 'Value': intercept } )
			except Exception:
				pass
		
		if detail_rows:
			df_details = pd.DataFrame( detail_rows )
			st.data_editor( df_details, use_container_width=True )
		else:
			st.info( 'No additional model details are exposed for this regressor.' )
		
		# ------------------------------------------------------------------
		# SCATTER PLOT
		# ------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Observed vs Predicted' )
		plt.close( 'all' )
		model.scatter_plot( X_test, y_test )
		st.pyplot( plt.gcf( ) )
		plt.close( 'all' )


# ============================================
# CLUSTERING MODELS MODE
# ============================================
elif mode == 'Clustering Models':
	df_original = st.session_state.get( 'df_dataset', None )
	df_dataset = st.session_state.get( 'df_dataset', None )
	df_working = st.session_state.get( 'df_working', None )
	df_processed = st.session_state.get( 'df_processed', None )
	df_cluster = st.session_state.get( 'df_cluster', None )
	numeric_columns = st.session_state.get( 'numeric_columns', [ ] )
	categorical_columns = st.session_state.get( 'categorical_columns', [ ] )
	features = st.session_state.get( 'features', [ ] )
	targets = st.session_state.get( 'targets', [ ] )
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Clustering Models' ] )
		st.divider( )
		
		if df_dataset is None or df_dataset.empty:
			st.warning( '⚠️ No dataset loaded.' )
			st.stop( )
		
		df_original = df_dataset.copy( )
		st.session_state[ 'df_original' ] = df_original.copy( )
		numeric_columns = [ c for c in df_original.columns
		                    if pd.api.types.is_numeric_dtype( df_original[ c ] ) ]
		
		categorical_columns = [ c for c in df_original.columns if c not in numeric_columns ]
		
		if not numeric_columns or not categorical_columns:
			st.warning( '⚠️ Clustering requires numeric features' )
			st.stop( )
		
		df_cluster = st.session_state.get( 'df_cluster', df_original.copy( ) ).copy( )
		
		# ======================================================================================
		# Data Selection
		# ======================================================================================
		st.markdown( '##### Feature Selection' )
		st.caption( f'Inputs: {len( df_original ):,} | Features: {len( df_original.columns ):,}' )
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			features = st.multiselect( 'Select Features', options=df_working.columns  )
		
		with col_c2:
			target_options = [ c for c in numeric_columns if c not in features ]
			targets = st.multiselect( 'Select Targets', options=target_options )
		
		sel_b1, sel_b2 = st.columns( [ 0.5, 0.5 ] )
		with sel_b1:
			if st.button( 'Create Working Dataset', icon='➕', key='cluster_create_dataset',
					use_container_width=True ):
				selected_all = features + [ c for c in targets if c not in features ]
				if selected_all:
					df_working = df_original[ selected_all ].copy( )
					st.session_state[ 'df_working' ] = df_working
				else:
					df_working = df_original.copy( )
					st.session_state[ 'df_working' ] = df_working
				
				st.session_state[ 'features' ] = features.copy( )
				st.session_state[ 'targets' ] = targets.copy( )
				st.session_state[ 'df_processed' ] = df_working.copy( )
				df_processed = pd.DataFrame( )
				
				commit_frame( df_working )
				st.success( 'Working Dataset Created!' )
		
		with sel_b2:
			if st.button( 'Reset To Original', icon='🔁', key='cluster_reset_to_original',
					use_container_width=True ):
				df_original = df_dataset.copy( )
				df_working = df_original.copy( )
				df_processed = pd.DataFrame( )
				st.session_state[ 'features' ] = [ ]
				st.session_state[ 'targets' ] = [ ]
				st.session_state[ 'df_working' ] = df_working.copy( )
				st.session_state[ 'df_processed' ] = df_processed.copy( )
				commit_frame( df_working )
				st.success( 'Reset to Original' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Working Data' )
		st.caption( f'Samples: {len( df_working ):,} | Features: {len( df_working.columns ):,}' )
		
		st.data_editor( df_working, key='clusters_working_data' )
		
		# ------------------------------------------------------------------
		# Training Target & Features
		# ------------------------------------------------------------------
		if df_working.empty:
			st.warning( '⚠️ No complete rows remain after preprocessing and target/feature selection.' )
			st.stop( )
		
		y = df_working[ targets ]
		
		if len( np.unique( y ) ) < 2:
			st.warning( '⚠️ Classification requires at least two classes in the selected target.' )
			st.stop( )
		
		# -----------------------------------------------------------------
		# Data Processing
		# -----------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Feature-Engineering' )
		
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			with st.expander( label='Data Scaling', icon='⚖️', key='cluster_scalers' ):
				
				with st.expander( 'Standard Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.STANDARD_SCALER )
					scale_cols = st.multiselect( 'Columns', options=targets,
						key='cluster_standard_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_standard_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = StandardScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'Standard Scaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_standard_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Min-Max Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.MINMAX_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_minmax_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_minmax_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MinMaxScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'Min-Max Scaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='cluster_minmax_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.ROBUST_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_robust_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_robust_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = RobustScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'RobustScaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_robust_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Normal Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.NORMAL_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ],
						index=1, key='cluster_normal_scaler_norm' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_normal_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = NormalScaler( norm=norm )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'NormalScaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_normal_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.MAXABS_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_maxabs_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_maxabs_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MaxAbsScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'MaxAbsScaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_maxabs_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Data Imputation', icon='🧹', key='cluster_imputers' ):
				with st.expander( 'Mean Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.MEAN_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='cluster_mean_imputer_indicator' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_mean_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = MeanImputer( strategy='mean', add_indicator=add_indicator )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'mean_imputer' )
								
								commit_frame( df_processed )
								st.success( 'MeanImputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_mean_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.NEAREST_NEIGHBOR_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1,
						value=5, step=1, key='cluster_nearest_imputer_neighbors' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_nearest_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = NearestImputer( neighbors=int( neighbors ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'nearest_imputer' )
								
								commit_frame( df_processed )
								st.success( 'Nearest Imputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_nearest_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.ITERATIVE_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_iterative_imputer_cols' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1,
						value=10, step=1, key='cluster_iterative_imputer_max_iter' )
					
					random_state = st.number_input( 'Random State', min_value=0,
						value=0, step=1, key='cluster_iterative_imputer_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Iterative Imputer',
								key='cluster_iterative_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = IterativeImputer( max_iter=int( max_iter ),
									random_state=int( random_state ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'iterative_imputer' )
								commit_frame( df_processed )
								st.success( 'Iterative Imputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_iterative_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.SIMPLE_IMPUTER )
					impute_cols = st.multiselect( 'Columns',
						options=numeric_columns,
						key='cluster_simple_imputer_cols' )
					
					strategy = st.selectbox( 'Strategy',
						options=[ 'mean', 'median', 'most_frequent', 'constant' ],
						key='cluster_simple_imputer_strategy' )
					
					fill_value = st.text_input( 'Fill Value', value='0.0',
						key='cluster_simple_imputer_fill_value' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='cluster_simple_imputer_indicator' )
					
					keep_empty_features = st.checkbox( 'Keep Empty Features', value=False,
						key='cluster_simple_imputer_keep_empty' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SimpleImputer', key='cluster_simpleimputer_apply',
								use_container_width=True ):
							if impute_cols:
								if strategy in [ 'mean', 'median' ]:
									df_input = df_processed[ impute_cols ].apply(
										pd.to_numeric, errors='coerce' )
									fill_object: object = 0.0
								elif strategy == 'constant':
									df_input = df_processed[ impute_cols ].copy( )
									fill_object = fill_value
								else:
									df_input = df_processed[ impute_cols ].copy( )
									fill_object = fill_value
								
								imputer = SimpleImputer( strategy=strategy, fill_value=fill_object,
									add_indicator=add_indicator,
									keep_empty_features=keep_empty_features )
								
								result = imputer.train_transform( df_input.to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'simple_imputer' )
								
								commit_frame( df_processed )
								st.success( 'Simple Imputer Applied' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_simple_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Data Encoding', icon='🔣', key='cluster_encoders' ):
				with st.expander( 'One-Hot Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.ONEHOT_ENCODER )
					encode_cols = st.multiselect( 'Columns', options=features,
						key='cluster_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='cluster_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0,
						key='cluster_onehot_unknown' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_onehot_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OneHotEncoder( sparse=bool( sparse ), unknown=unknown )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, encode_cols,
									result, 'onehot' )
								commit_frame( df_processed )
								st.success( 'OneHotEncoder applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_onehot_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.ORDINAL_ENCODER )
					encode_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='cluster_ordinal_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_ordinal_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OrdinalEncoder( )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								df_processed[ encode_cols ] = result
								commit_frame( df_processed )
								st.success( 'Ordinal Encoder Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_ordinal_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Label Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.LABEL_ENCODER )
					target_col = st.selectbox( 'Column', options=categorical_columns,
						key='cluster_label_encoder_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_label_encoder_apply',
								use_container_width=True ):
							if target_col:
								encoder = LabelEncoder( )
								result = encoder.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed[ target_col ] = result
								commit_frame( df_processed )
								st.success( 'Label Encoder Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_label_encoder_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Target Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.TARGET_ENCODER )
					encode_cols = st.multiselect( 'Categorical Feature Columns',
						options=categorical_columns, key='cluster_target_encoder_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_target_encoder_target_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_target_encoder_apply',
								use_container_width=True ):
							if encode_cols and target_col:
								df_processed = df_working.copy( )
								encoder = TargetEncoder( )
								X_enc = df_processed[ encode_cols ].astype( str ).to_numpy( )
								y_enc = df_processed[ target_col ].to_numpy( )
								result = encoder.train_transform( X_enc, y_enc )
								
								df_processed = replace_columns( df_processed, encode_cols, result,
									'target_encoder' )
								
								commit_frame( df_processed )
								st.success( 'Target Encoder Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_target_encoder_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left', help=cfg.POLYNOMIAL_FEATURES )
					poly_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4,
						value=2, key='cluster_polynomial_degree' )
					
					interaction = st.checkbox( 'Interaction Only', value=True,
						key='cluster_polynomial_interaction' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_polynomial_apply',
								use_container_width=True ):
							if poly_cols:
								df_processed = df_working.copy( )
								
								encoder = PolynomialFeatures( degree=int( degree ),
									interaction=bool( interaction ) )
								
								result = encoder.train_transform(
									df_processed[ poly_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, poly_cols, result,
									'polynomial' )
								
								commit_frame( df_processed )
								st.success( 'PolynomialFeatures applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_polynomial_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
		
		with feature_c2:
			with st.expander( label='Data Transformation', icon='⚡', key='cluster_transformers' ):
				with st.expander( 'Binarizer', expanded=False ):
					transform_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_binarizer_cols' )
					
					threshold = st.number_input( 'Threshold', value=0.0, step=0.1,
						key='cluster_binarizer_threshold' )
					
					copy = st.checkbox( 'Copy', value=True, key='cluster_binarizer_copy' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Binarizer',
								key='cluster_binarizer_apply',
								use_container_width=True ):
							if transform_cols:
								df_processed = df_working.copy( )
								transformer = Binarizer(
									threshold=float( threshold ),
									copy=bool( copy ) )
								result = transformer.train_transform(
									df_processed[ transform_cols ].to_numpy( ) )
								
								df_processed[ transform_cols ] = result
								commit_frame( df_processed )
								st.success( 'Binarizer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_binarizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column',
						options=categorical_columns,
						key='cluster_label_binarizer_col' )
					
					pos_label = st.number_input( 'Positive Label', value=1, step=1,
						key='cluster_label_binarizer_pos' )
					
					neg_label = st.number_input( 'Negative Label', value=0, step=1,
						key='cluster_label_binarizer_neg' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='cluster_label_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply LabelBinarizer',
								key='cluster_label_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ), sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, [
										target_col ], result,
									'label_binarizer' )
								commit_frame( df_processed )
								st.success( 'Label Binarizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_label_binarizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column',
						options=categorical_columns,
						key='cluster_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='cluster_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='cluster_multilabel_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_multilabel_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								y_multi = parse_multilabel_series( df_processed[ target_col ],
									delimiter=delimiter )
								
								transformer = MultiLabelBinarizer( classes=None,
									sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform( y_multi )
								df_processed = replace_columns( df_processed, [ target_col ],
									result, 'multilabel_binarizer' )
								
								commit_frame( df_processed )
								st.success( 'Multi-Label Binarizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_multilabel_binarizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'TF-IDF Transformer', expanded=False ):
					text_count_cols = st.multiselect( 'Count Matrix Columns',
						options=numeric_columns,
						key='cluster_tfidf_transformer_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', None ],
						index=1, key='cluster_tfidf_transformer_norm' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='cluster_tfidf_transformer_use_idf' )
					
					smooth_idf = st.checkbox( 'Smooth IDF', value=True,
						key='cluster_tfidf_transformer_smooth_idf' )
					
					sublinear_tf = st.checkbox( 'Sublinear TF', value=False,
						key='cluster_tfidf_transformer_sublinear' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_tfidf_transformer_apply',
								use_container_width=True ):
							if text_count_cols:
								df_processed = df_working.copy( )
								transformer = TfidfTransformer( norm=norm, use_idf=bool( use_idf ),
									smooth_idf=bool( smooth_idf ), sublinear_tf=bool( sublinear_tf ) )
								
								result = transformer.train_transform(
									df_processed[ text_count_cols ].apply(
										pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, text_count_cols,
									result, 'tfidf_transformer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Transformer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_tfidf_transformer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Column Transformer', expanded=False ):
					numeric_columns = st.multiselect( 'Numeric Columns', options=numeric_columns,
						key='cluster_column_transformer_numeric_columns' )
					
					categorical_columns = st.multiselect( 'Categorical Columns',
						options=categorical_columns,
						key='cluster_column_transformer_categorical_columns' )
					
					numeric_transform = st.selectbox( 'Numeric Transformer',
						options=[ 'StandardScaler', 'MinMaxScaler', 'RobustScaler',
						          'MaxAbsScaler', 'Binarizer', 'None' ],
						key='cluster_column_transformer_numeric_transform' )
					
					categorical_transform = st.selectbox( 'Categorical Transformer',
						options=[ 'OneHotEncoder', 'OrdinalEncoder', 'None' ],
						key='cluster_column_transformer_categorical_transform' )
					
					remainder = st.selectbox( 'Remainder', options=[ 'drop', 'passthrough' ],
						key='cluster_column_transformer_remainder' )
					
					sparse_threshold = st.slider( 'Sparse Threshold', min_value=0.0,
						max_value=1.0, value=0.3,
						key='cluster_column_transformer_sparse_threshold' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Column Transformer',
								key='cluster_column_transformer_apply',
								use_container_width=True ):
							df_processed = df_working.copy( )
							transformers = [ ]
							
							if numeric_columns and numeric_transform != 'None':
								if numeric_transform == 'StandardScaler':
									numeric_model = StandardScaler( ).model
								elif numeric_transform == 'MinMaxScaler':
									numeric_model = MinMaxScaler( ).model
								elif numeric_transform == 'RobustScaler':
									numeric_model = RobustScaler( ).model
								elif numeric_transform == 'MaxAbsScaler':
									numeric_model = MaxAbsScaler( ).model
								else:
									numeric_model = Binarizer( ).model
								
								transformers.append( ('numeric', numeric_model, numeric_columns) )
							
							if categorical_columns and categorical_transform != 'None':
								if categorical_transform == 'OneHotEncoder':
									categorical_model = OneHotEncoder( sparse=False,
										unknown='ignore' ).model
								else:
									categorical_model = OrdinalEncoder( ).model
								
								transformers.append( ('categorical', categorical_model,
								                      categorical_columns) )
							
							if transformers:
								transformer = ColumnTransformer( transformers=transformers,
									remainder=remainder, sparse_threshold=float( sparse_threshold ),
									n_jobs=None, transformer_weights=None, verbose=False )
								
								result = transformer.train_transform( df_processed )
								df_processed = normalize_result_frame( result=result,
									index=df_processed.index, prefix='column_transformer',
									columns=None )
								
								commit_frame( df_processed )
								st.success( 'ColumnTransformer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_column_transformer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Feature Extration', icon='⛏️', key='cluster_extractors' ):
				with st.expander( 'TF-IDF Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns',
						options=categorical_columns,
						key='cluster_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='cluster_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='cluster_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='cluster_tfidf_vectorizer_use_idf' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_tfidf_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = TfidfVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int( max_features ),
									use_idf=bool( use_idf ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'tfidf_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Vectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_tfidf_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns',
						options=categorical_columns,
						key='cluster_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='cluster_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0,
						step=1, key='cluster_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='cluster_count_vectorizer_binary' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_count_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = CountVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int( max_features ),
									binary=bool( binary ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'count_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'Count Vectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_count_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns',
						options=categorical_columns,
						key='cluster_hash_vectorizer_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='cluster_hash_vectorizer_n_features' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3,
						value=1, key='cluster_hash_vectorizer_ngram_max' )
					
					binary = st.checkbox( 'Binary', value=False,
						key='cluster_hash_vectorizer_binary' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='cluster_hash_vectorizer_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_hash_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = HashVectorizer( num=int( n_features ),
									ngram_range=(1, int( ngram_max )), binary=bool( binary ),
									alternate_sign=bool( alternate_sign ) )
								df_processed = apply_text_vectorizer( df_processed,
									text_cols, transformer, 'hash_vectorizer' )
								commit_frame( df_processed )
								st.success( 'HashVectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_hash_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					dict_cols = st.multiselect( 'Columns',
						options=categorical_columns,
						key='cluster_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='cluster_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='cluster_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='cluster_dict_vectorizer_sort' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_dict_vectorizer_apply',
								use_container_width=True ):
							if dict_cols:
								df_processed = df_working.copy( )
								transformer = DictVectorizer( dtype=np.float64, separator=separator,
									sparse=bool( sparse ), sort=bool( sort ) )
								
								df_processed = apply_dict_transform( df_processed, dict_cols,
									transformer, 'dict_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'DictVectorizer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_dict_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					hash_cols = st.multiselect( 'Columns',
						options=categorical_columns,
						key='cluster_feature_hasher_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='cluster_feature_hasher_n_features' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='cluster_feature_hasher_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_feature_hasher_apply',
								use_container_width=True ):
							if hash_cols:
								df_processed = df_working.copy( )
								transformer = FeatureHasher( n_features=int( n_features ),
									input_type='dict', dtype=np.float64,
									alternate_sign=bool( alternate_sign ) )
								
								df_processed = apply_dict_transform( df_processed, hash_cols,
									transformer, 'feature_hasher' )
								commit_frame( df_processed )
								st.success( 'FeatureHasher applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_feature_hasher_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Dimensionality Reduction', icon='🎚️', key='cluster_selectors' ):
				with st.expander( 'Variance Threshold', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0,
						step=0.01, key='cluster_variance_threshold_value' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_variance_threshold_apply',
								use_container_width=True ):
							if select_cols:
								df_processed = df_working.copy( )
								selector = VarianceThreshold( thresh=float( threshold ) )
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result,
									'variance_threshold' )
								
								commit_frame( df_processed )
								st.success( 'VarianceThreshold applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_variance_threshold_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Canonical Correlation Analysis', expanded=False ):
					X_cols = st.multiselect( 'Predictor Columns', options=numeric_columns,
						key='cluster_cca_x_cols' )
					
					y_cols = st.multiselect( 'Target Columns', options=numeric_columns,
						key='cluster_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2,
						step=1, key='cluster_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True,
						key='cluster_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500,
						step=1, key='cluster_cca_max_iter' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_cca_apply',
								use_container_width=True ):
							if X_cols and y_cols:
								df_processed = df_working.copy( )
								selector = CCA( num=int( n_components ), scale=bool( scale ),
									size=int( max_iter ) )
								
								result = selector.train_transform(
									df_processed[ X_cols ].to_numpy( ),
									df_processed[ y_cols ].to_numpy( ) )
								
								df_result = normalize_result_frame( result=result,
									index=df_processed.index, prefix='cca', columns=None )
								
								df_processed = pd.concat(
									[ df_processed.drop( columns=X_cols + y_cols, errors='ignore' ),
									  df_result ], axis=1 )
								
								commit_frame( df_processed )
								st.success( 'CCA Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_cca_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Principle Component Analysis', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_pca_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2,
						step=1, key='cluster_pca_components' )
					
					solver = st.selectbox( 'SVD Solver',
						options=[ 'auto', 'full', 'randomized', 'covariance_eigh', 'arpack' ],
						key='cluster_pca_solver' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_pca_apply',
								use_container_width=True ):
							if select_cols:
								df_processed = df_working.copy( )
								selector = PCA( num=int( n_components ), solver=solver )
								
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result, 'pca' )
								commit_frame( df_processed )
								st.success( 'PCA applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_pca_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Select-Best', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_selectbest_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
						          'mutual_info_regression' ],
						key='cluster_selectbest_score_name' )
					
					k_best = st.number_input( 'K', min_value=1, value=5, step=1,
						key='cluster_selectbest_k' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_selectbest_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectBest(
									score_func=score_function_from_name( score_name ),
									num=int( k_best ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'select_best' )
								commit_frame( df_processed )
								st.success( 'Select Best Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_selectbest_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Select-Percent', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_selectpercent_x_cols' )
					
					target_col = st.selectbox( 'Target Column',
						options=categorical_columns,
						key='cluster_selectpercent_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
						          'mutual_info_regression' ],
						key='cluster_selectpercent_score_name' )
					
					percentile = st.slider( 'Percentile', min_value=1, max_value=100, value=10,
						key='cluster_selectpercent_percentile' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SelectPercent',
								key='cluster_selectpercent_apply', use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectPercent(
									score_func=score_function_from_name( score_name ),
									pct=int( percentile ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'select_percent' )
								commit_frame( df_processed )
								st.success( 'SelectPercent applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_selectpercent_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Sequential Back Selection', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_sbs_x_cols' )
					
					target_col = st.selectbox( 'Target Column',
						options=categorical_columns,
						key='cluster_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='cluster_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='cluster_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1,
						step=1, key='cluster_sbs_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_sbs_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SBS( classifier=None, k_features=int( k_features ),
									test_size=float( test_size ), random_state=int( random_state ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'sbs' )
								
								commit_frame( df_processed )
								st.success( 'SBS applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_sbs_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Recursive Feature Elimination', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_rfe_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain',
						min_value=1, value=1, step=1, key='cluster_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0,
						step=1, key='cluster_rfe_verbose' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_rfe_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = RFE( k_features=int( k_features ), verbose=int( verbose ) )
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'rfe' )
								commit_frame( df_processed )
								st.success( 'RFE applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_rfe_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed', df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Processed Data' )
		st.caption( f'Samples: {len( df_processed ):,} | Features: {len( df_processed.columns ):,}' )
		st.data_editor( df_processed, key='cluster_processed_data' )
		
		# ------------------------------------------------------------------
		# MODEL SELECTION
		# ------------------------------------------------------------------
		st.markdown( '##### Clustering Model' )
		
		if 'cluster_kmeans_n_clusters' not in st.session_state:
			st.session_state[ 'cluster_kmeans_n_clusters' ] = 3
		
		if 'cluster_kmeans_init' not in st.session_state:
			st.session_state[ 'cluster_kmeans_init' ] = 'k-means++'
		
		if 'cluster_kmeans_n_init_mode' not in st.session_state:
			st.session_state[ 'cluster_kmeans_n_init_mode' ] = 'auto'
		
		if 'cluster_kmeans_n_init_value' not in st.session_state:
			st.session_state[ 'cluster_kmeans_n_init_value' ] = 10
		
		if 'cluster_kmeans_max_iter' not in st.session_state:
			st.session_state[ 'cluster_kmeans_max_iter' ] = 300
		
		if 'cluster_kmeans_tol' not in st.session_state:
			st.session_state[ 'cluster_kmeans_tol' ] = 0.0001
		
		if 'cluster_kmeans_random_state' not in st.session_state:
			st.session_state[ 'cluster_kmeans_random_state' ] = 42
		
		if 'cluster_kmeans_verbose' not in st.session_state:
			st.session_state[ 'cluster_kmeans_verbose' ] = 0
		
		if 'cluster_kmeans_copy_x' not in st.session_state:
			st.session_state[ 'cluster_kmeans_copy_x' ] = True
		
		if 'cluster_kmeans_algorithm' not in st.session_state:
			st.session_state[ 'cluster_kmeans_algorithm' ] = 'lloyd'
		
		with st.expander( 'K-Means', expanded=True ):
			st.caption( 'Prototype-based clustering using centroid minimization.' )
			
			km_c1, km_c2, km_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			
			with km_c1:
				n_clusters = st.number_input(
					'Number of Clusters (K)',
					min_value=2,
					step=1,
					key='cluster_kmeans_n_clusters'
				)
				
				init = st.selectbox(
					'Initialization',
					options=[ 'k-means++', 'random' ],
					key='cluster_kmeans_init'
				)
				
				algorithm = st.selectbox(
					'Algorithm',
					options=[ 'lloyd', 'elkan' ],
					key='cluster_kmeans_algorithm'
				)
			
			with km_c2:
				n_init_mode = st.selectbox(
					'Initialization Runs',
					options=[ 'auto', 'manual' ],
					key='cluster_kmeans_n_init_mode'
				)
				
				if n_init_mode == 'manual':
					n_init = int(
						st.number_input(
							'Number of Initializations',
							min_value=1,
							step=1,
							key='cluster_kmeans_n_init_value'
						)
					)
				else:
					n_init = 'auto'
					st.text_input(
						'Number of Initializations',
						value='auto',
						disabled=True,
						key='cluster_kmeans_n_init_display'
					)
				
				max_iter = st.number_input(
					'Maximum Iterations',
					min_value=1,
					step=1,
					key='cluster_kmeans_max_iter'
				)
			
			with km_c3:
				tol = st.number_input(
					'Tolerance',
					min_value=0.0,
					step=0.0001,
					format='%.4f',
					key='cluster_kmeans_tol'
				)
				
				random_state = st.number_input( 'Random State', step=1,
					key='cluster_kmeans_random_state' )
				
				verbose = st.number_input( 'Verbose', min_value=0, step=1,
					key='cluster_kmeans_verbose' )
				
				copy_x = st.checkbox( 'Copy Input Data', key='cluster_kmeans_copy_x' )
			
			model = KMeans( clusters=int( n_clusters ), init=init, n_init=n_init,
				tol=float( tol ), rando=int( random_state ), max_iter=int( max_iter ),
				verbose=int( verbose ), copy_x=bool( copy_x ), algorithm=algorithm )
			
			model_parameters = {
					'Model': 'K-Means',
					'n_clusters': int( n_clusters ),
					'init': init,
					'n_init': n_init,
					'max_iter': int( max_iter ),
					'tol': float( tol ),
					'random_state': int( random_state ),
					'verbose': int( verbose ),
					'copy_x': bool( copy_x ),
					'algorithm': algorithm
			}
			
			km_b1, km_b2 = st.columns( 2 )
			
			with km_b1:
				if st.button( 'Run K-Means', icon='🏃', key='cluster_kmeans_run',
						use_container_width=True ):
					cluster_signature = (
							tuple( feature_columns ),
							'K-Means',
							tuple( (k, str( v )) for k, v in model_parameters.items( ) ) )
					
					try:
						start_time = time.time( )
						labels = model.project( X )
						elapsed_seconds = time.time( ) - start_time
						
						df_results = df_cluster_input.copy( )
						df_results[ 'Cluster' ] = labels
						
						df_counts = (
								df_results[ 'Cluster' ]
								.value_counts( dropna=False )
								.rename_axis( 'Cluster' )
								.reset_index( name='Count' )
								.sort_values( by='Cluster' )
								.reset_index( drop=True ) )
						
						try:
							df_metrics = model.score( X )
							if df_metrics is None:
								df_metrics = pd.DataFrame( )
						except Exception:
							df_metrics = pd.DataFrame( )
						
						if df_metrics is None or df_metrics.empty:
							df_metrics = pd.DataFrame(
								[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ]
							)
						else:
							df_metrics = df_metrics.copy( )
							df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
						
						detail_rows = [ ]
						for prop in [
								'features',
								'inertia',
								'iterations',
								'metric',
								'algorithm',
								'n_clusters',
								'n_init',
								'max_iter',
								'tolerance',
								'random_state'
						]:
							if hasattr( model, prop ):
								try:
									value = getattr( model, prop )
									if value is not None and not isinstance(
											value,
											(np.ndarray, pd.DataFrame)
									):
										detail_rows.append(
											{ 'Property': prop, 'Value': value }
										)
								except Exception:
									pass
						
						df_details = (
								pd.DataFrame( detail_rows )
								if detail_rows
								else pd.DataFrame( )
						)
						
						df_centroids = pd.DataFrame( )
						if hasattr( model, 'centroids_' ):
							try:
								centroids = model.centroids_
								if centroids is not None:
									df_centroids = pd.DataFrame(
										centroids,
										columns=feature_columns
									)
									df_centroids.insert(
										0,
										'Cluster',
										range( len( df_centroids ) )
									)
							except Exception:
								df_centroids = pd.DataFrame( )
						
						st.session_state[ 'df_cluster_results' ] = df_results
						st.session_state[ 'df_cluster_counts' ] = df_counts
						st.session_state[ 'df_cluster_metrics' ] = df_metrics
						st.session_state[ 'df_cluster_centroids' ] = df_centroids
						st.session_state[ 'df_cluster_details' ] = df_details
						st.session_state[ 'cluster_plot_features' ] = feature_columns.copy( )
						st.session_state[ 'cluster_signature' ] = cluster_signature
						
						st.success( 'K-Means clustering complete.' )
					
					except Exception as ex:
						st.session_state[ 'df_cluster_results' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_counts' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_metrics' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_centroids' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_details' ] = pd.DataFrame( )
						st.session_state[ 'cluster_plot_features' ] = [ ]
						st.session_state[ 'cluster_signature' ] = None
						st.error( f'K-Means clustering failed: {ex}' )
			
			with km_b2:
				if st.button( 'Reset K-Means', icon='🔁', key='cluster_kmeans_reset',
						use_container_width=True ):
					
					st.session_state[ 'cluster_kmeans_n_clusters' ] = 3
					st.session_state[ 'cluster_kmeans_init' ] = 'k-means++'
					st.session_state[ 'cluster_kmeans_n_init_mode' ] = 'auto'
					st.session_state[ 'cluster_kmeans_n_init_value' ] = 10
					st.session_state[ 'cluster_kmeans_max_iter' ] = 300
					st.session_state[ 'cluster_kmeans_tol' ] = 0.0001
					st.session_state[ 'cluster_kmeans_random_state' ] = 42
					st.session_state[ 'cluster_kmeans_verbose' ] = 0
					st.session_state[ 'cluster_kmeans_copy_x' ] = True
					st.session_state[ 'cluster_kmeans_algorithm' ] = 'lloyd'
					st.session_state[ 'df_cluster_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_plot_features' ] = [ ]
					st.session_state[ 'cluster_signature' ] = None
					st.rerun( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
			st.markdown( '##### Cluster Summary' )
			
			if df_counts is not None and not df_counts.empty:
				st.data_editor( df_counts, use_container_width=True )
				
				if df_metrics is not None and not df_metrics.empty:
					st.caption( 'Metrics' )
					st.data_editor( df_metrics, use_container_width=True )
				
				if df_details is not None and not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True )
			else:
				st.info( 'Run clustering to view cluster counts and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
			st.subheader( 'Cluster Visualization' )
			
			if df_results is not None and not df_results.empty:
				if len( feature_columns ) == 2:
					plt.close( 'all' )
					fig, ax = plt.subplots( )
					ax.scatter(
						df_results[ feature_columns[ 0 ] ],
						df_results[ feature_columns[ 1 ] ],
						c=df_results[ 'Cluster' ],
						alpha=0.7
					)
					ax.set_xlabel( feature_columns[ 0 ] )
					ax.set_ylabel( feature_columns[ 1 ] )
					ax.set_title( 'Cluster Assignments' )
					
					if df_centroids is not None and not df_centroids.empty:
						try:
							ax.scatter(
								df_centroids[ feature_columns[ 0 ] ],
								df_centroids[ feature_columns[ 1 ] ],
								marker='x',
								s=100
							)
						except Exception:
							pass
					
					st.pyplot( fig )
					plt.close( fig )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run clustering to view the scatter plot.' )
			
			# ------------------------------------------------------------------
			# CENTROIDS (IF AVAILABLE)
			# ------------------------------------------------------------------
			if df_centroids is not None and not df_centroids.empty:
				st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
				st.markdown( '##### Cluster Centroids' )
				st.data_editor( df_centroids, use_container_width=True )

# ============================================
# TIME SERIES MODE
# ============================================
elif mode == 'Time-Series Models':
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Time-Series Models' ] )
		st.divider( )
		
		# ------------------------------------------------------------------
		# DATA VALIDATION
		# ------------------------------------------------------------------
		df_dataset = st.session_state.get( 'df_infer', None )
		numeric_columns = st.session_state.get( 'numeric_columns', [ ] )
		
		if df_dataset is None or df_dataset.empty:
			st.warning( '⚠️ No dataset loaded.' )
			st.stop( )
		
		if not numeric_columns:
			st.warning( '⚠️ No numeric columns available for time-series analysis.' )
			st.stop( )
		
		# ------------------------------------------------------------------
		# SERIES SELECTION
		# ------------------------------------------------------------------
		st.markdown( '##### Time-Series Selection' )
		series_col = st.selectbox( 'Select Numeric Time-Series Column', numeric_columns,
			key='timeseries_col_box' )
		series = df_dataset[ series_col ].dropna( ).to_numpy( )
		
		if series.ndim != 1 or len( series ) < 10:
			st.warning( '⚠️ Selected series is too short for modeling.' )
			st.stop( )
		
		# ------------------------------------------------------------------
		# MODEL SELECTION
		# ------------------------------------------------------------------
		model_map = \
			{
					'Lagged Linear Regression': 'lag',
					'Lagged Boosting Regression': 'boost',
					'ARIMA': 'arima',
					'SARIMA': 'sarima'
			}
		
		st.markdown( '##### Model Selection' )
		model_name = st.selectbox( 'Select time-series model', list( model_map.keys( ) ),
			key='model_name_box' )
		
		# ------------------------------------------------------------------
		# MODEL PARAMETERS
		# ------------------------------------------------------------------
		st.subheader( 'Model Parameters' )
		model = None
		
		if model_name == 'Lagged Linear Regression':
			lag = st.number_input( 'Lag order', min_value=1, value=5, key='lag_input' )
			model = LaggingSeries( lag=int( lag ) )
		
		elif model_name == 'Lagged Boosting Regression':
			lag = st.number_input( 'Lag order', min_value=1, value=12 )
			loss = st.selectbox( 'Loss',
				[ 'squared_error', 'absolute_error', 'gamma', 'poisson', 'quantile' ],
				index=0, key='loss_box' )
			
			quantile = None
			if loss == 'quantile':
				quantile = st.number_input(
					'Quantile',
					min_value=0.01,
					max_value=0.99,
					value=0.50,
					step=0.01
				)
			
			rate = st.number_input(
				'Learning Rate',
				min_value=0.001,
				max_value=1.0,
				value=0.1,
				step=0.001,
				format='%.3f'
			)
			iters = st.number_input( 'Max Iterations', min_value=10, value=100 )
			leaf_nodes = st.number_input( 'Max Leaf Nodes (0 = None)', min_value=0, value=31 )
			depth = st.number_input( 'Max Depth (0 = None)', min_value=0, value=0 )
			leaf = st.number_input( 'Min Samples Leaf', min_value=1, value=20 )
			regularization = st.number_input(
				'L2 Regularization',
				min_value=0.0,
				value=0.0,
				step=0.001,
				format='%.3f'
			)
			features = st.number_input(
				'Max Features',
				min_value=0.1,
				max_value=1.0,
				value=1.0,
				step=0.1,
				format='%.1f'
			)
			bins = st.number_input( 'Max Bins', min_value=2, max_value=255, value=255 )
			stopping = st.selectbox( 'Early Stopping', [ 'auto', True, False ],
				index=0, key='stop_box' )
			validation = st.number_input( 'Validation Fraction', min_value=0.01, max_value=0.50,
				value=0.10, step=0.01, format='%.2f', key='validation_input' )
			no_change = st.number_input( 'Iterations No Change', min_value=1, value=10 )
			tol = st.number_input(
				'Tolerance',
				min_value=0.0,
				value=1e-7,
				step=1e-7,
				format='%.7f'
			)
			verbose = st.number_input( 'Verbose', min_value=0, value=0 )
			rando = st.number_input( 'Random State (-1 = None)', min_value=-1, value=-1 )
			
			model = LagBoostingSeries(
				lag=int( lag ),
				loss=loss,
				quantile=float( quantile ) if quantile is not None else None,
				rate=float( rate ),
				iters=int( iters ),
				leaf_nodes=int( leaf_nodes ) if int( leaf_nodes ) > 0 else None,
				depth=int( depth ) if int( depth ) > 0 else None,
				leaf=int( leaf ),
				regularization=float( regularization ),
				features=float( features ),
				bins=int( bins ),
				stopping=stopping,
				validation=float( validation ),
				no_change=int( no_change ),
				tol=float( tol ),
				verbose=int( verbose ),
				rando=None if int( rando ) < 0 else int( rando )
			)
		
		elif model_name == 'ARIMA':
			p = st.number_input( 'p (AR)', min_value=0, value=1 )
			d = st.number_input( 'd (I)', min_value=0, value=0 )
			q = st.number_input( 'q (MA)', min_value=0, value=0 )
			model = ARIMA( order=(int( p ), int( d ), int( q )) )
		
		elif model_name == 'SARIMA':
			p = st.number_input( 'p (AR)', min_value=0, value=1 )
			d = st.number_input( 'd (I)', min_value=0, value=1 )
			q = st.number_input( 'q (MA)', min_value=0, value=1 )
			P = st.number_input( 'P (Seasonal AR)', min_value=0, value=0 )
			D = st.number_input( 'D (Seasonal I)', min_value=0, value=0 )
			Q = st.number_input( 'Q (Seasonal MA)', min_value=0, value=0 )
			s = st.number_input( 'Season Length', min_value=0, value=0 )
			model = SARIMA(
				order=(int( p ), int( d ), int( q )),
				seasonal=(int( P ), int( D ), int( Q ), int( s ))
			)
		
		# ------------------------------------------------------------------
		# TRAIN / FORECAST
		# ------------------------------------------------------------------
		st.subheader( 'Train & Forecast' )
		forecast_horizon = st.number_input( 'Forecast Horizon (Steps)', min_value=1, value=5 )
		
		if st.button( '🚀 Run Time-Series Model' ):
			try:
				plt.close( 'all' )
				model.train( series )
				forecast = model.project( n_steps=int( forecast_horizon ) )
				
				st.subheader( 'Model Evaluation' )
				metrics = model.analyze( )
				
				if isinstance( metrics, pd.DataFrame ):
					st.data_editor( metrics, use_container_width=True )
				else:
					df_metrics = pd.DataFrame( metrics, index=[ 'Value' ] ).T
					st.data_editor( df_metrics, use_container_width=True )
				
				st.subheader( 'Observed vs Forecast' )
				fig, ax = plt.subplots( )
				ax.plot( range( len( series ) ), series, label='Observed' )
				ax.plot(
					range( len( series ), len( series ) + len( forecast ) ),
					forecast,
					label='Forecast',
					linestyle='--'
				)
				ax.set_title( 'Time-Series Forecast' )
				ax.legend( )
				st.pyplot( fig )
				plt.close( fig )
			except Exception as e:
				st.error( f'Time-Series Modeling failed: {e}' )
		
		# ------------------------------------------------------------------
		# TIME-SERIES CROSS-VALIDATION
		# ------------------------------------------------------------------
		st.subheader( 'Time-Series Cross-Validation' )
		
		with st.expander( 'Show time-series splits' ):
			splits = st.number_input( 'Number of splits', min_value=2, value=5 )
			test_size = st.number_input( 'Test window size', min_value=1, value=10 )
			gap = st.number_input( 'Gap size', min_value=0, value=0 )
			max_train_size = st.number_input( 'Max train size (0 = unlimited)', min_value=0, value=0 )
			
			splitter = TimeSeriesSpliter(
				splits=int( splits ),
				test_size=int( test_size ),
				gap=int( gap ),
				max_train_size=int( max_train_size ) if int( max_train_size ) > 0 else None
			)
			
			if st.button( 'Visualize CV Splits' ):
				try:
					plt.close( 'all' )
					fig = splitter.visualize( series )
					st.pyplot( fig )
					plt.close( fig )
				except Exception as e:
					st.error( f'Time-Series split visualization failed: {e}' )

# ============================================
# DATA MANAGEMENT MODE
# ============================================
elif mode == 'Data Management':
	st.subheader( cfg.MODE[ 'Database' ] )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		tabs = st.tabs( [ 'Import', 'Browse', 'CRUD', 'Explore', 'Filter',
		                  'Aggregate', 'Visualize', 'Admin', 'SQL' ] )
		
		tables = list_tables( )
		if not tables:
			st.info( 'No tables available.' )
		else:
			table = st.selectbox( 'Table', tables, key='table_selectbox' )
			df_full = read_table( table )
		
		# ------------------------------------------------------------------------------
		# UPLOAD TAB
		# ------------------------------------------------------------------------------
		with tabs[ 0 ]:
			upl_c1, upl_c2 = st.columns( [ 0.5, 0.5 ], border=True )
			with upl_c1:
				uploaded_file = st.file_uploader( 'Upload Excel File', type=[ 'xlsx' ] )
				
			with upl_c2:
				overwrite = st.checkbox( 'Overwrite existing tables', value=True )
				if uploaded_file:
					try:
						sheets = pd.read_excel( uploaded_file, sheet_name=None )
						with create_connection( ) as conn:
							conn.execute( 'BEGIN' )
							for sheet_name, df in sheets.items( ):
								table_name = create_identifier( sheet_name )
								if overwrite:
									conn.execute( f'DROP TABLE IF EXISTS "{table_name}"' )
								
								# --- Create Table ---
								columns = [ ]
								df.columns = [ create_identifier( c ) for c in df.columns ]
								for col in df.columns:
									sql_type = get_sqlite_type( df[ col ].dtype )
									columns.append( f'"{col}" {sql_type}' )
								
								create_stmt = ( f'CREATE TABLE "{table_name}" '
										f'({", ".join( columns )});' )
								
								conn.execute( create_stmt )
								
								# --- Insert Data ---
								placeholders = ", ".join( [ "?" ] * len( df.columns ) )
								insert_stmt = ( f'INSERT INTO "{table_name}" '
										f'VALUES ({placeholders});' )
								
								conn.executemany( insert_stmt,
									df.where( pd.notnull( df ), None ).values.tolist( ) )
							
							conn.commit( )
						
						st.success( 'Import completed successfully (transaction committed).' )
						st.rerun( )
					
					except Exception as e:
						try:
							conn.rollback( )
						except:
							pass
						st.error( f'Import failed — transaction rolled back.\n\n{e}' )
		
		# ------------------------------------------------------------------------------
		# BROWSE TAB
		# ------------------------------------------------------------------------------
		with tabs[ 1 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='table_name' )
				df = read_table( table )
				st.dataframe( df, use_container_width=True )
			else:
				st.info( 'No tables available.' )
		
		# ------------------------------------------------------------------------------
		# CRUD (Schema-Aware)
		# ------------------------------------------------------------------------------
		with tabs[ 2 ]:
			tables = list_tables( )
			if not tables:
				st.info( 'No tables available.' )
			else:
				table = st.selectbox( 'Select Table', tables, key='crud_table' )
				df = read_table( table )
				schema = create_schema( table )
				
				# Build type map
				type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != 'rowid' }
				
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				st.markdown( '##### Insert Row' )
				insert_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						insert_data[ column ] = st.number_input( column, step=1,
							key=f'ins_{column}' )
					
					elif 'REAL' in col_type:
						insert_data[ column ] = st.number_input( column, format='%.6f',
							key=f'ins_{column}' )
					
					elif 'BOOL' in col_type:
						insert_data[ column ] = 1 if st.checkbox( column,
							key=f'ins_{column}' ) else 0
					
					else:
						insert_data[ column ] = st.text_input( column, key=f'ins_{column}' )
				
				if st.button( 'Insert Row' ):
					cols = list( insert_data.keys( ) )
					placeholders = ', '.join( [ '?' ] * len( cols ) )
					stmt = f'INSERT INTO "{table}" ({", ".join( cols )}) VALUES ({placeholders});'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( insert_data.values( ) ) )
						conn.commit( )
					
					st.success( 'Row inserted.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# UPDATE
				# ------------------------------------------------------------------
				st.markdown( '##### Update Row' )
				rowid = st.number_input( 'Row ID', min_value=1, step=1 )
				update_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						val = st.number_input( column, step=1, key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'REAL' in col_type:
						val = st.number_input( column, format='%.6f', key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'BOOL' in col_type:
						val = 1 if st.checkbox( column, key=f'upd_{column}' ) else 0
						update_data[ column ] = val
					
					else:
						val = st.text_input( column, key=f"upd_{column}" )
						update_data[ column ] = val
				
				if st.button( 'Update Row' ):
					set_clause = ', '.join( [ f'{c}=?' for c in update_data ] )
					stmt = f'UPDATE {table} SET {set_clause} WHERE rowid=?;'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( update_data.values( ) ) + [ rowid ] )
						conn.commit( )
					
					st.success( 'Row updated.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# DELETE
				# ------------------------------------------------------------------
				st.markdown( '##### Delete Row' )
				delete_id = st.number_input( 'Row ID to Delete', min_value=1, step=1 )
				if st.button( 'Delete Row' ):
					with create_connection( ) as conn:
						conn.execute( f'DELETE FROM {table} WHERE rowid=?;', (delete_id,) )
						conn.commit( )
					
					st.success( 'Row deleted.' )
					st.rerun( )
		
		# ------------------------------------------------------------------------------
		# EXPLORE
		# ------------------------------------------------------------------------------
		with tabs[ 3 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='explore_table' )
				page_size = st.slider( 'Rows per page', 10, 500, 50 )
				page = st.number_input( 'Page', min_value=1, step=1 )
				offset = (page - 1) * page_size
				df_page = read_table( table, page_size, offset )
				st.dataframe( df_page, use_container_width=True )
		
		# ------------------------------------------------------------------------------
		# FILTER
		# ------------------------------------------------------------------------------
		with tabs[ 4 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='filter_table' )
				df = read_table( table )
				column = st.selectbox( 'Column', df.columns, key='filter_column_box' )
				value = st.text_input( 'Contains' )
				if value:
					df = df[ df[ column ].astype( str ).str.contains( value ) ]
				st.dataframe( df, use_container_width=True )
		
		# ------------------------------------------------------------------------------
		# AGGREGATE
		# ------------------------------------------------------------------------------
		with tabs[ 5 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='agg_table' )
				df = read_table( table )
				numeric_columns = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_columns:
					col = st.selectbox( 'Column', numeric_columns, key='col_box' )
					agg = st.selectbox( 'Function', [ 'SUM', 'AVG', 'COUNT' ], key='agg_box' )
					if agg == 'SUM':
						st.metric( 'Result', df[ col ].sum( ) )
					elif agg == 'AVG':
						st.metric( 'Result', df[ col ].mean( ) )
					elif agg == 'COUNT':
						st.metric( 'Result', df[ col ].count( ) )
		
		# ------------------------------------------------------------------------------
		# VISUALIZE
		# ------------------------------------------------------------------------------
		with tabs[ 6 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='viz_table' )
				df = read_table( table )
				numeric_columns = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_columns:
					col = st.selectbox( 'Column', numeric_columns, key='numeric_col_box' )
					fig = px.histogram( df, x=col )
					st.plotly_chart( fig, use_container_width=True )
		
		# ------------------------------------------------------------------------------
		# ADMIN
		# ------------------------------------------------------------------------------
		with tabs[ 7 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='admin_table' )
			
			st.divider( )
			st.markdown( '##### Data Profiling' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='profile_table' )
				if st.button( 'Generate Profile' ):
					profile_df = create_profile_table( table )
					st.dataframe( profile_df, use_container_width=True )
			
			st.markdown( '##### Drop Table' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table to Drop', tables, key='admin_drop_table' )
				
				# Initialize confirmation state
				if 'dm_confirm_drop' not in st.session_state:
					st.session_state.dm_confirm_drop = False
				
				# Step 1: Initial Drop click
				if st.button( 'Drop Table', key='admin_drop_button' ):
					st.session_state.dm_confirm_drop = True
				
				# Step 2: Confirmation UI
				if st.session_state.dm_confirm_drop:
					st.warning( f'You are about to permanently delete table {table}. '
					            'This action cannot be undone.' )
					
					col1, col2 = st.columns( 2 )
					
					if col1.button( 'Confirm Drop', key='admin_confirm_drop' ):
						try:
							drop_table( table )
							st.success( f'Table {table} dropped successfully.' )
						except Exception as e:
							st.error( f'Drop failed: {e}' )
						
						st.session_state.dm_confirm_drop = False
						st.rerun( )
					
					if col2.button( 'Cancel', key='admin_cancel_drop' ):
						st.session_state.dm_confirm_drop = False
						st.rerun( )
				
				df = read_table( table )
				col = st.selectbox( 'Create Index On', df.columns, key='index_box' )
				
				if st.button( 'Create Index' ):
					create_index( table, col )
					st.success( 'Index created.' )
			
			st.divider( )
			
			st.markdown( '##### Create Custom Table' )
			new_table_name = st.text_input( 'Table Name' )
			column_count = st.number_input( 'Number of Columns', min_value=1, max_value=20, value=1 )
			columns = [ ]
			for i in range( column_count ):
				st.markdown( f'### Column {i + 1}' )
				col_name = st.text_input( 'Column Name', key=f'col_name_{i}' )
				col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
					key=f'col_type_{i}' )
				
				not_null = st.checkbox( 'NOT NULL', key=f'not_null_{i}' )
				primary_key = st.checkbox( 'PRIMARY KEY', key=f'pk_{i}' )
				auto_inc = st.checkbox( 'AUTOINCREMENT (INTEGER only)', key=f'ai_{i}' )
				
				columns.append( {
						'name': col_name,
						'type': col_type,
						'not_null': not_null,
						'primary_key': primary_key,
						'auto_increment': auto_inc } )
			
			if st.button( 'Create Table' ):
				try:
					create_custom_table( new_table_name, columns )
					st.success( 'Table created successfully.' )
					st.rerun( )
				
				except Exception as e:
					st.error( f'Error: {e}' )
			
			st.divider( )
			st.markdown( '##### Schema Viewer' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='schema_view_table' )
				schema = create_schema( table )
				schema_df = pd.DataFrame(
					schema,
					columns=[ 'cid', 'name', 'type', 'notnull', 'default', 'pk' ] )
				
				st.markdown( "##### Columns" )
				st.dataframe( schema_df, use_container_width=True )
				
				# Row count
				with create_connection( ) as conn:
					count = conn.execute(
						f'SELECT COUNT(*) FROM "{table}"'
					).fetchone( )[ 0 ]
				
				st.metric( "Row Count", f"{count:,}" )
				
				# Indexes
				indexes = get_indexes( table )
				if indexes:
					idx_df = pd.DataFrame( indexes,
						columns=[ 'seq', 'name', 'unique', 'origin', 'partial' ] )
					
					st.markdown( "##### Indexes" )
					st.dataframe( idx_df, use_container_width=True )
				else:
					st.info( "No indexes defined." )
			
			st.divider( )
			st.subheader( "ALTER TABLE Operations" )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='alter_table_select' )
				operation = st.selectbox( 'Operation',
					[ 'Add Column', 'Rename Column', 'Rename Table', 'Drop Column' ],
					key='operation_box' )
				
				if operation == 'Add Column':
					new_col = st.text_input( 'Column Name' )
					col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
						key='type_box' )
					
					if st.button( 'Add Column' ):
						add_column( table, new_col, col_type )
						st.success( 'Column added.' )
						st.rerun( )
				
				elif operation == 'Rename Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					old_col = st.selectbox( 'Column to Rename', col_names, key='column_selectbox' )
					new_col = st.text_input( 'New Column Name' )
					
					if st.button( 'Rename Column' ):
						rename_column( table, old_col, new_col )
						st.success( 'Column renamed.' )
						st.rerun( )
				
				elif operation == 'Rename Table':
					new_name = st.text_input( 'New Table Name' )
					
					if st.button( 'Rename Table' ):
						rename_table( table, new_name )
						st.success( 'Table renamed.' )
						st.rerun( )
				
				elif operation == 'Drop Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					drop_col = st.selectbox( 'Column to Drop', col_names, key='drop_box' )
					
					if st.button( 'Drop Column' ):
						drop_column( table, drop_col )
						st.success( 'Column dropped.' )
						st.rerun( )
		
		# ------------------------------------------------------------------------------
		# SQL
		# ------------------------------------------------------------------------------
		with tabs[ 8 ]:
			st.markdown( '##### SQL Console' )
			query = st.text_area( 'Enter SQL Query' )
			if st.button( 'Run Query' ):
				if not is_safe_query( query ):
					st.error( 'Query blocked: Only read-only SELECT statements are allowed.' )
				else:
					try:
						start_time = time.perf_counter( )
						with create_connection( ) as conn:
							result = pd.read_sql_query( query, conn )
						
						end_time = time.perf_counter( )
						elapsed = end_time - start_time
						
						# ----------------------------------------------------------
						# Display Results
						# ----------------------------------------------------------
						st.dataframe( result, use_container_width=True )
						row_count = len( result )
						
						# ----------------------------------------------------------
						# Execution Metrics
						# ----------------------------------------------------------
						col1, col2 = st.columns( 2 )
						col1.metric( 'Rows Returned', f'{row_count:,}' )
						col2.metric( 'Execution Time (seconds)', f'{elapsed:.6f}' )
						
						# Optional slow query warning
						if elapsed > 2.0:
							st.warning( 'Slow query detected (> 2 seconds). Consider indexing.' )
						
						# ----------------------------------------------------------
						# Download
						# ----------------------------------------------------------
						if not result.empty:
							csv = result.to_csv( index=False ).encode( 'utf-8' )
							st.download_button( 'Download CSV', csv,
								'query_results.csv', 'text/csv' )
					
					except Exception as e:
						st.error( f'Execution failed: {e}' )