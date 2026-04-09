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
	confusion_matrix, roc_curve, auc,
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

from classifications import ( Perceptron, LeastSquares, LogisticRegression, DecisionTree,
	SupportVector, RandomForest, NearestNeighbor, BaggingModel, AdaptiveBoost, GradientBoost )

from encoders import (OneHotEncoder, OrdinalEncoder, TargetEncoder)

from regressions import ( LeastSquares, Ridge, Lasso, ElasticNet, BayesianRidge, SupportVector,
	GradientDescent, NearestNeighbor, BaggingModel, ExtraTreesModel, AdaptiveBoost,
	GradientBoost, RandomForest, VotingModel, StackingModel )

from imputers import (MeanImputer, SimpleImputer, NearestImputer, IterativeImputer)
from forecasting import ( LaggingSeries, LagBoostingSeries, ARIMA, SARIMA, TimeSeriesSpliter )

# ============================================
# Session State
# ============================================

if 'mode' not in st.session_state or st.session_state[ 'mode' ] is None:
	st.session_state[ 'mode' ] = 'Data Transformation'

if 'df_dataset' not in st.session_state or st.session_state[ 'df_dataset' ] is None:
	st.session_state[ 'df_dataset'] = pd.DataFrame( )
	
# ------ Data Plumbing Members

if 'df_original' not in st.session_state or st.session_state[ 'df_original' ] is None:
	st.session_state[ 'df_original' ] = pd.DataFrame( )

if 'df_features' not in st.session_state:
	st.session_state[ 'df_features' ] = pd.DataFrame( )

if 'df_targets' not in st.session_state:
	st.session_state[ 'df_targets' ] = pd.DataFrame( )

if 'df_processed' not in st.session_state or st.session_state[ 'df_processed' ] is None:
	st.session_state[ 'df_processed' ] = pd.DataFrame( )

if 'df_cluster_results' not in st.session_state:
	st.session_state[ 'df_cluster_results' ] = pd.DataFrame( )
	
if 'df_dataset' not in st.session_state or st.session_state[ 'df_dataset' ] is None:
	st.session_state[ 'df_dataset' ] = pd.DataFrame( )

if 'df_working' not in st.session_state or st.session_state[ 'df_working' ] is None:
	st.session_state[ 'df_working' ] = pd.DataFrame( )

if 'df_regression' not in st.session_state or st.session_state[ 'df_regression' ] is None:
	st.session_state[ 'df_regression' ] = pd.DataFrame( )
	
if 'df_classification' not in st.session_state or st.session_state[ 'df_classification' ] is None:
	st.session_state[ 'df_classification' ] = pd.DataFrame( )

if 'numeric_columns' not in st.session_state:
	st.session_state[ 'numeric_columns' ] = [ ]

if 'categorical_columns' not in st.session_state:
	st.session_state[ 'categorical_columns' ] = [ ]
	
if 'features' not in st.session_state:
	st.session_state[ 'features' ] = [ ]

if 'targets' not in st.session_state:
	st.session_state[ 'targets' ] = [ ]


# ----------- Clustering Members

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
	st.session_state[ 'df_processed' ] = df_frame.copy( )
	
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
pd.options.display.float_format = '{:,.4f}'.format

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
				row_idx = st.number_input( 'Select Row Index', min_value=0,
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
		st.markdown( '##### Cardinality' )
		
		v3, v4 = st.columns( 2, border=True )
		with v3:
			cardinality = df_dataset.nunique( dropna=True ).sort_values( ascending=False ).head( 10 )
			fig, ax = plt.subplots( figsize=(6, 4.5) )
			cardinality.sort_values( ascending=True ).plot( kind='barh', ax=ax, width=0.75,
				edgecolor='#0f172a', linewidth=0.9 )
			
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
						label=f'Mean: {mean_val:,.3f}' )
					
					ax.axvline( median_val, linestyle=':', linewidth=1.5,
						label=f'Median: {median_val:,.3f}' )
					
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
					m4.metric( 'Std', f'{float( s.std( ddof=1 ) ):,.3f}' if len( s ) > 1 else '0.000' )
		
		
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
					linewidth=1.5, label=f'Mean: {mean_val:,.3f}' )
				
				ax.axvline( median_val, linestyle=':', linewidth=1.5,
					label=f'Median: {median_val:,.3f}' )
				
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

		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Correlation Structure', help=cfg.CORRELATION_STRUCTURE )
		
		cor_c1, cor_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with cor_c1:
			corr_vars = st.multiselect( 'Variables for Correlation', all_num_cols,
				default=default_pick( all_num_cols, 4 ) )
		
		with cor_c2:
			corr_method = st.radio( 'Method', options=[ 'Pearson', 'Spearman' ],
				horizontal=True, key='desc_corr_method' )
		
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
			""", unsafe_allow_html=True )
		
		# -------------------------------------------------------------------------------------
		# INFERENTIAL SUMMARY
		# -------------------------------------------------------------------------------------
		st.markdown( '##### Summary' )
		
		sum_r1c1, sum_r1c2, sum_r1c3 = st.columns( 3, border=True )
		with sum_r1c1:
			summary_y = st.selectbox( 'Summary Outcome Variable', numeric_columns, key='infer_summary_y' )
		
		with sum_r1c2:
			summary_x = st.selectbox( 'Summary Second Numeric Variable',
				[ '<None>' ] + [ c for c in numeric_columns if c != summary_y ], key='infer_summary_x' )
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
					'Notes': 'Normality assessment'
				})
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
					cramers_v = ( np.sqrt( phi2 / min( c_dim - 1, r_dim - 1 ) )
							if min( c_dim - 1, r_dim - 1 ) > 0
							else np.nan )
					
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
					'Statistic': st.column_config.NumberColumn( 'Statistic', format='%.4f' ),
					'P-Value': st.column_config.NumberColumn( 'P-Value', format='%.4g' ),
					'DoF': st.column_config.NumberColumn( 'DoF', format='%.0f' ),
					'Effect Size': st.column_config.NumberColumn( 'Effect Size', format='%.4f' ),
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
				m2.metric( 'Shapiro W', f'{stat:,.4f}' )
				m3.metric( 'Shapiro p', f'{p_value:,.4g}' )
				
				if p_value < 0.05:
					st.caption( 'Distribution departs from normality at α = 0.05.' )
				else:
					st.caption( 'Distribution does not significantly depart from normality at α = 0.05.' )
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
					g2.metric( 'ANOVA F', f'{f_stat:,.4f}' )
					g3.metric( 'ANOVA p', f'{p_anova:,.4g}' )
					g4.metric( 'Kruskal p', f'{p_kw:,.4g}' )
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
			cat_c1, cat_c2 = st.columns( [ 0.5, 0.5 ], border=True )
			with cat_c1:
				col_cat1 = st.selectbox( 'Select First Categorical Variable', categorical_columns )
				
			with cat_c2:
				col_cat2 = st.selectbox( 'Select Second Categorical Variable',
					[ c for c in categorical_columns if c != col_cat1 ] )
			
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
			""", unsafe_allow_html=True )
		
		all_num_cols = df_numeric.columns.tolist( )
		preferred = [ c for c in all_num_cols if c.lower( ) in ('py', 'cy', 'by') ]
		default_vars = preferred if preferred else default_pick( all_num_cols, 2 )
		vars_sel = st.multiselect( 'Variables to Analyze', all_num_cols, default=default_vars )
		
		if not vars_sel:
			st.info( 'Select at least one numeric variable to run anomaly detection.' )
			st.stop( )
		
		analysis_scale = st.checkbox( 'Use analysis-only standardization', value=False )
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
			iqr_mult = st.slider( 'IQR multiplier', 1.0, 3.0, 1.5, 0.1, help=cfg.IQR_MULTIPLIER )
		
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
		m3.metric(
			'Flag Rate %',
			f'{(100.0 * len( anomalies ) / max( 1, len( df_analysis ) )):,.2f}'
		)
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
				ax.set_title( 'Consensus Strength', fontsize=12, fontweight='bold' )
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
		st.markdown( '##### Anomalous Distributions' )
		
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
					label=f'Mean: {mean_val:,.3f}' )
				
				ax.axvline( median_val, linestyle=':', linewidth=1.4,
					label=f'Median: {median_val:,.3f}' )
				
				ax.set_title( f'{col} — ECDF with Anomalies', fontsize=12, fontweight='bold' )
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
				
				ax.set_title( f'{col} — Violin / Box Summary', fontsize=12, fontweight='bold' )
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
# DATA PLUMBING MODE
# ============================================
elif mode == 'Data Plumbing':
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Data Plumbing' ] )
		st.divider( )
		if df_dataset is None or df_dataset.empty:
			st.warning( 'No dataset loaded.' )
			st.stop( )
		
		df_original = df_dataset.copy( )
		numeric_columns = [ c for c in df_original.columns
		                    if pd.api.types.is_numeric_dtype( df_original[ c ] ) ]
		
		categorical_columns = [ c for c in df_original.columns if c not in numeric_columns ]
		
		# ======================================================================================
		# Data Selection
		# ======================================================================================
		st.markdown( '##### Feature Selection' )
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			selected_columns = st.multiselect( 'Select Features',
				options=df_original.columns.tolist( ),
				default=st.session_state.get( 'features', [ ] ),
				key='plumbing_features' )
		
		with col_c2:
			selected_target_options = [ c for c in df_original.columns
					if c not in selected_columns ]
			
			selected_targets = st.multiselect( 'Select Targets', options=selected_target_options,
				default=st.session_state.get( 'targets', [ ] ),
				key='plumbing_stargets' )
		
		sel_b1, sel_b2, sel_b3 = st.columns( [ 0.34, 0.33, 0.33 ] )
		with sel_b1:
			if st.button( 'Create Working Dataset', key='plumbing_create_dataset',
					use_container_width=True ):
				
				selected_all = selected_columns + [ c for c in selected_targets
						if c not in selected_columns ]
				
				if selected_all:
					df_working = df_original[ selected_all ].copy( )
				else:
					df_working = df_original.copy( )
				
				st.session_state[ 'features' ] = selected_columns.copy( )
				st.session_state[ 'targets' ] = selected_targets.copy( )
				st.session_state[ 'df_working' ] = df_working.copy( )
				commit_frame( df_working )
				st.success( 'Working dataframe created.' )
		
		with sel_b2:
			if st.button( 'Reset Working Dataset', key='plumbing_reset_working_dataset',
					use_container_width=True ):
				
				df_working = st.session_state.get( 'df_working' )
				if df_working is None or df_working.empty:
					df_working = st.session_state[ 'df_original' ].copy( )
				commit_frame( df_working.copy( ) )
				st.success( 'Working dataframe reset.' )
		
		with sel_b3:
			if st.button( 'Reset To Original', key='plumbing_reset_to_original',
					use_container_width=True ):
				
				st.session_state[ 'features' ] = [ ]
				st.session_state[ 'targets' ] = [ ]
				working_to_original( )
				st.success( 'Reset back to df_original.' )
		
		df_working = get_working_frame( )
		active_numeric_columns = get_numeric_columns( df_working )
		categorical_columns = get_categorical_columns( df_working )
		
		st.caption( f'Working rows: {len( df_working ):,}| Working columns: {len( df_working.columns ):,}')
		
		# ======================================================================================
		# Data Processing
		# ======================================================================================
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Data Transformations' )
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			
			with st.expander( label='Data Scaling', icon='⚖️', key='scalers' ):
				
				with st.expander( 'Standard Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_standard_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_standard_scaler_apply',
								use_container_width=True ):
							
							if scale_cols:
								df_processed = get_working_frame( )
								scaler = StandardScaler( )
								result = scaler.train_transform( df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'StandardScaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_standard_scaler_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Min-Max Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_minmax_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_minmax_scaler_apply',
								use_container_width=True ):
							
							if scale_cols:
								df_processed = get_working_frame( )
								scaler = MinMaxScaler( )
								result = scaler.train_transform( df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'MinMaxScaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_minmax_scaler_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_robust_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_robust_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								df_processed = get_working_frame( )
								scaler = RobustScaler( )
								result = scaler.train_transform( df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'RobustScaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_robust_scaler_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Normal Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ],
						index=1, key='plumbing_normal_scaler_norm' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply', key='plumbing_normal_scaler_apply',
								use_container_width=True ):
							
							if scale_cols:
								df_processed = get_working_frame( )
								scaler = NormalScaler( norm=norm )
								result = scaler.train_transform( 
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'NormalScaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_normal_scaler_reset', 
								use_container_width=True ):
							
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_maxabs_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_maxabs_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								df_processed = get_working_frame( )
								scaler = MaxAbsScaler( )
								result = scaler.train_transform( df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'MaxAbsScaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_maxabs_scaler_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
			
			with st.expander( label='Data Imputation', icon='🛡️', key='imputers' ):
				
				with st.expander( 'Mean Imputer', expanded=False ):
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='plumbing_mean_imputer_indicator' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_mean_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								df_processed = get_working_frame( )
								imputer = MeanImputer( strategy='mean', add_indicator=add_indicator )
								result = imputer.train_transform( df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'mean_imputer' )
								commit_frame( df_processed )
								st.success( 'MeanImputer applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_mean_imputer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1,
						value=5, step=1, key='plumbing_nearest_imputer_neighbors' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply', key='plumbing_nearest_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								df_processed = get_working_frame( )
								imputer = NearestImputer( neighbors=int( neighbors ) )
								result = imputer.train_transform( df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'nearest_imputer' )
								
								commit_frame( df_processed )
								st.success( 'Nearest Imputer applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_nearest_imputer_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset Data' )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_iterative_imputer_cols' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1,
						value=10, step=1, key='plumbing_iterative_imputer_max_iter' )
					
					random_state = st.number_input( 'Random State', min_value=0,
						value=0, step=1, key='plumbing_iterative_imputer_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Iterative Imputer', key='plumbing_iterative_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								df_processed = get_working_frame( )
								imputer = IterativeImputer( max_iter=int( max_iter ),
									random_state=int( random_state ) )
								result = imputer.train_transform( df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'iterative_imputer' )
								commit_frame( df_processed )
								st.success( 'Iterative Imputer applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_iterative_imputer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset Data' )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_simple_imputer_cols' )
					
					strategy = st.selectbox( 'Strategy',
						options=[ 'mean', 'median', 'most_frequent', 'constant' ],
						key='plumbing_simple_imputer_strategy' )
					
					fill_value = st.text_input( 'Fill Value', value='0.0',
						key='plumbing_simple_imputer_fill_value' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='plumbing_simple_imputer_indicator' )
					
					keep_empty_features = st.checkbox( 'Keep Empty Features', value=False,
						key='plumbing_simple_imputer_keep_empty' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SimpleImputer', key='classification_simple_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								df_processed = get_working_frame( )
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
									add_indicator=add_indicator, keep_empty_features=keep_empty_features )
								
								result = imputer.train_transform( df_input.to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols,
									result, 'simple_imputer' )
								commit_frame( df_processed )
								st.success( 'Simple Imputer Applied' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_simple_imputer_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.success( 'Reset to Original.' )
			
			with st.expander( label='Data Encoding', icon='🔠', key='encoders' ):
				
				with st.expander( 'One-Hot Encoder', expanded=False ):
					encode_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='plumbing_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='plumbing_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0,
						key='plumbing_onehot_unknown' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_onehot_apply', use_container_width=True ):
							if encode_cols:
								df_processed = get_working_frame( )
								encoder = OneHotEncoder( sparse=bool( sparse ), unknown=unknown )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, encode_cols,
									result, 'onehot' )
								commit_frame( df_processed )
								st.success( 'OneHotEncoder applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_onehot_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					encode_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='plumbing_ordinal_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_ordinal_apply', use_container_width=True ):
							if encode_cols:
								df_processed = get_working_frame( )
								encoder = OrdinalEncoder( )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								df_processed[ encode_cols ] = result
								commit_frame( df_processed )
								st.success( 'Ordinal Encoder Applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_ordinal_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Label Encoder', expanded=False ):
					target_col = st.selectbox( 'Column', options=df_working.columns.tolist( ),
						key='plumbing_label_encoder_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_label_encoder_apply',
								use_container_width=True ):
							if target_col:
								df_processed = get_working_frame( )
								encoder = LabelEncoder( )
								result = encoder.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed[ target_col ] = result
								commit_frame( df_processed )
								st.success( 'Label Encoder Applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_label_encoder_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Target Encoder', expanded=False ):
					encode_cols = st.multiselect( 'Categorical Feature Columns',
						options=categorical_columns, key='plumbing_target_encoder_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns.tolist( ),
						key='plumbing_target_encoder_target_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_target_encoder_apply',
								use_container_width=True ):
							if encode_cols and target_col:
								df_processed = get_working_frame( )
								encoder = TargetEncoder( )
								X = df_processed[ encode_cols ].astype( str ).to_numpy( )
								y = df_processed[ target_col ].to_numpy( )
								result = encoder.train_transform( X, y )
								df_processed = replace_columns( df_processed, encode_cols, result,
									'target_encoder' )
								commit_frame( df_processed )
								st.success( 'Target Encoder Applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_target_encoder_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					poly_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4,
						value=2, key='plumbing_polynomial_degree' )
					
					interaction = st.checkbox( 'Interaction Only', value=True,
						key='plumbing_polynomial_interaction' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply', key='plumbing_polynomial_apply',
								use_container_width=True ):
							if poly_cols:
								df_processed = get_working_frame( )
								encoder = PolynomialFeatures(
									degree=int( degree ),
									interaction=bool( interaction ) )
								
								result = encoder.train_transform( df_processed[ poly_cols ].to_numpy( ))
								df_processed = replace_columns( df_processed, poly_cols, result,
									'polynomial' )
								commit_frame( df_processed )
								st.success( 'PolynomialFeatures applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_polynomial_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
		
		with feature_c2:
			
			with st.expander( label='Data Transformation', icon='↔️', key='transformers' ):
				
				with st.expander( 'Binarizer', expanded=False ):
					transform_cols = st.multiselect( 'Columns',
						options=numeric_columns, key='plumbing_binarizer_cols' )
					
					threshold = st.number_input( 'Threshold', value=0.0, step=0.1,
						key='plumbing_binarizer_threshold' )
					
					copy = st.checkbox( 'Copy', value=True, key='plumbing_binarizer_copy' )
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Binarizer', key='plumbing_binarizer_apply',
								use_container_width=True ):
							
							if transform_cols:
								df_processed = get_working_frame( )
								transformer = Binarizer( threshold=float( threshold ),
									copy=bool( copy ) )
								result = transformer.train_transform(
									df_processed[ transform_cols ].to_numpy( ) )
								
								df_processed[ transform_cols ] = result
								commit_frame( df_processed )
								st.success( 'Binarizer applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_binarizer_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column', options=df_working.columns.tolist( ),
						key='plumbing_label_binarizer_col' )
					
					pos_label = st.number_input( 'Positive Label', value=1, step=1,
						key='plumbing_label_binarizer_pos' )
					
					neg_label = st.number_input( 'Negative Label', value=0, step=1,
						key='plumbing_label_binarizer_neg' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='plumbing_label_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply LabelBinarizer', key='plumbing_label_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = get_working_frame( )
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ), sparse_output=bool( sparse_output ) )
								result = transformer.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, [ target_col ], result,
									'label_binarizer' )
								commit_frame( df_processed )
								st.success( 'Label Binarizer Applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_label_binarizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column', options=df_working.columns.tolist( ),
						key='plumbing_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='plumbing_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='plumbing_multilabel_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_multilabel_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = get_working_frame( )
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
						if st.button( 'Reset', key='plumbing_multilabel_binarizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'TFIDF Transformer', expanded=False ):
					text_count_cols = st.multiselect( 'Count Matrix Columns',
						options=numeric_columns, key='plumbing_tfidf_transformer_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', None ],
						index=1, key='plumbing_tfidf_transformer_norm' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='plumbing_tfidf_transformer_use_idf' )
					
					smooth_idf = st.checkbox( 'Smooth IDF', value=True,
						key='plumbing_tfidf_transformer_smooth_idf' )
					
					sublinear_tf = st.checkbox( 'Sublinear TF', value=False,
						key='plumbing_tfidf_transformer_sublinear' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_tfidf_transformer_apply',
								use_container_width=True ):
							
							if text_count_cols:
								df_processed = get_working_frame( )
								transformer = TfidfTransformer( norm=norm, use_idf=bool( use_idf ),
									smooth_idf=bool( smooth_idf ), sublinear_tf=bool( sublinear_tf ))
								
								result = transformer.train_transform(
									df_processed[ text_count_cols ].apply(
										pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, text_count_cols, result,
									'tfidf_transformer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Transformer Applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_tfidf_transformer_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Column Transformer', expanded=False ):
					numeric_columns = st.multiselect( 'Numeric Columns', options=numeric_columns,
						key='plumbing_column_transformer_numeric_columns' )
					
					categorical_columns = st.multiselect( 'Categorical Columns',
						options=categorical_columns,
						key='plumbing_column_transformer_categorical_columns' )
					
					numeric_transform = st.selectbox( 'Numeric Transformer',
						options=[ 'StandardScaler', 'MinMaxScaler', 'RobustScaler',
						          'MaxAbsScaler', 'Binarizer', 'None' ],
						key='plumbing_column_transformer_numeric_transform' )
					
					categorical_transform = st.selectbox( 'Categorical Transformer',
						options=[ 'OneHotEncoder', 'OrdinalEncoder', 'None' ],
						key='plumbing_column_transformer_categorical_transform' )
					
					remainder = st.selectbox( 'Remainder', options=[ 'drop', 'passthrough' ],
						key='plumbing_column_transformer_remainder' )
					
					sparse_threshold = st.slider( 'Sparse Threshold', min_value=0.0, max_value=1.0,
						value=0.3, key='plumbing_column_transformer_sparse_threshold' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Column Transformer', key='plumbing_column_transformer_apply',
								use_container_width=True ):
							df_processed = get_working_frame( )
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
									categorical_model = OneHotEncoder(
										sparse=False, unknown='ignore' ).model
								else:
									categorical_model = OrdinalEncoder( ).model
								
								transformers.append( 'categorical', categorical_model,
									categorical_columns )
							
							if transformers:
								transformer = ColumnTransformer( transformers=transformers,
									remainder=remainder, sparse_threshold=float( sparse_threshold ),
									n_jobs=None, transformer_weights=None, verbose=False )
								
								result = transformer.train_transform( df_processed )
								df_processed = normalize_result_frame( result=result, index=df_processed.index,
									prefix='column_transformer', columns=None )
								
								commit_frame( df_processed )
								st.success( 'ColumnTransformer applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_column_transformer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'TFIDF Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns.tolist( ),
						key='plumbing_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='plumbing_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='plumbing_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='plumbing_tfidf_vectorizer_use_idf' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_tfidf_vectorizer_apply',
								use_container_width=True ):
							
							if text_cols:
								df_processed = get_working_frame( )
								transformer = TfidfVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int( max_features ),
									use_idf=bool( use_idf ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols, transformer,
									'tfidf_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Vectorizer Applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_tfidf_vectorizer_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns.tolist( ),
						key='plumbing_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='plumbing_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0,
						step=1, key='plumbing_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='plumbing_count_vectorizer_binary' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply', key='plumbing_count_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = get_working_frame( )
								transformer = CountVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int( max_features ),
									binary=bool( binary ) )
								df_processed = apply_text_vectorizer( df_processed, text_cols, transformer,
									'count_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'Count Vectorizer Applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_count_vectorizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns.tolist( ),
						key='plumbing_hash_vectorizer_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='plumbing_hash_vectorizer_n_features' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3,
						value=1, key='plumbing_hash_vectorizer_ngram_max' )
					
					binary = st.checkbox( 'Binary', value=False,
						key='plumbing_hash_vectorizer_binary' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='plumbing_hash_vectorizer_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_hash_vectorizer_apply',
								use_container_width=True ):
							
							if text_cols:
								df_processed = get_working_frame( )
								transformer = HashVectorizer( num=int( n_features ),
									ngram_range=(1, int( ngram_max )), binary=bool( binary ),
									alternate_sign=bool( alternate_sign ) )
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'hash_vectorizer' )
								commit_frame( df_processed )
								st.success( 'HashVectorizer Applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_hash_vectorizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					dict_cols = st.multiselect( 'Columns', options=df_working.columns.tolist( ),
						key='plumbing_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='plumbing_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='plumbing_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='plumbing_dict_vectorizer_sort' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_dict_vectorizer_apply',
								use_container_width=True ):
							
							if dict_cols:
								df_processed = get_working_frame( )
								transformer = DictVectorizer( dtype=np.float64, separator=separator,
									sparse=bool( sparse ), sort=bool( sort )  )
								
								df_processed = apply_dict_transform( df_processed, dict_cols,
									transformer, 'dict_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'DictVectorizer applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_dict_vectorizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					hash_cols = st.multiselect( 'Columns', options=df_working.columns.tolist( ),
						key='plumbing_feature_hasher_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='plumbing_feature_hasher_n_features' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='plumbing_feature_hasher_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply', key='plumbing_feature_hasher_apply',
								use_container_width=True ):
							if hash_cols:
								df_processed = get_working_frame( )
								transformer = FeatureHasher( n_features=int( n_features ),
									input_type='dict', dtype=np.float64,
									alternate_sign=bool( alternate_sign ) )
								
								df_processed = apply_dict_transform( df_processed, hash_cols,
									transformer, 'feature_hasher' )
								commit_frame( df_processed )
								st.success( 'FeatureHasher applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_feature_hasher_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
			
			with st.expander( label='Feature Selection', icon='🔍', key='selectors' ):
				
				with st.expander( 'Variance Threshold', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0,
						step=0.01, key='plumbing_variance_threshold_value' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_variance_threshold_apply',
								use_container_width=True ):
							if select_cols:
								df_processed = get_working_frame( )
								selector = VarianceThreshold( thresh=float( threshold ) )
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result,
									'variance_threshold' )
								
								commit_frame( df_processed )
								st.success( 'VarianceThreshold applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_variance_threshold_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Canonical Correlation Analysis', expanded=False ):
					X_cols = st.multiselect( 'Predictor Columns', options=numeric_columns,
						key='plumbing_cca_x_cols' )
					
					y_cols = st.multiselect( 'Target Columns', options=numeric_columns,
						key='plumbing_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2,
						step=1, key='plumbing_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True, key='plumbing_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500,
						step=1, key='plumbing_cca_max_iter' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply', key='plumbing_cca_apply', use_container_width=True ):
							if X_cols and y_cols:
								df_processed = get_working_frame( )
								selector = CCA( num=int( n_components ), scale=bool( scale ),
									size=int( max_iter ) )
								
								result = selector.train_transform( df_processed[ X_cols ].to_numpy( ),
									df_processed[ y_cols ].to_numpy( ) )
								
								df_result = normalize_result_frame( result=result,
									index=df_processed.index, prefix='cca', columns=None )
								
								df_processed = pd.concat(
									[ df_processed.drop( columns=X_cols + y_cols, errors='ignore' ),
											df_result ], axis=1 )
								commit_frame( df_processed )
								st.success( 'CCA Applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_cca_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Principle Component Analysis', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='plumbing_pca_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2,
						step=1, key='plumbing_pca_components' )
					
					solver = st.selectbox( 'SVD Solver',
						options=[ 'auto', 'full', 'randomized', 'covariance_eigh', 'arpack' ],
						key='plumbing_pca_solver' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='plumbing_pca_apply', use_container_width=True ):
							if select_cols:
								df_processed = get_working_frame( )
								selector = PCA( num=int( n_components ),
									solver=solver )
								
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result, 'pca' )
								commit_frame( df_processed )
								st.success( 'PCA applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_pca_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Select-Best', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns',
						options=numeric_columns, key='plumbing_selectbest_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns.tolist( ),
						key='plumbing_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
								'mutual_info_regression' ], key='plumbing_selectbest_score_name' )
					
					k_best = st.number_input( 'K', min_value=1, value=5, step=1,
						key='plumbing_selectbest_k' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply', key='plumbing_selectbest_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = get_working_frame( )
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
						if st.button( 'Reset', key='plumbing_selectbest_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Select-Percent', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns',
						options=numeric_columns, key='plumbing_selectpercent_x_cols' )
					
					target_col = st.selectbox( 'Target Column',
						options=df_working.columns.tolist( ),
						key='plumbing_selectpercent_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression',
						          'mutual_info_classif', 'mutual_info_regression' ],
						key='plumbing_selectpercent_score_name' )
					
					percentile = st.slider( 'Percentile', min_value=1, max_value=100,
						value=10, key='plumbing_selectpercent_percentile' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply SelectPercent', key='plumbing_selectpercent_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = get_working_frame( )
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
						if st.button( 'Reset', key='plumbing_selectpercent_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Sequential Back Selection', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='plumbing_sbs_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns.tolist( ),
						key='plumbing_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='plumbing_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='plumbing_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1,
						step=1, key='plumbing_sbs_random_state' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply', key='plumbing_sbs_apply', use_container_width=True ):
							if X_cols and target_col:
								df_processed = get_working_frame( )
								selector = SBS( classifier=None, k_features=int( k_features ),
									test_size=float( test_size ), random_state=int( random_state ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'sbs' )
								
								commit_frame( df_processed )
								st.success( 'SBS applied.' )
					
					with a2:
						if st.button( 'Reset', key='plumbing_sbs_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Recursive Feature Elimination', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='plumbing_rfe_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns.tolist( ),
						key='plumbing_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain',
						min_value=1, value=1, step=1, key='plumbing_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0,
						step=1, key='plumbing_rfe_verbose' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply', key='plumbing_rfe_apply', use_container_width=True ):
							if X_cols and target_col:
								df_processed = get_working_frame( )
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
						if st.button( 'Reset', key='plumbing_rfe_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
		
		# ======================================================================================
		# Data Transformation
		# ======================================================================================
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Feature-Engineered Dataset' )
		st.data_editor( get_working_frame( ), key='engineering_table',
			use_container_width=True, height=420 )
		
		st.divider( )
		st.subheader( 'Download or Export' )
		df_export = get_working_frame( )
		st.download_button( label='Download Feature-Engineered Dataset (CSV)',
			data=df_export.to_csv( index=False ), file_name='feature_engineered_data.csv',
			mime='text/csv' )

# ============================================
# FEATURE ENGINEERING MODE
# ============================================
elif mode == 'Feature Engineering':
	df_dataset = st.session_state.df_dataset
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Feature Engineering' ] )
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
		numeric_columns = [ c for c in df_original.columns if
		                    df_original[ c ].dtype.kind in { 'i', 'f' } ]
		
		categorical_columns = [ c for c in df_original.columns if c not in numeric_columns ]
		
		# ------------------------------------------------------------------
		# Column selection
		# ------------------------------------------------------------------
		st.markdown( '##### Column Selection' )
		selected_columns = st.multiselect( 'Select columns for feature engineering',
			options=df_original.columns.tolist( ) )
		
		if not selected_columns:
			st.info( 'Select one or more columns to begin.' )
			st.stop( )
		
		df_features = df_original[ selected_columns ].copy( )
		
		# ======================================================================================
		# Missing Value Handling
		# ======================================================================================
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Missing Value Handling' )
		sel_c1, sel_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with sel_c1:
			impute_columns = st.multiselect( 'Columns to Impute', options=df_features.columns.tolist( ) )
		
		with sel_c2:
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
		st.data_editor( df_features.head( ), key='imputation_data' )
		
		# ======================================================================================
		# Encoding
		# ======================================================================================
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Encoding' )
		enc_c1, enc_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with enc_c1:
			encode_columns = st.multiselect( 'Categorical Columns to Encode',
				options=[ c for c in df_features.columns if c in categorical_columns ] )
		
		with enc_c2:
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
		st.data_editor( df_features.head( ), key='encoding_data' )
		
		# ======================================================================================
		# Scaling / Normalization
		# ======================================================================================
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Scaling & Normalization' )
		sca_c1, sca_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with sca_c1:
			scale_columns = st.multiselect( 'Numeric Columns to Scale',
				options=[ c for c in df_features.columns if c in numeric_columns ] )
		
		with sca_c2:
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
		st.data_editor( df_features.head( ), key='scaling_data' )
		
		# ======================================================================================
		# Feature Generation
		# ======================================================================================
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Feature Generation' )
		fet_c1, fet_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with fet_c1:
			poly_columns = st.multiselect( 'Columns for Polynomial Features',
				options=[ c for c in df_features.columns if c in numeric_columns ] )
		
		with fet_c2:
			poly_degree = st.slider( 'Polynomial Degree', min_value=2, max_value=4, value=2 )
			if poly_columns:
				X_poly = df_features[ poly_columns ].to_numpy( )
				poly = PolynomialFeatures( degree=poly_degree )
				X_poly_out = poly.train_transform( X_poly )
				df_polynomial = pd.DataFrame( X_poly_out, index=df_features.index )
				df_features = df_features.drop( columns=poly_columns )
				df_features = pd.concat( [ df_features, df_polynomial ], axis=1 )
			
		st.caption( 'Polynomial Feature preview (First 5 Rows)' )
		st.data_editor( df_features.head( ), key='feature_data' )
		
		# ======================================================================================
		# Apply / Export
		# ======================================================================================
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Apply or Export' )
		
		app_c1, app_c2, app_c3 = st.columns( [ 0.2, 0.5, 0.3 ] )
		with app_c1:
			if st.button( 'Apply Feature Engineering' ):
				st.session_state[ 'df_features' ] = df_features.copy( )
				st.success( 'Feature-Engineered Dataset Stored in Session State.' )
		
		with app_c2:
			st.download_button( label='Download Feature-Engineered Dataset (CSV)',
				data=df_features.to_csv( index=False ), file_name='feature_engineered_data.csv',
				mime='text/csv' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Feature-Transformed Data')
		if st.download_button:
			st.data_editor( df_features )

# ============================================
# CLASSIFICATION MODE
# ============================================
elif mode == 'Classifications':
	df_original = st.session_state.get( 'df_dataset', None )
	df_dataset = st.session_state.get( 'df_dataset', None )
	df_working = st.session_state.get( 'df_working', None )
	df_processed = st.session_state.get( 'df_processed', None )
	df_classification = st.session_state.get( 'df_classification', None )
	numeric_columns = st.session_state.get( 'numeric_columns', [ ] )
	categorical_columns = st.session_state.get( 'categorical_columns', [ ] )
	features = st.session_state.get( 'features', [ ] )
	targets = st.session_state.get( 'targets', [ ] )
	left, center, right = st.columns( [ 0.25, 0.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Classifications' ] )
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
			st.warning( '⚠️ Classification requires numeric features and a categorical target.' )
			st.stop( )
		
		df_classification = st.session_state.get( 'df_classification', df_original.copy( ) ).copy( )
		
		# ======================================================================================
		# Data Selection
		# ======================================================================================
		st.markdown( '##### Data Selection' )
		st.caption( f'Records: {len( df_original ):,}  | Fields: {len( df_original.columns ):,}' )
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			features = st.multiselect( 'Select Features', options=categorical_columns,
				default=[ c for c in st.session_state.get( 'features', [ ] )
						if c in numeric_columns ], key='classification_features' )
		
		with col_c2:
			target_options = [ c for c in numeric_columns if c not in features ]
			targets = st.multiselect( 'Select Targets', options=target_options,
				default=[ c for c in st.session_state.get( 'targets', [ ] )
						if c in target_options ], key='classification_targets' )
		
		sel_b1, sel_b2, sel_b3 = st.columns( [ 0.34, 0.33, 0.33 ] )
		with sel_b1:
			if st.button( 'Create Working Dataset', key='classification_create_dataset',
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
				st.success( 'Working Dataframe Created.' )
		
		with sel_b2:
			if st.button( 'Reset Working Dataset', key='classification_reset_working_dataset',
					use_container_width=True ):
				
				st.session_state[ 'df_working' ] = df_working.copy( )
				commit_frame( df_working )
				st.success( 'Working dataframe reset.' )
		
		with sel_b3:
			if st.button( 'Reset To Original', key='classification_reset_to_original',
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
		st.markdown( '##### Working Data')
		st.caption( f'Rows: {len( df_working ):,}  |  Columns: {len( df_working.columns ):,}' )
		
		render_table( df_working )
		
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
		st.markdown( '##### Data Transformations' )
		
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			
			with st.expander( label='Data Scaling', icon='⚖️', key='classification_scalers' ):
				
				with st.expander( 'Standard Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=targets,
						key='classification_standard_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_standard_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = StandardScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								st.success( 'Standard Scaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='classification_standard_scaler_reset',
								use_container_width=True ):
							
							df_processed = pd.DataFrame( )
							st.success( 'Reset to Working Dataset.' )
				
				with st.expander( 'Min-Max Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='classification_minmax_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_minmax_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MinMaxScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'Min-Max Scaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='classification_minmax_scaler_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_processed' ] = df_working.copy( )
							st.success( 'Reset to Working' )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='classification_robust_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_robust_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = RobustScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'RobustScaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='classification_robust_scaler_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_processed' ] = df_working.copy( )
							st.success( 'Reset to Working' )
				
				with st.expander( 'Normal Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='classification_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ],
						index=1, key='classification_normal_scaler_norm' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_normal_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = NormalScaler( norm=norm )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'NormalScaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='classification_normal_scaler_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='classification_maxabs_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_maxabs_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MaxAbsScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'MaxAbsScaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='classification_maxabs_scaler_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
			
			with st.expander( label='Data Imputation', icon='➕', key='classification_imputers' ):
				
				with st.expander( 'Mean Imputer', expanded=False ):
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='classification_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='classification_mean_imputer_indicator' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_mean_imputer_apply',
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
						if st.button( 'Reset', key='classification_mean_imputer_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='classification_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1,
						value=5, step=1, key='classification_nearest_imputer_neighbors' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_nearest_imputer_apply',
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
						if st.button( 'Reset', key='classification_nearest_imputer_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset Data' )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='classification_iterative_imputer_cols' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1,
						value=10, step=1, key='classification_iterative_imputer_max_iter' )
					
					random_state = st.number_input( 'Random State', min_value=0,
						value=0, step=1, key='classification_iterative_imputer_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Iterative Imputer',
								key='classification_iterative_imputer_apply',
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
						if st.button( 'Reset', key='classification_iterative_imputer_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset Data' )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					impute_cols = st.multiselect( 'Columns',
						options=numeric_columns,
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
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SimpleImputer', key='classification_simpleimputer_apply',
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
						if st.button( 'Reset', key='classification_simple_imputer_reset',
								use_container_width=True ):
							
							
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
			
			with st.expander( label='Data Encoding', icon='🔣', key='classification_encoders' ):
				
				with st.expander( 'One-Hot Encoder', expanded=False ):
					encode_cols = st.multiselect( 'Columns', options=features,
						key='classification_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='classification_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0,
						key='classification_onehot_unknown' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_onehot_apply',
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
						if st.button( 'Reset', key='classification_onehot_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					encode_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='classification_ordinal_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_ordinal_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OrdinalEncoder( )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								df_processed[ encode_cols ] = result
								commit_frame( df_processed )
								st.success( 'Ordinal Encoder Applied.' )
					
					with a2:
						if st.button( 'Reset', key='classification_ordinal_reset',
								use_container_width=True ):
							
							df_processed = pd.DataFrame( )
							st.success( 'Reset to Working' )
				
				with st.expander( 'Label Encoder', expanded=False ):
					target_col = st.selectbox( 'Column',
						options=categorical_columns,
						key='classification_label_encoder_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_label_encoder_apply',
								use_container_width=True ):
							
							if target_col:
								encoder = LabelEncoder( )
								result = encoder.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed[ target_col ] = result
								commit_frame( df_processed )
								st.success( 'Label Encoder Applied.' )
					
					with a2:
						if st.button( 'Reset', key='classification_label_encoder_reset',
								use_container_width=True ):
							
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Target Encoder', expanded=False ):
					encode_cols = st.multiselect( 'Categorical Feature Columns',
						options=categorical_columns, key='classification_target_encoder_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='classification_target_encoder_target_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_target_encoder_apply',
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
						if st.button( 'Reset', key='classification_target_encoder_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					poly_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='classification_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4,
						value=2, key='classification_polynomial_degree' )
					
					interaction = st.checkbox( 'Interaction Only', value=True,
						key='classification_polynomial_interaction' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_polynomial_apply',
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
						if st.button( 'Reset', key='classification_polynomial_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
		with feature_c2:
			
			with st.expander( label='Data Transformation', icon='⚡', key='classification_transformers' ):
				
				with st.expander( 'Binarizer', expanded=False ):
					transform_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='classification_binarizer_cols' )
					
					threshold = st.number_input( 'Threshold', value=0.0, step=0.1,
						key='classification_binarizer_threshold' )
					
					copy = st.checkbox( 'Copy', value=True, key='classification_binarizer_copy' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Binarizer',
								key='classification_binarizer_apply',
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
						if st.button( 'Reset', key='classification_binarizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column',
						options=categorical_columns,
						key='classification_label_binarizer_col' )
					
					pos_label = st.number_input( 'Positive Label', value=1, step=1,
						key='classification_label_binarizer_pos' )
					
					neg_label = st.number_input( 'Negative Label', value=0, step=1,
						key='classification_label_binarizer_neg' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='classification_label_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply LabelBinarizer',
								key='classification_label_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ), sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, [ target_col ], result,
									'label_binarizer' )
								commit_frame( df_processed )
								st.success( 'Label Binarizer Applied.' )
					
					with a2:
						if st.button( 'Reset', key='classification_label_binarizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column',
						options=categorical_columns,
						key='classification_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='classification_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='classification_multilabel_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_multilabel_binarizer_apply',
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
						if st.button( 'Reset', key='classification_multilabel_binarizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'TFIDF Transformer', expanded=False ):
					text_count_cols = st.multiselect( 'Count Matrix Columns',
						options=numeric_columns,
						key='classification_tfidf_transformer_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', None ],
						index=1, key='classification_tfidf_transformer_norm' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='classification_tfidf_transformer_use_idf' )
					
					smooth_idf = st.checkbox( 'Smooth IDF', value=True,
						key='classification_tfidf_transformer_smooth_idf' )
					
					sublinear_tf = st.checkbox( 'Sublinear TF', value=False,
						key='classification_tfidf_transformer_sublinear' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_tfidf_transformer_apply',
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
						if st.button( 'Reset', key='classification_tfidf_transformer_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_processed' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Column Transformer', expanded=False ):
					numeric_columns = st.multiselect( 'Numeric Columns', options=numeric_columns,
						key='classification_column_transformer_numeric_columns' )
					
					categorical_columns = st.multiselect( 'Categorical Columns',
						options=categorical_columns,
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
								
								commit_frame( df_processed )
								st.success( 'ColumnTransformer applied.' )
					
					with a2:
						if st.button( 'Reset', key='classification_column_transformer_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
			
			with st.expander( label='Feature Extration', icon='⛏️', key='classification_extractors' ):
				
				with st.expander( 'TFIDF Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns',
						options=categorical_columns,
						key='classification_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='classification_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='classification_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='classification_tfidf_vectorizer_use_idf' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_tfidf_vectorizer_apply',
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
						if st.button( 'Reset', key='classification_tfidf_vectorizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns',
						options=categorical_columns,
						key='classification_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='classification_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0,
						step=1, key='classification_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='classification_count_vectorizer_binary' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_count_vectorizer_apply',
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
						if st.button( 'Reset', key='classification_count_vectorizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns',
						options=categorical_columns,
						key='classification_hash_vectorizer_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='classification_hash_vectorizer_n_features' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3,
						value=1, key='classification_hash_vectorizer_ngram_max' )
					
					binary = st.checkbox( 'Binary', value=False,
						key='classification_hash_vectorizer_binary' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='classification_hash_vectorizer_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_hash_vectorizer_apply',
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
						if st.button( 'Reset', key='classification_hash_vectorizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					dict_cols = st.multiselect( 'Columns',
						options=categorical_columns,
						key='classification_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='classification_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='classification_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='classification_dict_vectorizer_sort' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_dict_vectorizer_apply',
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
						if st.button( 'Reset', key='classification_dict_vectorizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					hash_cols = st.multiselect( 'Columns',
						options=categorical_columns,
						key='classification_feature_hasher_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='classification_feature_hasher_n_features' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='classification_feature_hasher_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_feature_hasher_apply',
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
						if st.button( 'Reset', key='classification_feature_hasher_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.session_state[ 'df_processed' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
			
			with st.expander( label='Dimensionality Reduction', icon='🎚️', key='classification_selectors' ):
				
				with st.expander( 'Variance Threshold', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='classification_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0,
						step=0.01, key='classification_variance_threshold_value' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_variance_threshold_apply',
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
						if st.button( 'Reset', key='classification_variance_threshold_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.session_state[ 'df_processed' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Canonical Correlation Analysis', expanded=False ):
					X_cols = st.multiselect( 'Predictor Columns', options=numeric_columns,
						key='classification_cca_x_cols' )
					
					y_cols = st.multiselect( 'Target Columns', options=numeric_columns,
						key='classification_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2,
						step=1, key='classification_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True,
						key='classification_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500,
						step=1, key='classification_cca_max_iter' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_cca_apply',
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
								
								commit_frame( df_processed )
								st.success( 'CCA Applied.' )
					
					with a2:
						if st.button( 'Reset', key='classification_cca_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.session_state[ 'df_processed' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Principle Component Analysis', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='classification_pca_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2,
						step=1, key='classification_pca_components' )
					
					solver = st.selectbox( 'SVD Solver',
						options=[ 'auto', 'full', 'randomized', 'covariance_eigh', 'arpack' ],
						key='classification_pca_solver' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_pca_apply',
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
						if st.button( 'Reset', key='classification_pca_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.session_state[ 'df_processed' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Select-Best', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='classification_selectbest_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='classification_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
								'mutual_info_regression' ],
						key='classification_selectbest_score_name' )
					
					k_best = st.number_input( 'K', min_value=1, value=5, step=1,
						key='classification_selectbest_k' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_selectbest_apply',
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
						if st.button( 'Reset', key='classification_selectbest_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.session_state[ 'df_processed' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Select-Percent', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='classification_selectpercent_x_cols' )
					
					target_col = st.selectbox( 'Target Column',
						options=categorical_columns,
						key='classification_selectpercent_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
								'mutual_info_regression' ],
						key='classification_selectpercent_score_name' )
					
					percentile = st.slider( 'Percentile', min_value=1, max_value=100, value=10,
						key='classification_selectpercent_percentile' )
					
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
								commit_frame( df_processed )
								st.success( 'SelectPercent applied.' )
					
					with a2:
						if st.button( 'Reset', key='classification_selectpercent_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.session_state[ 'df_processed' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Sequential Back Selection', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='classification_sbs_x_cols' )
					
					target_col = st.selectbox( 'Target Column',
						options=categorical_columns,
						key='classification_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='classification_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='classification_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1,
						step=1, key='classification_sbs_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_sbs_apply',
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
						if st.button( 'Reset', key='classification_sbs_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.session_state[ 'df_processed' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Recursive Feature Elimination', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='classification_rfe_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='classification_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain',
						min_value=1, value=1, step=1, key='classification_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0,
						step=1, key='classification_rfe_verbose' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='classification_rfe_apply',
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
						if st.button( 'Reset', key='classification_rfe_reset',
								use_container_width=True ):
							
							working_to_original( )
							st.session_state[ 'df_processed ' ] = get_working_frame( ).copy( )
							st.success( 'Reset to Original.' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Processed Data' )
		st.render_table( df_processed )
		
		# ------------------------------------------------------------------
		# MODEL SELECTION
		# ------------------------------------------------------------------
		model_map = \
		{
				'Perceptron': Perceptron,
				'Least Squares Classifier': LeastSquares,
				'Logistic Regression': LogisticRegression,
				'Decision Tree': DecisionTree,
				'Support Vector Machine': SupportVector,
				'Random Forest': RandomForest,
				'k-Nearest Neighbors': NearestNeighbor,
				'Bagging': BaggingModel,
				'AdaBoost': AdaptiveBoost,
				'Gradient Boosting': GradientBoost
		}
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Model Selection & Configuration' )
		
		
		mdl_c1, mdl_c2, mdl_c3 = st.columns( [ 0.33, 0.33, 0.33 ], border=True )
		with mdl_c1:
			model_name = st.selectbox( 'Select Classification Model', list( model_map.keys( ) ) )
			model = model_map[ model_name ]( )
		
		with mdl_c2:
			test_sz = st.slider( 'Test Set Size (%)', 10, 30, 20, key='classifications-1' ) / 100.0
		
		with mdl_c3:
			random_state = st.number_input( 'Random State', value=42, step=1,
				key='classifications-2' )
		
		
		if st.button( '🚀 Train Classifier' ):
			try:
				X = df_processed[ features ].to_numpy( )
				y = df_processed[ targets ].to_numpy( dtype=float )
				X_train, X_test, y_train, y_test = model.split_data( X, y, size=test_sz,
					random=random_state )
				
				model.train( X_train, y_train )
				y_pred = model.project( X_test )
				target_count = len( np.unique( y_test ) )
				
				# ------------------------------------------------------------------
				# METRICS & ANALYSIS
				# ------------------------------------------------------------------
				st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
				st.markdown( '##### Model Performance' )
				df_classifier = model.analyze( X_test, y_test )
				st.data_editor( df_classifier, use_container_width=True )
				
				# ------------------------------------------------------------------
				# CONFUSION MATRIX
				# ------------------------------------------------------------------
				st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
				st.markdown( '##### Confusion Matrix' )
				plt.close( 'all' )
				model.confusion_matrix( X_test, y_test )
				st.pyplot( plt.gcf( ) )
				plt.close( 'all' )
				
				# ------------------------------------------------------------------
				# ACTUAL VS PREDICTED CLASS COUNTS
				# ------------------------------------------------------------------
				st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
				st.markdown( '##### Actual vs Predicted Counts' )
				actual_counts = pd.Series( y_test ).value_counts( ).sort_index( )
				pred_counts = pd.Series( y_pred ).value_counts( ).sort_index( )
				df_counts = pd.DataFrame(
					{ 'Actual': actual_counts, 'Predicted': pred_counts } ).fillna( 0 )
				
				fig_counts, ax_counts = plt.subplots( figsize=( 8, 5 ) )
				df_counts.plot( kind='bar', ax=ax_counts )
				ax_counts.set_xlabel( 'Class' )
				ax_counts.set_ylabel( 'Count' )
				ax_counts.set_title( 'Actual vs Predicted Class Counts' )
				ax_counts.grid( axis='y', alpha=0.3 )
				fig_counts.tight_layout( )
				st.pyplot( fig_counts )
				plt.close( fig_counts )
				
				# ------------------------------------------------------------------
				# PER-CLASS ACCURACY
				# ------------------------------------------------------------------
				st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
				st.markdown( '##### Per-Class Accuracy' )
				df_eval = pd.DataFrame( { 'Actual': y_test, 'Predicted': y_pred } )
				df_eval[ 'Correct' ] = (df_eval[ 'Actual' ] == df_eval[ 'Predicted' ]).astype( int )
				df_class_acc = df_eval.groupby(
					'Actual', dropna=False )[ 'Correct' ].mean( ).sort_index( )
				
				fig_acc, ax_acc = plt.subplots( figsize=(8, 5) )
				ax_acc.bar( df_class_acc.index.astype( str ), df_class_acc.values )
				ax_acc.set_xlabel( 'Class' )
				ax_acc.set_ylabel( 'Accuracy' )
				ax_acc.set_ylim( 0.0, 1.05 )
				ax_acc.set_title( 'Per-Class Accuracy' )
				ax_acc.grid( axis='y', alpha=0.3 )
				fig_acc.tight_layout( )
				st.pyplot( fig_acc )
				plt.close( fig_acc )
				
				# ------------------------------------------------------------------
				# PREDICTION CONFIDENCE
				# ------------------------------------------------------------------
				if hasattr( model, 'predict_probability' ):
					try:
						proba = model.predict_probability( X_test )
						if isinstance( proba, np.ndarray ) and proba.ndim == 2 and proba.shape[
							1 ] > 1:
							st.subheader( 'Prediction Confidence' )
							max_conf = np.max( proba, axis=1 )
							fig_conf, ax_conf = plt.subplots( figsize=(8, 5) )
							ax_conf.hist( max_conf, bins=20 )
							ax_conf.set_xlabel( 'Maximum Predicted Probability' )
							ax_conf.set_ylabel( 'Frequency' )
							ax_conf.set_title( 'Prediction Confidence Distribution' )
							ax_conf.grid( axis='y', alpha=0.3 )
							fig_conf.tight_layout( )
							st.pyplot( fig_conf )
							plt.close( fig_conf )
					except Exception:
						pass
				
				# ------------------------------------------------------------------
				# OBSERVED VS PREDICTED
				# ------------------------------------------------------------------
				st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
				st.markdown( '##### Observed vs Predicted' )
				if target_count <= 2 and hasattr( model, 'scatter_plot' ):
					try:
						plt.close( 'all' )
						model.scatter_plot( X_test, y_test )
						st.pyplot( plt.gcf( ) )
						plt.close( 'all' )
					except Exception as e:
						st.info( f'Observed vs Predicted plot skipped: {e}' )
						plt.close( 'all' )
				else:
					st.info( 'Observed vs Predicted is shown only when the target has < 2 classes.' )
				
				# ------------------------------------------------------------------
				# ROC CURVE
				# ------------------------------------------------------------------
				st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
				st.subheader( 'ROC Curve' )
				if target_count == 2 and hasattr( model, 'roc_curve' ):
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
			
			except Exception as e:
				st.error( f'Classification Failed: {e}' )

# ============================================
# REGRESSION MODE
# ============================================
elif mode == 'Regressions':
	df_original = st.session_state.get( 'df_dataset', None )
	df_dataset = st.session_state.get( 'df_dataset', None )
	df_working = st.session_state.get( 'df_working', None )
	df_processed = st.session_state.get( 'df_processed', None )
	df_classification = st.session_state.get( 'df_classification', None )
	numeric_columns = st.session_state.get( 'numeric_columns', [ ] )
	categorical_columns = st.session_state.get( 'categorical_columns', [ ] )
	features = st.session_state.get( 'features', [ ] )
	targets = st.session_state.get( 'targets', [ ] )
	
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Regressions' ] )
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
			st.warning( '⚠️ Classification requires numeric features and a categorical target.' )
			st.stop( )
		
		df_regression = st.session_state.get( 'df_regression', df_original.copy( ) ).copy( )
		
		# ======================================================================================
		# Data Selection
		# ======================================================================================
		st.markdown( '##### Data Selection' )
		st.caption( f'Records: {len( df_original ):,}  |  Features: {len( df_original.columns ):,}' )
		
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			features = st.multiselect( 'Select Features', options=categorical_columns,
				default=[ c for c in st.session_state.get( 'features', [ ] )
				          if c in numeric_columns ], key='regression_features' )
		
		with col_c2:
			target_options = [ c for c in numeric_columns if c not in features ]
			targets = st.multiselect( 'Select Targets', options=target_options,
				default=[ c for c in st.session_state.get( 'targets', [ ] )
				          if c in target_options ], key='regression_targets' )
		
		sel_b1, sel_b2, sel_b3 = st.columns( [ 0.34, 0.33, 0.33 ] )
		with sel_b1:
			if st.button( 'Create Working Dataset', key='regression_create_dataset',
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
				st.success( 'Working dataframe created.' )
		
		with sel_b2:
			if st.button( 'Reset Working Dataset', key='regression_reset_working_dataset',
					use_container_width=True ):
				st.session_state[ 'df_working' ] = df_working.copy( )
				commit_frame( df_working )
				st.success( 'Working dataframe reset.' )
		
		with sel_b3:
			if st.button( 'Reset To Original', key='regression_reset_to_original',
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
		st.caption( f'Rows: {len( df_working ):,}  |  Columns: {len( df_working.columns ):,}' )
		
		render_table( df_working )
		
		# ------------------------------------------------------------------
		# Training Target & Features
		# ------------------------------------------------------------------
		if df_working.empty:
			st.warning( '⚠️ No complete rows remain after preprocessing and target/feature selection.' )
			st.stop( )
		
		y = df_working[ targets ]
		
		if len( np.unique( y ) ) < 2:
			st.warning( '⚠️ The selected numeric target must contain at least two distinct values.' )
			st.stop( )
		
		# -----------------------------------------------------------------
		# Data Processing
		# -----------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Feature Engineering' )
		
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			with st.expander( label='Data Scaling', icon='⚖️', key='regression_scalers' ):
				
				with st.expander( 'Standard Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=targets,
						key='regression_standard_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_standard_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = StandardScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								st.success( 'Standard Scaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_standard_scaler_reset',
								use_container_width=True ):
							df_processed = pd.DataFrame( )
							st.success( 'Reset to Working Dataset.' )
				
				with st.expander( 'Min-Max Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_minmax_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_minmax_scaler_apply',
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
							st.success( 'Reset to Working' )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_robust_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_robust_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = RobustScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'RobustScaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_robust_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							st.success( 'Reset to Working' )
				
				with st.expander( 'Normal Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ],
						index=1, key='regression_normal_scaler_norm' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_normal_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = NormalScaler( norm=norm )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'NormalScaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_normal_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_maxabs_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_maxabs_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MaxAbsScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'MaxAbsScaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_maxabs_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
			
			with st.expander( label='Data Imputation', icon='➕', key='regression_imputers' ):
				
				with st.expander( 'Mean Imputer', expanded=False ):
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='regression_mean_imputer_indicator' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_mean_imputer_apply',
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
						if st.button( 'Reset', key='regression_mean_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1,
						value=5, step=1, key='regression_nearest_imputer_neighbors' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_nearest_imputer_apply',
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
						if st.button( 'Reset', key='regression_nearest_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset Data' )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
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
						if st.button( 'Reset', key='regression_iterative_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset Data' )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					impute_cols = st.multiselect( 'Columns',
						options=numeric_columns,
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
						if st.button( 'Reset', key='regression_simple_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
			
			with st.expander( label='Data Encoding', icon='🔢', key='regression_encoders' ):
				
				with st.expander( 'One-Hot Encoder', expanded=False ):
					encode_cols = st.multiselect( 'Columns', options=features,
						key='regression_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='regression_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0,
						key='regression_onehot_unknown' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_onehot_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OneHotEncoder( sparse=bool( sparse ), unknown=unknown )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, encode_cols,
									result, 'onehot' )
								commit_frame( df_processed )
								st.success( 'One-Hot Encoder Applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_onehot_reset',
								use_container_width=True ):
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					encode_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='regression_ordinal_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_ordinal_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OrdinalEncoder( )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								df_processed[ encode_cols ] = result
								commit_frame( df_processed )
								st.success( 'Ordinal Encoder Applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_ordinal_reset',
								use_container_width=True ):
							df_processed = pd.DataFrame( )
							st.success( 'Reset to Working' )
				
				with st.expander( 'Label Encoder', expanded=False ):
					target_col = st.selectbox( 'Column',
						options=categorical_columns,
						key='regression_label_encoder_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_label_encoder_apply',
								use_container_width=True ):
							if target_col:
								encoder = LabelEncoder( )
								result = encoder.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed[ target_col ] = result
								commit_frame( df_processed )
								st.success( 'Label Encoder Applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_label_encoder_reset',
								use_container_width=True ):
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Target Encoder', expanded=False ):
					encode_cols = st.multiselect( 'Categorical Feature Columns',
						options=categorical_columns, key='regression_target_encoder_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='regression_target_encoder_target_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_target_encoder_apply',
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
						if st.button( 'Reset', key='regression_target_encoder_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					poly_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='regression_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4,
						value=2, key='regression_polynomial_degree' )
					
					interaction = st.checkbox( 'Interaction Only', value=True,
						key='regression_polynomial_interaction' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_polynomial_apply',
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
						if st.button( 'Reset', key='regression_polynomial_reset',
								use_container_width=True ):
							working_to_original( )
							st.session_state[ 'df_working' ] = df_working.copy( )
							st.success( 'Reset to Original.' )
		
		with feature_c2:
			
			with st.expander( label='Data Transformation', icon='⚡', key='transformers' ):
				
				with st.expander( 'Binarizer', expanded=False ):
					transform_cols = st.multiselect( 'Columns',
						options=numeric_columns, key='regression_binarizer_cols' )
					
					threshold = st.number_input( 'Threshold', value=0.0, step=0.1,
						key='regression_binarizer_threshold' )
					
					copy = st.checkbox( 'Copy', value=True, key='regression_binarizer_copy' )
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Binarizer', key='regression_binarizer_apply',
								use_container_width=True ):
							if transform_cols:
								df_processed = get_working_frame( )
								transformer = Binarizer( threshold=float( threshold ),
									copy=bool( copy ) )
								result = transformer.train_transform(
									df_processed[ transform_cols ].to_numpy( ) )
								
								df_processed[ transform_cols ] = result
								commit_frame( df_processed )
								st.success( 'Binarizer applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_binarizer_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column', options=df_working.columns.tolist( ),
						key='regression_label_binarizer_col' )
					
					pos_label = st.number_input( 'Positive Label', value=1, step=1,
						key='regression_label_binarizer_pos' )
					
					neg_label = st.number_input( 'Negative Label', value=0, step=1,
						key='regression_label_binarizer_neg' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='regression_label_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply LabelBinarizer', key='regression_label_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = get_working_frame( )
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ), sparse_output=bool( sparse_output ) )
								result = transformer.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, [ target_col ], result,
									'label_binarizer' )
								commit_frame( df_processed )
								st.success( 'Label Binarizer Applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_label_binarizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column', options=df_working.columns.tolist( ),
						key='regression_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='regression_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='regression_multilabel_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_multilabel_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = get_working_frame( )
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
						if st.button( 'Reset', key='regression_multilabel_binarizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'TF-IDF Transformer', expanded=False ):
					text_count_cols = st.multiselect( 'Count Matrix Columns',
						options=df_working.columns.to_list( ), key='regression_tfidf_transformer_cols' )
					
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
						if st.button( 'Apply', key='regression_tfidf_transformer_apply',
								use_container_width=True ):
							if text_count_cols:
								df_processed = get_working_frame( )
								transformer = TfidfTransformer( norm=norm, use_idf=bool( use_idf ),
									smooth_idf=bool( smooth_idf ), sublinear_tf=bool( sublinear_tf ) )
								
								result = transformer.train_transform(
									df_processed[ text_count_cols ].apply(
										pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, text_count_cols, result,
									'tfidf_transformer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Transformer Applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_tfidf_transformer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Column Transformer', expanded=False ):
					numeric_columns = st.multiselect( 'Numeric Columns', options=df_working.columns.to_list(),
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
					
					sparse_threshold = st.slider( 'Sparse Threshold', min_value=0.0, max_value=1.0,
						value=0.3, key='regression_column_transformer_sparse_threshold' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Column Transformer', key='regression_column_transformer_apply',
								use_container_width=True ):
							df_processed = get_working_frame( )
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
									categorical_model = OneHotEncoder(
										sparse=False, unknown='ignore' ).model
								else:
									categorical_model = OrdinalEncoder( ).model
								
								transformers.append( 'categorical', categorical_model,
									categorical_columns )
							
							if transformers:
								transformer = ColumnTransformer( transformers=transformers,
									remainder=remainder, sparse_threshold=float( sparse_threshold ),
									n_jobs=None, transformer_weights=None, verbose=False )
								
								result = transformer.train_transform( df_processed )
								df_processed = normalize_result_frame( result=result, index=df_processed.index,
									prefix='column_transformer', columns=None )
								
								commit_frame( df_processed )
								st.success( 'ColumnTransformer applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_column_transformer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
			
			with st.expander( label='Feature Extration', icon='⛏️', key='regression_extractors' ):
				
				with st.expander( 'TF-IDF Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns.tolist( ),
						key='regression_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='regression_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='regression_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='regression_tfidf_vectorizer_use_idf' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_tfidf_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = get_working_frame( )
								transformer = TfidfVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int( max_features ),
									use_idf=bool( use_idf ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols, transformer,
									'tfidf_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Vectorizer Applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_tfidf_vectorizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns.tolist( ),
						key='regression_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='regression_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0,
						step=1, key='regression_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='regression_count_vectorizer_binary' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_count_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = get_working_frame( )
								transformer = CountVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int( max_features ),
									binary=bool( binary ) )
								df_processed = apply_text_vectorizer( df_processed, text_cols, transformer,
									'count_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'Count Vectorizer Applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_count_vectorizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns.tolist( ),
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
						if st.button( 'Apply', key='regression_hash_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = get_working_frame( )
								transformer = HashVectorizer( num=int( n_features ),
									ngram_range=(1, int( ngram_max )), binary=bool( binary ),
									alternate_sign=bool( alternate_sign ) )
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'hash_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'HashVectorizer Applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_hash_vectorizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					dict_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='regression_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='regression_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='regression_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='regression_dict_vectorizer_sort' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_dict_vectorizer_apply',
								use_container_width=True ):
							if dict_cols:
								df_processed = get_working_frame( )
								transformer = DictVectorizer( dtype=np.float64, separator=separator,
									sparse=bool( sparse ), sort=bool( sort ) )
								
								df_processed = apply_dict_transform( df_processed, dict_cols,
									transformer, 'dict_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'DictVectorizer applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_dict_vectorizer_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					hash_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='regression_feature_hasher_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='regression_feature_hasher_n_features' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='regression_feature_hasher_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply', key='regression_feature_hasher_apply',
								use_container_width=True ):
							if hash_cols:
								df_processed = get_working_frame( )
								transformer = FeatureHasher( n_features=int( n_features ),
									input_type='dict', dtype=np.float64,
									alternate_sign=bool( alternate_sign ) )
								
								df_processed = apply_dict_transform( df_processed, hash_cols,
									transformer, 'feature_hasher' )
								commit_frame( df_processed )
								st.success( 'FeatureHasher applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_feature_hasher_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
			
			with st.expander( label='Dimensionality Reduction', icon='🎚️', key='selectors' ):
				
				with st.expander( 'Variance Threshold', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=df_working.columns.to_list( ),
						key='regression_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0,
						step=0.01, key='regression_variance_threshold_value' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_variance_threshold_apply',
								use_container_width=True ):
							if select_cols:
								df_processed = get_working_frame( )
								selector = VarianceThreshold( thresh=float( threshold ) )
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result,
									'variance_threshold' )
								
								commit_frame( df_processed )
								st.success( 'VarianceThreshold applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_variance_threshold_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Canonical Correlation Analysis', expanded=False ):
					X_cols = st.multiselect( 'Predictor Columns', options=numeric_columns,
						key='regression_cca_x_cols' )
					
					y_cols = st.multiselect( 'Target Columns', options=numeric_columns,
						key='regression_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2,
						step=1, key='regression_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True, key='regression_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500,
						step=1, key='regression_cca_max_iter' )
					
					a1, a2 = st.columns( 2 )
					
					with a1:
						if st.button( 'Apply', key='regression_cca_apply', use_container_width=True ):
							if X_cols and y_cols:
								df_processed = get_working_frame( )
								selector = CCA( num=int( n_components ), scale=bool( scale ),
									size=int( max_iter ) )
								
								result = selector.train_transform( df_processed[ X_cols ].to_numpy( ),
									df_processed[ y_cols ].to_numpy( ) )
								
								df_result = normalize_result_frame( result=result,
									index=df_processed.index, prefix='cca', columns=None )
								
								df_processed = pd.concat(
									[ df_processed.drop( columns=X_cols + y_cols, errors='ignore' ),
									  df_result ], axis=1 )
								commit_frame( df_processed )
								st.success( 'CCA Applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_cca_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
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
						if st.button( 'Apply', key='regression_pca_apply', use_container_width=True ):
							if select_cols:
								df_processed = get_working_frame( )
								selector = PCA( num=int( n_components ),
									solver=solver )
								
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols,
									result, 'pca' )
								
								commit_frame( df_processed )
								st.success( 'PCA applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_pca_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Select-Best', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns',
						options=numeric_columns, key='regression_selectbest_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=numeric_columns,
						key='regression_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
						          'mutual_info_regression' ], key='regression_selectbest_score_name' )
					
					k_best = st.number_input( 'K', min_value=1, value=5, step=1,
						key='regression_selectbest_k' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_selectbest_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = get_working_frame( )
								selector = SelectBest(
									score_func=score_function_from_name( score_name ),
									num=int( k_best ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'select_best' )
								
								commit_frame( df_processed )
								st.success( 'Select Best Applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_selectbest_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Select-Percent', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns',
						options=numeric_columns, key='regression_selectpercent_x_cols' )
					
					target_col = st.selectbox( 'Target Column',
						options=df_working.columns.tolist( ),
						key='regression_selectpercent_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression',
						          'mutual_info_classif', 'mutual_info_regression' ],
						key='regression_selectpercent_score_name' )
					
					percentile = st.slider( 'Percentile', min_value=1, max_value=100,
						value=10, key='regression_selectpercent_percentile' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SelectPercent', key='regression_selectpercent_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = get_working_frame( )
								selector = SelectPercent(
									score_func=score_function_from_name( score_name ),
									pct=int( percentile ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'select_percent' )
								
								commit_frame( df_processed )
								st.success( 'SelectPercent applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_selectpercent_reset',
								use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Sequential Back Selection', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='regression_sbs_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns.tolist( ),
						key='regression_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='regression_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='regression_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1,
						step=1, key='regression_sbs_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply', key='regression_sbs_apply', use_container_width=True ):
							if X_cols and target_col:
								df_processed = get_working_frame( )
								selector = SBS( classifier=None, k_features=int( k_features ),
									test_size=float( test_size ), random_state=int( random_state ) )
								
								X_input = df_processed[ X_cols ].apply(
									pd.to_numeric, errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result, 'sbs' )
								commit_frame( df_processed )
								st.success( 'SBS Applied.' )
					
					with a2:
						if st.button( 'Reset', key='regression_sbs_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
				
				with st.expander( 'Recursive Feature Elimination', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='regression_rfe_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns.tolist( ),
						key='regression_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain',
						min_value=1, value=1, step=1, key='regression_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0,
						step=1, key='regression_rfe_verbose' )
					
					a1, a2 = st.columns( 2 )					
					with a1:
						if st.button( 'Apply', key='regression_rfe_apply', use_container_width=True ):
							if X_cols and target_col:
								df_processed = get_working_frame( )
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
						if st.button( 'Reset', key='regression_rfe_reset', use_container_width=True ):
							working_to_original( )
							st.success( 'Reset to Original.' )
		
		# ------------------------------------------------------------------
		# MODEL SELECTION
		# ------------------------------------------------------------------
		model_map = \
		{
			'Ordinary Least Squares': LeastSquares,
			'Ridge Regression': Ridge,
			'Lasso Regression': Lasso,
			'Elastic Net': ElasticNet,
			'Bayesian Ridge': BayesianRidge,
			'Support Vector': SupportVector,
			'Stochastic Gradient Descent': GradientDescent,
			'k-Nearest Neighbors': NearestNeighbor,
			'Bagging Regressor': BaggingModel,
			'Extra Trees Regressor': ExtraTreesModel,
			'AdaBoost Regressor': AdaptiveBoost,
			'Gradient Boosting': GradientBoost,
			'Random Forest': RandomForest,
			'Voting Regressor': VotingModel,
			'Stacking Regressor': StackingModel
		}
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		
		sel_c4, sel_c5, sel_c6 = st.columns( [ 0.33, 0.33, 0.33 ], border=True )
		with sel_c4:
			st.markdown( '##### Model Selection' )
			model_name = st.selectbox( 'Select Regression Model', list( model_map.keys( ) ) )
			model = model_map[ model_name ]( )
		
		# ------------------------------------------------------------------
		# TRAIN / TEST SPLIT
		# ------------------------------------------------------------------
		with sel_c5:
			st.markdown( '##### Training Configuration' )
			test_size = st.slider( 'Test Set Size (%)', 10, 40, 20, key='regressions-1' ) / 100.0
		
		with sel_c6:
			st.markdown( '##### Random State' )
			random_state = int( st.number_input( 'Seed', value=42, step=1, key='regressions-2' ) )
			
			min_test_rows = max( 2, int( np.ceil( len( df_regression ) * test_size ) ) )
			min_train_rows = len( df_regression ) - min_test_rows
			
			if min_train_rows < 2:
				st.warning( '⚠️ The selected test size leaves too few training rows. Reduce the test size or load more data.' )
				st.stop( )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		X = df_processed[ features ].to_numpy( )
		y = df_processed[ targets ].to_numpy( dtype=float )
		if st.button( '🚀 Train Model' ):
			try:
				X_train, X_test, y_train, y_test = model.split_data( X, y, size=test_size,
					random=random_state )
				
				model.train( X_train, y_train )
				
				# ------------------------------------------------------------------
				# METRICS
				# ------------------------------------------------------------------
				st.markdown( '##### Model Performance' )
				df_regressor = model.analyze( X_test, y_test )
				st.data_editor( df_regressor, use_container_width=True )
				
				# ------------------------------------------------------------------
				# PREDICTIONS
				# ------------------------------------------------------------------
				st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
				st.markdown( '##### Predictions' )
				y_pred = model.project( X_test )
				df_predictions = pd.DataFrame(
					{
							'Observed': y_test,
							'Predicted': y_pred,
							'Residual': y_test - y_pred
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
			
			except Exception as e:
				st.error( f'Regression failed: {e}' )

# ============================================
# CLUSTERING MODELS MODE
# ============================================
elif mode == 'Clustering':
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Clustering' ] )
		st.divider( )
		st.caption( 'Unsupervised Learning Models' )
		
		# ------------------------------------------------------------------
		# DATA SOURCE RESOLUTION
		# ------------------------------------------------------------------
		if 'df_features' in st.session_state and \
				st.session_state[ 'df_features' ] is not None and \
				not st.session_state[ 'df_features' ].empty:
			df_cluster = st.session_state[ 'df_features' ].copy( )
			data_source = 'features'
			st.info( 'Using Feature-Engineered Dataset.' )
		else:
			df_cluster = df_dataset.copy( )
			data_source = 'dataset'
			st.info( 'Using Original Dataset.' )
		
		if df_cluster is None or df_cluster.empty:
			st.warning( 'No Dataset Available for Clustering.' )
			st.stop( )
		
		# ------------------------------------------------------------------
		# COLUMN CLASSIFICATION (NUMERIC ONLY)
		# ------------------------------------------------------------------
		numeric_columns = [ c for c in df_cluster.columns
				if df_cluster[ c ].dtype.kind in { 'i', 'f' } ]
		
		if len( numeric_columns ) < 2:
			st.warning( 'At least two numeric columns are required for clustering.' )
			st.stop( )
		
		# ------------------------------------------------------------------
		# FEATURE SELECTION
		# ------------------------------------------------------------------
		csr_c1, csr_c2 = st.columns( [ 0.5, 0.5 ],  border=True )
		with csr_c1:
			st.markdown( '##### Feature Selection' )
			feature_columns = st.multiselect( 'Select Features for Clustering',
				options=numeric_columns )
			
			if len( feature_columns ) < 2:
				st.info( 'Select at least two features to continue.' )
				st.stop( )
			
			df_cluster_input = df_cluster[ feature_columns ].copy( )
			df_cluster_input = df_cluster_input.replace( [ np.inf, -np.inf ], np.nan )
			rows_before = len( df_cluster_input )
			df_cluster_input = df_cluster_input.dropna( axis=0, how='any' )
			rows_after = len( df_cluster_input )
			
			if rows_after == 0:
				st.warning( 'No complete numeric rows remain after removing missing or invalid values.' )
				st.stop( )
			
			if rows_after != rows_before:
				st.info(
					f'Using {rows_after:,} complete rows after removing '
					f'{rows_before - rows_after:,} row(s) with missing or invalid values.'
				)
			
			X = df_cluster_input.to_numpy( )
		
		# ------------------------------------------------------------------
		# MODEL SELECTION
		# ------------------------------------------------------------------
		with csr_c2:
			st.markdown( '##### Clustering Model' )
			model_name = st.selectbox( 'Clustering Algorithm',
				[ 'K-Means', 'DBSCAN', 'Agglomerative', 'Spectral', 'OPTICS', 'MeanShift',
				  'AffinityPropagation', 'Birch' ] )
		
		# ------------------------------------------------------------------
		# MODEL PARAMETERS
		# ------------------------------------------------------------------
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Model Parameters' )
		
		model = None
		model_parameters = { }
			
		if model_name == 'K-Means':
			prm_c1, prm_c2, prm_c3 = st.columns( [ 0.33, 0.33, 0.33 ], border=True )
			with prm_c1:
				n_clusters = st.number_input( 'Number of Clusters (K)',
					min_value=2, value=3 )
			with prm_c2:
				n_init = st.number_input( 'Number of Initializations', min_value=1, value=10 )
			
			with prm_c3:
				max_iter = st.number_input( 'Maximum Iterations', min_value=1,
					value=300 )
	
			model = KMeans( clusters=int( n_clusters ), n_init=int( n_init ),
				max_iter=int( max_iter ) )
			model_parameters = {
					'Model': model_name,
					'Clusters': int( n_clusters ),
					'N-Init': int( n_init ),
					'Max-Iter': int( max_iter )
			}
		
		elif model_name == 'DBSCAN':
			prm_c1, prm_c2, prm_c3 = st.columns( [ 0.33, 0.33, 0.33 ], border=True )
			with prm_c1:
				eps = st.number_input( 'Epsilon (eps)', min_value=0.01, value=0.5 )
			
			with prm_c2:
				min_samples = st.number_input( 'Min Samples', min_value=1, value=5 )
			
			with prm_c3:
				metric = st.selectbox( 'Metric',
					[ 'euclidean', 'manhattan', 'minkowski', 'cosine' ] )
				
			model = DBSCAN( eps=float( eps ), samples=int( min_samples ), metric=metric )
			model_parameters = {
					'Model': model_name,
					'Eps': float( eps ),
					'Min-Samples': int( min_samples ),
					'Metric': metric
			}
		
		elif model_name == 'Agglomerative':
			prm_c1, prm_c2, prm_c3 = st.columns( [ 0.33, 0.33, 0.33 ], border=True )
			with prm_c1:
				n_clusters = st.number_input( 'Number of Clusters', min_value=2, value=3 )
			
			with prm_c2:
				linkage = st.selectbox( 'Linkage',
					[ 'ward', 'complete', 'average', 'single' ] )
				
			with prm_c3:
				if linkage == 'ward':
					metric = 'euclidean'
					st.caption( 'Ward linkage requires euclidean metric.' )
				else:
					metric = st.selectbox( 'Metric',
						[ 'euclidean', 'manhattan', 'cosine', 'l1', 'l2' ] )
				
			model = Agglomerative( n_clusters=int( n_clusters ), linkage=linkage, metric=metric )
			model_parameters = {
					'Model': model_name,
					'Clusters': int( n_clusters ),
					'Linkage': linkage,
					'Metric': metric
			}
		
		elif model_name == 'Spectral':
			prm_c1, prm_c2, prm_c3, prm_c4, prm_c5 = st.columns( [ 0.2, 0.2, 0.2, 0.2, 0.2 ],
				border=True )
			
			with prm_c1:
				n_clusters = st.number_input( 'Number of Clusters', min_value=2, value=3 )
			
			with prm_c2:
				affinity = st.selectbox( 'Affinity', [ 'rbf', 'nearest_neighbors' ] )
				
			with prm_c3:
				n_neighbors = st.number_input( 'Neighbors', min_value=1, value=10 )
			
			with prm_c4:
				gamma = st.number_input( 'Gamma', min_value=0.0001, value=1.0 )
			
			with prm_c5:
				assign_labels = st.selectbox( 'Assign Labels', [ 'kmeans', 'discretize', 'cluster_qr' ])
			
			model = Spectral( n_clusters=int( n_clusters ), affinity=affinity,
				n_neighbors=int( n_neighbors ), gamma=float( gamma ),
				assign_labels=assign_labels )
			
			model_parameters = {
					'Model': model_name,
					'Clusters': int( n_clusters ),
					'Affinity': affinity,
					'N-Neighbors': int( n_neighbors ),
					'Gamma': float( gamma ),
					'Assign-Labels': assign_labels
			}
		
		elif model_name == 'OPTICS':
			prm_c1, prm_c2, prm_c3, prm_c4, prm_c5 = st.columns( [ 0.2, 0.2, 0.2, 0.2, 0.2 ],
				border=True )
			
			with prm_c1:
				min_samples = st.number_input( 'Min Samples', min_value=2, value=5 )
			
			with prm_c2:
				max_eps = st.number_input( 'Max Epsilon', min_value=0.01, value=10.0 )
			
			with prm_c3:
				cluster_method = st.selectbox( 'Cluster Method', [ 'xi', 'dbscan' ] )
			
			with prm_c4:
				xi = st.number_input( 'Xi', min_value=0.0001, max_value=0.9999,
					value=0.05 )
			
			with prm_c5:
				eps_value = None
				if cluster_method == 'dbscan':
					eps_value = st.number_input( 'Extraction Epsilon', min_value=0.01,
						value=0.5 )
				
			model = OPTICS( min_samples=int( min_samples ), max_eps=float( max_eps ),
				cluster_method=cluster_method, xi=float( xi ),
				eps=float( eps_value ) if eps_value is not None else None )
			
			model_parameters = {
					'Model': model_name,
					'Min-Samples': int( min_samples ),
					'Max-Eps': float( max_eps ),
					'Cluster-Method': cluster_method,
					'Xi': float( xi ),
					'Eps': float( eps_value ) if eps_value is not None else None
			}
		
		elif model_name == 'MeanShift':
			prm_c1, prm_c2, prm_c3, prm_c4, prm_c5, prm_c6 = st.columns(
				[ 0.16, 0.16, 0.16, 0.16, 0.16, 0.16 ], border=True )
			
			with prm_c1:
				use_bandwidth = st.checkbox( 'Specify Bandwidth', value=False )
			
			with prm_c2:
				bandwidth = None
				if use_bandwidth:
					bandwidth = st.number_input( 'Bandwidth', min_value=0.0001, value=1.0 )
			with prm_c3:
				bin_seeding = st.checkbox( 'Use Bin Seeding', value=False )
			
			with prm_c4:
				min_bin_freq = st.number_input( 'Min Bin Frequency', min_value=1, value=1 )
			
			with prm_c5:
				cluster_all = st.checkbox( 'Cluster All Samples', value=True )
			
			with prm_c6:
				max_iter = st.number_input( 'Maximum Iterations', min_value=1, value=300 )
			
			model = MeanShift( bandwidth=float( bandwidth ) if bandwidth is not None else None,
				bin_seeding=bin_seeding, min_bin_freq=int( min_bin_freq ),
				cluster_all=cluster_all, max_iter=int( max_iter ) )
			model_parameters = {
					'Model': model_name,
					'Bandwidth': float( bandwidth ) if bandwidth is not None else None,
					'Bin-Seeding': bin_seeding,
					'Min-Bin-Freq': int( min_bin_freq ),
					'Cluster-All': cluster_all,
					'Max-Iter': int( max_iter )
			}
		
		elif model_name == 'AffinityPropagation':
			prm_c1, prm_c2, prm_c3, prm_c4, prm_c5, prm_c6 = st.columns(
				[ 0.16, 0.16, 0.16, 0.16, 0.16, 0.16 ], border=True )
			
			with prm_c1:
				damping = st.number_input( 'Damping', min_value=0.5, max_value=0.9999, value=0.5 )
			
			with prm_c2:
				max_iter = st.number_input( 'Maximum Iterations', min_value=1, value=200 )
			
			with prm_c3:
				convergence_iter = st.number_input( 'Convergence Iterations', min_value=1, value=15 )
			
			with prm_c4:
				use_preference = st.checkbox( 'Specify Preference', value=False )
				
			with prm_c5:
				preference = None
				if use_preference:
					preference = st.number_input( 'Preference', value=0.0 )
			
			with prm_c6:
				affinity = st.selectbox( 'Affinity', [ 'euclidean', 'precomputed' ] )
			
			model = AffinityPropagation( damping=float( damping ),
				max_iter=int( max_iter ), convergence_iter=int( convergence_iter ),
				preference=float( preference ) if preference is not None else None,
				affinity=affinity )
			model_parameters = {
					'Model': model_name,
					'Damping': float( damping ),
					'Max-Iter': int( max_iter ),
					'Convergence-Iter': int( convergence_iter ),
					'Preference': float( preference ) if preference is not None else None,
					'Affinity': affinity
			}
		
		elif model_name == 'Birch':
			prm_c1, prm_c2, prm_c3, prm_c4, prm_c5, prm_c6 = st.columns(
				[ 0.2, 0.2, 0.2, 0.2, 0.2 ], border=True )
			
			with prm_c1:
				threshold = st.number_input( 'Threshold', min_value=0.0001, value=0.5 )
			
			with prm_c2:
				branching_factor = st.number_input( 'Branching Factor', min_value=2, value=50 )
			
			with prm_c3:
				use_global_clusters = st.checkbox( 'Use Global Clusters', value=True )
			
			with prm_c4:
				if use_global_clusters:
					n_clusters = st.number_input( 'Number of Global Clusters', min_value=2, value=3 )
					n_cluster_value = int( n_clusters )
				else:
					n_cluster_value = None
			
			with prm_c5:
				compute_labels = st.checkbox( 'Compute Labels', value=True )
			
			model = Birch( threshold=float( threshold ), branching_factor=int( branching_factor ),
				n_clusters=n_cluster_value, compute_labels=compute_labels )
			model_parameters = {
					'Model': model_name,
					'Threshold': float( threshold ),
					'Branching-Factor': int( branching_factor ),
					'N-Clusters': n_cluster_value,
					'Compute-Labels': compute_labels }
		
		cluster_signature = ( data_source, tuple( feature_columns ),
				model_name, tuple( (k, str( v )) for k, v in model_parameters.items( ) ) )
		
		# ------------------------------------------------------------------
		# FIT CLUSTERING MODEL
		# ------------------------------------------------------------------
		st.markdown( '##### Run Clustering' )
		
		if st.button( 'Run Clustering' ):
			try:
				labels = model.project( X )
				
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
				
				detail_rows = [ ]
				for prop in [ 'features', 'inertia', 'iterations', 'epsilon', 'eps',
						'min_samples', 'metric', 'linkage', 'cluster_method', 'bandwidth',
						'threshold', 'branching_factor', 'damping', 'convergence_iter', 'affinity' ]:
					if hasattr( model, prop ):
						try:
							value = getattr( model, prop )
							if value is not None and not isinstance( value, (np.ndarray,
							                                                 pd.DataFrame) ):
								detail_rows.append( { 'Property': prop, 'Value': value } )
						except Exception:
							pass
				
				df_details = pd.DataFrame( detail_rows ) if detail_rows else pd.DataFrame( )
				
				df_centroids = pd.DataFrame( )
				if hasattr( model, 'centroids_' ):
					try:
						centroids = model.centroids_
						if centroids is not None:
							df_centroids = pd.DataFrame( centroids, columns=feature_columns )
							df_centroids.insert( 0, 'Cluster', range( len( df_centroids ) ) )
					except Exception:
						df_centroids = pd.DataFrame( )
				
				st.session_state[ 'df_cluster_results' ] = df_results
				st.session_state[ 'df_cluster_counts' ] = df_counts
				st.session_state[ 'df_cluster_metrics' ] = df_metrics
				st.session_state[ 'df_cluster_centroids' ] = df_centroids
				st.session_state[ 'df_cluster_details' ] = df_details
				st.session_state[ 'cluster_plot_features' ] = feature_columns.copy( )
				st.session_state[ 'cluster_signature' ] = cluster_signature
				
				st.success( 'Clustering complete.' )
			
			except Exception as e:
				st.session_state[ 'df_cluster_results' ] = pd.DataFrame( )
				st.session_state[ 'df_cluster_counts' ] = pd.DataFrame( )
				st.session_state[ 'df_cluster_metrics' ] = pd.DataFrame( )
				st.session_state[ 'df_cluster_centroids' ] = pd.DataFrame( )
				st.session_state[ 'df_cluster_details' ] = pd.DataFrame( )
				st.session_state[ 'cluster_plot_features' ] = [ ]
				st.session_state[ 'cluster_signature' ] = None
				st.error( f'Clustering failed: {e}' )
		
		df_results = pd.DataFrame( )
		df_counts = pd.DataFrame( )
		df_metrics = pd.DataFrame( )
		df_centroids = pd.DataFrame( )
		df_details = pd.DataFrame( )
		
		if st.session_state.get( 'cluster_signature', None ) == cluster_signature:
			df_results = st.session_state.get( 'df_cluster_results', pd.DataFrame( ) )
			df_counts = st.session_state.get( 'df_cluster_counts', pd.DataFrame( ) )
			df_metrics = st.session_state.get( 'df_cluster_metrics', pd.DataFrame( ) )
			df_centroids = st.session_state.get( 'df_cluster_centroids', pd.DataFrame( ) )
			df_details = st.session_state.get( 'df_cluster_details', pd.DataFrame( ) )
		
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
elif mode == 'Time-Series':
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Time-Series' ] )
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