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
from boogr import Error, Logger
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
from sklearn.metrics import (confusion_matrix, roc_curve, auc, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score)

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
from scalers import (StandardScaler, MinMaxScaler, RobustScaler, NormalScaler, MaxAbsScaler)
from imputers import (MeanImputer, NearestImputer, IterativeImputer, SimpleImputer)
from encoders import (OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder,
                      PolynomialFeatures)

from transformers import (Binarizer, LabelBinarizer, MultiLabelBinarizer, TfidfTransformer,
                          ColumnTransformer, TfidfVectorizer, CountVectorizer, HashVectorizer,
                          DictVectorizer, FeatureHasher)

from clusters import (KMeans, DBSCAN, Agglomerative, Spectral, OPTICS, MeanShift,
                      AffinityPropagation, Birch)

from features import (VarianceThreshold, CCA, PCA, SelectBest, SelectPercent, SBS, RFE)
import classifications as classification_model
import regressions as regression_model

from classifications import (Perceptron, LogisticRegression, DecisionTree, SupportVector,
                             RandomForest, NearestNeighbor, BaggingModel, AdaptiveBoost,
                             GradientBoost)

from encoders import (OneHotEncoder, OrdinalEncoder, TargetEncoder)
from imputers import (MeanImputer, SimpleImputer, NearestImputer, IterativeImputer)
from forecasting import (LaggingSeries, LagQuantileSeries, LagBoostingSeries, ARIMA, SARIMA,
                         TimeSeriesSpliter)

# ============================================
# Session State
# ============================================

if 'df_original' not in st.session_state or st.session_state[ 'df_original' ] is None:
	st.session_state[ 'df_original' ] = pd.DataFrame( )

if 'df_profile' not in st.session_state or st.session_state[ 'df_profile' ] is None:
	st.session_state[ 'df_profile' ] = pd.DataFrame( )

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

if 'df_regression_scores' not in st.session_state or st.session_state[
	'df_regression_scores' ] is None:
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

def throw_if( name: str, value: object ) -> None:
	"""Input guard.
	
	Purpose:
	    Validates that a required argument contains a usable value before the surrounding workflow
	    continues. This guard centralizes early validation so provider wrappers and UI routines fail
	    with consistent, readable error messages.
	
	Args:
	    name (str): Name value used by the operation.
	    value (object): Value value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )
	
def init_state( ) -> None:
	defaults = { 'df_dataset': None, 'df_original': None, 'df_processed': None,
		'numeric_columns': [ ], 'categorical_columns': [ ], 'features': [ ], 'targets': [ ],
		'pipeline_log': [ ] }
	for k, v in defaults.items( ):
		if k not in st.session_state:
			st.session_state[ k ] = v

init_state( )

def has_loaded_dataset( df_frame: object ) -> bool:
	"""
		Purpose:
		--------
		Determine whether an object is a valid loaded dataframe.

		Parameters:
		-----------
		df_frame ( object ): Candidate dataframe object.

		Returns:
		--------
		bool:
			True when the object is a non-empty dataframe with at least one column.
	"""
	return (isinstance( df_frame, pd.DataFrame ) and not df_frame.empty and len(
		df_frame.columns ) > 0)

def get_loaded_dataset( ) -> pd.DataFrame | None:
	"""
		Purpose:
		--------
		Return the currently loaded dataset from session state when valid.

		Parameters:
		-----------
		None

		Returns:
		--------
		pd.DataFrame | None:
			Copy of the loaded dataset, or None when no valid dataset exists.
	"""
	df_frame = st.session_state.get( 'df_dataset', None )
	if not has_loaded_dataset( df_frame ):
		return None
	
	return df_frame.copy( )

def store_loaded_dataset( df_dataset: pd.DataFrame, df_original: pd.DataFrame = None ) -> None:
	"""
		Purpose:
		--------
		Persist a successfully loaded dataset to session state.

		Parameters:
		-----------
		df_dataset ( pd.DataFrame ): Loaded dataset.
		df_original ( pd.DataFrame | None ): Optional original copy.

		Returns:
		--------
		None
	"""
	if not has_loaded_dataset( df_dataset ):
		return
	
	df_source = df_dataset.copy( )
	df_base = df_original.copy( ) if isinstance( df_original, pd.DataFrame ) else df_source.copy( )
	
	st.session_state[ 'raw_df' ] = df_source.copy( )
	st.session_state[ 'df_original' ] = df_base.copy( )
	st.session_state[ 'df_dataset' ] = df_source.copy( )

def clear_keys( keys: List[ str ] ) -> None:
	"""
	
		Purpose:
		--------
		Remove a list of session-state keys if they exist.
	
		Parameters:
		-----------
		keys: List[str]
			The session-state keys to remove.
	
		Returns:
		--------
		None
	
	"""
	for key in keys:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_classification_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Clear session-state values owned by Classification mode when the user
		activates that mode.
	
		Parameters:
		-----------
		None
	
		Returns:
		--------
		None
	
	"""
	classification_keys = [ 'df_working', 'df_processed', 'df_model', 'df_scores',
		'df_predictions',
		'df_classification', 'df_classification_scores', 'df_features', 'df_targets', 'features',
		'targets', 'selected_all', 'active_features', 'active_targets', 'X_data', 'X_train',
		'X_test', 'y_train', 'y_test', 'y_series', 'y_prediction', 'model', 'elapsed_seconds',
		'target_count' ]
	
	clear_keys( classification_keys )
	st.session_state[ 'df_working' ] = pd.DataFrame( )
	st.session_state[ 'df_processed' ] = pd.DataFrame( )
	st.session_state[ 'df_model' ] = pd.DataFrame( )
	st.session_state[ 'df_scores' ] = pd.DataFrame( )
	st.session_state[ 'df_predictions' ] = pd.DataFrame( )
	st.session_state[ 'df_classification' ] = pd.DataFrame( )
	st.session_state[ 'df_classification_scores' ] = pd.DataFrame( )
	st.session_state[ 'df_features' ] = pd.DataFrame( )
	st.session_state[ 'df_targets' ] = pd.DataFrame( )
	st.session_state[ 'features' ] = [ ]
	st.session_state[ 'targets' ] = [ ]
	st.session_state[ 'selected_all' ] = [ ]
	st.session_state[ 'active_features' ] = [ ]
	st.session_state[ 'active_targets' ] = [ ]
	st.session_state[ 'X_data' ] = None
	st.session_state[ 'X_train' ] = None
	st.session_state[ 'X_test' ] = None
	st.session_state[ 'y_train' ] = None
	st.session_state[ 'y_test' ] = None
	st.session_state[ 'y_series' ] = None
	st.session_state[ 'y_prediction' ] = None
	st.session_state[ 'model' ] = None
	st.session_state[ 'elapsed_seconds' ] = 0.0
	st.session_state[ 'target_count' ] = 0.0

def reset_regression_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Clear session-state values owned by Regression mode when the user
		activates that mode.
	
		Parameters:
		-----------
		None
	
		Returns:
		--------
		None
	
	"""
	regression_keys = [ 'df_working', 'df_processed', 'df_model', 'df_scores', 'df_predictions',
		'df_regression', 'df_regression_scores', 'df_features', 'df_targets', 'features',
		'targets',
		'selected_all', 'active_features', 'active_targets', 'X_data', 'X_train', 'X_test',
		'y_train', 'y_test', 'y_series', 'y_prediction', 'model', 'elapsed_seconds',
		'target_count' ]
	
	clear_keys( regression_keys )
	st.session_state[ 'df_working' ] = pd.DataFrame( )
	st.session_state[ 'df_processed' ] = pd.DataFrame( )
	st.session_state[ 'df_model' ] = pd.DataFrame( )
	st.session_state[ 'df_scores' ] = pd.DataFrame( )
	st.session_state[ 'df_predictions' ] = pd.DataFrame( )
	st.session_state[ 'df_regression' ] = pd.DataFrame( )
	st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
	st.session_state[ 'df_features' ] = pd.DataFrame( )
	st.session_state[ 'df_targets' ] = pd.DataFrame( )
	st.session_state[ 'features' ] = [ ]
	st.session_state[ 'targets' ] = [ ]
	st.session_state[ 'selected_all' ] = [ ]
	st.session_state[ 'active_features' ] = [ ]
	st.session_state[ 'active_targets' ] = [ ]
	st.session_state[ 'X_data' ] = None
	st.session_state[ 'X_train' ] = None
	st.session_state[ 'X_test' ] = None
	st.session_state[ 'y_train' ] = None
	st.session_state[ 'y_test' ] = None
	st.session_state[ 'y_series' ] = None
	st.session_state[ 'y_prediction' ] = None
	st.session_state[ 'model' ] = None
	st.session_state[ 'elapsed_seconds' ] = 0.0
	st.session_state[ 'target_count' ] = 0.0

# ============================================
# Utilities
# ============================================
def inferential_plot( title: str, subtitle: str | None = None, figsize: tuple[ int, int ] = (6, 4),
	grid: bool=True, ref_line: float | None=None, legend: bool=True ):
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
	fig, ax = plt.subplots( figsize=figsize )
	
	# Grid (subtle)
	if grid:
		ax.grid( True, alpha=0.25, linewidth=0.8 )
	
	# Titles
	ax.set_title( title, fontsize=12, fontweight="bold", pad=8 )
	if subtitle:
		ax.text( 0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
			alpha=0.85 )
	
	# Reference line
	if ref_line is not None:
		ax.axhline( ref_line, color="black", linestyle="--", linewidth=1.2, alpha=0.7 )
	
	# Legend handling
	if not legend:
		ax.get_legend( ).remove( )
	else:
		ax.legend( frameon=False )
	
	return fig, ax

def blue_divider( ) -> None:
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

def log_step( msg: str ) -> None:
	st.session_state.pipeline_log.append( msg )

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

def styled_scatter( ax: plt.Axes, x: np.ndarray, y: np.ndarray, series_index: int=0,
	label: Optional[ str ] = None, size: int=30, ) -> None:
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
	ax.scatter( x, y, s=size, alpha=0.9, edgecolors="#020617", linewidths=0.6, c=[ color ],
		marker=marker, label=label, )
	ax.grid( True, alpha=0.25 )

def auto_float_format( series: pd.Series, max_decimals: int=4 ) -> str:
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
	out = df.replace( [ np.inf, -np.inf ], np.nan )
	for c in out.columns:
		out[ c ] = pd.to_numeric( out[ c ], errors="coerce" )
	out = out.dropna( axis=1, how="all" )
	out = out.loc[ :, out.nunique( dropna=True ) > 1 ]
	return out

def analysis_fillna_mean( df: pd.DataFrame ) -> pd.DataFrame:
	return df.apply( lambda c: c.fillna( c.mean( ) ) if c.dtype.kind in "fc" else c )

def default_pick( items: List[ str ], k: int=2 ) -> List[ str ]:
	return items[ : min( k, len( items ) ) ] if items else [ ]

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown( """
		<style>
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stMarkdownContainer"] h4,
		div[data-testid="stMarkdownContainer"] h6,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h4 {
			color: rgb(053, 149, 252) !important;
		}
		</style>
		""", unsafe_allow_html=True, )

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
	return [ c for c in df_frame.columns if pd.api.types.is_numeric_dtype( df_frame[ c ] ) ]

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
	return [ c for c in df_frame.columns if not pd.api.types.is_numeric_dtype( df_frame[ c ] ) ]

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
	return [ c for c in st.session_state.get( 'features', [ ] ) if c in df_frame.columns ]

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
	return [ c for c in st.session_state.get( 'targets', [ ] ) if c in df_frame.columns ]

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

def normalize_result_frame( result: object, index: pd.Index, prefix: str,
	columns: list[ str ] ) -> pd.DataFrame:
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
			df_part = normalize_result_frame( item, index=index, prefix=f'{prefix}_{i + 1}',
				columns=None )
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

def replace_columns( df_frame: pd.DataFrame, column_names: list[ str ], result: object,
	prefix: str ) -> pd.DataFrame:
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
	df_result = normalize_result_frame( result=result, index=df_frame.index, prefix=prefix,
		columns=column_names )
	
	df_updated = df_frame.drop( columns=column_names, errors='ignore' )
	df_updated = pd.concat( [ df_updated, df_result ], axis=1 )
	return df_updated

def apply_text_vectorizer( df_frame: pd.DataFrame, column_names: list[ str ], vectorizer: object,
	prefix: str ) -> pd.DataFrame:
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
	df_result = normalize_result_frame( result=result, index=df_frame.index, prefix=prefix,
		columns=None )
	
	df_updated = df_frame.drop( columns=column_names, errors='ignore' )
	df_updated = pd.concat( [ df_updated, df_result ], axis=1 )
	return df_updated

def apply_dict_transform( df_frame: pd.DataFrame, column_names: list[ str ], transformer: object,
	prefix: str ) -> pd.DataFrame:
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
	df_result = normalize_result_frame( result=result, index=df_frame.index, prefix=prefix,
		columns=None )
	
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
		lambda s: [ item.strip( ) for item in s.split( delimiter ) if item.strip( ) ] )
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
	mapper = { 'chi2': sf.chi2, 'f_classif': sf.f_classif, 'f_regression': sf.f_regression,
		'mutual_info_classif': sf.mutual_info_classif,
		'mutual_info_regression': sf.mutual_info_regression }
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
		conn.execute( """
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
		              """ )
		
		conn.execute( """
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
		              """ )
		
		conn.execute( """
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
		              """ )
		
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

def read_table( table: str, limit: int=None, offset: int=0 ) -> pd.DataFrame:
	"""
	
		Purpose:
		--------
		Read a SQLite table into a pandas DataFrame using a normalized scalar-only path.
	
		Parameters:
		-----------
		table : str
			Table name.
		limit : int=None
			Optional row limit.
		offset : int=0
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
		display_df[ col ] = display_df[ col ].map( lambda x: '' if x is None else str( x ) )
	
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
			df_plot[ col ] = df_plot[ col ].map( lambda x: '' if x is None else str( x ) )
	
	numeric_cols: List[ str ] = [ ]
	for col in df_plot.columns:
		series_num = pd.to_numeric( df_plot[ col ], errors='coerce' )
		if series_num.notna( ).any( ):
			numeric_cols.append( col )
	
	categorical_columns: List[ str ] = [ col for col in df_plot.columns if col not in
	                                                                       numeric_cols ]
	
	chart = st.selectbox( 'Chart Type',
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
		
		fig = go.Figure( data=[ go.Heatmap( z=corr.values.tolist( ), x=corr.columns.tolist( ),
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
	blocked_keywords = ('insert ', 'update ', 'delete ', 'drop ', 'alter ', 'create ', 'attach ',
		'detach ', 'vacuum ', 'replace ', 'trigger ')
	
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
		conn.execute( f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};' )
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
				f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute( """
                            SELECT sql
                            FROM sqlite_master
                            WHERE type ='table' AND name =?
		                    """, (table_name,) ).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute( """
                                SELECT sql
                                FROM sqlite_master
                                WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
		                        """, (table_name,) ).fetchall( )
		
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
			f'INSERT INTO "{temp_table}" ({new_insert}) SELECT {old_select} FROM "{table_name}";' )
		
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
		row = { 'column': col, 'dtype': str( series.dtype ),
			'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
			'distinct_%': round( (distinct_count / total_rows) * 100, 2 ) if total_rows else 0, }
		
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
		row = conn.execute( """
                            SELECT sql
                            FROM sqlite_master
                            WHERE type ='table' AND name =?
		                    """, (table,) ).fetchone( )
		
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
		
		new_create_sql = (f'CREATE TABLE "{temp_table}" (' + ", ".join( new_defs ) + ");")
		
		# ------------------------------------------------------------
		# Begin transaction
		# ------------------------------------------------------------
		conn.execute( "BEGIN" )
		
		conn.execute( new_create_sql )
		
		remaining_cols = [ c.split( )[ 0 ].strip( '"' ) for c in new_defs ]
		
		col_list = ", ".join( [ f'"{c}"' for c in remaining_cols ] )
		
		conn.execute( f'INSERT INTO "{temp_table}" ({col_list}) '
		              f'SELECT {col_list} FROM "{table}";' )
		
		# Preserve indexes
		indexes = conn.execute( """
                                SELECT sql
                                FROM sqlite_master
                                WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
		                        """, (table,) ).fetchall( )
		
		conn.execute( f'DROP TABLE "{table}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table}";' )
		
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
		
		row = conn.execute( """
                            SELECT sql
                            FROM sqlite_master
                            WHERE type ='table' AND name =?
		                    """, (old_name,) ).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute( """
                                SELECT sql
                                FROM sqlite_master
                                WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
		                        """, (old_name,) ).fetchall( )
		
		open_paren = create_sql.find( "(" )
		if open_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		temp_name = f"{new_name}__rebuild_temp"
		
		conn.execute( "BEGIN" )
		conn.execute( f'CREATE TABLE "{temp_name}" {create_sql[ open_paren: ]}' )
		
		cols = [ r[ 1 ] for r in conn.execute( f'PRAGMA table_info("{old_name}");' ).fetchall( ) ]
		col_list = ", ".join( [ f'"{c}"' for c in cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_name}" ({col_list}) SELECT {col_list} FROM "{old_name}";' )
		
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
st.set_page_config( page_title='Mathy', layout='wide', page_icon=cfg.FAVICON,
	initial_sidebar_state='expanded' )

st.logo( image=cfg.LOGO, size='large', link=cfg.REPO_URL )
pd.options.display.float_format = '{:,.2f}'.format

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
	st.sidebar.divider( )
	st.subheader( 'Data Source' )
	
	with st.expander( 'Select Source', expanded=False ):
		source = st.selectbox( label='Select Source',
			options=[ 'Default Data', 'Database Data', 'Custom Data' ], key='source_selectbox' )
	
	uploaded = st.file_uploader( label='Upload Spreadsheet', type=[ 'xlsx', 'xls', 'csv' ],
		key='source_uploader' )
	
	loaded_df: pd.DataFrame | None = None
	loaded_original: pd.DataFrame | None = None
	
	if source == 'Default Data':
		loaded_df = pd.read_excel( cfg.DEFAULT_DATA )
		loaded_original = loaded_df.copy( )
		log_step( 'Loaded Default Dataset' )
	
	elif source == 'Database Data':
		try:
			with sqlite3.connect( cfg.DB_PATH ) as connection:
				df_tables = pd.read_sql_query( """
                                               SELECT name
                                               FROM sqlite_master
                                               WHERE type = 'table'
                                                 AND name NOT LIKE 'sqlite_%'
                                               ORDER BY name;
				                               """, connection )
				
				table_options = df_tables[ 'name' ].tolist( )[ : ]
				if table_options:
					selected_table = st.selectbox( label='Select Database Table',
						options=table_options, key='database_table_selectbox' )
					
					if selected_table:
						loaded_df = pd.read_sql_query( f'SELECT * FROM "{selected_table}"',
							connection )
						loaded_original = loaded_df.copy( )
						log_step( f'Loaded Database Table: {selected_table}' )
				else:
					st.warning( 'No tables were found in the database.' )
		except Exception as ex:
			st.error( f'Error loading database data: {ex}' )
	
	elif source == 'Custom Data':
		if uploaded is not None:
			if uploaded.name.lower( ).endswith( ('.xlsx', '.xls') ):
				loaded_df = pd.read_excel( uploaded )
			else:
				loaded_df = pd.read_csv( uploaded )
			
			loaded_original = loaded_df.copy( )
			log_step( f'Loaded uploaded file: {uploaded.name}' )
		else:
			st.info( 'Upload a spreadsheet to load data.' )
	
	if has_loaded_dataset( loaded_df ):
		store_loaded_dataset( loaded_df, loaded_original )
	
	def get_visualization_modes( df_frame: pd.DataFrame | None ) -> list[ str ]:
		"""
			Purpose:
			--------
			Identify visualization modes supported by the schema and contents of the
			currently loaded dataframe.

			Parameters:
			-----------
			df_frame ( pd.DataFrame | None ): Currently loaded dataframe.

			Returns:
			--------
			list[ str ]:
				Visualization modes supported by the available data.
		"""
		if df_frame is None or df_frame.empty:
			return [ ]
		
		numeric_columns = [ column for column in df_frame.columns
			if pd.api.types.is_numeric_dtype( df_frame[ column ] )
			and not pd.api.types.is_bool_dtype( df_frame[ column ] ) ]
		
		datetime_columns = [ column for column in df_frame.columns
			if pd.api.types.is_datetime64_any_dtype( df_frame[ column ] ) ]
		
		categorical_columns = [ column for column in df_frame.columns
			if column not in numeric_columns and column not in datetime_columns ]
		
		visualization_modes = [ 'Data Overview' ]
		
		if numeric_columns:
			visualization_modes.append( 'Numeric Distributions' )
		
		if len( numeric_columns ) >= 2:
			visualization_modes.append( 'Correlation Analysis' )
			visualization_modes.append( 'Scatter Analysis' )
		
		if categorical_columns:
			visualization_modes.append( 'Categorical Distributions' )
		
		if numeric_columns and categorical_columns:
			visualization_modes.append( 'Category Comparisons' )
		
		if datetime_columns and numeric_columns:
			visualization_modes.append( 'Time-Series Visualization' )
		
		if df_frame.isna( ).any( ).any( ):
			visualization_modes.append( 'Missing Data Visualization' )
		
		return visualization_modes
	
	
	def handle_ml_mode_change( ) -> None:
		"""
			Purpose:
			--------
			Activate the selected machine-learning mode, clear inactive selections,
			and reset mode-specific processing state when required.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		"""
		selected_mode = st.session_state.get( 'ml_mode_radio', None )
		
		if selected_mode is None:
			return
		
		st.session_state[ 'db_mode_radio' ] = None
		st.session_state[ 'visualization_mode_radio' ] = None
		st.session_state[ 'active_mode' ] = selected_mode
		
		if selected_mode == 'Classification Models':
			reset_classification_mode_state( )
		elif selected_mode == 'Regression Models':
			reset_regression_mode_state( )
	
	
	def handle_db_mode_change( ) -> None:
		"""
			Purpose:
			--------
			Activate the selected data-management mode and clear inactive mode
			selections.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		"""
		selected_mode = st.session_state.get( 'db_mode_radio', None )
		
		if selected_mode is None:
			return
		
		st.session_state[ 'ml_mode_radio' ] = None
		st.session_state[ 'visualization_mode_radio' ] = None
		st.session_state[ 'active_mode' ] = selected_mode
	
	
	def handle_visualization_mode_change( ) -> None:
		"""
			Purpose:
			--------
			Activate the selected visualization mode and clear inactive mode
			selections.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		"""
		selected_mode = st.session_state.get( 'visualization_mode_radio', None )
		
		if selected_mode is None:
			return
		
		st.session_state[ 'ml_mode_radio' ] = None
		st.session_state[ 'db_mode_radio' ] = None
		st.session_state[ 'active_mode' ] = selected_mode
	
	
	# ------- Available Modes
	ml_modes = list( cfg.ML_MODE )
	db_modes = list( cfg.DB_MODE )
	
	df_loaded = st.session_state.get( 'df_dataset', None )
	visualization_modes = get_visualization_modes( df_loaded )
	
	# ------- Mode State Initialization
	if 'active_mode' not in st.session_state:
		st.session_state[ 'active_mode' ] = ml_modes[ 0 ] if ml_modes else None
	
	if 'ml_mode_radio' not in st.session_state:
		st.session_state[ 'ml_mode_radio' ] = st.session_state[ 'active_mode' ]
	
	if 'db_mode_radio' not in st.session_state:
		st.session_state[ 'db_mode_radio' ] = None
	
	if 'visualization_mode_radio' not in st.session_state:
		st.session_state[ 'visualization_mode_radio' ] = None
	
	available_modes = ml_modes + db_modes + visualization_modes
	if st.session_state[ 'active_mode' ] not in available_modes:
		st.session_state[ 'active_mode' ] = ml_modes[ 0 ] if ml_modes else None
		st.session_state[ 'ml_mode_radio' ] = st.session_state[ 'active_mode' ]
		st.session_state[ 'db_mode_radio' ] = None
		st.session_state[ 'visualization_mode_radio' ] = None
	
	if st.session_state.get( 'ml_mode_radio', None ) not in ml_modes:
		st.session_state[ 'ml_mode_radio' ] = None
	
	if st.session_state.get( 'db_mode_radio', None ) not in db_modes:
		st.session_state[ 'db_mode_radio' ] = None
	
	if st.session_state.get( 'visualization_mode_radio', None ) not in visualization_modes:
		st.session_state[ 'visualization_mode_radio' ] = None
	
	# ------- Machine Learning Selection Mode
	st.sidebar.divider( )
	with st.expander( 'Machine Learning', expanded=True ):
		st.radio( label='Select', options=ml_modes, index=None,
			key='ml_mode_radio', on_change=handle_ml_mode_change )
	
	# ------- Data Management Selection Mode
	st.sidebar.divider( )
	with st.expander( 'Data Management', expanded=True ):
		st.radio( label='Select', options=db_modes, index=None,
			key='db_mode_radio', on_change=handle_db_mode_change )
	
	# ------- Data Visualization Selection Mode
	st.sidebar.divider( )
	with st.expander( 'Data Visualization', expanded=True ):
		if not visualization_modes:
			st.info( 'Load a dataset to enable visualization modes.' )
		else:
			st.radio( label='Select', options=visualization_modes, index=None,
				key='visualization_mode_radio',
				on_change=handle_visualization_mode_change )
	
	# ------- Active Application Mode
	mode = st.session_state[ 'active_mode' ]

style_subheaders( )

# ============================================
# DATA PROFILING MODE
# ============================================
if mode == 'Data Profile':
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Data Profile' ] )
		st.divider( )
		
		df_dataset = get_loaded_dataset( )
		if df_dataset is None:
			st.info( 'No data loaded.' )
			st.stop( )
		
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
					if ('id' in name) or ('code' in name) or ('key' in name) or (
							unique_ratio > 0.8):
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
		blue_divider( )
		st.markdown( '##### Types' )
		type_counts = pd.Series( schema ).value_counts( )
		m1, m2, m3, m4, m5 = st.columns( 5, border=True )
		m1.metric( 'Rows', len( df_dataset ) )
		m2.metric( 'Numeric', type_counts.get( 'numeric', 0 ) )
		m3.metric( 'Ordinal / ID',
			type_counts.get( 'ordinal', 0 ) + type_counts.get( 'identifier', 0 ) )
		m4.metric( 'Categorical', type_counts.get( 'categorical', 0 ) )
		m5.metric( 'Datetime', type_counts.get( 'datetime', 0 ) )
		
		blue_divider( )
		st.markdown( '##### Records' )
		
		with st.expander( label='Edit', icon='✏️', expanded=True ):
			if df_dataset is None or df_dataset.empty:
				st.info( 'No rows are available to edit.' )
			else:
				top_c1, top_c2 = st.columns( [ 0.20, 0.80 ] )
				
				with top_c1:
					max_row_index = len( df_dataset ) - 1
					default_row_index = int( st.session_state.get( 'row_editor_index', 0 ) )
					default_row_index = max( 0, min( default_row_index, max_row_index ) )
					
					row_idx = st.number_input( 'Select Index', min_value=0,
						max_value=max_row_index,
						value=default_row_index, step=1, key='row_editor_index' )
				
				row = df_dataset.iloc[ row_idx ]
				updated = { }
				
				col_left, col_right = st.columns( 2, border=True )
				
				with st.form( 'row_edit_form' ):
					for i, (col, dtype) in enumerate( schema.items( ) ):
						target = col_left if i % 2 == 0 else col_right
						val = row[ col ]
						
						with target:
							if dtype == 'numeric':
								updated[ col ] = st.number_input( col,
									value=float( val ) if pd.notna( val ) else 0.0 )
							elif dtype == 'ordinal':
								updated[ col ] = st.number_input( col,
									value=int( val ) if pd.notna( val ) else 0 )
							elif dtype == 'datetime':
								updated[ col ] = st.date_input( col,
									value=pd.to_datetime( val ).date( ) if pd.notna(
										val ) else pd.Timestamp.today( ).date( ) )
							elif dtype == 'categorical':
								options = df_dataset[ col ].dropna( ).unique( ).tolist( )
								if options:
									updated[ col ] = st.selectbox( col, options,
										index=options.index( val ) if val in options else 0 )
								else:
									updated[ col ] = st.text_input( col,
										value='' if pd.isna( val ) else str( val ) )
							else:
								updated[ col ] = st.text_input( col, value=str( val ),
									disabled=True )
					
					submitted = st.form_submit_button( 'Apply Row Update' )
				
				if submitted:
					before = df_dataset.loc[ row_idx ].copy( )
					for col, value in updated.items( ):
						if schema[ col ] == 'datetime':
							st.session_state.df_dataset.at[ row_idx, col ] = pd.to_datetime(
								value )
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
		blue_divider( )
		st.markdown( '##### Diagnostics' )
		
		v1, v2 = st.columns( 2, border=True )
		with v1:
			fig, ax = plt.subplots( figsize=(6, 4.5) )
			type_counts.sort_values( ascending=False ).plot( kind='bar', ax=ax, width=0.75,
				edgecolor='#0f172a', linewidth=0.9 )
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
				missing_pct.sort_values( ascending=True ).plot( kind='barh', ax=ax, width=0.75,
					edgecolor='#0f172a', linewidth=0.9 )
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
		
		blue_divider( )
		st.markdown( '##### Cardinality', help=cfg.DATA_CARDINALITY )
		
		v3, v4 = st.columns( 2, border=True )
		with v3:
			cardinality = df_dataset.nunique( dropna=True ).sort_values( ascending=False ).head(
				10 )
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
		blue_divider( )
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
				rename_col = st.selectbox( 'Rename Column',
					[ '<None>' ] + df_dataset.columns.tolist( ) )
				new_name = st.text_input( 'New Column Name' )
				if st.button( 'Rename' ):
					if rename_col != '<None>' and new_name:
						if new_name in df_dataset.columns:
							st.error( 'Column Name Already Exists.' )
						else:
							st.session_state.df_dataset = df_dataset.rename(
								columns={ rename_col: new_name } )
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
					st.session_state.df_dataset.to_csv( index=False ), 'dataset.csv', 'text/csv',
					icon='📥', )
		
		# -------------------------------------------------------------------------------------
		# Probability Distributions
		# -------------------------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Numeric Distributions' )
		
		numeric_dist_cols = [ c for c in df_dataset.columns if
			pd.api.types.is_numeric_dtype( df_dataset[ c ] ) and not pd.api.types.is_bool_dtype(
				df_dataset[ c ] ) ]
		
		if not numeric_dist_cols:
			st.info( 'No numeric columns detected.' )
		else:
			st.caption( f'{len( numeric_dist_cols )} numeric column(s) detected.' )
			
			ctrl1, ctrl2, ctrl3 = st.columns( 3, border=True )
			with ctrl1:
				dist_bins = st.slider( 'Bins', min_value=10, max_value=60, value=30, step=5,
					key='profile_numeric_dist_bins' )
			
			with ctrl2:
				show_kde = st.checkbox( 'Show KDE Overlay', value=True,
					key='profile_numeric_dist_kde' )
			
			with ctrl3:
				dist_mode = st.radio( 'Display', options=[ 'Density', 'Frequency' ],
					horizontal=True, key='profile_numeric_dist_mode' )
			
			st.markdown( """
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
					sns.histplot( s, bins=dist_bins, kde=show_kde, stat=stat_mode, ax=ax,
						edgecolor='#0f172a', line_kws={ 'linewidth': 2.0 } if show_kde else None )
					
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
					m4.metric( 'Std',
						f'{float( s.std( ddof=1 ) ):,.2f}' if len( s ) > 1 else '0.00' )
		
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
		
		st.markdown( """
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
			summary_vars = st.multiselect( 'Variables for Summary Table', all_num_cols,
				default=all_num_cols[ : min( 8, len( all_num_cols ) ) ], key='desc_summary_vars' )
		
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
					df_dataset[ summary_vars ].isna( ).mean( ).values * 100.0)
			
			df_descriptive[ 'Skew' ] = df_summary_source.skew( ).values
			df_descriptive[ 'Kurtosis' ] = df_summary_source.kurtosis( ).values
			df_descriptive[ 'Zeros' ] = (df_summary_source == 0).sum( ).values
			df_descriptive[ 'Zeros %' ] = ((df_summary_source == 0).mean( ).values * 100.0)
			
			ordered_cols = [ 'Variable', 'count', 'mean', 'std', 'Variance', 'min' ]
			if show_percentiles:
				for pcol in [ '5%', '25%', '50%', '75%', '95%' ]:
					if pcol in df_descriptive.columns:
						ordered_cols.append( pcol )
			ordered_cols += [ 'max', 'Missing', 'Missing %', 'Zeros', 'Zeros %', 'Skew',
				'Kurtosis' ]
			ordered_cols = [ c for c in ordered_cols if c in df_descriptive.columns ]
			df_descriptive = df_descriptive[ ordered_cols ]
			
			for c in df_descriptive.columns:
				if c != 'Variable':
					df_descriptive[ c ] = pd.to_numeric( df_descriptive[ c ], errors='coerce' )
			
			column_config = { 'Variable': st.column_config.TextColumn( 'Variable',
				width='medium' ),
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
				'Kurtosis': st.column_config.NumberColumn( 'Kurtosis', format='%.4f' ) }
			
			column_config = { k: v for k, v in column_config.items( ) if
				k in df_descriptive.columns }
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
			
			blue_divider( )
			st.markdown( f'##### Distribution & Shape — {col}' )
			c1, c2 = st.columns( 2, border=True )
			with c1:
				fig, ax = plt.subplots( figsize=(7, 4.75) )
				sns.histplot( s, bins=dist_bins, kde=True, stat='count', ax=ax,
					edgecolor='#0f172a',
					line_kws={ 'linewidth': 2.0 } )
				
				mean_val = float( s.mean( ) )
				median_val = float( s.median( ) )
				ax.axvline( mean_val, linestyle='--', linewidth=1.5,
					label=f'Mean: {mean_val:,.2f}' )
				
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
				m4.metric( 'Deviation',
					f'{float( s.std( ddof=1 ) ):,.2f}' if len( s ) > 1 else '0.000' )
			
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
						q1.metric( 'Skew',
							f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
						q2.metric( 'Kurtosis',
							f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
						q3.metric( 'Shapiro P', f'{shapiro_p:,.3f}' )
					else:
						q1, q2, q3 = st.columns( 3, border=True )
						q1.metric( 'Skew',
							f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
						q2.metric( 'Kurtosis',
							f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
						q3.metric( 'Shapiro P', 'n/a' )
				except Exception:
					q1, q2, q3 = st.columns( 3, border=True )
					q1.metric( 'Skew',
						f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
					q2.metric( 'Kurtosis',
						f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
					q3.metric( 'Shapiro P', 'n/a' )
		
		blue_divider( )
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
				sns.heatmap( corr, ax=ax, cmap='coolwarm', vmin=-1, vmax=1, center=0, annot=True,
					fmt='.2f', square=False, linewidths=0.5,
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
		
		blue_divider( )
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
			
			df_explained = pd.DataFrame( { 'Component': [ f'PC{i + 1}' for i in range( n_comp ) ],
				'Explained Variance (%)': pca.explained_variance_ratio * 100 } )
			
			with c5:
				render_table( df_explained )
			
			with c6:
				fig, ax = plt.subplots( figsize=(7, 5) )
				bars = ax.bar( df_explained[ 'Component' ],
					df_explained[ 'Explained Variance (%)' ], edgecolor='#0f172a', linewidth=0.9 )
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
				st.caption(
					'Explained variance chart will appear here once at least two variables are '
					'selected.' )

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
		
		st.markdown( """
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
				infer_rows.append( { 'Analysis': 'Outcome Distribution', 'Test': 'Shapiro-Wilk',
					'Statistic': shapiro_stat, 'P-Value': shapiro_p, 'DoF': np.nan,
					'Effect Size': np.nan, 'N': float( len( summary_series ) ),
					'Notes': 'Normality Assessment' } )
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
					infer_rows.append( { 'Analysis': 'Group Comparison', 'Test': 'One-Way ANOVA',
						'Statistic': f_stat, 'P-Value': p_anova,
						'DoF': float( len( valid_group_arrays ) - 1 ), 'Effect Size': np.nan,
						'N': float( sum( len( g ) for g in valid_group_arrays ) ),
						'Notes': f'{summary_y} by {summary_group}' } )
				except Exception:
					pass
				
				try:
					h_stat, p_kw = stats.kruskal( *valid_group_arrays )
					infer_rows.append( { 'Analysis': 'Group Comparison', 'Test': 'Kruskal-Wallis',
						'Statistic': h_stat, 'P-Value': p_kw,
						'DoF': float( len( valid_group_arrays ) - 1 ), 'Effect Size': np.nan,
						'N': float( sum( len( g ) for g in valid_group_arrays ) ),
						'Notes': f'{summary_y} by {summary_group}' } )
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
					
					infer_rows.append( { 'Analysis': 'Association', 'Test': 'Pearson Correlation',
						'Statistic': pearson_r, 'P-Value': pearson_p,
						'DoF': float( pair_mask.sum( ) - 2 ), 'Effect Size': abs( pearson_r ),
						'N': float( pair_mask.sum( ) ), 'Notes': f'{summary_y} vs {summary_x}' } )
				except Exception:
					pass
				
				try:
					spearman_rho, spearman_p = stats.spearmanr( x_summary[ pair_mask ],
						y_summary[ pair_mask ] )
					
					infer_rows.append( { 'Analysis': 'Association', 'Test': 'Spearman Correlation',
						'Statistic': spearman_rho, 'P-Value': spearman_p, 'DoF': np.nan,
						'Effect Size': abs( spearman_rho ), 'N': float( pair_mask.sum( ) ),
						'Notes': f'{summary_y} vs {summary_x}' } )
				except Exception:
					pass
		
		# -----------------------------------------------------------------
		# Categorical Association Summary
		# -----------------------------------------------------------------
		if summary_cat1 and summary_cat2:
			contingency_summary = pd.crosstab( df_dataset[ summary_cat1 ],
				df_dataset[ summary_cat2 ] )
			
			if not contingency_summary.empty and contingency_summary.shape[ 0 ] >= 2 and \
					contingency_summary.shape[ 1 ] >= 2:
				try:
					chi2_stat, chi2_p, chi2_dof, expected = stats.chi2_contingency(
						contingency_summary )
					
					n_total = contingency_summary.to_numpy( ).sum( )
					phi2 = chi2_stat / n_total if n_total > 0 else np.nan
					r_dim, c_dim = contingency_summary.shape
					cramers_v = (np.sqrt( phi2 / min( c_dim - 1, r_dim - 1 ) ) if min( c_dim - 1,
						r_dim - 1 ) > 0 else np.nan)
					
					infer_rows.append(
						{ 'Analysis': 'Categorical Association', 'Test': 'Chi-Square',
							'Statistic': chi2_stat, 'P Value': chi2_p, 'DoF': float( chi2_dof ),
							'Effect Size': cramers_v, 'N': float( n_total ),
							'Notes': f'{summary_cat1} vs {summary_cat2}' } )
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
				'Notes': st.column_config.TextColumn( 'Notes', width='large' ) }
			
			st.data_editor( df_infer_summary, use_container_width=True, hide_index=True,
				disabled=True, column_config=infer_column_config, key='infer_summary_editor' )
		else:
			st.info( 'Unable to compute inferential summary for the current selections.' )
		
		blue_divider( )
		
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
					st.caption(
						'Distribution does not significantly depart from normality at α=0.05' )
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
					sns.stripplot( data=df_group, x=col_group, y=col_y, ax=ax, color='black',
						alpha=0.45, size=4 )
					ax.set_title( f'Group Comparison — {col_y} by {col_group}', fontsize=12,
						fontweight='bold', pad=10 )
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
					            f'Use the nonparametric result when normality or homoscedasticity '
					            f'is doubtful.' )
				else:
					st.info( 'Not enough valid groups for group comparison.' )
			else:
				st.info( 'Select a grouping variable to compare groups.' )
		
		blue_divider( )
		
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
						xline = np.linspace( float( x[ mask ].min( ) ), float( x[ mask ].max( ) ),
							100 )
						ax.plot( xline, m * xline + b, linewidth=2.0, linestyle='--' )
					except Exception:
						pass
				
				ax.set_title( f'Correlation — {col_y} vs {col_x2}', fontsize=12, fontweight='bold',
					pad=10 )
				
				ax.set_xlabel( col_x2 )
				ax.set_ylabel( col_y )
				ax.grid( True, alpha=0.20, linestyle='--' )
				ax.spines[ 'top' ].set_visible( False )
				ax.spines[ 'right' ].set_visible( False )
				fig.tight_layout( )
				st.pyplot( fig )
				plt.close( fig )
		
		blue_divider( )
		
		# -------------------------------------------------------------------------------------
		# CATEGORICAL ASSOCIATION
		# -------------------------------------------------------------------------------------
		st.markdown( '##### Categorical Association' )
		if not categorical_columns or len( categorical_columns ) < 2:
			st.info(
				'At least two categorical variables are required for categorical association.' )
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
				cramers_v = np.sqrt( phi2 / min( k - 1, r - 1 ) ) if min( k - 1,
					r - 1 ) > 0 else np.nan
				
				ca1, ca2 = st.columns( 2, border=True )
				with ca1:
					st.data_editor( contingency, key='inference_data', height='stretch',
						num_rows='dynamic' )
				
				with ca2:
					fig, ax = plt.subplots( figsize=(7, 5.5) )
					sns.heatmap( contingency, annot=True, fmt='d', cmap='Blues', linewidths=0.5,
						ax=ax, cbar_kws={ 'shrink': 0.85, 'label': 'Count' } )
					
					ax.set_title( f'Contingency Heatmap — {col_cat1} vs {col_cat2}', fontsize=12,
						fontweight='bold', pad=10 )
					
					ax.set_xlabel( col_cat2 )
					ax.set_ylabel( col_cat1 )
					fig.tight_layout( )
					st.pyplot( fig )
					plt.close( fig )
				
				cm1, cm2, cm3, cm4 = st.columns( 4, border=True )
				cm1.metric( 'Chi-Square', f'{chi2:,.2f}' )
				cm2.metric( 'P Value', f'{p_chi:,.2g}' )
				cm3.metric( 'DoF', f'{dof:,}' )
				cm4.metric( "Cramér's V",
					f'{cramers_v:,.2f}' if np.isfinite( cramers_v ) else 'n/a' )

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
		
		st.markdown( """
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
				df_analysis = pd.DataFrame( SKStandardScaler( ).fit_transform(
					df_analysis.values ),
					columns=df_analysis.columns, index=df_analysis.index )
			
			if analysis_scale and len( vars_sel ) > 1:
				df_analysis[ : ] = SKStandardScaler( ).fit_transform( df_analysis.values )
		
		# -------------------------------------------------------------------------
		# Method Selection
		# -------------------------------------------------------------------------
		blue_divider( )
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
		blue_divider( )
		st.markdown( '##### Thresholds' )
		
		c_t1, c_t2 = st.columns( 2, border=True )
		with c_t1:
			z_thresh = st.slider( 'Z / Modified Z threshold', 2.0, 5.0, 3.0, 0.1,
				help=cfg.MODIFIED_Z )
			iqr_mult = st.slider( 'IQR Multiplier', 1.0, 3.0, 1.5, 0.1, help=cfg.IQR_MULTIPLIER )
		
		with c_t2:
			lof_k = st.slider( 'LOF Neighbors (k)', 5, 50, 20, 1, help=cfg.LOF_K )
			min_methods = st.slider( 'Consensus: minimum methods flagging a row', 1, 4, 1, 1,
				help=cfg.MIN_METHODS )
		
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
		blue_divider( )
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
				bars = ax.bar( vc.index.astype( str ), vc.values, width=0.75, edgecolor='black',
					linewidth=0.9 )
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
		blue_divider( )
		st.markdown( '##### Empirical Cumulative Distribution Function (ECDF)', help=cfg.ECDF )
		
		for col in vars_sel:
			if col not in df_analysis.columns:
				continue
			
			s = pd.to_numeric( df_analysis[ col ], errors='coerce' ).replace( [ np.inf, -np.inf ],
				np.nan )
			
			s_clean = s.dropna( )
			if s_clean.empty:
				continue
			
			flagged_idx = anomalies.index.intersection( s_clean.index )
			flagged_vals = s_clean.loc[ flagged_idx ] if not flagged_idx.empty else pd.Series(
				dtype=float )
			
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
					flierprops=dict( marker='o', markerfacecolor='#475569',
						markeredgecolor='black',
						markersize=4, alpha=0.7 ) )
				
				if not flagged_vals.empty:
					ax.scatter( flagged_vals.values, np.ones( len( flagged_vals ) ),
						color='crimson', alpha=0.85, s=34, edgecolors='black', linewidths=0.4,
						label='Flagged', zorder=3 )
				
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
		
		blue_divider( )
		
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
					df_scatter.loc[ ~flag_mask, y_col ].values, s=34, alpha=0.70,
					edgecolors='black', linewidths=0.5, label='Inliers' )
				
				if flag_mask.any( ):
					ax.scatter( df_scatter.loc[ flag_mask, x_col ].values,
						df_scatter.loc[ flag_mask, y_col ].values, s=52, alpha=0.92,
						edgecolors='black', linewidths=0.7, c='crimson', marker='X',
						label='Flagged' )
				
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
	selected_all = st.session_state.get( 'selected_all', [ ] )
	active_features = st.session_state.get( 'active_features', [ ] )
	active_targets = st.session_state.get( 'active_targets', [ ] )
	model = st.session_state.get( 'model', None )
	X_data = st.session_state.get( 'X_data', None )
	X_train = st.session_state.get( 'X_train', None )
	X_test = st.session_state.get( 'X_test', None )
	y_train = st.session_state.get( 'y_train', None )
	y_test = st.session_state.get( 'y_test', None )
	y_series = st.session_state.get( 'y_series', None )
	y_prediction = st.session_state.get( 'y_prediction', None )
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
		numeric_columns = [ column for column in df_original.columns if
			pd.api.types.is_numeric_dtype( df_original[ column ] ) ]
		
		categorical_columns = [ column for column in df_original.columns if
			column not in numeric_columns ]
		
		st.session_state[ 'numeric_columns' ] = numeric_columns
		st.session_state[ 'categorical_columns' ] = categorical_columns
		if len( df_original.columns ) < 2:
			warn = '⚠️ Classification requires at least one feature column and one target column.'
			st.warning( warn )
			st.stop( )
		
		# ======================================================================================
		# Data Selection
		# ======================================================================================
		st.markdown( '##### Feature Selection' )
		st.caption( f'Samples: {len( df_original ):,} | Features: {len( df_original.columns ):,}' )
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			features = st.multiselect( 'Select Features', options=df_original.columns.tolist( ),
				key='classification_features' )
		
		with col_c2:
			target_options = [ column for column in df_original.columns if column not in features ]
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
		
		blue_divider( )
		if df_working is None:
			st.stop( )
		st.markdown( '##### Working Data' )
		
		st.caption( f'Samples: {len( df_working ):,} | Feautres: {len( df_working.columns ):,}' )
		st.data_editor( df_working, key='classification_working_data' )
		
		# -----------------------------------------------------------------
		# Data Processing
		# -----------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Feature-Engineering' )
		
		def has_classification_frame( df_frame: object ) -> bool:
			"""

				Purpose:
				--------
				Determine whether the supplied object is a usable classification dataframe.

				Parameters:
				-----------
				df_frame ( object ): Candidate dataframe object.

				Returns:
				--------
				bool: True when the object is a non-empty dataframe with one or more
				columns.

			"""
			return (isinstance( df_frame, pd.DataFrame ) and not df_frame.empty and len(
				df_frame.columns ) > 0)
		
		def get_classification_source_signature( df_frame: pd.DataFrame ) -> tuple:
			"""

				Purpose:
				--------
				Create a stable signature for the user-selected classification working frame.

				Parameters:
				-----------
				df_frame ( pd.DataFrame ): Current classification working dataframe.

				Returns:
				--------
				tuple: Source signature used to detect changed working datasets.

			"""
			index_sample = tuple( str( value ) for value in df_frame.index[ :10 ].tolist( ) )
			return (tuple( df_frame.columns.tolist( ) ), int( len( df_frame ) ), index_sample)
		
		def clear_classification_processed_state( ) -> None:
			"""

				Purpose:
				--------
				Clear only classification processing outputs without modifying df_working.

				Parameters:
				-----------
				None

				Returns:
				--------
				None

			"""
			st.session_state[ 'df_processed' ] = pd.DataFrame( )
			st.session_state[ 'df_features' ] = pd.DataFrame( )
			st.session_state[ 'df_targets' ] = pd.DataFrame( )
			st.session_state[ 'active_features' ] = [ ]
			st.session_state[ 'active_targets' ] = [ ]
			st.session_state[ 'df_model' ] = pd.DataFrame( )
			st.session_state[ 'df_scores' ] = pd.DataFrame( )
			st.session_state[ 'df_predictions' ] = pd.DataFrame( )
			st.session_state[ 'X_data' ] = None
			st.session_state[ 'X_train' ] = None
			st.session_state[ 'X_test' ] = None
			st.session_state[ 'y_train' ] = None
			st.session_state[ 'y_test' ] = None
			st.session_state[ 'y_series' ] = None
			st.session_state[ 'y_prediction' ] = None
			st.session_state[ 'model' ] = None
			st.session_state[ 'elapsed_seconds' ] = 0.0
		
		def get_classification_pipeline_frame( ) -> pd.DataFrame:
			"""

				Purpose:
				--------
				Return the current input frame for a classification preprocessing operation.

				Parameters:
				-----------
				None

				Returns:
				--------
				pd.DataFrame: df_processed when it exists; otherwise df_working.

			"""
			df_current = st.session_state.get( 'df_processed', pd.DataFrame( ) )
			if has_classification_frame( df_current ):
				return df_current.copy( )
			
			return st.session_state[ 'df_working' ].copy( )
		
		def get_classification_columns( numeric_only: bool = False,
			categorical_only: bool = False ) -> list[ str ]:
			"""

				Purpose:
				--------
				Return selectable preprocessing columns from the active pipeline frame.

				Parameters:
				-----------
				numeric_only ( bool ): Return only numeric columns when True.
				categorical_only ( bool ): Return only non-numeric columns when True.

				Returns:
				--------
				list[ str ]: Column names available for the requested operation.

			"""
			df_input = get_classification_pipeline_frame( )
			if numeric_only:
				return [ col for col in df_input.columns if
					pd.api.types.is_numeric_dtype( df_input[ col ] ) ]
			
			if categorical_only:
				return [ col for col in df_input.columns if
					not pd.api.types.is_numeric_dtype( df_input[ col ] ) ]
			
			return df_input.columns.tolist( )
		
		def prune_classification_multiselect( key: str, options: list[ str ] ) -> None:
			"""

				Purpose:
				--------
				Remove stale multiselect values before the widget is instantiated.

				Parameters:
				-----------
				key ( str ): Streamlit session-state key.
				options ( list[ str ] ): Current legal column choices.

				Returns:
				--------
				None

			"""
			if key in st.session_state and isinstance( st.session_state[ key ], list ):
				st.session_state[ key ] = [ value for value in st.session_state[ key ] if
					value in options ]
		
		def prune_classification_selectbox( key: str, options: list[ str ] ) -> None:
			"""

				Purpose:
				--------
				Remove a stale selectbox value before the widget is instantiated.

				Parameters:
				-----------
				key ( str ): Streamlit session-state key.
				options ( list[ str ] ): Current legal column choices.

				Returns:
				--------
				None

			"""
			if key in st.session_state and st.session_state[ key ] not in options:
				del st.session_state[ key ]
		
		def get_valid_classification_columns( column_names: list[ str ],
			df_frame: pd.DataFrame ) -> list[ str ]:
			"""

				Purpose:
				--------
				Filter selected column names to columns that still exist in the input frame.

				Parameters:
				-----------
				column_names ( list[ str ] ): Selected column names.
				df_frame ( pd.DataFrame ): Current input dataframe.

				Returns:
				--------
				list[ str ]: Valid selected column names.

			"""
			if not column_names:
				return [ ]
			
			return [ col for col in column_names if col in df_frame.columns ]
		
		def get_numeric_classification_subset( df_frame: pd.DataFrame,
			column_names: list[ str ] ) -> pd.DataFrame:
			"""

				Purpose:
				--------
				Return a numeric dataframe subset for estimators requiring numeric input.

				Parameters:
				-----------
				df_frame ( pd.DataFrame ): Input dataframe.
				column_names ( list[ str ] ): Columns to coerce to numeric values.

				Returns:
				--------
				pd.DataFrame: Numeric subset with invalid values converted and filled.

			"""
			return df_frame[ column_names ].apply( pd.to_numeric, errors='coerce' ).fillna( 0.0 )
		
		def require_classification_columns( column_names: list[ str ], label: str ) -> bool:
			"""

				Purpose:
				--------
				Display a warning when an operation has no valid selected columns.

				Parameters:
				-----------
				column_names ( list[ str ] ): Selected column names.
				label ( str ): Operation label.

				Returns:
				--------
				bool: True when one or more columns are selected.

			"""
			if not column_names:
				st.warning( f'Select at least one available column for {label}.' )
				return False
			
			return True
		
		def update_classification_pipeline_contract( df_before: pd.DataFrame,
			df_after: pd.DataFrame, source_columns: list[ str ],
			preserve_source_names: bool ) -> None:
			"""

				Purpose:
				--------
				Update active feature and target column contracts after preprocessing.

				Parameters:
				-----------
				df_before ( pd.DataFrame ): Dataframe before transformation.
				df_after ( pd.DataFrame ): Dataframe after transformation.
				source_columns ( list[ str ] ): Source columns consumed by the operation.
				preserve_source_names ( bool ): True when transformed columns retain their
				source names.

				Returns:
				--------
				None

			"""
			source_columns = source_columns or [ ]
			features_current = st.session_state.get( 'features', [ ] )
			targets_current = st.session_state.get( 'targets', [ ] )
			
			if preserve_source_names:
				st.session_state[ 'features' ] = [ col for col in features_current if
					col in df_after.columns ]
				st.session_state[ 'targets' ] = [ col for col in targets_current if
					col in df_after.columns ]
				return
			
			replacement_columns = [ col for col in df_after.columns if
				col not in df_before.columns ]
			
			if not replacement_columns:
				replacement_columns = [ col for col in df_after.columns if
					col not in [ old_col for old_col in df_before.columns if
						old_col not in source_columns ] ]
			
			features_next = [ ]
			for col in features_current:
				if col in source_columns:
					for new_col in replacement_columns:
						if new_col in df_after.columns and new_col not in features_next:
							features_next.append( new_col )
				elif col in df_after.columns and col not in features_next:
					features_next.append( col )
			
			targets_next = [ ]
			for col in targets_current:
				if col in source_columns:
					for new_col in replacement_columns:
						if new_col in df_after.columns and new_col not in targets_next:
							targets_next.append( new_col )
				elif col in df_after.columns and col not in targets_next:
					targets_next.append( col )
			
			st.session_state[ 'features' ] = features_next
			st.session_state[ 'targets' ] = targets_next
		
		def commit_classification_processed_frame( df_before: pd.DataFrame, df_after: pd.DataFrame,
			source_columns: list[ str ] | None = None, preserve_source_names: bool = True ) -> None:
			"""

				Purpose:
				--------
				Commit the current classification processed dataframe without changing
				df_working.

				Parameters:
				-----------
				df_before ( pd.DataFrame ): Dataframe before transformation.
				df_after ( pd.DataFrame ): Dataframe after transformation.
				source_columns ( list[ str ] | None ): Source columns transformed.
				preserve_source_names ( bool ): True when transformed columns retain their
				source names.

				Returns:
				--------
				None

			"""
			update_classification_pipeline_contract( df_before=df_before, df_after=df_after,
				source_columns=source_columns or [ ], preserve_source_names=preserve_source_names )
			
			st.session_state[ 'df_processed' ] = df_after.copy( )
			commit_frame( df_after )
		
		df_working_current = st.session_state[ 'df_working' ].copy( )
		source_signature = get_classification_source_signature( df_working_current )
		prior_signature = st.session_state.get( 'classification_processing_source_signature',
			None )
		
		if prior_signature is not None and prior_signature != source_signature:
			clear_classification_processed_state( )
		
		st.session_state[ 'classification_processing_source_signature' ] = source_signature
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			with st.expander( label='Data Scaling', icon='⚖️', key='classification_scalers' ):
				
				with st.expander( 'Standard Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.STANDARD_SCALER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_standard_scaler_cols',
						options )
					columns = st.multiselect( 'Columns', options=options,
						key='classification_standard_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', use_container_width=True,
								key='classification_standard_scaler_apply' ):
							df_input = get_classification_pipeline_frame( )
							columns = get_valid_classification_columns( columns, df_input )
							if require_classification_columns( columns, 'Standard Scaler' ):
								scaler = StandardScaler( )
								df_output = df_input.copy( )
								result = scaler.train_transform(
									get_numeric_classification_subset( df_input,
										columns ).to_numpy( ) )
								df_output[ columns ] = result
								commit_classification_processed_frame( df_input, df_output,
									columns,
									True )
								st.success( 'Standard Scaler applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_standard_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Min-Max Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.MINMAX_SCALER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_minmax_scaler_cols',
						options )
					scale_cols = st.multiselect( 'Columns', options=options,
						key='classification_minmax_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_minmax_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							scale_cols = get_valid_classification_columns( scale_cols, df_input )
							if require_classification_columns( scale_cols, 'Min-Max Scaler' ):
								scaler = MinMaxScaler( )
								df_output = df_input.copy( )
								result = scaler.train_transform(
									get_numeric_classification_subset( df_input,
										scale_cols ).to_numpy( ) )
								df_output[ scale_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									scale_cols, True )
								st.success( 'Min-Max Scaler applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔄',
								key='classification_minmax_scaler_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.ROBUST_SCALER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_robust_scaler_cols',
						options )
					scale_cols = st.multiselect( 'Columns', options=options,
						key='classification_robust_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_robust_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							scale_cols = get_valid_classification_columns( scale_cols, df_input )
							if require_classification_columns( scale_cols, 'Robust Scaler' ):
								scaler = RobustScaler( )
								df_output = df_input.copy( )
								result = scaler.train_transform(
									get_numeric_classification_subset( df_input,
										scale_cols ).to_numpy( ) )
								df_output[ scale_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									scale_cols, True )
								st.success( 'Robust Scaler applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_robust_scaler_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Normal Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.NORMAL_SCALER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_normal_scaler_cols',
						options )
					scale_cols = st.multiselect( 'Columns', options=options,
						key='classification_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ], index=1,
						key='classification_normal_scaler_norm' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_normal_scaler_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							scale_cols = get_valid_classification_columns( scale_cols, df_input )
							if require_classification_columns( scale_cols, 'Normal Scaler' ):
								scaler = NormalScaler( norm=norm )
								df_output = df_input.copy( )
								result = scaler.train_transform(
									get_numeric_classification_subset( df_input,
										scale_cols ).to_numpy( ) )
								df_output[ scale_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									scale_cols, True )
								st.success( 'NormalScaler applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_normal_scaler_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.MAXABS_SCALER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_maxabs_scaler_cols',
						options )
					scale_cols = st.multiselect( 'Columns', options=options,
						key='classification_maxabs_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_maxabs_scaler_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							scale_cols = get_valid_classification_columns( scale_cols, df_input )
							if require_classification_columns( scale_cols, 'Max-Absolute Scaler' ):
								scaler = MaxAbsScaler( )
								df_output = df_input.copy( )
								result = scaler.train_transform(
									get_numeric_classification_subset( df_input,
										scale_cols ).to_numpy( ) )
								df_output[ scale_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									scale_cols, True )
								st.success( 'MaxAbsScaler applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_maxabs_scaler_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
			
			with st.expander( label='Data Imputation', icon='🧹', key='classification_imputers' ):
				
				with st.expander( 'Mean Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.MEAN_IMPUTER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_mean_imputer_cols', options )
					impute_cols = st.multiselect( 'Columns', options=options,
						key='classification_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='classification_mean_imputer_indicator' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_mean_imputer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							impute_cols = get_valid_classification_columns( impute_cols, df_input )
							if require_classification_columns( impute_cols, 'Mean Imputer' ):
								imputer = MeanImputer( strategy='mean',
									add_indicator=add_indicator )
								result = imputer.train_transform(
									get_numeric_classification_subset( df_input,
										impute_cols ).to_numpy( ) )
								
								df_output = replace_columns( df_input, impute_cols, result,
									'mean_imputer' )
								commit_classification_processed_frame( df_input, df_output,
									impute_cols, not bool( add_indicator ) )
								st.success( 'MeanImputer applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_mean_imputer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.NEAREST_NEIGHBOR_IMPUTER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_nearest_imputer_cols',
						options )
					impute_cols = st.multiselect( 'Columns', options=options,
						key='classification_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1, value=5, step=1,
						key='classification_nearest_imputer_neighbors' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_nearest_imputer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							impute_cols = get_valid_classification_columns( impute_cols, df_input )
							if require_classification_columns( impute_cols,
									'Nearest Neighbor Imputer' ):
								imputer = NearestImputer( neighbors=int( neighbors ) )
								result = imputer.train_transform(
									get_numeric_classification_subset( df_input,
										impute_cols ).to_numpy( ) )
								
								df_output = replace_columns( df_input, impute_cols, result,
									'nearest_imputer' )
								commit_classification_processed_frame( df_input, df_output,
									impute_cols, True )
								st.success( 'Nearest Imputer applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_nearest_imputer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.ITERATIVE_IMPUTER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_iterative_imputer_cols',
						options )
					impute_cols = st.multiselect( 'Columns', options=options,
						key='classification_iterative_imputer_cols' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=10, step=1,
						key='classification_iterative_imputer_max_iter' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=0, step=1,
						key='classification_iterative_imputer_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Iterative Imputer', icon='✔️',
								key='classification_iterative_imputer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							impute_cols = get_valid_classification_columns( impute_cols, df_input )
							if require_classification_columns( impute_cols, 'Iterative Imputer' ):
								imputer = IterativeImputer( max_iter=int( max_iter ),
									random_state=int( random_state ) )
								result = imputer.train_transform(
									get_numeric_classification_subset( df_input,
										impute_cols ).to_numpy( ) )
								
								df_output = replace_columns( df_input, impute_cols, result,
									'iterative_imputer' )
								commit_classification_processed_frame( df_input, df_output,
									impute_cols, True )
								st.success( 'Iterative Imputer applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_iterative_imputer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.SIMPLE_IMPUTER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_simple_imputer_cols',
						options )
					impute_cols = st.multiselect( 'Columns', options=options,
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
						if st.button( 'Apply SimpleImputer', icon='✔️',
								key='classification_simpleimputer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							impute_cols = get_valid_classification_columns( impute_cols, df_input )
							if require_classification_columns( impute_cols, 'Simple Imputer' ):
								if strategy in [ 'mean', 'median' ]:
									df_values = get_numeric_classification_subset( df_input,
										impute_cols )
									fill_object: object = 0.0
								else:
									df_values = df_input[ impute_cols ].copy( )
									fill_object = fill_value
								
								imputer = SimpleImputer( strategy=strategy, fill_value=fill_object,
									add_indicator=add_indicator,
									keep_empty_features=keep_empty_features )
								
								result = imputer.train_transform( df_values.to_numpy( ) )
								df_output = replace_columns( df_input, impute_cols, result,
									'simple_imputer' )
								commit_classification_processed_frame( df_input, df_output,
									impute_cols, not bool( add_indicator ) )
								st.success( 'Simple Imputer Applied' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_simple_imputer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
			
			with st.expander( label='Data Encoding', icon='🔣', key='classification_encoders' ):
				
				with st.expander( 'One-Hot Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.ONEHOT_ENCODER )
					
					options = get_classification_columns( categorical_only=True )
					prune_classification_multiselect( 'classification_onehot_cols', options )
					encode_cols = st.multiselect( 'Columns', options=options,
						key='classification_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='classification_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0,
						key='classification_onehot_unknown' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_onehot_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							encode_cols = get_valid_classification_columns( encode_cols, df_input )
							if require_classification_columns( encode_cols, 'One-Hot Encoder' ):
								encoder = OneHotEncoder( sparse=bool( sparse ), unknown=unknown )
								result = encoder.train_transform(
									df_input[ encode_cols ].fillna( '' ).astype( str ).to_numpy(
									
									) )
								df_output = replace_columns( df_input, encode_cols, result,
									'onehot' )
								commit_classification_processed_frame( df_input, df_output,
									encode_cols, False )
								st.success( 'One-Hot Encoder applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_onehot_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.ORDINAL_ENCODER )
					
					options = get_classification_columns( categorical_only=True )
					prune_classification_multiselect( 'classification_ordinal_cols', options )
					encode_cols = st.multiselect( 'Columns', options=options,
						key='classification_ordinal_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_ordinal_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							encode_cols = get_valid_classification_columns( encode_cols, df_input )
							if require_classification_columns( encode_cols, 'Ordinal Encoder' ):
								encoder = OrdinalEncoder( )
								df_output = df_input.copy( )
								result = encoder.train_transform(
									df_output[ encode_cols ].fillna( '' ).astype(
										str ).to_numpy( ) )
								df_output[ encode_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									encode_cols, True )
								st.success( 'Ordinal Encoder Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_ordinal_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Label Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_ENCODER )
					
					options = get_classification_columns( )
					prune_classification_selectbox( 'classification_label_encoder_col', options )
					target_col = st.selectbox( 'Column', options=options,
						key='classification_label_encoder_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_label_encoder_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							if target_col and target_col in df_input.columns:
								encoder = LabelEncoder( )
								df_output = df_input.copy( )
								result = encoder.train_transform(
									df_output[ target_col ].fillna( '' ).astype( str ).to_numpy(
									
									) )
								df_output[ target_col ] = result
								commit_classification_processed_frame( df_input, df_output,
									[ target_col ], True )
								st.success( 'Label Encoder Applied.' )
								st.rerun( )
							else:
								st.warning( 'Select an available column for Label Encoder.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_label_encoder_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Target Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.TARGET_ENCODER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_target_encoder_cols',
						options )
					encode_cols = st.multiselect( 'Categorical Feature Columns', options=options,
						key='classification_target_encoder_cols' )
					
					prune_classification_selectbox( 'classification_target_encoder_target_col',
						options )
					target_col = st.selectbox( 'Target Column', options=options,
						key='classification_target_encoder_target_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_target_encoder_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							encode_cols = get_valid_classification_columns( encode_cols, df_input )
							if require_classification_columns( encode_cols, 'Target Encoder' ):
								if target_col and target_col in df_input.columns:
									encoder = TargetEncoder( )
									X_enc = df_input[ encode_cols ].fillna( '' ).astype(
										str ).to_numpy( )
									y_enc = df_input[ target_col ].to_numpy( )
									result = encoder.train_transform( X_enc, y_enc )
									df_output = replace_columns( df_input, encode_cols, result,
										'target_encoder' )
									commit_classification_processed_frame( df_input, df_output,
										encode_cols, False )
									st.success( 'Target Encoder Applied.' )
									st.rerun( )
								else:
									st.warning(
										'Select an available target column for Target Encoder.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_target_encoder_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.POLYNOMIAL_FEATURES )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_polynomial_cols', options )
					poly_cols = st.multiselect( 'Columns', options=options,
						key='classification_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4, value=2,
						key='classification_polynomial_degree' )
					
					interaction = st.checkbox( 'Interaction Only', value=True,
						key='classification_polynomial_interaction' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_polynomial_apply', use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							poly_cols = get_valid_classification_columns( poly_cols, df_input )
							if require_classification_columns( poly_cols, 'Polynomial Features' ):
								encoder = PolynomialFeatures( degree=int( degree ),
									interaction=bool( interaction ) )
								result = encoder.train_transform(
									get_numeric_classification_subset( df_input,
										poly_cols ).to_numpy( ) )
								df_output = replace_columns( df_input, poly_cols, result,
									'polynomial' )
								commit_classification_processed_frame( df_input, df_output,
									poly_cols, False )
								st.success( 'PolynomialFeatures applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_polynomial_reset', use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
		
		with feature_c2:
			with st.expander( label='Data Transformation', icon='⚡',
					key='classification_transformers' ):
				
				with st.expander( 'Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.BINARIZER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_binarizer_cols', options )
					transform_cols = st.multiselect( 'Columns', options=options,
						key='classification_binarizer_cols' )
					
					threshold = st.number_input( 'Threshold', value=0.0, step=0.1,
						key='classification_binarizer_threshold' )
					
					copy = st.checkbox( 'Copy', value=True, key='classification_binarizer_copy' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Binarizer', key='classification_binarizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							transform_cols = get_valid_classification_columns( transform_cols,
								df_input )
							if require_classification_columns( transform_cols, 'Binarizer' ):
								transformer = Binarizer( threshold=float( threshold ),
									copy=bool( copy ) )
								df_output = df_input.copy( )
								result = transformer.train_transform(
									get_numeric_classification_subset( df_input,
										transform_cols ).to_numpy( ) )
								df_output[ transform_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									transform_cols, True )
								st.success( 'Binarizer applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_binarizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_BINARIZER )
					
					options = get_classification_columns( )
					prune_classification_selectbox( 'classification_label_binarizer_col', options )
					target_col = st.selectbox( 'Column', options=options,
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
								key='classification_lblbinarizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							if target_col and target_col in df_input.columns:
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ),
									sparse_output=bool( sparse_output ) )
								result = transformer.train_transform(
									df_input[ target_col ].fillna( '' ).astype( str ).to_numpy( ) )
								df_output = replace_columns( df_input, [ target_col ], result,
									'label_binarizer' )
								preserve_name = np.asarray( result ).ndim == 1
								commit_classification_processed_frame( df_input, df_output,
									[ target_col ], preserve_name )
								st.success( 'Label Binarizer Applied.' )
								st.rerun( )
							else:
								st.warning( 'Select an available column for Label Binarizer.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_lblbinarizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.MULTILABEL_BINARIZER )
					
					options = get_classification_columns( )
					prune_classification_selectbox( 'classification_multilabel_binarizer_col',
						options )
					target_col = st.selectbox( 'Column', options=options,
						key='classification_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='classification_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='classification_multilabel_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_multilabel_binarizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							if target_col and target_col in df_input.columns:
								y_multi = parse_multilabel_series( df_input[ target_col ],
									delimiter=delimiter )
								transformer = MultiLabelBinarizer( classes=None,
									sparse_output=bool( sparse_output ) )
								result = transformer.train_transform( y_multi )
								df_output = replace_columns( df_input, [ target_col ], result,
									'multilabel_binarizer' )
								commit_classification_processed_frame( df_input, df_output,
									[ target_col ], False )
								st.success( 'Multi-Label Binarizer Applied.' )
								st.rerun( )
							else:
								st.warning(
									'Select an available column for Multi-Label Binarizer.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_multilabel_binarizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'TF-IDF Transformer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.TDIDF_TRANSFORMER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_tfidf_transformer_cols',
						options )
					text_count_cols = st.multiselect( 'Count Matrix Columns', options=options,
						key='classification_tfidf_transformer_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', None ], index=1,
						key='classification_tfidf_transformer_norm' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='classification_tfidf_transformer_use_idf' )
					
					smooth_idf = st.checkbox( 'Smooth IDF', value=True,
						key='classification_tfidf_transformer_smooth_idf' )
					
					sublinear_tf = st.checkbox( 'Sublinear TF', value=False,
						key='classification_tfidf_transformer_sublinear' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_tfidf_transformer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							text_count_cols = get_valid_classification_columns( text_count_cols,
								df_input )
							if require_classification_columns( text_count_cols,
									'TF-IDF Transformer' ):
								transformer = TfidfTransformer( norm=norm, use_idf=bool( use_idf ),
									smooth_idf=bool( smooth_idf ),
									sublinear_tf=bool( sublinear_tf ) )
								result = transformer.train_transform(
									get_numeric_classification_subset( df_input,
										text_count_cols ).to_numpy( ) )
								df_output = replace_columns( df_input, text_count_cols, result,
									'tfidf_transformer' )
								commit_classification_processed_frame( df_input, df_output,
									text_count_cols, True )
								st.success( 'TFIDF Transformer Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_tfidf_transformer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Column Transformer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.COLUMN_TRANSFORMER )
					
					all_options = get_classification_columns( )
					df_column_input = get_classification_pipeline_frame( )
					numeric_options = [ col for col in all_options if
						pd.api.types.is_numeric_dtype( df_column_input[ col ] ) ]
					categorical_options = [ col for col in all_options if
						col not in numeric_options ]
					
					prune_classification_multiselect(
						'classification_column_transformer_numeric_columns', numeric_options )
					numeric_columns = st.multiselect( 'Numeric Columns', options=numeric_options,
						key='classification_column_transformer_numeric_columns' )
					
					prune_classification_multiselect(
						'classification_column_transformer_categorical_columns',
						categorical_options )
					categorical_columns = st.multiselect( 'Categorical Columns',
						options=categorical_options,
						key='classification_column_transformer_categorical_columns' )
					
					numeric_transform = st.selectbox( 'Numeric Transformer',
						options=[ 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler',
							'Binarizer', 'None' ],
						key='classification_column_transformer_numeric_transform' )
					
					categorical_transform = st.selectbox( 'Categorical Transformer',
						options=[ 'OneHotEncoder', 'OrdinalEncoder', 'None' ],
						key='classification_column_transformer_categorical_transform' )
					
					remainder = st.selectbox( 'Remainder', options=[ 'drop', 'passthrough' ],
						key='classification_column_transformer_remainder' )
					
					sparse_threshold = st.slider( 'Sparse Threshold', min_value=0.0, max_value=1.0,
						value=0.3, key='classification_column_transformer_sparse_threshold' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Column Transformer',
								key='classification_column_transformer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							numeric_columns = get_valid_classification_columns( numeric_columns,
								df_input )
							categorical_columns = get_valid_classification_columns(
								categorical_columns, df_input )
							
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
								
								transformers.append(
									('categorical', categorical_model, categorical_columns) )
							
							if transformers:
								transformer = ColumnTransformer( transformers=transformers,
									remainder=remainder, sparse_threshold=float(
										sparse_threshold ),
									n_jobs=None, transformer_weights=None, verbose=False )
								result = transformer.train_transform( df_input )
								df_output = normalize_result_frame( result=result,
									index=df_input.index, prefix='column_transformer',
									columns=None )
								commit_classification_processed_frame( df_input, df_output,
									numeric_columns + categorical_columns, False )
								st.success( 'ColumnTransformer applied.' )
								st.rerun( )
							else:
								st.warning( 'Select at least one transformer and matching '
								            'column.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_column_transformer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
			
			with st.expander( label='Feature Extration', icon='⛏️',
					key='classification_extractors' ):
				
				with st.expander( 'TF-IDF Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.TDIDF_VECTORIZER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_tfidf_vectorizer_cols',
						options )
					text_cols = st.multiselect( 'Text Columns', options=options,
						key='classification_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='classification_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='classification_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='classification_tfidf_vectorizer_use_idf' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_tfidf_vectorizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							text_cols = get_valid_classification_columns( text_cols, df_input )
							if require_classification_columns( text_cols, 'TF-IDF Vectorizer' ):
								transformer = TfidfVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), use_idf=bool( use_idf ) )
								df_output = apply_text_vectorizer( df_input, text_cols,
									transformer,
									'tfidf_vectorizer' )
								commit_classification_processed_frame( df_input, df_output,
									text_cols, False )
								st.success( 'TFIDF Vectorizer Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_tfidf_vectorizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.COUNT_VECTORIZER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_count_vectorizer_cols',
						options )
					text_cols = st.multiselect( 'Text Columns', options=options,
						key='classification_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='classification_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='classification_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='classification_count_vectorizer_binary' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_count_vectorizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							text_cols = get_valid_classification_columns( text_cols, df_input )
							if require_classification_columns( text_cols, 'Count Vectorizer' ):
								transformer = CountVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), binary=bool( binary ) )
								df_output = apply_text_vectorizer( df_input, text_cols,
									transformer,
									'count_vectorizer' )
								commit_classification_processed_frame( df_input, df_output,
									text_cols, False )
								st.success( 'Count Vectorizer Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_count_vectorizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.HASH_VECTORIZER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_hash_vectorizer_cols',
						options )
					text_cols = st.multiselect( 'Text Columns', options=options,
						key='classification_hash_vectorizer_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='classification_hash_vectorizer_n_features' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='classification_hash_vectorizer_ngram_max' )
					
					binary = st.checkbox( 'Binary', value=False,
						key='classification_hash_vectorizer_binary' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='classification_hash_vectorizer_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_hash_vectorizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							text_cols = get_valid_classification_columns( text_cols, df_input )
							if require_classification_columns( text_cols, 'Hash Vectorizer' ):
								transformer = HashVectorizer( num=int( n_features ),
									ngram_range=(1, int( ngram_max )), binary=bool( binary ),
									alternate_sign=bool( alternate_sign ) )
								df_output = apply_text_vectorizer( df_input, text_cols,
									transformer,
									'hash_vectorizer' )
								commit_classification_processed_frame( df_input, df_output,
									text_cols, False )
								st.success( 'HashVectorizer Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_hash_vectorizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.DICT_VECTORIZER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_dict_vectorizer_cols',
						options )
					dict_cols = st.multiselect( 'Columns', options=options,
						key='classification_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='classification_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='classification_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='classification_dict_vectorizer_sort' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_dict_vectorizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							dict_cols = get_valid_classification_columns( dict_cols, df_input )
							if require_classification_columns( dict_cols, 'Dictionary '
							                                              'Vectorizer' ):
								transformer = DictVectorizer( dtype=np.float64,
									separator=separator,
									sparse=bool( sparse ), sort=bool( sort ) )
								df_output = apply_dict_transform( df_input, dict_cols, transformer,
									'dict_vectorizer' )
								commit_classification_processed_frame( df_input, df_output,
									dict_cols, False )
								st.success( 'DictVectorizer applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_dict_vectorizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.FEATURE_HASHER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_feature_hasher_cols',
						options )
					hash_cols = st.multiselect( 'Columns', options=options,
						key='classification_feature_hasher_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='classification_feature_hasher_n_features' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='classification_feature_hasher_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_feature_hasher_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							hash_cols = get_valid_classification_columns( hash_cols, df_input )
							if require_classification_columns( hash_cols, 'Feature Hasher' ):
								transformer = FeatureHasher( n_features=int( n_features ),
									input_type='dict', dtype=np.float64,
									alternate_sign=bool( alternate_sign ) )
								df_output = apply_dict_transform( df_input, hash_cols, transformer,
									'feature_hasher' )
								commit_classification_processed_frame( df_input, df_output,
									hash_cols, False )
								st.success( 'FeatureHasher applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_feature_hasher_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
			
			with st.expander( label='Dimensionality Reduction', icon='🎚️',
					key='classification_selectors' ):
				
				with st.expander( 'Variance Threshold', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.VARIANCE_THRESHOLD )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_variance_threshold_cols',
						options )
					select_cols = st.multiselect( 'Columns', options=options,
						key='classification_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0, step=0.01,
						key='classification_variance_threshold_value' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_variance_threshold_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							select_cols = get_valid_classification_columns( select_cols, df_input )
							if require_classification_columns( select_cols, 'Variance Threshold' ):
								selector = VarianceThreshold( thresh=float( threshold ) )
								result = selector.train_transform(
									get_numeric_classification_subset( df_input,
										select_cols ).to_numpy( ) )
								df_output = replace_columns( df_input, select_cols, result,
									'variance_threshold' )
								preserve_name = np.asarray( result ).ndim == 2 and \
								                np.asarray( result ).shape[ 1 ] == len(
									select_cols )
								commit_classification_processed_frame( df_input, df_output,
									select_cols, preserve_name )
								st.success( 'VarianceThreshold applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_variance_threshold_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Canonical Correlation Analysis (CCA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.CCA )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_cca_x_cols', options )
					X_cols = st.multiselect( 'Predictor Columns', options=options,
						key='classification_cca_x_cols' )
					
					prune_classification_multiselect( 'classification_cca_y_cols', options )
					y_cols = st.multiselect( 'Target Columns', options=options,
						key='classification_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='classification_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True, key='classification_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500, step=1,
						key='classification_cca_max_iter' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_cca_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							X_cols = get_valid_classification_columns( X_cols, df_input )
							y_cols = get_valid_classification_columns( y_cols, df_input )
							if require_classification_columns( X_cols,
									'CCA predictors' ) and require_classification_columns( y_cols,
								'CCA targets' ):
								selector = CCA( num=int( n_components ), scale=bool( scale ),
									size=int( max_iter ) )
								result = selector.train_transform(
									get_numeric_classification_subset( df_input,
										X_cols ).to_numpy( ),
									get_numeric_classification_subset( df_input,
										y_cols ).to_numpy( ) )
								df_result = normalize_result_frame( result=result,
									index=df_input.index, prefix='cca', columns=None )
								df_output = pd.concat(
									[ df_input.drop( columns=X_cols + y_cols, errors='ignore' ),
										df_result ], axis=1 )
								commit_classification_processed_frame( df_input, df_output,
									X_cols + y_cols, False )
								st.success( 'CCA Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_cca_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Principle Component Analysis (PCA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.PCA )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_pca_cols', options )
					select_cols = st.multiselect( 'Columns', options=options,
						key='classification_pca_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='classification_pca_components' )
					
					solver = st.selectbox( 'SVD Solver',
						options=[ 'auto', 'full', 'randomized', 'covariance_eigh', 'arpack' ],
						key='classification_pca_solver' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_pca_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							select_cols = get_valid_classification_columns( select_cols, df_input )
							if require_classification_columns( select_cols, 'PCA' ):
								selector = PCA( num=int( n_components ), solver=solver )
								result = selector.train_transform(
									get_numeric_classification_subset( df_input,
										select_cols ).to_numpy( ) )
								df_output = replace_columns( df_input, select_cols, result, 'pca' )
								commit_classification_processed_frame( df_input, df_output,
									select_cols, False )
								st.success( 'PCA applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_pca_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Select-Best', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SELECT_BEST )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_selectbest_x_cols', options )
					X_cols = st.multiselect( 'Feature Columns', options=options,
						key='classification_selectbest_x_cols' )
					
					all_options = get_classification_columns( )
					prune_classification_selectbox( 'classification_selectbest_target_col',
						all_options )
					target_col = st.selectbox( 'Target Column', options=all_options,
						key='classification_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
							'mutual_info_regression' ],
						key='classification_selectbest_score_name' )
					
					k_best = st.number_input( 'K', min_value=1, value=5, step=1,
						key='classification_selectbest_k' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_selectbest_apply', use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							X_cols = get_valid_classification_columns( X_cols, df_input )
							if require_classification_columns( X_cols, 'Select Best' ):
								if target_col and target_col in df_input.columns:
									selector = SelectBest(
										score_func=score_function_from_name( score_name ),
										num=int( min( k_best, len( X_cols ) ) ) )
									result = selector.train_transform(
										get_numeric_classification_subset( df_input,
											X_cols ).to_numpy( ),
										df_input[ target_col ].to_numpy( ) )
									df_output = replace_columns( df_input, X_cols, result,
										'select_best' )
									commit_classification_processed_frame( df_input, df_output,
										X_cols, False )
									st.success( 'Select Best Applied.' )
									st.rerun( )
								else:
									st.warning(
										'Select an available target column for Select Best.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_selectbest_reset', use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Select-Percent', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SELECT_PERCENT )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_selectpercent_x_cols',
						options )
					X_cols = st.multiselect( 'Feature Columns', options=options,
						key='classification_selectpercent_x_cols' )
					
					all_options = get_classification_columns( )
					prune_classification_selectbox( 'classification_selectpercent_target_col',
						all_options )
					target_col = st.selectbox( 'Target Column', options=all_options,
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
								key='classification_selectpercent_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							X_cols = get_valid_classification_columns( X_cols, df_input )
							if require_classification_columns( X_cols, 'Select Percent' ):
								if target_col and target_col in df_input.columns:
									selector = SelectPercent(
										score_func=score_function_from_name( score_name ),
										pct=int( percentile ) )
									result = selector.train_transform(
										get_numeric_classification_subset( df_input,
											X_cols ).to_numpy( ),
										df_input[ target_col ].to_numpy( ) )
									df_output = replace_columns( df_input, X_cols, result,
										'select_percent' )
									commit_classification_processed_frame( df_input, df_output,
										X_cols, False )
									st.success( 'SelectPercent applied.' )
									st.rerun( )
								else:
									st.warning(
										'Select an available target column for Select Percent.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_selectpercent_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Sequential Back Selection (SBS)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SBS )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_sbs_x_cols', options )
					X_cols = st.multiselect( 'Feature Columns', options=options,
						key='classification_sbs_x_cols' )
					
					all_options = get_classification_columns( )
					prune_classification_selectbox( 'classification_sbs_target_col', all_options )
					target_col = st.selectbox( 'Target Column', options=all_options,
						key='classification_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='classification_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='classification_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1, step=1,
						key='classification_sbs_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_sbs_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							X_cols = get_valid_classification_columns( X_cols, df_input )
							if require_classification_columns( X_cols, 'SBS' ):
								if target_col and target_col in df_input.columns:
									selector = SBS( classifier=None,
										k_features=int( min( k_features, len( X_cols ) ) ),
										test_size=float( test_size ),
										random_state=int( random_state ) )
									X_input = get_numeric_classification_subset( df_input,
										X_cols ).to_numpy( )
									y_input = df_input[ target_col ].to_numpy( )
									selector.train( X_input, y_input )
									result = selector.transform( X_input )
									df_output = replace_columns( df_input, X_cols, result, 'sbs' )
									commit_classification_processed_frame( df_input, df_output,
										X_cols, False )
									st.success( 'SBS applied.' )
									st.rerun( )
								else:
									st.warning( 'Select an available target column for SBS.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_sbs_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Recursive Feature Elimination (RFA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.RFE )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_rfe_x_cols', options )
					X_cols = st.multiselect( 'Feature Columns', options=options,
						key='classification_rfe_x_cols' )
					
					all_options = get_classification_columns( )
					prune_classification_selectbox( 'classification_rfe_target_col', all_options )
					target_col = st.selectbox( 'Target Column', options=all_options,
						key='classification_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='classification_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0, step=1,
						key='classification_rfe_verbose' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_rfe_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							X_cols = get_valid_classification_columns( X_cols, df_input )
							if require_classification_columns( X_cols, 'RFE' ):
								if target_col and target_col in df_input.columns:
									selector = RFE(
										k_features=int( min( k_features, len( X_cols ) ) ),
										verbose=int( verbose ) )
									X_input = get_numeric_classification_subset( df_input,
										X_cols ).to_numpy( )
									y_input = df_input[ target_col ].to_numpy( )
									selector.train( X_input, y_input )
									result = selector.transform( X_input )
									df_output = replace_columns( df_input, X_cols, result, 'rfe' )
									commit_classification_processed_frame( df_input, df_output,
										X_cols, False )
									st.success( 'RFE applied.' )
									st.rerun( )
								else:
									st.warning( 'Select an available target column for RFE.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_rfe_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
		
		blue_divider( )
		df_processed = st.session_state.get( 'df_processed', pd.DataFrame( ) )
		if not has_classification_frame( df_processed ):
			st.info( 'No processed classification data has been created.' )
			st.stop( )
		
		st.markdown( '##### Processed Data' )
		st.caption(
			f'Samples: {len( df_processed ):,} | Features: {len( df_processed.columns ):,}' )
		st.data_editor( df_processed, key='classification_processed_data' )
		
		# ------------------------------------------------------------------
		# MODEL TRAINING
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Model Training', help=cfg.CLASSIFICATION_MODELS )
		
		active_features = [ ftr for ftr in st.session_state.get( 'features', [ ] ) if
			ftr in df_processed.columns ]
		
		active_targets = [ tgt for tgt in st.session_state.get( 'targets', [ ] ) if
			tgt in df_processed.columns ]
		
		st.session_state[ 'active_features' ] = active_features
		st.session_state[ 'active_targets' ] = active_targets
		if not active_features:
			warn = '⚠️ Classification training requires at least one processed feature column.'
			st.warning( warn )
			st.stop( )
		
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
				            'Use a label encoder/binarizer for classification targets, '
				            'not a scaler.' )
				
				st.stop( )
		
		class_counts = pd.Series( y ).value_counts( dropna=False )
		if len( class_counts ) < 2:
			st.warning( '⚠️ Classification requires at least two classes.' )
			st.stop( )
		
		df_classification = df_model.copy( )
		st.session_state[ 'df_classification' ] = df_classification.copy( )
		
		# ------------------------------------------------------------------
		# Classification Models
		# ------------------------------------------------------------------
		with st.expander( 'Linear Models', expanded=True ):
			
			with st.expander( 'Perceptron', expanded=True ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.PERCEPTRON_CLASSIFIER )
				
				perceptron_defaults = { 'classification_perceptron_alpha': 0.001000,
					'classification_perceptron_eta': 1.000000,
					'classification_perceptron_iters': 1000,
					'classification_perceptron_shuffle': False,
					'classification_perceptron_penalty': None,
					'classification_perceptron_test_size': 20,
					'classification_perceptron_random_state': 1 }
				
				def reset_perceptron_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Perceptron widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in perceptron_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in perceptron_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				per_c1, per_c2, per_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with per_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					perceptron_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_perceptron_alpha' ] ),
						step=0.000100, format='%.6f', key='classification_perceptron_alpha' )
					
					perceptron_eta = st.number_input( 'Eta', min_value=0.000001,
						value=float( st.session_state[ 'classification_perceptron_eta' ] ),
						step=0.100000, format='%.6f', key='classification_perceptron_eta' )
					
					perceptron_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_perceptron_iters' ] ), step=1,
						key='classification_perceptron_iters' )
				
				with per_c2:
					st.markdown( '###### 🚦 Regularization / Split' )
					perceptron_shuffle = st.checkbox( 'Shuffle',
						value=bool( st.session_state[ 'classification_perceptron_shuffle' ] ),
						key='classification_perceptron_shuffle' )
					
					perceptron_penalty = st.selectbox( 'Penalty',
						options=[ None, 'l2', 'l1', 'elasticnet' ],
						index=[ None, 'l2', 'l1', 'elasticnet' ].index(
							st.session_state[ 'classification_perceptron_penalty' ] ),
						format_func=lambda value: 'None' if value is None else str( value ),
						key='classification_perceptron_penalty' )
					
					perceptron_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30,
						value=int( st.session_state[ 'classification_perceptron_test_size' ] ),
						step=1, key='classification_perceptron_test_size' ) / 100.0
				
				with per_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					perceptron_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_perceptron_random_state' ] ),
						step=1, key='classification_perceptron_random_state' )
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				per_btn_1, per_btn_2 = st.columns( 2 )
				with per_btn_1:
					train_perceptron = st.button( '🚂 Train Perceptron',
						key='classification_perceptron_train', use_container_width=True )
				
				with per_btn_2:
					st.button( '🔁 Reset Perceptron', key='classification_perceptron_reset',
						use_container_width=True, on_click=reset_perceptron_state )
				
				if train_perceptron:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Perceptron requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Perceptron requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Perceptron requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						model = classification_model.Perceptron( alpha=float( perceptron_alpha ),
							eta=float( perceptron_eta ), iters=int( perceptron_iters ),
							shuffle=bool( perceptron_shuffle ), penalty=perceptron_penalty,
							random=int( perceptron_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( perceptron_test_size ),
							random=int( perceptron_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Perceptron training completed.' )
					except Exception as ex:
						st.error( f'Perceptron training failed: {ex}' )
			
			with st.expander( 'Ordinary Least Squares', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.LEAST_SQUARES_CLASSIFIER )
				
				least_squares_defaults = { 'classification_least_squares_alpha': 0.000100,
					'classification_least_squares_eta': 0.010000,
					'classification_least_squares_iters': 1000,
					'classification_least_squares_shuffle': False,
					'classification_least_squares_penalty': 'l2',
					'classification_least_squares_test_size': 20,
					'classification_least_squares_random_state': 42 }
				
				def reset_least_squares_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Ordinary Least Squares widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in least_squares_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in least_squares_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				ls_c1, ls_c2, ls_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ls_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
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
					st.markdown( '###### 🚦 Regularization / Split' )
					least_squares_shuffle = st.checkbox( 'Shuffle',
						value=bool( st.session_state[ 'classification_least_squares_shuffle' ] ),
						key='classification_leastsquares_shuffle' )
					
					least_squares_penalty = st.selectbox( 'Penalty',
						options=[ None, 'l2', 'l1', 'elasticnet' ],
						index=[ None, 'l2', 'l1', 'elasticnet' ].index(
							st.session_state[ 'classification_least_squares_penalty' ] ),
						format_func=lambda value: 'None' if value is None else str( value ),
						key='classification_leastsquares_penalty' )
					
					least_squares_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30, step=1,
						value=int( st.session_state[ 'classification_least_squares_test_size' ] ),
						key='classification_leastsquares_test_size' ) / 100.0
				
				with ls_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					least_squares_random_state = st.number_input( 'Random State', min_value=0,
						value=int(
							st.session_state[ 'classification_least_squares_random_state' ] ),
						step=1, key='classification_leastsquares_random_state' )
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				ls_btn_1, ls_btn_2 = st.columns( 2 )
				with ls_btn_1:
					train_least_squares = st.button( '🚂 Train Least Squares',
						key='classification_leastsquares_train', use_container_width=True )
				
				with ls_btn_2:
					st.button( '🔁 Reset Least Squares', key='classification_leastsquares_reset',
						use_container_width=True, on_click=reset_least_squares_state )
				
				if train_least_squares:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Least Squares requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Least Squares requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Least Squares requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.LeastSquares(
							alpha=float( least_squares_alpha ), eta=float( least_squares_eta ),
							iters=int( least_squares_iters ), shuffle=bool(
								least_squares_shuffle ),
							penalty=least_squares_penalty,
							random=int( least_squares_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( least_squares_test_size ),
							random=int( least_squares_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Least Squares training completed.' )
					
					except Exception as ex:
						st.error( f'Least Squares training failed: {ex}' )
			
			with st.expander( 'Logistic Regression', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.LOGISTIC_REGRESSION )
				
				logistic_defaults = { 'classification_logistic_c': 1.000000,
					'classification_logistic_penalty': 'l2', 'classification_logistic_iters': 1000,
					'classification_logistic_multiclass': 'multinomial',
					'classification_logistic_solver': 'lbfgs',
					'classification_logistic_test_size': 20,
					'classification_logistic_random_state': 42 }
				
				for key, value in logistic_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				log_c1, log_c2, log_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with log_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					logistic_c = st.number_input( 'C', min_value=0.000001,
						value=float( st.session_state[ 'classification_logistic_c' ] ),
						step=0.100000, format='%.6f', key='classification_logistic_c' )
					
					logistic_penalty = st.selectbox( 'Penalty',
						options=[ 'l2', 'l1', 'elasticnet', 'none', None ],
						index=[ 'l2', 'l1', 'elasticnet', 'none', None ].index(
							st.session_state[ 'classification_logistic_penalty' ] ),
						format_func=lambda v: 'None' if v is None else str( v ),
						key='classification_logistic_penalty' )
					
					logistic_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_logistic_iters' ] ), step=1,
						key='classification_logistic_iters' )
				
				with log_c2:
					st.markdown( '###### Strategy / Solver' )
					logistic_multiclass = st.selectbox( 'Multiclass',
						options=[ 'multinomial', 'ovr', 'auto' ],
						index=[ 'multinomial', 'ovr', 'auto' ].index(
							st.session_state[ 'classification_logistic_multiclass' ] ),
						key='classification_logistic_multiclass' )
					
					logistic_solver = st.selectbox( 'Solver',
						options=[ 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag',
							'saga' ],
						index=[ 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag',
							'saga' ].index( st.session_state[ 'classification_logistic_solver' ] ),
						key='classification_logistic_solver' )
					
					logistic_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30,
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
					train_logistic = st.button( '🚂 Train Logistic Regression',
						key='classification_logistic_train', use_container_width=True )
				
				with log_btn_2:
					reset_logistic = st.button( '🔁 Reset Logistic Regression',
						key='classification_logistic_reset', use_container_width=True )
				
				if reset_logistic:
					for key, value in logistic_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
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
						
						df_predictions = pd.DataFrame(
							{ 'Actual': y_test, 'Predicted': y_prediction } )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
					except Exception as ex:
						st.error( f'Logistic Regression training failed: {ex}' )
			
			with st.expander( 'Ridge Classification', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.RIDGE_CLASSIFIER )
				
				ridge_defaults = { 'classification_ridge_alpha': 1.000000,
					'classification_ridge_solver': 'auto', 'classification_ridge_iters': 1000,
					'classification_ridge_test_size': 20, 'classification_ridge_random_state': 42 }
				
				def reset_ridge_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Ridge Classification widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in ridge_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in ridge_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				ridge_c1, ridge_c2, ridge_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ridge_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					ridge_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_ridge_alpha' ] ),
						step=0.100000, format='%.6f', key='classification_ridge_alpha' )
					
					ridge_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_ridge_iters' ] ), step=1,
						key='classification_ridge_iters' )
				
				with ridge_c2:
					st.markdown( '###### Solver / Split' )
					ridge_solver = st.selectbox( 'Solver',
						options=[ 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga',
							'lbfgs' ],
						index=[ 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga',
							'lbfgs' ].index( st.session_state[ 'classification_ridge_solver' ] ),
						key='classification_ridge_solver' )
					
					ridge_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_ridge_test_size' ] ), step=1,
						key='classification_ridge_test_size' ) / 100.0
				
				with ridge_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					ridge_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_ridge_random_state' ] ),
						step=1, key='classification_ridge_random_state' )
					
					st.caption( f'Samples: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				ridge_btn_1, ridge_btn_2 = st.columns( 2 )
				with ridge_btn_1:
					train_ridge = st.button( '🚂 Train Ridge', key='classification_ridge_train',
						use_container_width=True )
				
				with ridge_btn_2:
					st.button( '🔁 Reset Ridge', key='classification_ridge_reset',
						use_container_width=True, on_click=reset_ridge_state )
				
				if train_ridge:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Ridge Classification requires prepared feature and target '
								'arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning(
								'⚠️ Ridge Classification requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ Ridge Classification requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.Ridge( alpha=float( ridge_alpha ),
							solver=str( ridge_solver ), iters=int( ridge_iters ),
							rando=int( ridge_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( ridge_test_size ), random=int( ridge_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Ridge Classification training completed.' )
					
					except Exception as ex:
						st.error( f'Ridge training failed: {ex}' )
			
			with st.expander( 'Lasso Classification', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.LASSO_CLASSIFIER )
				
				lasso_defaults = { 'classification_lasso_alpha': 1.000000,
					'classification_lasso_iters': 500, 'classification_lasso_threshold': 0.500000,
					'classification_lasso_selection': 'random',
					'classification_lasso_test_size': 20, 'classification_lasso_random_state': 42 }
				
				def reset_lasso_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Lasso Classification widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in lasso_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in lasso_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				lasso_c1, lasso_c2, lasso_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with lasso_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					lasso_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_lasso_alpha' ] ),
						step=0.100000, format='%.6f', key='classification_lasso_alpha' )
					
					lasso_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_lasso_iters' ] ), step=1,
						key='classification_lasso_iters' )
					
					lasso_threshold = st.number_input( 'Threshold', min_value=0.000000,
						max_value=1.000000,
						value=float( st.session_state[ 'classification_lasso_threshold' ] ),
						step=0.050000, format='%.6f', key='classification_lasso_threshold' )
				
				with lasso_c2:
					st.markdown( '###### ↔️ Selection / Split' )
					lasso_selection = st.selectbox( 'Selection', options=[ 'cyclic', 'random' ],
						index=[ 'cyclic', 'random' ].index(
							st.session_state[ 'classification_lasso_selection' ] ),
						key='classification_lasso_selection' )
					
					lasso_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_lasso_test_size' ] ), step=1,
						key='classification_lasso_test_size' ) / 100.0
				
				with lasso_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					lasso_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_lasso_random_state' ] ),
						step=1, key='classification_lasso_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				lasso_btn_1, lasso_btn_2 = st.columns( 2 )
				with lasso_btn_1:
					train_lasso = st.button( '🚂 Train Lasso', key='classification_lasso_train',
						use_container_width=True )
				
				with lasso_btn_2:
					st.button( '🔁 Reset Lasso', key='classification_lasso_reset',
						use_container_width=True, on_click=reset_lasso_state )
				
				if train_lasso:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Lasso Classification requires prepared feature and target '
								'arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning(
								'⚠️ Lasso Classification requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ Lasso Classification requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.Lasso( alpha=float( lasso_alpha ),
							iters=int( lasso_iters ), rando=int( lasso_random_state ),
							threshold=float( lasso_threshold ), selection=str( lasso_selection ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( lasso_test_size ), random=int( lasso_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.score( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Lasso Classification training completed.' )
					
					except Exception as ex:
						st.error( f'Lasso training failed: {ex}' )
			
			with st.expander( 'Gradient Descent', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.GRADIENT_DESCENT_CLASSIFIER )
				
				gradient_defaults = { 'classification_gradient_loss': 'hinge',
					'classification_gradient_penalty': 'l2',
					'classification_gradient_alpha': 0.000100,
					'classification_gradient_iters': 1000, 'classification_gradient_shuffle': True,
					'classification_gradient_eta': 0.010000,
					'classification_gradient_learning': 'optimal',
					'classification_gradient_power': 0.500000,
					'classification_gradient_epsilon': 0.100000,
					'classification_gradient_test_size': 20,
					'classification_gradient_random_state': 42 }
				
				def reset_gradient_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Gradient Descent widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in gradient_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in gradient_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				gd_c1, gd_c2, gd_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with gd_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					gradient_loss = st.selectbox( 'Loss',
						options=[ 'hinge', 'log_loss', 'modified_huber', 'squared_hinge',
							'perceptron', 'squared_error', 'huber', 'epsilon_insensitive',
							'squared_epsilon_insensitive' ],
						index=[ 'hinge', 'log_loss', 'modified_huber', 'squared_hinge',
							'perceptron', 'squared_error', 'huber', 'epsilon_insensitive',
							'squared_epsilon_insensitive' ].index(
							st.session_state[ 'classification_gradient_loss' ] ),
						key='classification_gradient_loss' )
					
					gradient_penalty = st.selectbox( 'Penalty',
						options=[ None, 'l2', 'l1', 'elasticnet' ],
						index=[ None, 'l2', 'l1', 'elasticnet' ].index(
							st.session_state[ 'classification_gradient_penalty' ] ),
						format_func=lambda value: 'None' if value is None else str( value ),
						key='classification_gradient_penalty' )
					
					gradient_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_gradient_alpha' ] ),
						step=0.000100, format='%.6f', key='classification_gradient_alpha' )
					
					gradient_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_gradient_iters' ] ), step=1,
						key='classification_gradient_iters' )
				
				with gd_c2:
					st.markdown( '###### Learning / Split' )
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
						step=0.100000, format='%.6f', key='classification_gradient_power' )
					
					gradient_epsilon = st.number_input( 'Epsilon', min_value=0.000000,
						value=float( st.session_state[ 'classification_gradient_epsilon' ] ),
						step=0.010000, format='%.6f', key='classification_gradient_epsilon' )
				
				with gd_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					gradient_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30,
						value=int( st.session_state[ 'classification_gradient_test_size' ] ),
						step=1, key='classification_gradient_test_size' ) / 100.0
					
					gradient_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_gradient_random_state' ] ),
						step=1, key='classification_gradient_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				gd_btn_1, gd_btn_2 = st.columns( 2 )
				with gd_btn_1:
					train_gradient = st.button( '🚂 Train Gradient Descent',
						key='classification_gradient_train', use_container_width=True )
				
				with gd_btn_2:
					st.button( '🔁 Reset Gradient Descent', key='classification_gradient_reset',
						use_container_width=True, on_click=reset_gradient_state )
				
				if train_gradient:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Gradient Descent requires prepared feature and target '
							            'arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning(
								'⚠️ Gradient Descent requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ Gradient Descent requires at least two target classes.' )
							st.stop( )
						
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
						
						df_scores = model.score( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Gradient Descent training completed.' )
					
					except Exception as ex:
						st.error( f'Gradient Descent training failed: {ex}' )
		
		with st.expander( 'Instance Models', expanded=True ):
			
			with st.expander( 'Nearest Neighbor', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.NEAREST_NEIGHBOR_CLASSFIER )
				
				nearest_defaults = { 'classification_nearest_num': 5,
					'classification_nearest_algorithm': 'auto', 'classification_nearest_power': 2,
					'classification_nearest_metric': 'minkowski',
					'classification_nearest_leafs': 30, 'classification_nearest_test_size': 20,
					'classification_nearest_random_state': 42 }
				
				def reset_nearest_state( ) -> None:
					"""
						
							Purpose:
							--------
							Reset Nearest Neighbor widget values and model outputs before
							widget instantiation on the next Streamlit run.
						
							Parameters:
							-----------
							None
						
							Returns:
							--------
							None
						
						"""
					for reset_key, reset_value in nearest_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in nearest_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				nn_c1, nn_c2, nn_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with nn_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					nearest_num = st.number_input( 'Neighbors', min_value=1,
						value=int( st.session_state[ 'classification_nearest_num' ] ), step=1,
						key='classification_nearest_num' )
					
					nearest_power = st.number_input( 'Power', min_value=1,
						value=int( st.session_state[ 'classification_nearest_power' ] ), step=1,
						key='classification_nearest_power' )
					
					nearest_leafs = st.number_input( 'Leaf Size', min_value=1,
						value=int( st.session_state[ 'classification_nearest_leafs' ] ), step=1,
						key='classification_nearest_leafs' )
				
				with nn_c2:
					st.markdown( '###### Distance / Search' )
					nearest_algorithm = st.selectbox( 'Algorithm',
						options=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ],
						index=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ].index(
							st.session_state[ 'classification_nearest_algorithm' ] ),
						key='classification_nearest_algorithm' )
					
					nearest_metric = st.selectbox( 'Metric',
						options=[ 'minkowski', 'euclidean', 'manhattan', 'chebyshev', 'hamming',
							'canberra', 'braycurtis', 'cityblock', 'cosine', 'l1', 'l2',
							'nan_euclidean', 'mahalanobis', 'seuclidean' ],
						index=[ 'minkowski', 'euclidean', 'manhattan', 'chebyshev', 'hamming',
							'canberra', 'braycurtis', 'cityblock', 'cosine', 'l1', 'l2',
							'nan_euclidean', 'mahalanobis', 'seuclidean' ].index(
							st.session_state[ 'classification_nearest_metric' ] ),
						key='classification_nearest_metric' )
					
					nearest_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_nearest_test_size' ] ),
						step=1,
						key='classification_nearest_test_size' ) / 100.0
				
				with nn_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					nearest_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_nearest_random_state' ] ),
						step=1, key='classification_nearest_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				nn_btn_1, nn_btn_2 = st.columns( 2, border=True )
				with nn_btn_1:
					train_nearest = st.button( '🚂 Train Nearest Neighbor',
						key='classification_nearest_train', use_container_width=True )
				
				with nn_btn_2:
					st.button( '🔁 Reset Nearest Neighbor', key='classification_nearest_reset',
						use_container_width=True, on_click=reset_nearest_state )
				
				if train_nearest:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Nearest Neighbor requires prepared feature and target '
							            'arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning(
								'⚠️ Nearest Neighbor requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ Nearest Neighbor requires at least two target classes.' )
							st.stop( )
						
						if int( nearest_num ) > len( X ):
							st.warning(
								'⚠️ Neighbors cannot exceed the number of available samples.' )
							st.stop( )
						
						if str( nearest_metric ) in [ 'mahalanobis', 'seuclidean' ]:
							st.warning( '⚠️ The selected metric requires additional parameters. '
								'Use minkowski, euclidean, manhattan, cityblock, l1, l2, '
								'cosine, chebyshev, hamming, canberra, braycurtis, or '
								'nan_euclidean for this UI path.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						model = classification_model.NearestNeighbor( num=int( nearest_num ),
							algorithm=str( nearest_algorithm ), power=int( nearest_power ),
							metric=str( nearest_metric ), leafs=int( nearest_leafs ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( nearest_test_size ), random=int( nearest_random_state ) )
						
						if int( nearest_num ) > len( X_train ):
							st.warning( '⚠️ Neighbors cannot exceed the number of training '
							            'samples after the train/test split.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Nearest Neighbor training completed.' )
					except Exception as ex:
						st.error( f'Nearest Neighbor training failed: {ex}' )
			
			with st.expander( 'Support Vector Machine', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.SUPPORT_VECTOR_CLASSIFIER )
				
				svm_defaults = { 'classification_svm_c': 1.000000,
					'classification_svm_kernel': 'rbf', 'classification_svm_degree': 3,
					'classification_svm_test_size': 20, 'classification_svm_random_state': 42 }
				
				def reset_svm_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Support Vector Machine widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in svm_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in svm_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				svm_c1, svm_c2, svm_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with svm_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					svm_c = st.number_input( 'C', min_value=0.000001,
						value=float( st.session_state[ 'classification_svm_c' ] ), step=0.100000,
						format='%.6f', key='classification_svm_c' )
					
					svm_kernel = st.selectbox( 'Kernel',
						options=[ 'linear', 'poly', 'rbf', 'sigmoid' ],
						index=[ 'linear', 'poly', 'rbf', 'sigmoid' ].index(
							st.session_state[ 'classification_svm_kernel' ] ) if st.session_state[
								                                                     'classification_svm_kernel' ] in [
								                                                     'linear',
								                                                     'poly', 'rbf',
								                                                     'sigmoid' ]
						else 2,
						key='classification_svm_kernel' )
					
					svm_degree = st.number_input( 'Degree', min_value=1,
						value=int( st.session_state[ 'classification_svm_degree' ] ), step=1,
						key='classification_svm_degree' )
				
				with svm_c2:
					st.markdown( '###### ↔️ Split' )
					svm_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_svm_test_size' ] ), step=1,
						key='classification_svm_test_size' ) / 100.0
					
					if svm_kernel != 'poly':
						st.caption( 'Degree is only used when kernel = poly.' )
				
				with svm_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					svm_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_svm_random_state' ] ), step=1,
						key='classification_svm_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				svm_btn_1, svm_btn_2 = st.columns( 2 )
				with svm_btn_1:
					train_svm = st.button( '🚂 Train Support Vector',
						key='classification_svm_train',
						use_container_width=True )
				
				with svm_btn_2:
					st.button( '🔁 Reset Support Vector', key='classification_svm_reset',
						use_container_width=True, on_click=reset_svm_state )
				
				if train_svm:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Support Vector Machine requires prepared feature and target '
								'arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Support Vector Machine requires at least one numeric '
							            'feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ Support Vector Machine requires at least two target classes.' )
							st.stop( )
						
						if str( svm_kernel ) == 'precomputed':
							st.warning(
								'⚠️ The precomputed kernel requires a square kernel matrix and is '
								'not supported by this Classification UI path.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.SupportVector( C=float( svm_c ),
							kernel=str( svm_kernel ), degree=int( svm_degree ),
							random=int( svm_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( svm_test_size ), random=int( svm_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Support Vector Machine training completed.' )
					
					except Exception as ex:
						st.error( f'Support Vector training failed: {ex}' )
		
		with st.expander( 'Tree Models', expanded=True ):
			
			with st.expander( 'Decision Tree', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.DESICION_TREE_CLASSIFIER )
				
				tree_defaults = { 'classification_tree_criterion': 'gini',
					'classification_tree_splitter': 'best', 'classification_tree_depth': 0,
					'classification_tree_min_split': 2, 'classification_tree_min_leaf': 1,
					'classification_tree_test_size': 20, 'classification_tree_random_state': 42 }
				
				def reset_tree_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Decision Tree widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in tree_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in tree_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				tree_c1, tree_c2, tree_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with tree_c1:
					st.markdown( '###### 🎚️ Hyper-Parameters' )
					tree_criterion = st.selectbox( 'Criterion',
						options=[ 'gini', 'entropy', 'log_loss' ],
						index=[ 'gini', 'entropy', 'log_loss' ].index(
							st.session_state[ 'classification_tree_criterion' ] ),
						key='classification_tree_criterion' )
					
					tree_splitter = st.selectbox( 'Splitter', options=[ 'best', 'random' ],
						index=[ 'best', 'random' ].index(
							st.session_state[ 'classification_tree_splitter' ] ),
						key='classification_tree_splitter' )
					
					tree_depth = st.number_input( 'Max Depth (0 = None)', min_value=0,
						value=int( st.session_state[ 'classification_tree_depth' ] ), step=1,
						key='classification_tree_depth' )
				
				with tree_c2:
					st.markdown( '###### Node Constraints' )
					tree_min_split = st.number_input( 'Min Samples Split', min_value=2,
						value=int( st.session_state[ 'classification_tree_min_split' ] ), step=1,
						key='classification_tree_min_split' )
					
					tree_min_leaf = st.number_input( 'Min Samples Leaf', min_value=1,
						value=int( st.session_state[ 'classification_tree_min_leaf' ] ), step=1,
						key='classification_tree_min_leaf' )
					
					tree_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_tree_test_size' ] ), step=1,
						key='classification_tree_test_size' ) / 100.0
				
				with tree_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					tree_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_tree_random_state' ] ),
						step=1,
						key='classification_tree_random_state' )
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				tree_btn_1, tree_btn_2 = st.columns( 2 )
				with tree_btn_1:
					train_tree = st.button( '🚂 Train Decision Tree',
						key='classification_tree_train', use_container_width=True )
				
				with tree_btn_2:
					st.button( '🔁 Reset Decision Tree', key='classification_tree_reset',
						use_container_width=True, on_click=reset_tree_state )
				
				if train_tree:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Decision Tree requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Decision Tree requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Decision Tree requires at least two target classes.' )
							st.stop( )
						
						if int( tree_min_leaf ) > int( tree_min_split ):
							st.warning( '⚠️ Min Samples Leaf should not exceed Min Samples Split '
							            'for this UI path.' )
							st.stop( )
						
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
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Decision Tree training completed.' )
					
					except Exception as ex:
						st.error( f'Decision Tree training failed: {ex}' )
			
			with st.expander( 'Random Forest', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.RANDOM_FOREST_CLASSIFIER )
				
				forest_defaults = { 'classification_forest_estimators': 100,
					'classification_forest_criterion': 'gini', 'classification_forest_depth': 0,
					'classification_forest_min_split': 2, 'classification_forest_min_leaf': 1,
					'classification_forest_test_size': 20,
					'classification_forest_random_state': 42 }
				
				def reset_forest_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Random Forest widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in forest_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in forest_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				forest_c1, forest_c2, forest_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with forest_c1:
					st.markdown( '###### 🎚️ Hyper-Parameters' )
					forest_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_forest_estimators' ] ),
						step=1,
						key='classification_forest_estimators' )
					
					forest_criterion = st.selectbox( 'Criterion',
						options=[ 'gini', 'entropy', 'log_loss' ],
						index=[ 'gini', 'entropy', 'log_loss' ].index(
							st.session_state[ 'classification_forest_criterion' ] ),
						key='classification_forest_criterion' )
					
					forest_depth = st.number_input( 'Max Depth (0 = None)', min_value=0,
						value=int( st.session_state[ 'classification_forest_depth' ] ), step=1,
						key='classification_forest_depth' )
				
				with forest_c2:
					st.markdown( '###### Node Constraints' )
					forest_min_split = st.number_input( 'Min Samples Split', min_value=2,
						value=int( st.session_state[ 'classification_forest_min_split' ] ), step=1,
						key='classification_forest_min_split' )
					
					forest_min_leaf = st.number_input( 'Min Samples Leaf', min_value=1,
						value=int( st.session_state[ 'classification_forest_min_leaf' ] ), step=1,
						key='classification_forest_min_leaf' )
					
					forest_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_forest_test_size' ] ), step=1,
						key='classification_forest_test_size' ) / 100.0
				
				with forest_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					forest_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_forest_random_state' ] ),
						step=1, key='classification_forest_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				forest_btn_1, forest_btn_2 = st.columns( 2 )
				with forest_btn_1:
					train_forest = st.button( '🚂 Train Random Forest',
						key='classification_forest_train', use_container_width=True )
				
				with forest_btn_2:
					st.button( '🔁 Reset Random Forest', key='classification_forest_reset',
						use_container_width=True, on_click=reset_forest_state )
				
				if train_forest:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Random Forest requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Random Forest requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Random Forest requires at least two target classes.' )
							st.stop( )
						
						if int( forest_min_leaf ) > int( forest_min_split ):
							st.warning( '⚠️ Min Samples Leaf should not exceed Min Samples Split '
							            'for this UI path.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.RandomForest(
							n_estimators=int( forest_estimators ),
							criterion=str( forest_criterion ),
							depth=None if int( forest_depth ) == 0 else int( forest_depth ),
							min_split=int( forest_min_split ), min_leaf=int( forest_min_leaf ),
							random=int( forest_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( forest_test_size ), random=int( forest_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Random Forest training completed.' )
					
					except Exception as ex:
						st.error( f'Random Forest training failed: {ex}' )
		
		with st.expander( 'Ensemble Models', expanded=False ):
			
			with st.expander( 'Gradient Boost', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.GRADIENT_BOOST_CLASSIFIER )
				
				gb_defaults = { 'classification_gb_estimators': 100,
					'classification_gb_rate': 0.100000, 'classification_gb_depth': 3,
					'classification_gb_criterion': 'friedman_mse',
					'classification_gb_test_size': 20, 'classification_gb_random_state': 42 }
				
				def reset_gb_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Gradient Boost widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in gb_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in gb_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				gb_c1, gb_c2, gb_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with gb_c1:
					st.markdown( '###### 🎚️ Hyper-Parameters' )
					gb_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_gb_estimators' ] ), step=1,
						key='classification_gb_estimators' )
					
					gb_rate = st.number_input( 'Learning Rate', min_value=0.000001,
						value=float( st.session_state[ 'classification_gb_rate' ] ), step=0.010000,
						format='%.6f', key='classification_gb_rate' )
					
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
					gb_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_gb_random_state' ] ), step=1,
						key='classification_gb_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				gb_btn_1, gb_btn_2 = st.columns( 2 )
				with gb_btn_1:
					train_gb = st.button( '🚂 Train Gradient Boost', key='classification_gb_train',
						use_container_width=True )
				
				with gb_btn_2:
					st.button( '🔁 Reset Gradient Boost', key='classification_gb_reset',
						use_container_width=True, on_click=reset_gb_state )
				
				if train_gb:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Gradient Boost requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Gradient Boost requires at least one numeric '
							            'feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Gradient Boost requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						model = classification_model.GradientBoost( estimators=int(
							gb_estimators ),
							rate=float( gb_rate ), depth=int( gb_depth ),
							criterion=str( gb_criterion ), random=int( gb_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( gb_test_size ), random=int( gb_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Gradient Boost training completed.' )
					except Exception as ex:
						st.error( f'Gradient Boost training failed: {ex}' )
			
			with st.expander( 'Adaptive Boost', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.ADAPTIVE_BOOST_CLASSIFIER )
				
				ab_defaults = { 'classification_ab_estimators': 50,
					'classification_ab_rate': 1.000000, 'classification_ab_algorithm': 'SAMME',
					'classification_ab_test_size': 20, 'classification_ab_random_state': 42 }
				
				def reset_ab_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Adaptive Boost widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in ab_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in ab_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				ab_c1, ab_c2, ab_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ab_c1:
					st.markdown( '###### 🎚️ Hyper-Parameters' )
					ab_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_ab_estimators' ] ), step=1,
						key='classification_ab_estimators' )
					
					ab_rate = st.number_input( 'Learning Rate', min_value=0.000001,
						value=float( st.session_state[ 'classification_ab_rate' ] ), step=0.010000,
						format='%.6f', key='classification_ab_rate' )
				
				with ab_c2:
					st.markdown( '###### ♟️ Algorithm / Split' )
					ab_algorithm = st.selectbox( 'Algorithm', options=[ 'SAMME' ], index=0,
						key='classification_ab_algorithm' )
					
					ab_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_ab_test_size' ] ), step=1,
						key='classification_ab_test_size' ) / 100.0
				
				with ab_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					ab_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_ab_random_state' ] ), step=1,
						key='classification_ab_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				ab_btn_1, ab_btn_2 = st.columns( 2 )
				with ab_btn_1:
					train_ab = st.button( '🚂 Train Adaptive Boost', key='classification_ab_train',
						use_container_width=True )
				
				with ab_btn_2:
					st.button( '🔁 Reset Adaptive Boost', key='classification_ab_reset',
						use_container_width=True, on_click=reset_ab_state )
				
				if train_ab:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Adaptive Boost requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Adaptive Boost requires at least one numeric '
							            'feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Adaptive Boost requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.AdaptiveBoost( base=None,
							estimators=int( ab_estimators ), rate=float( ab_rate ),
							algorithm=str( ab_algorithm ), random=int( ab_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( ab_test_size ), random=int( ab_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Adaptive Boost training completed.' )
					
					except Exception as ex:
						st.error( f'Adaptive Boost training failed: {ex}' )
			
			with st.expander( 'Bagging Model', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.BAGGING_CLASSIFIER )
				
				bag_defaults = { 'classification_bag_estimators': 50,
					'classification_bag_test_size': 20, 'classification_bag_random_state': 42 }
				
				def reset_bag_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Bagging Model widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in bag_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in bag_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				bag_c1, bag_c2, bag_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with bag_c1:
					st.markdown( '###### 🎚️ Hyper-Parameters' )
					bag_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_bag_estimators' ] ), step=1,
						key='classification_bag_estimators' )
				
				with bag_c2:
					st.markdown( '###### ↔️ Split' )
					bag_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_bag_test_size' ] ), step=1,
						key='classification_bag_test_size' ) / 100.0
				
				with bag_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					bag_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_bag_random_state' ] ), step=1,
						key='classification_bag_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				bag_btn_1, bag_btn_2 = st.columns( 2 )
				with bag_btn_1:
					train_bag = st.button( '🚂 Train Bagging Model',
						key='classification_bag_train',
						use_container_width=True )
				
				with bag_btn_2:
					st.button( '🔁 Reset Bagging Model', key='classification_bag_reset',
						use_container_width=True, on_click=reset_bag_state )
				
				if train_bag:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Bagging Model requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Bagging Model requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Bagging Model requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						model = classification_model.BaggingModel(
							estimators=int( bag_estimators ), random=int( bag_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( bag_test_size ), random=int( bag_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Bagging Model training completed.' )
					except Exception as ex:
						st.error( f'Bagging Model training failed: {ex}' )
			
			with st.expander( 'Voting Model', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.VOTING_CLASSFIER )
				
				vote_defaults = { 'classification_vote_mode': 'hard',
					'classification_vote_include_logistic': True,
					'classification_vote_include_tree': True,
					'classification_vote_include_knn': True,
					'classification_vote_include_forest': False,
					'classification_vote_include_nb': False, 'classification_vote_test_size': 20,
					'classification_vote_random_state': 42 }
				
				def reset_vote_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Voting Model widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in vote_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in vote_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				vote_c1, vote_c2, vote_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with vote_c1:
					st.markdown( '###### ♟️ Voting Strategy' )
					vote_mode = st.selectbox( 'Vote', options=[ 'hard', 'soft' ],
						index=[ 'hard', 'soft' ].index(
							st.session_state[ 'classification_vote_mode' ] ),
						key='classification_vote_mode' )
					
					st.caption( 'Select at least two base estimators.' )
				
				with vote_c2:
					st.markdown( '###### 📐 Base Estimators' )
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
					vote_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_vote_test_size' ] ), step=1,
						key='classification_vote_test_size' ) / 100.0
					
					vote_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_vote_random_state' ] ),
						step=1,
						key='classification_vote_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				vote_btn_1, vote_btn_2 = st.columns( 2 )
				with vote_btn_1:
					train_vote = st.button( '🚂 Train Voting Model',
						key='classification_vote_train',
						use_container_width=True )
				
				with vote_btn_2:
					st.button( '🔁 Reset Voting Model', key='classification_vote_reset',
						use_container_width=True, on_click=reset_vote_state )
				
				if train_vote:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Voting Model requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Voting Model requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Voting Model requires at least two target classes.' )
							st.stop( )
						
						estimators = [ ]
						
						if vote_include_logistic:
							estimators.append( ('logistic', LogisticRegression( max_iter=1000,
								random_state=int( vote_random_state ) )) )
						
						if vote_include_tree:
							estimators.append( ('tree',
								DecisionTreeClassifier( random_state=int( vote_random_state ) )) )
						
						if vote_include_knn:
							estimators.append( ('knn', KNeighborsClassifier( )) )
						
						if vote_include_forest:
							estimators.append( ('forest',
								RandomForestClassifier( random_state=int( vote_random_state ) )) )
						
						if vote_include_nb:
							estimators.append( ('naive_bayes', GaussianNB( )) )
						
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
						df_scores = model.score( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Estimator Count',
							int( len( estimators ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Vote Mode', str( vote_mode ) )
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Voting Model training completed.' )
					except Exception as ex:
						st.error( f'Voting Model training failed: {ex}' )
			
			with st.expander( 'Stacking Model', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.STACKING_CLASSIFIER )
				
				stack_defaults = { 'classification_stack_include_logistic': True,
					'classification_stack_include_tree': True,
					'classification_stack_include_knn': True,
					'classification_stack_include_forest': False,
					'classification_stack_include_nb': False,
					'classification_stack_final': 'logistic', 'classification_stack_test_size': 20,
					'classification_stack_random_state': 42 }
				
				def reset_stack_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Stacking Model widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in stack_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in stack_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				stack_c1, stack_c2, stack_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with stack_c1:
					st.markdown( '###### 🛑 Final Estimator' )
					stack_final = st.selectbox( 'Final Estimator', options=[ 'logistic', 'tree' ],
						index=[ 'logistic', 'tree' ].index(
							st.session_state[ 'classification_stack_final' ] ),
						key='classification_stack_final' )
					
					st.caption( 'Select at least two base estimators.' )
				
				with stack_c2:
					st.markdown( '###### 📐 Base Estimators' )
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
					
					stack_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_stack_random_state' ] ),
						step=1, key='classification_stack_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				stack_btn_1, stack_btn_2 = st.columns( 2 )
				with stack_btn_1:
					train_stack = st.button( '🚂 Train Stacking Model',
						key='classification_stack_train', use_container_width=True )
				
				with stack_btn_2:
					st.button( '🔁 Reset Stacking Model', key='classification_stack_reset',
						use_container_width=True, on_click=reset_stack_state )
				
				if train_stack:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Stacking Model requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Stacking Model requires at least one numeric '
							            'feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Stacking Model requires at least two target classes.' )
							st.stop( )
						
						estimators = [ ]
						if stack_include_logistic:
							estimators.append( ('logistic', LogisticRegression( max_iter=1000,
								random_state=int( stack_random_state ) )) )
						
						if stack_include_tree:
							estimators.append( ('tree',
								DecisionTreeClassifier( random_state=int( stack_random_state ) )) )
						
						if stack_include_knn:
							estimators.append( ('knn', KNeighborsClassifier( )) )
						
						if stack_include_forest:
							estimators.append( ('forest',
								RandomForestClassifier( random_state=int( stack_random_state ) )) )
						
						if stack_include_nb:
							estimators.append( ('naive_bayes', GaussianNB( )) )
						
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
						df_scores = model.score( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Estimator Count',
							int( len( estimators ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Final Estimator',
							str( stack_final ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Stacking Model training completed.' )
					except Exception as ex:
						st.error( f'Stacking Model training failed: {ex}' )
		
		with st.expander( 'Neural Models', expanded=True ):
			
			with st.expander( 'Multi-Layer Perceptron', expanded=False ):
				mlp_defaults = { 'classification_mlp_hidden_1': 100,
					'classification_mlp_hidden_2': 0, 'classification_mlp_activation': 'logistic',
					'classification_mlp_solver': 'lbfgs', 'classification_mlp_alpha': 0.000100,
					'classification_mlp_learning': 'constant', 'classification_mlp_test_size': 20,
					'classification_mlp_random_state': 42 }
				
				def reset_mlp_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Multi-Layer Perceptron widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in mlp_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'y_predictions' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in mlp_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				mlp_c1, mlp_c2, mlp_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with mlp_c1:
					st.markdown( '###### 🛡️ Network Structure' )
					mlp_hidden_1 = st.number_input( 'Hidden Layer 1', min_value=1,
						value=int( st.session_state[ 'classification_mlp_hidden_1' ] ), step=1,
						key='classification_mlp_hidden_1' )
					
					mlp_hidden_2 = st.number_input( 'Hidden Layer 2 (0 = none)', min_value=0,
						value=int( st.session_state[ 'classification_mlp_hidden_2' ] ), step=1,
						key='classification_mlp_hidden_2' )
					
					mlp_activation = st.selectbox( 'Activation',
						options=[ 'identity', 'logistic', 'tanh', 'relu' ],
						index=[ 'identity', 'logistic', 'tanh', 'relu' ].index(
							st.session_state[ 'classification_mlp_activation' ] ),
						key='classification_mlp_activation' )
				
				with mlp_c2:
					st.markdown( '###### 🎚️ Optimization' )
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
							st.session_state[ 'classification_mlp_learning' ] ),
						key='classification_mlp_learning' )
					
					mlp_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_mlp_test_size' ] ), step=1,
						key='classification_mlp_test_size' ) / 100.0
				
				with mlp_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					mlp_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_mlp_random_state' ] ), step=1,
						key='classification_mlp_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				mlp_btn_1, mlp_btn_2 = st.columns( 2 )
				with mlp_btn_1:
					train_mlp = st.button( '🚂 Train Multi-Layer Perceptron',
						key='classification_mlp_train', use_container_width=True )
				
				with mlp_btn_2:
					st.button( '🔁 Reset Multi-Layer Perceptron', key='classification_mlp_reset',
						use_container_width=True, on_click=reset_mlp_state )
				
				if train_mlp:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Multi-Layer Perceptron requires prepared feature and target '
								'arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Multi-Layer Perceptron requires at least one numeric '
							            'feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ Multi-Layer Perceptron requires at least two target classes.' )
							st.stop( )
						
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
						df_scores = model.score( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Hidden Layers',
							str( hidden_layers ) )
						
						df_scores.insert( len( df_scores.columns ), 'Activation',
							str( mlp_activation ) )
						
						df_scores.insert( len( df_scores.columns ), 'Solver', str( mlp_solver ) )
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Multi-Layer Perceptron training completed.' )
					except Exception as ex:
						st.error( f'Multi-Layer Perceptron training failed: {ex}' )
		
		if st.session_state.get( 'model', None ) is None:
			st.stop( )
		
		# ------------------------------------------------------------------
		# Performance Metrics & Visualizations
		# ------------------------------------------------------------------
		target_count = int( st.session_state.get( 'target_count', 0 ) )
		model = st.session_state.get( 'model', None )
		X_test = st.session_state.get( 'X_test', None )
		y_test = st.session_state.get( 'y_test', None )
		y_prediction = st.session_state.get( 'y_prediction', None )
		if y_prediction is None:
			y_prediction = st.session_state.get( 'y_prediction', None )
		
		df_scores = st.session_state.get( 'df_scores', pd.DataFrame( ) )
		df_predictions = st.session_state.get( 'df_predictions', pd.DataFrame( ) )
		elapsed_seconds = float( st.session_state.get( 'elapsed_seconds', 0.0 ) )
		has_metric_frame = (isinstance( df_scores,
			pd.DataFrame ) and not df_scores.empty and 'Accuracy Score' in df_scores.columns and
		                    'Mis-Classifications' in df_scores.columns)
		
		has_prediction_frame = (isinstance( df_predictions,
			pd.DataFrame ) and not df_predictions.empty and 'Actual' in df_predictions.columns and
		                        'Predicted' in df_predictions.columns)
		
		has_visual_context = (
				model is not None and X_test is not None and y_test is not None and y_prediction
				is not None)
		
		blue_divider( )
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
		
		blue_divider( )
		st.markdown( '##### Predictions' )
		
		if has_prediction_frame:
			st.data_editor( df_predictions, use_container_width=True,
				key='classification_performance_predictions' )
		else:
			st.info( 'No predictions are available for the current classification result.' )
		
		blue_divider( )
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
		# Actual vs Predicted Class Counts
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Actual vs Predicted Counts' )
		
		if has_visual_context:
			try:
				actual_counts = pd.Series( y_test ).value_counts( ).sort_index( )
				pred_counts = pd.Series( y_prediction ).value_counts( ).sort_index( )
				df_counts = pd.DataFrame(
					{ 'Actual': actual_counts, 'Predicted': pred_counts } ).fillna( 0 )
				
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
		# Per-Class Accuracy
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Per-Class Accuracy', help=cfg.PERCLASS_ACCURACY )
		
		if has_visual_context:
			try:
				df_evaluation = pd.DataFrame( { 'Actual': y_test, 'Predicted': y_prediction } )
				df_evaluation[ 'Correct' ] = (
						df_evaluation[ 'Actual' ] == df_evaluation[ 'Predicted' ]).astype( int )
				
				df_class_acc = df_evaluation.groupby( 'Actual', dropna=False )[
					'Correct' ].mean( ).sort_index( )
				
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
		# Prediction Confidence
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Prediction Confidence', help=cfg.PREDICTION_CONFIDENCE )
		
		if has_visual_context and hasattr( model, 'predict_probability' ):
			try:
				proba = model.predict_probability( X_test )
				if (isinstance( proba, np.ndarray ) and proba.ndim == 2 and proba.shape[ 1 ] > 1):
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
				else:
					st.info( 'Prediction confidence is unavailable for this model output.' )
			except Exception as e:
				st.info( f'Prediction Confidence skipped: {e}' )
				plt.close( 'all' )
		else:
			st.info( 'Prediction Confidence is unavailable until a model is trained.' )
		
		# ------------------------------------------------------------------
		# Observed vs Predicted
		# ------------------------------------------------------------------
		blue_divider( )
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
		# ROC Curve
		# ------------------------------------------------------------------
		blue_divider( )
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
	df_dataset = st.session_state.get( 'df_dataset', None )
	df_original = st.session_state.get( 'df_original', None )
	df_working = st.session_state.get( 'df_working', pd.DataFrame( ) )
	df_processed = st.session_state.get( 'df_processed', pd.DataFrame( ) )
	df_regression = st.session_state.get( 'df_regression', pd.DataFrame( ) )
	df_model = st.session_state.get( 'df_model', pd.DataFrame( ) )
	df_scores = st.session_state.get( 'df_scores', pd.DataFrame( ) )
	df_predictions = st.session_state.get( 'df_predictions', pd.DataFrame( ) )
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
		st.caption( 'Predictive Models for Continuous Values' )
		st.divider( )
		
		if not has_loaded_dataset( df_dataset ):
			st.warning( '⚠️ No dataset loaded.' )
			st.stop( )
		
		df_original = df_dataset.copy( )
		st.session_state[ 'df_original' ] = df_original.copy( )
		
		numeric_columns = [ column for column in df_original.columns if
			pd.api.types.is_numeric_dtype( df_original[ column ] ) ]
		
		categorical_columns = [ column for column in df_original.columns if
			column not in numeric_columns ]
		
		st.session_state[ 'numeric_columns' ] = numeric_columns.copy( )
		st.session_state[ 'categorical_columns' ] = categorical_columns.copy( )
		if len( df_original.columns ) < 2:
			st.warning(
				'⚠️ Regression requires at least one feature column and one target column.' )
			st.stop( )
		
		if not numeric_columns:
			st.warning( '⚠️ Regression requires at least one numeric target column.' )
			st.stop( )
		
		# ======================================================================================
		# Data Selection
		# ======================================================================================
		st.markdown( '##### Feature Selection' )
		st.caption( f'Samples: {len( df_original ):,} | '
		            f'Features: {len( df_original.columns ):,}' )
		
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			features = st.multiselect( 'Select Features', options=df_original.columns.tolist( ),
				key='regression_features' )
		
		with col_c2:
			target_options = [ column for column in numeric_columns if column not in features ]
			if target_options:
				target = st.selectbox( 'Select Target', options=target_options,
					key='regression_target' )
			else:
				target = None
				st.info(
					'At least one numeric column must remain available as the regression target.' )
		
		sel_b1, sel_b2 = st.columns( [ 0.5, 0.5 ] )
		with sel_b1:
			if st.button( 'Create Working Dataset', icon='➕', key='regression_create_dataset',
					use_container_width=True ):
				if not features:
					st.warning( '⚠️ Select at least one feature column.' )
					st.stop( )
				
				if target is None:
					st.warning( '⚠️ Select one numeric target column.' )
					st.stop( )
				
				if target in features:
					st.warning( '⚠️ The regression target cannot also be selected as a feature.' )
					st.stop( )
				
				selected_all = features.copy( )
				selected_all.append( target )
				df_working = df_original[ selected_all ].copy( )
				if df_working.empty:
					st.warning(
						'⚠️ The selected feature and target columns contain no observations.' )
					st.stop( )
				
				st.session_state[ 'features' ] = features.copy( )
				st.session_state[ 'targets' ] = [ target ]
				st.session_state[ 'selected_all' ] = selected_all.copy( )
				st.session_state[ 'df_working' ] = df_working.copy( )
				st.session_state[ 'df_processed' ] = pd.DataFrame( )
				st.session_state[ 'df_model' ] = pd.DataFrame( )
				st.session_state[ 'df_scores' ] = pd.DataFrame( )
				st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				st.session_state[ 'df_regression' ] = pd.DataFrame( )
				st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
				st.session_state[ 'active_features' ] = [ ]
				st.session_state[ 'active_targets' ] = [ ]
				st.session_state[ 'X_data' ] = None
				st.session_state[ 'X_train' ] = None
				st.session_state[ 'X_test' ] = None
				st.session_state[ 'y_train' ] = None
				st.session_state[ 'y_test' ] = None
				st.session_state[ 'y_series' ] = None
				st.session_state[ 'y_prediction' ] = None
				st.session_state[ 'model' ] = None
				st.session_state[ 'elapsed_seconds' ] = 0.0
				st.session_state[ 'target_count' ] = 1.0
				commit_frame( df_working )
				st.success( 'Working Dataset Created!' )
				st.rerun( )
		
		with sel_b2:
			if st.button( 'Reset Working Dataset', icon='🔁', key='regression_reset_to_original',
					use_container_width=True ):
				reset_regression_mode_state( )
				st.success( 'Regression working data reset.' )
				st.rerun( )
		
		blue_divider( )
		df_working = st.session_state.get( 'df_working', pd.DataFrame( ) )
		if not has_loaded_dataset( df_working ):
			i = ('Select regression features and one numeric target, then create the working '
			     'dataset.')
			st.info( i )
			st.stop( )
		
		features = [ column for column in st.session_state.get( 'features', [ ] ) if
			column in df_working.columns ]
		
		targets = [ column for column in st.session_state.get( 'targets', [ ] ) if
			column in df_working.columns ]
		
		if not features:
			st.warning( '⚠️ The working dataset does not contain any selected feature columns.' )
			st.stop( )
		
		if len( targets ) != 1:
			st.warning( '⚠️ Regression requires exactly one numeric target column.' )
			st.stop( )
		
		target_name = targets[ 0 ]
		if not pd.api.types.is_numeric_dtype( df_working[ target_name ] ):
			st.warning( '⚠️ The selected regression target must be numeric.' )
			st.stop( )
		
		st.markdown( '##### Working Data' )
		st.caption( f'Samples: {len( df_working ):,} | '
		            f'Features: {len( features ):,} | Target: {target_name}' )
		
		st.data_editor( df_working, key='regression_working_data', use_container_width=True )
		
		# -----------------------------------------------------------------
		# Data Processing
		# -----------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Feature-Engineering' )
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			with st.expander( label='Data Scaling', icon='⚖️', key='regression_scalers' ):
				with st.expander( 'Standard Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.STANDARD_SCALER )
					
					columns = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_standard_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', use_container_width=True,
								key='regression_standard_scaler_apply' ):
							if columns:
								scaler = StandardScaler( )
								df_processed = df_working.copy( )
								result = scaler.train_transform(
									df_processed[ columns ].to_numpy( ) )
								df_processed[ columns ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Standard Scaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_standard_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Min-Max Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.MINMAX_SCALER )
					
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_minmax_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_minmax_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MinMaxScaler( )
								df_processed = df_working.copy( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Min-Max Scaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔄',
								key='regression_minmax_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = None
							df_processed = None
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.ROBUST_SCALER )
					
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_robust_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_robust_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = RobustScaler( )
								df_processed = df_working.copy( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ scale_cols ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Robust Scaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_robust_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = None
							df_processed = None
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Normal Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.NORMAL_SCALER )
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ], index=1,
						key='regression_normal_scaler_norm' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_normal_scaler_apply', use_container_width=True ):
							if scale_cols:
								scaler = NormalScaler( norm=norm )
								df_processed = df_working.copy( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ columns ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'NormalScaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_normal_scaler_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.MAXABS_SCALER )
					
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_maxabs_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_maxabs_scaler_apply', use_container_width=True ):
							if scale_cols:
								scaler = MaxAbsScaler( )
								df_processed = df_working.copy( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ columns ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'MaxAbsScaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_maxabs_scaler_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
			
			with st.expander( label='Data Imputation', icon='🧹', key='regression_imputers' ):
				with st.expander( 'Mean Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.MEAN_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='regression_mean_imputer_indicator' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_mean_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = MeanImputer( strategy='mean',
									add_indicator=add_indicator )
								
								df_processed = df_working.copy( )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, impute_cols, result,
									'mean_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'MeanImputer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_mean_imputer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.NEAREST_NEIGHBOR_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1, value=5, step=1,
						key='regression_nearest_imputer_neighbors' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_nearest_imputer_apply', use_container_width=True ):
							if impute_cols:
								imputer = NearestImputer( neighbors=int( neighbors ) )
								df_processed = df_working.copy( )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, impute_cols, result,
									'nearest_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Nearest Imputer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_nearest_imputer_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.ITERATIVE_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_iterative_imputer_cols' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=10, step=1,
						key='regression_iterative_imputer_max_iter' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=0, step=1,
						key='regression_iterative_imputer_random_state' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Iterative Imputer', icon='✔️',
								key='regression_iterative_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = IterativeImputer( max_iter=int( max_iter ),
									random_state=int( random_state ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, impute_cols, result,
									'iterative_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Iterative Imputer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_iterative_imputer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.SIMPLE_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
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
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SimpleImputer', icon='✔️',
								key='regression_simpleimputer_apply', use_container_width=True ):
							if impute_cols:
								if strategy in [ 'mean', 'median' ]:
									df_input = df_processed[ impute_cols ].apply( pd.to_numeric,
										errors='coerce' )
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
								df_processed = replace_columns( df_processed, impute_cols, result,
									'simple_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Simple Imputer Applied' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_simple_imputer_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
			
			with st.expander( label='Data Encoding', icon='🔣', key='regression_encoders' ):
				with st.expander( 'One-Hot Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.ONEHOT_ENCODER )
					
					encode_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='regression_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0, key='regression_onehot_unknown' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_onehot_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OneHotEncoder( sparse=bool( sparse ), unknown=unknown )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, encode_cols, result,
									'onehot' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'One-Hot Encoder applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_onehot_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.ORDINAL_ENCODER )
					
					encode_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_ordinal_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_ordinal_apply',
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
						if st.button( label='Reset', icon='🔁', key='regression_ordinal_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Label Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_ENCODER )
					
					target_col = st.selectbox( 'Column', options=df_working.columns,
						key='regression_label_encoder_col' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_label_encoder_apply', use_container_width=True ):
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
						if st.button( label='Reset', icon='🔁',
								key='regression_label_encoder_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
					
					st.session_state[ 'df_processed' ] = df_processed
				
				with st.expander( 'Target Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.TARGET_ENCODER )
					
					encode_cols = st.multiselect( 'Categorical Feature Columns',
						options=df_working.columns, key='regression_target_encoder_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='regression_target_encoder_target_col' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_target_encoder_apply', use_container_width=True ):
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
						if st.button( label='Reset', icon='🔁',
								key='regression_target_encoder_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.POLYNOMIAL_FEATURES )
					
					poly_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4, value=2,
						key='regression_polynomial_degree' )
					
					interaction = st.checkbox( 'Interaction Only', value=True,
						key='regression_polynomial_interaction' )
					
					# Apply Button
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
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'PolynomialFeatures applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_polynomial_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
		
		with feature_c2:
			with st.expander( label='Data Transformation', icon='⚡',
					key='regression_transformers' ):
				with st.expander( 'Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.BINARIZER )
					
					transform_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_binarizer_cols' )
					
					threshold = st.number_input( 'Threshold', value=0.0, step=0.1,
						key='regression_binarizer_threshold' )
					
					copy = st.checkbox( 'Copy', value=True, key='regression_binarizer_copy' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Binarizer', key='regression_binarizer_apply',
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
						if st.button( label='Reset', icon='🔁', key='regression_binarizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_BINARIZER )
					
					target_col = st.selectbox( 'Column', options=df_working.columns,
						key='regression_label_binarizer_col' )
					
					pos_label = st.number_input( 'Positive Label', value=1, step=1,
						key='regression_label_binarizer_pos' )
					
					neg_label = st.number_input( 'Negative Label', value=0, step=1,
						key='regression_label_binarizer_neg' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='regression_label_binarizer_sparse' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply LabelBinarizer', key='regression_lblbinarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ),
									sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, [ target_col ],
									result, 'label_binarizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Label Binarizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_lblbinarizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.MULTILABEL_BINARIZER )
					
					target_col = st.selectbox( 'Column', options=df_working.columns,
						key='regression_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='regression_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='regression_multilabel_binarizer_sparse' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_multilabel_binarizer_apply',
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
						if st.button( label='Reset', icon='🔁',
								key='regression_multilabel_binarizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'TF-IDF Transformer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.TDIDF_TRANSFORMER )
					
					text_count_cols = st.multiselect( 'Count Matrix Columns',
						options=df_working.columns, key='regression_tfidf_transformer_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', None ], index=1,
						key='regression_tfidf_transformer_norm' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='regression_tfidf_transformer_use_idf' )
					
					smooth_idf = st.checkbox( 'Smooth IDF', value=True,
						key='regression_tfidf_transformer_smooth_idf' )
					
					sublinear_tf = st.checkbox( 'Sublinear TF', value=False,
						key='regression_tfidf_transformer_sublinear' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_tfidf_transformer_apply',
								use_container_width=True ):
							if text_count_cols:
								df_processed = df_working.copy( )
								transformer = TfidfTransformer( norm=norm, use_idf=bool( use_idf ),
									smooth_idf=bool( smooth_idf ),
									sublinear_tf=bool( sublinear_tf ) )
								
								result = transformer.train_transform(
									df_processed[ text_count_cols ].apply( pd.to_numeric,
										errors='coerce' ).fillna( 0.0 ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, text_count_cols,
									result, 'tfidf_transformer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'TFIDF Transformer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_tfidf_transformer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Column Transformer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.COLUMN_TRANSFORMER )
					
					numeric_columns = st.multiselect( 'Numeric Columns',
						options=df_working.columns,
						key='regression_column_transformer_numeric_columns' )
					
					categorical_columns = st.multiselect( 'Categorical Columns',
						options=df_working.columns,
						key='regression_column_transformer_categorical_columns' )
					
					numeric_transform = st.selectbox( 'Numeric Transformer',
						options=[ 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler',
							'Binarizer', 'None' ],
						key='regression_column_transformer_numeric_transform' )
					
					categorical_transform = st.selectbox( 'Categorical Transformer',
						options=[ 'OneHotEncoder', 'OrdinalEncoder', 'None' ],
						key='regression_column_transformer_categorical_transform' )
					
					remainder = st.selectbox( 'Remainder', options=[ 'drop', 'passthrough' ],
						key='regression_column_transformer_remainder' )
					
					sparse_threshold = st.slider( 'Sparse Threshold', min_value=0.0, max_value=1.0,
						value=0.3, key='regression_column_transformer_sparse_threshold' )
					
					# Apply Button
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
								
								transformers.append(
									('categorical', categorical_model, categorical_columns) )
							
							if transformers:
								transformer = ColumnTransformer( transformers=transformers,
									remainder=remainder, sparse_threshold=float(
										sparse_threshold ),
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
						if st.button( label='Reset', icon='🔁',
								key='regression_column_transformer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
			
			with st.expander( label='Feature Extration', icon='⛏️', key='regression_extractors' ):
				with st.expander( 'TF-IDF Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.TDIDF_VECTORIZER )
					
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns,
						key='regression_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='regression_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='regression_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='regression_tfidf_vectorizer_use_idf' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_tfidf_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = TfidfVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), use_idf=bool( use_idf ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'tfidf_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'TFIDF Vectorizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_tfidf_vectorizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.COUNT_VECTORIZER )
					
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns,
						key='regression_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='regression_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='regression_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='regression_count_vectorizer_binary' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_count_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = CountVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), binary=bool( binary ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'count_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Count Vectorizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_count_vectorizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.HASH_VECTORIZER )
					
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns,
						key='regression_hash_vectorizer_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='regression_hash_vectorizer_n_features' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='regression_hash_vectorizer_ngram_max' )
					
					binary = st.checkbox( 'Binary', value=False,
						key='regression_hash_vectorizer_binary' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='regression_hash_vectorizer_alternate_sign' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_hash_vectorizer_apply', use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = HashVectorizer( num=int( n_features ),
									ngram_range=(1, int( ngram_max )), binary=bool( binary ),
									alternate_sign=bool( alternate_sign ) )
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'hash_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'HashVectorizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_hash_vectorizer_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.DICT_VECTORIZER )
					
					dict_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='regression_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='regression_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='regression_dict_vectorizer_sort' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_dict_vectorizer_apply', use_container_width=True ):
							if dict_cols:
								df_processed = df_working.copy( )
								transformer = DictVectorizer( dtype=np.float64,
									separator=separator,
									sparse=bool( sparse ), sort=bool( sort ) )
								
								df_processed = apply_dict_transform( df_processed, dict_cols,
									transformer, 'dict_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'DictVectorizer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_dict_vectorizer_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.FEATURE_HASHER )
					
					hash_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_feature_hasher_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='regression_feature_hasher_n_features' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='regression_feature_hasher_alternate_sign' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_feature_hasher_apply', use_container_width=True ):
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
						if st.button( label='Reset', icon='🔁',
								key='regression_feature_hasher_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
			
			with st.expander( label='Dimensionality Reduction', icon='🎚️',
					key='regression_selectors' ):
				with st.expander( 'Variance Threshold', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.VARIANCE_THRESHOLD )
					
					select_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0, step=0.01,
						key='regression_variance_threshold_value' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_variance_threshold_apply',
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
						if st.button( label='Reset', icon='🔁',
								key='regression_variance_threshold_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Canonical Correlation Analysis (CCA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.CCA )
					
					X_cols = st.multiselect( 'Predictor Columns', options=df_working.columns,
						key='regression_cca_x_cols' )
					
					y_cols = st.multiselect( 'Target Columns', options=df_working.columns,
						key='regression_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='regression_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True, key='regression_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500, step=1,
						key='regression_cca_max_iter' )
					
					# Apply Button
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
									[ df_processed.drop( columns=X_cols + y_cols,
										errors='ignore' ),
										df_result ], axis=1 )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'CCA Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_cca_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Principle Component Analysis (PCA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.PCA )
					
					select_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_pca_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='regression_pca_components' )
					
					solver = st.selectbox( 'SVD Solver',
						options=[ 'auto', 'full', 'randomized', 'covariance_eigh', 'arpack' ],
						key='regression_pca_solver' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_pca_apply',
								use_container_width=True ):
							if select_cols:
								df_processed = df_working.copy( )
								selector = PCA( num=int( n_components ), solver=solver )
								
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result,
									'pca' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'PCA applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_pca_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Select-Best', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SELECT_BEST )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='regression_selectbest_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='regression_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
							'mutual_info_regression' ], key='regression_selectbest_score_name' )
					
					k_best = st.number_input( 'K', min_value=1, value=5, step=1,
						key='regression_selectbest_k' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_selectbest_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectBest(
									score_func=score_function_from_name( score_name ),
									num=int( k_best ) )
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'select_best' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Select Best Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_selectbest_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Select-Percent', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SELECT_PERCENT )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='regression_selectpercent_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='regression_selectpercent_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
							'mutual_info_regression' ], key='regression_selectpercent_score_name' )
					
					percentile = st.slider( 'Percentile', min_value=1, max_value=100, value=10,
						key='regression_selectpercent_percentile' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SelectPercent', key='regression_selectpercent_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectPercent(
									score_func=score_function_from_name( score_name ),
									pct=int( percentile ) )
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'select_percent' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'SelectPercent applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_selectpercent_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Sequential Back Selection (SBS)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SBS )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='regression_sbs_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='regression_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='regression_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='regression_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1, step=1,
						key='regression_sbs_random_state' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_sbs_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SBS( classifier=None, k_features=int( k_features ),
									test_size=float( test_size ), random_state=int( random_state
									) )
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'sbs' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'SBS applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_sbs_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Recursive Feature Elimination (RFA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.RFE )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='regression_rfe_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='regression_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='regression_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0, step=1,
						key='regression_rfe_verbose' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_rfe_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = RFE( k_features=int( k_features ),
									verbose=int( verbose ) )
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'rfe' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'RFE applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_rfe_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ] = None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
		
		blue_divider( )
		if df_processed is None:
			st.stop( )
		
		st.markdown( '##### Processed Data' )
		st.caption(
			f'Samples: {len( df_processed ):,} | Features: {len( df_processed.columns ):,}' )
		st.data_editor( df_processed, key='regression_processed_data' )
		
		# ------------------------------------------------------------------
		# MODEL TRAINING
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Model Training' )
		active_features = list( st.session_state.get( 'df_features', pd.DataFrame( ) ).columns )
		active_targets = list( st.session_state.get( 'df_targets', pd.DataFrame( ) ).columns )
		st.session_state[ 'active_features' ] = active_features.copy( )
		st.session_state[ 'active_targets' ] = active_targets.copy( )
		
		if not active_features:
			st.warning( '⚠️ No processed feature columns are available for training.' )
			st.stop( )
		
		if not active_targets:
			st.warning( '⚠️ Regression training requires exactly one processed target column.' )
			st.stop( )
		
		target_name = active_targets[ 0 ]
		missing_columns = [ c for c in active_features + [ target_name ] if
			c not in df_processed.columns ]
		if missing_columns:
			st.warning(
				f'⚠️ The processed dataframe is missing required columns: {missing_columns}. '
				f'Apply preprocessing again or reset the working dataset.' )
			st.stop( )
		
		active_features = list( st.session_state.get( 'df_features', pd.DataFrame( ) ).columns )
		active_targets = list( st.session_state.get( 'df_targets', pd.DataFrame( ) ).columns )
		st.session_state[ 'active_features' ] = active_features.copy( )
		st.session_state[ 'active_targets' ] = active_targets.copy( )
		if not active_features:
			st.warning( '⚠️ Regression training requires at least one processed feature column.' )
			st.stop( )
		
		if len( active_targets ) != 1:
			st.warning( '⚠️ Regression training requires exactly one processed target column.' )
			st.stop( )
		
		target_name = active_targets[ 0 ]
		if target_name in active_features:
			st.warning( '⚠️ The regression target cannot also be used as a feature.' )
			st.stop( )
		
		required_columns = active_features + [ target_name ]
		missing_columns = [ column for column in required_columns if
			column not in df_processed.columns ]
		
		if missing_columns:
			st.warning( f'⚠️ The processed dataframe is missing required columns: '
			            f'{missing_columns}. Apply preprocessing again or reset the '
			            f'working dataset.' )
			st.stop( )
		
		df_model = df_processed[ required_columns ].copy( )
		for column in required_columns:
			df_model[ column ] = pd.to_numeric( df_model[ column ], errors='coerce' )
		
		df_model = df_model.replace( [ np.inf, -np.inf ], np.nan )
		df_model = df_model.dropna( subset=required_columns ).copy( )
		
		if df_model.empty:
			st.warning( '⚠️ No complete numeric observations remain after model-input '
			            'validation.' )
			st.stop( )
		
		if len( df_model ) < 2:
			st.warning( '⚠️ Regression training requires at least two complete observations.' )
			st.stop( )
		
		X_data = df_model[ active_features ].copy( )
		y_series = df_model[ target_name ].copy( )
		if X_data.empty or len( X_data.columns ) == 0:
			st.warning( '⚠️ Regression training requires at least one numeric feature.' )
			st.stop( )
		
		if X_data.isna( ).any( ).any( ):
			st.warning( '⚠️ One or more feature columns contain invalid numeric values.' )
			st.stop( )
		
		if y_series.isna( ).any( ):
			st.warning( '⚠️ The regression target contains invalid numeric values.' )
			st.stop( )
		
		if int( y_series.nunique( dropna=True ) ) < 2:
			st.warning( '⚠️ The regression target must contain at least two distinct values.' )
			st.stop( )
		
		X = X_data.to_numpy( dtype=float )
		y = y_series.to_numpy( dtype=float ).reshape( -1 )
		if X.ndim != 2 or X.shape[ 1 ] < 1:
			st.warning( '⚠️ Regression training requires a two-dimensional feature matrix.' )
			st.stop( )
		
		if y.ndim != 1:
			st.warning( '⚠️ The regression target must resolve to a one-dimensional vector.' )
			st.stop( )
		
		if len( X ) != len( y ):
			st.warning( '⚠️ Feature and target row counts do not match.' )
			st.stop( )
		
		if not np.isfinite( X ).all( ):
			st.warning( '⚠️ The regression feature matrix contains non-finite values.' )
			st.stop( )
		
		if not np.isfinite( y ).all( ):
			st.warning( '⚠️ The regression target contains non-finite values.' )
			st.stop( )
		
		df_regression = df_model.copy( )
		st.session_state[ 'df_model' ] = df_model.copy( )
		st.session_state[ 'df_regression' ] = df_regression.copy( )
		st.session_state[ 'X_data' ] = X_data.copy( )
		st.session_state[ 'y_series' ] = y_series.copy( )
		st.session_state[ 'X_train' ] = None
		st.session_state[ 'X_test' ] = None
		st.session_state[ 'y_train' ] = None
		st.session_state[ 'y_test' ] = None
		st.session_state[ 'y_prediction' ] = None
		
		# ------------------------------------------------------------------
		# REGRESSION MODELS
		# ------------------------------------------------------------------
		with st.expander( 'Linear Models', expanded=False ):
			with st.expander( 'Ordinary Least Squares', expanded=False ):
				ols_defaults = { 'regression_ols_test_size': 0.20,
					'regression_ols_random_state': 42, 'regression_ols_fit_intercept': True,
					'regression_ols_copy_x': True, 'regression_ols_tol': 1e-6,
					'regression_ols_n_jobs': 1, 'regression_ols_positive': False }
				
				for key, value in ols_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Linear regression for continuous targets.' )
				ols_c1, ols_c2, ols_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ols_c1:
					st.markdown( '###### ↔️ Data Split' )
					ols_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_ols_test_size' ] * 100 ), step=5,
						key='regression_ols_test_size_slider' ) / 100.0
					
					ols_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_ols_random_state' ] ), step=1,
						key='regression_ols_random_state_input' ) )
				
				with ols_c2:
					st.markdown( '###### 🎚️ Hyper Parameters' )
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
					st.markdown( '###### ♟️ Solver Settings' )
					ols_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_ols_tol' ] ), step=0.000001,
						format='%.6f', key='regression_ols_tol_input' ) )
					
					ols_n_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
						value=int( st.session_state[ 'regression_ols_n_jobs' ] ), step=1,
						key='regression_ols_n_jobs_input' ) )
				
				ols_btn_1, ols_btn_2 = st.columns( 2 )
				with ols_btn_1:
					train_ols = st.button( '🚂 Train Ordinary Least Squares',
						key='regression_ols_train', use_container_width=True )
				
				with ols_btn_2:
					reset_ols = st.button( '🔄 Reset Ordinary Least Squares',
						key='regression_ols_reset', use_container_width=True )
				
				if reset_ols:
					for key, value in ols_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_ols_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_ols:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Ordinary Least Squares requires prepared '
							            'feature and target arrays.' )
							st.stop( )
						
						X_ols = np.asarray( X, dtype=float )
						y_ols = np.asarray( y, dtype=float ).reshape( -1 )
						if X_ols.ndim != 2 or X_ols.shape[ 1 ] < 1:
							st.warning( '⚠️ Ordinary Least Squares requires at least '
							            'one numeric feature.' )
							st.stop( )
						
						if y_ols.ndim != 1:
							st.warning( '⚠️ The Ordinary Least Squares target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_ols ) != len( y_ols ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_ols ) < 2:
							st.warning( '⚠️ Ordinary Least Squares requires at least '
							            'two observations.' )
							st.stop( )
						
						if not np.isfinite( X_ols ).all( ):
							st.warning( '⚠️ The Ordinary Least Squares feature matrix '
							            'contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_ols ).all( ):
							st.warning( '⚠️ The Ordinary Least Squares target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_ols ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						st.session_state[ 'regression_ols_test_size' ] = float( ols_test_size )
						st.session_state[ 'regression_ols_random_state' ] = int( ols_random_state )
						st.session_state[ 'regression_ols_fit_intercept' ] = bool(
							ols_fit_intercept )
						st.session_state[ 'regression_ols_copy_x' ] = bool( ols_copy_x )
						st.session_state[ 'regression_ols_tol' ] = float( ols_tol )
						st.session_state[ 'regression_ols_n_jobs' ] = int( ols_n_jobs )
						st.session_state[ 'regression_ols_positive' ] = bool( ols_positive )
						start_time = time.perf_counter( )
						
						model = regression_model.LeastSquares( fit=bool( ols_fit_intercept ),
							copy=bool( ols_copy_x ), tol=float( ols_tol ), jobs=int( ols_n_jobs ),
							positive=bool( ols_positive ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_ols, y_ols,
							size=float( ols_test_size ), random=int( ols_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( )
						else:
							df_scores = df_scores.copy( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_ols_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Ordinary Least Squares training completed.' )
					except Exception as ex:
						st.error( f'Ordinary Least Squares training failed: {ex}' )
			
			with st.expander( 'Ridge Regression', expanded=False ):
				ridge_defaults = { 'regression_ridge_alpha': 1.0,
					'regression_ridge_fit_intercept': True, 'regression_ridge_copy_x': True,
					'regression_ridge_max_iter': 0, 'regression_ridge_tol': 0.0001,
					'regression_ridge_solver': 'auto', 'regression_ridge_positive': False,
					'regression_ridge_test_size': 0.20, 'regression_ridge_random_state': 42 }
				
				for key, value in ridge_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'L2-regularized linear regression for continuous targets.' )
				ridge_c1, ridge_c2, ridge_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ridge_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					ridge_alpha = float( st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'regression_ridge_alpha' ] ), step=0.100000,
						format='%.6f', key='regression_ridge_alpha_input' ) )
					
					ridge_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_ridge_fit_intercept' ] ),
						key='regression_ridge_fit_intercept_check' )
					
					ridge_copy_x = st.checkbox( 'Copy X',
						value=bool( st.session_state[ 'regression_ridge_copy_x' ] ),
						key='regression_ridge_copy_x_check' )
				
				with ridge_c2:
					st.markdown( '###### ♟️ Solver / Iteration' )
					ridge_solvers = [ 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag',
						'saga',
						'lbfgs' ]
					
					ridge_solver_value = st.session_state.get( 'regression_ridge_solver', 'auto' )
					if ridge_solver_value not in ridge_solvers:
						ridge_solver_value = 'auto'
					
					ridge_solver = st.selectbox( 'Solver', options=ridge_solvers,
						index=ridge_solvers.index( ridge_solver_value ),
						key='regression_ridge_solver_select' )
					
					ridge_max_iter_raw = int(
						st.number_input( 'Max Iterations (0 = Auto)', min_value=0,
							value=int( st.session_state[ 'regression_ridge_max_iter' ] ), step=1,
							key='regression_ridge_max_iter_input' ) )
					
					ridge_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_ridge_tol' ] ), step=0.000100,
						format='%.6f', key='regression_ridge_tol_input' ) )
					
					ridge_positive = st.checkbox( 'Positive Coefficients',
						value=bool( st.session_state[ 'regression_ridge_positive' ] ),
						key='regression_ridge_positive_check' )
					
					if ridge_positive and ridge_solver != 'lbfgs':
						st.info( "Positive coefficients require the 'lbfgs' solver. "
						         "The solver will be set to 'lbfgs' during training." )
					
					if not ridge_positive and ridge_solver == 'lbfgs':
						st.warning( "The 'lbfgs' solver requires Positive Coefficients." )
				
				with ridge_c3:
					st.markdown( '###### ↔️ Data Split' )
					ridge_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_ridge_test_size' ] * 100 ),
						step=5,
						key='regression_ridge_test_size_slider' ) / 100.0
					
					ridge_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_ridge_random_state' ] ), step=1,
						key='regression_ridge_random_state_input' ) )
				
				ridge_btn_1, ridge_btn_2 = st.columns( 2 )
				with ridge_btn_1:
					train_ridge = st.button( '🚂 Train Ridge Regression',
						key='regression_ridge_train', use_container_width=True )
				
				with ridge_btn_2:
					reset_ridge = st.button( '🔄 Reset Ridge Regression',
						key='regression_ridge_reset', use_container_width=True )
				
				if reset_ridge:
					for key, value in ridge_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_ridge_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_ridge:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Ridge Regression requires prepared feature & target arrays.' )
							st.stop( )
						
						X_ridge = np.asarray( X, dtype=float )
						y_ridge = np.asarray( y, dtype=float ).reshape( -1 )
						if X_ridge.ndim != 2 or X_ridge.shape[ 1 ] < 1:
							st.warning(
								'⚠️ Ridge Regression requires at least one numeric feature.' )
							st.stop( )
						
						if y_ridge.ndim != 1:
							st.warning( '⚠️ The Ridge Regression target must be one-dimensional.' )
							st.stop( )
						
						if len( X_ridge ) != len( y_ridge ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_ridge ) < 2:
							st.warning( '⚠️ Ridge Regression requires at least two observations.' )
							st.stop( )
						
						if not np.isfinite( X_ridge ).all( ):
							st.warning( '⚠️ The Ridge feature matrix contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_ridge ).all( ):
							st.warning( '⚠️ The Ridge target contains non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_ridge ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						if ridge_alpha <= 0.0:
							st.warning( '⚠️ Ridge alpha must be greater than zero.' )
							st.stop( )
						
						if ridge_tol < 0.0:
							st.warning( '⚠️ Ridge tolerance cannot be negative.' )
							st.stop( )
						
						if (not ridge_positive and ridge_solver == 'lbfgs'):
							st.warning( "⚠️ The 'lbfgs' solver requires Positive Coefficients." )
							st.stop( )
						
						effective_solver = ('lbfgs' if ridge_positive else str( ridge_solver ))
						effective_max_iter = (
							None if ridge_max_iter_raw == 0 else int( ridge_max_iter_raw ))
						
						st.session_state[ 'regression_ridge_alpha' ] = float( ridge_alpha )
						st.session_state[ 'regression_ridge_fit_intercept' ] = bool(
							ridge_fit_intercept )
						st.session_state[ 'regression_ridge_copy_x' ] = bool( ridge_copy_x )
						st.session_state[ 'regression_ridge_max_iter' ] = int( ridge_max_iter_raw )
						st.session_state[ 'regression_ridge_tol' ] = float( ridge_tol )
						st.session_state[ 'regression_ridge_solver' ] = str( ridge_solver )
						st.session_state[ 'regression_ridge_positive' ] = bool( ridge_positive )
						st.session_state[ 'regression_ridge_test_size' ] = float( ridge_test_size )
						st.session_state[ 'regression_ridge_random_state' ] = int(
							ridge_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.Ridge( alpha=float( ridge_alpha ),
							fit=bool( ridge_fit_intercept ), copy=bool( ridge_copy_x ),
							iters=effective_max_iter, tol=float( ridge_tol ),
							solver=effective_solver, positive=bool( ridge_positive ),
							rando=int( ridge_random_state ) )
						
						X_train, X_test, y_train, y_test = (
							model.split_data( X_ridge, y_ridge, size=float( ridge_test_size ),
								random=int( ridge_random_state ) ))
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Alpha', 'Solver', 'Maximum Iterations',
								'Positive Coefficients' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), float( ridge_alpha ),
								str( effective_solver ),
								('Auto' if effective_max_iter is None else int(
									effective_max_iter )), bool( ridge_positive ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						coefficient_values = np.asarray( model.weights, dtype=float ).reshape( -1 )
						if len( coefficient_values ) != len( active_features ):
							coefficient_names = [ f'Feature {index + 1}' for index in
								range( len( coefficient_values ) ) ]
						else:
							coefficient_names = active_features.copy( )
						
						df_coefficients = pd.DataFrame(
							{ 'Feature': coefficient_names, 'Coefficient': coefficient_values } )
						
						st.session_state[ 'regression_ridge_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.session_state[ 'df_coefficients' ] = df_coefficients.copy( )
						st.success( 'Ridge Regression training completed.' )
					except Exception as ex:
						st.error( f'Ridge Regression training failed: {ex}' )
			
			with st.expander( 'Lasso Regression', expanded=False ):
				lasso_defaults = { 'regression_lasso_alpha': 1.0,
					'regression_lasso_fit_intercept': True, 'regression_lasso_precompute': False,
					'regression_lasso_copy_x': True, 'regression_lasso_max_iter': 1000,
					'regression_lasso_tol': 0.0001, 'regression_lasso_warm_start': False,
					'regression_lasso_positive': False, 'regression_lasso_random_state': 42,
					'regression_lasso_selection': 'cyclic', 'regression_lasso_test_size': 0.20 }
				
				for key, value in lasso_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'L1-regularized linear regression for continuous targets.' )
				lasso_c1, lasso_c2, lasso_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with lasso_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					lasso_alpha = float( st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'regression_lasso_alpha' ] ), step=0.100000,
						format='%.6f', key='regression_lasso_alpha_input' ) )
					
					lasso_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_lasso_fit_intercept' ] ),
						key='regression_lasso_fit_intercept_check' )
					
					lasso_precompute = st.checkbox( 'Precompute',
						value=bool( st.session_state[ 'regression_lasso_precompute' ] ),
						key='regression_lasso_precompute_check' )
					
					lasso_copy_x = st.checkbox( 'Copy X',
						value=bool( st.session_state[ 'regression_lasso_copy_x' ] ),
						key='regression_lasso_copy_x_check' )
				
				with lasso_c2:
					st.markdown( '###### Solver / Iteration' )
					
					lasso_max_iter = int( st.number_input( 'Max Iterations', min_value=1,
						value=int( st.session_state[ 'regression_lasso_max_iter' ] ), step=1,
						key='regression_lasso_max_iter_input' ) )
					
					lasso_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_lasso_tol' ] ), step=0.000100,
						format='%.6f', key='regression_lasso_tol_input' ) )
					
					lasso_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_lasso_warm_start' ] ),
						key='regression_lasso_warm_start_check' )
					
					lasso_positive = st.checkbox( 'Positive Coefficients',
						value=bool( st.session_state[ 'regression_lasso_positive' ] ),
						key='regression_lasso_positive_check' )
					
					lasso_selection_options = [ 'cyclic', 'random' ]
					
					lasso_selection_value = st.session_state.get( 'regression_lasso_selection',
						'cyclic' )
					
					if lasso_selection_value not in lasso_selection_options:
						lasso_selection_value = 'cyclic'
					
					lasso_selection = st.selectbox( 'Selection', options=lasso_selection_options,
						index=lasso_selection_options.index( lasso_selection_value ),
						key='regression_lasso_selection_select' )
				
				with lasso_c3:
					st.markdown( '###### Data Split' )
					lasso_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_lasso_test_size' ] * 100 ),
						step=5,
						key='regression_lasso_test_size_slider' ) / 100.0
					
					lasso_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_lasso_random_state' ] ), step=1,
						key='regression_lasso_random_state_input' ) )
				
				lasso_btn_1, lasso_btn_2 = st.columns( 2 )
				with lasso_btn_1:
					train_lasso = st.button( '🚂 Train Lasso Regression',
						key='regression_lasso_train', use_container_width=True )
				
				with lasso_btn_2:
					reset_lasso = st.button( '🔄 Reset Lasso Regression',
						key='regression_lasso_reset', use_container_width=True )
				
				if reset_lasso:
					for key, value in lasso_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_lasso_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_lasso:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Lasso Regression requires prepared '
							            'feature and target arrays.' )
							st.stop( )
						
						X_lasso = np.asarray( X, dtype=float )
						y_lasso = np.asarray( y, dtype=float ).reshape( -1 )
						if X_lasso.ndim != 2 or X_lasso.shape[ 1 ] < 1:
							st.warning( '⚠️ Lasso Regression requires at least one '
							            'numeric feature.' )
							st.stop( )
						
						if y_lasso.ndim != 1:
							st.warning( '⚠️ The Lasso Regression target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_lasso ) != len( y_lasso ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_lasso ) < 2:
							st.warning( '⚠️ Lasso Regression requires at least two '
							            'observations.' )
							st.stop( )
						
						if not np.isfinite( X_lasso ).all( ):
							st.warning( '⚠️ The Lasso feature matrix contains '
							            'non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_lasso ).all( ):
							st.warning( '⚠️ The Lasso target contains non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_lasso ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						if lasso_alpha <= 0.0:
							st.warning( '⚠️ Lasso alpha must be greater than zero.' )
							st.stop( )
						
						if lasso_max_iter < 1:
							st.warning( '⚠️ Maximum iterations must be greater than zero.' )
							st.stop( )
						
						if lasso_tol < 0.0:
							st.warning( '⚠️ Lasso tolerance cannot be negative.' )
							st.stop( )
						
						st.session_state[ 'regression_lasso_alpha' ] = float( lasso_alpha )
						st.session_state[ 'regression_lasso_fit_intercept' ] = bool(
							lasso_fit_intercept )
						st.session_state[ 'regression_lasso_precompute' ] = bool(
							lasso_precompute )
						st.session_state[ 'regression_lasso_copy_x' ] = bool( lasso_copy_x )
						st.session_state[ 'regression_lasso_max_iter' ] = int( lasso_max_iter )
						st.session_state[ 'regression_lasso_tol' ] = float( lasso_tol )
						st.session_state[ 'regression_lasso_warm_start' ] = bool(
							lasso_warm_start )
						st.session_state[ 'regression_lasso_positive' ] = bool( lasso_positive )
						st.session_state[ 'regression_lasso_random_state' ] = int(
							lasso_random_state )
						st.session_state[ 'regression_lasso_selection' ] = str( lasso_selection )
						st.session_state[ 'regression_lasso_test_size' ] = float( lasso_test_size )
						start_time = time.perf_counter( )
						model = regression_model.Lasso( alpha=float( lasso_alpha ),
							fit=bool( lasso_fit_intercept ), precompute=bool( lasso_precompute ),
							copy=bool( lasso_copy_x ), iters=int( lasso_max_iter ),
							tol=float( lasso_tol ), warm=bool( lasso_warm_start ),
							positive=bool( lasso_positive ), rando=int( lasso_random_state ),
							selection=str( lasso_selection ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_lasso, y_lasso,
							size=float( lasso_test_size ), random=int( lasso_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Alpha', 'Selection', 'Maximum Iterations',
								'Positive Coefficients' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), float( lasso_alpha ), str( lasso_selection ),
								int( lasso_max_iter ), bool( lasso_positive ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						coefficient_values = np.asarray( model.weights, dtype=float ).reshape( -1 )
						if len( coefficient_values ) == len( active_features ):
							coefficient_names = active_features.copy( )
						else:
							coefficient_names = [ f'Feature {index + 1}' for index in
								range( len( coefficient_values ) ) ]
						
						df_coefficients = pd.DataFrame(
							{ 'Feature': coefficient_names, 'Coefficient': coefficient_values } )
						
						st.session_state[ 'regression_lasso_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.session_state[ 'df_coefficients' ] = df_coefficients.copy( )
						
						st.success( 'Lasso Regression training completed.' )
					
					except Exception as ex:
						st.error( f'Lasso Regression training failed: {ex}' )
			
			with st.expander( 'Elastic Net', expanded=False ):
				elastic_defaults = { 'regression_elastic_alpha': 1.0,
					'regression_elastic_ratio': 0.5, 'regression_elastic_fit_intercept': True,
					'regression_elastic_precompute': False, 'regression_elastic_copy_x': True,
					'regression_elastic_max_iter': 1000, 'regression_elastic_tol': 0.0001,
					'regression_elastic_warm_start': False, 'regression_elastic_positive': False,
					'regression_elastic_random_state': 42, 'regression_elastic_selection':
						'cyclic',
					'regression_elastic_test_size': 0.20 }
				
				for key, value in elastic_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Combined L1/L2-regularized linear regression for continuous '
				            'targets.' )
				
				elastic_c1, elastic_c2, elastic_c3 = st.columns( [ 0.34, 0.33, 0.33 ],
					border=True )
				
				with elastic_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					elastic_alpha = float( st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'regression_elastic_alpha' ] ),
						step=0.100000, format='%.6f', key='regression_elastic_alpha_input' ) )
					
					elastic_ratio = float( st.slider( 'L1 Ratio', min_value=0.0, max_value=1.0,
						value=float( st.session_state[ 'regression_elastic_ratio' ] ), step=0.05,
						key='regression_elastic_ratio_slider' ) )
					
					elastic_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_elastic_fit_intercept' ] ),
						key='regression_elastic_fit_intercept_check' )
					
					elastic_precompute = st.checkbox( 'Precompute',
						value=bool( st.session_state[ 'regression_elastic_precompute' ] ),
						key='regression_elastic_precompute_check' )
				
				with elastic_c2:
					st.markdown( '###### Solver / Iteration' )
					elastic_copy_x = st.checkbox( 'Copy X',
						value=bool( st.session_state[ 'regression_elastic_copy_x' ] ),
						key='regression_elastic_copy_x_check' )
					
					elastic_max_iter = int( st.number_input( 'Max Iterations', min_value=1,
						value=int( st.session_state[ 'regression_elastic_max_iter' ] ), step=1,
						key='regression_elastic_max_iter_input' ) )
					
					elastic_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_elastic_tol' ] ), step=0.000100,
						format='%.6f', key='regression_elastic_tol_input' ) )
					
					elastic_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_elastic_warm_start' ] ),
						key='regression_elastic_warm_start_check' )
					
					elastic_positive = st.checkbox( 'Positive Coefficients',
						value=bool( st.session_state[ 'regression_elastic_positive' ] ),
						key='regression_elastic_positive_check' )
					
					elastic_selection_options = [ 'cyclic', 'random' ]
					elastic_selection_value = st.session_state.get( 'regression_elastic_selection',
						'cyclic' )
					
					if elastic_selection_value not in elastic_selection_options:
						elastic_selection_value = 'cyclic'
					
					elastic_selection = st.selectbox( 'Selection',
						options=elastic_selection_options,
						index=elastic_selection_options.index( elastic_selection_value ),
						key='regression_elastic_selection_select' )
				
				with elastic_c3:
					st.markdown( '###### Data Split' )
					elastic_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_elastic_test_size' ] * 100 ),
						step=5, key='regression_elastic_test_size_slider' ) / 100.0
					
					elastic_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_elastic_random_state' ] ), step=1,
						key='regression_elastic_random_state_input' ) )
					
					if elastic_ratio == 0.0:
						st.info( 'l1_ratio = 0.0 uses only L2 regularization.' )
					
					if elastic_ratio == 1.0:
						st.info( 'l1_ratio = 1.0 is equivalent to Lasso.' )
				
				elastic_btn_1, elastic_btn_2 = st.columns( 2 )
				with elastic_btn_1:
					train_elastic = st.button( '🚂 Train Elastic Net',
						key='regression_elastic_train', use_container_width=True )
				
				with elastic_btn_2:
					reset_elastic = st.button( '🔄 Reset Elastic Net',
						key='regression_elastic_reset', use_container_width=True )
				
				if reset_elastic:
					for key, value in elastic_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_elastic_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_elastic:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Elastic Net requires prepared feature and target arrays.' )
							st.stop( )
						
						X_elastic = np.asarray( X, dtype=float )
						
						y_elastic = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_elastic.ndim != 2 or X_elastic.shape[ 1 ] < 1:
							st.warning( '⚠️ Elastic Net requires at least one numeric feature.' )
							st.stop( )
						
						if y_elastic.ndim != 1:
							st.warning( '⚠️ The Elastic Net target must be one-dimensional.' )
							st.stop( )
						
						if len( X_elastic ) != len( y_elastic ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_elastic ) < 2:
							st.warning( '⚠️ Elastic Net requires at least two observations.' )
							st.stop( )
						
						if not np.isfinite( X_elastic ).all( ):
							st.warning(
								'⚠️ The Elastic Net feature matrix contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_elastic ).all( ):
							st.warning( '⚠️ The Elastic Net target contains non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_elastic ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						if elastic_alpha <= 0.0:
							st.warning( '⚠️ Elastic Net alpha must be greater than zero.' )
							st.stop( )
						
						if elastic_ratio < 0.0 or elastic_ratio > 1.0:
							st.warning( '⚠️ The Elastic Net L1 ratio must be between '
							            '0.0 and 1.0.' )
							st.stop( )
						
						if elastic_max_iter < 1:
							st.warning( '⚠️ Maximum iterations must be greater than zero.' )
							st.stop( )
						
						if elastic_tol < 0.0:
							st.warning( '⚠️ Elastic Net tolerance cannot be negative.' )
							st.stop( )
						
						st.session_state[ 'regression_elastic_alpha' ] = float( elastic_alpha )
						st.session_state[ 'regression_elastic_ratio' ] = float( elastic_ratio )
						st.session_state[ 'regression_elastic_fit_intercept' ] = bool(
							elastic_fit_intercept )
						st.session_state[ 'regression_elastic_precompute' ] = bool(
							elastic_precompute )
						st.session_state[ 'regression_elastic_copy_x' ] = bool( elastic_copy_x )
						st.session_state[ 'regression_elastic_max_iter' ] = int( elastic_max_iter )
						st.session_state[ 'regression_elastic_tol' ] = float( elastic_tol )
						st.session_state[ 'regression_elastic_warm_start' ] = bool(
							elastic_warm_start )
						st.session_state[ 'regression_elastic_positive' ] = bool(
							elastic_positive )
						st.session_state[ 'regression_elastic_random_state' ] = int(
							elastic_random_state )
						st.session_state[ 'regression_elastic_selection' ] = str(
							elastic_selection )
						st.session_state[ 'regression_elastic_test_size' ] = float(
							elastic_test_size )
						start_time = time.perf_counter( )
						model = regression_model.ElasticNet( alpha=float( elastic_alpha ),
							ratio=float( elastic_ratio ), fit=bool( elastic_fit_intercept ),
							precompute=bool( elastic_precompute ), iters=int( elastic_max_iter ),
							copy=bool( elastic_copy_x ), tol=float( elastic_tol ),
							warm=bool( elastic_warm_start ), positive=bool( elastic_positive ),
							rando=int( elastic_random_state ), select=str( elastic_selection ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_elastic, y_elastic,
							size=float( elastic_test_size ), random=int( elastic_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Alpha', 'L1 Ratio', 'Selection',
								'Maximum Iterations', 'Positive Coefficients' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), float( elastic_alpha ),
								float( elastic_ratio ), str( elastic_selection ),
								int( elastic_max_iter ), bool( elastic_positive ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						coefficient_values = np.asarray( model.weights, dtype=float ).reshape( -1 )
						if len( coefficient_values ) == len( active_features ):
							coefficient_names = active_features.copy( )
						else:
							coefficient_names = [ f'Feature {index + 1}' for index in
								range( len( coefficient_values ) ) ]
						
						df_coefficients = pd.DataFrame(
							{ 'Feature': coefficient_names, 'Coefficient': coefficient_values } )
						
						st.session_state[ 'regression_elastic_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.session_state[ 'df_coefficients' ] = df_coefficients.copy( )
						st.success( 'Elastic Net training completed.' )
					except Exception as ex:
						st.error( f'Elastic Net training failed: {ex}' )
			
			with st.expander( 'Bayesian Ridge', expanded=False ):
				bayes_defaults = { 'regression_bayes_max_iter': 300,
					'regression_bayes_shape_alpha': 0.000001,
					'regression_bayes_scale_alpha': 0.000001,
					'regression_bayes_shape_lambda': 0.000001,
					'regression_bayes_scale_lambda': 0.000001, 'regression_bayes_tol': 0.001000,
					'regression_bayes_alpha_init': 0.0, 'regression_bayes_lambda_init': 0.0,
					'regression_bayes_compute_score': False, 'regression_bayes_fit_intercept':
						True,
					'regression_bayes_copy_x': True, 'regression_bayes_verbose': False,
					'regression_bayes_test_size': 0.20, 'regression_bayes_random_state': 42 }
				
				for key, value in bayes_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Bayesian linear regression with automatic regularization '
				            'estimation.' )
				
				bayes_c1, bayes_c2, bayes_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with bayes_c1:
					st.markdown( '###### Prior / Precision Parameters' )
					
					bayes_shape_alpha = float( st.number_input( 'Alpha Shape', min_value=0.0,
						value=float( st.session_state[ 'regression_bayes_shape_alpha' ] ),
						step=0.000001, format='%.6f', key='regression_bayes_shape_alpha_input' ) )
					
					bayes_scale_alpha = float( st.number_input( 'Alpha Scale', min_value=0.0,
						value=float( st.session_state[ 'regression_bayes_scale_alpha' ] ),
						step=0.000001, format='%.6f', key='regression_bayes_scale_alpha_input' ) )
					
					bayes_shape_lambda = float( st.number_input( 'Lambda Shape', min_value=0.0,
						value=float( st.session_state[ 'regression_bayes_shape_lambda' ] ),
						step=0.000001, format='%.6f', key='regression_bayes_shape_lambda_input' ) )
					
					bayes_scale_lambda = float( st.number_input( 'Lambda Scale', min_value=0.0,
						value=float( st.session_state[ 'regression_bayes_scale_lambda' ] ),
						step=0.000001, format='%.6f', key='regression_bayes_scale_lambda_input' ) )
				
				with bayes_c2:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					
					bayes_max_iter = int( st.number_input( 'Max Iterations', min_value=1,
						value=int( st.session_state[ 'regression_bayes_max_iter' ] ), step=1,
						key='regression_bayes_max_iter_input' ) )
					
					bayes_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_bayes_tol' ] ), step=0.000100,
						format='%.6f', key='regression_bayes_tol_input' ) )
					
					bayes_alpha_init_raw = float(
						st.number_input( 'Alpha Init (0 = None)', min_value=0.0,
							value=float( st.session_state[ 'regression_bayes_alpha_init' ] ),
							step=0.000100, format='%.6f',
							key='regression_bayes_alpha_init_input' ) )
					
					bayes_lambda_init_raw = float(
						st.number_input( 'Lambda Init (0 = None)', min_value=0.0,
							value=float( st.session_state[ 'regression_bayes_lambda_init' ] ),
							step=0.000100, format='%.6f',
							key='regression_bayes_lambda_init_input' ) )
					
					bayes_compute_score = st.checkbox( 'Compute Marginal Log Likelihood',
						value=bool( st.session_state[ 'regression_bayes_compute_score' ] ),
						key='regression_bayes_compute_score_check' )
					
					bayes_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_bayes_fit_intercept' ] ),
						key='regression_bayes_fit_intercept_check' )
					
					bayes_copy_x = st.checkbox( 'Copy X',
						value=bool( st.session_state[ 'regression_bayes_copy_x' ] ),
						key='regression_bayes_copy_x_check' )
					
					bayes_verbose = st.checkbox( 'Verbose',
						value=bool( st.session_state[ 'regression_bayes_verbose' ] ),
						key='regression_bayes_verbose_check' )
				
				with bayes_c3:
					st.markdown( '###### Data Split' )
					
					bayes_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_bayes_test_size' ] * 100 ),
						step=5,
						key='regression_bayes_test_size_slider' ) / 100.0
					
					bayes_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_bayes_random_state' ] ), step=1,
						key='regression_bayes_random_state_input' ) )
				
				bayes_btn_1, bayes_btn_2 = st.columns( 2 )
				with bayes_btn_1:
					train_bayes = st.button( '🚂 Train Bayesian Ridge',
						key='regression_bayes_train',
						use_container_width=True )
				
				with bayes_btn_2:
					reset_bayes = st.button( '🔄 Reset Bayesian Ridge',
						key='regression_bayes_reset',
						use_container_width=True )
				
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
							bayes_shape_alpha )
						st.session_state[ 'regression_bayes_scale_alpha' ] = float(
							bayes_scale_alpha )
						st.session_state[ 'regression_bayes_shape_lambda' ] = float(
							bayes_shape_lambda )
						st.session_state[ 'regression_bayes_scale_lambda' ] = float(
							bayes_scale_lambda )
						st.session_state[ 'regression_bayes_tol' ] = float( bayes_tol )
						st.session_state[ 'regression_bayes_alpha_init' ] = float(
							bayes_alpha_init_raw )
						st.session_state[ 'regression_bayes_lambda_init' ] = float(
							bayes_lambda_init_raw )
						st.session_state[ 'regression_bayes_compute_score' ] = bool(
							bayes_compute_score )
						st.session_state[ 'regression_bayes_fit_intercept' ] = bool(
							bayes_fit_intercept )
						st.session_state[ 'regression_bayes_copy_x' ] = bool( bayes_copy_x )
						st.session_state[ 'regression_bayes_verbose' ] = bool( bayes_verbose )
						st.session_state[ 'regression_bayes_test_size' ] = float( bayes_test_size )
						st.session_state[ 'regression_bayes_random_state' ] = int(
							bayes_random_state )
						df_training = df_model.copy( )
						X = df_training[ active_features ].apply( pd.to_numeric,
							errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric( df_training[ target_name ], errors='coerce' ).fillna(
							0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ The target must contain at least two distinct '
							            'values.' )
							st.stop( )
						
						bayes_alpha_init = None if bayes_alpha_init_raw == 0.0 else float(
							bayes_alpha_init_raw )
						
						bayes_lambda_init = None if bayes_lambda_init_raw == 0.0 else float(
							bayes_lambda_init_raw )
						
						start_time = time.perf_counter( )
						
						model = regression_model.BayesianRidge( max=int( bayes_max_iter ),
							shape_alpha=float( bayes_shape_alpha ),
							scale_alpha=float( bayes_scale_alpha ),
							shape_lambda=float( bayes_shape_lambda ),
							scale_lambda=float( bayes_scale_lambda ), tol=float( bayes_tol ),
							alpha_init=bayes_alpha_init, lambda_init=bayes_lambda_init,
							compute_score=bool( bayes_compute_score ),
							fit=bool( bayes_fit_intercept ), copy=bool( bayes_copy_x ),
							verbose=bool( bayes_verbose ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( bayes_test_size ), random=int( bayes_random_state ) )
						
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
							
							df_extra = pd.DataFrame( {
								'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
									'Testing Rows', 'Max Iterations' ],
								'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
									int( len( X_test ) ), int( bayes_max_iter ) ] } )
							
							df_scores = pd.concat( [ df_scores, df_extra ], ignore_index=True )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': y_test, 'Predicted': y_prediction } )
						
						df_coefficients = pd.DataFrame( { 'Feature': active_features,
							'Coefficient': np.asarray( model.model.coef_ ).reshape( -1 ) } )
						
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
				sgd_defaults = { 'regression_sgd_loss': 'squared_error',
					'regression_sgd_penalty': 'l2', 'regression_sgd_alpha': 0.0001,
					'regression_sgd_iters': 1000, 'regression_sgd_shuffle': True,
					'regression_sgd_learning_rate': 'invscaling', 'regression_sgd_l1_ratio': 0.15,
					'regression_sgd_fit_intercept': True, 'regression_sgd_tol': 0.001,
					'regression_sgd_verbose': 0, 'regression_sgd_epsilon': 0.1,
					'regression_sgd_eta0': 0.01, 'regression_sgd_power_t': 0.25,
					'regression_sgd_early_stopping': False,
					'regression_sgd_validation_fraction': 0.1,
					'regression_sgd_n_iter_no_change': 5,
					'regression_sgd_warm_start': False, 'regression_sgd_average': False,
					'regression_sgd_test_size': 0.20, 'regression_sgd_random_state': 42 }
				
				for key, value in sgd_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption(
					'Linear regression trained with SGD for large-scale continuous targets.' )
				
				sgd_c1, sgd_c2, sgd_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with sgd_c1:
					st.markdown( '###### Loss / Penalty' )
					
					sgd_loss_options = [ 'squared_error', 'huber', 'epsilon_insensitive',
						'squared_epsilon_insensitive' ]
					
					sgd_loss_value = st.session_state.get( 'regression_sgd_loss', 'squared_error' )
					
					if sgd_loss_value not in sgd_loss_options:
						sgd_loss_value = 'squared_error'
					
					sgd_loss = st.selectbox( 'Loss', options=sgd_loss_options,
						index=sgd_loss_options.index( sgd_loss_value ),
						key='regression_sgd_loss_select' )
					
					sgd_penalty_options = [ None, 'l2', 'l1', 'elasticnet' ]
					sgd_penalty_value = st.session_state.get( 'regression_sgd_penalty', 'l2' )
					if sgd_penalty_value not in sgd_penalty_options:
						sgd_penalty_value = 'l2'
					
					sgd_penalty = st.selectbox( 'Penalty', options=sgd_penalty_options,
						index=sgd_penalty_options.index( sgd_penalty_value ),
						format_func=lambda value: 'None' if value is None else str( value ),
						key='regression_sgd_penalty_select' )
					
					sgd_alpha = float( st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'regression_sgd_alpha' ] ), step=0.0001,
						format='%.6f', key='regression_sgd_alpha_input' ) )
					
					sgd_iters = int( st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'regression_sgd_iters' ] ), step=1,
						key='regression_sgd_iters_input' ) )
					
					sgd_l1_ratio = float( st.slider( 'L1 Ratio', min_value=0.0, max_value=1.0,
						value=float( st.session_state[ 'regression_sgd_l1_ratio' ] ), step=0.05,
						key='regression_sgd_l1_ratio_slider' ) )
				
				with sgd_c2:
					st.markdown( '###### Learning Controls' )
					
					sgd_shuffle = st.checkbox( 'Shuffle',
						value=bool( st.session_state[ 'regression_sgd_shuffle' ] ),
						key='regression_sgd_shuffle_check' )
					
					sgd_learning_rate_options = [ 'constant', 'optimal', 'invscaling', 'adaptive' ]
					
					sgd_learning_rate_value = st.session_state.get( 'regression_sgd_learning_rate',
						'invscaling' )
					
					if sgd_learning_rate_value not in sgd_learning_rate_options:
						sgd_learning_rate_value = 'invscaling'
					
					sgd_learning_rate = st.selectbox( 'Learning Rate Schedule',
						options=sgd_learning_rate_options,
						index=sgd_learning_rate_options.index( sgd_learning_rate_value ),
						key='regression_sgd_learning_rate_select' )
					
					sgd_eta0 = float( st.number_input( 'Eta0', min_value=0.000001,
						value=float( st.session_state[ 'regression_sgd_eta0' ] ), step=0.01,
						format='%.6f', key='regression_sgd_eta0_input' ) )
					
					sgd_power_t = float( st.number_input( 'Power T', min_value=0.0,
						value=float( st.session_state[ 'regression_sgd_power_t' ] ), step=0.1,
						format='%.6f', key='regression_sgd_power_t_input' ) )
					
					sgd_epsilon = float( st.number_input( 'Epsilon', min_value=0.0,
						value=float( st.session_state[ 'regression_sgd_epsilon' ] ), step=0.01,
						format='%.6f', key='regression_sgd_epsilon_input' ) )
					
					sgd_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_sgd_tol' ] ), step=0.0001,
						format='%.6f', key='regression_sgd_tol_input' ) )
					
					sgd_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_sgd_fit_intercept' ] ),
						key='regression_sgd_fit_intercept_check' )
				
				with sgd_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					sgd_early_stopping = st.checkbox( 'Early Stopping',
						value=bool( st.session_state[ 'regression_sgd_early_stopping' ] ),
						key='regression_sgd_early_stopping_check' )
					
					sgd_validation_fraction = float(
						st.slider( 'Validation Fraction', min_value=0.05, max_value=0.40,
							value=float( st.session_state[ 'regression_sgd_validation_fraction'
							] ),
							step=0.05, key='regression_sgd_validation_fraction_slider' ) )
					
					sgd_n_iter_no_change = int( st.number_input( 'N Iter No Change', min_value=1,
						value=int( st.session_state[ 'regression_sgd_n_iter_no_change' ] ), step=1,
						key='regression_sgd_n_iter_no_change_input' ) )
					
					sgd_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_sgd_warm_start' ] ),
						key='regression_sgd_warm_start_check' )
					
					sgd_average = st.checkbox( 'Average Weights',
						value=bool( st.session_state[ 'regression_sgd_average' ] ),
						key='regression_sgd_average_check' )
					
					sgd_verbose = int( st.number_input( 'Verbose', min_value=0,
						value=int( st.session_state[ 'regression_sgd_verbose' ] ), step=1,
						key='regression_sgd_verbose_input' ) )
					
					sgd_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_sgd_test_size' ] * 100 ), step=5,
						key='regression_sgd_test_size_slider' ) / 100.0
					
					sgd_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_sgd_random_state' ] ), step=1,
						key='regression_sgd_random_state_input' ) )
					
					if sgd_penalty != 'elasticnet':
						st.caption( 'L1 Ratio is only used when Penalty = elasticnet.' )
					
					epsilon_losses = [ 'huber', 'epsilon_insensitive',
						'squared_epsilon_insensitive' ]
					
					if sgd_loss not in epsilon_losses:
						st.caption( 'Epsilon is only used by Huber and '
						            'epsilon-insensitive losses.' )
				
				sgd_btn_1, sgd_btn_2 = st.columns( 2 )
				
				with sgd_btn_1:
					train_sgd = st.button( '🚂 Train Stochastic Gradient Descent',
						key='regression_sgd_train', use_container_width=True )
				
				with sgd_btn_2:
					reset_sgd = st.button( '🔄 Reset Stochastic Gradient Descent',
						key='regression_sgd_reset', use_container_width=True )
				
				if reset_sgd:
					for key, value in sgd_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_sgd_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_sgd:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Stochastic Gradient Descent requires prepared '
							            'feature and target arrays.' )
							st.stop( )
						
						X_sgd = np.asarray( X, dtype=float )
						y_sgd = np.asarray( y, dtype=float ).reshape( -1 )
						if X_sgd.ndim != 2 or X_sgd.shape[ 1 ] < 1:
							st.warning( '⚠️ Stochastic Gradient Descent requires at least '
							            'one numeric feature.' )
							st.stop( )
						
						if y_sgd.ndim != 1:
							st.warning( '⚠️ The Stochastic Gradient Descent target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_sgd ) != len( y_sgd ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_sgd ) < 2:
							st.warning( '⚠️ Stochastic Gradient Descent requires at least '
							            'two observations.' )
							st.stop( )
						
						if not np.isfinite( X_sgd ).all( ):
							st.warning( '⚠️ The Stochastic Gradient Descent feature matrix '
							            'contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_sgd ).all( ):
							st.warning( '⚠️ The Stochastic Gradient Descent target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_sgd ) ) < 2:
							st.warning( '⚠️ The regression target must contain at least '
							            'two distinct values.' )
							st.stop( )
						
						if sgd_alpha <= 0.0:
							st.warning( '⚠️ SGD alpha must be greater than zero.' )
							st.stop( )
						
						if sgd_iters < 1:
							st.warning( '⚠️ SGD iterations must be greater than zero.' )
							st.stop( )
						
						if sgd_tol < 0.0:
							st.warning( '⚠️ SGD tolerance cannot be negative.' )
							st.stop( )
						
						if sgd_eta0 <= 0.0:
							st.warning( '⚠️ SGD Eta0 must be greater than zero.' )
							st.stop( )
						
						if sgd_power_t < 0.0:
							st.warning( '⚠️ SGD Power T cannot be negative.' )
							st.stop( )
						
						if sgd_epsilon < 0.0:
							st.warning( '⚠️ SGD epsilon cannot be negative.' )
							st.stop( )
						
						if sgd_l1_ratio < 0.0 or sgd_l1_ratio > 1.0:
							st.warning( '⚠️ SGD L1 Ratio must be between 0.0 and 1.0.' )
							st.stop( )
						
						if (sgd_validation_fraction <= 0.0 or sgd_validation_fraction >= 1.0):
							st.warning( '⚠️ Validation Fraction must be greater than zero '
							            'and less than one.' )
							st.stop( )
						
						if sgd_n_iter_no_change < 1:
							st.warning( '⚠️ N Iter No Change must be greater than zero.' )
							st.stop( )
						
						st.session_state[ 'regression_sgd_loss' ] = str( sgd_loss )
						st.session_state[ 'regression_sgd_penalty' ] = sgd_penalty
						st.session_state[ 'regression_sgd_alpha' ] = float( sgd_alpha )
						st.session_state[ 'regression_sgd_iters' ] = int( sgd_iters )
						st.session_state[ 'regression_sgd_shuffle' ] = bool( sgd_shuffle )
						st.session_state[ 'regression_sgd_learning_rate' ] = str(
							sgd_learning_rate )
						st.session_state[ 'regression_sgd_l1_ratio' ] = float( sgd_l1_ratio )
						st.session_state[ 'regression_sgd_fit_intercept' ] = bool(
							sgd_fit_intercept )
						st.session_state[ 'regression_sgd_tol' ] = float( sgd_tol )
						st.session_state[ 'regression_sgd_verbose' ] = int( sgd_verbose )
						st.session_state[ 'regression_sgd_epsilon' ] = float( sgd_epsilon )
						st.session_state[ 'regression_sgd_eta0' ] = float( sgd_eta0 )
						st.session_state[ 'regression_sgd_power_t' ] = float( sgd_power_t )
						st.session_state[ 'regression_sgd_early_stopping' ] = bool(
							sgd_early_stopping )
						st.session_state[ 'regression_sgd_validation_fraction' ] = float(
							sgd_validation_fraction )
						st.session_state[ 'regression_sgd_n_iter_no_change' ] = int(
							sgd_n_iter_no_change )
						st.session_state[ 'regression_sgd_warm_start' ] = bool( sgd_warm_start )
						st.session_state[ 'regression_sgd_average' ] = bool( sgd_average )
						st.session_state[ 'regression_sgd_test_size' ] = float( sgd_test_size )
						st.session_state[ 'regression_sgd_random_state' ] = int( sgd_random_state )
						start_time = time.perf_counter( )
						
						model = regression_model.GradientDescent( loss=str( sgd_loss ),
							iters=int( sgd_iters ), penalty=sgd_penalty, alpha=float( sgd_alpha ),
							rando=int( sgd_random_state ), learning_rate=str( sgd_learning_rate ),
							l1_ratio=float( sgd_l1_ratio ), fit=bool( sgd_fit_intercept ),
							tol=float( sgd_tol ), shuffle=bool( sgd_shuffle ),
							verbose=int( sgd_verbose ), epsilon=float( sgd_epsilon ),
							eta0=float( sgd_eta0 ), power_t=float( sgd_power_t ),
							early_stopping=bool( sgd_early_stopping ),
							validation_fraction=float( sgd_validation_fraction ),
							n_iter_no_change=int( sgd_n_iter_no_change ),
							warm=bool( sgd_warm_start ), average=bool( sgd_average ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_sgd, y_sgd,
							size=float( sgd_test_size ), random=int( sgd_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						if sgd_early_stopping and len( X_train ) < 10:
							st.warning( '⚠️ Early stopping requires a larger training set. '
							            'Disable Early Stopping or use more observations.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Loss', 'Penalty', 'Learning Rate', 'Iterations',
								'Alpha' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), str( sgd_loss ),
								('None' if sgd_penalty is None else str( sgd_penalty )),
								str( sgd_learning_rate ), int( sgd_iters ), float( sgd_alpha ) ]
						} )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						coefficient_values = np.asarray( model.weights, dtype=float ).reshape( -1 )
						if len( coefficient_values ) == len( active_features ):
							coefficient_names = active_features.copy( )
						else:
							coefficient_names = [ f'Feature {index + 1}' for index in
								range( len( coefficient_values ) ) ]
						
						df_coefficients = pd.DataFrame(
							{ 'Feature': coefficient_names, 'Coefficient': coefficient_values } )
						
						st.session_state[ 'regression_sgd_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.session_state[ 'df_coefficients' ] = df_coefficients.copy( )
						
						st.success( 'Stochastic Gradient Descent training completed.' )
					
					except Exception as ex:
						st.error( f'Stochastic Gradient Descent training failed: {ex}' )
		
		with st.expander( 'Instance Models', expanded=False ):
			with st.expander( 'k-Nearest Neighbors', expanded=False ):
				knn_defaults = { 'regression_knn_neighbors': 5, 'regression_knn_weights':
					'uniform',
					'regression_knn_algorithm': 'auto', 'regression_knn_leaf_size': 30,
					'regression_knn_power': 2.0, 'regression_knn_metric': 'minkowski',
					'regression_knn_jobs': 1, 'regression_knn_test_size': 0.20,
					'regression_knn_random_state': 42 }
				
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
					
					knn_weights_options = [ 'uniform', 'distance' ]
					
					knn_weights_value = st.session_state.get( 'regression_knn_weights', 'uniform' )
					
					if knn_weights_value not in knn_weights_options:
						knn_weights_value = 'uniform'
					
					knn_weights = st.selectbox( 'Weights', options=knn_weights_options,
						index=knn_weights_options.index( knn_weights_value ),
						key='regression_knn_weights_select' )
					
					knn_power = float( st.number_input( 'Power', min_value=1.0,
						value=float( st.session_state[ 'regression_knn_power' ] ), step=1.0,
						format='%.1f', key='regression_knn_power_input' ) )
					
					knn_leaf_size = int( st.number_input( 'Leaf Size', min_value=1,
						value=int( st.session_state[ 'regression_knn_leaf_size' ] ), step=1,
						key='regression_knn_leaf_size_input' ) )
				
				with knn_c2:
					st.markdown( '###### Distance / Search' )
					
					knn_algorithm_options = [ 'auto', 'ball_tree', 'kd_tree', 'brute' ]
					
					knn_algorithm_value = st.session_state.get( 'regression_knn_algorithm',
						'auto' )
					
					if knn_algorithm_value not in knn_algorithm_options:
						knn_algorithm_value = 'auto'
					
					knn_algorithm = st.selectbox( 'Algorithm', options=knn_algorithm_options,
						index=knn_algorithm_options.index( knn_algorithm_value ),
						key='regression_knn_algorithm_select' )
					
					knn_metric_options = [ 'minkowski', 'euclidean', 'manhattan', 'chebyshev',
						'canberra', 'braycurtis', 'cityblock', 'cosine', 'l1', 'l2',
						'nan_euclidean', 'hamming' ]
					
					knn_metric_value = st.session_state.get( 'regression_knn_metric', 'minkowski' )
					
					if knn_metric_value not in knn_metric_options:
						knn_metric_value = 'minkowski'
					
					knn_metric = st.selectbox( 'Metric', options=knn_metric_options,
						index=knn_metric_options.index( knn_metric_value ),
						key='regression_knn_metric_select' )
					
					knn_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
						value=int( st.session_state[ 'regression_knn_jobs' ] ), step=1,
						key='regression_knn_jobs_input' ) )
					
					if knn_metric != 'minkowski':
						st.caption( 'Power is primarily used with the Minkowski metric.' )
				
				with knn_c3:
					st.markdown( '###### Data Split' )
					
					knn_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_knn_test_size' ] * 100 ), step=5,
						key='regression_knn_test_size_slider' ) / 100.0
					
					knn_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_knn_random_state' ] ), step=1,
						key='regression_knn_random_state_input' ) )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Target: {target_name}' )
				
				knn_btn_1, knn_btn_2 = st.columns( 2 )
				
				with knn_btn_1:
					train_knn = st.button( '🚂 Train k-Nearest Neighbors',
						key='regression_knn_train', use_container_width=True )
				
				with knn_btn_2:
					reset_knn = st.button( '🔄 Reset k-Nearest Neighbors',
						key='regression_knn_reset', use_container_width=True )
				
				if reset_knn:
					for key, value in knn_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_knn_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_knn:
					try:
						if X is None or y is None:
							st.warning( '⚠️ k-Nearest Neighbors requires prepared '
							            'feature and target arrays.' )
							st.stop( )
						
						X_knn = np.asarray( X, dtype=float )
						
						y_knn = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_knn.ndim != 2 or X_knn.shape[ 1 ] < 1:
							st.warning( '⚠️ k-Nearest Neighbors requires at least one '
							            'numeric feature.' )
							st.stop( )
						
						if y_knn.ndim != 1:
							st.warning( '⚠️ The k-Nearest Neighbors target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_knn ) != len( y_knn ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_knn ) < 2:
							st.warning( '⚠️ k-Nearest Neighbors requires at least two '
							            'observations.' )
							st.stop( )
						
						if not np.isfinite( X_knn ).all( ):
							st.warning( '⚠️ The k-Nearest Neighbors feature matrix '
							            'contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_knn ).all( ):
							st.warning( '⚠️ The k-Nearest Neighbors target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_knn ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						if knn_neighbors < 1:
							st.warning( '⚠️ The number of neighbors must be greater '
							            'than zero.' )
							st.stop( )
						
						if knn_leaf_size < 1:
							st.warning( '⚠️ Leaf Size must be greater than zero.' )
							st.stop( )
						
						if knn_power < 1.0:
							st.warning( '⚠️ Minkowski Power must be at least 1.0.' )
							st.stop( )
						
						if knn_jobs < 1:
							st.warning( '⚠️ Parallel Jobs must be greater than zero.' )
							st.stop( )
						
						st.session_state[ 'regression_knn_neighbors' ] = int( knn_neighbors )
						st.session_state[ 'regression_knn_weights' ] = str( knn_weights )
						st.session_state[ 'regression_knn_algorithm' ] = str( knn_algorithm )
						st.session_state[ 'regression_knn_leaf_size' ] = int( knn_leaf_size )
						st.session_state[ 'regression_knn_power' ] = float( knn_power )
						st.session_state[ 'regression_knn_metric' ] = str( knn_metric )
						st.session_state[ 'regression_knn_jobs' ] = int( knn_jobs )
						st.session_state[ 'regression_knn_test_size' ] = float( knn_test_size )
						st.session_state[ 'regression_knn_random_state' ] = int( knn_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.NearestNeighbor( num=int( knn_neighbors ),
							weight=str( knn_weights ), algo=str( knn_algorithm ),
							leaf=int( knn_leaf_size ), power=float( knn_power ),
							metric=str( knn_metric ), metric_params=None, jobs=int( knn_jobs ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_knn, y_knn,
							size=float( knn_test_size ), random=int( knn_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						if knn_neighbors > len( X_train ):
							st.warning( '⚠️ Neighbors cannot exceed the number of '
							            'training observations.' )
							st.stop( )
						
						model.train( X_train, y_train )
						
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Neighbors', 'Weights', 'Algorithm', 'Metric',
								'Power' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), int( knn_neighbors ), str( knn_weights ),
								str( knn_algorithm ), str( knn_metric ), float( knn_power ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_knn_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'k-Nearest Neighbors training completed.' )
					
					except Exception as ex:
						st.error( f'k-Nearest Neighbors training failed: {ex}' )
			
			with st.expander( 'Support Vector Machine', expanded=False ):
				svr_defaults = { 'regression_svr_kernel': 'rbf', 'regression_svr_degree': 3,
					'regression_svr_gamma_mode': 'scale', 'regression_svr_gamma_value': 0.1,
					'regression_svr_coef0': 0.0, 'regression_svr_tol': 0.001,
					'regression_svr_c': 1.0, 'regression_svr_epsilon': 0.1,
					'regression_svr_shrinking': True, 'regression_svr_cache_size': 200.0,
					'regression_svr_verbose': False, 'regression_svr_max_iter': -1,
					'regression_svr_test_size': 0.20, 'regression_svr_random_state': 42 }
				
				for key, value in svr_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Support Vector Regression for continuous targets.' )
				
				svr_c1, svr_c2, svr_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with svr_c1:
					st.markdown( '###### Kernel Parameters' )
					
					svr_kernel_options = [ 'linear', 'poly', 'rbf', 'sigmoid' ]
					
					svr_kernel_value = st.session_state.get( 'regression_svr_kernel', 'rbf' )
					
					if svr_kernel_value not in svr_kernel_options:
						svr_kernel_value = 'rbf'
					
					svr_kernel = st.selectbox( 'Kernel', options=svr_kernel_options,
						index=svr_kernel_options.index( svr_kernel_value ),
						key='regression_svr_kernel_select' )
					
					svr_degree = int( st.number_input( 'Degree', min_value=1,
						value=int( st.session_state[ 'regression_svr_degree' ] ), step=1,
						key='regression_svr_degree_input' ) )
					
					svr_gamma_options = [ 'scale', 'auto', 'custom' ]
					
					svr_gamma_mode_value = st.session_state.get( 'regression_svr_gamma_mode',
						'scale' )
					
					if svr_gamma_mode_value not in svr_gamma_options:
						svr_gamma_mode_value = 'scale'
					
					svr_gamma_mode = st.selectbox( 'Gamma', options=svr_gamma_options,
						index=svr_gamma_options.index( svr_gamma_mode_value ),
						key='regression_svr_gamma_mode_select' )
					
					svr_gamma_value = float( st.number_input( 'Gamma Value', min_value=0.000001,
						value=float( st.session_state[ 'regression_svr_gamma_value' ] ),
						step=0.010000, format='%.6f', key='regression_svr_gamma_value_input' ) )
					
					svr_coef0 = float( st.number_input( 'Coef0',
						value=float( st.session_state[ 'regression_svr_coef0' ] ), step=0.100000,
						format='%.6f', key='regression_svr_coef0_input' ) )
				
				with svr_c2:
					st.markdown( '###### Regularization / Solver' )
					
					svr_c = float( st.number_input( 'C', min_value=0.000001,
						value=float( st.session_state[ 'regression_svr_c' ] ), step=0.100000,
						format='%.6f', key='regression_svr_c_input' ) )
					
					svr_epsilon = float( st.number_input( 'Epsilon', min_value=0.0,
						value=float( st.session_state[ 'regression_svr_epsilon' ] ), step=0.010000,
						format='%.6f', key='regression_svr_epsilon_input' ) )
					
					svr_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_svr_tol' ] ), step=0.000100,
						format='%.6f', key='regression_svr_tol_input' ) )
					
					svr_shrinking = st.checkbox( 'Shrinking Heuristic',
						value=bool( st.session_state[ 'regression_svr_shrinking' ] ),
						key='regression_svr_shrinking_check' )
					
					svr_cache_size = float( st.number_input( 'Cache Size (MB)', min_value=1.0,
						value=float( st.session_state[ 'regression_svr_cache_size' ] ), step=10.0,
						format='%.1f', key='regression_svr_cache_size_input' ) )
					
					svr_verbose = st.checkbox( 'Verbose',
						value=bool( st.session_state[ 'regression_svr_verbose' ] ),
						key='regression_svr_verbose_check' )
					
					svr_max_iter = int(
						st.number_input( 'Max Iterations (-1 = No Limit)', min_value=-1,
							value=int( st.session_state[ 'regression_svr_max_iter' ] ), step=1,
							key='regression_svr_max_iter_input' ) )
				
				with svr_c3:
					st.markdown( '###### Data Split' )
					
					svr_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_svr_test_size' ] * 100 ), step=5,
						key='regression_svr_test_size_slider' ) / 100.0
					
					svr_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_svr_random_state' ] ), step=1,
						key='regression_svr_random_state_input' ) )
					
					if svr_kernel != 'poly':
						st.caption( 'Degree is only used when Kernel = poly.' )
					
					if svr_kernel not in [ 'poly', 'sigmoid' ]:
						st.caption( 'Coef0 is mainly used by poly and sigmoid kernels.' )
					
					if svr_gamma_mode != 'custom':
						st.caption( 'Gamma Value is only used when Gamma = custom.' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Target: {target_name}' )
				
				svr_btn_1, svr_btn_2 = st.columns( 2 )
				
				with svr_btn_1:
					train_svr = st.button( '🚂 Train Support Vector', key='regression_svr_train',
						use_container_width=True )
				
				with svr_btn_2:
					reset_svr = st.button( '🔄 Reset Support Vector', key='regression_svr_reset',
						use_container_width=True )
				
				if reset_svr:
					for key, value in svr_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_svr_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_svr:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Support Vector Regression requires prepared '
							            'feature and target arrays.' )
							st.stop( )
						
						X_svr = np.asarray( X, dtype=float )
						y_svr = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_svr.ndim != 2 or X_svr.shape[ 1 ] < 1:
							st.warning( '⚠️ Support Vector Regression requires at least '
							            'one numeric feature.' )
							st.stop( )
						
						if y_svr.ndim != 1:
							st.warning( '⚠️ The Support Vector Regression target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_svr ) != len( y_svr ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_svr ) < 2:
							st.warning( '⚠️ Support Vector Regression requires at least '
							            'two observations.' )
							st.stop( )
						
						if not np.isfinite( X_svr ).all( ):
							st.warning( '⚠️ The Support Vector Regression feature matrix '
							            'contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_svr ).all( ):
							st.warning( '⚠️ The Support Vector Regression target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_svr ) ) < 2:
							st.warning( '⚠️ The regression target must contain at least '
							            'two distinct values.' )
							st.stop( )
						
						if svr_c <= 0.0:
							st.warning( '⚠️ Support Vector C must be greater than zero.' )
							st.stop( )
						
						if svr_epsilon < 0.0:
							st.warning( '⚠️ Support Vector epsilon cannot be negative.' )
							st.stop( )
						
						if svr_tol < 0.0:
							st.warning( '⚠️ Support Vector tolerance cannot be negative.' )
							st.stop( )
						
						if svr_cache_size <= 0.0:
							st.warning( '⚠️ Support Vector cache size must be greater '
							            'than zero.' )
							st.stop( )
						
						if svr_max_iter == 0 or svr_max_iter < -1:
							st.warning( '⚠️ Max Iterations must be -1 or greater than zero.' )
							st.stop( )
						
						if svr_gamma_mode == 'custom' and svr_gamma_value <= 0.0:
							st.warning( '⚠️ Custom gamma must be greater than zero.' )
							st.stop( )
						
						effective_gamma = (
							float( svr_gamma_value ) if svr_gamma_mode == 'custom' else str(
								svr_gamma_mode ))
						
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
						
						start_time = time.perf_counter( )
						
						model = regression_model.SupportVector( kernel=str( svr_kernel ),
							degree=int( svr_degree ), gamma=effective_gamma,
							coef0=float( svr_coef0 ), tol=float( svr_tol ), penalty=float( svr_c ),
							epsilon=float( svr_epsilon ), shrinking=bool( svr_shrinking ),
							cache=float( svr_cache_size ), verbose=bool( svr_verbose ),
							iters=int( svr_max_iter ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_svr, y_svr,
							size=float( svr_test_size ), random=int( svr_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Kernel', 'C', 'Epsilon', 'Gamma',
								'Maximum Iterations' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), str( svr_kernel ), float( svr_c ),
								float( svr_epsilon ), effective_gamma, int( svr_max_iter ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_svr_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Support Vector Regression training completed.' )
					
					except Exception as ex:
						st.error( f'Support Vector Regression training failed: {ex}' )
		
		with st.expander( 'Tree Models', expanded=False ):
			with st.expander( 'Extra Trees Regressor', expanded=False ):
				extra_defaults = { 'regression_extra_estimators': 100,
					'regression_extra_criterion': 'squared_error',
					'regression_extra_max_depth_mode': 'none',
					'regression_extra_max_depth_value': 10,
					'regression_extra_max_features_mode': 'all',
					'regression_extra_max_features_value': 1.0, 'regression_extra_bootstrap':
						False,
					'regression_extra_oob_score': False, 'regression_extra_warm_start': False,
					'regression_extra_jobs': 1, 'regression_extra_verbose': 0,
					'regression_extra_test_size': 0.20, 'regression_extra_random_state': 42 }
				
				for key, value in extra_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Highly randomized tree ensemble for continuous targets.' )
				
				extra_c1, extra_c2, extra_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with extra_c1:
					st.markdown( '###### Forest Parameters' )
					
					extra_estimators = int( st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'regression_extra_estimators' ] ), step=1,
						key='regression_extra_estimators_input' ) )
					
					extra_criterion_options = [ 'squared_error', 'absolute_error', 'friedman_mse',
						'poisson' ]
					
					extra_criterion_value = st.session_state.get( 'regression_extra_criterion',
						'squared_error' )
					
					if extra_criterion_value not in extra_criterion_options:
						extra_criterion_value = 'squared_error'
					
					extra_criterion = st.selectbox( 'Criterion', options=extra_criterion_options,
						index=extra_criterion_options.index( extra_criterion_value ),
						key='regression_extra_criterion_select' )
					
					extra_depth_options = [ 'none', 'custom' ]
					
					extra_depth_value = st.session_state.get( 'regression_extra_max_depth_mode',
						'none' )
					
					if extra_depth_value not in extra_depth_options:
						extra_depth_value = 'none'
					
					extra_max_depth_mode = st.selectbox( 'Max Depth', options=extra_depth_options,
						index=extra_depth_options.index( extra_depth_value ),
						key='regression_extra_max_depth_mode_select' )
					
					extra_max_depth_value = int( st.number_input( 'Max Depth Value', min_value=1,
						value=int( st.session_state[ 'regression_extra_max_depth_value' ] ),
						step=1,
						key='regression_extra_max_depth_value_input' ) )
				
				with extra_c2:
					st.markdown( '###### Feature / Run Settings' )
					
					extra_feature_options = [ 'all', 'sqrt', 'log2', 'fraction' ]
					
					extra_feature_value = st.session_state.get(
						'regression_extra_max_features_mode', 'all' )
					
					if extra_feature_value not in extra_feature_options:
						extra_feature_value = 'all'
					
					extra_max_features_mode = st.selectbox( 'Max Features',
						options=extra_feature_options,
						index=extra_feature_options.index( extra_feature_value ),
						key='regression_extra_max_features_mode_select' )
					
					extra_max_features_value = float(
						st.slider( 'Max Features Fraction', min_value=0.10, max_value=1.00,
							value=float(
								st.session_state[ 'regression_extra_max_features_value' ] ),
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
						step=5,
						key='regression_extra_test_size_slider' ) / 100.0
					
					extra_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_extra_random_state' ] ), step=1,
						key='regression_extra_random_state_input' ) )
					
					if extra_oob_score and not extra_bootstrap:
						st.info( 'Out-of-bag scoring requires Bootstrap Samples.' )
					
					if extra_max_features_mode != 'fraction':
						st.caption( 'Max Features Fraction is only used when '
						            'Max Features = fraction.' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Target: {target_name}' )
				
				extra_btn_1, extra_btn_2 = st.columns( 2 )
				
				with extra_btn_1:
					train_extra = st.button( '🚂 Train Extra Trees Regressor',
						key='regression_extra_train', use_container_width=True )
				
				with extra_btn_2:
					reset_extra = st.button( '🔄 Reset Extra Trees Regressor',
						key='regression_extra_reset', use_container_width=True )
				
				if reset_extra:
					for key, value in extra_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_extra_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_extra:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Extra Trees Regressor requires prepared '
							            'feature and target arrays.' )
							st.stop( )
						
						X_extra = np.asarray( X, dtype=float )
						
						y_extra = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_extra.ndim != 2 or X_extra.shape[ 1 ] < 1:
							st.warning( '⚠️ Extra Trees Regressor requires at least '
							            'one numeric feature.' )
							st.stop( )
						
						if y_extra.ndim != 1:
							st.warning( '⚠️ The Extra Trees target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_extra ) != len( y_extra ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_extra ) < 2:
							st.warning( '⚠️ Extra Trees Regressor requires at least '
							            'two observations.' )
							st.stop( )
						
						if not np.isfinite( X_extra ).all( ):
							st.warning( '⚠️ The Extra Trees feature matrix contains '
							            'non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_extra ).all( ):
							st.warning( '⚠️ The Extra Trees target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_extra ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						if extra_estimators < 1:
							st.warning( '⚠️ Estimators must be greater than zero.' )
							st.stop( )
						
						if extra_jobs < 1:
							st.warning( '⚠️ Parallel Jobs must be greater than zero.' )
							st.stop( )
						
						if (extra_max_features_value <= 0.0 or extra_max_features_value > 1.0):
							st.warning( '⚠️ Max Features Fraction must be greater '
							            'than zero and no greater than one.' )
							st.stop( )
						
						if extra_criterion == 'poisson' and np.any( y_extra < 0.0 ):
							st.warning( '⚠️ Poisson criterion requires a '
							            'non-negative target.' )
							st.stop( )
						
						if extra_max_depth_mode == 'none':
							effective_depth = None
						else:
							effective_depth = int( extra_max_depth_value )
						
						if extra_max_features_mode == 'all':
							effective_features = 1.0
						elif extra_max_features_mode == 'sqrt':
							effective_features = 'sqrt'
						elif extra_max_features_mode == 'log2':
							effective_features = 'log2'
						else:
							effective_features = float( extra_max_features_value )
						
						effective_oob = (bool( extra_oob_score ) and bool( extra_bootstrap ))
						
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
						st.session_state[ 'regression_extra_warm_start' ] = bool(
							extra_warm_start )
						st.session_state[ 'regression_extra_jobs' ] = int( extra_jobs )
						st.session_state[ 'regression_extra_verbose' ] = int( extra_verbose )
						st.session_state[ 'regression_extra_test_size' ] = float( extra_test_size )
						st.session_state[ 'regression_extra_random_state' ] = int(
							extra_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.ExtraTreesModel(
							estimators=int( extra_estimators ), criterion=str( extra_criterion ),
							depth=effective_depth, features=effective_features,
							bootstrap=bool( extra_bootstrap ), oob_score=bool( effective_oob ),
							jobs=int( extra_jobs ), rando=int( extra_random_state ),
							verbose=int( extra_verbose ), warm=bool( extra_warm_start ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_extra, y_extra,
							size=float( extra_test_size ), random=int( extra_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Estimators', 'Criterion', 'Max Depth',
								'Max Features', 'Bootstrap' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), int( extra_estimators ),
								str( extra_criterion ),
								('None' if effective_depth is None else int( effective_depth )),
								str( effective_features ), bool( extra_bootstrap ) ] } )
						
						if (effective_oob and hasattr( model.model, 'oob_score_' )):
							df_oob = pd.DataFrame( { 'Metric': [ 'OOB Score' ],
								'Value': [ float( model.model.oob_score_ ) ] } )
							
							df_metadata = pd.concat( [ df_metadata, df_oob ], ignore_index=True )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_extra_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Extra Trees Regressor training completed.' )
					
					except Exception as ex:
						st.error( f'Extra Trees Regressor training failed: {ex}' )
			
			with st.expander( 'Random Forest', expanded=False ):
				rf_defaults = { 'regression_rf_estimators': 100,
					'regression_rf_criterion': 'squared_error',
					'regression_rf_max_depth_mode': 'none', 'regression_rf_max_depth_value': 10,
					'regression_rf_min_samples_split': 2, 'regression_rf_min_samples_leaf': 1,
					'regression_rf_min_weight_fraction_leaf': 0.0,
					'regression_rf_max_features_mode': 'all',
					'regression_rf_max_features_value': 1.0,
					'regression_rf_max_leaf_nodes_mode': 'none',
					'regression_rf_max_leaf_nodes_value': 31,
					'regression_rf_min_impurity_decrease': 0.0, 'regression_rf_bootstrap': True,
					'regression_rf_oob_score': False, 'regression_rf_jobs': 1,
					'regression_rf_verbose': 0, 'regression_rf_warm_start': False,
					'regression_rf_ccp_alpha': 0.0, 'regression_rf_max_samples_mode': 'all',
					'regression_rf_max_samples_value': 1.0, 'regression_rf_test_size': 0.20,
					'regression_rf_random_state': 42 }
				
				for key, value in rf_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Bootstrap random forest for continuous targets.' )
				
				rf_c1, rf_c2, rf_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with rf_c1:
					st.markdown( '###### Forest Parameters' )
					
					rf_estimators = int( st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'regression_rf_estimators' ] ), step=1,
						key='regression_rf_estimators_input' ) )
					
					rf_criterion_options = [ 'squared_error', 'absolute_error', 'friedman_mse',
						'poisson' ]
					
					rf_criterion_value = st.session_state.get( 'regression_rf_criterion',
						'squared_error' )
					
					if rf_criterion_value not in rf_criterion_options:
						rf_criterion_value = 'squared_error'
					
					rf_criterion = st.selectbox( 'Criterion', options=rf_criterion_options,
						index=rf_criterion_options.index( rf_criterion_value ),
						key='regression_rf_criterion_select' )
					
					rf_depth_options = [ 'none', 'custom' ]
					
					rf_depth_value = st.session_state.get( 'regression_rf_max_depth_mode', 'none' )
					
					if rf_depth_value not in rf_depth_options:
						rf_depth_value = 'none'
					
					rf_max_depth_mode = st.selectbox( 'Max Depth', options=rf_depth_options,
						index=rf_depth_options.index( rf_depth_value ),
						key='regression_rf_max_depth_mode_select' )
					
					rf_max_depth_value = int( st.number_input( 'Max Depth Value', min_value=1,
						value=int( st.session_state[ 'regression_rf_max_depth_value' ] ), step=1,
						key='regression_rf_max_depth_value_input' ) )
					
					rf_min_samples_split = int( st.number_input( 'Min Samples Split', min_value=2,
						value=int( st.session_state[ 'regression_rf_min_samples_split' ] ), step=1,
						key='regression_rf_min_samples_split_input' ) )
					
					rf_min_samples_leaf = int( st.number_input( 'Min Samples Leaf', min_value=1,
						value=int( st.session_state[ 'regression_rf_min_samples_leaf' ] ), step=1,
						key='regression_rf_min_samples_leaf_input' ) )
				
				with rf_c2:
					st.markdown( '###### Node / Feature Controls' )
					
					rf_min_weight_fraction_leaf = float(
						st.number_input( 'Min Weight Fraction Leaf', min_value=0.0, max_value=0.5,
							value=float(
								st.session_state[ 'regression_rf_min_weight_fraction_leaf' ] ),
							step=0.010000, format='%.6f',
							key='regression_rf_min_weight_fraction_leaf_input' ) )
					
					rf_feature_options = [ 'all', 'sqrt', 'log2', 'fraction' ]
					
					rf_feature_value = st.session_state.get( 'regression_rf_max_features_mode',
						'all' )
					
					if rf_feature_value not in rf_feature_options:
						rf_feature_value = 'all'
					
					rf_max_features_mode = st.selectbox( 'Max Features',
						options=rf_feature_options,
						index=rf_feature_options.index( rf_feature_value ),
						key='regression_rf_max_features_mode_select' )
					
					rf_max_features_value = float(
						st.slider( 'Max Features Fraction', min_value=0.10, max_value=1.00,
							value=float( st.session_state[ 'regression_rf_max_features_value' ] ),
							step=0.05, key='regression_rf_max_features_value_slider' ) )
					
					rf_leaf_node_options = [ 'none', 'custom' ]
					
					rf_leaf_node_value = st.session_state.get( 'regression_rf_max_leaf_nodes_mode',
						'none' )
					
					if rf_leaf_node_value not in rf_leaf_node_options:
						rf_leaf_node_value = 'none'
					
					rf_max_leaf_nodes_mode = st.selectbox( 'Max Leaf Nodes',
						options=rf_leaf_node_options,
						index=rf_leaf_node_options.index( rf_leaf_node_value ),
						key='regression_rf_max_leaf_nodes_mode_select' )
					
					rf_max_leaf_nodes_value = int(
						st.number_input( 'Max Leaf Nodes Value', min_value=2,
							value=int( st.session_state[ 'regression_rf_max_leaf_nodes_value' ] ),
							step=1, key='regression_rf_max_leaf_nodes_value_input' ) )
					
					rf_min_impurity_decrease = float(
						st.number_input( 'Min Impurity Decrease', min_value=0.0, value=float(
							st.session_state[ 'regression_rf_min_impurity_decrease' ] ),
							step=0.000100, format='%.6f',
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
						step=0.000100,
						format='%.6f', key='regression_rf_ccp_alpha_input' ) )
					
					rf_sample_options = [ 'all', 'fraction' ]
					
					rf_sample_value = st.session_state.get( 'regression_rf_max_samples_mode',
						'all' )
					
					if rf_sample_value not in rf_sample_options:
						rf_sample_value = 'all'
					
					rf_max_samples_mode = st.selectbox( 'Max Samples', options=rf_sample_options,
						index=rf_sample_options.index( rf_sample_value ),
						key='regression_rf_max_samples_mode_select' )
					
					rf_max_samples_value = float(
						st.slider( 'Max Samples Fraction', min_value=0.10, max_value=1.00,
							value=float( st.session_state[ 'regression_rf_max_samples_value' ] ),
							step=0.05, key='regression_rf_max_samples_value_slider' ) )
					
					rf_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_rf_test_size' ] * 100 ), step=5,
						key='regression_rf_test_size_slider' ) / 100.0
					
					rf_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_rf_random_state' ] ), step=1,
						key='regression_rf_random_state_input' ) )
					
					if rf_oob_score and not rf_bootstrap:
						st.info( 'Out-of-bag scoring requires Bootstrap Samples.' )
					
					if rf_max_features_mode != 'fraction':
						st.caption( 'Max Features Fraction is only used when '
						            'Max Features = fraction.' )
					
					if rf_max_samples_mode != 'fraction':
						st.caption( 'Max Samples Fraction is only used when '
						            'Max Samples = fraction.' )
				
				rf_btn_1, rf_btn_2 = st.columns( 2 )
				
				with rf_btn_1:
					train_rf = st.button( '🚂 Train Random Forest', key='regression_rf_train',
						use_container_width=True )
				
				with rf_btn_2:
					reset_rf = st.button( '🔄 Reset Random Forest', key='regression_rf_reset',
						use_container_width=True )
				
				if reset_rf:
					for key, value in rf_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_rf_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_rf:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Random Forest requires prepared feature and '
							            'target arrays.' )
							st.stop( )
						
						X_rf = np.asarray( X, dtype=float )
						y_rf = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_rf.ndim != 2 or X_rf.shape[ 1 ] < 1:
							st.warning( '⚠️ Random Forest requires at least one numeric feature.' )
							st.stop( )
						
						if len( X_rf ) != len( y_rf ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_rf ) < 2:
							st.warning( '⚠️ Random Forest requires at least two observations.' )
							st.stop( )
						
						if not np.isfinite( X_rf ).all( ):
							st.warning( '⚠️ The Random Forest feature matrix contains '
							            'non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_rf ).all( ):
							st.warning( '⚠️ The Random Forest target contains non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_rf ) ) < 2:
							st.warning( '⚠️ The target must contain at least two distinct '
							            'values.' )
							st.stop( )
						
						if rf_criterion == 'poisson' and np.any( y_rf < 0.0 ):
							st.warning( '⚠️ Poisson criterion requires a non-negative target.' )
							st.stop( )
						
						effective_depth = (
							None if rf_max_depth_mode == 'none' else int( rf_max_depth_value ))
						
						if rf_max_features_mode == 'all':
							effective_features = 1.0
						elif rf_max_features_mode == 'sqrt':
							effective_features = 'sqrt'
						elif rf_max_features_mode == 'log2':
							effective_features = 'log2'
						else:
							effective_features = float( rf_max_features_value )
						
						effective_leaf_nodes = (None if rf_max_leaf_nodes_mode == 'none' else int(
							rf_max_leaf_nodes_value ))
						
						effective_samples = (
							None if rf_max_samples_mode == 'all' else float(
								rf_max_samples_value ))
						
						effective_oob = bool( rf_oob_score ) and bool( rf_bootstrap )
						
						if not rf_bootstrap:
							effective_samples = None
						
						st.session_state[ 'regression_rf_estimators' ] = int( rf_estimators )
						st.session_state[ 'regression_rf_criterion' ] = str( rf_criterion )
						st.session_state[ 'regression_rf_max_depth_mode' ] = str(
							rf_max_depth_mode )
						st.session_state[ 'regression_rf_max_depth_value' ] = int(
							rf_max_depth_value )
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
						st.session_state[ 'regression_rf_random_state' ] = int( rf_random_state )
						
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
						
						X_train, X_test, y_train, y_test = model.split_data( X_rf, y_rf,
							size=float( rf_test_size ), random=int( rf_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce valid '
							            'training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Estimators', 'Criterion', 'Max Depth',
								'Max Features', 'Bootstrap' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), int( rf_estimators ), str( rf_criterion ),
								('None' if effective_depth is None else int( effective_depth )),
								str( effective_features ), bool( rf_bootstrap ) ] } )
						
						if effective_oob and hasattr( model.model, 'oob_score_' ):
							df_oob = pd.DataFrame( { 'Metric': [ 'OOB Score' ],
								'Value': [ float( model.model.oob_score_ ) ] } )
							
							df_metadata = pd.concat( [ df_metadata, df_oob ], ignore_index=True )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_rf_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Random Forest training completed.' )
					
					except Exception as ex:
						st.error( f'Random Forest training failed: {ex}' )
		
		with st.expander( 'Ensemble Models', expanded=False ):
			with st.expander( 'Adaptive Boosting', expanded=False ):
				ada_defaults = { 'regression_ada_estimators': 50,
					'regression_ada_learning_rate': 1.0, 'regression_ada_loss': 'linear',
					'regression_ada_test_size': 0.20, 'regression_ada_random_state': 42 }
				
				for key, value in ada_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'AdaBoost ensemble for continuous targets.' )
				
				ada_c1, ada_c2, ada_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with ada_c1:
					st.markdown( '###### Ensemble Parameters' )
					
					ada_estimators = int( st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'regression_ada_estimators' ] ), step=1,
						key='regression_ada_estimators_input' ) )
					
					ada_learning_rate = float( st.number_input( 'Learning Rate',
						min_value=0.000001,
						value=float( st.session_state[ 'regression_ada_learning_rate' ] ),
						step=0.100000, format='%.6f', key='regression_ada_learning_rate_input' ) )
					
					ada_loss_options = [ 'linear', 'square', 'exponential' ]
					
					ada_loss_value = st.session_state.get( 'regression_ada_loss', 'linear' )
					
					if ada_loss_value not in ada_loss_options:
						ada_loss_value = 'linear'
					
					ada_loss = st.selectbox( 'Loss', options=ada_loss_options,
						index=ada_loss_options.index( ada_loss_value ),
						key='regression_ada_loss_select' )
				
				with ada_c2:
					st.markdown( '###### Data Split' )
					
					ada_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_ada_test_size' ] * 100 ), step=5,
						key='regression_ada_test_size_slider' ) / 100.0
					
					ada_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_ada_random_state' ] ), step=1,
						key='regression_ada_random_state_input' ) )
				
				with ada_c3:
					st.markdown( '###### Context' )
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Target: {target_name}' )
				
				ada_btn_1, ada_btn_2 = st.columns( 2 )
				
				with ada_btn_1:
					train_ada = st.button( '🚂 Train Adaptive Boosting',
						key='regression_ada_train',
						use_container_width=True )
				
				with ada_btn_2:
					reset_ada = st.button( '🔄 Reset Adaptive Boosting',
						key='regression_ada_reset',
						use_container_width=True )
				
				if reset_ada:
					for key, value in ada_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_ada_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_ada:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Adaptive Boosting requires prepared feature '
							            'and target arrays.' )
							st.stop( )
						
						X_ada = np.asarray( X, dtype=float )
						y_ada = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_ada.ndim != 2 or X_ada.shape[ 1 ] < 1:
							st.warning( '⚠️ Adaptive Boosting requires at least one '
							            'numeric feature.' )
							st.stop( )
						
						if y_ada.ndim != 1:
							st.warning( '⚠️ The Adaptive Boosting target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_ada ) != len( y_ada ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_ada ) < 2:
							st.warning( '⚠️ Adaptive Boosting requires at least two '
							            'observations.' )
							st.stop( )
						
						if not np.isfinite( X_ada ).all( ):
							st.warning( '⚠️ The Adaptive Boosting feature matrix contains '
							            'non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_ada ).all( ):
							st.warning( '⚠️ The Adaptive Boosting target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_ada ) ) < 2:
							st.warning( '⚠️ The regression target must contain at least '
							            'two distinct values.' )
							st.stop( )
						
						if ada_estimators < 1:
							st.warning( '⚠️ Estimators must be greater than zero.' )
							st.stop( )
						
						if ada_learning_rate <= 0.0:
							st.warning( '⚠️ Learning Rate must be greater than zero.' )
							st.stop( )
						
						st.session_state[ 'regression_ada_estimators' ] = int( ada_estimators )
						st.session_state[ 'regression_ada_learning_rate' ] = float(
							ada_learning_rate )
						st.session_state[ 'regression_ada_loss' ] = str( ada_loss )
						st.session_state[ 'regression_ada_test_size' ] = float( ada_test_size )
						st.session_state[ 'regression_ada_random_state' ] = int( ada_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.AdaptiveBoost( estimators=int( ada_estimators ),
							rate=float( ada_learning_rate ), loss=str( ada_loss ),
							rando=int( ada_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_ada, y_ada,
							size=float( ada_test_size ), random=int( ada_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce valid '
							            'training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Estimators', 'Learning Rate', 'Loss' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), int( ada_estimators ),
								float( ada_learning_rate ), str( ada_loss ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_ada_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Adaptive Boosting training completed.' )
					
					except Exception as ex:
						st.error( f'Adaptive Boosting training failed: {ex}' )
			
			with st.expander( 'Gradient Boosting', expanded=False ):
				gb_defaults = { 'regression_gb_loss': 'squared_error',
					'regression_gb_learning_rate': 0.100000, 'regression_gb_estimators': 100,
					'regression_gb_subsample': 1.0, 'regression_gb_criterion': 'friedman_mse',
					'regression_gb_min_samples_split': 2, 'regression_gb_min_samples_leaf': 1,
					'regression_gb_min_weight_fraction_leaf': 0.0,
					'regression_gb_max_depth_mode': 'custom', 'regression_gb_max_depth_value': 3,
					'regression_gb_min_impurity_decrease': 0.0,
					'regression_gb_max_features_mode': 'none',
					'regression_gb_max_features_value': 1.0, 'regression_gb_alpha': 0.9,
					'regression_gb_verbose': 0, 'regression_gb_max_leaf_nodes_mode': 'none',
					'regression_gb_max_leaf_nodes_value': 31, 'regression_gb_warm_start': False,
					'regression_gb_validation_fraction': 0.1,
					'regression_gb_n_iter_no_change_mode': 'none',
					'regression_gb_n_iter_no_change_value': 5, 'regression_gb_tol': 0.000100,
					'regression_gb_ccp_alpha': 0.0, 'regression_gb_test_size': 0.20,
					'regression_gb_random_state': 42 }
				
				for key, value in gb_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Sequential tree boosting for continuous targets.' )
				
				gb_c1, gb_c2, gb_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with gb_c1:
					st.markdown( '###### Boosting Parameters' )
					
					gb_loss_options = [ 'squared_error', 'absolute_error', 'huber', 'quantile' ]
					
					gb_loss_value = st.session_state.get( 'regression_gb_loss', 'squared_error' )
					
					if gb_loss_value not in gb_loss_options:
						gb_loss_value = 'squared_error'
					
					gb_loss = st.selectbox( 'Loss', options=gb_loss_options,
						index=gb_loss_options.index( gb_loss_value ),
						key='regression_gb_loss_select' )
					
					gb_learning_rate = float( st.number_input( 'Learning Rate', min_value=0.000001,
						value=float( st.session_state[ 'regression_gb_learning_rate' ] ),
						step=0.010000, format='%.6f', key='regression_gb_learning_rate_input' ) )
					
					gb_estimators = int( st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'regression_gb_estimators' ] ), step=1,
						key='regression_gb_estimators_input' ) )
					
					gb_subsample = float( st.slider( 'Subsample', min_value=0.10, max_value=1.00,
						value=float( st.session_state[ 'regression_gb_subsample' ] ), step=0.05,
						key='regression_gb_subsample_slider' ) )
					
					gb_alpha = float( st.slider( 'Alpha', min_value=0.01, max_value=0.99,
						value=float( st.session_state[ 'regression_gb_alpha' ] ), step=0.01,
						key='regression_gb_alpha_slider' ) )
				
				with gb_c2:
					st.markdown( '###### Tree Parameters' )
					
					gb_criterion_options = [ 'friedman_mse', 'squared_error' ]
					
					gb_criterion_value = st.session_state.get( 'regression_gb_criterion',
						'friedman_mse' )
					
					if gb_criterion_value not in gb_criterion_options:
						gb_criterion_value = 'friedman_mse'
					
					gb_criterion = st.selectbox( 'Criterion', options=gb_criterion_options,
						index=gb_criterion_options.index( gb_criterion_value ),
						key='regression_gb_criterion_select' )
					
					gb_min_samples_split = int( st.number_input( 'Min Samples Split', min_value=2,
						value=int( st.session_state[ 'regression_gb_min_samples_split' ] ), step=1,
						key='regression_gb_min_samples_split_input' ) )
					
					gb_min_samples_leaf = int( st.number_input( 'Min Samples Leaf', min_value=1,
						value=int( st.session_state[ 'regression_gb_min_samples_leaf' ] ), step=1,
						key='regression_gb_min_samples_leaf_input' ) )
					
					gb_min_weight_fraction_leaf = float(
						st.number_input( 'Min Weight Fraction Leaf', min_value=0.0, max_value=0.5,
							value=float(
								st.session_state[ 'regression_gb_min_weight_fraction_leaf' ] ),
							step=0.010000, format='%.6f',
							key='regression_gb_min_weight_fraction_leaf_input' ) )
					
					gb_depth_options = [ 'none', 'custom' ]
					
					gb_depth_value = st.session_state.get( 'regression_gb_max_depth_mode',
						'custom' )
					
					if gb_depth_value not in gb_depth_options:
						gb_depth_value = 'custom'
					
					gb_max_depth_mode = st.selectbox( 'Max Depth', options=gb_depth_options,
						index=gb_depth_options.index( gb_depth_value ),
						key='regression_gb_max_depth_mode_select' )
					
					gb_max_depth_value = int( st.number_input( 'Max Depth Value', min_value=1,
						value=int( st.session_state[ 'regression_gb_max_depth_value' ] ), step=1,
						key='regression_gb_max_depth_value_input' ) )
					
					gb_min_impurity_decrease = float(
						st.number_input( 'Min Impurity Decrease', min_value=0.0, value=float(
							st.session_state[ 'regression_gb_min_impurity_decrease' ] ),
							step=0.000100, format='%.6f',
							key='regression_gb_min_impurity_decrease_input' ) )
				
				with gb_c3:
					st.markdown( '###### Feature / Run Controls' )
					
					gb_feature_options = [ 'none', 'sqrt', 'log2', 'fraction' ]
					
					gb_feature_value = st.session_state.get( 'regression_gb_max_features_mode',
						'none' )
					
					if gb_feature_value not in gb_feature_options:
						gb_feature_value = 'none'
					
					gb_max_features_mode = st.selectbox( 'Max Features',
						options=gb_feature_options,
						index=gb_feature_options.index( gb_feature_value ),
						key='regression_gb_max_features_mode_select' )
					
					gb_max_features_value = float(
						st.slider( 'Max Features Fraction', min_value=0.10, max_value=1.00,
							value=float( st.session_state[ 'regression_gb_max_features_value' ] ),
							step=0.05, key='regression_gb_max_features_value_slider' ) )
					
					gb_leaf_node_options = [ 'none', 'custom' ]
					
					gb_leaf_node_value = st.session_state.get( 'regression_gb_max_leaf_nodes_mode',
						'none' )
					
					if gb_leaf_node_value not in gb_leaf_node_options:
						gb_leaf_node_value = 'none'
					
					gb_max_leaf_nodes_mode = st.selectbox( 'Max Leaf Nodes',
						options=gb_leaf_node_options,
						index=gb_leaf_node_options.index( gb_leaf_node_value ),
						key='regression_gb_max_leaf_nodes_mode_select' )
					
					gb_max_leaf_nodes_value = int(
						st.number_input( 'Max Leaf Nodes Value', min_value=2,
							value=int( st.session_state[ 'regression_gb_max_leaf_nodes_value' ] ),
							step=1, key='regression_gb_max_leaf_nodes_value_input' ) )
					
					gb_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_gb_warm_start' ] ),
						key='regression_gb_warm_start_check' )
					
					gb_validation_fraction = float(
						st.slider( 'Validation Fraction', min_value=0.05, max_value=0.40,
							value=float( st.session_state[ 'regression_gb_validation_fraction' ] ),
							step=0.05, key='regression_gb_validation_fraction_slider' ) )
					
					gb_no_change_options = [ 'none', 'custom' ]
					
					gb_no_change_value = st.session_state.get(
						'regression_gb_n_iter_no_change_mode', 'none' )
					
					if gb_no_change_value not in gb_no_change_options:
						gb_no_change_value = 'none'
					
					gb_n_iter_no_change_mode = st.selectbox( 'N Iter No Change',
						options=gb_no_change_options,
						index=gb_no_change_options.index( gb_no_change_value ),
						key='regression_gb_n_iter_no_change_mode_select' )
					
					gb_n_iter_no_change_value = int(
						st.number_input( 'N Iter No Change Value', min_value=1,
							value=int( st.session_state[ 'regression_gb_n_iter_no_change_value'
							] ),
							step=1, key='regression_gb_n_iter_no_change_value_input' ) )
					
					gb_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_gb_tol' ] ), step=0.000100,
						format='%.6f', key='regression_gb_tol_input' ) )
					
					gb_ccp_alpha = float( st.number_input( 'CCP Alpha', min_value=0.0,
						value=float( st.session_state[ 'regression_gb_ccp_alpha' ] ),
						step=0.000100,
						format='%.6f', key='regression_gb_ccp_alpha_input' ) )
					
					gb_verbose = int( st.number_input( 'Verbose', min_value=0,
						value=int( st.session_state[ 'regression_gb_verbose' ] ), step=1,
						key='regression_gb_verbose_input' ) )
					
					gb_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_gb_test_size' ] * 100 ), step=5,
						key='regression_gb_test_size_slider' ) / 100.0
					
					gb_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_gb_random_state' ] ), step=1,
						key='regression_gb_random_state_input' ) )
					
					if gb_loss not in [ 'huber', 'quantile' ]:
						st.caption( 'Alpha is only used by huber and quantile losses.' )
					
					if gb_max_features_mode != 'fraction':
						st.caption( 'Max Features Fraction is only used when '
						            'Max Features = fraction.' )
				
				gb_btn_1, gb_btn_2 = st.columns( 2 )
				
				with gb_btn_1:
					train_gb = st.button( '🚂 Train Gradient Boosting', key='regression_gb_train',
						use_container_width=True )
				
				with gb_btn_2:
					reset_gb = st.button( '🔄 Reset Gradient Boosting', key='regression_gb_reset',
						use_container_width=True )
				
				if reset_gb:
					for key, value in gb_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_gb_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ] = None
					st.session_state[ 'X_train' ] = None
					st.session_state[ 'X_test' ] = None
					st.session_state[ 'y_train' ] = None
					st.session_state[ 'y_test' ] = None
					st.session_state[ 'y_prediction' ] = None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_gb:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Gradient Boosting requires prepared feature '
							            'and target arrays.' )
							st.stop( )
						
						X_gb = np.asarray( X, dtype=float )
						y_gb = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_gb.ndim != 2 or X_gb.shape[ 1 ] < 1:
							st.warning( '⚠️ Gradient Boosting requires at least one '
							            'numeric feature.' )
							st.stop( )
						
						if y_gb.ndim != 1:
							st.warning( '⚠️ The Gradient Boosting target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_gb ) != len( y_gb ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_gb ) < 2:
							st.warning( '⚠️ Gradient Boosting requires at least two '
							            'observations.' )
							st.stop( )
						
						if not np.isfinite( X_gb ).all( ):
							st.warning( '⚠️ The Gradient Boosting feature matrix contains '
							            'non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_gb ).all( ):
							st.warning( '⚠️ The Gradient Boosting target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_gb ) ) < 2:
							st.warning( '⚠️ The regression target must contain at least '
							            'two distinct values.' )
							st.stop( )
						
						if gb_learning_rate <= 0.0:
							st.warning( '⚠️ Learning Rate must be greater than zero.' )
							st.stop( )
						
						if gb_estimators < 1:
							st.warning( '⚠️ Estimators must be greater than zero.' )
							st.stop( )
						
						if gb_subsample <= 0.0 or gb_subsample > 1.0:
							st.warning( '⚠️ Subsample must be greater than zero and '
							            'no greater than one.' )
							st.stop( )
						
						if gb_tol < 0.0:
							st.warning( '⚠️ Tolerance cannot be negative.' )
							st.stop( )
						
						effective_depth = (
							None if gb_max_depth_mode == 'none' else int( gb_max_depth_value ))
						
						if gb_max_features_mode == 'none':
							effective_features = None
						elif gb_max_features_mode == 'sqrt':
							effective_features = 'sqrt'
						elif gb_max_features_mode == 'log2':
							effective_features = 'log2'
						else:
							effective_features = float( gb_max_features_value )
						
						effective_leaf_nodes = (None if gb_max_leaf_nodes_mode == 'none' else int(
							gb_max_leaf_nodes_value ))
						
						effective_no_change = (None if gb_n_iter_no_change_mode == 'none' else int(
							gb_n_iter_no_change_value ))
						
						st.session_state[ 'regression_gb_loss' ] = str( gb_loss )
						st.session_state[ 'regression_gb_learning_rate' ] = float(
							gb_learning_rate )
						st.session_state[ 'regression_gb_estimators' ] = int( gb_estimators )
						st.session_state[ 'regression_gb_subsample' ] = float( gb_subsample )
						st.session_state[ 'regression_gb_criterion' ] = str( gb_criterion )
						st.session_state[ 'regression_gb_min_samples_split' ] = int(
							gb_min_samples_split )
						st.session_state[ 'regression_gb_min_samples_leaf' ] = int(
							gb_min_samples_leaf )
						st.session_state[ 'regression_gb_min_weight_fraction_leaf' ] = float(
							gb_min_weight_fraction_leaf )
						st.session_state[ 'regression_gb_max_depth_mode' ] = str(
							gb_max_depth_mode )
						st.session_state[ 'regression_gb_max_depth_value' ] = int(
							gb_max_depth_value )
						st.session_state[ 'regression_gb_min_impurity_decrease' ] = float(
							gb_min_impurity_decrease )
						st.session_state[ 'regression_gb_max_features_mode' ] = str(
							gb_max_features_mode )
						st.session_state[ 'regression_gb_max_features_value' ] = float(
							gb_max_features_value )
						st.session_state[ 'regression_gb_alpha' ] = float( gb_alpha )
						st.session_state[ 'regression_gb_verbose' ] = int( gb_verbose )
						st.session_state[ 'regression_gb_max_leaf_nodes_mode' ] = str(
							gb_max_leaf_nodes_mode )
						st.session_state[ 'regression_gb_max_leaf_nodes_value' ] = int(
							gb_max_leaf_nodes_value )
						st.session_state[ 'regression_gb_warm_start' ] = bool( gb_warm_start )
						st.session_state[ 'regression_gb_validation_fraction' ] = float(
							gb_validation_fraction )
						st.session_state[ 'regression_gb_n_iter_no_change_mode' ] = str(
							gb_n_iter_no_change_mode )
						st.session_state[ 'regression_gb_n_iter_no_change_value' ] = int(
							gb_n_iter_no_change_value )
						st.session_state[ 'regression_gb_tol' ] = float( gb_tol )
						st.session_state[ 'regression_gb_ccp_alpha' ] = float( gb_ccp_alpha )
						st.session_state[ 'regression_gb_test_size' ] = float( gb_test_size )
						st.session_state[ 'regression_gb_random_state' ] = int( gb_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.GradientBoost( loss=str( gb_loss ),
							rate=float( gb_learning_rate ), estimators=int( gb_estimators ),
							subsample=float( gb_subsample ), criterion=str( gb_criterion ),
							split=int( gb_min_samples_split ), leaf=int( gb_min_samples_leaf ),
							weight_fraction=float( gb_min_weight_fraction_leaf ),
							depth=effective_depth, impurity=float( gb_min_impurity_decrease ),
							init=None, rando=int( gb_random_state ), features=effective_features,
							alpha=float( gb_alpha ), verbose=int( gb_verbose ),
							leaf_nodes=effective_leaf_nodes, warm=bool( gb_warm_start ),
							validation_fraction=float( gb_validation_fraction ),
							no_change=effective_no_change, tol=float( gb_tol ),
							ccp_alpha=float( gb_ccp_alpha ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_gb, y_gb,
							size=float( gb_test_size ), random=int( gb_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce valid '
							            'training and testing partitions.' )
							st.stop( )
						
						if effective_no_change is not None and len( X_train ) < 10:
							st.warning( '⚠️ Early stopping requires a larger training set. '
							            'Disable N Iter No Change or use more observations.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Estimators', 'Learning Rate', 'Loss', 'Subsample',
								'Criterion' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), int( gb_estimators ),
								float( gb_learning_rate ), str( gb_loss ), float( gb_subsample ),
								str( gb_criterion ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_gb_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Gradient Boosting training completed.' )
					
					except Exception as ex:
						st.error( f'Gradient Boosting training failed: {ex}' )
			
			with st.expander( 'Voting Regressor', expanded=False ):
				vote_defaults = { 'regression_vote_include_ols': True,
					'regression_vote_include_ridge': True, 'regression_vote_include_knn': True,
					'regression_vote_use_weights': False, 'regression_vote_weight_ols': 1.0,
					'regression_vote_weight_ridge': 1.0, 'regression_vote_weight_knn': 1.0,
					'regression_vote_jobs': 1, 'regression_vote_verbose': False,
					'regression_vote_test_percent': 20, 'regression_vote_test_size': 0.20,
					'regression_vote_random_state': 42 }
				
				for key, value in vote_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				def reset_voting_regressor( ) -> None:
					"""Reset Voting Regressor state.

						Purpose:
							Restores Voting Regressor controls and output dataframes to their
							default
							state without modifying dataset, feature, or target selections.

						Returns:
							None
					"""
					for key, value in vote_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_vote_elapsed_seconds' ] = None
				
				st.caption(
					'Average predictions from multiple regressors fit on the full dataset.' )
				
				vote_c1, vote_c2, vote_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with vote_c1:
					st.markdown( '###### Base Estimators' )
					
					vote_include_ols = st.checkbox( 'Ordinary Least Squares',
						key='regression_vote_include_ols' )
					
					vote_include_ridge = st.checkbox( 'Ridge Regression',
						key='regression_vote_include_ridge' )
					
					vote_include_knn = st.checkbox( 'k-Nearest Neighbors',
						key='regression_vote_include_knn' )
					
					st.caption( 'Select at least two base estimators.' )
				
				with vote_c2:
					st.markdown( '###### Weighting' )
					
					vote_use_weights = st.checkbox( 'Use Custom Weights',
						key='regression_vote_use_weights' )
					
					vote_weight_ols = float(
						st.number_input( 'OLS Weight', min_value=0.0, step=0.10, format='%.2f',
							key='regression_vote_weight_ols' ) )
					
					vote_weight_ridge = float(
						st.number_input( 'Ridge Weight', min_value=0.0, step=0.10, format='%.2f',
							key='regression_vote_weight_ridge' ) )
					
					vote_weight_knn = float(
						st.number_input( 'kNN Weight', min_value=0.0, step=0.10, format='%.2f',
							key='regression_vote_weight_knn' ) )
					
					st.caption( 'Weights are only applied when custom weighting is enabled.' )
				
				with vote_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					
					vote_jobs = int( st.number_input( 'Parallel Jobs', min_value=1, step=1,
						key='regression_vote_jobs' ) )
					
					vote_verbose = st.checkbox( 'Verbose', key='regression_vote_verbose' )
					
					vote_test_percent = int(
						st.slider( 'Test Set Size (%)', min_value=10, max_value=30, step=1,
							key='regression_vote_test_percent' ) )
					vote_test_size = float( vote_test_percent / 100.0 )
					
					vote_random_state = int( st.number_input( 'Random State', min_value=0, step=1,
						key='regression_vote_random_state' ) )
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Target: {target_name}' )
				
				vote_btn_1, vote_btn_2 = st.columns( 2 )
				with vote_btn_1:
					train_vote = st.button( '🚂 Train Voting Regressor',
						key='regression_vote_train',
						use_container_width=True )
				
				with vote_btn_2:
					st.button( '🔄 Reset Voting Regressor', key='regression_vote_reset',
						use_container_width=True, on_click=reset_voting_regressor )
				
				if train_vote:
					try:
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply( pd.to_numeric,
							errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric( df_training[ target_name ], errors='coerce' ).fillna(
							0.0 ).to_numpy( ).reshape( -1 )
						
						if len( X ) < 3:
							st.warning( '⚠️ Voting Regressor requires at least three rows.' )
							st.stop( )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two '
								'distinct '
								'values.' )
							st.stop( )
						
						estimated_training_rows = max( 1,
							int( np.floor( len( X ) * (1.0 - vote_test_size) ) ) )
						vote_neighbors = min( 5, estimated_training_rows )
						
						estimators = [ ]
						weights = [ ]
						
						if vote_include_ols:
							estimators.append(
								('least_squares', regression_model.skl.LinearRegression( )) )
							weights.append( float( vote_weight_ols ) )
						
						if vote_include_ridge:
							estimators.append( ('ridge', regression_model.skl.Ridge(
								random_state=int( vote_random_state ) )) )
							weights.append( float( vote_weight_ridge ) )
						
						if vote_include_knn:
							estimators.append( ('nearest_neighbor',
								regression_model.skn.KNeighborsRegressor(
									n_neighbors=vote_neighbors )) )
							weights.append( float( vote_weight_knn ) )
						
						if len( estimators ) < 2:
							st.warning(
								'⚠️ Voting Regressor requires at least two base estimators.' )
							st.stop( )
						
						if vote_use_weights and all( weight == 0.0 for weight in weights ):
							st.warning( '⚠️ At least one voting weight must be greater than '
							            'zero.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = regression_model.VotingModel( est=estimators,
							weights=weights if vote_use_weights else None, jobs=int( vote_jobs ),
							verbose=bool( vote_verbose ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( vote_test_size ), random=int( vote_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = np.asarray( model.project( X_test ), dtype=float ).reshape(
							-1 )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if df_scores is None:
							df_scores = pd.DataFrame( )
						else:
							df_scores = df_scores.copy( )
						
						if not df_scores.empty:
							df_extra = pd.DataFrame( {
								'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
									'Testing Rows', 'Estimator Count', 'Weighted Voting' ],
								'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
									int( len( X_test ) ), int( len( estimators ) ),
									bool( vote_use_weights ) ] } )
							
							df_scores = pd.concat( [ df_scores, df_extra ], ignore_index=True )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_vote_test_size' ] = vote_test_size
						
						st.session_state[ 'regression_vote_elapsed_seconds' ] = elapsed_seconds
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Voting Regressor training completed.' )
					except Exception as ex:
						st.error( f'Voting Regressor training failed: {ex}' )
			
			with st.expander( 'Stacking Regressor', expanded=False ):
				stack_defaults = { 'regression_stack_include_ols': True,
					'regression_stack_include_ridge': True, 'regression_stack_include_knn': True,
					'regression_stack_final_estimator': 'ridge',
					'regression_stack_cv_mode': 'default', 'regression_stack_cv_value': 5,
					'regression_stack_jobs': 1, 'regression_stack_passthrough': False,
					'regression_stack_verbose': 0, 'regression_stack_test_size': 20,
					'regression_stack_random_state': 42 }
				
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
					
					stack_final_estimator = st.selectbox( 'Final Estimator',
						options=[ 'linear_regression', 'ridge', 'knn' ],
						index=[ 'linear_regression', 'ridge', 'knn' ].index(
							st.session_state[ 'regression_stack_final_estimator' ] ),
						key='regression_stack_final_estimator' )
					
					stack_cv_mode = st.selectbox( 'Cross-Validation',
						options=[ 'default', 'custom' ], index=[ 'default', 'custom' ].index(
							st.session_state[ 'regression_stack_cv_mode' ] ),
						key='regression_stack_cv_mode' )
					
					stack_cv_value = int( st.number_input( 'CV Folds', min_value=2,
						value=int( st.session_state[ 'regression_stack_cv_value' ] ), step=1,
						key='regression_stack_cv_value' ) )
					
					stack_passthrough = st.checkbox( 'Passthrough Original Features',
						value=bool( st.session_state[ 'regression_stack_passthrough' ] ),
						key='regression_stack_passthrough' )
				
				with stack_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					
					stack_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
						value=int( st.session_state[ 'regression_stack_jobs' ] ), step=1,
						key='regression_stack_jobs' ) )
					
					stack_verbose = int( st.number_input( 'Verbose', min_value=0,
						value=int( st.session_state[ 'regression_stack_verbose' ] ), step=1,
						key='regression_stack_verbose' ) )
					
					stack_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'regression_stack_test_size' ] ), step=1,
						key='regression_stack_test_size' ) / 100.0
					
					stack_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_stack_random_state' ] ), step=1,
						key='regression_stack_random_state' ) )
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Target: {target_name}' )
				
				stack_btn_1, stack_btn_2 = st.columns( 2 )
				
				with stack_btn_1:
					train_stack = st.button( '🚂 Train Stacking Regressor',
						key='regression_stack_train', use_container_width=True )
				
				with stack_btn_2:
					reset_stack = st.button( '🔄 Reset Stacking Regressor',
						key='regression_stack_reset', use_container_width=True )
				
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
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply( pd.to_numeric,
							errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric( df_training[ target_name ], errors='coerce' ).fillna(
							0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct'
								' values.' )
							st.stop( )
						
						estimators = [ ]
						
						if stack_include_ols:
							estimators.append(
								('least_squares', regression_model.skl.LinearRegression( )) )
						
						if stack_include_ridge:
							estimators.append( ('ridge', regression_model.skl.Ridge(
								random_state=int( stack_random_state ) )) )
						
						if stack_include_knn:
							estimators.append(
								('nearest_neighbor', regression_model.skn.KNeighborsRegressor( )) )
						
						if len( estimators ) < 2:
							st.warning(
								'⚠️ Stacking Regressor requires at least two base estimators.' )
							st.stop( )
						
						if stack_final_estimator == 'linear_regression':
							final_estimator = regression_model.skl.LinearRegression( )
						elif stack_final_estimator == 'ridge':
							final_estimator = regression_model.skl.Ridge(
								random_state=int( stack_random_state ) )
						else:
							final_estimator = regression_model.skn.KNeighborsRegressor( )
						
						effective_cv = None if stack_cv_mode == 'default' else int(
							stack_cv_value )
						
						start_time = time.perf_counter( )
						
						model = regression_model.StackingModel( est=estimators,
							final=final_estimator, cv=effective_cv, jobs=int( stack_jobs ),
							passthrough=bool( stack_passthrough ), verbose=int( stack_verbose ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( stack_test_size ), random=int( stack_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							elif df_scores.shape[ 1 ] == 1:
								df_scores = df_scores.reset_index( )
								df_scores.columns = [ 'Metric', 'Value' ]
							
							df_extra = pd.DataFrame( {
								'Metric': [ 'Training Score', 'Testing Score',
									'Processing Time (Seconds)', 'Training Rows', 'Testing Rows',
									'Estimator Count', 'Passthrough' ],
								'Value': [ float( model.training_score ),
									float( model.testing_score ), round( elapsed_seconds, 4 ),
									int( len( X_train ) ), int( len( X_test ) ),
									int( len( estimators ) ), bool( stack_passthrough ) ] } )
							
							df_scores = pd.concat( [ df_scores, df_extra ], ignore_index=True )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': y_test, 'Predicted': y_prediction } )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Stacking Regressor training failed: {ex}' )
		
		model = st.session_state.get( 'model', None )
		X_test = st.session_state.get( 'X_test', None )
		y_test = st.session_state.get( 'y_test', None )
		
		if model is None or X_test is None or y_test is None:
			st.stop( )
		
		# ------------------------------------------------------------------
		# PREDICTIONS
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Predictions' )
		y_prediction = model.project( X_test )
		df_predictions = pd.DataFrame(
			{ 'Observed': y_test, 'Predicted': y_prediction, 'Residual': y_test - y_prediction } )
		
		st.session_state[ 'y_prediction' ] = y_prediction.copy( )
		st.session_state[ 'df_predictions' ] = df_predictions.copy( )
		
		st.data_editor( df_predictions, use_container_width=True )
		
		# ------------------------------------------------------------------
		# MODEL DETAILS
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Model Details' )
		detail_rows = [ ]
		
		if hasattr( model, 'features' ):
			try:
				detail_rows.append( { 'Property': 'Features', 'Value': model.features } )
			except Exception:
				pass
		
		if hasattr( model, 'training_score' ):
			try:
				detail_rows.append(
					{ 'Property': 'Training Score', 'Value': model.training_score } )
			except Exception:
				pass
		
		if hasattr( model, 'testing_score' ):
			try:
				detail_rows.append( { 'Property': 'Testing Score', 'Value': model.testing_score } )
			except Exception:
				pass
		
		if hasattr( model, 'weights' ):
			try:
				weights = model.weights
				if weights is not None:
					df_weights = pd.DataFrame(
						{ 'Feature': features, 'Weight': np.asarray( weights ).reshape( -1 ) } )
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
		blue_divider( )
		st.markdown( '##### Observed vs Predicted' )
		plt.close( 'all' )
		model.scatter_plot( X_test, y_test )
		st.pyplot( plt.gcf( ) )
		plt.close( 'all' )

# ============================================
# CLUSTERING MODELS MODE
# ============================================
elif mode == 'Clustering Models':
	df_dataset = st.session_state.get( 'df_dataset', None )
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Clustering Models' ] )
		st.divider( )
		
		if df_dataset is None or df_dataset.empty:
			st.warning( '⚠️ No dataset loaded.' )
			st.stop( )
		
		df_original = df_dataset.copy( )
		numeric_columns = [ column for column in df_original.columns if
			pd.api.types.is_numeric_dtype( df_original[ column ] ) ]
		categorical_columns = [ column for column in df_original.columns if
			column not in numeric_columns ]
		
		if not numeric_columns:
			st.warning( '⚠️ Clustering requires at least one numeric feature.' )
			st.stop( )
		
		cluster_source_signature = (tuple( df_original.columns.tolist( ) ),
			tuple( str( dtype ) for dtype in df_original.dtypes.tolist( ) ),
			int( len( df_original ) ),
			int( pd.util.hash_pandas_object( df_original, index=True ).sum( ) ))
		
		prior_cluster_source_signature = st.session_state.get( 'clustering_source_signature',
			None )
		
		if prior_cluster_source_signature != cluster_source_signature:
			cluster_state_keys = [ key for key in list( st.session_state.keys( ) ) if
				key.startswith( 'cluster_' ) or key in [ 'clusters_working_data', 'df_cluster',
					'df_cluster_results', 'df_cluster_counts', 'df_cluster_metrics',
					'df_cluster_centroids', 'df_cluster_details', 'df_cluster_ordering',
					'cluster_plot_features', 'cluster_signature' ] ]
			clear_keys( cluster_state_keys )
			st.session_state[ 'df_working' ] = df_original.copy( )
			st.session_state[ 'df_processed' ] = df_original.copy( )
			st.session_state[ 'features' ] = [ ]
			st.session_state[ 'targets' ] = [ ]
			st.session_state[ 'active_features' ] = [ ]
			st.session_state[ 'active_targets' ] = [ ]
		
		st.session_state[ 'clustering_source_signature' ] = cluster_source_signature
		st.session_state[ 'df_original' ] = df_original.copy( )
		st.session_state[ 'numeric_columns' ] = numeric_columns.copy( )
		st.session_state[ 'categorical_columns' ] = categorical_columns.copy( )
		df_working = st.session_state.get( 'df_working', pd.DataFrame( ) )
		df_processed = st.session_state.get( 'df_processed', pd.DataFrame( ) )
		df_cluster = st.session_state.get( 'df_cluster', pd.DataFrame( ) )
		features = st.session_state.get( 'features', [ ] )
		targets = st.session_state.get( 'targets', [ ] )
		if df_working is None or df_working.empty:
			df_working = df_original.copy( )
			st.session_state[ 'df_working' ] = df_working.copy( )
		
		if df_processed is None or df_processed.empty:
			df_processed = df_working.copy( )
			st.session_state[ 'df_processed' ] = df_processed.copy( )
		
		if df_cluster is None or df_cluster.empty:
			df_cluster = df_original.copy( )
			st.session_state[ 'df_cluster' ] = df_cluster.copy( )
		
		# ======================================================================================
		# Data Selection
		# ======================================================================================
		st.markdown( '##### Feature Selection' )
		st.caption( f'Inputs: {len( df_original ):,} | Features: {len( df_original.columns ):,}' )
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			features = st.multiselect( 'Select Features', options=df_working.columns )
		
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
				else:
					df_working = df_original.copy( )
				
				df_processed = df_working.copy( )
				st.session_state[ 'features' ] = features.copy( )
				st.session_state[ 'targets' ] = targets.copy( )
				st.session_state[ 'df_working' ] = df_working.copy( )
				st.session_state[ 'df_processed' ] = df_processed.copy( )
				commit_frame( df_working )
				st.success( 'Working Dataset Created!' )
		
		with sel_b2:
			if st.button( 'Reset To Original', icon='🔁', key='cluster_reset_to_original',
					use_container_width=True ):
				df_original = df_dataset.copy( )
				df_working = df_original.copy( )
				df_processed = df_working.copy( )
				st.session_state[ 'features' ] = [ ]
				st.session_state[ 'targets' ] = [ ]
				st.session_state[ 'df_working' ] = df_working.copy( )
				st.session_state[ 'df_processed' ] = df_processed.copy( )
				commit_frame( df_working )
				st.success( 'Reset to Original' )
		
		blue_divider( )
		st.markdown( '##### Working Data' )
		st.caption( f'Samples: {len( df_working ):,} | Features: {len( df_working.columns ):,}' )
		st.data_editor( df_working, key='clusters_working_data', disabled=True )
		
		# ------------------------------------------------------------------
		# Training Features
		# ------------------------------------------------------------------
		if df_working.empty:
			st.warning( '⚠️ No complete rows remain after preprocessing and feature selection.' )
			st.stop( )
		
		# -----------------------------------------------------------------
		# Data Processing
		# -----------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Feature-Engineering' )
		
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			with st.expander( label='Data Scaling', icon='⚖️', key='cluster_scalers' ):
				
				with st.expander( 'Standard Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.STANDARD_SCALER )
					scale_cols = st.multiselect( 'Columns', options=targets,
						key='cluster_standard_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_standard_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = StandardScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'Standard Scaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_standard_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Min-Max Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.MINMAX_SCALER )
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
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.ROBUST_SCALER )
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
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Normal Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.NORMAL_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ], index=1,
						key='cluster_normal_scaler_norm' )
					
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
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.MAXABS_SCALER )
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
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Data Imputation', icon='🧹', key='cluster_imputers' ):
				
				with st.expander( 'Mean Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.MEAN_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='cluster_mean_imputer_indicator' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_mean_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = MeanImputer( strategy='mean',
									add_indicator=add_indicator )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols, result,
									'mean_imputer' )
								
								commit_frame( df_processed )
								st.success( 'MeanImputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_mean_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.NEAREST_NEIGHBOR_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1, value=5, step=1,
						key='cluster_nearest_imputer_neighbors' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_nearest_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = NearestImputer( neighbors=int( neighbors ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols, result,
									'nearest_imputer' )
								
								commit_frame( df_processed )
								st.success( 'Nearest Imputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_nearest_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.ITERATIVE_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_iterative_imputer_cols' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=10, step=1,
						key='cluster_iterative_imputer_max_iter' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=0, step=1,
						key='cluster_iterative_imputer_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Iterative Imputer',
								key='cluster_iterative_imputer_apply', use_container_width=True ):
							if impute_cols:
								imputer = IterativeImputer( max_iter=int( max_iter ),
									random_state=int( random_state ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols, result,
									'iterative_imputer' )
								commit_frame( df_processed )
								st.success( 'Iterative Imputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_iterative_imputer_reset', use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.SIMPLE_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
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
									df_input = df_processed[ impute_cols ].apply( pd.to_numeric,
										errors='coerce' )
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
								df_processed = replace_columns( df_processed, impute_cols, result,
									'simple_imputer' )
								
								commit_frame( df_processed )
								st.success( 'Simple Imputer Applied' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_simple_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Data Encoding', icon='🔣', key='cluster_encoders' ):
				
				with st.expander( 'One-Hot Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.ONEHOT_ENCODER )
					encode_cols = st.multiselect( 'Columns', options=features,
						key='cluster_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='cluster_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0, key='cluster_onehot_unknown' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_onehot_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OneHotEncoder( sparse=bool( sparse ), unknown=unknown )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, encode_cols, result,
									'onehot' )
								commit_frame( df_processed )
								st.success( 'OneHotEncoder applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_onehot_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.ORDINAL_ENCODER )
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
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Label Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_ENCODER )
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
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Target Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.TARGET_ENCODER )
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
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.POLYNOMIAL_FEATURES )
					poly_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4, value=2,
						key='cluster_polynomial_degree' )
					
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
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
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
						if st.button( 'Apply Binarizer', key='cluster_binarizer_apply',
								use_container_width=True ):
							if transform_cols:
								df_processed = df_working.copy( )
								transformer = Binarizer( threshold=float( threshold ),
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
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column', options=categorical_columns,
						key='cluster_label_binarizer_col' )
					
					pos_label = st.number_input( 'Positive Label', value=1, step=1,
						key='cluster_label_binarizer_pos' )
					
					neg_label = st.number_input( 'Negative Label', value=0, step=1,
						key='cluster_label_binarizer_neg' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='cluster_label_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply LabelBinarizer', key='cluster_label_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ),
									sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, [ target_col ],
									result, 'label_binarizer' )
								commit_frame( df_processed )
								st.success( 'Label Binarizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_label_binarizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column', options=categorical_columns,
						key='cluster_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='cluster_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='cluster_multilabel_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_multilabel_binarizer_apply',
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
						if st.button( label='Reset', icon='🔁',
								key='cluster_multilabel_binarizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'TF-IDF Transformer', expanded=False ):
					text_count_cols = st.multiselect( 'Count Matrix Columns',
						options=numeric_columns, key='cluster_tfidf_transformer_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', None ], index=1,
						key='cluster_tfidf_transformer_norm' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='cluster_tfidf_transformer_use_idf' )
					
					smooth_idf = st.checkbox( 'Smooth IDF', value=True,
						key='cluster_tfidf_transformer_smooth_idf' )
					
					sublinear_tf = st.checkbox( 'Sublinear TF', value=False,
						key='cluster_tfidf_transformer_sublinear' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_tfidf_transformer_apply', use_container_width=True ):
							if text_count_cols:
								df_processed = df_working.copy( )
								transformer = TfidfTransformer( norm=norm, use_idf=bool( use_idf ),
									smooth_idf=bool( smooth_idf ),
									sublinear_tf=bool( sublinear_tf ) )
								
								result = transformer.train_transform(
									df_processed[ text_count_cols ].apply( pd.to_numeric,
										errors='coerce' ).fillna( 0.0 ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, text_count_cols,
									result, 'tfidf_transformer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Transformer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_tfidf_transformer_reset', use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Column Transformer', expanded=False ):
					numeric_columns = st.multiselect( 'Numeric Columns', options=numeric_columns,
						key='cluster_column_transformer_numeric_columns' )
					
					categorical_columns = st.multiselect( 'Categorical Columns',
						options=categorical_columns,
						key='cluster_column_transformer_categorical_columns' )
					
					numeric_transform = st.selectbox( 'Numeric Transformer',
						options=[ 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler',
							'Binarizer', 'None' ],
						key='cluster_column_transformer_numeric_transform' )
					
					categorical_transform = st.selectbox( 'Categorical Transformer',
						options=[ 'OneHotEncoder', 'OrdinalEncoder', 'None' ],
						key='cluster_column_transformer_categorical_transform' )
					
					remainder = st.selectbox( 'Remainder', options=[ 'drop', 'passthrough' ],
						key='cluster_column_transformer_remainder' )
					
					sparse_threshold = st.slider( 'Sparse Threshold', min_value=0.0, max_value=1.0,
						value=0.3, key='cluster_column_transformer_sparse_threshold' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Column Transformer',
								key='cluster_column_transformer_apply', use_container_width=True ):
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
								
								transformers.append(
									('categorical', categorical_model, categorical_columns) )
							
							if transformers:
								transformer = ColumnTransformer( transformers=transformers,
									remainder=remainder, sparse_threshold=float(
										sparse_threshold ),
									n_jobs=None, transformer_weights=None, verbose=False )
								
								result = transformer.train_transform( df_processed )
								df_processed = normalize_result_frame( result=result,
									index=df_processed.index, prefix='column_transformer',
									columns=None )
								
								commit_frame( df_processed )
								st.success( 'ColumnTransformer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_column_transformer_reset', use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Feature Extration', icon='⛏️', key='cluster_extractors' ):
				
				with st.expander( 'TF-IDF Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=categorical_columns,
						key='cluster_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='cluster_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='cluster_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='cluster_tfidf_vectorizer_use_idf' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_tfidf_vectorizer_apply', use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = TfidfVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), use_idf=bool( use_idf ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'tfidf_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Vectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_tfidf_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=categorical_columns,
						key='cluster_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='cluster_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='cluster_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='cluster_count_vectorizer_binary' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_count_vectorizer_apply', use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = CountVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), binary=bool( binary ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'count_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'Count Vectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_count_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=categorical_columns,
						key='cluster_hash_vectorizer_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='cluster_hash_vectorizer_n_features' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='cluster_hash_vectorizer_ngram_max' )
					
					binary = st.checkbox( 'Binary', value=False,
						key='cluster_hash_vectorizer_binary' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='cluster_hash_vectorizer_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_hash_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = HashVectorizer( num=int( n_features ),
									ngram_range=(1, int( ngram_max )), binary=bool( binary ),
									alternate_sign=bool( alternate_sign ) )
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'hash_vectorizer' )
								commit_frame( df_processed )
								st.success( 'HashVectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_hash_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					dict_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='cluster_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='cluster_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='cluster_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='cluster_dict_vectorizer_sort' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_dict_vectorizer_apply',
								use_container_width=True ):
							if dict_cols:
								df_processed = df_working.copy( )
								transformer = DictVectorizer( dtype=np.float64,
									separator=separator,
									sparse=bool( sparse ), sort=bool( sort ) )
								
								df_processed = apply_dict_transform( df_processed, dict_cols,
									transformer, 'dict_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'DictVectorizer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_dict_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					hash_cols = st.multiselect( 'Columns', options=categorical_columns,
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
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Dimensionality Reduction', icon='🎚️',
					key='cluster_selectors' ):
				
				with st.expander( 'Variance Threshold', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0, step=0.01,
						key='cluster_variance_threshold_value' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_variance_threshold_apply', use_container_width=True ):
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
						if st.button( label='Reset', icon='🔁',
								key='cluster_variance_threshold_reset', use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Canonical Correlation Analysis', expanded=False ):
					X_cols = st.multiselect( 'Predictor Columns', options=numeric_columns,
						key='cluster_cca_x_cols' )
					
					y_cols = st.multiselect( 'Target Columns', options=numeric_columns,
						key='cluster_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='cluster_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True, key='cluster_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500, step=1,
						key='cluster_cca_max_iter' )
					
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
									[ df_processed.drop( columns=X_cols + y_cols,
										errors='ignore' ),
										df_result ], axis=1 )
								
								commit_frame( df_processed )
								st.success( 'CCA Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_cca_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Principle Component Analysis', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_pca_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='cluster_pca_components' )
					
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
								
								df_processed = replace_columns( df_processed, select_cols, result,
									'pca' )
								commit_frame( df_processed )
								st.success( 'PCA applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_pca_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Select-Best', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_selectbest_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
							'mutual_info_regression' ], key='cluster_selectbest_score_name' )
					
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
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'select_best' )
								commit_frame( df_processed )
								st.success( 'Select Best Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_selectbest_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Select-Percent', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_selectpercent_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_selectpercent_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
							'mutual_info_regression' ], key='cluster_selectpercent_score_name' )
					
					percentile = st.slider( 'Percentile', min_value=1, max_value=100, value=10,
						key='cluster_selectpercent_percentile' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SelectPercent', key='cluster_selectpercent_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectPercent(
									score_func=score_function_from_name( score_name ),
									pct=int( percentile ) )
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'select_percent' )
								commit_frame( df_processed )
								st.success( 'SelectPercent applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_selectpercent_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Sequential Back Selection', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_sbs_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='cluster_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='cluster_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1, step=1,
						key='cluster_sbs_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_sbs_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SBS( classifier=None, k_features=int( k_features ),
									test_size=float( test_size ), random_state=int( random_state
									) )
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'sbs' )
								
								commit_frame( df_processed )
								st.success( 'SBS applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_sbs_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Recursive Feature Elimination', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_rfe_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='cluster_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0, step=1,
						key='cluster_rfe_verbose' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_rfe_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = RFE( k_features=int( k_features ),
									verbose=int( verbose ) )
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'rfe' )
								commit_frame( df_processed )
								st.success( 'RFE applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_rfe_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
		
		blue_divider( )
		st.markdown( '##### Processed Data' )
		st.caption(
			f'Samples: {len( df_processed ):,} | Features: {len( df_processed.columns ):,}' )
		st.data_editor( df_processed, key='cluster_processed_data' )
		
		# ------------------------------------------------------------------
		# MODEL INPUT
		# ------------------------------------------------------------------
		active_features = [ feature for feature in st.session_state.get( 'features', [ ] ) if
			feature in df_processed.columns and pd.api.types.is_numeric_dtype(
				df_processed[ feature ] ) ]
		
		if not active_features:
			active_features = [ column for column in df_processed.columns if
				pd.api.types.is_numeric_dtype(
					df_processed[ column ] ) and column not in st.session_state.get( 'targets',
					[ ] ) ]
		
		if not active_features:
			st.warning( '⚠️ Clustering requires at least one processed numeric feature column.' )
			st.stop( )
		
		feature_columns = active_features.copy( )
		df_cluster_input = df_processed[ feature_columns ].copy( )
		for column in feature_columns:
			df_cluster_input[ column ] = pd.to_numeric( df_cluster_input[ column ],
				errors='coerce' )
		
		df_cluster_input = df_cluster_input.replace( [ np.inf, -np.inf ], np.nan )
		if df_cluster_input.isna( ).any( ).any( ):
			st.warning(
				'⚠️ One or more clustering feature columns contain invalid numeric values.' )
			st.stop( )
		
		if len( df_cluster_input ) < 2:
			st.warning( '⚠️ Clustering requires at least two complete rows.' )
			st.stop( )
		
		X = df_cluster_input.to_numpy( dtype=float )
		cluster_data_signature = int( pd.util.hash_pandas_object( df_cluster_input, index=True ).sum( ) )
		st.session_state[ 'active_features' ] = active_features.copy( )
		st.session_state[ 'X_data' ] = df_cluster_input.copy( )
		
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
		
		cluster_max_clusters = max( 2, len( df_cluster_input ) - 1 )
		st.session_state[ 'cluster_kmeans_n_clusters' ] = min(
			max( 2, int( st.session_state[ 'cluster_kmeans_n_clusters' ] ) ),
			cluster_max_clusters )
		
		with st.expander( 'K-Means', expanded=True ):
			st.caption( 'Prototype-based clustering using centroid minimization.' )
			
			km_c1, km_c2, km_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with km_c1:
				n_clusters = st.number_input( 'Number of Clusters (K)', min_value=2,
					max_value=cluster_max_clusters, step=1, key='cluster_kmeans_n_clusters' )
				
				init = st.selectbox( 'Initialization', options=[ 'k-means++', 'random' ],
					key='cluster_kmeans_init' )
				
				algorithm = st.selectbox( 'Algorithm', options=[ 'lloyd', 'elkan' ],
					key='cluster_kmeans_algorithm' )
			
			with km_c2:
				n_init_mode = st.selectbox( 'Initialization Runs', options=[ 'auto', 'manual' ],
					key='cluster_kmeans_n_init_mode' )
				
				if n_init_mode == 'manual':
					n_init = int( st.number_input( 'Number of Initializations', min_value=1,
						step=1,
						key='cluster_kmeans_n_init_value' ) )
				else:
					n_init = 'auto'
					st.text_input( 'Number of Initializations', value='auto', disabled=True,
						key='cluster_kmeans_n_init_display' )
				
				max_iter = st.number_input( 'Maximum Iterations', min_value=1, step=1,
					key='cluster_kmeans_max_iter' )
			
			with km_c3:
				tol = st.number_input( 'Tolerance', min_value=0.0, step=0.0001, format='%.4f',
					key='cluster_kmeans_tol' )
				
				random_state = st.number_input( 'Random State', step=1,
					key='cluster_kmeans_random_state' )
				
				verbose = st.number_input( 'Verbose', min_value=0, step=1,
					key='cluster_kmeans_verbose' )
				
				copy_x = st.checkbox( 'Copy Input Data', key='cluster_kmeans_copy_x' )
			
			model = KMeans( clusters=int( n_clusters ), init=init, n_init=n_init, tol=float( tol ),
				rando=int( random_state ), max_iter=int( max_iter ), verbose=int( verbose ),
				copy_x=bool( copy_x ), algorithm=algorithm )
			
			model_parameters = { 'Model': 'K-Means', 'n_clusters': int( n_clusters ), 'init': init,
				'n_init': n_init, 'max_iter': int( max_iter ), 'tol': float( tol ),
				'random_state': int( random_state ), 'verbose': int( verbose ),
				'copy_x': bool( copy_x ), 'algorithm': algorithm }
			
			km_b1, km_b2 = st.columns( 2 )
			with km_b1:
				if st.button( 'Run K-Means', icon='🏃', key='cluster_kmeans_run',
						use_container_width=True ):
					cluster_kmeans_signature = ((tuple( active_features ), cluster_data_signature),
						'K-Means', tuple( (k, str( v )) for k, v in model_parameters.items( ) ))
					
					try:
						start_time = time.time( )
						labels = model.project( X )
						elapsed_seconds = time.time( ) - start_time
						df_results = df_cluster_input.copy( )
						df_results[ 'Cluster' ] = labels
						
						df_counts = ( df_results[ 'Cluster' ].value_counts(
							dropna=False ).rename_axis( 'Cluster' ).reset_index(
							name='Count' ).sort_values( by='Cluster' ).reset_index( drop=True ))
						
						try:
							df_metrics = model.score( X )
							if df_metrics is None:
								df_metrics = pd.DataFrame( )
						except Exception:
							df_metrics = pd.DataFrame( )
						
						if df_metrics is None or df_metrics.empty:
							df_metrics = pd.DataFrame(
								[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
						else:
							df_metrics = df_metrics.copy( )
							df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
						
						detail_rows = [ ]
						for prop in [ 'features', 'inertia', 'iterations', 'metric', 'algorithm',
							'n_clusters', 'n_init', 'max_iter', 'tolerance', 'random_state' ]:
							if hasattr( model, prop ):
								try:
									value = getattr( model, prop )
									if value is not None and not isinstance( value,
											(np.ndarray, pd.DataFrame) ):
										detail_rows.append( { 'Property': prop, 'Value': value } )
								except Exception:
									pass
						
						df_details = (
							pd.DataFrame( detail_rows ) if detail_rows else pd.DataFrame( ))
						df_centroids = pd.DataFrame( )
						if hasattr( model, 'centroids_' ):
							try:
								centroids = model.centroids_
								if centroids is not None:
									df_centroids = pd.DataFrame( centroids,
										columns=feature_columns )
									df_centroids.insert( 0, 'Cluster',
										range( len( df_centroids ) ) )
							except Exception:
								df_centroids = pd.DataFrame( )
						
						st.session_state[ 'df_cluster_kmeans_results' ] = df_results
						st.session_state[ 'df_cluster_kmeans_counts' ] = df_counts
						st.session_state[ 'df_cluster_kmeans_metrics' ] = df_metrics
						st.session_state[ 'df_cluster_kmeans_centroids' ] = df_centroids
						st.session_state[ 'df_cluster_kmeans_details' ] = df_details
						st.session_state[ 'cluster_kmeans_plot_features' ] = feature_columns.copy( )
						st.session_state[ 'cluster_kmeans_signature' ] = cluster_kmeans_signature
						st.success( 'K-Means clustering complete.' )
					except Exception as ex:
						st.session_state[ 'df_cluster_kmeans_results' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_kmeans_counts' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_kmeans_metrics' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_kmeans_centroids' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_kmeans_details' ] = pd.DataFrame( )
						st.session_state[ 'cluster_kmeans_plot_features' ] = [ ]
						st.session_state[ 'cluster_kmeans_signature' ] = None
						st.error( f'K-Means clustering failed: {ex}' )
			
			kmeans_reset_keys = [ 'cluster_kmeans_n_clusters', 'cluster_kmeans_init',
				'cluster_kmeans_n_init_mode', 'cluster_kmeans_n_init_value',
				'cluster_kmeans_n_init_display', 'cluster_kmeans_max_iter', 'cluster_kmeans_tol',
				'cluster_kmeans_random_state', 'cluster_kmeans_verbose', 'cluster_kmeans_copy_x',
				'cluster_kmeans_algorithm', 'df_cluster_kmeans_results',
				'df_cluster_kmeans_counts',
				'df_cluster_kmeans_metrics', 'df_cluster_kmeans_centroids',
				'df_cluster_kmeans_details', 'cluster_kmeans_plot_features',
				'cluster_kmeans_signature' ]
			
			with km_b2:
				st.button( 'Reset K-Means', icon='🔁', key='cluster_kmeans_reset',
					use_container_width=True, on_click=clear_keys, args=(kmeans_reset_keys,) )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
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
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			if df_results is not None and not df_results.empty:
				if len( feature_columns ) == 2:
					plt.close( 'all' )
					fig, ax = plt.subplots( )
					ax.scatter( df_results[ feature_columns[ 0 ] ],
						df_results[ feature_columns[ 1 ] ], c=df_results[ 'Cluster' ], alpha=0.7 )
					ax.set_xlabel( feature_columns[ 0 ] )
					ax.set_ylabel( feature_columns[ 1 ] )
					ax.set_title( 'Cluster Assignments' )
					if df_centroids is not None and not df_centroids.empty:
						try:
							ax.scatter( df_centroids[ feature_columns[ 0 ] ],
								df_centroids[ feature_columns[ 1 ] ], marker='x', s=100 )
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
				blue_divider( )
				st.markdown( '##### Cluster Centroids' )
				st.data_editor( df_centroids, use_container_width=True )
		
		if 'cluster_dbscan_eps' not in st.session_state:
			st.session_state[ 'cluster_dbscan_eps' ] = 0.5
		
		if 'cluster_dbscan_min_samples' not in st.session_state:
			st.session_state[ 'cluster_dbscan_min_samples' ] = 5
		
		if 'cluster_dbscan_metric' not in st.session_state:
			st.session_state[ 'cluster_dbscan_metric' ] = 'euclidean'
		
		if 'cluster_dbscan_algorithm' not in st.session_state:
			st.session_state[ 'cluster_dbscan_algorithm' ] = 'auto'
		
		if 'cluster_dbscan_leaf_size' not in st.session_state:
			st.session_state[ 'cluster_dbscan_leaf_size' ] = 30
		
		if 'cluster_dbscan_p' not in st.session_state:
			st.session_state[ 'cluster_dbscan_p' ] = 2.0
		
		if 'cluster_dbscan_n_jobs' not in st.session_state:
			st.session_state[ 'cluster_dbscan_n_jobs' ] = 1
		
		with st.expander( 'DBSCAN', expanded=False ):
			st.caption(
				'Density-based clustering that identifies dense regions and isolates noise.' )
			
			db_c1, db_c2, db_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with db_c1:
				epsilon = st.number_input( 'Neighborhood Radius', min_value=0.0001, step=0.1,
					format='%.4f', key='cluster_dbscan_eps' )
				
				min_samples = st.number_input( 'Minimum Samples', min_value=1, step=1,
					key='cluster_dbscan_min_samples' )
			
			with db_c2:
				metric = st.selectbox( 'Distance Metric',
					options=[ 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine' ],
					key='cluster_dbscan_metric' )
				
				algorithm = st.selectbox( 'Neighbor Algorithm',
					options=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ],
					key='cluster_dbscan_algorithm' )
			
			with db_c3:
				leaf_size = st.number_input( 'Leaf Size', min_value=1, step=1,
					key='cluster_dbscan_leaf_size' )
				
				p_value = st.number_input( 'Minkowski Power', min_value=1.0, step=1.0,
					format='%.1f', key='cluster_dbscan_p' )
				
				n_jobs = st.number_input( 'Parallel Jobs', min_value=-1, step=1,
					key='cluster_dbscan_n_jobs' )
			
			db_b1, db_b2 = st.columns( 2 )
			with db_b1:
				run_dbscan = st.button( 'Run DBSCAN', icon='🏃', key='cluster_dbscan_run',
					use_container_width=True )
			
			dbscan_reset_keys = [ 'cluster_dbscan_eps', 'cluster_dbscan_min_samples',
				'cluster_dbscan_metric', 'cluster_dbscan_algorithm', 'cluster_dbscan_leaf_size',
				'cluster_dbscan_p', 'cluster_dbscan_n_jobs', 'df_cluster_dbscan_results',
				'df_cluster_dbscan_counts', 'df_cluster_dbscan_metrics',
				'df_cluster_dbscan_centroids', 'df_cluster_dbscan_details',
				'cluster_dbscan_plot_features', 'cluster_dbscan_signature' ]
			
			with db_b2:
				st.button( 'Reset DBSCAN', icon='🔁', key='cluster_dbscan_reset',
					use_container_width=True, on_click=clear_keys, args=(dbscan_reset_keys,) )
			
			if run_dbscan:
				model_parameters = { 'Model': 'DBSCAN', 'eps': float( epsilon ),
					'min_samples': int( min_samples ), 'metric': metric, 'algorithm': algorithm,
					'leaf_size': int( leaf_size ), 'p': float( p_value ), 'n_jobs': int( n_jobs ) }
				
				cluster_dbscan_signature = ((tuple( active_features ), cluster_data_signature),
					'DBSCAN', tuple( (k, str( value )) for k, value in model_parameters.items( ) ))
				
				try:
					model = DBSCAN( eps=float( epsilon ), samples=int( min_samples ),
						metric=metric,
						metric_params=None, algorithm=algorithm, leaf_size=int( leaf_size ),
						p=float( p_value ), n_jobs=int( n_jobs ) )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					try:
						df_metrics = model.score( X )
						if df_metrics is None:
							df_metrics = pd.DataFrame( )
					except Exception:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					unique_labels = np.unique( labels )
					cluster_labels = unique_labels[ unique_labels != -1 ]
					noise_count = int( np.sum( labels == -1 ) )
					detail_rows = [ { 'Property': 'features', 'Value': int( model.features ) },
						{ 'Property': 'clusters', 'Value': int( len( cluster_labels ) ) },
						{ 'Property': 'noise_samples', 'Value': noise_count },
						{ 'Property': 'core_samples', 'Value': int( len( model.core_samples ) ) },
						{ 'Property': 'epsilon', 'Value': float( model.epsilon ) },
						{ 'Property': 'min_samples', 'Value': int( model.min_samples ) },
						{ 'Property': 'metric', 'Value': model.metric },
						{ 'Property': 'algorithm', 'Value': model.algorithm },
						{ 'Property': 'leaf_size', 'Value': int( model.leaf_size ) },
						{ 'Property': 'p', 'Value': model.p },
						{ 'Property': 'n_jobs', 'Value': model.n_jobs } ]
					
					df_details = pd.DataFrame( detail_rows )
					df_centroids = pd.DataFrame( )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_dbscan_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_dbscan_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_dbscan_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_dbscan_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_dbscan_details' ] = df_details.copy( )
					st.session_state[ 'cluster_dbscan_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_dbscan_signature' ] = cluster_dbscan_signature
					st.success( 'DBSCAN clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_dbscan_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_dbscan_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_dbscan_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_dbscan_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_dbscan_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_dbscan_plot_features' ] = [ ]
					st.session_state[ 'cluster_dbscan_signature' ] = None
					st.error( f'DBSCAN clustering failed: {ex}' )
			
			dbscan_signature = st.session_state.get( 'cluster_dbscan_signature', None )
			if (isinstance( dbscan_signature, tuple ) and len( dbscan_signature ) > 1 and
					dbscan_signature[ 1 ] == 'DBSCAN'):
				df_results = st.session_state.get( 'df_cluster_dbscan_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_dbscan_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_dbscan_metrics', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_dbscan_details', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				st.data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					st.data_editor( df_metrics, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True )
			else:
				st.info( 'Run DBSCAN to view cluster counts, noise counts, and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					plt.close( 'all' )
					fig, ax = plt.subplots( )
					ax.scatter( df_results[ feature_columns[ 0 ] ],
						df_results[ feature_columns[ 1 ] ], c=df_results[ 'Cluster' ], alpha=0.7 )
					
					ax.set_xlabel( feature_columns[ 0 ] )
					ax.set_ylabel( feature_columns[ 1 ] )
					ax.set_title( 'DBSCAN Cluster Assignments' )
					st.pyplot( fig )
					plt.close( fig )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run DBSCAN to view the scatter plot.' )
		
		if 'cluster_agglomerative_n_clusters' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_n_clusters' ] = 2
		
		if 'cluster_agglomerative_metric' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_metric' ] = 'euclidean'
		
		if 'cluster_agglomerative_linkage' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_linkage' ] = 'ward'
		
		if 'cluster_agglomerative_compute_full_tree' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_compute_full_tree' ] = 'auto'
		
		if 'cluster_agglomerative_use_distance_threshold' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_use_distance_threshold' ] = False
		
		if 'cluster_agglomerative_distance_threshold' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_distance_threshold' ] = 0.0
		
		if 'cluster_agglomerative_compute_distances' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_compute_distances' ] = False
		
		cluster_max_clusters = max( 2, len( df_cluster_input ) - 1 )
		if 'cluster_agglomerative_n_clusters' in st.session_state:
			st.session_state[ 'cluster_agglomerative_n_clusters' ] = min(
				max( 2, int( st.session_state[ 'cluster_agglomerative_n_clusters' ] ) ),
				cluster_max_clusters )
		
		with st.expander( 'Agglomerative Clustering', expanded=False ):
			st.caption( 'Hierarchical bottom-up clustering that successively merges related '
			            'observations.' )
			
			ag_c1, ag_c2, ag_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with ag_c1:
				agglomerative_clusters = int( st.number_input( 'Number of Clusters', min_value=2,
					max_value=max( 2, len( df_cluster_input ) - 1 ), step=1,
					key='cluster_agglomerative_n_clusters' ) )
				
				agglomerative_linkage = st.selectbox( 'Linkage',
					options=[ 'ward', 'complete', 'average', 'single' ],
					key='cluster_agglomerative_linkage' )
			
			with ag_c2:
				agglomerative_metric = st.selectbox( 'Distance Metric',
					options=[ 'euclidean', 'manhattan', 'cosine', 'l1', 'l2' ],
					key='cluster_agglomerative_metric' )
				
				agglomerative_compute_full_tree = st.selectbox( 'Compute Full Tree',
					options=[ 'auto', True, False ],
					key='cluster_agglomerative_compute_full_tree' )
			
			with ag_c3:
				agglomerative_use_distance_threshold = st.checkbox( 'Use Distance Threshold',
					key='cluster_agglomerative_use_distance_threshold' )
				
				agglomerative_distance_threshold = float(
					st.number_input( 'Distance Threshold', min_value=0.0, step=0.1, format='%.4f',
						disabled=not agglomerative_use_distance_threshold,
						key='cluster_agglomerative_distance_threshold' ) )
				
				agglomerative_compute_distances = st.checkbox( 'Compute Merge Distances',
					key='cluster_agglomerative_compute_distances' )
			
			ag_b1, ag_b2 = st.columns( 2 )
			
			with ag_b1:
				run_agglomerative = st.button( 'Run Agglomerative Clustering', icon='🏃',
					key='cluster_agglomerative_run', use_container_width=True )
			
			agglomerative_reset_keys = [ 'cluster_agglomerative_n_clusters',
				'cluster_agglomerative_metric', 'cluster_agglomerative_linkage',
				'cluster_agglomerative_compute_full_tree',
				'cluster_agglomerative_use_distance_threshold',
				'cluster_agglomerative_distance_threshold',
				'cluster_agglomerative_compute_distances', 'df_cluster_agglomerative_results',
				'df_cluster_agglomerative_counts', 'df_cluster_agglomerative_metrics',
				'df_cluster_agglomerative_centroids', 'df_cluster_agglomerative_details',
				'cluster_agglomerative_plot_features', 'cluster_agglomerative_signature' ]
			
			with ag_b2:
				st.button( 'Reset Agglomerative Clustering', icon='🔁',
					key='cluster_agglomerative_reset', use_container_width=True,
					on_click=clear_keys, args=(agglomerative_reset_keys,) )
			
			if run_agglomerative:
				try:
					if (agglomerative_linkage == 'ward' and agglomerative_metric != 'euclidean'):
						st.warning( '⚠️ Ward linkage requires the Euclidean distance metric.' )
						st.stop( )
					
					if ( agglomerative_use_distance_threshold and
							agglomerative_distance_threshold <= 0.0):
						st.warning(
							'⚠️ Distance Threshold must be greater than zero when enabled.' )
						st.stop( )
					
					distance_threshold = (float(
						agglomerative_distance_threshold ) if agglomerative_use_distance_threshold
					                      else None)
					
					model_parameters = { 'Model': 'Agglomerative', 'n_clusters': (
						None if distance_threshold is not None else int( agglomerative_clusters )),
						'metric': agglomerative_metric, 'linkage': agglomerative_linkage,
						'compute_full_tree': agglomerative_compute_full_tree,
						'distance_threshold': distance_threshold,
						'compute_distances': bool( agglomerative_compute_distances ) }
					
					cluster_agglomerative_signature = (
						(tuple( active_features ), cluster_data_signature), 'Agglomerative',
						tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = Agglomerative( clusters=int( agglomerative_clusters ),
						affinity=agglomerative_metric, linkage=agglomerative_linkage,
						metric=agglomerative_metric, memory=None, connectivity=None,
						compute_full_tree=agglomerative_compute_full_tree,
						distance_threshold=distance_threshold,
						compute_distances=bool( agglomerative_compute_distances ), n_clusters=(
							None if distance_threshold is not None else int(
								agglomerative_clusters )) )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					try:
						df_metrics = model.score( X )
						if df_metrics is None:
							df_metrics = pd.DataFrame( )
					except Exception:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					detail_rows = [ ]
					for property_name in [ 'features', 'leaves', 'connected_components',
						'n_clusters', 'metric', 'linkage', 'compute_full_tree',
						'distance_threshold', 'compute_distances' ]:
						if hasattr( model, property_name ):
							try:
								property_value = getattr( model, property_name )
								
								if (property_value is not None and not isinstance( property_value,
										(np.ndarray, pd.DataFrame) )):
									detail_rows.append(
										{ 'Property': property_name, 'Value': property_value } )
							except Exception:
								pass
					
					df_details = (pd.DataFrame( detail_rows ) if detail_rows else pd.DataFrame( ))
					df_centroids = pd.DataFrame( )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_agglomerative_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_agglomerative_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_agglomerative_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_agglomerative_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_agglomerative_details' ] = df_details.copy( )
					st.session_state[ 'cluster_agglomerative_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_agglomerative_signature' ] = cluster_agglomerative_signature
					st.success( 'Agglomerative clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_agglomerative_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_agglomerative_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_agglomerative_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_agglomerative_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_agglomerative_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_agglomerative_plot_features' ] = [ ]
					st.session_state[ 'cluster_agglomerative_signature' ] = None
					st.error( f'Agglomerative clustering failed: {ex}' )
			
			agglomerative_signature = st.session_state.get( 'cluster_agglomerative_signature',
				None )
			
			if (isinstance( agglomerative_signature, tuple ) and len(
					agglomerative_signature ) > 1 and agglomerative_signature[ 1 ] == 'Agglomerative'):
				df_results = st.session_state.get( 'df_cluster_agglomerative_results',
					pd.DataFrame( ) )
				
				df_counts = st.session_state.get( 'df_cluster_agglomerative_counts',
					pd.DataFrame( ) )
				
				df_metrics = st.session_state.get( 'df_cluster_agglomerative_metrics',
					pd.DataFrame( ) )
				
				df_details = st.session_state.get( 'df_cluster_agglomerative_details',
					pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				st.data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					st.data_editor( df_metrics, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True )
			else:
				st.info( 'Run Agglomerative Clustering to view cluster counts and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					plt.close( 'all' )
					fig, ax = plt.subplots( )
					
					ax.scatter( df_results[ feature_columns[ 0 ] ],
						df_results[ feature_columns[ 1 ] ], c=df_results[ 'Cluster' ], alpha=0.7 )
					
					ax.set_xlabel( feature_columns[ 0 ] )
					ax.set_ylabel( feature_columns[ 1 ] )
					ax.set_title( 'Agglomerative Cluster Assignments' )
					st.pyplot( fig )
					plt.close( fig )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run Agglomerative Clustering to view the scatter plot.' )
		
		if 'cluster_spectral_n_clusters' not in st.session_state:
			st.session_state[ 'cluster_spectral_n_clusters' ] = 2
		
		if 'cluster_spectral_affinity' not in st.session_state:
			st.session_state[ 'cluster_spectral_affinity' ] = 'rbf'
		
		if 'cluster_spectral_gamma' not in st.session_state:
			st.session_state[ 'cluster_spectral_gamma' ] = 1.0
		
		if 'cluster_spectral_n_neighbors' not in st.session_state:
			st.session_state[ 'cluster_spectral_n_neighbors' ] = 10
		
		if 'cluster_spectral_eigen_solver' not in st.session_state:
			st.session_state[ 'cluster_spectral_eigen_solver' ] = 'None'
		
		if 'cluster_spectral_n_components' not in st.session_state:
			st.session_state[ 'cluster_spectral_n_components' ] = 0
		
		if 'cluster_spectral_n_init' not in st.session_state:
			st.session_state[ 'cluster_spectral_n_init' ] = 10
		
		if 'cluster_spectral_eigen_tolerance' not in st.session_state:
			st.session_state[ 'cluster_spectral_eigen_tolerance' ] = 'auto'
		
		if 'cluster_spectral_assign_labels' not in st.session_state:
			st.session_state[ 'cluster_spectral_assign_labels' ] = 'kmeans'
		
		if 'cluster_spectral_degree' not in st.session_state:
			st.session_state[ 'cluster_spectral_degree' ] = 3.0
		
		if 'cluster_spectral_coef0' not in st.session_state:
			st.session_state[ 'cluster_spectral_coef0' ] = 1.0
		
		if 'cluster_spectral_n_jobs' not in st.session_state:
			st.session_state[ 'cluster_spectral_n_jobs' ] = 1
		
		if 'cluster_spectral_verbose' not in st.session_state:
			st.session_state[ 'cluster_spectral_verbose' ] = False
		
		if 'cluster_spectral_random_state' not in st.session_state:
			st.session_state[ 'cluster_spectral_random_state' ] = 42
		
		cluster_max_clusters = max( 2, len( df_cluster_input ) - 1 )
		if 'cluster_spectral_n_clusters' in st.session_state:
			st.session_state[ 'cluster_spectral_n_clusters' ] = min(
				max( 2, int( st.session_state[ 'cluster_spectral_n_clusters' ] ) ),
				cluster_max_clusters )
		
		with st.expander( 'Spectral Clustering', expanded=False ):
			st.caption( 'Graph-based clustering for identifying nonlinear cluster structures.' )
			
			sp_c1, sp_c2, sp_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with sp_c1:
				spectral_clusters = int( st.number_input( 'Number of Clusters', min_value=2,
					max_value=max( 2, len( df_cluster_input ) - 1 ), step=1,
					key='cluster_spectral_n_clusters' ) )
				
				spectral_affinity = st.selectbox( 'Affinity',
					options=[ 'rbf', 'nearest_neighbors', 'polynomial', 'sigmoid', 'cosine' ],
					key='cluster_spectral_affinity' )
				
				spectral_gamma = float(
					st.number_input( 'Gamma', min_value=0.0001, step=0.1, format='%.4f',
						key='cluster_spectral_gamma' ) )
				
				spectral_neighbors = int( st.number_input( 'Nearest Neighbors', min_value=1,
					max_value=max( 1, len( df_cluster_input ) - 1 ), step=1,
					key='cluster_spectral_n_neighbors' ) )
			
			with sp_c2:
				spectral_eigen_solver_display = st.selectbox( 'Eigen Solver',
					options=[ 'None', 'arpack', 'lobpcg' ], key='cluster_spectral_eigen_solver' )
				
				spectral_components = int(
					st.number_input( 'Embedding Components (0 = Default)', min_value=0,
						max_value=max( 1, len( df_cluster_input ) - 1 ), step=1,
						key='cluster_spectral_n_components' ) )
				
				spectral_n_init = int(
					st.number_input( 'K-Means Initializations', min_value=1, step=1,
						key='cluster_spectral_n_init' ) )
				
				spectral_eigen_tolerance = st.selectbox( 'Eigen Tolerance',
					options=[ 'auto', 0.0, 0.0001, 0.001, 0.01 ],
					key='cluster_spectral_eigen_tolerance' )
			
			with sp_c3:
				spectral_assign_labels = st.selectbox( 'Label Assignment',
					options=[ 'kmeans', 'discretize', 'cluster_qr' ],
					key='cluster_spectral_assign_labels' )
				
				spectral_degree = float(
					st.number_input( 'Polynomial Degree', min_value=1.0, step=1.0, format='%.1f',
						key='cluster_spectral_degree' ) )
				
				spectral_coef0 = float(
					st.number_input( 'Kernel Coefficient', step=0.1, format='%.4f',
						key='cluster_spectral_coef0' ) )
				
				spectral_jobs = int( st.number_input( 'Parallel Jobs', min_value=-1, step=1,
					key='cluster_spectral_n_jobs' ) )
				
				spectral_verbose = st.checkbox( 'Verbose', key='cluster_spectral_verbose' )
				
				spectral_random_state = int( st.number_input( 'Random State', min_value=0, step=1,
					key='cluster_spectral_random_state' ) )
			
			sp_b1, sp_b2 = st.columns( 2 )
			with sp_b1:
				run_spectral = st.button( 'Run Spectral Clustering', icon='🏃',
					key='cluster_spectral_run', use_container_width=True )
			
			spectral_reset_keys = [ 'cluster_spectral_n_clusters', 'cluster_spectral_affinity',
				'cluster_spectral_gamma', 'cluster_spectral_n_neighbors',
				'cluster_spectral_eigen_solver', 'cluster_spectral_n_components',
				'cluster_spectral_n_init', 'cluster_spectral_eigen_tolerance',
				'cluster_spectral_assign_labels', 'cluster_spectral_degree',
				'cluster_spectral_coef0', 'cluster_spectral_n_jobs', 'cluster_spectral_verbose',
				'cluster_spectral_random_state', 'df_cluster_spectral_results',
				'df_cluster_spectral_counts', 'df_cluster_spectral_metrics',
				'df_cluster_spectral_centroids', 'df_cluster_spectral_details',
				'cluster_spectral_plot_features', 'cluster_spectral_signature' ]
			
			with sp_b2:
				st.button( 'Reset Spectral Clustering', icon='🔁', key='cluster_spectral_reset',
					use_container_width=True, on_click=clear_keys, args=(spectral_reset_keys,) )
			
			if run_spectral:
				try:
					if spectral_clusters >= len( df_cluster_input ):
						st.warning(
							'⚠️ Number of Clusters must be less than the number of samples.' )
						st.stop( )
					
					if (spectral_affinity == 'nearest_neighbors' and spectral_neighbors >= len(
							df_cluster_input )):
						st.warning(
							'⚠️ Nearest Neighbors must be less than the number of samples.' )
						st.stop( )
					
					if (spectral_components > 0 and spectral_components >= len( df_cluster_input
					)):
						st.warning(
							'⚠️ Embedding Components must be less than the number of samples.' )
						st.stop( )
					
					spectral_eigen_solver = (
						None if spectral_eigen_solver_display == 'None' else
						spectral_eigen_solver_display)
					
					spectral_n_components = (
						None if spectral_components == 0 else spectral_components)
					
					model_parameters = { 'Model': 'Spectral', 'n_clusters': spectral_clusters,
						'affinity': spectral_affinity, 'gamma': spectral_gamma,
						'n_neighbors': spectral_neighbors, 'eigen_solver': spectral_eigen_solver,
						'n_components': spectral_n_components, 'n_init': spectral_n_init,
						'eigen_tol': spectral_eigen_tolerance,
						'assign_labels': spectral_assign_labels, 'degree': spectral_degree,
						'coef0': spectral_coef0, 'n_jobs': spectral_jobs,
						'verbose': spectral_verbose, 'random_state': spectral_random_state }
					
					cluster_spectral_signature = (
						(tuple( active_features ), cluster_data_signature), 'Spectral',
						tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = Spectral( clusters=spectral_clusters,
						random_state=spectral_random_state, n_init=spectral_n_init,
						gama=spectral_gamma, distance=spectral_affinity,
						neighbors=spectral_neighbors, tolerance=spectral_eigen_tolerance,
						assign=spectral_assign_labels, eigen_solver=spectral_eigen_solver,
						n_components=spectral_n_components, degree=spectral_degree,
						coef0=spectral_coef0, kernel_params=None, n_jobs=spectral_jobs,
						verbose=spectral_verbose )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					try:
						df_metrics = model.score( X )
						if df_metrics is None:
							df_metrics = pd.DataFrame( )
					except Exception:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					detail_rows = [ { 'Property': 'features', 'Value': int( model.features ) },
						{ 'Property': 'n_clusters', 'Value': int( model.n_clusters ) },
						{ 'Property': 'affinity', 'Value': model.affinity },
						{ 'Property': 'gamma', 'Value': float( model.gamma ) },
						{ 'Property': 'n_neighbors', 'Value': int( model.n_neighbors ) },
						{ 'Property': 'eigen_solver', 'Value': model.eigen_solver },
						{ 'Property': 'n_components', 'Value': model.n_components },
						{ 'Property': 'n_init', 'Value': int( model.n_init ) },
						{ 'Property': 'eigen_tolerance', 'Value': model.eigen_tolerance },
						{ 'Property': 'assign_labels', 'Value': model.assign_labels },
						{ 'Property': 'degree', 'Value': model.degree },
						{ 'Property': 'coef0', 'Value': model.coef0 },
						{ 'Property': 'n_jobs', 'Value': model.n_jobs },
						{ 'Property': 'verbose', 'Value': model.verbose },
						{ 'Property': 'random_state', 'Value': model.random_state } ]
					
					df_details = pd.DataFrame( detail_rows )
					df_centroids = pd.DataFrame( )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_spectral_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_spectral_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_spectral_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_spectral_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_spectral_details' ] = df_details.copy( )
					st.session_state[ 'cluster_spectral_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_spectral_signature' ] = cluster_spectral_signature
					st.success( 'Spectral clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_spectral_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_spectral_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_spectral_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_spectral_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_spectral_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_spectral_plot_features' ] = [ ]
					st.session_state[ 'cluster_spectral_signature' ] = None
					st.error( f'Spectral clustering failed: {ex}' )
			
			spectral_signature = st.session_state.get( 'cluster_spectral_signature', None )
			if (isinstance( spectral_signature, tuple ) and len( spectral_signature ) > 1 and
					spectral_signature[ 1 ] == 'Spectral'):
				df_results = st.session_state.get( 'df_cluster_spectral_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_spectral_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_spectral_metrics', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_spectral_details', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				st.data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					st.data_editor( df_metrics, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True )
			else:
				st.info( 'Run Spectral Clustering to view cluster counts and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					plt.close( 'all' )
					fig, ax = plt.subplots( )
					ax.scatter( df_results[ feature_columns[ 0 ] ],
						df_results[ feature_columns[ 1 ] ], c=df_results[ 'Cluster' ], alpha=0.7 )
					
					ax.set_xlabel( feature_columns[ 0 ] )
					ax.set_ylabel( feature_columns[ 1 ] )
					ax.set_title( 'Spectral Cluster Assignments' )
					st.pyplot( fig )
					plt.close( fig )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run Spectral Clustering to view the scatter plot.' )
		
		if 'cluster_optics_min_samples' not in st.session_state:
			st.session_state[ 'cluster_optics_min_samples' ] = 5
		
		if 'cluster_optics_max_eps' not in st.session_state:
			st.session_state[ 'cluster_optics_max_eps' ] = 0.0
		
		if 'cluster_optics_metric' not in st.session_state:
			st.session_state[ 'cluster_optics_metric' ] = 'minkowski'
		
		if 'cluster_optics_p' not in st.session_state:
			st.session_state[ 'cluster_optics_p' ] = 2.0
		
		if 'cluster_optics_cluster_method' not in st.session_state:
			st.session_state[ 'cluster_optics_cluster_method' ] = 'xi'
		
		if 'cluster_optics_eps' not in st.session_state:
			st.session_state[ 'cluster_optics_eps' ] = 0.5
		
		if 'cluster_optics_xi' not in st.session_state:
			st.session_state[ 'cluster_optics_xi' ] = 0.05
		
		if 'cluster_optics_predecessor_correction' not in st.session_state:
			st.session_state[ 'cluster_optics_predecessor_correction' ] = True
		
		if 'cluster_optics_min_cluster_size' not in st.session_state:
			st.session_state[ 'cluster_optics_min_cluster_size' ] = 0
		
		if 'cluster_optics_algorithm' not in st.session_state:
			st.session_state[ 'cluster_optics_algorithm' ] = 'auto'
		
		if 'cluster_optics_leaf_size' not in st.session_state:
			st.session_state[ 'cluster_optics_leaf_size' ] = 30
		
		if 'cluster_optics_n_jobs' not in st.session_state:
			st.session_state[ 'cluster_optics_n_jobs' ] = 1
		
		with st.expander( 'OPTICS Clustering', expanded=False ):
			st.caption(
				'Density-based clustering across varying neighborhood radii with explicit noise '
				'detection.' )
			
			op_c1, op_c2, op_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with op_c1:
				optics_min_samples = int( st.number_input( 'Minimum Samples', min_value=2,
					max_value=max( 2, len( df_cluster_input ) ), step=1,
					key='cluster_optics_min_samples' ) )
				
				optics_max_eps_input = float(
					st.number_input( 'Maximum Epsilon (0 = Unlimited)', min_value=0.0, step=0.1,
						format='%.4f', key='cluster_optics_max_eps' ) )
				
				optics_metric = st.selectbox( 'Distance Metric',
					options=[ 'minkowski', 'euclidean', 'manhattan', 'chebyshev', 'cosine' ],
					key='cluster_optics_metric' )
				
				optics_p = float(
					st.number_input( 'Minkowski Power', min_value=1.0, step=1.0, format='%.1f',
						key='cluster_optics_p' ) )
			
			with op_c2:
				optics_cluster_method = st.selectbox( 'Cluster Extraction Method',
					options=[ 'xi', 'dbscan' ], key='cluster_optics_cluster_method' )
				
				optics_eps = float(
					st.number_input( 'DBSCAN Epsilon', min_value=0.0001, step=0.1, format='%.4f',
						disabled=optics_cluster_method != 'dbscan', key='cluster_optics_eps' ) )
				
				optics_xi = float(
					st.number_input( 'Xi', min_value=0.0001, max_value=1.0, step=0.01,
						format='%.4f', disabled=optics_cluster_method != 'xi',
						key='cluster_optics_xi' ) )
				
				optics_predecessor_correction = st.checkbox( 'Predecessor Correction',
					disabled=optics_cluster_method != 'xi',
					key='cluster_optics_predecessor_correction' )
			
			with op_c3:
				optics_min_cluster_size_input = int(
					st.number_input( 'Minimum Cluster Size (0 = Default)', min_value=0,
						max_value=max( 1, len( df_cluster_input ) ), step=1,
						key='cluster_optics_min_cluster_size' ) )
				
				optics_algorithm = st.selectbox( 'Neighbor Algorithm',
					options=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ],
					key='cluster_optics_algorithm' )
				
				optics_leaf_size = int( st.number_input( 'Leaf Size', min_value=1, step=1,
					key='cluster_optics_leaf_size' ) )
				
				optics_n_jobs = int( st.number_input( 'Parallel Jobs', min_value=-1, step=1,
					key='cluster_optics_n_jobs' ) )
			
			op_b1, op_b2 = st.columns( 2 )
			with op_b1:
				run_optics = st.button( 'Run OPTICS Clustering', icon='🏃',
					key='cluster_optics_run', use_container_width=True )
			
			optics_reset_keys = [ 'cluster_optics_min_samples', 'cluster_optics_max_eps',
				'cluster_optics_metric', 'cluster_optics_p', 'cluster_optics_cluster_method',
				'cluster_optics_eps', 'cluster_optics_xi', 'cluster_optics_predecessor_correction',
				'cluster_optics_min_cluster_size', 'cluster_optics_algorithm',
				'cluster_optics_leaf_size', 'cluster_optics_n_jobs', 'df_cluster_optics_results',
				'df_cluster_optics_counts', 'df_cluster_optics_metrics',
				'df_cluster_optics_centroids', 'df_cluster_optics_details',
				'df_cluster_optics_ordering', 'cluster_optics_plot_features',
				'cluster_optics_signature' ]
			
			with op_b2:
				st.button( 'Reset OPTICS Clustering', icon='🔁', key='cluster_optics_reset',
					use_container_width=True, on_click=clear_keys, args=(optics_reset_keys,) )
			
			if run_optics:
				try:
					if optics_min_samples > len( df_cluster_input ):
						st.warning( '⚠️ Minimum Samples cannot exceed the number of samples.' )
						st.stop( )
					
					if (optics_min_cluster_size_input > 0 and optics_min_cluster_size_input > len(
							df_cluster_input )):
						st.warning( '⚠️ Minimum Cluster Size cannot exceed the number of '
						            'samples.' )
						st.stop( )
					
					optics_max_eps = (
						np.inf if optics_max_eps_input == 0.0 else optics_max_eps_input)
					
					optics_min_cluster_size = (
						None if optics_min_cluster_size_input == 0 else
						optics_min_cluster_size_input)
					
					optics_effective_eps = (
						optics_eps if optics_cluster_method == 'dbscan' else None)
					
					model_parameters = { 'Model': 'OPTICS', 'min_samples': optics_min_samples,
						'max_eps': optics_max_eps, 'metric': optics_metric, 'p': optics_p,
						'cluster_method': optics_cluster_method, 'eps': optics_effective_eps,
						'xi': optics_xi, 'predecessor_correction': optics_predecessor_correction,
						'min_cluster_size': optics_min_cluster_size, 'algorithm': optics_algorithm,
						'leaf_size': optics_leaf_size, 'n_jobs': optics_n_jobs }
					
					cluster_optics_signature = ((tuple( active_features ), cluster_data_signature),
						'OPTICS', tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = OPTICS( samples=optics_min_samples, max_eps=optics_max_eps,
						metric=optics_metric, algorithm=optics_algorithm,
						leaf_size=optics_leaf_size, eps=optics_effective_eps,
						predecessor_correction=optics_predecessor_correction,
						min_cluster_size=optics_min_cluster_size, p=optics_p, metric_params=None,
						cluster_method=optics_cluster_method, xi=optics_xi, memory=None,
						n_jobs=optics_n_jobs )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					df_metrics = model.score( X )
					if df_metrics is None:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					unique_labels = np.unique( labels )
					cluster_labels = unique_labels[ unique_labels != -1 ]
					noise_count = int( np.sum( labels == -1 ) )
					detail_rows = [ { 'Property': 'features', 'Value': int( model.features ) },
						{ 'Property': 'clusters', 'Value': int( len( cluster_labels ) ) },
						{ 'Property': 'noise_samples', 'Value': noise_count },
						{ 'Property': 'min_samples', 'Value': model.min_samples },
						{ 'Property': 'max_eps', 'Value': model.max_eps },
						{ 'Property': 'metric', 'Value': model.metric },
						{ 'Property': 'p', 'Value': model.p },
						{ 'Property': 'cluster_method', 'Value': model.cluster_method },
						{ 'Property': 'eps', 'Value': model.eps },
						{ 'Property': 'xi', 'Value': model.xi },
						{ 'Property': 'predecessor_correction',
							'Value': model.predecessor_correction },
						{ 'Property': 'min_cluster_size', 'Value': model.min_cluster_size },
						{ 'Property': 'algorithm', 'Value': model.algorithm },
						{ 'Property': 'leaf_size', 'Value': model.leaf_size },
						{ 'Property': 'n_jobs', 'Value': model.n_jobs } ]
					
					df_details = pd.DataFrame( detail_rows )
					df_ordering = pd.DataFrame(
						{ 'Sample': np.arange( len( model.ordering ) ), 'Ordering': model.ordering,
							'Reachability': model.reachability,
							'Core Distance': model.core_distances,
							'Predecessor': model.predecessor } )
					
					df_centroids = pd.DataFrame( )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_optics_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_optics_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_optics_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_optics_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_optics_details' ] = df_details.copy( )
					st.session_state[ 'df_cluster_optics_ordering' ] = df_ordering.copy( )
					st.session_state[ 'cluster_optics_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_optics_signature' ] = cluster_optics_signature
					st.success( 'OPTICS clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_optics_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_optics_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_optics_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_optics_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_optics_details' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_optics_ordering' ] = pd.DataFrame( )
					st.session_state[ 'cluster_optics_plot_features' ] = [ ]
					st.session_state[ 'cluster_optics_signature' ] = None
					st.error( f'OPTICS clustering failed: {ex}' )
			
			optics_signature = st.session_state.get( 'cluster_optics_signature', None )
			if (isinstance( optics_signature, tuple ) and len( optics_signature ) > 1 and
					optics_signature[ 1 ] == 'OPTICS'):
				df_results = st.session_state.get( 'df_cluster_optics_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_optics_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_optics_metrics', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_optics_details', pd.DataFrame( ) )
				df_ordering = st.session_state.get( 'df_cluster_optics_ordering', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_details = pd.DataFrame( )
				df_ordering = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				st.data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					st.data_editor( df_metrics, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True )
				
				if not df_ordering.empty:
					st.caption( 'Reachability and Ordering' )
					st.data_editor( df_ordering, use_container_width=True )
			else:
				st.info( 'Run OPTICS Clustering to view cluster counts, noise counts, and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					plt.close( 'all' )
					fig, ax = plt.subplots( )
					
					ax.scatter( df_results[ feature_columns[ 0 ] ],
						df_results[ feature_columns[ 1 ] ], c=df_results[ 'Cluster' ], alpha=0.7 )
					
					ax.set_xlabel( feature_columns[ 0 ] )
					ax.set_ylabel( feature_columns[ 1 ] )
					ax.set_title( 'OPTICS Cluster Assignments' )
					st.pyplot( fig )
					plt.close( fig )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run OPTICS Clustering to view the scatter plot.' )
		
		if 'cluster_mean_shift_bandwidth' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_bandwidth' ] = 0.0
		
		if 'cluster_mean_shift_bin_seeding' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_bin_seeding' ] = False
		
		if 'cluster_mean_shift_min_bin_freq' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_min_bin_freq' ] = 1
		
		if 'cluster_mean_shift_cluster_all' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_cluster_all' ] = True
		
		if 'cluster_mean_shift_n_jobs' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_n_jobs' ] = 1
		
		if 'cluster_mean_shift_max_iter' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_max_iter' ] = 300
		
		with st.expander( 'Mean Shift Clustering', expanded=False ):
			st.caption(
				'Centroid-based clustering that discovers cluster centers from the density of the '
				'data.' )
			
			ms_c1, ms_c2, ms_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with ms_c1:
				mean_shift_bandwidth_input = float(
					st.number_input( 'Bandwidth (0 = Automatic)', min_value=0.0, step=0.1,
						format='%.4f', key='cluster_mean_shift_bandwidth' ) )
				
				mean_shift_bin_seeding = st.checkbox( 'Use Bin Seeding',
					key='cluster_mean_shift_bin_seeding' )
			
			with ms_c2:
				mean_shift_min_bin_freq = int( st.number_input( 'Minimum Bin Frequency',
					min_value=1, step=1, key='cluster_mean_shift_min_bin_freq' ) )
				
				mean_shift_cluster_all = st.checkbox( 'Assign All Samples',
					key='cluster_mean_shift_cluster_all' )
			
			with ms_c3:
				mean_shift_n_jobs = int( st.number_input( 'Parallel Jobs', min_value=-1, step=1,
					key='cluster_mean_shift_n_jobs' ) )
				
				mean_shift_max_iter = int( st.number_input( 'Maximum Iterations', min_value=1,
					step=1, key='cluster_mean_shift_max_iter' ) )
			
			ms_b1, ms_b2 = st.columns( 2 )
			
			with ms_b1:
				run_mean_shift = st.button( 'Run Mean Shift Clustering', icon='🏃',
					key='cluster_mean_shift_run', use_container_width=True )
			
			mean_shift_reset_keys = [ 'cluster_mean_shift_bandwidth',
				'cluster_mean_shift_bin_seeding', 'cluster_mean_shift_min_bin_freq',
				'cluster_mean_shift_cluster_all', 'cluster_mean_shift_n_jobs',
				'cluster_mean_shift_max_iter', 'df_cluster_mean_shift_results',
				'df_cluster_mean_shift_counts', 'df_cluster_mean_shift_metrics',
				'df_cluster_mean_shift_centroids', 'df_cluster_mean_shift_details',
				'cluster_mean_shift_plot_features', 'cluster_mean_shift_signature' ]
			
			with ms_b2:
				st.button( 'Reset Mean Shift Clustering', icon='🔁',
					key='cluster_mean_shift_reset',
					use_container_width=True, on_click=clear_keys, args=(mean_shift_reset_keys,) )
			
			if run_mean_shift:
				try:
					mean_shift_bandwidth = (
						None if mean_shift_bandwidth_input == 0.0 else mean_shift_bandwidth_input)
					
					model_parameters = { 'Model': 'MeanShift', 'bandwidth': mean_shift_bandwidth,
						'bin_seeding': mean_shift_bin_seeding,
						'min_bin_freq': mean_shift_min_bin_freq,
						'cluster_all': mean_shift_cluster_all, 'n_jobs': mean_shift_n_jobs,
						'max_iter': mean_shift_max_iter }
					
					cluster_mean_shift_signature = (
						(tuple( active_features ), cluster_data_signature), 'MeanShift',
						tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = MeanShift( min_bin=mean_shift_min_bin_freq,
						group_all=mean_shift_cluster_all, bandwidth=mean_shift_bandwidth,
						seeds=None, bin_seeding=mean_shift_bin_seeding, n_jobs=mean_shift_n_jobs,
						max_iter=mean_shift_max_iter, min_bin_freq=mean_shift_min_bin_freq,
						cluster_all=mean_shift_cluster_all )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					df_metrics = model.score( X )
					if df_metrics is None:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					centroids = np.asarray( model.centroids_ )
					df_centroids = pd.DataFrame( centroids, columns=feature_columns )
					df_centroids.insert( 0, 'Cluster', np.arange( len( df_centroids ) ) )
					detail_rows = [ { 'Property': 'features', 'Value': int( model.features ) },
						{ 'Property': 'clusters', 'Value': int( len( centroids ) ) },
						{ 'Property': 'bandwidth', 'Value': model.bandwidth },
						{ 'Property': 'bin_seeding', 'Value': model.bin_seeding },
						{ 'Property': 'min_bin_freq', 'Value': model.min_bin_freq },
						{ 'Property': 'cluster_all', 'Value': model.cluster_all },
						{ 'Property': 'n_jobs', 'Value': model.n_jobs },
						{ 'Property': 'max_iter', 'Value': model.max_iter },
						{ 'Property': 'iterations', 'Value': model.iterations } ]
					
					df_details = pd.DataFrame( detail_rows )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_mean_shift_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_mean_shift_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_mean_shift_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_mean_shift_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_mean_shift_details' ] = df_details.copy( )
					st.session_state[ 'cluster_mean_shift_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_mean_shift_signature' ] = cluster_mean_shift_signature
					st.success( 'Mean Shift clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_mean_shift_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_mean_shift_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_mean_shift_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_mean_shift_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_mean_shift_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_mean_shift_plot_features' ] = [ ]
					st.session_state[ 'cluster_mean_shift_signature' ] = None
					st.error( f'Mean Shift clustering failed: {ex}' )
			
			mean_shift_signature = st.session_state.get( 'cluster_mean_shift_signature', None )
			if (isinstance( mean_shift_signature, tuple ) and len( mean_shift_signature ) > 1 and
					mean_shift_signature[ 1 ] == 'MeanShift'):
				df_results = st.session_state.get( 'df_cluster_mean_shift_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_mean_shift_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_mean_shift_metrics', pd.DataFrame( ) )
				df_centroids = st.session_state.get( 'df_cluster_mean_shift_centroids', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_mean_shift_details', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_centroids = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				st.data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					st.data_editor( df_metrics, use_container_width=True )
				
				if not df_centroids.empty:
					st.caption( 'Cluster Centroids' )
					st.data_editor( df_centroids, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True )
			else:
				st.info(
					'Run Mean Shift Clustering to view cluster counts, centroids, and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					plt.close( 'all' )
					fig, ax = plt.subplots( )
					
					ax.scatter( df_results[ feature_columns[ 0 ] ],
						df_results[ feature_columns[ 1 ] ], c=df_results[ 'Cluster' ], alpha=0.7 )
					
					if not df_centroids.empty:
						ax.scatter( df_centroids[ feature_columns[ 0 ] ],
							df_centroids[ feature_columns[ 1 ] ], marker='X', s=180,
							edgecolors='black', linewidths=1.0, label='Centroids' )
						
						ax.legend( )
					
					ax.set_xlabel( feature_columns[ 0 ] )
					ax.set_ylabel( feature_columns[ 1 ] )
					ax.set_title( 'Mean Shift Cluster Assignments' )
					st.pyplot( fig )
					plt.close( fig )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run Mean Shift Clustering to view the scatter plot.' )
		
		with st.expander( 'Affinity Propagation Clustering', expanded=False ):
			affinity_defaults = { 'cluster_affinity_damping': 0.5, 'cluster_affinity_max_iter': 200,
				'cluster_affinity_convergence_iter': 15, 'cluster_affinity_use_preference': False,
				'cluster_affinity_preference': 0.0, 'cluster_affinity_copy': True,
				'cluster_affinity_verbose': False, 'cluster_affinity_random_state': 42 }
			
			for key, value in affinity_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			st.caption(
				'Exemplar-based clustering that determines cluster representatives from pairwise '
				'similarities.' )
			
			ap_c1, ap_c2, ap_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with ap_c1:
				affinity_damping = float( st.number_input( 'Damping', min_value=0.5,
					max_value=0.9999, step=0.01, format='%.4f', key='cluster_affinity_damping' ) )
				
				affinity_max_iter = int( st.number_input( 'Maximum Iterations', min_value=1,
					step=1, key='cluster_affinity_max_iter' ) )
			
			with ap_c2:
				affinity_convergence_iter = int( st.number_input( 'Convergence Iterations',
					min_value=1, step=1, key='cluster_affinity_convergence_iter' ) )
				
				affinity_copy = st.checkbox( 'Copy Input Data', key='cluster_affinity_copy' )
			
			with ap_c3:
				affinity_use_preference = st.checkbox( 'Use Custom Preference',
					key='cluster_affinity_use_preference' )
				
				affinity_preference_input = float( st.number_input( 'Preference', step=0.1,
					format='%.4f', disabled=not affinity_use_preference,
					key='cluster_affinity_preference' ) )
				
				affinity_verbose = st.checkbox( 'Verbose', key='cluster_affinity_verbose' )
				affinity_random_state = int( st.number_input( 'Random State', min_value=0, step=1,
					key='cluster_affinity_random_state' ) )
			
			affinity_reset_keys = list( affinity_defaults.keys( ) ) + [
				'df_cluster_affinity_results', 'df_cluster_affinity_counts',
				'df_cluster_affinity_metrics', 'df_cluster_affinity_centroids',
				'df_cluster_affinity_details', 'cluster_affinity_plot_features',
				'cluster_affinity_signature' ]
			
			ap_b1, ap_b2 = st.columns( 2 )
			with ap_b1:
				run_affinity = st.button( 'Run Affinity Propagation', icon='🏃',
					key='cluster_affinity_run', use_container_width=True )
			
			with ap_b2:
				st.button( 'Reset Affinity Propagation', icon='🔁', key='cluster_affinity_reset',
					use_container_width=True, on_click=clear_keys, args=(affinity_reset_keys,) )
			
			if run_affinity:
				try:
					if affinity_convergence_iter > affinity_max_iter:
						st.warning( '⚠️ Convergence Iterations cannot exceed Maximum Iterations.' )
						st.stop( )
					
					affinity_preference = ( affinity_preference_input if affinity_use_preference else None)
					model_parameters = { 'Model': 'AffinityPropagation',
						'damping': affinity_damping, 'max_iter': affinity_max_iter,
						'convergence_iter': affinity_convergence_iter,
						'preference': affinity_preference, 'affinity': 'euclidean',
						'copy': affinity_copy, 'verbose': affinity_verbose,
						'random_state': affinity_random_state }
					
					cluster_affinity_signature = (
						(tuple( active_features ), cluster_data_signature), 'AffinityPropagation',
						tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = AffinityPropagation( damping=affinity_damping,
						max_iter=affinity_max_iter, convergence_iter=affinity_convergence_iter,
						preference=affinity_preference, affinity='euclidean', copy=affinity_copy,
						verbose=affinity_verbose, random_state=affinity_random_state )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts(
						dropna=False ).rename_axis( 'Cluster' ).reset_index(
						name='Count' ).sort_values( by='Cluster' ).reset_index( drop=True ))
					
					try:
						df_metrics = model.score( X )
						if df_metrics is None:
							df_metrics = pd.DataFrame( )
					except Exception:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					centroids = np.asarray( model.centroids_ )
					df_centroids = pd.DataFrame( centroids, columns=feature_columns )
					df_centroids.insert( 0, 'Cluster', range( len( df_centroids ) ) )
					
					detail_rows = [ { 'Property': 'features', 'Value': model.features },
						{ 'Property': 'clusters', 'Value': len( df_centroids ) },
						{ 'Property': 'iterations', 'Value': model.iterations },
						{ 'Property': 'damping', 'Value': model.damping },
						{ 'Property': 'max_iter', 'Value': model.max_iter },
						{ 'Property': 'convergence_iter', 'Value': model.convergence_iter },
						{ 'Property': 'preference', 'Value': model.preference },
						{ 'Property': 'affinity', 'Value': model.affinity },
						{ 'Property': 'copy', 'Value': model.copy },
						{ 'Property': 'verbose', 'Value': model.verbose },
						{ 'Property': 'random_state', 'Value': model.random_state } ]
					
					df_details = pd.DataFrame( detail_rows )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_affinity_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_affinity_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_affinity_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_affinity_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_affinity_details' ] = df_details.copy( )
					st.session_state[ 'cluster_affinity_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_affinity_signature' ] = cluster_affinity_signature
					st.success( 'Affinity Propagation clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_affinity_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_affinity_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_affinity_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_affinity_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_affinity_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_affinity_plot_features' ] = [ ]
					st.session_state[ 'cluster_affinity_signature' ] = None
					st.error( f'Affinity Propagation clustering failed: {ex}' )
			
			affinity_signature = st.session_state.get( 'cluster_affinity_signature', None )
			if (isinstance( affinity_signature, tuple ) and len( affinity_signature ) > 1 and
					affinity_signature[ 1 ] == 'AffinityPropagation'):
				df_results = st.session_state.get( 'df_cluster_affinity_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_affinity_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_affinity_metrics', pd.DataFrame( ) )
				df_centroids = st.session_state.get( 'df_cluster_affinity_centroids', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_affinity_details', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_centroids = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				st.data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					st.data_editor( df_metrics, use_container_width=True )
				
				if not df_centroids.empty:
					st.caption( 'Cluster Exemplars' )
					st.data_editor( df_centroids, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True )
			else:
				st.info(
					'Run Affinity Propagation to view cluster counts, exemplars, and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					plt.close( 'all' )
					fig, ax = plt.subplots( )
					ax.scatter( df_results[ feature_columns[ 0 ] ],
						df_results[ feature_columns[ 1 ] ], c=df_results[ 'Cluster' ], alpha=0.7 )
					
					if not df_centroids.empty:
						ax.scatter( df_centroids[ feature_columns[ 0 ] ],
							df_centroids[ feature_columns[ 1 ] ], marker='X', s=180,
							edgecolors='black', linewidths=1.0, label='Exemplars' )
						ax.legend( )
					
					ax.set_xlabel( feature_columns[ 0 ] )
					ax.set_ylabel( feature_columns[ 1 ] )
					ax.set_title( 'Affinity Propagation Cluster Assignments' )
					st.pyplot( fig )
					plt.close( fig )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run Affinity Propagation to view the scatter plot.' )
		
		cluster_max_clusters = max( 2, len( df_cluster_input ) - 1 )
		if 'cluster_birch_n_clusters' in st.session_state:
			st.session_state[ 'cluster_birch_n_clusters' ] = min(
				max( 2, int( st.session_state[ 'cluster_birch_n_clusters' ] ) ),
				cluster_max_clusters )
		
		with st.expander( 'Birch Clustering', expanded=False ):
			birch_defaults = { 'cluster_birch_threshold': 0.5, 'cluster_birch_branching_factor':
				50,
				'cluster_birch_use_global_clusters': True, 'cluster_birch_n_clusters': 3,
				'cluster_birch_compute_labels': True }
			
			for key, value in birch_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			st.caption( 'Scalable hierarchical clustering using a clustering-feature tree.' )
			
			br_c1, br_c2, br_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with br_c1:
				birch_threshold = float(
					st.number_input( 'Threshold', min_value=0.0001, step=0.1, format='%.4f',
						key='cluster_birch_threshold' ) )
			
			with br_c2:
				birch_branching_factor = int(
					st.number_input( 'Branching Factor', min_value=2, step=1,
						key='cluster_birch_branching_factor' ) )
				
				birch_compute_labels = st.checkbox( 'Compute Labels',
					key='cluster_birch_compute_labels' )
			
			with br_c3:
				birch_use_global_clusters = st.checkbox( 'Apply Global Clustering',
					key='cluster_birch_use_global_clusters' )
				
				birch_n_clusters = int( st.number_input( 'Number of Clusters', min_value=2,
					max_value=max( 2, len( df_cluster_input ) - 1 ), step=1,
					disabled=not birch_use_global_clusters, key='cluster_birch_n_clusters' ) )
			
			br_b1, br_b2 = st.columns( 2 )
			with br_b1:
				run_birch = st.button( 'Run Birch Clustering', icon='🏃', key='cluster_birch_run',
					use_container_width=True )
			
			birch_reset_keys = [ 'cluster_birch_threshold', 'cluster_birch_branching_factor',
				'cluster_birch_use_global_clusters', 'cluster_birch_n_clusters',
				'cluster_birch_compute_labels', 'df_cluster_birch_results',
				'df_cluster_birch_counts', 'df_cluster_birch_metrics',
				'df_cluster_birch_centroids',
				'df_cluster_birch_details', 'cluster_birch_plot_features',
				'cluster_birch_signature' ]
			
			with br_b2:
				st.button( 'Reset Birch Clustering', icon='🔁', key='cluster_birch_reset',
					use_container_width=True, on_click=clear_keys, args=(birch_reset_keys,) )
			
			if run_birch:
				try:
					if (birch_use_global_clusters and birch_n_clusters >= len( df_cluster_input )):
						st.warning(
							'⚠️ Number of Clusters must be less than the number of samples.' )
						st.stop( )
					
					effective_clusters = (birch_n_clusters if birch_use_global_clusters else None)
					model_parameters = { 'Model': 'Birch', 'threshold': birch_threshold,
						'branching_factor': birch_branching_factor,
						'n_clusters': effective_clusters, 'compute_labels': birch_compute_labels }
					
					cluster_birch_signature = ((tuple( active_features ), cluster_data_signature),
						'Birch',
						tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = Birch( threshold=birch_threshold, branching_factor=birch_branching_factor,
						n_clusters=effective_clusters, compute_labels=birch_compute_labels )
					
					start_time = time.time( )
					if birch_compute_labels:
						labels = model.project( X )
					else:
						model.train( X )
						labels = model.predict( X )
					
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					try:
						df_metrics = model.score( X )
						if df_metrics is None:
							df_metrics = pd.DataFrame( )
					except Exception:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					subcluster_centers = np.asarray( model.subcluster_centers )
					df_centroids = pd.DataFrame( subcluster_centers, columns=feature_columns )
					df_centroids.insert( 0, 'Subcluster', range( len( df_centroids ) ) )
					try:
						subcluster_labels = np.asarray( model.subcluster_labels ).reshape( -1 )
						if len( subcluster_labels ) == len( df_centroids ):
							df_centroids.insert( 1, 'Cluster', subcluster_labels )
					except Exception:
						pass
					
					detail_rows = [ { 'Property': 'features', 'Value': model.features },
						{ 'Property': 'clusters', 'Value': len( np.unique( labels ) ) },
						{ 'Property': 'subclusters', 'Value': len( df_centroids ) },
						{ 'Property': 'threshold', 'Value': model.threshold },
						{ 'Property': 'branching_factor', 'Value': model.branching_factor },
						{ 'Property': 'n_clusters', 'Value': model.n_clusters },
						{ 'Property': 'compute_labels', 'Value': model.compute_labels } ]
					
					df_details = pd.DataFrame( detail_rows )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_birch_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_birch_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_birch_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_birch_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_birch_details' ] = df_details.copy( )
					st.session_state[ 'cluster_birch_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_birch_signature' ] = cluster_birch_signature
					st.success( 'Birch clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_birch_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_birch_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_birch_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_birch_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_birch_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_birch_plot_features' ] = [ ]
					st.session_state[ 'cluster_birch_signature' ] = None
					st.error( f'Birch clustering failed: {ex}' )
			
			birch_signature = st.session_state.get( 'cluster_birch_signature', None )
			if (isinstance( birch_signature, tuple ) and len( birch_signature ) > 1 and
					birch_signature[ 1 ] == 'Birch'):
				df_results = st.session_state.get( 'df_cluster_birch_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_birch_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_birch_metrics', pd.DataFrame( ) )
				df_centroids = st.session_state.get( 'df_cluster_birch_centroids', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_birch_details', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_centroids = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				st.data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					st.data_editor( df_metrics, use_container_width=True )
				
				if not df_centroids.empty:
					st.caption( 'Subcluster Centers' )
					st.data_editor( df_centroids, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True )
			else:
				st.info( 'Run Birch Clustering to view cluster counts, subclusters, and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					plt.close( 'all' )
					fig, ax = plt.subplots( )
					ax.scatter( df_results[ feature_columns[ 0 ] ],
						df_results[ feature_columns[ 1 ] ], c=df_results[ 'Cluster' ], alpha=0.7 )
					
					if not df_centroids.empty:
						ax.scatter( df_centroids[ feature_columns[ 0 ] ],
							df_centroids[ feature_columns[ 1 ] ], marker='X', s=140,
							edgecolors='black', linewidths=1.0, label='Subcluster Centers' )
						ax.legend( )
					
					ax.set_xlabel( feature_columns[ 0 ] )
					ax.set_ylabel( feature_columns[ 1 ] )
					ax.set_title( 'Birch Cluster Assignments' )
					st.pyplot( fig )
					plt.close( fig )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run Birch Clustering to view the scatter plot.' )

# ============================================
# TIME SERIES MODE
# ============================================
elif mode == 'Time-Series Models':
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		st.subheader( cfg.MODE[ 'Time-Series Models' ] )
		st.divider( )
		
		# ------------------------------------------------------------------
		# TIME-SERIES INPUT
		# ------------------------------------------------------------------
		df_dataset = get_loaded_dataset( )
		if df_dataset is None or df_dataset.empty:
			st.warning( '⚠️ No dataset loaded.' )
			st.stop( )
		
		numeric_columns = [ column for column in df_dataset.columns if
			pd.api.types.is_numeric_dtype(
				df_dataset[ column ] ) and not pd.api.types.is_bool_dtype( df_dataset[ column ] ) ]
		
		if not numeric_columns:
			st.warning( '⚠️ No numeric columns available for time-series analysis.' )
			st.stop( )
		
		timeseries_source_signature = (tuple( df_dataset.columns.tolist( ) ),
			tuple( str( dtype ) for dtype in df_dataset.dtypes.tolist( ) ),
			int( len( df_dataset ) ),
			int( pd.util.hash_pandas_object( df_dataset, index=True ).sum( ) ))
		
		prior_timeseries_source_signature = st.session_state.get( 'timeseries_source_signature',
			None )
		
		if ( prior_timeseries_source_signature is not None
				and prior_timeseries_source_signature != timeseries_source_signature):
			timeseries_keys = \
			[ key for key in list( st.session_state.keys( ) ) if key.startswith( 'timeseries_' ) ]
			clear_keys( timeseries_keys )
		
		st.session_state[ 'timeseries_source_signature' ] = timeseries_source_signature
		timeseries_defaults = { 'timeseries_col_box': numeric_columns[ 0 ] }
		for key, value in timeseries_defaults.items( ):
			if key not in st.session_state:
				st.session_state[ key ] = value
		
		if st.session_state[ 'timeseries_col_box' ] not in numeric_columns:
			st.session_state[ 'timeseries_col_box' ] = numeric_columns[ 0 ]
		
		st.markdown( '##### Time-Series Selection' )
		series_col = st.selectbox( 'Select Numeric Time-Series Column', numeric_columns,
			key='timeseries_col_box' )
		
		series_values = pd.to_numeric( df_dataset[ series_col ], errors='coerce' )
		series_values = series_values.replace( [ np.inf, -np.inf ], np.nan )
		invalid_observations = int( series_values.isna( ).sum( ) )
		if invalid_observations > 0:
			st.warning( f'⚠️ The selected series contains {invalid_observations:,} missing, '
			            'infinite, or nonnumeric observation(s). Resolve them before forecasting '
			            'to preserve temporal spacing.' )
			st.stop( )
		
		series = series_values.to_numpy( dtype=float )
		if series.ndim != 1:
			st.warning( '⚠️ The selected column could not be converted to a one-dimensional series.' )
			st.stop( )
		
		if len( series ) < 10:
			st.warning( '⚠️ Selected series must contain at least 10 valid observations.' )
			st.stop( )
		
		series_signature = int( pd.util.hash_pandas_object( series_values, index=True ).sum( ) )
		st.session_state[ 'timeseries_column' ] = series_col
		st.session_state[ 'timeseries_series' ] = series.copy( )
		df_timeseries_summary = pd.DataFrame(
			[ { 'Series': series_col, 'Observations': len( series ),
				'Minimum': float( np.min( series ) ), 'Maximum': float( np.max( series ) ),
				'Mean': float( np.mean( series ) ),
				'Standard Deviation': float( np.std( series ) ) } ] )
		
		st.data_editor( df_timeseries_summary, use_container_width=True, hide_index=True,
			disabled=True )
		
		# ------------------------------------------------------------------
		# LAGGED LINEAR REGRESSION
		# ------------------------------------------------------------------
		with st.expander( 'Lagged Linear Regression', expanded=False ):
			lag_linear_defaults = { 'timeseries_lag_linear_lag': 5,
				'timeseries_lag_linear_horizon': 5 }
			
			for key, value in lag_linear_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			maximum_lag = max( 1, len( series ) - 1 )
			st.session_state[ 'timeseries_lag_linear_lag' ] = min(
				int( st.session_state[ 'timeseries_lag_linear_lag' ] ), maximum_lag )
			
			st.caption( 'Linear autoregressive forecasting using lagged observations as predictors.' )
			ll_c1, ll_c2 = st.columns( 2, border=True )
			with ll_c1:
				lag_linear_order = int( st.number_input( 'Lag Order', min_value=1,
					max_value=max( 1, len( series ) - 1 ), step=1,
					key='timeseries_lag_linear_lag' ) )
			
			with ll_c2:
				lag_linear_horizon = int( st.number_input( 'Forecast Horizon', min_value=1, step=1,
					key='timeseries_lag_linear_horizon' ) )
			
			lag_linear_reset_keys = list( lag_linear_defaults.keys( ) ) + [
				'timeseries_lag_linear_model', 'timeseries_lag_linear_forecast',
				'timeseries_lag_linear_metrics', 'timeseries_lag_linear_results',
				'timeseries_lag_linear_signature' ]
			
			ll_b1, ll_b2 = st.columns( 2 )
			with ll_b1:
				run_lag_linear = st.button( 'Run Lagged Linear Regression', icon='🏃',
					key='timeseries_lag_linear_run', use_container_width=True )
			
			with ll_b2:
				st.button( 'Reset Lagged Linear Regression', icon='🔁',
					key='timeseries_lag_linear_reset', use_container_width=True,
					on_click=clear_keys, args=(lag_linear_reset_keys,) )
			
			if run_lag_linear:
				try:
					if lag_linear_order >= len( series ):
						st.warning( '⚠️ Lag Order must be less than the number of observations.' )
						st.stop( )
					
					model = LaggingSeries( lag=lag_linear_order )
					start_time = time.time( )
					model.train( series )
					metrics = model.analyze( )
					forecast = model.project( n_steps=lag_linear_horizon )
					elapsed_seconds = time.time( ) - start_time
					df_metrics = pd.DataFrame(
						[ { 'Metric': metric, 'Value': float( value ) } for metric, value in
							metrics.items( ) ] )
					
					df_metrics = pd.concat( [ df_metrics, pd.DataFrame(
						[ { 'Metric': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] ) ], ignore_index=True )
					
					forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
					df_results = pd.DataFrame( { 'Period': forecast_index,
						'Forecast': np.asarray( forecast, dtype=float ) } )
					
					lag_linear_signature = (series_col, len( series ), series_signature,
						lag_linear_order, lag_linear_horizon)
					
					st.session_state[ 'timeseries_lag_linear_model' ] = model
					st.session_state[ 'timeseries_lag_linear_forecast' ] = np.asarray( forecast,
						dtype=float )
					st.session_state[ 'timeseries_lag_linear_metrics' ] = df_metrics.copy( )
					st.session_state[ 'timeseries_lag_linear_results' ] = df_results.copy( )
					st.session_state[ 'timeseries_lag_linear_signature' ] = lag_linear_signature
					st.success( 'Lagged Linear Regression forecasting complete.' )
				except Exception as ex:
					st.session_state.pop( 'timeseries_lag_linear_model', None )
					st.session_state.pop( 'timeseries_lag_linear_forecast', None )
					st.session_state.pop( 'timeseries_lag_linear_metrics', None )
					st.session_state.pop( 'timeseries_lag_linear_results', None )
					st.session_state.pop( 'timeseries_lag_linear_signature', None )
					st.error( f'Lagged Linear Regression failed: {ex}' )
			
			lag_linear_signature = st.session_state.get( 'timeseries_lag_linear_signature', None )
			current_lag_linear_signature = (series_col, len( series ), series_signature,
				lag_linear_order, lag_linear_horizon)
			
			if lag_linear_signature == current_lag_linear_signature:
				df_metrics = st.session_state.get( 'timeseries_lag_linear_metrics',
					pd.DataFrame( ) )
				df_results = st.session_state.get( 'timeseries_lag_linear_results',
					pd.DataFrame( ) )
				forecast = st.session_state.get( 'timeseries_lag_linear_forecast',
					np.array( [ ], dtype=float ) )
			else:
				df_metrics = pd.DataFrame( )
				df_results = pd.DataFrame( )
				forecast = np.array( [ ], dtype=float )
			
			# ------------------------------------------------------------------
			# MODEL EVALUATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Model Evaluation' )
			
			if not df_metrics.empty:
				st.data_editor( df_metrics, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run Lagged Linear Regression to view model metrics.' )
			
			# ------------------------------------------------------------------
			# FORECAST RESULTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Forecast Results' )
			
			if not df_results.empty:
				st.data_editor( df_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run Lagged Linear Regression to view forecast values.' )
			
			# ------------------------------------------------------------------
			# FORECAST VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Observed vs Forecast' )
			if len( forecast ) > 0:
				plt.close( 'all' )
				fig, ax = plt.subplots( )
				observed_index = np.arange( len( series ) )
				forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
				ax.plot( observed_index, series, label='Observed' )
				ax.plot( forecast_index, forecast, label='Forecast', linestyle='--' )
				ax.set_xlabel( 'Period' )
				ax.set_ylabel( series_col )
				ax.set_title( 'Lagged Linear Regression Forecast' )
				ax.legend( )
				st.pyplot( fig )
				plt.close( fig )
			else:
				st.info( 'Run Lagged Linear Regression to view the forecast plot.' )
		
		# ------------------------------------------------------------------
		# LAGGED BOOSTING REGRESSION
		# ------------------------------------------------------------------
		with st.expander( 'Lagged Boosting Regression', expanded=False ):
			lag_boost_defaults = { 'timeseries_lag_boost_lag': 12,
				'timeseries_lag_boost_loss': 'squared_error', 'timeseries_lag_boost_quantile': 0.5,
				'timeseries_lag_boost_rate': 0.1, 'timeseries_lag_boost_iters': 100,
				'timeseries_lag_boost_leaf_nodes': 31, 'timeseries_lag_boost_use_depth': False,
				'timeseries_lag_boost_depth': 3, 'timeseries_lag_boost_leaf': 20,
				'timeseries_lag_boost_regularization': 0.0, 'timeseries_lag_boost_features': 1.0,
				'timeseries_lag_boost_bins': 255, 'timeseries_lag_boost_warm': False,
				'timeseries_lag_boost_stopping': 'auto', 'timeseries_lag_boost_scoring': 'loss',
				'timeseries_lag_boost_validation': 0.1, 'timeseries_lag_boost_no_change': 10,
				'timeseries_lag_boost_tolerance': 0.0000001, 'timeseries_lag_boost_verbose': 0,
				'timeseries_lag_boost_random_state': 42, 'timeseries_lag_boost_horizon': 5 }
			
			for key, value in lag_boost_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			maximum_lag = max( 1, len( series ) - 1 )
			st.session_state[ 'timeseries_lag_boost_lag' ] = min(
				int( st.session_state[ 'timeseries_lag_boost_lag' ] ), maximum_lag )
			
			st.caption( 'Nonlinear autoregressive forecasting using histogram gradient boosting over '
				'lagged observations.' )
			
			lb_c1, lb_c2, lb_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with lb_c1:
				lag_boost_order = int( st.number_input( 'Lag Order', min_value=1,
					max_value=max( 1, len( series ) - 1 ), step=1,
					key='timeseries_lag_boost_lag' ) )
				
				lag_boost_loss = st.selectbox( 'Loss',
					options=[ 'squared_error', 'absolute_error', 'gamma', 'poisson', 'quantile' ],
					key='timeseries_lag_boost_loss' )
				
				lag_boost_quantile = float(
					st.number_input( 'Quantile', min_value=0.0001, max_value=0.9999, step=0.05,
						format='%.4f', disabled=lag_boost_loss != 'quantile',
						key='timeseries_lag_boost_quantile' ) )
				
				lag_boost_rate = float(
					st.number_input( 'Learning Rate', min_value=0.0001, step=0.01, format='%.4f',
						key='timeseries_lag_boost_rate' ) )
				
				lag_boost_iters = int( st.number_input( 'Maximum Iterations', min_value=1, step=1,
					key='timeseries_lag_boost_iters' ) )
				
				lag_boost_leaf_nodes = int(
					st.number_input( 'Maximum Leaf Nodes', min_value=2, step=1,
						key='timeseries_lag_boost_leaf_nodes' ) )
				
				lag_boost_horizon = int( st.number_input( 'Forecast Horizon', min_value=1, step=1,
					key='timeseries_lag_boost_horizon' ) )
			
			with lb_c2:
				lag_boost_use_depth = st.checkbox( 'Limit Tree Depth',
					key='timeseries_lag_boost_use_depth' )
				
				lag_boost_depth_input = int( st.number_input( 'Maximum Depth', min_value=1, step=1,
					disabled=not lag_boost_use_depth, key='timeseries_lag_boost_depth' ) )
				
				lag_boost_leaf = int(
					st.number_input( 'Minimum Samples per Leaf', min_value=1, step=1,
						key='timeseries_lag_boost_leaf' ) )
				
				lag_boost_regularization = float(
					st.number_input( 'L2 Regularization', min_value=0.0, step=0.1, format='%.4f',
						key='timeseries_lag_boost_regularization' ) )
				
				lag_boost_features = float(
					st.number_input( 'Maximum Features', min_value=0.0001, max_value=1.0,
						step=0.05,
						format='%.4f', key='timeseries_lag_boost_features' ) )
				
				lag_boost_bins = int(
					st.number_input( 'Maximum Bins', min_value=2, max_value=255, step=1,
						key='timeseries_lag_boost_bins' ) )
				
				lag_boost_warm = st.checkbox( 'Warm Start', key='timeseries_lag_boost_warm' )
			
			with lb_c3:
				lag_boost_stopping = st.selectbox( 'Early Stopping',
					options=[ 'auto', True, False ], key='timeseries_lag_boost_stopping' )
				
				lag_boost_scoring = st.selectbox( 'Early-Stopping Scoring',
					options=[ 'loss', 'r2', 'neg_mean_absolute_error', 'neg_mean_squared_error' ],
					key='timeseries_lag_boost_scoring' )
				
				lag_boost_validation = float( st.number_input( 'Validation Fraction', min_value=0.01,
					max_value=0.99, step=0.01, format='%.4f', key='timeseries_lag_boost_validation' ) )
				
				lag_boost_no_change = int( st.number_input( 'Iterations Without Improvement',
					min_value=1, step=1, key='timeseries_lag_boost_no_change' ) )
				
				lag_boost_tolerance = float( st.number_input( 'Tolerance', min_value=0.0,
					step=0.0000001, format='%.8f', key='timeseries_lag_boost_tolerance' ) )
				
				lag_boost_verbose = int( st.number_input( 'Verbosity', min_value=0, step=1,
					key='timeseries_lag_boost_verbose' ) )
				
				lag_boost_random_state = int( st.number_input( 'Random State', min_value=0, step=1,
					key='timeseries_lag_boost_random_state' ) )
			
			lag_boost_reset_keys = list( lag_boost_defaults.keys( ) ) + [
				'timeseries_lag_boost_model', 'timeseries_lag_boost_forecast',
				'timeseries_lag_boost_metrics', 'timeseries_lag_boost_results',
				'timeseries_lag_boost_details', 'timeseries_lag_boost_signature' ]
			
			lb_b1, lb_b2 = st.columns( 2 )
			with lb_b1:
				run_lag_boost = st.button( 'Run Lagged Boosting Regression', icon='🏃',
					key='timeseries_lag_boost_run', use_container_width=True )
			
			with lb_b2:
				st.button( 'Reset Lagged Boosting Regression', icon='🔁',
					key='timeseries_lag_boost_reset', use_container_width=True,
					on_click=clear_keys,
					args=(lag_boost_reset_keys,) )
			
			if run_lag_boost:
				try:
					if lag_boost_order >= len( series ):
						st.warning( '⚠️ Lag Order must be less than the number of observations.' )
						st.stop( )
					
					if lag_boost_loss == 'quantile':
						quantile = lag_boost_quantile
					else:
						quantile = None
					
					if lag_boost_loss in [ 'gamma', 'poisson' ] and np.any( series <= 0 ):
						st.warning(
							'⚠️ Gamma and Poisson loss require strictly positive observations.' )
						st.stop( )
					
					if (lag_boost_stopping is not False and lag_boost_validation * (
							len( series ) - lag_boost_order) < 1):
						st.warning( '⚠️ Validation Fraction is too small for the available lagged '
						            'observations.' )
						st.stop( )
					
					max_depth = (lag_boost_depth_input if lag_boost_use_depth else None)
					model = LagBoostingSeries( lag=lag_boost_order, loss=lag_boost_loss,
						quantile=quantile, rate=lag_boost_rate, iters=lag_boost_iters,
						leaf_nodes=lag_boost_leaf_nodes, depth=max_depth, leaf=lag_boost_leaf,
						regularization=lag_boost_regularization, features=lag_boost_features,
						bins=lag_boost_bins, monotonic=None, interaction=None, warm=lag_boost_warm,
						stopping=lag_boost_stopping, scoring=lag_boost_scoring,
						validation=lag_boost_validation, no_change=lag_boost_no_change,
						tol=lag_boost_tolerance, verbose=lag_boost_verbose,
						rando=lag_boost_random_state )
					
					start_time = time.time( )
					model.train( series )
					metrics = model.analyze( )
					forecast = model.project( n_steps=lag_boost_horizon )
					elapsed_seconds = time.time( ) - start_time
					df_metrics = pd.DataFrame(
						[ { 'Metric': metric, 'Value': float( value ) } for metric, value in
							metrics.items( ) ] )
					
					df_metrics = pd.concat( [ df_metrics, pd.DataFrame(
						[ { 'Metric': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] ) ], ignore_index=True )
					
					forecast = np.asarray( forecast, dtype=float ).reshape( -1 )
					forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
					df_results = pd.DataFrame( { 'Period': forecast_index, 'Forecast': forecast } )
					detail_rows = [ { 'Property': 'Lag Order', 'Value': model.lag },
						{ 'Property': 'Loss', 'Value': model.loss },
						{ 'Property': 'Quantile', 'Value': model.quantile },
						{ 'Property': 'Learning Rate', 'Value': model.learning_rate },
						{ 'Property': 'Maximum Iterations', 'Value': model.max_iter },
						{ 'Property': 'Maximum Leaf Nodes', 'Value': model.max_leaf_nodes },
						{ 'Property': 'Maximum Depth', 'Value': model.max_depth },
						{ 'Property': 'Minimum Samples per Leaf', 'Value':
							model.min_samples_leaf },
						{ 'Property': 'L2 Regularization', 'Value': model.l2_regularization },
						{ 'Property': 'Maximum Features', 'Value': model.max_features },
						{ 'Property': 'Maximum Bins', 'Value': model.max_bins },
						{ 'Property': 'Warm Start', 'Value': model.warm_start },
						{ 'Property': 'Early Stopping', 'Value': model.early_stopping },
						{ 'Property': 'Scoring', 'Value': model.scoring },
						{ 'Property': 'Validation Fraction', 'Value': model.validation_fraction },
						{ 'Property': 'Iterations Without Improvement', 'Value': model.n_iter_no_change },
						{ 'Property': 'Tolerance', 'Value': model.tol },
						{ 'Property': 'Random State', 'Value': model.random_state } ]
					
					df_details = pd.DataFrame( detail_rows )
					lag_boost_signature = (series_col, len( series ), series_signature,
						lag_boost_order, lag_boost_loss, quantile, lag_boost_rate, lag_boost_iters,
						lag_boost_leaf_nodes, max_depth, lag_boost_leaf, lag_boost_regularization,
						lag_boost_features, lag_boost_bins, lag_boost_warm, lag_boost_stopping,
						lag_boost_scoring, lag_boost_validation, lag_boost_no_change,
						lag_boost_tolerance, lag_boost_verbose, lag_boost_random_state,
						lag_boost_horizon)
					
					st.session_state[ 'timeseries_lag_boost_model' ] = model
					st.session_state[ 'timeseries_lag_boost_forecast' ] = forecast.copy( )
					st.session_state[ 'timeseries_lag_boost_metrics' ] = df_metrics.copy( )
					st.session_state[ 'timeseries_lag_boost_results' ] = df_results.copy( )
					st.session_state[ 'timeseries_lag_boost_details' ] = df_details.copy( )
					st.session_state[ 'timeseries_lag_boost_signature' ] = lag_boost_signature
					st.success( 'Lagged Boosting Regression forecasting complete.' )
				except Exception as ex:
					for key in [ 'timeseries_lag_boost_model', 'timeseries_lag_boost_forecast',
						'timeseries_lag_boost_metrics', 'timeseries_lag_boost_results',
						'timeseries_lag_boost_details', 'timeseries_lag_boost_signature' ]:
						st.session_state.pop( key, None )
					
					st.error( f'Lagged Boosting Regression failed: {ex}' )
			
			current_quantile = (lag_boost_quantile if lag_boost_loss == 'quantile' else None)
			current_max_depth = (lag_boost_depth_input if lag_boost_use_depth else None)
			current_lag_boost_signature = (series_col, len( series ), series_signature,
				lag_boost_order, lag_boost_loss, current_quantile, lag_boost_rate, lag_boost_iters,
				lag_boost_leaf_nodes, current_max_depth, lag_boost_leaf, lag_boost_regularization,
				lag_boost_features, lag_boost_bins, lag_boost_warm, lag_boost_stopping,
				lag_boost_scoring, lag_boost_validation, lag_boost_no_change, lag_boost_tolerance,
				lag_boost_verbose, lag_boost_random_state, lag_boost_horizon)
			
			lag_boost_signature = st.session_state.get( 'timeseries_lag_boost_signature', None )
			if lag_boost_signature == current_lag_boost_signature:
				df_metrics = st.session_state.get( 'timeseries_lag_boost_metrics', pd.DataFrame( ) )
				df_results = st.session_state.get( 'timeseries_lag_boost_results', pd.DataFrame( ) )
				df_details = st.session_state.get( 'timeseries_lag_boost_details', pd.DataFrame( ) )
				forecast = st.session_state.get( 'timeseries_lag_boost_forecast',
					np.array( [ ], dtype=float ) )
			else:
				df_metrics = pd.DataFrame( )
				df_results = pd.DataFrame( )
				df_details = pd.DataFrame( )
				forecast = np.array( [ ], dtype=float )
			
			# ------------------------------------------------------------------
			# MODEL EVALUATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Model Evaluation' )
			
			if not df_metrics.empty:
				st.data_editor( df_metrics, use_container_width=True, hide_index=True,
					disabled=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True, hide_index=True,
						disabled=True )
			else:
				st.info( 'Run Lagged Boosting Regression to view model metrics.' )
			
			# ------------------------------------------------------------------
			# FORECAST RESULTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Forecast Results' )
			
			if not df_results.empty:
				st.data_editor( df_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run Lagged Boosting Regression to view forecast values.' )
			
			# ------------------------------------------------------------------
			# FORECAST VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Observed vs Forecast' )
			
			if len( forecast ) > 0:
				plt.close( 'all' )
				fig, ax = plt.subplots( )
				observed_index = np.arange( len( series ) )
				forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
				ax.plot( observed_index, series, label='Observed' )
				ax.plot( forecast_index, forecast, label='Forecast', linestyle='--' )
				ax.set_xlabel( 'Period' )
				ax.set_ylabel( series_col )
				ax.set_title( 'Lagged Boosting Regression Forecast' )
				ax.legend( )
				st.pyplot( fig )
				plt.close( fig )
			else:
				st.info( 'Run Lagged Boosting Regression to view the forecast plot.' )
		
		# ------------------------------------------------------------------
		# LAGGED QUANTILE REGRESSION
		# ------------------------------------------------------------------
		with st.expander( 'Lagged Quantile Regression', expanded=False ):
			lag_quantile_defaults = { 'timeseries_lag_quantile_lag': 12,
				'timeseries_lag_quantile_quantile': 0.5, 'timeseries_lag_quantile_alpha': 1.0,
				'timeseries_lag_quantile_fit': True, 'timeseries_lag_quantile_solver': 'highs',
				'timeseries_lag_quantile_horizon': 5 }
			
			for key, value in lag_quantile_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			maximum_lag = max( 1, len( series ) - 1 )
			st.session_state[ 'timeseries_lag_quantile_lag' ] = min( int(
				st.session_state[ 'timeseries_lag_quantile_lag' ] ), maximum_lag )
			
			st.caption( 'Conditional quantile forecasting using linear regression over lagged '
			            'observations.' )
			
			lq_c1, lq_c2, lq_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with lq_c1:
				lag_quantile_order = int( st.number_input( 'Lag Order', min_value=1,
					max_value=max( 1, len( series ) - 1 ), step=1,
					key='timeseries_lag_quantile_lag' ) )
				
				lag_quantile_value = float(
					st.number_input( 'Conditional Quantile', min_value=0.0001, max_value=0.9999,
						step=0.05, format='%.4f', key='timeseries_lag_quantile_quantile' ) )
			
			with lq_c2:
				lag_quantile_alpha = float( st.number_input( 'L1 Regularization', min_value=0.0,
					step=0.1, format='%.4f', key='timeseries_lag_quantile_alpha' ) )
				
				lag_quantile_fit = st.checkbox( 'Fit Intercept',
					key='timeseries_lag_quantile_fit' )
			
			with lq_c3:
				lag_quantile_solver = st.selectbox( 'Solver',
					options=[ 'highs', 'highs-ds', 'highs-ipm', 'interior-point',
						'revised simplex' ], key='timeseries_lag_quantile_solver' )
				
				lag_quantile_horizon = int(
					st.number_input( 'Forecast Horizon', min_value=1, step=1,
						key='timeseries_lag_quantile_horizon' ) )
			
			lag_quantile_reset_keys = list( lag_quantile_defaults.keys( ) ) + [
				'timeseries_lag_quantile_model', 'timeseries_lag_quantile_forecast',
				'timeseries_lag_quantile_metrics', 'timeseries_lag_quantile_results',
				'timeseries_lag_quantile_details', 'timeseries_lag_quantile_signature' ]
			
			lq_b1, lq_b2 = st.columns( 2 )
			with lq_b1:
				run_lag_quantile = st.button( 'Run Lagged Quantile Regression', icon='🏃',
					key='timeseries_lag_quantile_run', use_container_width=True )
			
			with lq_b2:
				st.button( 'Reset Lagged Quantile Regression', icon='🔁',
					key='timeseries_lag_quantile_reset', use_container_width=True,
					on_click=clear_keys, args=(lag_quantile_reset_keys,) )
			
			if run_lag_quantile:
				try:
					if lag_quantile_order >= len( series ):
						st.warning( '⚠️ Lag Order must be less than the number of observations.' )
						st.stop( )
					
					if not 0.0 < lag_quantile_value < 1.0:
						st.warning( '⚠️ Conditional Quantile must be strictly between 0 and 1.' )
						st.stop( )
					
					if lag_quantile_alpha < 0.0:
						st.warning( '⚠️ L1 Regularization cannot be negative.' )
						st.stop( )
					
					model = LagQuantileSeries( lag=lag_quantile_order, quantile=lag_quantile_value,
						alpha=lag_quantile_alpha, fit=lag_quantile_fit, solver=lag_quantile_solver,
						solver_options=None )
					
					start_time = time.time( )
					model.train( series )
					metrics = model.analyze( )
					forecast = model.project( n_steps=lag_quantile_horizon )
					elapsed_seconds = time.time( ) - start_time
					if metrics is None:
						metrics = { }
					
					df_metrics = pd.DataFrame(
						[ { 'Metric': metric, 'Value': float( value ) } for metric, value in
							metrics.items( ) ] )
					
					df_metrics = pd.concat( [ df_metrics, pd.DataFrame(
						[ { 'Metric': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] ) ], ignore_index=True )
					
					forecast = np.asarray( forecast, dtype=float ).reshape( -1 )
					forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
					df_results = pd.DataFrame( { 'Period': forecast_index, 'Quantile': lag_quantile_value,
							'Forecast': forecast } )
					
					detail_rows = [ { 'Property': 'Lag Order', 'Value': model.lag },
						{ 'Property': 'Conditional Quantile', 'Value': model.quantile },
						{ 'Property': 'L1 Regularization', 'Value': model.alpha },
						{ 'Property': 'Fit Intercept', 'Value': model.fit_intercept },
						{ 'Property': 'Solver', 'Value': model.solver },
						{ 'Property': 'Training Samples', 'Value': len( model.training_values ) },
						{ 'Property': 'Forecast Horizon', 'Value': lag_quantile_horizon } ]
					
					df_details = pd.DataFrame( detail_rows )
					lag_quantile_signature = (series_col, len( series ), series_signature,
						lag_quantile_order, lag_quantile_value, lag_quantile_alpha,
						lag_quantile_fit, lag_quantile_solver, lag_quantile_horizon)
					
					st.session_state[ 'timeseries_lag_quantile_model' ] = model
					st.session_state[ 'timeseries_lag_quantile_forecast' ] = forecast.copy( )
					st.session_state[ 'timeseries_lag_quantile_metrics' ] = df_metrics.copy( )
					st.session_state[ 'timeseries_lag_quantile_results' ] = df_results.copy( )
					st.session_state[ 'timeseries_lag_quantile_details' ] = df_details.copy( )
					st.session_state[ 'timeseries_lag_quantile_signature' ] = lag_quantile_signature
					st.success( 'Lagged Quantile Regression forecasting complete.' )
				except Exception as ex:
					for key in [ 'timeseries_lag_quantile_model',
						'timeseries_lag_quantile_forecast', 'timeseries_lag_quantile_metrics',
						'timeseries_lag_quantile_results', 'timeseries_lag_quantile_details',
						'timeseries_lag_quantile_signature' ]:
						st.session_state.pop( key, None )
					
					st.error( f'Lagged Quantile Regression failed: {ex}' )
			
			current_lag_quantile_signature = (series_col, len( series ), series_signature,
				lag_quantile_order, lag_quantile_value, lag_quantile_alpha, lag_quantile_fit,
				lag_quantile_solver, lag_quantile_horizon)
			
			lag_quantile_signature = st.session_state.get( 'timeseries_lag_quantile_signature',
				None )
			
			if lag_quantile_signature == current_lag_quantile_signature:
				df_metrics = st.session_state.get( 'timeseries_lag_quantile_metrics',
					pd.DataFrame( ) )
				df_results = st.session_state.get( 'timeseries_lag_quantile_results',
					pd.DataFrame( ) )
				df_details = st.session_state.get( 'timeseries_lag_quantile_details',
					pd.DataFrame( ) )
				forecast = st.session_state.get( 'timeseries_lag_quantile_forecast',
					np.array( [ ], dtype=float ) )
			else:
				df_metrics = pd.DataFrame( )
				df_results = pd.DataFrame( )
				df_details = pd.DataFrame( )
				forecast = np.array( [ ], dtype=float )
			
			# ------------------------------------------------------------------
			# MODEL EVALUATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Model Evaluation' )
			
			if not df_metrics.empty:
				st.data_editor( df_metrics, use_container_width=True, hide_index=True,
					disabled=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True, hide_index=True,
						disabled=True )
			else:
				st.info( 'Run Lagged Quantile Regression to view model metrics.' )
			
			# ------------------------------------------------------------------
			# FORECAST RESULTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Forecast Results' )
			
			if not df_results.empty:
				st.data_editor( df_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run Lagged Quantile Regression to view forecast values.' )
			
			# ------------------------------------------------------------------
			# FORECAST VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Observed vs Forecast' )
			
			if len( forecast ) > 0:
				plt.close( 'all' )
				fig, ax = plt.subplots( )
				observed_index = np.arange( len( series ) )
				forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
				ax.plot( observed_index, series, label='Observed' )
				ax.plot( forecast_index, forecast, label=f'Quantile {lag_quantile_value:.2f}',
					linestyle='--' )
				ax.set_xlabel( 'Period' )
				ax.set_ylabel( series_col )
				ax.set_title( 'Lagged Quantile Regression Forecast' )
				ax.legend( )
				st.pyplot( fig )
				plt.close( fig )
			else:
				st.info( 'Run Lagged Quantile Regression to view the forecast plot.' )
		
		# ------------------------------------------------------------------
		# ARIMA
		# ------------------------------------------------------------------
		with st.expander( 'ARIMA', expanded=False ):
			arima_defaults = { 'timeseries_arima_p': 1, 'timeseries_arima_d': 0,
				'timeseries_arima_q': 0, 'timeseries_arima_horizon': 5 }
			
			for key, value in arima_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			st.caption( 'Nonseasonal autoregressive integrated moving-average forecasting.' )
			ar_c1, ar_c2, ar_c3, ar_c4 = st.columns( 4, border=True )
			with ar_c1:
				arima_p = int( st.number_input( 'Autoregressive Order (p)', min_value=0, step=1,
					key='timeseries_arima_p' ) )
			
			with ar_c2:
				arima_d = int( st.number_input( 'Differencing Order (d)', min_value=0, step=1,
					key='timeseries_arima_d' ) )
			
			with ar_c3:
				arima_q = int( st.number_input( 'Moving-Average Order (q)', min_value=0, step=1,
					key='timeseries_arima_q' ) )
			
			with ar_c4:
				arima_horizon = int( st.number_input( 'Forecast Horizon', min_value=1, step=1,
					key='timeseries_arima_horizon' ) )
			
			arima_reset_keys = list( arima_defaults.keys( ) ) + [ 'timeseries_arima_model',
				'timeseries_arima_forecast', 'timeseries_arima_metrics',
				'timeseries_arima_results', 'timeseries_arima_details', 'timeseries_arima_signature' ]
			
			ar_b1, ar_b2 = st.columns( 2 )
			with ar_b1:
				run_arima = st.button( 'Run ARIMA', icon='🏃', key='timeseries_arima_run',
					use_container_width=True )
			
			with ar_b2:
				st.button( 'Reset ARIMA', icon='🔁', key='timeseries_arima_reset',
					use_container_width=True, on_click=clear_keys, args=(arima_reset_keys,) )
			
			if run_arima:
				try:
					minimum_observations = max( arima_p, arima_q, 1 )
					if len( series ) <= minimum_observations:
						st.warning(
							'⚠️ The selected series does not contain enough observations for the '
							'ARIMA order.' )
						st.stop( )
					
					if arima_p == 0 and arima_d == 0 and arima_q == 0:
						st.warning( '⚠️ At least one ARIMA order must be greater than zero.' )
						st.stop( )
					
					arima_order = (arima_p, arima_d, arima_q)
					model = ARIMA( order=arima_order )
					start_time = time.time( )
					model.train( series )
					metrics = model.analyze( )
					forecast = model.project( n_steps=arima_horizon )
					elapsed_seconds = time.time( ) - start_time
					if metrics is None:
						metrics = { }
					
					df_metrics = pd.DataFrame(
						[ { 'Metric': metric, 'Value': float( value ) } for metric, value in
							metrics.items( ) ] )
					
					df_metrics = pd.concat( [ df_metrics, pd.DataFrame(
						[ { 'Metric': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] ) ], ignore_index=True )
					
					forecast = np.asarray( forecast, dtype=float ).reshape( -1 )
					forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
					df_results = pd.DataFrame( { 'Period': forecast_index, 'Forecast': forecast } )
					detail_rows = [ { 'Property': 'Order', 'Value': str( model.order ) },
						{ 'Property': 'Autoregressive Order', 'Value': arima_p },
						{ 'Property': 'Differencing Order', 'Value': arima_d },
						{ 'Property': 'Moving-Average Order', 'Value': arima_q },
						{ 'Property': 'Training Observations', 'Value': len( model.train_data ) },
						{ 'Property': 'Forecast Horizon', 'Value': arima_horizon },
						{ 'Property': 'AIC', 'Value': float( model.results.aic ) },
						{ 'Property': 'BIC', 'Value': float( model.results.bic ) } ]
					
					df_details = pd.DataFrame( detail_rows )
					arima_signature = (series_col, len( series ), series_signature, arima_p,
						arima_d, arima_q, arima_horizon)
					
					st.session_state[ 'timeseries_arima_model' ] = model
					st.session_state[ 'timeseries_arima_forecast' ] = forecast.copy( )
					st.session_state[ 'timeseries_arima_metrics' ] = df_metrics.copy( )
					st.session_state[ 'timeseries_arima_results' ] = df_results.copy( )
					st.session_state[ 'timeseries_arima_details' ] = df_details.copy( )
					st.session_state[ 'timeseries_arima_signature' ] = arima_signature
					st.success( 'ARIMA forecasting complete.' )
				except Exception as ex:
					for key in [ 'timeseries_arima_model', 'timeseries_arima_forecast',
						'timeseries_arima_metrics', 'timeseries_arima_results',
						'timeseries_arima_details', 'timeseries_arima_signature' ]:
						st.session_state.pop( key, None )
					
					st.error( f'ARIMA forecasting failed: {ex}' )
			
			current_arima_signature = (series_col, len( series ), series_signature, arima_p,
				arima_d, arima_q, arima_horizon)
			
			arima_signature = st.session_state.get( 'timeseries_arima_signature', None )
			if arima_signature == current_arima_signature:
				df_metrics = st.session_state.get( 'timeseries_arima_metrics', pd.DataFrame( ) )
				df_results = st.session_state.get( 'timeseries_arima_results', pd.DataFrame( ) )
				df_details = st.session_state.get( 'timeseries_arima_details', pd.DataFrame( ) )
				forecast = st.session_state.get( 'timeseries_arima_forecast',
					np.array( [ ], dtype=float ) )
			else:
				df_metrics = pd.DataFrame( )
				df_results = pd.DataFrame( )
				df_details = pd.DataFrame( )
				forecast = np.array( [ ], dtype=float )
			
			# ------------------------------------------------------------------
			# MODEL EVALUATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Model Evaluation' )
			
			if not df_metrics.empty:
				st.data_editor( df_metrics, use_container_width=True, hide_index=True,
					disabled=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True, hide_index=True,
						disabled=True )
			else:
				st.info( 'Run ARIMA to view model metrics.' )
			
			# ------------------------------------------------------------------
			# FORECAST RESULTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Forecast Results' )
			
			if not df_results.empty:
				st.data_editor( df_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run ARIMA to view forecast values.' )
			
			# ------------------------------------------------------------------
			# FORECAST VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Observed vs Forecast' )
			
			if len( forecast ) > 0:
				plt.close( 'all' )
				fig, ax = plt.subplots( )
				
				observed_index = np.arange( len( series ) )
				forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
				
				ax.plot( observed_index, series, label='Observed' )
				ax.plot( forecast_index, forecast, label='Forecast', linestyle='--' )
				ax.set_xlabel( 'Period' )
				ax.set_ylabel( series_col )
				ax.set_title( f'ARIMA{(arima_p, arima_d, arima_q)} Forecast' )
				ax.legend( )
				
				st.pyplot( fig )
				plt.close( fig )
			else:
				st.info( 'Run ARIMA to view the forecast plot.' )
		
		# ------------------------------------------------------------------
		# SARIMA
		# ------------------------------------------------------------------
		with st.expander( 'SARIMA', expanded=False ):
			sarima_defaults = { 'timeseries_sarima_p': 1, 'timeseries_sarima_d': 0,
				'timeseries_sarima_q': 0, 'timeseries_sarima_seasonal_p': 0,
				'timeseries_sarima_seasonal_d': 0, 'timeseries_sarima_seasonal_q': 0,
				'timeseries_sarima_seasonal_period': 0, 'timeseries_sarima_horizon': 5 }
			
			for key, value in sarima_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			st.caption( 'Seasonal autoregressive integrated moving-average forecasting.' )
			sa_c1, sa_c2, sa_c3, sa_c4 = st.columns( 4, border=True )
			with sa_c1:
				sarima_p = int( st.number_input( 'Autoregressive Order (p)', min_value=0, step=1,
					key='timeseries_sarima_p' ) )
				
				sarima_seasonal_p = int(
					st.number_input( 'Seasonal Autoregressive Order (P)', min_value=0, step=1,
						key='timeseries_sarima_seasonal_p' ) )
			
			with sa_c2:
				sarima_d = int( st.number_input( 'Differencing Order (d)', min_value=0, step=1,
					key='timeseries_sarima_d' ) )
				
				sarima_seasonal_d = int(
					st.number_input( 'Seasonal Differencing Order (D)', min_value=0, step=1,
						key='timeseries_sarima_seasonal_d' ) )
			
			with sa_c3:
				sarima_q = int( st.number_input( 'Moving-Average Order (q)', min_value=0, step=1,
					key='timeseries_sarima_q' ) )
				
				sarima_seasonal_q = int(
					st.number_input( 'Seasonal Moving-Average Order (Q)', min_value=0, step=1,
						key='timeseries_sarima_seasonal_q' ) )
			
			with sa_c4:
				sarima_seasonal_period = int(
					st.number_input( 'Seasonal Period (s)', min_value=0, step=1,
						key='timeseries_sarima_seasonal_period' ) )
				
				sarima_horizon = int( st.number_input( 'Forecast Horizon', min_value=1, step=1,
					key='timeseries_sarima_horizon' ) )
			
			sarima_reset_keys = list( sarima_defaults.keys( ) ) + [ 'timeseries_sarima_model',
				'timeseries_sarima_forecast', 'timeseries_sarima_metrics',
				'timeseries_sarima_results', 'timeseries_sarima_details',
				'timeseries_sarima_signature' ]
			
			sa_b1, sa_b2 = st.columns( 2 )
			with sa_b1:
				run_sarima = st.button( 'Run SARIMA', icon='🏃', key='timeseries_sarima_run',
					use_container_width=True )
			
			with sa_b2:
				st.button( 'Reset SARIMA', icon='🔁', key='timeseries_sarima_reset',
					use_container_width=True, on_click=clear_keys, args=(sarima_reset_keys,) )
			
			if run_sarima:
				try:
					nonseasonal_order = (sarima_p, sarima_d, sarima_q)
					seasonal_terms = (sarima_seasonal_p, sarima_seasonal_d, sarima_seasonal_q)
					uses_seasonality = any( value > 0 for value in seasonal_terms )
					if (all( value == 0 for value in nonseasonal_order ) and not uses_seasonality):
						st.warning(
							'⚠️ At least one nonseasonal or seasonal order must be greater than '
							'zero.' )
						st.stop( )
					
					if uses_seasonality and sarima_seasonal_period < 2:
						st.warning(
							'⚠️ Seasonal Period must be at least 2 when seasonal orders are '
							'used.' )
						st.stop( )
					
					if (not uses_seasonality and sarima_seasonal_period not in [ 0, 1 ]):
						st.warning(
							'⚠️ Set Seasonal Period to 0 when no seasonal orders are used.' )
						st.stop( )
					
					effective_seasonal_period = (sarima_seasonal_period if uses_seasonality else 0)
					seasonal_order = (sarima_seasonal_p, sarima_seasonal_d, sarima_seasonal_q,
						effective_seasonal_period)
					
					minimum_observations = max( sarima_p + sarima_d + sarima_q,
						sarima_seasonal_p + sarima_seasonal_d + sarima_seasonal_q, 1 )
					
					if len( series ) <= minimum_observations:
						st.warning(
							'⚠️ The selected series does not contain enough observations for the '
							'SARIMA orders.' )
						st.stop( )
					
					if (uses_seasonality and len( series ) <= effective_seasonal_period):
						st.warning(
							'⚠️ The selected series must contain more observations than the '
							'Seasonal Period.' )
						st.stop( )
					
					model = SARIMA( order=nonseasonal_order, seasonal=seasonal_order )
					start_time = time.time( )
					model.train( series )
					metrics = model.analyze( )
					forecast = model.project( n_steps=sarima_horizon )
					elapsed_seconds = time.time( ) - start_time
					if metrics is None:
						metrics = { }
					
					df_metrics = pd.DataFrame(
						[ { 'Metric': metric, 'Value': float( value ) } for metric, value in
							metrics.items( ) ] )
					
					df_metrics = pd.concat( [ df_metrics, pd.DataFrame(
						[ { 'Metric': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] ) ], ignore_index=True )
					
					forecast = np.asarray( forecast, dtype=float ).reshape( -1 )
					forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
					df_results = pd.DataFrame( { 'Period': forecast_index, 'Forecast': forecast } )
					detail_rows = [ { 'Property': 'Order', 'Value': str( model.order ) },
						{ 'Property': 'Seasonal Order', 'Value': str( model.seasonal_order ) },
						{ 'Property': 'Training Observations',
							'Value': len( model.training_data ) },
						{ 'Property': 'Forecast Horizon', 'Value': sarima_horizon },
						{ 'Property': 'AIC', 'Value': float( model.results.aic ) },
						{ 'Property': 'BIC', 'Value': float( model.results.bic ) } ]
					
					df_details = pd.DataFrame( detail_rows )
					sarima_signature = (series_col, len( series ), series_signature, sarima_p,
						sarima_d, sarima_q, sarima_seasonal_p, sarima_seasonal_d,
						sarima_seasonal_q,
						effective_seasonal_period, sarima_horizon)
					
					st.session_state[ 'timeseries_sarima_model' ] = model
					st.session_state[ 'timeseries_sarima_forecast' ] = forecast.copy( )
					st.session_state[ 'timeseries_sarima_metrics' ] = df_metrics.copy( )
					st.session_state[ 'timeseries_sarima_results' ] = df_results.copy( )
					st.session_state[ 'timeseries_sarima_details' ] = df_details.copy( )
					st.session_state[ 'timeseries_sarima_signature' ] = sarima_signature
					st.success( 'SARIMA forecasting complete.' )
				except Exception as ex:
					for key in [ 'timeseries_sarima_model', 'timeseries_sarima_forecast',
						'timeseries_sarima_metrics', 'timeseries_sarima_results',
						'timeseries_sarima_details', 'timeseries_sarima_signature' ]:
						st.session_state.pop( key, None )
					
					st.error( f'SARIMA forecasting failed: {ex}' )
			
			current_uses_seasonality = any(
				value > 0 for value in (sarima_seasonal_p, sarima_seasonal_d, sarima_seasonal_q) )
			
			current_seasonal_period = (sarima_seasonal_period if current_uses_seasonality else 0)
			current_sarima_signature = (series_col, len( series ), series_signature, sarima_p,
				sarima_d, sarima_q, sarima_seasonal_p, sarima_seasonal_d, sarima_seasonal_q,
				current_seasonal_period, sarima_horizon)
			
			sarima_signature = st.session_state.get( 'timeseries_sarima_signature', None )
			if sarima_signature == current_sarima_signature:
				df_metrics = st.session_state.get( 'timeseries_sarima_metrics', pd.DataFrame( ) )
				df_results = st.session_state.get( 'timeseries_sarima_results', pd.DataFrame( ) )
				df_details = st.session_state.get( 'timeseries_sarima_details', pd.DataFrame( ) )
				forecast = st.session_state.get( 'timeseries_sarima_forecast',
					np.array( [ ], dtype=float ) )
			else:
				df_metrics = pd.DataFrame( )
				df_results = pd.DataFrame( )
				df_details = pd.DataFrame( )
				forecast = np.array( [ ], dtype=float )
			
			# ------------------------------------------------------------------
			# MODEL EVALUATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Model Evaluation' )
			
			if not df_metrics.empty:
				st.data_editor( df_metrics, use_container_width=True, hide_index=True,
					disabled=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					st.data_editor( df_details, use_container_width=True, hide_index=True,
						disabled=True )
			else:
				st.info( 'Run SARIMA to view model metrics.' )
			
			# ------------------------------------------------------------------
			# FORECAST RESULTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Forecast Results' )
			
			if not df_results.empty:
				st.data_editor( df_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run SARIMA to view forecast values.' )
			
			# ------------------------------------------------------------------
			# FORECAST VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Observed vs Forecast' )
			
			if len( forecast ) > 0:
				plt.close( 'all' )
				fig, ax = plt.subplots( )
				observed_index = np.arange( len( series ) )
				forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
				ax.plot( observed_index, series, label='Observed' )
				ax.plot( forecast_index, forecast, label='Forecast', linestyle='--' )
				ax.set_xlabel( 'Period' )
				ax.set_ylabel( series_col )
				ax.set_title( f'SARIMA{(sarima_p, sarima_d, sarima_q)}'
				              f'{(sarima_seasonal_p, sarima_seasonal_d, sarima_seasonal_q, current_seasonal_period)} Forecast' )
				ax.legend( )
				st.pyplot( fig )
				plt.close( fig )
			else:
				st.info( 'Run SARIMA to view the forecast plot.' )
		
		# ------------------------------------------------------------------
		# TIME-SERIES CROSS-VALIDATION
		# ------------------------------------------------------------------
		with st.expander( 'Time-Series Validation Windows', expanded=False ):
			splitter_defaults = { 'timeseries_splitter_splits': 5,
				'timeseries_splitter_use_train_size': False, 'timeseries_splitter_train_size': 20,
				'timeseries_splitter_use_test_size': False, 'timeseries_splitter_test_size': 5,
				'timeseries_splitter_gap': 0 }
			
			for key, value in splitter_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			maximum_split_count = max( 2, len( series ) - 1 )
			maximum_window_size = max( 1, len( series ) - 1 )
			st.session_state[ 'timeseries_splitter_splits' ] = min(
				max( 2, int( st.session_state[ 'timeseries_splitter_splits' ] ) ),
				maximum_split_count )
			st.session_state[ 'timeseries_splitter_train_size' ] = min(
				max( 1, int( st.session_state[ 'timeseries_splitter_train_size' ] ) ),
				maximum_window_size )
			st.session_state[ 'timeseries_splitter_test_size' ] = min(
				max( 1, int( st.session_state[ 'timeseries_splitter_test_size' ] ) ),
				maximum_window_size )
			st.session_state[ 'timeseries_splitter_gap' ] = min(
				max( 0, int( st.session_state[ 'timeseries_splitter_gap' ] ) ),
				max( 0, len( series ) - 2 ) )
			
			st.caption( 'Chronological windows for subsequent model evaluation without random shuffling.' )
			cv_c1, cv_c2, cv_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with cv_c1:
				splitter_splits = int( st.number_input( 'Number of Splits', min_value=2,
					max_value=max( 2, len( series ) - 1 ), step=1,
					key='timeseries_splitter_splits' ) )
				
				splitter_gap = int( st.number_input( 'Temporal Gap', min_value=0,
					max_value=max( 0, len( series ) - 2 ), step=1, key='timeseries_splitter_gap' ) )
			
			with cv_c2:
				splitter_use_train_size = st.checkbox( 'Limit Training Window',
					key='timeseries_splitter_use_train_size' )
				
				splitter_train_size_input = int( st.number_input( 'Maximum Training Size', min_value=1,
						max_value=max( 1, len( series ) - 1 ), step=1,
						disabled=not splitter_use_train_size, key='timeseries_splitter_train_size' ) )
			
			with cv_c3:
				splitter_use_test_size = st.checkbox( 'Use Fixed Test Size',
					key='timeseries_splitter_use_test_size' )
				
				splitter_test_size_input = int( st.number_input( 'Test Size', min_value=1,
					max_value=max( 1, len( series ) - 1 ), step=1,
					disabled=not splitter_use_test_size, key='timeseries_splitter_test_size' ) )
			
			splitter_reset_keys = list( splitter_defaults.keys( ) ) + [ 'timeseries_splitter_model',
				'timeseries_splitter_results', 'timeseries_splitter_summary',
				'timeseries_splitter_details', 'timeseries_splitter_figure',
				'timeseries_splitter_signature' ]
			
			cv_b1, cv_b2 = st.columns( 2 )
			with cv_b1:
				run_splitter = st.button( 'Generate Time-Series Splits', icon='🏃',
					key='timeseries_splitter_run', use_container_width=True )
			
			with cv_b2:
				st.button( 'Reset Time-Series Splits', icon='🔁', key='timeseries_splitter_reset',
					use_container_width=True, on_click=clear_keys, args=(splitter_reset_keys,) )
			
			if run_splitter:
				try:
					max_train_size = ( splitter_train_size_input if splitter_use_train_size else None)
					test_size = (splitter_test_size_input if splitter_use_test_size else None)
					computed_test_size = ( len( series ) // (splitter_splits + 1) if test_size is None else test_size)
					required_observations = (splitter_splits * computed_test_size + splitter_gap)
					if computed_test_size < 1:
						st.warning( '⚠️ The selected configuration produces an invalid test-window size.' )
						st.stop( )
					
					if required_observations >= len( series ):
						st.warning( '⚠️ The selected series does not contain enough observations for the '
							'requested splits, test size, and gap.' )
						st.stop( )
					
					splitter = TimeSeriesSpliter( splits=splitter_splits, max_train_size=max_train_size,
						test_size=test_size, gap=splitter_gap )
					
					start_time = time.time( )
					split_pairs = splitter.get_splits( series )
					figure = splitter.visualize( series )
					elapsed_seconds = time.time( ) - start_time
					
					if not split_pairs:
						st.warning( '⚠️ No time-series splits were generated.' )
						st.stop( )
					
					result_rows = [ ]
					summary_rows = [ ]
					for split_number, split_pair in enumerate( split_pairs, start=1 ):
						train_index, test_index = split_pair
						train_start = int( train_index[ 0 ] )
						train_end = int( train_index[ -1 ] )
						test_start = int( test_index[ 0 ] )
						test_end = int( test_index[ -1 ] )
						summary_rows.append( { 'Split': split_number, 'Train Start': train_start,
							'Train End': train_end, 'Train Size': len( train_index ),
							'Test Start': test_start, 'Test End': test_end,
							'Test Size': len( test_index ), 'Gap': splitter_gap } )
						
						for index in train_index:
							result_rows.append( { 'Split': split_number, 'Period': int( index ),
								'Value': float( series[ index ] ), 'Partition': 'Train' } )
						
						for index in test_index:
							result_rows.append( { 'Split': split_number, 'Period': int( index ),
								'Value': float( series[ index ] ), 'Partition': 'Test' } )
					
					df_split_results = pd.DataFrame( result_rows )
					df_split_summary = pd.DataFrame( summary_rows )
					df_processing = pd.DataFrame( [ { 'Property': 'Configured Splits',
						'Value': splitter.get_n_splits( series ) },
						{ 'Property': 'Generated Splits', 'Value': len( split_pairs ) },
						{ 'Property': 'Maximum Training Size', 'Value': max_train_size },
						{ 'Property': 'Test Size', 'Value': computed_test_size },
						{ 'Property': 'Temporal Gap', 'Value': splitter_gap },
						{ 'Property': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] )
					
					splitter_signature = (series_col, len( series ), series_signature,
						splitter_splits, max_train_size, test_size, splitter_gap)
					
					st.session_state[ 'timeseries_splitter_model' ] = splitter
					st.session_state[ 'timeseries_splitter_results' ] = df_split_results.copy( )
					st.session_state[ 'timeseries_splitter_summary' ] = df_split_summary.copy( )
					st.session_state[ 'timeseries_splitter_details' ] = df_processing.copy( )
					st.session_state[ 'timeseries_splitter_figure' ] = figure
					st.session_state[ 'timeseries_splitter_signature' ] = splitter_signature
					st.success( 'Time-series cross-validation splits generated.' )
				except Exception as ex:
					for key in [ 'timeseries_splitter_model', 'timeseries_splitter_results',
						'timeseries_splitter_summary', 'timeseries_splitter_details',
						'timeseries_splitter_figure', 'timeseries_splitter_signature' ]:
						st.session_state.pop( key, None )
					
					st.error( f'Time-Series Cross-Validation failed: {ex}' )
			
			current_max_train_size = ( splitter_train_size_input if splitter_use_train_size else None)
			current_test_size = (splitter_test_size_input if splitter_use_test_size else None)
			current_splitter_signature = (series_col, len( series ), series_signature,
				splitter_splits, current_max_train_size, current_test_size, splitter_gap)
			
			splitter_signature = st.session_state.get( 'timeseries_splitter_signature', None )
			if splitter_signature == current_splitter_signature:
				df_split_results = st.session_state.get( 'timeseries_splitter_results',
					pd.DataFrame( ) )
				df_split_summary = st.session_state.get( 'timeseries_splitter_summary',
					pd.DataFrame( ) )
				df_split_details = st.session_state.get( 'timeseries_splitter_details',
					pd.DataFrame( ) )
				split_figure = st.session_state.get( 'timeseries_splitter_figure', None )
			else:
				df_split_results = pd.DataFrame( )
				df_split_summary = pd.DataFrame( )
				df_split_details = pd.DataFrame( )
				split_figure = None
			
			# ------------------------------------------------------------------
			# SPLIT SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Split Summary' )
			
			if not df_split_summary.empty:
				st.data_editor( df_split_summary, use_container_width=True, hide_index=True,
					disabled=True )
				
				if not df_split_details.empty:
					st.caption( 'Splitter Details' )
					st.data_editor( df_split_details, use_container_width=True, hide_index=True,
						disabled=True )
			else:
				st.info( 'Generate time-series splits to view the train-and-test windows.' )
			
			# ------------------------------------------------------------------
			# SPLIT ASSIGNMENTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Split Assignments' )
			
			if not df_split_results.empty:
				st.data_editor( df_split_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Generate time-series splits to view individual period assignments.' )
			
			# ------------------------------------------------------------------
			# SPLIT VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Split Visualization' )
			
			if split_figure is not None:
				st.pyplot( split_figure )
			else:
				st.info( 'Generate time-series splits to view the validation windows.' )

# ============================================
# DATA MANAGEMENT MODE
# ============================================
elif mode == 'Data Management':
	st.subheader( cfg.MODE[ 'Data Management' ], divider='blue' )
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		tabs = st.tabs( [ 'Import', 'Browse', 'CRUD', 'Explore', 'Filter', 'Aggregate',
			'Visualize', 'Admin', 'SQL' ] )
		
		tables = list_tables( )
		if not tables:
			st.info( 'No tables available.' )
		
		# ------------------------------------------------------------------------------
		# UPLOAD TAB
		# ------------------------------------------------------------------------------
		with tabs[ 0 ]:
			st.header( '' )
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
								
								create_stmt = (f'CREATE TABLE "{table_name}" '
								               f'({", ".join( columns )});')
								
								conn.execute( create_stmt )
								
								# --- Insert Data ---
								placeholders = ", ".join( [ "?" ] * len( df.columns ) )
								insert_stmt = (f'INSERT INTO "{table_name}" '
								               f'VALUES ({placeholders});')
								
								conn.executemany( insert_stmt, df.where(
									pd.notnull( df ), None ).values.tolist( ) )
							
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
				st.header( '' )
				browse_left, browse_center, browse_right = st.columns( 3 )
				with browse_left:
					table = st.selectbox( 'Select Table:', tables, key='table_name' )
				
				blue_divider( )
				df = read_table( table )
				st.data_editor( df, use_container_width=True, height=400 )
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
				st.header( '' )
				st.markdown( '##### Data Table' )
				crud_left, crud_mid, crud_right = st.columns( 3 )
				with crud_left:
					table = st.selectbox( 'Select', tables, key='crud_table' )
				df = read_table( table )
				schema = create_schema( table )
				
				# ------------------------------------------------------------------
				# Build Type Map
				# ------------------------------------------------------------------
				type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != 'rowid' }
				
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				blue_divider( )
				st.markdown( '##### Insert Row' )
				insert_data = { }
				insert_columns = st.columns( 4 )
				
				for index, (column, col_type) in enumerate( type_map.items( ) ):
					target_column = insert_columns[ index % 4 ]
					
					with target_column:
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
							insert_data[ column ] = st.text_input( column,
								key=f'ins_{column}' )
				
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
				blue_divider( )
				st.markdown( '##### Update Row' )
				rowid = st.number_input( 'Row ID', min_value=1, step=1 )
				update_data = { }
				update_columns = st.columns( 4 )
				
				for index, (column, col_type) in enumerate( type_map.items( ) ):
					target_column = update_columns[ index % 4 ]
					
					with target_column:
						if 'INT' in col_type:
							val = st.number_input( column, step=1,
								key=f'upd_{column}' )
							update_data[ column ] = val
						
						elif 'REAL' in col_type:
							val = st.number_input( column, format='%.6f',
								key=f'upd_{column}' )
							update_data[ column ] = val
						
						elif 'BOOL' in col_type:
							val = 1 if st.checkbox( column,
								key=f'upd_{column}' ) else 0
							update_data[ column ] = val
						
						else:
							val = st.text_input( column, key=f'upd_{column}' )
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
				blue_divider( )
				st.markdown( '##### Delete Row' )
				delete_left, delete_mid, delete_right = st.columns( 3 )
				with delete_left:
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
			st.header( '' )
			tables = list_tables( )
			if tables:
				explore_c1, explore_c2, explore_c3 = st.columns( 3, border=True )			
				with explore_c1:
					table = st.selectbox( 'Table', tables, key='explore_table' )
				
				with explore_c2:
					page = st.number_input( 'Page', min_value=1, step=1 )
				
				with explore_c3:
					page_size = st.slider( 'Rows per page', 10, 500, 50 )
				
				blue_divider( )
				offset = (page - 1) * page_size
				df_page = read_table( table, page_size, offset )
				st.data_editor( df_page, use_container_width=True, height=400 )
		
		# ------------------------------------------------------------------------------
		# FILTER
		# ------------------------------------------------------------------------------
		with tabs[ 4 ]:
			st.header( '' )
			tables = list_tables( )
			if tables:
				filter_c1, filter_c2, filter_c3 = st.columns( 3, border=True )
				with filter_c1:
					table = st.selectbox( 'Table', tables, key='filter_table' )
					df = read_table( table )
				with filter_c2:
					column = st.selectbox( 'Filter Column', df.columns, key='filter_column_box' )
				with filter_c3:
					value = st.text_input( 'Column Criteria (Contains)' )
					if value:
						df = df[ df[ column ].astype( str ).str.contains( value ) ]
						
				blue_divider( )
				st.data_editor( df, use_container_width=True, key='filter_frame', height=400 )
		
		# ------------------------------------------------------------------------------
		# AGGREGATE
		# ------------------------------------------------------------------------------
		with tabs[ 5 ]:
			st.header( '' )
			tables = list_tables( )
			st.session_state.get( 'aggregation', None )
			if tables:
				agg_c1, agg_c2, agg_c3, agg_c4 = st.columns( 4, border=True )
				with agg_c1:
					table = st.selectbox( 'Table', tables, key='agg_table' )
					df = read_table( table )
				with agg_c2:
					numeric_columns = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
					if numeric_columns:
						col = st.selectbox( 'Column', numeric_columns, key='col_box' )
				with agg_c3:
					aggregation = st.selectbox( 'Function', [ 'SUM', 'AVG', 'COUNT' ], key='agg_box' )
				with agg_c4:
					if aggregation == 'SUM':
						st.metric( 'Result', df[ col ].sum( ) )
					elif aggregation == 'AVG':
						st.metric( 'Result', df[ col ].mean( ) )
					elif aggregation == 'COUNT':
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
			st.header( '' )
			df_profile = st.session_state.get( 'df_profile' )
			st.markdown( '##### Data Profiling' )
			tables = list_tables( )
			if tables:
				adm_c1, adm_c2, adm_c3 = st.columns( 3 )
				with adm_c1:
					table = st.selectbox( 'Select Table', tables, key='profile_table' )
					
				if st.button( label='Generate Profile', icon='⚡' ):
					df_profile = create_profile_table( table )
			
				st.data_editor( df_profile, use_container_width=True, height=400 )
			
			blue_divider( )
			st.markdown( '##### Drop Table' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table to Drop', tables, key='admin_drop_table' )
				
				# Initialize confirmation state
				if 'dm_confirm_drop' not in st.session_state:
					st.session_state.dm_confirm_drop = False
				
				# Step 1: Initial Drop click
				if st.button( label='Drop Table', key='admin_drop_button', icon='❌' ):
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
				
				if st.button( label='Create Index', icon='➕' ):
					create_index( table, col )
					st.success( 'Index created.' )
			
			blue_divider( )
			
			st.markdown( '##### Create Table' )
			new_table_name = st.text_input( 'Table Name' )
			column_count = st.number_input( 'Number of Columns', min_value=1, max_value=20,
				value=1 )
			columns = [ ]
			for i in range( column_count ):
				st.markdown( f'##### Column {i + 1}' )
				col_name = st.text_input( 'Column Name', key=f'col_name_{i}' )
				col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
					key=f'col_type_{i}' )
				
				not_null = st.checkbox( 'NOT NULL', key=f'not_null_{i}' )
				primary_key = st.checkbox( 'PRIMARY KEY', key=f'pk_{i}' )
				auto_inc = st.checkbox( 'AUTOINCREMENT (INTEGER only)', key=f'ai_{i}' )
				
				columns.append( { 'name': col_name, 'type': col_type, 'not_null': not_null,
					'primary_key': primary_key, 'auto_increment': auto_inc } )
			
			if st.button( label='Create Table', icon='➕' ):
				try:
					create_custom_table( new_table_name, columns )
					st.success( 'Table created successfully.' )
					st.rerun( )
				
				except Exception as e:
					st.error( f'Error: {e}' )
			
			blue_divider( )
			st.markdown( '##### Schema Viewer' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='schema_view_table' )
				schema = create_schema( table )
				schema_df = pd.DataFrame( schema,
					columns=[ 'cid', 'name', 'type', 'notnull', 'default', 'pk' ] )
				
				st.markdown( "##### Columns" )
				st.data_editor( schema_df, use_container_width=True,
					key='schema_editor', height=400 )
				
				# Row count
				with create_connection( ) as conn:
					count = conn.execute( f'SELECT COUNT(*) FROM "{table}"' ).fetchone( )[ 0 ]
				
				st.metric( "Row Count", f"{count:,}" )
				
				# Indexes
				indexes = get_indexes( table )
				if indexes:
					idx_df = pd.DataFrame( indexes,
						columns=[ 'seq', 'name', 'unique', 'origin', 'partial' ] )
					
					st.markdown( "##### Indexes" )
					st.data_editor( idx_df, use_container_width=True,
						key='schema_editor', height=400 )
				else:
					st.info( "No indexes defined." )
			
			blue_divider( )
			st.markdown( "##### ALTER TABLE" )
			
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
							st.download_button( 'Download CSV', csv, 'query_results.csv',
								'text/csv' )
					
					except Exception as e:
						st.error( f'Execution failed: {e}' )

# ============================================
# DATA UPLOAD MODE
# ============================================
elif mode == 'Data Upload':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Data Upload' ], divider='blue' )
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
							
							create_stmt = (f'CREATE TABLE "{table_name}" '
							               f'({", ".join( columns )});')
							
							conn.execute( create_stmt )
							
							# --- Insert Data ---
							placeholders = ", ".join( [ "?" ] * len( df.columns ) )
							insert_stmt = (f'INSERT INTO "{table_name}" '
							               f'VALUES ({placeholders});')
							
							conn.executemany( insert_stmt, df.where(
								pd.notnull( df ), None ).values.tolist( ) )
						
						conn.commit( )
					
					st.success( 'Import completed successfully (transaction committed).' )
					st.rerun( )
				
				except Exception as e:
					try:
						conn.rollback( )
					except:
						pass
					st.error( f'Import failed — transaction rolled back.\n\n{e}' )
		
# ============================================
# DATA BROWSE MODE
# ============================================
elif mode == 'Data Browse':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Data Browse' ], divider='blue' )
		tables = list_tables( )
		if tables:
			st.header( '' )
			browse_left, browse_center, browse_right = st.columns( 3 )
			with browse_left:
				table = st.selectbox( 'Select Table:', tables, key='table_name' )
			
			blue_divider( )
			df = read_table( table )
			st.data_editor( df, use_container_width=True, height=400 )
		else:
			st.info( 'No tables available.' )

# ============================================
# CRUD OPS MODE
# ============================================
elif mode == 'CRUD Ops':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'CRUD Ops' ], divider='blue' )
		tables = list_tables( )
		if not tables:
			st.info( 'No tables available.' )
		else:
			st.header( '' )
			st.markdown( '##### Data Table' )
			crud_left, crud_mid, crud_right = st.columns( 3 )
			with crud_left:
				table = st.selectbox( 'Select', tables, key='crud_table' )
			df = read_table( table )
			schema = create_schema( table )
			
			# ------------------------------------------------------------------
			# Build Type Map
			# ------------------------------------------------------------------
			type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != 'rowid' }
			
			# ------------------------------------------------------------------
			# INSERT
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Insert Row' )
			insert_data = { }
			insert_columns = st.columns( 4 )
			
			for index, (column, col_type) in enumerate( type_map.items( ) ):
				target_column = insert_columns[ index % 4 ]
				
				with target_column:
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
						insert_data[ column ] = st.text_input( column,
							key=f'ins_{column}' )
			
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
			blue_divider( )
			st.markdown( '##### Update Row' )
			rowid = st.number_input( 'Row ID', min_value=1, step=1 )
			update_data = { }
			update_columns = st.columns( 4 )
			
			for index, (column, col_type) in enumerate( type_map.items( ) ):
				target_column = update_columns[ index % 4 ]
				
				with target_column:
					if 'INT' in col_type:
						val = st.number_input( column, step=1,
							key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'REAL' in col_type:
						val = st.number_input( column, format='%.6f',
							key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'BOOL' in col_type:
						val = 1 if st.checkbox( column,
							key=f'upd_{column}' ) else 0
						update_data[ column ] = val
					
					else:
						val = st.text_input( column, key=f'upd_{column}' )
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
			blue_divider( )
			st.markdown( '##### Delete Row' )
			delete_left, delete_mid, delete_right = st.columns( 3 )
			with delete_left:
				delete_id = st.number_input( 'Row ID to Delete', min_value=1, step=1 )
				
			if st.button( 'Delete Row' ):
				with create_connection( ) as conn:
					conn.execute( f'DELETE FROM {table} WHERE rowid=?;', (delete_id,) )
					conn.commit( )
				
				st.success( 'Row deleted.' )
				st.rerun( )
	
# ============================================
# DATA FILTER MODE
# ============================================
elif mode == 'Data Filter':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Data Filter' ], divider='blue' )
		tables = list_tables( )
		if tables:
			explore_c1, explore_c2, explore_c3 = st.columns( 3, border=True )
			with explore_c1:
				table = st.selectbox( 'Table', tables, key='explore_table' )
			
			with explore_c2:
				page = st.number_input( 'Page', min_value=1, step=1 )
			
			with explore_c3:
				page_size = st.slider( 'Rows per page', 10, 500, 50 )
			
			blue_divider( )
			offset = (page - 1) * page_size
			df_page = read_table( table, page_size, offset )
			st.data_editor( df_page, use_container_width=True, height=400 )

# ============================================
# DATA AGGREGATION MODE
# ============================================
elif mode == 'Data Aggregation':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Data Aggregation' ], divider='blue' )
		tables = list_tables( )
		st.session_state.get( 'aggregation', None )
		if tables:
			agg_c1, agg_c2, agg_c3, agg_c4 = st.columns( 4, border=True )
			with agg_c1:
				table = st.selectbox( 'Table', tables, key='agg_table' )
				df = read_table( table )
			with agg_c2:
				numeric_columns = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_columns:
					col = st.selectbox( 'Column', numeric_columns, key='col_box' )
			with agg_c3:
				aggregation = st.selectbox( 'Function', [ 'SUM', 'AVG', 'COUNT' ], key='agg_box' )
			with agg_c4:
				if aggregation == 'SUM':
					st.metric( 'Result', df[ col ].sum( ) )
				elif aggregation == 'AVG':
					st.metric( 'Result', df[ col ].mean( ) )
				elif aggregation == 'COUNT':
					st.metric( 'Result', df[ col ].count( ) )

# ============================================
# SQL CONSOLE MODE
# ============================================
elif mode == 'SQL Console':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'SQL Console' ], divider='blue' )
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
						st.download_button( 'Download CSV', csv, 'query_results.csv',
							'text/csv' )
				
				except Exception as e:
						st.error( f'Execution failed: {e}' )

