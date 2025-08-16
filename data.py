'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                data.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="data.py" company="Terry D. Eppler">

     Mathy Data

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
	data.py
</summary>
******************************************************************************************
'''
from argparse import ArgumentError
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union, Sequence
from pandas.core.common import random_state
from pandas.core.reshape import pivot
from sklearn.model_selection import train_test_split
from sklearn.covariance import empirical_covariance
from sklearn.compose import ColumnTransformer
import sklearn.decomposition as sd
import sklearn.feature_selection as sf
from static import Scaler
from sklearn.metrics import silhouette_score
from sklearn.cross_decomposition import CCA
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from pydantic import BaseModel, Field, validator
from booger import Error, ErrorDialog
from preprocessors import Preprocessor


def entropy( p: float ) -> float | None:
	'''

	    Purpose:
	    _______
        Shannon entropy for a Bernoulli variable with success probability p.

	    Parameters:
	    __________
        p (float): Probability in [0, 1].

	    Returns:
	    ________
        float | None: Entropy in bits, or None on error.

	'''
	try:
		if p is None:
			raise Exception( 'Argument "p" cannot be None' )
		if p < 0 or p > 1:
			raise Exception( 'Argument "p" must be in [0, 1]' )
		eps = 1e-12
		p = np.clip( p, eps, 1 - eps )
		return -p * np.log2( p ) - (1 - p) * np.log2( 1 - p )
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'entropy( p: float ) -> float'
		error = ErrorDialog( exception )
		error.show( )


def gini_impurity( p: float ) -> float | None:
	'''

	    Purpose:
	    _______
	    Gini impurity for a Bernoulli variable with success probability p.

	    Parameters:
	    _________
	    p (float): Probability in [0, 1].

	    Returns:
	    _______
	    float | None: Gini impurity, or None on error.

	'''
	try:
		if p is None:
			raise Exception( 'Argument "p" cannot be None' )
		if p < 0 or p > 1:
			raise Exception( 'Argument "p" must be in [0, 1]' )
		return 1.0 - max( p, 1.0 - p )
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'gini_impurity( p: float ) -> float'
		error = ErrorDialog( exception )
		error.show( )


def misclassification_error( p: float ) -> float | None:
	'''

	    Purpose:
	    ________
        Misclassification error for Bernoulli (1 - max class probability).

	    Parameters:
	    ________
        p (float): Probability in [0, 1].

	    Returns:
	    ________
        float | None: Error rate, or None on error.

	'''
	try:
		if p is None:
			raise Exception( 'Argument "p" cannot be None' )
		else:
			return 1 - np.max( [ p, 1 - p ] )
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'misclassification_error( p: float ) -> float'
		error = ErrorDialog( exception )
		error.show( )


def sigmoid( z: float ) -> float | None:
	'''

		Purpose:
		_________
		While the logit function maps the probability to a real-number range, we can consider the
		inverse of this function to map the real-number range back to a [0, 1] range for the
		probability p. This inverse of the logit function is typically called the logistic sigmoid function,
		which is sometimes simply abbreviated to sigmoid function due to its characteristic S-shape

		Parameters:
		_________
	    z (float): Real-valued input.

		Returns:
		_________
	    float | None: σ(z), or None on error.

	'''
	try:
		if z is None:
			raise Exception( 'Argument "z" cannot be None' )
		z = float( np.clip( z, -709, 709 ) )
		return 1.0 / (1.0 + np.exp( -z ))
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'sigmoid( z: float ) -> float'
		error = ErrorDialog( exception )
		error.show( )


class DataSource( ):
	"""

		Purpose:
		-----------
		Utility class for preparing machine rate datasets from a pandas DataFrame.

		Members:
		------------
		dataframe: pd.DataFrame
		data: np.ndarray
		n_samples: int
		n_features: int
		target: str
		test_size: float
		random_state: int
		feature_names: list
		target_names
		categorical_columns
		numeric_columns: list
		X_training: pd.DataFrame
		y_training
		X_testing
		y_testing

	"""
	dataframe: pd.DataFrame
	target: np.ndarray
	test_size: float
	random_state: int
	data: Optional[ np.ndarray ]
	n_samples: Optional[ int ]
	n_features: Optional[ int ]
	feature_names: Optional[ List[ str ] ]
	target_names: Optional[ np.ndarray ]
	categorical_columns: Optional[ List[ str ] ]
	numeric_columns: Optional[ List[ str ] ]
	X_training: Optional[ np.ndarray ]
	X_testing: Optional[ np.ndarray ]
	y_training: Optional[ np.ndarray ]
	y_testing: Optional[ np.ndarray ]
	transtuple: Optional[ List[ Tuple[ str, Preprocessor, List[ str ] ] ] ]
	numeric_metrics: Optional[ pd.DataFrame ]
	categorical_metrics: Optional[ pd.DataFrame ]
	pivot_table: Optional[ pd.DataFrame ]
	mean_standard_error: Optional[ pd.DataFrame ]
	average: Optional[ pd.Series ]
	kurtosis: Optional[ pd.Series ]
	skew: Optional[ pd.Series ]
	variance: Optional[ pd.Series ]
	standard_deviation: Optional[ pd.Series ]



	def __init__( self, df: pd.DataFrame, target: str, size: float=0.25, rando: int=42 ):
		"""

			Purpose:
			-----------
			Initialize and split the dataset.

	        Parameters:
			-----------
            df (pd.DataFrame): Source dataframe.
            target (str): Name of the target column.
            size (float): Test set proportion.
            rando (int): Random seed for reproducibility.

	        Returns:
			-----------
	            None

		"""
		self.dataframe = df.copy( )
		self.test_size = size
		self.random_state = rando

		if target not in df.columns:
			raise ArgumentError( None, f'target "{target}" not in dataframe' )
		X = df.drop( columns = [ target ] )
		y = df[ target ]
		self.feature_names = list( X.columns )
		self.numeric_columns = X.select_dtypes( include = [ 'number' ] ).columns.tolist( )
		self.categorical_columns = X.select_dtypes(
			include = [ 'object', 'category' ] ).columns.tolist( )
		self.data = X.to_numpy( )
		self.n_samples = len( df )
		self.n_features = X.shape[ 1 ]
		self.target = y.to_numpy( )
		self.target_names = np.array( sorted( y.unique( ) ) )
		self.X_training, self.X_testing, self.y_training, self.y_testing = train_test_split(
			X, y, test_size=self.test_size, random_state=self.random_state, stratify=None)
		num_df = df.select_dtypes( include = 'number' )
		self.skew = num_df.skew( axis=0, numeric_only=True )
		self.variance = num_df.var( axis=0, ddof=1, numeric_only=True )
		self.kurtosis = num_df.kurt( axis=0, numeric_only=True )
		self.average = num_df.mean( axis=0, numeric_only=True )
		self.mean_standard_error = num_df.sem( axis=0, ddof = 1, numeric_only=True )
		self.standard_deviation = num_df.std( axis = 0, ddof = 1, numeric_only=True )
		self.transtuple: list[ tuple[ str, Preprocessor, list[ str ] ] ]=[ ]
		self.numeric_metrics = None
		self.categorical_metrics = None
		self.pivot_table = None

	def __dir__( self ):
		'''

			Purpose:
			-----------
			This function retuns a list of strings (members of the class)

		'''
		return [ 'dataframe', 'n_samples', 'n_features', 'target_names',
		         'feature_names', 'test_size', 'random_state', 'categorical_metrics',
		         'categorical_columns', 'transtuple', 'numeric_metrics',
		         'pivot_table', 'calculate_statistics', 'numeric_columns', 'mean_standard_error',
		         'X_training', 'X_testing', 'y_training', 'average', 'kurtosis', 'variance',
		         'y_testing', 'transform_columns', 'create_pivot_table', 'standard_deviation',
		         'export_excel', 'create_histogram', 'calculate_skew', 'calculate_average',
		         'calculate_deviation', 'calculate_kurtosis', 'calculate_standard_error',
		         'show_correlation_analysis', 'create_correlation_analysis']

	def transform_columns( self, name: str, encoder: Preprocessor, columns: List[ str ] ) -> None:
		"""

		    Purpose:
			-----------
	        Add a (name, transformer, columns) triple and fit/transform X using ColumnTransformer.

		    Parameters:
			-----------
	        name (str): Transformer name.
	        encoder (Preprocessor): Transformer implementing fit/transform.
	        columns (list[str]): Column names to transform.

		    Returns:
			-----------
	        None

		"""
		try:
			if not name:
				raise Exception( 'Argument "name" cannot be None or empty' )
			if encoder is None:
				raise Exception( 'Argument "encoder" cannot be None' )
			if not columns:
				raise Exception( 'Argument "columns" cannot be None or empty' )
			self.transtuple.append( (name, encoder, columns) )
			self.column_transformer = ColumnTransformer(
				transformers=self.transtuple,
				remainder='passthrough' )
			X = self.dataframe[ self.feature_names ]
			_ = self.column_transformer.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DataSource'
			exception.method = 'transform_columns( self, name: str, encoder: object, n_features: List[ str ] )'
			error = ErrorDialog( exception )
			error.show( )

	def calculate_numeric_statistics( self ) -> pd.DataFrame | None:
		"""

			Purpose:
			-----------
			Method calculating descriptive statistics for the datasets numeric n_features.

			Returns:
			-----------
			pd.DataFrame

		"""
		try:
			self.numeric_metrics = self.dataframe.describe(
				percentiles= [ .05, .1, .25, .3, .5, .75, .8, .9, .95 ],
				include=[ np.number ] )
			return self.numeric_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DataSource'
			exception.method = 'calculate_numeric_statistics( self ) -> pd.DataFrame'
			error = ErrorDialog( exception )
			error.show( )

	def calculate_categorical_statistics( self ) -> pd.DataFrame | None:
		"""

			Purpose:
			-----------
			Method calculating descriptive statistics for the datasets categorical n_features.

			Returns:
			-----------
			pd.DataFrame

		"""
		try:
			self.categorical_metrics = self.dataframe.describe( include=[ object ] )
			return self.categorical_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DataSource'
			exception.method = 'calculate_categorical_statistics( self ) -> pd.DataFrame '
			error = ErrorDialog( exception )
			error.show( )

	def create_pivot_table( self, df: pd.DataFrame, cols: list, vals: list, idx: list ) -> pd.DataFrame | None:
		'''

		    Purpose:
		    _______
	        Create a spreadsheet-style pivot table as a DataFrame.

		    Parameters:
		    __________
	        df (pd.DataFrame): Source dataframe.
	        cols (list): Columns to use for columns axis of pivot.
	        vals (list): Value columns to aggregate.
	        idx (list): Columns to use as row index of pivot.

		    Returns:
			________
		    pd.DataFrame | None: Pivot table or None on error.

		'''
		try:
			if df is None:
				raise Exception( 'Argument "df" cannot be None' )
			if not cols:
				raise Exception( 'Argument "cols" cannot be None or empty' )
			if not vals:
				raise Exception( 'Argument "vals" cannot be None or empty' )
			if not idx:
				raise Exception( 'Argument "idx" cannot be None or empty' )
			_df = df.copy( )
			self.pivot_table = pd.pivot_table( data=_df, index=idx, columns=cols,
				values=vals, aggfun='sum', dropna=True, margins=True )
			return self.pivot_table
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DataSource'
			exception.method = 'create_pivot_table( self ) -> pd.DataFrame '
			error = ErrorDialog( exception )
			error.show( )

	def export_excel( self, filepath: str=None ) -> None:
		'''

			Purpose:
			--------
			Exports dataframe to an Excel file.


			:param filepath:
			:type filepath:
			:return:
			:rtype:
		'''
		try:
			if filepath is None:
				raise Exception( 'Argument "filepath" cannot be None' )
			else:
				self.dataframe.to_excel( filepath )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DataSource'
			exception.method = 'export_excel( self, filepath: str=None ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def show_histogram( self ):
		'''

			Purpose:
			________

			Method to create histogram of numeric n_features.

		'''
		try:
			col_means = self.dataframe.select_dtypes( 'number' ).mean( axis = 0 )
			plt.figure( figsize = (10, 6) )
			sns.histplot( col_means, bins = 20, kde = True )
			plt.title( "Histogram of Column Means" )
			plt.xlabel( "Mean Value" )
			plt.ylabel( "Frequency" )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'data'
			exception.method = 'show_histogram( self )'
			error = ErrorDialog( exception )
			error.show( )

	def create_histogram( self, df: pd.DataFrame, axes: int=0, numbers_only=True ):
		'''

			Purpose:
			________

			Method to create histogram of from a dataframe.

		'''
		try:
			if df is None:
				raise Exception( 'Argument "df" cannot be None' )
			_df = df.select_dtypes( 'number' ) if numbers_only else df
			series = _df.mean( axis = axes )
			plt.figure( figsize = (8, 6) )
			sns.histplot( series, bins = 20, kde = True )
			plt.title( "Histogram of Means" )
			plt.xlabel( "Mean Value" )
			plt.ylabel( "Frequency" )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'data'
			exception.method = 'create_histogram( self, df: pd.DataFrame '
			error = ErrorDialog( exception )
			error.show( )

	def show_correlation_analysis( self, strategy='pearson', numbers_only: bool=True ):
		'''

			Purpose:
			--------
			Method to show the pearson-correlation analysis of the dataset.
		'''
		try:
			if strategy is None:
				raise Exception( 'Argument "strategy" cannot be None' )
			else:
				_correlation = self.dataframe.corr( method = strategy, numeric_only = numbers_only )
				plt.figure( figsize = (10, 6) )
				sns.heatmap( _correlation, cmap = "coolwarm", annot = True )
				plt.title( "Pearson Correlation" )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'data'
			exception.method = 'show_correlation_analysis( self )'
			error = ErrorDialog( exception )
			error.show( )

	def create_correlation_analysis( self, df: pd.DataFrame, strategy='pearson',
	                                 numbers_only: bool=True ):
		'''

			Purpose:
			--------
			Method to show the pearson-correlation analysis of the dataset.

		'''
		try:
			if df is None:
				raise Exception( 'Argument "df" cannot be None' )
			elif strategy is None:
				raise Exception( 'Argument "strategy" cannot be None' )
			else:
				_dataframe = df.copy( )
				_correlation = _dataframe.corr( method = strategy, numeric_only = numbers_only )
				plt.figure( figsize = (10, 6) )
				sns.heatmap( _correlation, cmap = 'coolwarm', annot = True )
				plt.title( 'Pearson Correlation' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'data'
			exception.method = 'create_correlation_analysis( self, df: pd.DataFrame )'
			error = ErrorDialog( exception )
			error.show( )

	def calculate_average( self, df: pd.DataFrame, axes: int=0, numeric: bool=True ) -> pd.Series | None:
		'''

		    Purpose:
		    ________
	        Compute the mean along the specified axis.

		    Parameters:
		    __________
	        df (pd.DataFrame): Source dataframe.
	        axes (int): Axis over which to compute mean (0=columns, 1=rows).
	        numeric (bool): If True, restrict to numeric dtypes.

		    Returns:
			________
	        pd.Series | None: Means by axis, or None on error.

		'''
		try:
			if axes is None:
				raise Exception( 'Argument "axes" cannot be None' )
			elif df is None:
				raise Exception( 'Argument "df" cannot be None' )
			elif numeric is None:
				raise Exception( 'Argument "numeric" cannot be None' )
			else:
				_dataframe = df.copy( )
				_deviation = _dataframe.mean( axis=axes, numeric_only=numeric )
				return _deviation
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DataSource'
			exception.method = ('calculate_average( self, df: pd.DataFrame, axes: int=0, '
			                    'numeric: bool=True ) -> pd.Series ')
			error = ErrorDialog( exception )
			error.show( )

	def calculate_variance( self, df: pd.DataFrame, axes: int=0, degree: int=1,
	                        numeric: bool=True ) -> pd.Series | None:
		'''

		    Purpose:
		    _______
	        Compute the variance along the specified axis.

		    Parameters:
		    _________
	        df (pd.DataFrame): Source dataframe.
	        axes (int): Axis over which to compute variance.
	        degree (int): Delta degrees of freedom (ddof).
	        numeric (bool): If True, restrict to numeric dtypes.

		    Returns:
		    _______
	        pd.Series | None: Variances by axis, or None on error.

		'''
		try:
			if df is None:
				raise Exception( 'Argument "df" cannot be None' )
			elif axes is None:
				raise Exception( 'Argument "axes" cannot be None' )
			elif degree is None:
				raise Exception( 'Argument "degree" cannot be None' )
			elif numeric is None:
				raise Exception( 'Argument "numeric" cannot be None' )
			else:
				_dataframe = df.copy( )
				_variance = _dataframe.var( axis=axes, ddof=degree,
					numeric_only=numeric )
				return _variance
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DataSource'
			exception.method = 'create_kurtosis( self ) -> pd.DataFrame '
			error = ErrorDialog( exception )
			error.show( )

	def calculate_skew( self, df: pd.DataFrame, axes: int=0, numeric: bool=True ) -> pd.Series | None:
		'''

			Purpose:
			--------
			Return unbiased skew over requested axis.


			:param dimension:
			:type dimension:
			:param degree:
			:type degree:
			:return: pd.Series
			:rtype: pd.Series | None
		'''
		try:
			if df is None:
				raise Exception( 'Argument "df" cannot be None' )
			elif axes is None:
				raise Exception( 'Argument "axis" cannot be None' )
			elif numeric is None:
				raise Exception( 'Argument "numeric" cannot be None' )
			else:
				_dataframe = df.copy( )
				_skew = _dataframe.skew( axis=axes, numeric_only=numeric )
				return _skew
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DataSource'
			exception.method = 'create_kurtosis( self ) -> pd.DataFrame '
			error = ErrorDialog( exception )
			error.show( )

	def calculate_kurtosis( self, df: pd.DataFrame, axes: int=0, numeric: bool=True ) -> pd.Series | None:
		'''

			Purpose:
			--------
			Return unbiased skutosis over requested axis.


			:param axes:
			:type axes: int
			:return: pd.Series
			:rtype: pd.Series | None
		'''
		try:
			if axes is None:
				raise Exception( 'Argument "axis" cannot be None' )
			elif df is None:
				raise Exception( 'Argument "df" cannot be None' )
			elif numeric is None:
				raise Exception( 'Argument "numeric" cannot be None' )
			else:
				_dataframe = df.copy( )
				_kurtosis = _dataframe.kurt( axis=axes, numeric_only=numeric )
				return _kurtosis
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DataSource'
			exception.method = 'create_kurtosis( self ) -> pd.DataFrame '
			error = ErrorDialog( exception )
			error.show( )

	def calculate_standard_error( self, df: pd.DataFrame, axes: int=0, degree: int=1,
	                              numeric: bool=True ) -> pd.Series | None:
		'''

			Purpose:
			--------
			Return unbiased standard error of the mean over requested axis. Normalized by N-1 by default.
			This can be changed using the degree argument.


			:param axes:
			:type axes: int
			:param degree:
			:type degree: int
			:return: pd.Series
			:rtype: pd.Series | None
		'''
		try:
			if axes is None:
				raise Exception( 'Argument "axis" cannot be None' )
			elif df is None:
				raise Exception( 'Argument "df" cannot be None' )
			elif degree is None:
				raise Exception( 'Argument "degree" cannot be None' )
			elif numeric is None:
				raise Exception( 'Argument "numeric" cannot be None' )
			else:
				_dataframe = df.copy( )
				_error = _dataframe.sem( axis=axes, ddof=degree,
					numeric_only=numeric )
				return _error
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DataSource'
			exception.method = 'calculate_standard_error( self, axes: int=0, degree: int=1 ) -> pd.Series'
			error = ErrorDialog( exception )
			error.show( )

	def calculate_deviation( self, df: pd.DataFrame, axes: int=0, degree: int=1,
	                         numeric: bool=True ) -> pd.Series | None:
		'''

			Purpose:
			--------
			Return unbiased standard deviation over requested axis. Normalized by N-1 by default.
			This can be changed using the degree argument.


			:param axes:
			:type axes: int
			:param degree:
			:type degree: int
			:return: pd.Series
			:rtype: pd.Series | None
		'''
		try:
			if axes is None:
				raise Exception( 'Argument "axis" cannot be None' )
			elif df is None:
				raise Exception( 'Argument "df" cannot be None' )
			elif degree is None:
				raise Exception( 'Argument "degree" cannot be None' )
			elif numeric is None:
				raise Exception( 'Argument "numeric" cannot be None' )
			else:
				_dataframe = df.copy( )
				_deviation = _dataframe.std( axis=axes,
					ddof=degree, numeric_only=numeric )
				return _deviation
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DataSource'
			exception.method = 'calculate_standard_deviation( self, axes: int=0, degree: int=1 ) -> pd.Series'
			error = ErrorDialog( exception )
			error.show( )

class VarianceThreshold( ):
	"""

		Purpose:
		---------
		VarianceThreshold is a simple baseline approach to feature selection. It removes all
		feature_names whose variance doesn’t meet some threshold. By default, it removes all
		zero-variance feature_names, i.e. feature_names that have the same value in all samples.

	"""
	variance_selector: sf.VarianceThreshold
	transformed_data: Optional[ np.ndarray ]
	threshold: Optional[ float ]


	def __init__( self, thresh: float=0.0 ) -> None:
		"""

			Purpose:
			---------
			Initialize VarianceThreshold.

			:param threshold: Features with variance below this are removed.
			:type threshold: float
		"""
		self.threshold = thresh
		self.variance_selector = sf.VarianceThreshold( threshold=self.threshold )
		self.transformed_data = None


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""

			Purpose:
			---------
			Fit the variance threshold model.

			:param X: Input feature matrix.
			:type X: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.variance_selector.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = 'fit( self, X: np.ndarray ) -> object | None'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Apply variance threshold selection.

			:param X: Feature matrix.
			:type X: np.ndarray
			:return: Reduced feature matrix.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.transformed_data = self.variance_selector.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Fit and transform the data using variance thresholding.

			:param X: Feature matrix.
			:type X: np.ndarray
			:return: Reduced feature matrix.
			:rtype: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.transformed_data = self.variance_selector.fit_transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class CorrelationAnalysis( ):
	"""

		Canonical Correlation Analysis (CCA) extracts the ‘directions of covariance’,
		i.e. the components of each datasets that explain the most shared variance
		between both datasets.

	"""
	correlation_analysis: CCA
	n_components: Optional[ int ]
	scale: bool
	max_iter: Optional[ int ]
	transformed_data: ( np.ndarray, np.ndarray )


	def __init__( self, num: int=2, scale: bool=True, max: int=500 ) -> None:
		"""

			Purpose:
			---------
			Initialize CCA.

			:param n: Number of components.
			:type n: int
		"""
		self.scale = scale
		self.n_components = num
		self.max_iter = max
		self.correlation_analysis = CCA( n_components=self.n_components,
			scale=self.scale, max_iter=self.max_iter )
		self.transformed_data = None


	def fit( self, X: np.ndarray, Y: np.ndarray ) -> CCA | None:
		"""

			Purpose:
			---------
			Fit the CCA model to X and Y.

			:param X: Feature matrix X.
			:type X: np.ndarray
			:param Y: Feature matrix Y.
			:type Y: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.correlation_analysis.fit( X, Y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'CorrelationAnalysis'
			exception.method = 'fit( self, X: np.ndarray, Y: np.ndarray ) -> object'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray, Y: np.ndarray ) -> ( np.ndarray, np.ndarray ):
		"""

			Purpose:
			---------
			Apply the CCA transformation.

			:param X: Feature matrix X.
			:type X: np.ndarray
			:param Y: Feature matrix Y.
			:type Y: np.ndarray
			:return: Transformed tuple (X_c, Y_c).
			:rtype: tuple[np.ndarray, np.ndarray]

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			elif Y is None:
				raise Exception( 'Argument "Y" is None' )
			else:
				self.transformed_data = self.correlation_analysis.transform( X, Y )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'CorrelationAnalysis'
			exception.method = 'transform( self, X: np.ndarray, Y: np.ndarray ) -> tuple'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: np.ndarray ) ->  ( np.ndarray, np.ndarray ):
		"""

			Purpose:
			---------
			Fit and transform with CCA.

			:param X: Feature matrix X.
			:type X: np.ndarray
			:param Y: Feature matrix Y.
			:type Y: np.ndarray
			:return: Transformed tuple (X_c, Y_c).
			:rtype: tuple[np.ndarray, np.ndarray]

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			elif y is None:
				raise Exception( 'Argument "Y" is None' )
			else:
				self.transformed_data = self.correlation_analysis.fit( X, y ).transform( X, y )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'CorrelationAnalysis'
			exception.method = 'fit_transform( self, X: np.ndarray, Y: np.ndarray ) -> tuple'
			error = ErrorDialog( exception )
			error.show( )


class ComponentAnalysis( ):
	"""

		Purpose:
		---------
		Principal Component Analysis (PCA). Linear dimensionality reduction using
		Singular Value Decomposition of the data to project it to a lower dimensional space.
		The input data is centered but not scaled for each feature before applying the SVD.
		It uses the LAPACK implementation of the full SVD or a randomized truncated SVD
		by the method of Halko et al. 2009, depending on the shape of the input data and
		the number of components to extract.

	"""
	component_analysis: sd.PCA
	svd_solver: Optional[ str ]
	n_components: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]


	def __init__( self, num: int=2, solver: str='auto' ) -> None:
		"""

			Purpose:
			---------
			Initialize PCA.

			:param n_components: Number of components.
			:type n_components: int

		"""
		self.n_components = num
		self.svd_solver = solver
		self.component_analysis = sd.PCA( n_components=num, svd_solver=self.svd_solver )
		self.transformed_data = None


	def fit( self, X: np.ndarray ) -> PCA | None:
		"""

			Purpose:
			---------
			Fit PCA to the input data.

			:param X: Feature matrix.
			:type X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.component_analysis.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ComponentAnalysis'
			exception.method = 'def fit( self, X: np.ndarray ) -> ComponentAnalysis'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Apply PCA transformation.

			:param X: Feature matrix.
			:type X: np.ndarray
			:return: Transformed matrix.
			:rtype: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.transformed_data = self.component_analysis.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ComponentAnalysis'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Fit PCA and transform input data.

			:param X: Feature matrix.
			:type X: np.ndarray
			:return: Transformed matrix.
			:rtype: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.transformed_data = self.component_analysis.fit_transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ComponentAnalysis'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
