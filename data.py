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
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from static import Scaler
from sklearn.metrics import silhouette_score
from sklearn.cross_decomposition import CCA
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from pydantic import BaseModel, Field, validator
from booger import Error, ErrorDialog
from preprocessors import Processor


def entropy( p: float ) -> float | None:
	'''

		Purpose:
		--------
		Method used to calculate the entropy of a numeric feature
		with a proportion of 'p'.

		:param p:
		:type p: float
		:return: entropy
		:rtype: float | None
	'''
	try:
		if p is None:
			raise Exception( 'Argument "p" cannot be None' )
		else:
			return - p * np.log2( p ) - ( 1 - p ) * np.log2( ( 1 - p ) )
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
			--------
			Method used to calculate the entropy of a numeric feature
			with a proportion of 'p'.

			:param p:
			:type p: float
			:return: impurity
			:rtype: float | None
	'''
	try:
		if p is None:
			raise Exception( 'Argument "p" cannot be None' )
		else:
			return p * ( 1 - p ) + ( 1 - p ) * ( 1 - ( 1 - p ) )
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
		--------
		Method used to calculate the entropy of a numeric feature
		with a proportion of 'p'.

		:param p:
		:type p: float
		:return: error rate
		:rtype: float | None
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

		:param z:
		:type z: float
		:return:
		:rtype: float | None
	'''
	try:
		if z is None:
			raise Exception( 'Argument "z" cannot be None' )
		else:
			return 1.0 / ( 1.0 + np.exp( -z ) )
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'sigmoid( z: float ) -> float'
		error = ErrorDialog( exception )
		error.show( )


class Dataset( ):
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
	transtuple: Optional[ List[ Tuple[ str, Processor, List[ str ] ] ] ]
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
			df (pd.DataFrame): Matrix text vector.
			target List[ str ]: Name of the target n_features.
			size (float): Proportion of df to use as test set.
			rando (int): Seed for reproducibility.

		"""
		self.dataframe = df
		self.data = df.to_numpy( )
		self.n_samples = len( df )
		self.n_features = len( df.columns )
		self.target = df[ target ].to_numpy( )
		self.test_size = size
		self.random_state = rando
		self.feature_names = [ column for column in df.columns ]
		self.target_names = np.array( df[ target ].unique( ) )
		self.categorical_columns = df.select_dtypes( include=[ 'object', 'category' ] ).columns.tolist( )
		self.numeric_columns = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
		self.X_training = train_test_split( self.data, self.target,
			test_size=self.test_size  )[ 0 ]
		self.X_testing = train_test_split( self.data, self.target,
			test_size=self.test_size  )[ 1 ]
		self.y_training = train_test_split( self.data, self.target,
			test_size=self.test_size  )[ 2 ]
		self.y_testing = train_test_split( self.data, self.target,
			test_size=self.test_size, random_state=self.random_state )[ 3 ]
		self.skew = df.skew( axis=0, numeric_only=True )
		self.variance = df.var( axis=0, ddof=1, numeric_only=True )
		self.kurtosis = df.kurt( axis=0, numeric_only=True )
		self.average = df.mean( axis=0, numeric_only=True )
		self.mean_standard_error = df.sem( axis=0, ddof=1, numeric_only=True )
		self.standard_deviation = df.std( axis=0, ddof=1, numeric_only=True  )
		self.transtuple = [ ]
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

	def transform_columns( self, name: str, encoder: Processor, columns: List[ str ] ) -> None:
		"""

			Purpose:
			-----------
				Scale numeric feature_names using selected scaler.

			Paramters:
			-----------
				name - the name of the encoder
				encoder - the encoder object to transform the df.
				n_features - the list of column names to apply the transformation to.

		"""
		try:
			if name is None:
				raise Exception( 'Arguent "name" cannot be None' )
			elif encoder is None:
				raise Exception( 'Arguent "encoder" cannot be None' )
			elif columns is None:
				raise Exception( 'Arguent "n_features" cannot be None' )
			else:
				_tuple = ( name, encoder, columns )
				self.transtuple.append( _tuple )
				self.column_transformer = ColumnTransformer( self.transtuple )
				self.column_transformer.fit_transform( self.data )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
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
				percentiles=[ .1, .25, .5, .75, .9 ],
				include=[ np.number ] )
			return self.numeric_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
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
			exception.cause = 'Dataset'
			exception.method = 'calculate_categorical_statistics( self ) -> pd.DataFrame '
			error = ErrorDialog( exception )
			error.show( )

	def create_pivot_table( self, df: pd.DataFrame, cols: List[ str ] ) -> pd.DataFrame | None:
		'''

			Purpose:
			________
			Create a spreadsheet-style pivot table as a DataFrame. The levels in the pivot table
			will be stored in MultiIndex objects (hierarchical indexes) on the index and n_features
			of the result DataFrame.

			:return: pivot table
			:rtype: pd.DataFrame
		'''
		try:
			if df is None:
				raise Exception( 'Argument "df" cannot be None' )
			elif cols is None:
				raise Exception( 'Argument "cols" cannot be None' )
			else:
				self.pivot_table = pandas.pivot( data=df, columns=cols )
				return self.pivot_table
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
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
			exception.cause = 'Dataset'
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
			plt.figure( figsize=( 10, 6 ) )
			sns.histplot( self.dataframe.mean( axis=0, numeric_only=True), bins=20, kde=True )
			plt.title( "Histogram (Mean)" )
			plt.xlabel( "Name" )
			plt.ylabel( "Value" )
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
			else:
				_dataframe = df.copy( )
				plt.figure( figsize = (8, 6) )
				sns.histplot( _dataframe.mean( axis=axes, numeric_only=numbers_only ), bins = 20,
					kde = True )
				plt.title( "Histogram (Mean)" )
				plt.xlabel( "Name" )
				plt.ylabel( "Value" )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'data'
			exception.method = 'create_histogram( self, df: pd.DataFrame '
			error = ErrorDialog( exception )
			error.show( )

	def show_correlation_analysis( self, strategy = 'pearson', numbers_only: bool = True ):
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

	def create_correlation_analysis( self, df: pd.DataFrame, strategy = 'pearson',
	                                 numbers_only: bool = True ):
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
			exception.cause = 'Dataset'
			exception.method = 'calculate_average( self, df: pd.DataFrame, axes: int=0, numeric: bool=True ) -> pd.Series '
			error = ErrorDialog( exception )
			error.show( )

	def calculate_variance( self, df: pd.DataFrame, axes: int=0, degree: int=1,
	                        numeric: bool=True ) -> pd.Series | None:
		'''

			Purpose:
			--------


			:param dimension:
			:type dimension:
			:param degree:
			:type degree:
			:return:
			:rtype:
		'''
		try:
			if df is None:
				raise Exception( 'Argument "df" cannot be None' )
			elif axes is None:
				raise Exception( 'Argument "axis" cannot be None' )
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
			exception.cause = 'Dataset'
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
			exception.cause = 'Dataset'
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
			exception.cause = 'Dataset'
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
			exception.cause = 'Dataset'
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
			exception.cause = 'Dataset'
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
	variance_selector: VarianceThreshold
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
		self.variance_selector = VarianceThreshold( threshold=self.threshold )
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
	component_analysis: PCA
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
		super( ).__init__( )
		self.n_components = num
		self.svd_solver = solver
		self.component_analysis = PCA( n_components=num, svd_solver=self.svd_solver )
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
