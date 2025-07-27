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
from typing import Optional, List, Dict, Tuple, Union, Sequence
from pandas.core.common import random_state
from pandas.core.reshape import pivot
from sklearn.model_selection import train_test_split
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

def entropy( p: float ) -> float | None:
	'''

		Purpose:
		--------
		Method used to calculate the entropy of a numeric feature
		with a probability or proportion of 'p'.

		:param p:
		:type p:
		:return:
		:rtype:
	'''
	if p is None:
		raise Exception( 'Argument "p" cannot be None' )
	else:
		return - p * np.log2( p ) - (1 - p) * np.log2( (1 - p) )


def gini_impurity( p: float ) -> float | None:
	'''

		Purpose:
		--------
		Method used to calculate the entropy of a numeric feature
		with a probability or proportion of 'p'.

		:param p:
		:type p:
		:return:
		:rtype:
	'''
	if p is None:
		raise Exception( 'Argument "p" cannot be None' )
	else:
		return p * (1 - p) + (1 - p) * (1 - (1 - p))

def misclassification_error( p: float ) -> float | None:
	'''

		Purpose:
		--------
		Method used to calculate the entropy of a numeric feature
		with a probability or proportion of 'p'.

		:param p:
		:type p:
		:return:
		:rtype:
	'''
	if p is None:
		raise Exception( 'Argument "p" cannot be None' )
	else:
		return 1 - np.max( [ p, 1 - p ] )


class Dataset( ):
	"""

		Purpose:
		-----------
		Utility class for preparing machine rate datasets from a pandas DataFrame.

		Members:
		------------
		dataframe: pd.DataFrame
		data: np.ndarray
		rows: int
		columns: int
		target: str
		test_size: float
		random_state: int
		features: list
		target_values
		numeric_columns
		text_columns: list
		training_data: pd.DataFrame
		training_values
		testing_data
		testing_values

	"""
	dataframe: pd.DataFrame
	target: str
	test_size: float
	random_state: int
	data: Optional[ np.ndarray ]
	rows: Optional[ int ]
	columns: Optional[ int ]
	features: Optional[ List[ str ] ]
	target_values: Optional[ np.ndarray ]
	numeric_columns: Optional[ List[ str ] ]
	text_columns: Optional[ List[ str ] ]
	training_data: Optional[ np.ndarray ]
	testing_data: Optional[ np.ndarray ]
	training_values: Optional[ np.ndarray ]
	testing_data: Optional[ np.ndarray ]
	testing_values: Optional[ np.ndarray ]
	transtuple: Optional[ List[ Tuple ] ]
	numeric_statistics: Optional[ pd.DataFrame ]
	categorical_statistics: Optional[ pd.DataFrame ]
	pivot_data: Optional[ pd.DataFrame ]
	kurtosis: Optional[ pd.Series ]
	skew: Optional[ pd.Series ]
	variance: Optional[ pd.Series ]
	standard_error: Optional[ pd.Series ]
	standeard_deviation: Optional[ pd.Series ]



	def __init__( self, df: pd.DataFrame, target: str, size: float = 0.25, rando: int = 42 ):
		"""

			Purpose:
			-----------
			Initialize and split the dataset.

			Parameters:
			-----------
			df (pd.DataFrame): Matrix text vector.
			target List[ str ]: Name of the target columns.
			size (float): Proportion of df to use as test set.
			rando (int): Seed for reproducibility.

		"""
		self.dataframe = df
		self.data = df.to_numpy( )
		self.rows = len( df )
		self.columns = len( df.columns )
		self.target = target
		self.test_size = size
		self.random_state = rando
		self.features = [ column for column in df.columns ]
		self.target_values = df[ target ].to_numpy( )
		self.numeric_columns = df.select_dtypes( include = [ 'number' ] ).columns.tolist( )
		self.text_columns = df.select_dtypes( include = [ 'object', 'category' ] ).columns.tolist( )
		self.training_data = train_test_split( self.data, self.target_values,
			test_size = self.test_size, random_state = self.random_state )[ 0 ]
		self.testing_data = train_test_split( self.data, self.target_values,
			test_size = self.test_size, random_state = self.random_state )[ 1 ]
		self.training_values = train_test_split( self.data, self.target_values,
			test_size = self.test_size, random_state = self.random_state )[ 2 ]
		self.testing_values = train_test_split( self.data, self.target_values,
			test_size = self.test_size, random_state = self.random_state )[ 3 ]
		self.transtuple = [ ]
		self.numeric_statistics = None
		self.categorical_statistics = None
		self.skew = None
		self.variance = None
		self.kurtosis = None
		self.standard_error = None
		self.standard_deviation = None


	def __dir__( self ):
		'''

			Purpose:
			-----------
			This function retuns a list of strings (members of the class)

		'''
		return [ 'dataframe', 'rows', 'columns', 'target_values',
		         'features', 'test_size', 'random_state', 'categorical_statistics',
		         'numeric_columns', 'text_columns', 'transtuple',
		         'calculate_statistics', 'numeric_statistics',
		         'target_values', 'training_data', 'testing_data', 'training_values',
		         'testing_values', 'transform_columns',
		         'create_pivot_table', 'calculate_skew', 'calculate_variance',
		         'calculate_standard_error', 'calculate_standeard_deviation', 'calculate_kurtosis']


	def transform_columns( self, name: str, encoder: object, columns: List[ str ] ) -> None:
		"""

			Purpose:
			-----------
				Scale numeric features using selected scaler.

			Paramters:
			-----------
				name - the name of the encoder
				encoder - the encoder object to transform the df.
				columns - the list of column names to apply the transformation to.

		"""
		try:
			if name is None:
				raise Exception( 'Arguent "name" cannot be None' )
			elif encoder is None:
				raise Exception( 'Arguent "encoder" cannot be None' )
			elif columns is None:
				raise Exception( 'Arguent "columns" cannot be None' )
			else:
				_tuple = (name, encoder, columns)
				self.transtuple.append( _tuple )
				self.column_transformer = ColumnTransformer( self.transtuple )
				self.column_transformer.fit_transform( self.data )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
			exception.method = 'transform_columns( self, name: str, encoder: object, columns: List[ str ] )'
			error = ErrorDialog( exception )
			error.show( )

	def calculate_numeric_statistics( self ) -> pd.DataFrame | None:
		"""

			Purpose:
			-----------
			Method calculating descriptive statistics for the datasets numeric columns.

			Returns:
			-----------
			pd.DataFrame

		"""
		try:
			self.numeric_statistics = self.dataframe.describe(
				percentiles = [ .1, .25, .5, .75, .9 ],
				include = [ np.number ] )
			return self.numeric_statistics
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
			Method calculating descriptive statistics for the datasets categorical columns.

			Returns:
			-----------
			pd.DataFrame

		"""
		try:
			self.categorical_statistics = self.dataframe.describe( include = [ object ] )
			return self.categorical_statistics
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
			will be stored in MultiIndex objects (hierarchical indexes) on the index and columns
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


	def calculate_variance( self, axes: int=0, degree: int=1 ) -> pd.Series | None:
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
			if axes is None:
				raise Exception( 'Argument "axis" cannot be None' )
			else:
				self.variance = self.dataframe.var( axis=axes, ddof=degree )
				return self.variance
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
			exception.method = 'create_kurtosis( self ) -> pd.DataFrame '
			error = ErrorDialog( exception )
			error.show( )


	def calculate_skew( self, axes: int=0 ) -> pd.Series | None:
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
			if axes is None:
				raise Exception( 'Argument "axis" cannot be None' )
			else:
				self.skew = self.dataframe.skew( axis=axes )
				return self.skew
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
			exception.method = 'create_kurtosis( self ) -> pd.DataFrame '
			error = ErrorDialog( exception )
			error.show( )


	def calculate_kurtosis( self, axes: int=0 ) -> pd.Series | None:
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
			else:
				self.kurtosis = self.dataframe.kurt( axis=axes )
				return self.kurtosis
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
			exception.method = 'create_kurtosis( self ) -> pd.DataFrame '
			error = ErrorDialog( exception )
			error.show( )


	def calculate_standard_error( self, axes: int=0, degree: int=1 ) -> pd.Series | None:
		'''

		Purpose:
		--------
		Return unbiased standard error of the mean over requested axis. Normalized by N-1 by default.
		This can be changed using the degree argument.


		:param dimension:
		:type dimension:
		:param degree:
		:type degree:
		:return:
		:rtype:
		'''
		try:
			if axes is None:
				raise Exception( 'Argument "axis" cannot be None' )
			elif degree is None:
				raise Exception( 'Argument "degree" cannot be None' )
			else:
				self.standard_error = self.dataframe.sem( axis=axes, ddof=degree )
				return self.standard_error
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
			exception.method = 'calculate_standard_error( self, axes: int=0, degree: int=1 ) -> pd.Series'
			error = ErrorDialog( exception )
			error.show( )


	def calculate_standard_deviation( self, axes: int=0, degree: int=1 ) -> pd.Series | None:
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
			elif degree is None:
				raise Exception( 'Argument "degree" cannot be None' )
			else:
				self.standard_deviation = self.dataframe.std( axis=axes, ddof=degree )
				return self.standard_deviation
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
		Wrapper for VarianceThreshold feature selector.
	"""
	variance_selector: VarianceThreshold
	transformed_data: Optional[ np.ndarray ]


	def __init__( self, thresh: float=0.0 ) -> None:
		"""

			Purpose:
			---------
			Initialize VarianceThreshold.

			:param threshold: Features with variance below this are removed.
			:type threshold: float
		"""
		super( ).__init__( )
		self.variance_selector = VarianceThreshold( threshold=thresh )
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

		Wrapper for Canonical Correlation Analysis (CCA).

	"""
	correlation_analysis: CCA
	transformed_data: ( np.ndarray, np.ndarray )


	def __init__( self, num: int=2 ) -> None:
		"""

			Purpose:
			---------
			Initialize CCA.

			:param n: Number of components.
			:type n: int
		"""
		super( ).__init__( )
		self.correlation_analysis = CCA( n_components=num )
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
		Wrapper for Principal Component Analysis (PCA).

	"""
	component_analysis: PCA
	transformed_data: Optional[ np.ndarray ]


	def __init__( self, num: int=2 ) -> None:
		"""

			Purpose:
			---------
			Initialize PCA.

			:param n_components: Number of components.
			:type n_components: int

		"""
		super( ).__init__( )
		self.component_analysis = PCA( n_components=num )
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
