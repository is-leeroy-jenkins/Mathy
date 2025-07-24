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
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union, Sequence
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from static import Scaler
from sklearn.metrics import silhouette_score
from sklearn.cross_decomposition import CCA
from sklearn.base import BaseEstimator
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Tuple


class Metric( BaseModel ):
	"""

		Purpose:
		---------
		Base interface for all preprocessors. Provides standard `fit`, `transform`, and
	    `fit_transform` methods.

	"""

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )
		self.pipeline = None
		self.transformed_data = [ ]
		self.transformed_values = [ ]


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
		"""

			Purpose:
			---------
			Fits the preprocessor to the text df.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (Optional[np.ndarray]): Optional target array.

		"""
		raise NotImplementedError

	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms the text df using the fitted preprocessor.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Fits the preprocessor and then transforms the text df.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (Optional[np.ndarray]): Optional target array.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			self.fit( X, y )
			return self.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Metric'
			exception.method = ('fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray '
			                    ']=None'
			                    ') -> np.ndarray')
			error = ErrorDialog( exception )
			error.show( )


class VarianceThreshold( Metric ):
	"""

		Purpose:
		---------
		Wrapper for VarianceThreshold feature selector.
	"""

	def __init__( self, thresh: float=0.0 ) -> None:
		"""

			Purpose:
			---------
			Initialize VarianceThreshold.

			:param threshold: Features with variance below this are removed.
			:type threshold: float
		"""
		super( ).__init__( )
		self.selector = VarianceThreshold( threshold=thresh )


	def fit( self, X: np.ndarray ) -> object | None:
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
				return self.selector.fit( X )
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
				return self.selector.transform( X )
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
				return self.selector.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class CorrelationAnalysis( Metric ):
	"""

		Wrapper for Canonical Correlation Analysis (CCA).

	"""

	def __init__( self, n: int=2 ) -> None:
		"""

			Purpose:
			---------
			Initialize CCA.

			:param n: Number of components.
			:type n: int
		"""
		super( ).__init__( )
		self.correlation_analysis = CCA( n_components=n )


	def fit( self, X: np.ndarray, Y: np.ndarray ) -> object | None:
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
				return self.correlation_analysis.fit( X, Y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = 'fit( self, X: np.ndarray, Y: np.ndarray ) -> object'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray, Y: np.ndarray ) -> tuple[ np.ndarray, np.ndarray ] | None:
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
				return self.correlation_analysis.transform( X, Y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, Y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ] | None:
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
			elif Y is None:
				raise Exception( 'Argument "Y" is None' )
			else:
				return self.correlation_analysis.fit( X, Y ).transform( X, Y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class ComponentAnalysis( Metric ):
	"""

		Purpose:
		---------
		Wrapper for Principal Component Analysis (PCA).

	"""

	def __init__( self, num: int ) -> None:
		"""

			Purpose:
			---------
			Initialize PCA.

			:param n_components: Number of components.
			:type n_components: int

		"""
		super( ).__init__( )
		self.component_analysis = PCA( n_components=num )


	def fit( self, X: np.ndarray ) -> object | None:
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
				return self.component_analysis.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = ''
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
				return self.component_analysis.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
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
				return self.component_analysis.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class Dataset( Metric ):
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
	data: np.ndarray
	rows: int
	columns: int
	target: str
	test_size: float
	random_state: int
	features: List[ str ]
	target_values: List[ object ]
	numeric_columns: List[ str ]
	text_columns: List[ str ]
	training_data: pd.DataFrame
	training_values: np.ndarray
	testing_data: pd.DataFrame
	testing_values: np.ndarray
	transtuple: List[ Tuple ]

	def __init__( self, df: pd.DataFrame, target: str, size: float=0.2, rando: int=42 ):
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
		super( ).__init__( )
		self.dataframe = df
		self.data = df[ 1:, : ]
		self.rows = len( df )
		self.columns = len( df.columns )
		self.target = target
		self.test_size = size
		self.random_state = rando
		self.features = [ column for column in df.columns ]
		self.target_values = [ value for value in df[ 1:, target ] ]
		self.numeric_columns = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
		self.text_columns = df.select_dtypes( include=[ 'object', 'category' ] ).columns.tolist( )
		self.training_data = \
		train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 0 ]
		self.testing_data = \
		train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 1 ]
		self.training_values = \
		train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 2 ]
		self.testing_values = \
		train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 3 ]
		self.transtuple = None


	def __dir__( self ):
		'''

			Purpose:
			-----------
			This function retuns a list of strings (members of the class)

		'''
		return [ 'dataframe', 'rows', 'columns', 'target_values', 'split_data',
		         'features', 'test_size', 'random_state', 'scale_data',
		         'numeric_columns', 'text_columns', 'scaler', 'transtuple', 'create_testing_data',
		         'calculate_statistics', 'create_training_data',
		         'target_values', 'training_data', 'testing_data', 'training_values',
		         'testing_values', 'transform_columns' ]


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
				_tuple = ( name, encoder, columns )
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


	def calculate_statistics( self ) -> pd.Series | None:
		"""

			Purpose:
			-----------
			Method calculating descriptive statistics.

			Returns:
			-----------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			statistics = self.dataframe.describe( )
			return statistics
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
			exception.method = 'caluclate_statistics( ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_training_data( self ) -> Tuple[ np.ndarray, np.ndarray ] | None:
		"""

			Purpose:
			-----------
			Return the training features and labels.

			Returns:
			-----------
			Tuple[ np.ndarray, np.ndarray ]: ( training_data, training_values )

		"""
		return tuple( self.training_data, self.training_values )


	def create_testing_data( self ) -> Tuple[ np.ndarray, np.ndarray ] | None:
		"""

			Purpose:
			-----------
			Return the test features and labels.

			Returns:
			-----------
			Tuple[ np.ndarray, np.ndarray ]: testing_data, testing_values

		"""
		return tuple( self.testing_data, self.testing_values )