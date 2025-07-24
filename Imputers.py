'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                Imputers.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="Imputers.py" company="Terry D. Eppler">

     Mathy Imputers

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
	Imputers.py
</summary>
******************************************************************************************
'''
from __future__ import annotations
from Booger import Error, ErrorDialog
import numpy as np
from typing import Optional
import sklearn.impute as sk
from pydantic import BaseModel


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


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> None:
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

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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


class MeanImputer( Metric ):
	"""

		Purpose:
		-----------
		Fills missing target_values using the average.

	"""


	def __init__( self, strat: str='mean' ) -> None:
		super( ).__init__( )
		self.mean_imputer = sk.SimpleImputer( strategy=strat )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""


			Purpose:
			---------
			Fits the simple_imputer to the df.

			Parameters:
			-----------
			X (np.ndarray): Input df with missing target_values.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.mean_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			---------
			Transforms the text
			df by filling in missing target_values.

			Parameters:
			-----------
			X (np.ndarray): Input df with missing target_values.

			Returns:
			-----------
			np.ndarray: Imputed df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.mean_imputer.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""
		Fit the iterative imputer and transform the data.

		:param y:
		:type y:
		:param X: Input array with missing values.
		:type X: np.ndarray
		:return: Transformed data with imputed values.
		:rtype: np.ndarray
		"""
		return self.mean_imputer.fit_transform( X )


class NearestNeighborImputer( Metric ):
	"""

		Purpose:
		---------
		Fills missing target_values using k-nearest neighbors.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.knn_imputer = sk.KNNImputer( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ):
		"""

			Purpose:
			________
			Fits the simple_imputer to the df.

			Parameters:
			_____
			X (np.ndarray): Input df with missing target_values.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.knn_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborImputer'
			exception.method = 'fit( self, X: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			_________

			Transforms the text df by imputing missing target_values.

			Parameters:
			-----------
			X (np.ndarray): Input df

			Returns:
			-----------
			np.ndarray: Imputed df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.knn_imputer.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Fit the iterative imputer and transform the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.knn_imputer.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class IterativeImputer( Metric ):
	"""


		Purpose:
		--------
		A strategy for imputing missing values by modeling each feature with
		missing values as a function of other features in a round-robin fashion.
	"""

	def __init__( self, max: int=10, rando: int=0 ) -> None:
		"""

			Purpose:
			--------
			Initialize the IterativeImputer.

			:param max: Maximum number of imputation iterations.
			:type max: int
			:param rando: Random seed.
			:type rando: int
		"""
		super( ).__init__( )
		self.iterative_imputer = sk.IterativeImputer( max_iter=max, random_state=rando )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""

			Purpose:
			--------
			Fit the iterative imputer to the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.iterative_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform data by iteratively imputing missing values.

			:param X: Data to transform.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray

		"""
		return self.iterative_imputer.transform( X )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit the iterative imputer and transform the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.iterative_imputer.fit_transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class SimpleImputer( Metric ):
	"""

		Wrapper for sklearn's SimpleImputer.

	"""

	def __init__( self, strategy: str='mean', fill_value: float=0.0 ) -> None:
		"""

			Purpose:
			---------
			Initialize the SimpleImputer.

			:param strategy: The imputation strategy ('mean', 'median', 'most_frequent', or 'constant').
			:type strategy: str
			:param fill_value: Value to use when strategy is 'constant'.
			:type fill_value: float

		"""
		super( ).__init__( )
		self.simple_imputer = sk.SimpleImputer( strategy=strategy, fill_value=fill_value )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""

			Purpose:
			--------
			Fit the imputer to the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.simple_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit( X: np.ndarray ) -> self'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Transform data by imputing missing values.

			:param X: Data to transform.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.simple_imputer.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit( X: np.ndarray ) -> self'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit the imputer and transform the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.simple_imputer.fit_transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
