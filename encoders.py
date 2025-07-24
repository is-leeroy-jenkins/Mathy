'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                scalers.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="encoders.py" company="Terry D. Eppler">

     Mathy Encoders

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
	encoders.py
</summary>
******************************************************************************************
'''
from __future__ import annotations

from booger import Error, ErrorDialog
import numpy as np
from typing import Optional
import sklearn.preprocessing as sk
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


class OneHotEncoder( Metric ):
	"""

		Purpose:
		---------
		Encodes categorical features as a one-hot numeric array.

	"""

	def __init__( self, unknown: str='ignore' ) -> None:
		super( ).__init__( )
		self.hot_encoder = sk.OneHotEncoder( sparse_output=False, handle_unknown=unknown )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""


			Purpose:
			---------
			Fits the hot_encoder to the categorical df.

			Parameters:
			-----------
			X (np.ndarray): Categorical text df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.hot_encoder.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			---------
			Transforms the text
			df into a one-hot encoded format.

			Parameters:
			-----------
			X (np.ndarray): Categorical text df.

			Returns:
			-----------
			np.ndarray: One-hot encoded matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.hot_encoder.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit the encoder and transform the data.

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
				return self.hot_encoder.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class OrdinalEncoder( Metric ):
	"""


			Purpose:
			---------
			Encodes categorical features as ordinal integers.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.ordinal_encoder = sk.OrdinalEncoder( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""

			Purpose:
			________
			Fits the ordial encoder to the categorical df.

			Parameters:
			_____
			X (np.ndarray): Categorical text df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.ordinal_encoder.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms the text df into ordinal-encoded format.


			Parameters:
			-----------
			X (np.ndarray): Categorical text df.

			Returns:
			-----------
			np.ndarray: Ordinal-encoded matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.ordinal_encoder.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit the encoder and transform the data.

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
				return self.ordinal_encoder.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class LabelEncoder( Metric ):
	"""

		Purpose:
		--------
		Wrapper for LabelEncoder.
	"""

	def __init__( self ) -> None:
		"""

			Purpose:
			--------
			Initialize LabelEncoder.
		"""
		super( ).__init__( )
		self.label_encoder = sk.LabelEncoder( )


	def fit( self, labels: list[ str ], y: Optional[ np.ndarray ]=None ) -> object | None:
		"""

			Purpose:
			--------
			Fit the label encoder to the data.

			:param y:
			:type y:
			:param labels: List of labels.
			:type labels: list[str]
		"""
		try:
			if labels is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.label_encoder.fit( labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LabellEncoder'
			exception.method = 'fit( self, labels: list[str] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, labels: list[ str ] ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Transform labels to encoded form.

			:param labels: List of labels.
			:type labels: list[str]
			:return: Encoded labels.
			:rtype: np.ndarray
		"""

		try:
			if labels is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.label_encoder.transform( labels )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LabellEncoder'
			exception.method = 'fit( self, labels: list[str] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, labels: list[ str ], y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit and transform the label data.

			:param y:
			:type y:
			:param labels: List of labels.
			:type labels: list[str]
			:return: Encoded labels.
			:rtype: np.ndarray
		"""

		try:
			if labels is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.label_encoder.fit_transform( labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LabellEncoder'
			exception.method = 'fit_transform( self, labels: list[str] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class PolynomialFeatures( Metric ):
	"""

		Purpose:
		--------
        Wrapper for PolynomialFeatures.
    """

	def __init__( self, degree: int=2 ) -> None:
		"""

			Purpose:
			--------
			Initialize PolynomialFeatures.

			:param degree: Degree of polynomial terms.
			:type degree: int
		"""
		super( ).__init__( )
		self.polynomial_features = sk.PolynomialFeatures( degree=degree )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""

			Purpose:
			--------
			Fit polynomial transformer to data.

			:param y:
			:type y:
			:param X: Feature matrix.
			:type X: np.ndarray
		"""
		self.polynomial_features.fit( X )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Transform data into polynomial features.

			:param X: Feature matrix.
			:type X: np.ndarray
			:return: Transformed feature matrix.
			:rtype: np.ndarray
		"""
		return self.polynomial_features.transform( X )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit and transform data using polynomial expansion.

			:param y:
			:type y:
			:param X: Feature matrix.
			:type X: np.ndarray
			:return: Transformed feature matrix.
			:rtype: np.ndarray
		"""
		return self.polynomial_features.fit_transform( X )