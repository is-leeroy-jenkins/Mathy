'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
	     Copyright ©  2022  Terry Eppler

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
    name.py
  </summary>
  ******************************************************************************************
  '''
from __future__ import annotations
from typing import Optional, List
import numpy as np
import sklearn.preprocessing as pp
from boogr import Error, ErrorDialog

def throw_if( name: str, value: object ):
	if not value:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Encoder( ):
	"""

		Purpose:
		---------
		Base interface for all preprocessors. Provides standard `fit`, `transform`, and
		`fit_transform` methods.

	"""
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ):
		self.transformed_data = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> object | None:
		"""

			Purpose:
			---------
			Train/flex hook; return self in concrete subclasses.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )
			y ( Optional[ np.ndarray ] ): Optional target array  of shape ( n_samples, ).

			Returns:
			-----------
			object | None

		"""
		raise NotImplementedError
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform X using a fitted preprocessor; return transformed X.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit to X (and y if used) then transform X in one step.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError
	
	def inverse_transform( self, text: list[ str ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Invert the transformation when supported; raise NotImplementedError otherwise.

			:param text: List of text.
			:type text: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		return NotImplementedError

class OneHotEncoder( Encoder ):
	"""

		Purpose:
		---------
		Encode categorical feature_names as a one-hot numeric array. The input to this transformer
		should be an array-like of integers or strings, denoting the values taken on by categorical
		(discrete) feature_names. The feature_names are encoded using a one-hot
		(aka ‘one-of-K’ or ‘dummy’) encoding scheme.
		This creates a binary column for each category and returns a sparse
		matrix or dense array (depending on the sparse_output parameter)

		By default, the encoder derives the categories based on the unique values in each feature.
		Alternatively, you can also specify the categories manually. This encoding is needed for
		feeding categorical stores to many scikit-learn estimators, notably linear models and SVMs
		with the standard kernels. Note: a one-hot encoding of y target_names should use a
		LabelBinarizer instead.

	"""
	unknown: Optional[ str ]
	sparse: Optional[ bool ]
	encoder: pp.OneHotEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, sparse: bool=False, unknown: str='ignore' ) -> None:
		super( ).__init__( )
		self.unknown = unknown
		self.sparse = sparse
		self.encoder = pp.OneHotEncoder( sparse_output=self.sparse, handle_unknown=self.unknown )
	
	@property
	def categories( self ):
		if self.encoder.categories_ is None:
			raise AttributeError( 'Hot Encoder data is untrained' )
		else:
			return self.encoder.categories_
	
	def train( self, X: np.ndarray, y: np.ndarray=None ) -> OneHotEncoder | None:
		"""


			Purpose:
			---------
			Fits the hot_encoder to the categorical df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): IGNORED. Target vector of shape ( n_samples, ).

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.encoder.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = ('fit( self, X: np.ndarray, y: np.ndarray=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""


			Purpose:
			---------
			Transforms the categorical matrix into one-hot encoded form

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): IGNORED. Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: One-hot encoded matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.encoder.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit the encoder and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): IGNORED. Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.encoder.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class OrdinalEncoder( Encoder ):
	"""


			Purpose:
			---------
			This estimator transforms each categorical feature to one new feature of integers
			(0 to n_categories - 1):

			Such integer representation can, however, not be used directly with all scikit-learn
			estimators, as these expect continuous input, and would interpret the categories as
			being ordered, which is often not desired (i.e. the set of browsers was
			ordered arbitrarily).

			By default, OrdinalEncoder will also passthrough missing values that are indicated
			by np.nan. OrdinalEncoder provides a parameter encoded_missing_value to encode
			the missing values without the need to create a pipeline and using SimpleImputer.

	"""
	encoder: pp.OrdinalEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.encoder = pp.OrdinalEncoder( )
		self.transformed_data = None
	
	@property
	def categories( self ):
		if self.encoder.categories_ is None:
			raise AttributeError( 'Encoder data is untrained' )
		else:
			return self.encoder.categories_
	
	def train( self, X: np.ndarray, y: np.ndarray=None ) -> OrdinalEncoder | None:
		"""

			Purpose:
			________
			Fits the ordial encoder to the categorical df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.encoder.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'fit( self, X: np.ndarray, y: np.ndarray=None ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms the text df into ordinal-encoded format.


			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Ordinal-encoded matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.encoder.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit the encoder and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.encoder.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Map ordinal-encoded matrix back to original categories.

			Parameters:
			-----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( '"X" cannot be None' )
			else:
				throw_if( 'X', X )
				return self.encoder.inverse_transform( X ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class LabelEncoder( Encoder ):
	"""

		Purpose:
		--------
		Encode target target_names with value between 0 and n_classes-1.
		This transformer should be used to encode target values, i.e. y, and not the input X.

	"""
	encoder: pp.LabelEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			--------
			Initialize LabelEncoder.

		"""
		super( ).__init__( )
		self.encoder = pp.LabelEncoder( )
	
	@property
	def classes( self ):
		if self.encoder.classes_ is None:
			raise AttributeError( 'The label encoder data is untrained.' )
		else:
			return self.encoder.classes_
	
	def train( self, y: np.ndarray ) -> LabelEncoder | None:
		"""

			Purpose:
			--------
			Fit the label encoder to the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'y', y )
			self.encoder.fit( y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'fit( self, y: np.ndarray ) -> LabelEncoder'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Transform target_names to encoded form.

			Parameters:
			-----------
			X ( List[ str ] ): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		
		try:
			throw_if( 'y', y )
			self.transformed_data = self.encoder.transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'transform( self, y: np.ndarray  ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit and transform the label stores.

			Parameters:
			-----------
			X ( List[ str ] ): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		
		try:
			throw_if( 'y', y )
			self.transformed_data = self.encoder.fit_transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'fit_transform( self, y: np.ndarray  ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Map integer labels back to original classes.

			Parameters:
			-----------
			y: np.ndarray

		"""
		try:
			throw_if( 'y', y )
			return self.encoder.inverse_transform( y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class PolynomialFeatures( Encoder ):
	"""

		Purpose:
		--------
		Generate a new feature matrix consisting of all polynomial combinations of the features
		with degree less than or equal to the specified degree. For example, if an input sample is
		two-dimensional and of the form [a, b], the degree-2 polynomial
		features are [1, a, b, a^2, ab, b^2].




	"""
	degree: Optional[ int ]
	interaction_only: Optional[ bool ]
	encoder: pp.PolynomialFeatures
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, degree: int = 2, interaction: bool = True ) -> None:
		"""

			Purpose:
			--------
			Initialize PolynomialFeatures.

			:param degree: Degree of polynomial terms.
			:type degree: int
		"""
		super( ).__init__( )
		self.degree = degree
		self.interaction_only = interaction
		self.encoder = pp.PolynomialFeatures( degree=self.degree,
			interaction_only=self.interaction_only )
	
	@property
	def powers( self ):
		if self.encoder.powers_ is None:
			raise AttributeError( 'The polynomial data is untrained.' )
		else:
			return self.encoder.powers_
	
	def train( self, X: np.ndarray, y: np.ndarray = None ) -> PolynomialFeatures | None:
		"""

			Purpose:
			--------
			Fit polynomial transformer to stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.encoder.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'fit( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray = None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Transform stores into polynomial feature_names.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.encoder.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray, y: np.ndarray = None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit and transform stores using polynomial expansion.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.encoder.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
