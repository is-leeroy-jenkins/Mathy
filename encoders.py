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
from boogr import Error

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )


class Encoder( ):
	"""

		Purpose:
		---------
		Base interface for encoder and transformer wrappers. Provides standard training,
		transformation, combined training/transformation, and inverse-transformation hooks.

	"""
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the shared encoder state.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		self.transformed_data = None
	
	def train( self, X: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
			Fit the underlying transformer to the input data.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples.

			Returns:
			--------
			object | None: The fitted encoder instance when implemented by a subclass.

		"""
		raise NotImplementedError( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform input data using a previously fitted encoder.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples.

			Returns:
			--------
			np.ndarray: Transformed output.

		"""
		raise NotImplementedError( )
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the encoder to the input data and return the transformed result.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples.

			Returns:
			--------
			np.ndarray: Transformed output.

		"""
		raise NotImplementedError( )
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Map transformed values back to their original representation when supported.

			Parameters:
			-----------
			X ( np.ndarray ): Transformed matrix or encoded values.

			Returns:
			--------
			np.ndarray: Inverse-transformed output.

		"""
		raise NotImplementedError( )


class OneHotEncoder( Encoder ):
	"""

		Purpose:
		---------
		Encode categorical features as a one-hot numeric array. The input to this
		transformer should be an array-like of integers or strings denoting the values
		taken on by categorical features. The features are encoded using a one-hot
		(aka one-of-K or dummy) encoding scheme.

		This creates a binary column for each category and returns a sparse matrix or
		dense array depending on the sparse_output parameter.

		By default, the encoder derives categories from the unique values in each
		feature. Alternatively, categories may be specified manually. This encoding is
		commonly used for feeding categorical data to scikit-learn estimators,
		especially linear models and support vector machines.

	"""
	unknown: Optional[ str ]
	sparse: Optional[ bool ]
	model: pp.OneHotEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, sparse: bool = False, unknown: str = 'ignore' ) -> None:
		"""

			Purpose:
			---------
			Initialize the one-hot encoder wrapper.

			Parameters:
			-----------
			sparse ( bool ): Specifies whether the transformed output should be returned
				as a sparse matrix.
			unknown ( str ): Strategy used to handle unknown categories during transform.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.unknown = unknown
		self.sparse = sparse
		self.model = pp.OneHotEncoder( sparse_output=self.sparse, handle_unknown=self.unknown )
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the list of exposed members for interactive inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: Member names exposed by the wrapper.

		"""
		return [
				'unknown',
				'model',
				'categories',
				'transformed_data',
				'sparse',
				'train',
				'transform',
				'train_transform',
				'inverse_transform',
		]
	
	@property
	def categories( self ):
		"""

			Purpose:
			---------
			Return the learned categories for each encoded feature.

			Parameters:
			-----------
			None

			Returns:
			--------
			object: Learned category arrays for each feature.

		"""
		if self.model.categories_ is None:
			raise AttributeError( 'Hot Encoder data is untrained' )
		else:
			return self.model.categories_
	
	def train( self, X: np.ndarray ) -> OneHotEncoder | None:
		"""

			Purpose:
			---------
			Fit the one-hot encoder to the categorical input matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).

			Returns:
			--------
			OneHotEncoder | None: Fitted encoder wrapper.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'train( self, X: np.ndarray ) -> OneHotEncoder | None'
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform the categorical matrix into one-hot encoded form.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: One-hot encoded matrix or sparse matrix, depending on the
				encoder configuration.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the one-hot encoder and transform the input matrix in one step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: One-hot encoded matrix or sparse matrix, depending on the
				encoder configuration.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Convert one-hot encoded values back to their original categorical form.

			Parameters:
			-----------
			X ( np.ndarray ): One-hot encoded matrix or sparse matrix.

			Returns:
			--------
			np.ndarray: Decoded categorical values.

		"""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception


class OrdinalEncoder( Encoder ):
	"""

		Purpose:
		---------
		Transform each categorical feature into a single integer-valued feature
		ranging from 0 to n_categories - 1.

		Although this representation is useful for some workflows, the encoded
		values may imply an ordering that does not exist in the original categories.
		As a result, ordinal encoding should be used with care when the source
		features are nominal rather than ordinal.

	"""
	model: pp.OrdinalEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the ordinal encoder wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.model = pp.OrdinalEncoder( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the list of exposed members for interactive inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: Member names exposed by the wrapper.

		"""
		return [
				'model',
				'categories',
				'transformed_data',
				'train',
				'transform',
				'train_transform',
				'inverse_transform',
		]
	
	@property
	def categories( self ):
		"""

			Purpose:
			---------
			Return the learned categories for each encoded feature.

			Parameters:
			-----------
			None

			Returns:
			--------
			object: Learned category arrays for each feature.

		"""
		if self.model.categories_ is None:
			raise AttributeError( 'Encoder data is untrained' )
		else:
			return self.model.categories_
	
	def train( self, X: np.ndarray ) -> OrdinalEncoder | None:
		"""

			Purpose:
			---------
			Fit the ordinal encoder to the categorical input matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).

			Returns:
			--------
			OrdinalEncoder | None: Fitted encoder wrapper.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'train( self, X: np.ndarray ) -> OrdinalEncoder | None'
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform the categorical matrix into ordinal-encoded form.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Ordinal-encoded matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the ordinal encoder and transform the input matrix in one step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Ordinal-encoded matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Map ordinal-encoded values back to their original categories.

			Parameters:
			-----------
			X ( np.ndarray ): Ordinal-encoded matrix.

			Returns:
			--------
			np.ndarray: Decoded categorical matrix.

		"""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception


class LabelEncoder( Encoder ):
	"""

		Purpose:
		---------
		Encode target labels with values between 0 and n_classes - 1.
		This transformer is intended for encoding a one-dimensional target vector,
		not a feature matrix.

	"""
	model: pp.LabelEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the label encoder wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.model = pp.LabelEncoder( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the list of exposed members for interactive inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: Member names exposed by the wrapper.

		"""
		return [
				'model',
				'classes',
				'transformed_data',
				'train',
				'transform',
				'train_transform',
				'inverse_transform',
		]
	
	@property
	def classes( self ):
		"""

			Purpose:
			---------
			Return the learned class labels.

			Parameters:
			-----------
			None

			Returns:
			--------
			object: Learned class labels.

		"""
		if self.model.classes_ is None:
			raise AttributeError( 'Label Encoder data is untrained' )
		else:
			return self.model.classes_
	
	def train( self, X: np.ndarray ) -> LabelEncoder | None:
		"""

			Purpose:
			---------
			Fit the label encoder to the target vector.

			Parameters:
			-----------
			X ( np.ndarray ): Target labels of shape ( n_samples, ).

			Returns:
			--------
			LabelEncoder | None: Fitted encoder wrapper.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'train( self, X: np.ndarray ) -> LabelEncoder | None'
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform the target labels into encoded integer values.

			Parameters:
			-----------
			X ( np.ndarray ): Target labels of shape ( n_samples, ).

			Returns:
			--------
			np.ndarray: Encoded label vector.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the label encoder and transform the target labels in one step.

			Parameters:
			-----------
			X ( np.ndarray ): Target labels of shape ( n_samples, ).

			Returns:
			--------
			np.ndarray: Encoded label vector.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Map encoded integer labels back to their original values.

			Parameters:
			-----------
			X ( np.ndarray ): Encoded label vector.

			Returns:
			--------
			np.ndarray: Decoded label vector.

		"""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception


class TargetEncoder( Encoder ):
	"""

		Purpose:
		---------
		Encode categorical features using the target values associated with each category.
		Each category is encoded using a shrunk estimate of the target mean conditioned on
		the category value and the global target mean.

		For multiclass targets, encodings are based on one-vs-all conditional target
		probabilities, which produces n_features * n_classes encoded output features.

		Missing values are treated as their own category. Categories not seen during
		training are encoded with the learned global target mean.

	"""
	model: pp.TargetEncoder
	transformed_data: Optional[ np.ndarray ]
	categories: Optional[ str ]
	smoothing: Optional[ str ]
	target_type: Optional[ str ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the target encoder wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.model = pp.TargetEncoder( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the list of exposed members for interactive inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: Member names exposed by the wrapper.

		"""
		return [
				'model',
				'classes',
				'encodings',
				'features_in',
				'categories',
				'transformed_data',
				'train',
				'transform',
				'train_transform',
		]
	
	@property
	def classes( self ):
		"""

			Purpose:
			---------
			Return the learned class labels when the target type is multiclass.

			Parameters:
			-----------
			None

			Returns:
			--------
			object: Learned class labels.

		"""
		if self.model.classes_ is None:
			raise AttributeError( 'The target encoder data is untrained.' )
		else:
			return self.model.classes_
	
	@property
	def encodings( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the learned category encodings.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray: Encodings learned from the training data.

		"""
		if self.model.encodings_ is None:
			raise AttributeError( 'The target encoder data is untrained.' )
		else:
			return self.model.encodings_
	
	@property
	def features_in( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of input features seen during fitting.

			Parameters:
			-----------
			None

			Returns:
			--------
			int: Number of fitted input features.

		"""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The target encoder data is untrained.' )
		else:
			return self.model.n_features_in_
	
	@property
	def categories( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the learned categories for each input feature.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray: Learned categories for each input feature.

		"""
		if self.model.categories_ is None:
			raise AttributeError( 'The target encoder data is untrained.' )
		else:
			return self.model.categories_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> TargetEncoder | None:
		"""

			Purpose:
			---------
			Fit the target encoder to the feature matrix and target vector.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).
			y ( np.ndarray ): Target vector of shape ( n_samples, ).

			Returns:
			--------
			TargetEncoder | None: Fitted encoder wrapper.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TargetEncoder'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> TargetEncoder | None'
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform the feature matrix using encodings learned during fitting.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Target-encoded feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TargetEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the target encoder and transform the training data in one step using
			scikit-learn's internal cross-fitting behavior.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).
			y ( np.ndarray ): Target vector of shape ( n_samples, ).

			Returns:
			--------
			np.ndarray: Target-encoded feature matrix.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TargetEncoder'
			exception.method = 'train_transform( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception


class PolynomialFeatures( Encoder ):
	"""

		Purpose:
		---------
		Generate polynomial and interaction features from the input feature matrix.
		This transformer creates a new feature matrix consisting of all polynomial
		combinations of the input features with degree less than or equal to the
		specified degree. For example, if an input sample is two-dimensional and of
		the form [a, b], the degree-2 polynomial features are
		[1, a, b, a^2, ab, b^2].

	"""
	degree: Optional[ int ]
	interaction_only: Optional[ bool ]
	model: pp.PolynomialFeatures
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, degree: int = 2, interaction: bool = True ) -> None:
		"""

			Purpose:
			---------
			Initialize the polynomial feature transformer wrapper.

			Parameters:
			-----------
			degree ( int ): Maximum polynomial degree to generate.
			interaction ( bool ): Specifies whether only interaction terms should be
				produced, excluding powers of the same feature.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.degree = degree
		self.interaction_only = interaction
		self.model = pp.PolynomialFeatures( degree=self.degree, interaction_only=self.interaction_only )
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the list of exposed members for interactive inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: Member names exposed by the wrapper.

		"""
		return [
				'model',
				'degree',
				'powers',
				'interaction_only',
				'transformed_data',
				'train',
				'transform',
				'train_transform',
		]
	
	@property
	def powers( self ):
		"""

			Purpose:
			---------
			Return the exponent for each input feature in each generated output feature.

			Parameters:
			-----------
			None

			Returns:
			--------
			object: Polynomial exponent mapping for generated features.

		"""
		if self.model.powers_ is None:
			raise AttributeError( 'The polynomial data is untrained.' )
		else:
			return self.model.powers_
	
	def train( self, X: np.ndarray ) -> PolynomialFeatures | None:
		"""

			Purpose:
			---------
			Fit the polynomial transformer to the input feature matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).

			Returns:
			--------
			PolynomialFeatures | None: Fitted transformer wrapper.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'train( self, X: np.ndarray ) -> PolynomialFeatures | None'
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform the input matrix into polynomial feature space.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Polynomially expanded feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the polynomial transformer and transform the input matrix in one step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix or input samples of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Polynomially expanded feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
