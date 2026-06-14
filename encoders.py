"""******************************************************************************************
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
    Provides encoder and feature-expansion wrappers over sklearn.preprocessing for Mathy
    preprocessing workflows. The module centralizes categorical encoding, label encoding,
    target encoding, and polynomial feature generation behind a consistent training,
    transformation, and inverse-transformation interface.
</summary>
******************************************************************************************"""
from __future__ import annotations
from typing import Optional, List, Any
import numpy as np
import sklearn.preprocessing as pp
from boogr import Error, Logger

def throw_if( name: str, value: object ):
	"""Validate a required argument.

	Purpose:
	    Raises a validation error when a required argument is missing. The helper provides a
	    consistent guard for encoder operations before sklearn preprocessing methods are called.

	Args:
	    name (str): Argument name used in the validation error message.
	    value (object): Argument value checked for missing state.

	Raises:
	    ValueError: Raised when `value` is `None`."""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Encoder( ):
	"""Define the encoder interface.

	Purpose:
	    Defines the shared encoder contract used by Mathy preprocessing wrappers. The base class
	    establishes a common transformed-data attribute and requires concrete subclasses to
	    implement fit, transform, fit-transform, and inverse-transform behavior where supported.

	Attributes:
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by a
	        concrete encoder wrapper."""
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""Initialize encoder state.

		Purpose:
		    Initializes the common transformed-data cache used by concrete encoder wrappers to store
		    the most recent output returned by sklearn preprocessing operations."""
		self.transformed_data = None
	
	def train( self, X: np.ndarray ) -> object | None:
		"""Fit an encoder.

		Purpose:
		    Defines the required training contract for concrete encoder wrappers. Subclasses must fit
		    their underlying sklearn preprocessing object to the supplied input and return a fitted
		    wrapper or compatible result.

		Args:
		    X (np.ndarray): Feature matrix, categorical matrix, or target vector supplied to the
		        concrete encoder implementation.

		Returns:
		    Fitted concrete encoder wrapper or implementation-specific training result.

		Raises:
		    NotImplementedError: Raised when the base method is called directly."""
		raise NotImplementedError( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""Transform input values.

		Purpose:
		    Defines the required transformation contract for concrete encoder wrappers. Subclasses
		    must transform supplied input with a fitted preprocessing object and return the encoded
		    output.

		Args:
		    X (np.ndarray): Feature matrix, categorical matrix, or target vector supplied to the
		        concrete encoder implementation.

		Returns:
		    Encoded output produced by the concrete encoder wrapper.

		Raises:
		    NotImplementedError: Raised when the base method is called directly."""
		raise NotImplementedError( )
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Fit and transform input values.

		Purpose:
		    Defines the required combined fit-transform contract for concrete encoder wrappers.
		    Subclasses must fit their preprocessing object and return the encoded output in one
		    operation.

		Args:
		    X (np.ndarray): Feature matrix, categorical matrix, or target vector supplied to the
		        concrete encoder implementation.

		Returns:
		    Encoded output produced after fitting the concrete encoder wrapper.

		Raises:
		    NotImplementedError: Raised when the base method is called directly."""
		raise NotImplementedError( )
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Decode transformed values.

		Purpose:
		    Defines the required inverse-transformation contract for concrete encoder wrappers.
		    Subclasses must map encoded values back to their source representation when the wrapped
		    sklearn object supports decoding.

		Args:
		    X (np.ndarray): Encoded matrix or vector supplied to the concrete encoder implementation.

		Returns:
		    Decoded output produced by the concrete encoder wrapper.

		Raises:
		    NotImplementedError: Raised when the base method is called directly."""
		raise NotImplementedError( )

class OneHotEncoder( Encoder ):
	"""Wrap sklearn OneHotEncoder.

	Purpose:
	    Encodes categorical feature columns as a one-hot numeric representation. The wrapper
	    configures sparse-output behavior and unknown-category handling while preserving a uniform
	    Mathy encoder interface for training, transformation, and inverse transformation.

	Attributes:
	    unknown (Optional[str]): Strategy used by sklearn for unknown categories during transform.
	    sparse (Optional[bool]): Flag controlling sparse or dense transformed output.
	    model (pp.OneHotEncoder): Underlying sklearn preprocessing estimator.
	    transformed_data (Optional[np.ndarray]): Most recent encoded output produced by the wrapper."""
	unknown: Optional[ str ]
	sparse: Optional[ bool ]
	model: pp.OneHotEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, sparse: bool = False, unknown: str = 'ignore' ) -> None:
		"""Initialize OneHotEncoder.

		Purpose:
		    Initializes the one-hot encoder wrapper with sklearn sparse-output and unknown-category
		    configuration. The constructor prepares the backing sklearn model without fitting it to
		    any data.

		Args:
		    sparse (bool): Flag indicating whether transformed output should be returned as a sparse
		        matrix.
		    unknown (str): Strategy used to handle unknown categories during transformation."""
		super( ).__init__( )
		self.unknown = unknown
		self.sparse = sparse
		self.model = pp.OneHotEncoder( sparse_output=self.sparse, handle_unknown=self.unknown )
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.

		Purpose:
		    Returns the stable set of member names exposed by the one-hot encoder wrapper for
		    interactive inspection, notebook exploration, and IDE discovery.

		Returns:
		    Public member names exposed by the wrapper."""
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
	def categories( self ) -> List[ Any ]:
		"""Return learned categories.

		Purpose:
		    Returns the category arrays learned by the fitted one-hot encoder for each input feature.
		    The property enforces fitted-state access by raising an attribute error when categories
		    are not available.

		Returns:
		    Learned category arrays for each encoded feature.

		Raises:
		    AttributeError: Raised when the encoder has not been fitted."""
		if self.model.categories_ is None:
			raise AttributeError( 'Hot Encoder data is untrained' )
		else:
			return self.model.categories_
	
	def train( self, X: np.ndarray ) -> OneHotEncoder | None:
		"""Fit the one-hot encoder.

		Purpose:
		    Fits the underlying sklearn one-hot encoder to a categorical input matrix and returns the
		    wrapper instance for consistent preprocessing chains.

		Args:
		    X (np.ndarray): Categorical feature matrix used to learn category levels.

		Returns:
		    Fitted one-hot encoder wrapper.

		Raises:
		    Error: Raised when validation or sklearn fitting fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'train( self, X: np.ndarray ) -> OneHotEncoder | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""Transform categorical values.

		Purpose:
		    Encodes the supplied categorical matrix using category levels learned during fitting and
		    stores the encoded output on the wrapper for later inspection.

		Args:
		    X (np.ndarray): Categorical feature matrix transformed by the fitted encoder.

		Returns:
		    One-hot encoded matrix produced by the fitted encoder.

		Raises:
		    Error: Raised when validation or sklearn transformation fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Fit and transform categorical values.

		Purpose:
		    Fits the underlying one-hot encoder to the supplied categorical matrix and returns the
		    encoded matrix in one operation. The encoded output is cached on the wrapper.

		Args:
		    X (np.ndarray): Categorical feature matrix used for fitting and transformation.

		Returns:
		    One-hot encoded matrix produced after fitting the encoder.

		Raises:
		    Error: Raised when validation or sklearn fit-transform execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Decode one-hot values.

		Purpose:
		    Converts one-hot encoded rows back to their original categorical representation using the
		    inverse transformation provided by the fitted sklearn encoder.

		Args:
		    X (np.ndarray): One-hot encoded matrix to decode.

		Returns:
		    Decoded categorical matrix.

		Raises:
		    Error: Raised when validation or sklearn inverse transformation fails."""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class OrdinalEncoder( Encoder ):
	"""Wrap sklearn OrdinalEncoder.

	Purpose:
	    Converts categorical feature values into integer-like ordinal codes ranging from zero to
	    one less than the number of observed categories. The wrapper exposes sklearn ordinal
	    encoding through the shared Mathy encoder interface.

	Attributes:
	    model (pp.OrdinalEncoder): Underlying sklearn preprocessing estimator.
	    transformed_data (Optional[np.ndarray]): Most recent encoded output produced by the wrapper."""
	model: pp.OrdinalEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""Initialize OrdinalEncoder.

		Purpose:
		    Initializes the ordinal encoder wrapper by constructing the underlying sklearn encoder and
		    preparing the transformed-data cache used by later transformation methods."""
		super( ).__init__( )
		self.model = pp.OrdinalEncoder( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.

		Purpose:
		    Returns the stable set of member names exposed by the ordinal encoder wrapper for
		    interactive inspection, notebook exploration, and IDE discovery.

		Returns:
		    Public member names exposed by the wrapper."""
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
	def categories( self ) -> List[ Any ]:
		"""Return learned categories.

		Purpose:
		    Returns the category arrays learned by the fitted ordinal encoder for each input feature.
		    The property enforces fitted-state access by raising an attribute error when categories
		    are not available.

		Returns:
		    Learned category arrays for each encoded feature.

		Raises:
		    AttributeError: Raised when the encoder has not been fitted."""
		if self.model.categories_ is None:
			raise AttributeError( 'Encoder data is untrained' )
		else:
			return self.model.categories_
	
	def train( self, X: np.ndarray ) -> OrdinalEncoder | None:
		"""Fit the ordinal encoder.

		Purpose:
		    Fits the underlying sklearn ordinal encoder to a categorical input matrix and returns the
		    wrapper instance for consistent preprocessing chains.

		Args:
		    X (np.ndarray): Categorical feature matrix used to learn category levels.

		Returns:
		    Fitted ordinal encoder wrapper.

		Raises:
		    Error: Raised when validation or sklearn fitting fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'train( self, X: np.ndarray ) -> OrdinalEncoder | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""Transform categorical values.

		Purpose:
		    Encodes the supplied categorical matrix using ordinal category codes learned during
		    fitting and stores the encoded output on the wrapper.

		Args:
		    X (np.ndarray): Categorical feature matrix transformed by the fitted encoder.

		Returns:
		    Ordinal-encoded matrix produced by the fitted encoder.

		Raises:
		    Error: Raised when validation or sklearn transformation fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Fit and transform categorical values.

		Purpose:
		    Fits the underlying ordinal encoder to the supplied categorical matrix and returns the
		    encoded matrix in one operation. The encoded output is cached on the wrapper.

		Args:
		    X (np.ndarray): Categorical feature matrix used for fitting and transformation.

		Returns:
		    Ordinal-encoded matrix produced after fitting the encoder.

		Raises:
		    Error: Raised when validation or sklearn fit-transform execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Decode ordinal values.

		Purpose:
		    Converts ordinal-encoded rows back to their original categorical representation using the
		    inverse transformation provided by the fitted sklearn encoder.

		Args:
		    X (np.ndarray): Ordinal-encoded matrix to decode.

		Returns:
		    Decoded categorical matrix.

		Raises:
		    Error: Raised when validation or sklearn inverse transformation fails."""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class LabelEncoder( Encoder ):
	"""Wrap sklearn LabelEncoder.

	Purpose:
	    Encodes one-dimensional target labels as integer values from zero to one less than the
	    number of observed classes. The wrapper exposes sklearn label encoding through the shared
	    Mathy encoder interface.

	Attributes:
	    model (pp.LabelEncoder): Underlying sklearn preprocessing estimator.
	    transformed_data (Optional[np.ndarray]): Most recent encoded label vector produced by the wrapper."""
	model: pp.LabelEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""Initialize LabelEncoder.

		Purpose:
		    Initializes the label encoder wrapper by constructing the underlying sklearn encoder and
		    preparing the transformed-data cache used by later label transformation methods."""
		super( ).__init__( )
		self.model = pp.LabelEncoder( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.

		Purpose:
		    Returns the stable set of member names exposed by the label encoder wrapper for interactive
		    inspection, notebook exploration, and IDE discovery.

		Returns:
		    Public member names exposed by the wrapper."""
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
	def classes( self ) -> Any:
		"""Return learned classes.

		Purpose:
		    Returns the class labels learned by the fitted label encoder. The property enforces
		    fitted-state access by raising an attribute error when class labels are not available.

		Returns:
		    Learned class labels.

		Raises:
		    AttributeError: Raised when the encoder has not been fitted."""
		if self.model.classes_ is None:
			raise AttributeError( 'Label Encoder data is untrained' )
		else:
			return self.model.classes_
	
	def train( self, X: np.ndarray ) -> LabelEncoder | None:
		"""Fit the label encoder.

		Purpose:
		    Fits the underlying sklearn label encoder to a one-dimensional label vector and returns the
		    wrapper instance for consistent preprocessing chains.

		Args:
		    X (np.ndarray): Target-label vector used to learn class labels.

		Returns:
		    Fitted label encoder wrapper.

		Raises:
		    Error: Raised when validation or sklearn fitting fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'train( self, X: np.ndarray ) -> LabelEncoder | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""Transform label values.

		Purpose:
		    Encodes the supplied target-label vector using class labels learned during fitting and
		    stores the encoded output on the wrapper.

		Args:
		    X (np.ndarray): Target-label vector transformed by the fitted encoder.

		Returns:
		    Encoded label vector.

		Raises:
		    Error: Raised when validation or sklearn transformation fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Fit and transform label values.

		Purpose:
		    Fits the underlying label encoder to the supplied target vector and returns the encoded
		    label vector in one operation. The encoded output is cached on the wrapper.

		Args:
		    X (np.ndarray): Target-label vector used for fitting and transformation.

		Returns:
		    Encoded label vector produced after fitting the encoder.

		Raises:
		    Error: Raised when validation or sklearn fit-transform execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Decode label values.

		Purpose:
		    Converts encoded integer labels back to their original class labels using the inverse
		    transformation provided by the fitted sklearn label encoder.

		Args:
		    X (np.ndarray): Encoded label vector to decode.

		Returns:
		    Decoded label vector.

		Raises:
		    Error: Raised when validation or sklearn inverse transformation fails."""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class TargetEncoder( Encoder ):
	"""Wrap sklearn TargetEncoder.

	Purpose:
	    Encodes categorical feature values with target-conditioned statistics learned from paired
	    features and target values. The wrapper exposes sklearn target encoding through the shared
	    Mathy encoder interface and caches transformed output for downstream modeling workflows.

	Attributes:
	    model (pp.TargetEncoder): Underlying sklearn preprocessing estimator.
	    transformed_data (Optional[np.ndarray]): Most recent target-encoded matrix produced by the wrapper.
	    categories (Optional[str]): Category configuration metadata retained for interface compatibility.
	    smoothing (Optional[str]): Smoothing configuration metadata retained for interface compatibility.
	    target_type (Optional[str]): Target-type metadata retained for interface compatibility."""
	model: pp.TargetEncoder
	transformed_data: Optional[ np.ndarray ]
	categories: Optional[ str ]
	smoothing: Optional[ str ]
	target_type: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize TargetEncoder.

		Purpose:
		    Initializes the target encoder wrapper by constructing the underlying sklearn encoder and
		    preparing the transformed-data cache used by later target-aware transformations."""
		super( ).__init__( )
		self.model = pp.TargetEncoder( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.

		Purpose:
		    Returns the stable set of member names exposed by the target encoder wrapper for interactive
		    inspection, notebook exploration, and IDE discovery.

		Returns:
		    Public member names exposed by the wrapper."""
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
	def classes( self ) -> Any:
		"""Return learned target classes.

		Purpose:
		    Returns the learned class labels when the fitted target encoder is operating with a
		    multiclass target. The property enforces fitted-state access by raising an attribute error
		    when classes are not available.

		Returns:
		    Learned class labels for multiclass target encoding.

		Raises:
		    AttributeError: Raised when the target encoder has not been fitted."""
		if self.model.classes_ is None:
			raise AttributeError( 'The target encoder data is untrained.' )
		else:
			return self.model.classes_
	
	@property
	def encodings( self ) -> np.ndarray:
		"""Return learned target encodings.

		Purpose:
		    Returns the category encoding arrays learned during target encoder fitting. These values
		    represent target-conditioned mappings used to transform categorical feature values.

		Returns:
		    Learned category encodings from the fitted target encoder.

		Raises:
		    AttributeError: Raised when the target encoder has not been fitted."""
		if self.model.encodings_ is None:
			raise AttributeError( 'The target encoder data is untrained.' )
		else:
			return self.model.encodings_
	
	@property
	def features_in( self ) -> int:
		"""Return fitted feature count.

		Purpose:
		    Returns the number of input features observed by the target encoder during fitting. The
		    property provides fitted metadata required for inspection and compatibility checks.

		Returns:
		    Number of input features seen during fitting.

		Raises:
		    AttributeError: Raised when the target encoder has not been fitted."""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The target encoder data is untrained.' )
		else:
			return self.model.n_features_in_
	
	@property
	def categories( self ) -> np.ndarray:
		"""Return learned categories.

		Purpose:
		    Returns the category arrays learned by the fitted target encoder for each input feature.
		    These categories define the values eligible for target-conditioned encoding during
		    transformation.

		Returns:
		    Learned categories for each input feature.

		Raises:
		    AttributeError: Raised when the target encoder has not been fitted."""
		if self.model.categories_ is None:
			raise AttributeError( 'The target encoder data is untrained.' )
		else:
			return self.model.categories_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> TargetEncoder | None:
		"""Fit the target encoder.

		Purpose:
		    Fits the underlying sklearn target encoder to categorical features and aligned target
		    values so category-level encodings can be learned from target-conditioned statistics.

		Args:
		    X (np.ndarray): Categorical feature matrix used to learn target encodings.
		    y (np.ndarray): Target vector aligned to the rows of `X`.

		Returns:
		    Fitted target encoder wrapper.

		Raises:
		    Error: Raised when validation or sklearn fitting fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TargetEncoder'
			exception.method = 'train( self, *args ) -> TargetEncoder | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""Transform categorical values.

		Purpose:
		    Applies learned target encodings to the supplied categorical feature matrix and stores the
		    target-encoded output on the wrapper.

		Args:
		    X (np.ndarray): Categorical feature matrix transformed by the fitted target encoder.

		Returns:
		    Target-encoded feature matrix.

		Raises:
		    Error: Raised when validation or sklearn transformation fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TargetEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Fit and transform categorical values.

		Purpose:
		    Fits the underlying target encoder and returns target-encoded training data in one
		    operation. The method uses sklearn target encoding behavior for paired feature and target
		    arrays and caches the transformed result on the wrapper.

		Args:
		    X (np.ndarray): Categorical feature matrix used for fitting and transformation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.

		Returns:
		    Target-encoded feature matrix produced after fitting the encoder.

		Raises:
		    Error: Raised when validation or sklearn fit-transform execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TargetEncoder'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class PolynomialFeatures( Encoder ):
	"""Wrap sklearn PolynomialFeatures.

	Purpose:
	    Generates polynomial and interaction terms from numeric input features. The wrapper exposes
	    sklearn polynomial feature expansion through the shared Mathy encoder interface and retains
	    the generated output for downstream modeling workflows.

	Attributes:
	    degree (Optional[int]): Maximum polynomial degree generated by the transformer.
	    interaction_only (Optional[bool]): Flag indicating whether only interaction terms are generated.
	    model (pp.PolynomialFeatures): Underlying sklearn preprocessing estimator.
	    transformed_data (Optional[np.ndarray]): Most recent polynomial feature matrix produced by the wrapper."""
	degree: Optional[ int ]
	interaction_only: Optional[ bool ]
	model: pp.PolynomialFeatures
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, degree: int = 2, interaction: bool = True ) -> None:
		"""Initialize PolynomialFeatures.

		Purpose:
		    Initializes the polynomial feature wrapper by configuring maximum degree and interaction-only
		    behavior on the underlying sklearn transformer. The constructor prepares the model without
		    fitting it to any data.

		Args:
		    degree (int): Maximum polynomial degree generated by the transformer.
		    interaction (bool): Flag indicating whether only interaction terms are produced."""
		super( ).__init__( )
		self.degree = degree
		self.interaction_only = interaction
		self.model = pp.PolynomialFeatures( degree=self.degree,
			interaction_only=self.interaction_only )
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.

		Purpose:
		    Returns the stable set of member names exposed by the polynomial feature wrapper for
		    interactive inspection, notebook exploration, and IDE discovery.

		Returns:
		    Public member names exposed by the wrapper."""
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
	def powers( self ) -> np.ndarray:
		"""Return polynomial powers.

		Purpose:
		    Returns the exponent matrix generated by the fitted polynomial transformer. Each row
		    describes the powers applied to input features for one generated output feature.

		Returns:
		    Polynomial exponent matrix for generated output features.

		Raises:
		    AttributeError: Raised when the polynomial transformer has not been fitted."""
		if self.model.powers_ is None:
			raise AttributeError( 'The polynomial data is untrained.' )
		else:
			return self.model.powers_
	
	def train( self, X: np.ndarray ) -> PolynomialFeatures | None:
		"""Fit the polynomial transformer.

		Purpose:
		    Fits the underlying sklearn polynomial feature transformer to the supplied feature matrix
		    and returns the wrapper instance for consistent preprocessing chains.

		Args:
		    X (np.ndarray): Feature matrix used to establish polynomial feature metadata.

		Returns:
		    Fitted polynomial feature wrapper.

		Raises:
		    Error: Raised when validation or sklearn fitting fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'train( self, X: np.ndarray ) -> PolynomialFeatures | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""Transform polynomial features.

		Purpose:
		    Expands the supplied feature matrix into the polynomial feature space learned or configured
		    by the underlying sklearn transformer and caches the transformed matrix on the wrapper.

		Args:
		    X (np.ndarray): Feature matrix transformed into polynomial feature space.

		Returns:
		    Polynomially expanded feature matrix.

		Raises:
		    Error: Raised when validation or sklearn transformation fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Fit and transform polynomial features.

		Purpose:
		    Fits the polynomial feature transformer to the supplied feature matrix and returns the
		    polynomially expanded output in one operation. The expanded matrix is cached on the wrapper.

		Args:
		    X (np.ndarray): Feature matrix used for fitting and polynomial expansion.

		Returns:
		    Polynomially expanded feature matrix produced after fitting the transformer.

		Raises:
		    Error: Raised when validation or sklearn fit-transform execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception