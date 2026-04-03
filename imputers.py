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

from typing import Optional
import numpy as np
import sklearn.impute as im
from boogr import Error
from sklearn.experimental import enable_iterative_imputer


def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )


class Imputer( ):
	"""

		Purpose:
		---------
		Provides a common interface for imputations built on top of the
		scikit-learn transformer API. Concrete subclasses must implement training
		and transformation behavior for a specific imputer.

	"""
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the common imputer state.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		self.transformed_data = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
		"""

			Purpose:
			---------
			Fit the imputer to the input feature matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. Imputers do not use this value.

			Returns:
			--------
			object | None: Concrete wrapper instance when implemented by a subclass.

		"""
		raise NotImplementedError( )
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform the input feature matrix by imputing missing values.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. Imputers do not use this value.

			Returns:
			--------
			np.ndarray: Imputed feature matrix.

		"""
		raise NotImplementedError( )
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the imputer to the input feature matrix and return the transformed
			result in a single step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. Imputers do not use this value.

			Returns:
			--------
			np.ndarray: Imputed feature matrix.

		"""
		raise NotImplementedError( )
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Attempt to invert a previously transformed feature matrix when the
			underlying sklearn imputer supports inverse transformation.

			Parameters:
			-----------
			X ( np.ndarray ): Transformed feature matrix.

			Returns:
			--------
			np.ndarray: Reconstructed feature matrix.

		"""
		raise NotImplementedError( )


class MeanImputer( Imputer ):
	"""

		Purpose:
		---------
		Impute missing values by replacing them with the arithmetic mean of each
		feature column.

	"""
	strategy: Optional[ str ]
	add_indicator: bool
	imputer: im.SimpleImputer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, strategy: str = 'mean', add_indicator: bool = False ) -> None:
		"""

			Purpose:
			---------
			Initialize a mean-based simple imputer wrapper.

			Parameters:
			-----------
			strategy ( str ): Imputation strategy. This wrapper is intended for
				mean imputation and defaults to 'mean'.
			add_indicator ( bool ): Indicates whether missing-value indicator
				columns should be appended during transformation.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.strategy = strategy
		self.add_indicator = add_indicator
		self.imputer = im.SimpleImputer(
			strategy=self.strategy,
			add_indicator=self.add_indicator )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the available public members for the wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: List of public member names.

		"""
		return [
				'imputer',
				'transformed_data',
				'strategy',
				'add_indicator',
				'train',
				'transform',
				'train_transform',
				'inverse_transform'
		]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> MeanImputer | None:
		"""

			Purpose:
			---------
			Fit the mean imputer to the input feature matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. The underlying imputer does not use this value.

			Returns:
			--------
			MeanImputer | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = (
					'train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> MeanImputer | None')
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Impute missing values in the input feature matrix using the learned
			column means.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. The underlying imputer does not use this value.

			Returns:
			--------
			np.ndarray: Imputed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = (
					'transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> np.ndarray')
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the mean imputer and transform the input feature matrix in a
			single step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. The underlying imputer does not use this value.

			Returns:
			--------
			np.ndarray: Imputed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = (
					'train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> np.ndarray')
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Attempt to invert a transformed matrix back toward its pre-imputed
			form when missing-value indicators were included during fitting.

			Parameters:
			-----------
			X ( np.ndarray ): Transformed feature matrix.

			Returns:
			--------
			np.ndarray: Reconstructed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			if not self.add_indicator:
				raise ValueError(
					'inverse_transform requires add_indicator=True in the '
					'underlying sklearn.impute.SimpleImputer.' )
			return self.imputer.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			

class NearestImputer( Imputer ):
	"""

		Purpose:
		---------
		The NearestNeighborImputer class provides imputation for filling in missing values using
		the k-Nearest Neighbors approach. By default, a euclidean distance metric that supports
		missing values, nan_euclidean_distances, is used to find the nearest neighbors.
		Each missing feature is imputed using values from n_neighbors nearest neighbors that have
		a value for the feature. The feature of the neighbors are averaged uniformly or weighted
		by distance to each neighbor.

		If a sample has more than one feature missing, then the neighbors for that sample can be
		different depending on the particular feature being imputed. When the number of available
		neighbors is less than n_neighbors and there are no defined distances to the training set,
		the training set average for that feature is used during imputation. If there is at least
		one neighbor with a defined distance, the weighted or unweighted average of the
		remaining neighbors will be used during imputation. If a feature is always missing in
		training, it is removed during transform.

	"""
	n_neighbors: Optional[ int ]
	imputer: im.KNNImputer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, neighbors: int = 5 ) -> None:
		"""

			Purpose:
			---------
			Initialize a K-nearest neighbors imputer wrapper.

			Parameters:
			-----------
			neighbors ( int ): Number of neighboring samples to use when imputing
				missing values.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.n_neighbors = neighbors
		self.imputer = im.KNNImputer( n_neighbors=self.n_neighbors )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the available public members for the wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: List of public member names.

		"""
		return [
				'imputer',
				'transformed_data',
				'n_neighbors',
				'train',
				'transform',
				'train_transform'
		]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> NearestImputer | None:
		"""

			Purpose:
			---------
			Fit the nearest-neighbor imputer to the input feature matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. The underlying imputer does not use this value.

			Returns:
			--------
			NearestImputer | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = (
					'train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> NearestImputer | None')
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Impute missing values in the input feature matrix by averaging values
			from neighboring samples.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. The underlying imputer does not use this value.

			Returns:
			--------
			np.ndarray: Imputed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = (
					'transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> np.ndarray')
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the nearest-neighbor imputer and transform the input feature matrix
			in a single step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. The underlying imputer does not use this value.

			Returns:
			--------
			np.ndarray: Imputed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = (
					'train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> np.ndarray')
			raise exception
			

class IterativeImputer( Imputer ):
	"""
		Purpose:
		--------
		The IterativeImputer class, which models each feature with missing values as a function of
		other features, and uses that estimate for imputation. It does so in an iterated
		round-robin fashion: at each step, a feature column is designated as output y and the
		other feature columns are treated as inputs X. A regressor is fit on (X, y) for known y.

		Then, the regressor is used to predict the missing values of y. This is done for each
		feature in an iterative fashion, and then is repeated for max_iter imputation rounds.
		The results of the final imputation round are returned.

	"""
	imputer: im.IterativeImputer
	max_iter: Optional[ int ]
	random_state: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, max_iter: int=10, random_state: int=0 ) -> None:
		"""

			Purpose:
			---------
			Initialize an iterative imputer wrapper.

			Parameters:
			-----------
			max_iter ( int ): Maximum number of imputation rounds to perform.
			random_state ( int ): Random seed used by the underlying sklearn
				iterative imputer.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.max_iter = max_iter
		self.random_state = random_state
		self.imputer = im.IterativeImputer(
			max_iter=self.max_iter,
			random_state=self.random_state )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the available public members for the wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: List of public member names.

		"""
		return [
				'imputer',
				'transformed_data',
				'max_iter',
				'random_state',
				'train',
				'transform',
				'train_transform'
		]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> IterativeImputer | None:
		"""

			Purpose:
			---------
			Fit the iterative imputer to the input feature matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				consistency. The underlying imputer does not use this value.

			Returns:
			--------
			IterativeImputer | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = (
					'train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> IterativeImputer | None')
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Impute missing values in the input feature matrix using the fitted
			iterative imputation model.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				consistency. The underlying imputer does not use this value.

			Returns:
			--------
			np.ndarray: Imputed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = (
					'transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> np.ndarray')
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the iterative imputer and transform the input feature matrix in a
			single step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				consistency. The underlying imputer does not use this value.

			Returns:
			--------
			np.ndarray: Imputed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = (
					'train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> np.ndarray')
			raise exception


class SimpleImputer( Imputer ):
	"""

		Purpose:
		---------
		Impute missing values using sklearn's SimpleImputer wrapper with support
		for common strategy-based replacement operations.

	"""
	imputer: im.SimpleImputer
	transformed_data: Optional[ np.ndarray ]
	strategy: Optional[ str ]
	fill_value: Optional[ object ]
	add_indicator: bool
	keep_empty_features: bool
	
	def __init__( self, strategy: str='mean', fill_value: object=0.0,
			add_indicator: bool=False, keep_empty_features: bool=False ) -> None:
		"""

			Purpose:
			---------
			Initialize the simple imputer wrapper.

			Parameters:
			-----------
			strategy ( str ): Imputation strategy. Supported values are 'mean',
				'median', 'most_frequent', and 'constant'.
			fill_value ( object ): Replacement value used when strategy is
				'constant'.
			add_indicator ( bool ): Indicates whether missing-value indicator
				columns should be appended during transformation.
			keep_empty_features ( bool ): Indicates whether features that are
				entirely missing at fit time should be preserved during transform.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.strategy = strategy
		self.fill_value = fill_value
		self.add_indicator = add_indicator
		self.keep_empty_features = keep_empty_features
		self.imputer = im.SimpleImputer(
			strategy=self.strategy,
			fill_value=self.fill_value,
			add_indicator=self.add_indicator,
			keep_empty_features=self.keep_empty_features )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the available public members for the wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: List of public member names.

		"""
		return [
				'imputer',
				'transformed_data',
				'strategy',
				'fill_value',
				'add_indicator',
				'keep_empty_features',
				'train',
				'transform',
				'train_transform',
				'inverse_transform'
		]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> SimpleImputer | None:
		"""

			Purpose:
			---------
			Fit the simple imputer to the input feature matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. The underlying imputer does not use this value.

			Returns:
			--------
			SimpleImputer | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SimpleImputer'
			exception.method = (
					'train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> SimpleImputer | None')
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform the input feature matrix by imputing missing values using
			the fitted simple imputer.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. The underlying imputer does not use this value.

			Returns:
			--------
			np.ndarray: Imputed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SimpleImputer'
			exception.method = (
					'transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> np.ndarray')
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the simple imputer and transform the input feature matrix in a
			single step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array accepted for API
				compatibility. The underlying imputer does not use this value.

			Returns:
			--------
			np.ndarray: Imputed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SimpleImputer'
			exception.method = (
					'train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) '
					'-> np.ndarray')
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Attempt to invert the transformed matrix back toward its pre-imputed
			form when missing-value indicators were included during fitting.

			Parameters:
			-----------
			X ( np.ndarray ): Transformed feature matrix.

			Returns:
			--------
			np.ndarray: Reconstructed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			if not self.add_indicator:
				raise ValueError(
					'inverse_transform requires add_indicator=True in the '
					'underlying sklearn.impute.SimpleImputer.' )
			return self.imputer.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SimpleImputer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
