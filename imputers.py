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

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Imputer( ):
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

class MeanImputer( Imputer ):
	"""

		Purpose:
		-----------
		Fills missing target_names using the average.

	"""
	strategy: Optional[ str ]
	imputer: im.SimpleImputer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, strategy: str='mean' ) -> None:
		super( ).__init__( )
		self.strategy = strategy
		self.imputer = im.SimpleImputer( strategy=self.strategy )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'imputer',
		  'transformed_data',
		  'strategy',
		  'train',
		  'transform',
		  'train_transform',
		  'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: np.ndarray=None ) -> MeanImputer | None:
		"""


			Purpose:
			---------
			Fits the simple_imputer to the df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			--------
			Pipeline

		"""
		try:
			throw_if( 'X', X )
			self.imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'fit( self, X: np.ndarray, y: np.ndarray=None ) -> MeanImputer'
			raise exception
			
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""


			Purpose:
			---------
			Transforms the text df by filling in missing target_names.

			Parameters:
			-----------
			X (np.ndarray): Input df with missing target_names.

			Returns:
			-----------
			np.ndarray: Imputed df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def train_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit the iterative imputer and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.imputer.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform text to TF-IDF vectors.

			:param X: np.ndarray
		"""
		try:
			throw_if( 'X', X )
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
	
	def __init__( self, neighbors: int=5 ) -> None:
		super( ).__init__( )
		self.n_neighbors = neighbors
		self.imputer = im.KNNImputer( n_neighbors=self.n_neighbors )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'imputer',
		  'transformed_data',
		  'n_neighbors',
		  'train',
		  'transform',
		  'train_transform',
		  'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> NearestImputer | None:
		"""

			Purpose:
			________
			Fits the simple_imputer to the df.

			Parameters:
			___________
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborImputer'
			exception.method = 'fit( self, X: np.ndarray ) -> NearestNeighborImputer'
			raise exception
			
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			_________

			Transforms the text df by imputing missing target_names.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Imputed df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborImputer'
			exception.method = 'transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray'
			raise exception
			
	
	def train_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the iterative imputer and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.imputer.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray'
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
		self.max_iter = max
		self.random_state = rando
		self.imputer = im.IterativeImputer( max_iter=self.max_iter, random_state=self.random_state )
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'imputer',
		  'transformed_data',
		  'max_iter',
		  'random_state',
		  'train',
		  'transform',
		  'train_transform', ]
	
	def train( self, X: np.ndarray, y: np.ndarray=None ) -> IterativeImputer | None:
		"""

			Purpose:
			--------
			Fit the iterative imputer to the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray=None ) -> IterativeImputer'
			raise exception
			
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform stores by iteratively imputing missing values.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = 'transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray'
			raise exception
			
	
	def train_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit the iterative imputer and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.imputer.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			

class SimpleImputer( Imputer ):
	"""

		Wrapper for sklearn's SimpleImputer.

	"""
	imputer: im.SimpleImputer
	transformed_data: Optional[ np.ndarray ]
	strategy: Optional[ str ]
	fill_value: Optional[ float ]
	
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
		self.strategy = strategy
		self.fill_value = fill_value
		self.imputer = im.SimpleImputer( strategy=self.strategy, fill_value=self.fill_value )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'imputer',
		  'transformed_data',
		  'stratgey',
		  'fill_value',
		  'train',
		  'transform',
		  'train_transform', ]
	
	def train( self, X: np.ndarray, y: np.ndarray=None ) -> SimpleImputer | None:
		"""

			Purpose:
			--------
			Fit the imputer to the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'train( self, X: np.ndarray ) -> SimpleImputer'
			raise exception
			
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Transform stores by imputing missing values.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.imputer.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def train_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit the imputer and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
