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
import sklearn.preprocessing as pp
from boogr import Error

def throw_if( name: str, value: object ):
	if value is None:
		raise Exception( f'Argument "{name}" cannot be empty!' )

class Scaler( ):
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
		
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]  ) -> np.ndarray:
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

class StandardScaler( Scaler ):
	"""

		Purpose:
		--------
		Standardize feature_names by removing the mean and scaling to unit variance. The standard score
		of a sample x is calculated as: z = (x - u) / s where u is the mean of the training
		samples or zero if with_mean=False, and s is the standard deviation of the training
		samples or one if with_std=False.

	"""
	model: pp.StandardScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.model = pp.StandardScaler( )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'scaler',
		  'transformed_data',
		  'train',
		  'transform',
		  'train_transform',
		  'inverse_transform' ]
	
	def train( self, X: np.ndarray ) -> StandardScaler | None:
		"""


			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ). IGNORED

			Returns:
			--------
			StandardScaler | None

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> '
			                    'Pipeline')
			raise exception
			
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms the df using the fitted StandardScaler.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ). IGNORED

			Returns:
			-----------
			np.ndarray: Scaled df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fits & Transforms the df using the fitted MinMaxScaler.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ). IGNORED

			Returns:
			-----------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'fit_transform( self, X: np.ndarray, y:np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms into a standardized np.ndarray

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )

			Returns:
			-----------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			_original = self.model.inverse_transform( X )
			return _original
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			

class MinMaxScaler( Scaler ):
	"""

		Purpose:
		---------
		This estimator scales and translates each feature individually such that it is in the
		given range on the training set, e.g. between zero and one. This transformation is often
		used as an alternative to zero mean, unit variance scaling.

		MinMaxScaler doesn’t reduce the effect of outliers, but it linearly scales them down
		into a fixed range, where the largest occurring stores point corresponds to the maximum
		value and the smallest one corresponds to the minimum value

	"""
	model: pp.MinMaxScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.model = pp.MinMaxScaler( )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'scaler',
		  'transformed_data',
		  'train',
		  'transform',
		  'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: np.ndarray=None ) -> MinMaxScaler | None:
		"""

			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ). IGNORED

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = ('fit( self, X: np.ndarray, y: np.ndarray=None ) -> '
			                    'Pipeline')
			raise exception
			
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms the df using the fitted MinMaxScaler.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ). IGNORED

			Returns:
			-----------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def train_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fits & Transforms the df using the fitted MinMaxScaler.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ). IGNORED

			Returns:
			-----------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms back to original stores

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )

			Returns:
			-----------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			

class RobustScaler( Scaler ):
	"""

		Purpose:
		--------
		This Scaler removes the median and scales the stores according to the quantile range
		(defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile
		(25th quantile) and the 3rd quartile (75th quantile).

		Centering and scaling happen independently on each feature by computing the relevant
		statistics on the samples in the training set. Median and interquartile range are
		then stored to be used on later stores using the transform method.

		Standardization of a dataset is a common preprocessing for many machine learning estimators.
		Typically this is done by removing the mean and scaling to unit variance.
		However, outliers can often influence the sample mean / variance in a negative way.
		In such cases, using the median and the interquartile range often give better results.

	"""
	model: pp.RobustScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.model = pp.RobustScaler( )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'scaler',
		  'transformed_data',
		  'train',
		  'transform',
		  'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: np.ndarray=None ) -> RobustScaler | None:
		"""


			Purpose:
			---------
			Fits the standard_scaler to the df.

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
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = 'fit( self, X: np.ndarray, y: np.ndarray=None ) -> Pipeline'
			raise exception
			
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms the df using the fitted RobustScaler.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Scaled df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms into robust stores.

			:param X: List of text.
			:type X: list[str]
			:return: Standardized stores.
			:rtype: np.ndarray
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.inverse_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			

class NormalScaler( Scaler ):
	"""

		Purpose:
		---------
		Normalize samples individually to unit norm. Each sample (i.e. each row of the stores matrix)
		with at least one non zero component is rescaled independently of other samples so that
		its norm (l1, l2 or inf) equals one.

		This transformer is able to work both with dense numpy arrays and scipy.sparse matrix
		(use CSR format if you want to avoid the burden of a copy / conversion). Scaling inputs to
		unit norms is a common operation for text classification or clustering for instance.
		For instance the dot product of two l2-normalized TF-IDF vectors is the cosine similarity
		of the vectors and is the base similarity metric for the Vector Space Model.

	"""
	norm: Optional[ str ]
	model: pp.Normalizer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, reg: str='l2' ) -> None:
		super( ).__init__( )
		self.norm = reg
		self.model = pp.Normalizer( norm=self.norm )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'scaler',
		  'transformed_data',
		  'norm',
		  'train',
		  'transform',
		  'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: np.ndarray=None ) -> NormalScaler | None:
		"""


			Purpose:
			---------
			Fits the normalizer (no-op for Normalizer).

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
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = ('fit( self, X: np.ndarray, y: np.ndarray=None ) -> '
			                    'Pipeline')
			raise exception
			
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""


			Purpose:
			---------
			Applies normalization to each sample.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): IGNORED. Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Normalized df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def train_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""


			Purpose:
			---------
			Applies normalization to each sample.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): IGNORED. Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Normalized df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			

