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
		Provide a common base interface for scaler and normalization wrappers built
		over sklearn.preprocessing transformers.

	"""
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize shared scaler state.

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
			Fit the underlying preprocessing model to the input data.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array. Present for API
				consistency; ignored by most scaler implementations.

			Returns:
			--------
			object | None: The fitted wrapper instance when implemented by a
				concrete subclass.

		"""
		try:
			raise NotImplementedError(
				'Concrete scaler wrappers must implement train( self, X, y=None ).'
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Scaler'
			exception.method = (
					'train( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> object | None'
			)
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform input features using a previously fitted preprocessing model.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array. Present for API
				consistency; ignored by most scaler implementations.

			Returns:
			--------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			raise NotImplementedError(
				'Concrete scaler wrappers must implement transform( self, X, y=None ).'
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Scaler'
			exception.method = (
					'transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the preprocessing model to the input data and return the transformed
			result in one step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array. Present for API
				consistency; ignored by most scaler implementations.

			Returns:
			--------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			raise NotImplementedError(
				'Concrete scaler wrappers must implement train_transform( self, X, y=None ).'
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Scaler'
			exception.method = (
					'train_transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform scaled data back to its original representation when the
			underlying preprocessing model supports inverse transformation.

			Parameters:
			-----------
			X ( np.ndarray ): Transformed feature matrix of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Original-scale feature matrix.

		"""
		try:
			raise NotImplementedError(
				'Concrete scaler wrappers must implement inverse_transform( self, X ).'
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Scaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception


class StandardScaler( Scaler ):
	"""

		Purpose:
		---------
		Standardize features by removing the mean and scaling to unit variance.
		The standard score of a sample x is calculated as: z = ( x - u ) / s,
		where u is the mean of the training samples or zero if with_mean=False,
		and s is the standard deviation of the training samples or one if
		with_std=False.

	"""
	model: pp.StandardScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the StandardScaler wrapper and its backing sklearn model.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.model = pp.StandardScaler( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the list of class members exposed by this wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: Member names exposed by the wrapper.

		"""
		return [
				'scaler',
				'transformed_data',
				'train',
				'transform',
				'train_transform',
				'inverse_transform'
		]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> StandardScaler | None:
		"""

			Purpose:
			---------
			Fit the StandardScaler to the input feature matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by StandardScaler.

			Returns:
			--------
			StandardScaler | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = (
					'train( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> StandardScaler | None'
			)
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform the input feature matrix using the fitted StandardScaler.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by StandardScaler.

			Returns:
			--------
			np.ndarray: Standardized feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = (
					'transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the StandardScaler to the input feature matrix and return the
			standardized result in one step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by StandardScaler.

			Returns:
			--------
			np.ndarray: Standardized feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = (
					'train_transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Scale standardized data back to the original representation.

			Parameters:
			-----------
			X ( np.ndarray ): Standardized feature matrix of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Feature matrix restored to the original scale.

		"""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
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
		Transform features by scaling each feature to a given range.
		This estimator scales and translates each feature individually such
		that it is in the given range on the training set, e.g. between zero
		and one. This transformation is often used as an alternative to zero
		mean, unit variance scaling.

		MinMaxScaler does not reduce the effect of outliers, but it linearly
		scales them down into a fixed range, where the largest occurring data
		point corresponds to the maximum value and the smallest one
		corresponds to the minimum value.

	"""
	model: pp.MinMaxScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the MinMaxScaler wrapper and its backing sklearn model.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.model = pp.MinMaxScaler( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the list of class members exposed by this wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: Member names exposed by the wrapper.

		"""
		return [
				'scaler',
				'transformed_data',
				'train',
				'transform',
				'train_transform',
				'inverse_transform'
		]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> MinMaxScaler | None:
		"""

			Purpose:
			---------
			Fit the MinMaxScaler to the input feature matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by MinMaxScaler.

			Returns:
			--------
			MinMaxScaler | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = (
					'train( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> MinMaxScaler | None'
			)
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform the input feature matrix using the fitted MinMaxScaler.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by MinMaxScaler.

			Returns:
			--------
			np.ndarray: Min-max scaled feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = (
					'transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the MinMaxScaler to the input feature matrix and return the
			scaled result in one step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by MinMaxScaler.

			Returns:
			--------
			np.ndarray: Min-max scaled feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = (
					'train_transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Undo the scaling of the feature matrix according to the fitted
			feature range.

			Parameters:
			-----------
			X ( np.ndarray ): Scaled feature matrix of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Feature matrix restored to the original scale.

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
		---------
		Remove the median and scale features according to the quantile range.

		By default, the quantile range is the interquartile range ( IQR ), which
		is the range between the 1st quartile ( 25th quantile ) and the 3rd
		quartile ( 75th quantile ).

		Centering and scaling happen independently on each feature by computing
		the relevant statistics on the samples in the training set. The median
		and interquartile range are then stored for use on later data during
		transformation.

		Robust scaling is useful when outliers would otherwise negatively affect
		mean- and variance-based scaling methods.

	"""
	model: pp.RobustScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the RobustScaler wrapper and its backing sklearn model.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.model = pp.RobustScaler( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the list of class members exposed by this wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: Member names exposed by the wrapper.

		"""
		return [
				'scaler',
				'transformed_data',
				'train',
				'transform',
				'train_transform',
				'inverse_transform'
		]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> RobustScaler | None:
		"""

			Purpose:
			---------
			Fit the RobustScaler to the input feature matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by RobustScaler.

			Returns:
			--------
			RobustScaler | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = (
					'train( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> RobustScaler | None'
			)
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform the input feature matrix using the fitted RobustScaler.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by RobustScaler.

			Returns:
			--------
			np.ndarray: Robust-scaled feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = (
					'transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the RobustScaler to the input feature matrix and return the
			robust-scaled result in one step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by RobustScaler.

			Returns:
			--------
			np.ndarray: Robust-scaled feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = (
					'train_transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Undo the robust scaling and restore the original feature scale.

			Parameters:
			-----------
			X ( np.ndarray ): Robust-scaled feature matrix of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Feature matrix restored to the original scale.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.inverse_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception


class NormalScaler( Scaler ):
	"""

		Purpose:
		---------
		Wrap sklearn.preprocessing.Normalizer while preserving the module's
		existing public class name.

		Normalize samples individually to unit norm. Each sample ( that is,
		each row of the feature matrix ) with at least one non-zero component
		is rescaled independently of the other samples so that its norm
		( l1, l2, or max ) equals one.

		This transformer can work with dense NumPy arrays and sparse matrices.
		Scaling inputs to unit norms is a common preprocessing step for text
		classification and clustering. For example, the dot product of two
		l2-normalized TF-IDF vectors is the cosine similarity between them.

	"""
	model: pp.Normalizer
	transformed_data: Optional[ np.ndarray ]
	norm: str
	
	def __init__( self, norm: str = 'l2' ) -> None:
		"""

			Purpose:
			---------
			Initialize the NormalScaler wrapper and its backing sklearn
			Normalizer model.

			Parameters:
			-----------
			norm ( str ): Norm used to normalize each sample. Supported values
				are typically 'l1', 'l2', and 'max'.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.norm = norm
		self.model = pp.Normalizer( norm=self.norm )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the list of class members exposed by this wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: Member names exposed by the wrapper.

		"""
		return [
				'scaler',
				'transformed_data',
				'norm',
				'train',
				'transform',
				'train_transform',
				'inverse_transform'
		]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> NormalScaler | None:
		"""

			Purpose:
			---------
			Fit the backing Normalizer model.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by Normalizer.

			Returns:
			--------
			NormalScaler | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = (
					'train( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> NormalScaler | None'
			)
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Normalize each sample in the feature matrix to unit norm.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by Normalizer.

			Returns:
			--------
			np.ndarray: Normalized feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = (
					'transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the Normalizer model and return the normalized feature matrix
			in one step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by Normalizer.

			Returns:
			--------
			np.ndarray: Normalized feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = (
					'train_transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Reject inverse transformation because sklearn.preprocessing.
			Normalizer does not support inverse_transform.

			Parameters:
			-----------
			X ( np.ndarray ): Normalized feature matrix of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: This method does not return successfully.

		"""
		try:
			throw_if( 'X', X )
			raise NotImplementedError(
				'Normalizer does not support inverse_transform( X ).'
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception


class MaxAbsScaler( Scaler ):
	"""

		Purpose:
		---------
		Scale each feature by its maximum absolute value.

		This estimator scales and transforms each feature individually such
		that the maximal absolute value of each feature in the training set
		will be 1.0. It does not shift or center the data, and therefore
		does not destroy sparsity.

		This scaler can also be applied to sparse CSR or CSC matrices.
		MaxAbsScaler does not reduce the effect of outliers; it only linearly
		scales them down.

	"""
	model: pp.MaxAbsScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the MaxAbsScaler wrapper and its backing sklearn model.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.model = pp.MaxAbsScaler( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the list of class members exposed by this wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			list[ str ]: Member names exposed by the wrapper.

		"""
		return [
				'scaler',
				'transformed_data',
				'train',
				'transform',
				'train_transform',
				'inverse_transform'
		]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> MaxAbsScaler | None:
		"""

			Purpose:
			---------
			Fit the MaxAbsScaler to the input feature matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by MaxAbsScaler.

			Returns:
			--------
			MaxAbsScaler | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MaxAbsScaler'
			exception.method = (
					'train( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> MaxAbsScaler | None'
			)
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform the input feature matrix using the fitted MaxAbsScaler.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by MaxAbsScaler.

			Returns:
			--------
			np.ndarray: Max-absolute scaled feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MaxAbsScaler'
			exception.method = (
					'transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the MaxAbsScaler to the input feature matrix and return the
			scaled result in one step.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target vector of shape
				( n_samples, ). Ignored by MaxAbsScaler.

			Returns:
			--------
			np.ndarray: Max-absolute scaled feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MaxAbsScaler'
			exception.method = (
					'train_transform( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> np.ndarray'
			)
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Scale max-absolute transformed data back to the original
			representation.

			Parameters:
			-----------
			X ( np.ndarray ): Max-absolute scaled feature matrix of shape
				( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Feature matrix restored to the original scale.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.inverse_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MaxAbsScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception