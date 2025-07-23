'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                Scalers.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="Processors.py" company="Terry D. Eppler">

     Mathy Scalers

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
	Scalers.py
</summary>
******************************************************************************************
'''
from Data import Metric
from Booger import Error, ErrorDialog
import numpy as np
from typing import Optional
import sklearn.preprocessing as skp


class StandardScaler( Metric ):
	"""

		Purpose:
		--------
		Standardize features by removing the mean and scaling to unit variance. The standard score
		of a sample x is calculated as: z = (x - u) / s where u is the mean of the training
		samples or zero if with_mean=False, and s is the standard deviation of the training
		samples or one if with_std=False.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.standard_scaler = skp.StandardScaler( )

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object:
		"""


			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Input df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.standard_scaler.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StandardScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms the df using the fitted StandardScaler.

			Parameters:
			-----------
			X (np.ndarray): Input df.

			Returns:
			-----------
			np.ndarray: Scaled df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.standard_scaler.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class MinMaxScaler( Metric ):
	"""

		Purpose:
		---------
		Transforms features by scaling each feature to a given range. This estimator scales and
		translates each feature individually such that it is in the given range on the
		training set, e.g. between zero and one.

	"""


	def __init__( self ) -> None:
		super( ).__init__( )
		self.minmax_scaler = skp.MinMaxScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""

			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Input df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.minmax_scaler.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms the df using the fitted MinMaxScaler.

			Parameters:
			-----------
			X (np.ndarray): Input df.

			Returns:
			-----------
			np.ndarray: Scaled df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.minmax_scaler.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class RobustScaler( Metric ):
	"""

		Purpose:
		--------
		This Scaler wraps the RobustScaler and removes the median and scales
		the data according to the quantile range (defaults to IQR: Interquartile Range).
		The IQR is the range between the 1st quartile (25th quantile)
		and the 3rd quartile (75th quantile).

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.robust_scaler = skp.RobustScaler( )

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""


			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Input df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.robust_scaler.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RobustScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )

	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms the df using the fitted RobustScaler.

			Parameters:
			-----------
			X (np.ndarray): Input df.

			Returns:
			-----------
			np.ndarray: Scaled df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.robust_scaler.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RobustScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class NormalScaler( Metric ):
	"""

		Purpose:
		---------
		Normalize samples individually to unit norm. Each sample (i.e. each row of the data matrix)
		with at least one non zero component is rescaled independently of other samples
		so that its norm (l1 or l2) equals one.

	"""

	def __init__( self, norm: str = 'l2' ) -> None:
		super( ).__init__( )
		self.normal_scaler = skp.Normalizer( norm = norm )

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""


			Purpose:
			---------
			Fits the normalizer (no-op for Normalizer).

			Parameters:
			-----------
			X (np.ndarray): Input df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.normal_scaler.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Normalizer'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )

	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			---------
			Applies normalization to each sample.

			Parameters:
			-----------
			X (np.ndarray): Input df.

			Returns:
			-----------
			np.ndarray: Normalized df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.normal_scaler.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Normalizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
