'''
******************************************************************************************
  Assembly:                Name
  Filename:                features.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="features.py" company="Terry D. Eppler">

     features.py
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
features.py
</summary>
******************************************************************************************
'''
from typing import Optional, Tuple

import numpy as np
import sklearn.cross_decomposition as sd
import sklearn.decomposition as sd
import sklearn.feature_selection as sf
from sklearn.feature_selection import chi2

from boogr import Error, ErrorDialog

def throw_if( name: str, value: object ):
    if value is None:
        raise ValueError( f'Argument "{name}" cannot be empty!' )


class VarianceThreshold( ):
	"""

		Purpose:
		---------
		VarianceThreshold is a simple baseline approach to feature selection. It removes all
		feature_names whose variance doesn’t meet some threshold. By default, it removes all
		zero-variance feature_names, i.e. feature_names that have the same value in all samples.

	"""
	model: sf.VarianceThreshold
	transformed_data: Optional[ np.ndarray ]
	threshold: Optional[ float ]
	
	def __init__( self, thresh: float = 0.0 ) -> None:
		"""

			Purpose:
			---------
			Initialize VarianceThreshold.

			:param threshold: Features with variance below this are removed.
			:type threshold: float

		"""
		self.threshold = thresh
		self.model = sf.VarianceThreshold( threshold=self.threshold )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members

		'''
		return [ 'threshold',
		         'model',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def train( self, X: np.ndarray ) -> sf.VarianceThreshold | None:
		"""

			Purpose:
			---------
			Fit the variance threshold model.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'fit( self, X: np.ndarray ) -> object | None'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Apply variance threshold selection.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit and transform the stores using variance thresholding.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).


			Return:
			-------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )

class CCA( ):
	"""

		Canonical Correlation Analysis (CCA) extracts the ‘directions of covariance’,
		i.e. the components of each datasets that explain the most shared variance
		between both datasets.

	"""
	analysis: Optional[ sd.CCA ]
	n_components: Optional[ int ]
	scale: Optional[ bool ]
	max_iter: Optional[ int ]
	transformed_data: Optional[ Tuple[ np.ndarray, np.ndarray ] ]
	
	def __init__( self, num: int=2, scale: bool=True, size: int=500 ) -> None:
		"""

			Purpose:
			---------
			Initialize CCA.

			Parameters:
			-----------
			num (int): Number of components to extract.
			scale (bool): Whether to scale the correlation analysis.
			max (int): The maximum number of components to extract.

		"""
		self.scale = scale
		self.n_components = num
		self.max_iter = size
		self.analysis = sd.CCA( n_components=self.n_components, scale=self.scale, max_iter=self.max_iter )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			Returns a list of strings representing class members.

		'''
		return [ 'analysis',
		         'n_components',
		         'max_iter',
		         'analysis',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> sd.CCA:
		"""

			Purpose:
			---------
			Fit the CCA model to X and Y.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).

			Return:
			-------
			CCA or None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.analysis.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CorrelationAnalysis'
			exception.method = 'train( self, X: np.ndarray, Y: np.ndarray ) -> object'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
		"""

			Purpose:
			---------
			Apply the CCA transformation.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).

			Return:
			-------
			(np.ndarray, np.ndarray): Transformed X and Y.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.transformed_data = self.analysis.transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CAA'
			exception.method = 'transform( self, X: np.ndarray, Y: np.ndarray ) -> tuple'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
		"""

			Purpose:
			---------
			Fit and transform with CCA.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).


		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.transformed_data = self.analysis.fit( X, y ).transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CAA'
			exception.method = 'train_transform( self, X: np.ndarray, Y: np.ndarray ) -> tuple'
			error = ErrorDialog( exception )
			error.show( )

class PCA( ):
	"""

		Purpose:
		---------
		Principal Component Analysis (PCA). Linear dimensionality reduction using
		Singular Value Decomposition of the stores to project it to a lower dimensional space.
		The input stores is centered but not scaled for each feature before applying the SVD.
		It uses the LAPACK implementation of the full SVD or a randomized truncated SVD
		by the method of Halko et al. 2009, depending on the shape of the input stores and
		the number of components to extract.

	"""
	model: sd.PCA
	svd_solver: Optional[ str ]
	n_components: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, num: int=2, solver: str='auto' ) -> None:
		"""

			Purpose:
			---------
			Initialize PCA.

			:param num: Number of components.
			:type num: int

			:param solver: The solver used by the model
			:type solver: str

		"""
		self.n_components = num
		self.svd_solver = solver
		self.model = sd.PCA( n_components=self.n_components, svd_solver=self.svd_solver )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members.

		'''
		return [ 'component_analysis',
		         'svd_solver',
		         'n_components',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def train( self, X: np.ndarray ) -> sd.PCA:
		"""

			Purpose:
			---------
			Fit PCA to the input stores.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).\

			Return:
			-------
			sd.PCA

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'def fit( self, X: np.ndarray ) -> ComponentAnalysis'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Apply PCA transformation.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).

			Return:
			--------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit PCA and transform input stores.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).

			Return:
			-------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class SelectBest( ):
	"""

		Purpose:
		---------


	"""
	model: sf.SelectKBest
	k_best: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	threshold: Optional[ float ]
	
	def __init__( self, k: int = 3 ) -> None:
		"""

			Purpose:
			---------
			Initialize SelectBest.

			:param threshold: Features with variance below this are removed.
			:type threshold: float

		"""
		self.k_best = k
		self.model = sf.SelectKBest( score_func=chi2, k=self.k_best )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members

		'''
		return [ 'k_best',
		         'model',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def train( self, X: np.ndarray ) -> sf.VarianceThreshold | None:
		"""

			Purpose:
			---------
			Fit the model.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'fit( self, X: np.ndarray ) -> object | None'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Apply selection.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit and transform the data.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).


			Return:
			-------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class SelectPercent( ):
	"""

		Purpose:
		---------


	"""
	model: sf.SelectPercentile
	transformed_data: Optional[ np.ndarray ]
	threshold: Optional[ float ]
	
	def __init__( self, percent: int = 10 ) -> None:
		"""

			Purpose:
			---------
			Initialize SelectBest.

			:param threshold: Features with variance below this are removed.
			:type threshold: float

		"""
		self.percent = percent
		self.model = sf.SelectPercentile( percentile=self.percent )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members

		'''
		return [ 'k',
		         'model',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def train( self, X: np.ndarray ) -> sf.SelectPercentile | None:
		"""

			Purpose:
			---------
			Fit the model.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'fit( self, X: np.ndarray ) -> object | None'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Apply selection.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit and transform the data.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).


			Return:
			-------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
