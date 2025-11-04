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
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import sklearn.cross_decomposition as sd
import sklearn.decomposition as sd
import sklearn.feature_selection as sf
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from itertools import combinations
from sklearn.model_selection import train_test_split as split
from classifications import Classifier, NearestNeighbor
from boogr import Error, ErrorDialog

def throw_if( name: str, value: object ):
    if value is None:
        raise ValueError( f'Argument "{name}" cannot be empty!' )

class Selector( ):
	'''
		
		Purpose:
		--------
		Base class for implementing feature selection functionality
		
		
	'''
	markers: Optional[ List[ str ] ]
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	
	def __init__( self ):
		self.markers = [ '.',
		                 'o',
		                 'v',
		                 '^',
		                 '<',
		                 '>',
		                 '1',
		                 '2',
		                 '3',
		                 '4',
		                 '8',
		                 's',
		                 'p',
		                 'P',
		                 '*',
		                 'h',
		                 'H',
		                 '+',
		                 'x',
		                 'X',
		                 'd',
		                 'D' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray ) -> (
			(np.ndarray, np.ndarray, np.ndarray, np.ndarray,) | None):
		'''

			Purpose:
			_______


			Parameters:
			__________


			Returns:
			________


		'''
		raise NotImplementedError
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
			Fit the linerar_model to the training df.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).

			Returns:
			--------
				None

		"""
		raise NotImplementedError
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Generate predictions from  the trained linerar_model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True target target_names.

			Returns:
			-----------
			np.ndarray: Predicted target_names or class target_names.

		"""
		raise NotImplementedError
	
	def transform( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
			Fit the linerar_model to the training df.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).

			Returns:
			--------
				None

		"""
		raise NotImplementedError
	
	def train_transform( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
			Fit the linerar_model to the training df.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).

			Returns:
			--------
				None

		"""
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			---------
			Compute the core metric (e.g., R²) of the model on test df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True target target_names.

			Returns:
			-----------
				float: Score value (e.g., R² for regressors).

		"""
		raise NotImplementedError

class VarianceThreshold( Selector ):
	"""

		Purpose:
		---------
		VarianceThreshold is a simple baseline approach to feature selection. It removes all
		feature_names whose variance doesn’t meet some threshold. By default, it removes all
		zero-variance feature_names, i.e. feature_names that have the same value in all samples.

	"""
	model: sf.VarianceThreshold
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	threshold: Optional[ float ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, thresh: float=0.0 ) -> None:
		"""

			Purpose:
			---------
			Initialize VarianceThreshold.

			:param threshold: Features with variance below this are removed.
			:type threshold: float

		"""
		super( ).__init__( )
		self.threshold = thresh
		self.model = sf.VarianceThreshold( threshold=self.threshold )
		self.prediction = None
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members

		'''
		return [ 'threshold',
		         'model',
		         'prediction',
		         'transformed_data',
		         'split_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: int=0.2, random: int=42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		'''

			Purpose:
			_______


			Parameters:
			---------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary class target_names. ( n_samples, ).
			size (int): The size of the testing data set
			random (int): A random seed.


			Returns:
			________
			tuple ( np.ndarray, np.ndarray, np.ndarray, np.ndarray )
			ex. ( X_train, X_test, y_train, y_test )


		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return ( X_train, X_test, y_train, y_test )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = ('split_data( self, X: ndarray, y: ndarray, size: int=0.2, '
			                    'random: int=42 ) -> ( ndarray, ndarray, ndarray, ndarray )')
			error = ErrorDialog( exception )
			error.show( )
	
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
	
	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels from input features using the regression model.


			Parameters:
			---------
			X (np.ndarray | pd.DataFrame):
			Input features.


			Returns:
			--------
			np.ndarray:

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'project( self, X: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Returns the coefficient of determination R^2 of the prediction.
			The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
			((y_true - y_pred) ** 2).sum() and v is the total sum of squares
			((y_true - y_true.mean()) ** 2).sum().

			The best possible score is 1.0 and it can be negative
			(because the model can be arbitrarily worse). A constant model that always predicts
			the expected value of y, disregarding the input features, would get a R^2 score of 0.0.


			Parameters:
			-----------
			X (np.ndarray ): Input features.
			y (np.ndarray ): True binary class labels.

			Returns:
			--------
			Dict[ str, float]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			X_training, X_testing, y_training, y_testing = self.split_data( X, y )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.accuracy = accuracy_score( y, y_pred )
			_metrics = \
				{
						'Training Score': self.training_score,
						'Testing Score': self.testing_score,
						'Accuracy Score': self.accuracy,
				}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'project( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
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
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class CCA( Selector ):
	"""

		Canonical Correlation Analysis (CCA) extracts the ‘directions of covariance’,
		i.e. the components of each datasets that explain the most shared variance
		between both datasets.

	"""
	model: Optional[ sd.CCA ]
	prediction: Optional[ np.ndarray ]
	n_components: Optional[ int ]
	scale: Optional[ bool ]
	max_iter: Optional[ int ]
	transformed_data: Optional[ Tuple[ np.ndarray, np.ndarray ] ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
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
		super( ).__init__( )
		self.scale = scale
		self.n_components = num
		self.max_iter = size
		self.model = sd.CCA( n_components=self.n_components, scale=self.scale, max_iter=self.max_iter )
		self.prediction = None
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			Returns a list of strings representing class members.

		'''
		return [ 'mdel',
		         'n_components',
		         'max_iter',
		         'prediction',
		         'transformed_data',
		         'split_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: int=0.2, random: int=42 ) -> ( np.ndarray, np.ndarray, np.ndarray, np.ndarray ):
		'''

			Purpose:
			_______


			Parameters:
			---------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary class target_names. ( n_samples, ).
			size (int): The size of the testing data set
			random (int): A random seed.


			Returns:
			________
			tuple ( np.ndarray, np.ndarray, np.ndarray, np.ndarray )
			ex. ( X_train, X_test, y_train, y_test )


		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return ( X_train, X_test, y_train, y_test )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = ('split_data( self, X: ndarray, y: ndarray, size: int=0.2, '
			                    'random: int=42 ) -> ( ndarray, ndarray, ndarray, ndarray )')
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels from input features using the regression model.


			Parameters:
			---------
			X (np.ndarray | pd.DataFrame):
			Input features.


			Returns:
			--------
			np.ndarray:

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'project( self, X: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Returns the coefficient of determination R^2 of the prediction.
			The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
			((y_true - y_pred) ** 2).sum() and v is the total sum of squares
			((y_true - y_true.mean()) ** 2).sum().
			
			The best possible score is 1.0 and it can be negative
			(because the model can be arbitrarily worse). A constant model that always predicts
			the expected value of y, disregarding the input features, would get a R^2 score of 0.0.


			Parameters:
			-----------
			X (np.ndarray ): Input features.
			y (np.ndarray ): True binary class labels.

			Returns:
			--------
			Dict[ str, float]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			X_training, X_testing, y_training, y_testing = self.split_data( X, y )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.accuracy = accuracy_score( y, y_pred )
			_metrics = \
			{
				'Training Score': self.training_score,
				'Testing Score': self.testing_score,
				'Accuracy Score': self.accuracy,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'project( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.train( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
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
			self.transformed_data = self.model.transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
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
			self.transformed_data = self.model.train( X, y ).transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'train_transform( self, X: np.ndarray, Y: np.ndarray ) -> tuple'
			error = ErrorDialog( exception )
			error.show( )

class PCA( Selector ):
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
	prediction: Optional[ np.ndarray ]
	svd_solver: Optional[ str ]
	n_components: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
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
		super( ).__init__( )
		self.n_components = num
		self.svd_solver = solver
		self.model = sd.PCA( n_components=self.n_components, svd_solver=self.svd_solver )
		self.prediction = None
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members.

		'''
		return [ 'model',
		         'prediction',
		         'svd_solver',
		         'n_components',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: int=0.2, random: int=42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		'''

			Purpose:
			_______


			Parameters:
			---------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary class target_names. ( n_samples, ).
			size (int): The size of the testing data set
			random (int): A random seed.


			Returns:
			________
			tuple ( np.ndarray, np.ndarray, np.ndarray, np.ndarray )
			ex. ( X_train, X_test, y_train, y_test )


		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return ( X_train, X_test, y_train, y_test )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = ('split_data( self, X: ndarray, y: ndarray, size: int=0.2, '
			                    'random: int=42 ) -> ( ndarray, ndarray, ndarray, ndarray )')
			error = ErrorDialog( exception )
			error.show( )
	
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
	
	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels from input features using the regression model.


			Parameters:
			---------
			X (np.ndarray | pd.DataFrame):
			Input features.


			Returns:
			--------
			np.ndarray:

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'project( self, X: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Returns the coefficient of determination R^2 of the prediction.
			The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
			((y_true - y_pred) ** 2).sum() and v is the total sum of squares
			((y_true - y_true.mean()) ** 2).sum().

			The best possible score is 1.0 and it can be negative
			(because the model can be arbitrarily worse). A constant model that always predicts
			the expected value of y, disregarding the input features, would get a R^2 score of 0.0.


			Parameters:
			-----------
			X (np.ndarray ): Input features.
			y (np.ndarray ): True binary class labels.

			Returns:
			--------
			Dict[ str, float]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			X_training, X_testing, y_training, y_testing = self.split_data( X, y )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.accuracy = accuracy_score( y, y_pred )
			_metrics = \
				{
					'Training Score': self.training_score,
					'Testing Score': self.testing_score,
					'Accuracy Score': self.accuracy,
				}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Apply the PCA transformation.

			Parameters:
			-----------
			X : Feature vector w/shape ( n_samples, n_features ).

			Return:
			-------
			(np.ndarray, np.ndarray): Transformed X

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

class SelectBest( Selector ):
	"""

		Purpose:
		---------
		Class that removes all but the 'k' highest scoring features

	"""
	model: sf.SelectKBest
	prediction: Optional[ np.ndarray ]
	k_best: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	score_function: Optional[ callable ]
	chi2_score: Optional[ Tuple[ float, float ] ]
	
	def __init__( self, score_func: callable=sf.chi2,  k: int=3 ) -> None:
		"""

			Purpose:
			---------
			Initialize SelectBest.

			:param threshold: Features with variance below this are removed.
			:type threshold: float

		"""
		super( ).__init__( )
		self.k_best = k
		self.score_function = score_func
		self.model = sf.SelectKBest( score_func=self.score_function, k=self.k_best )
		self.prediction = None
		self.transformed_data = None
		self.chi2_score = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members

		'''
		return [ 'k_best',
		         'model',
		         'prediction',
		         'accuracy',
		         'score_function',
		         'chi2_score',
		         'transformed_data',
		         'chi_square', ]
	
	def chi_square( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ float, float ]:
		'''
			
			Purpose:
			-------
			Compute chi-squared stats between each non-negative feature and class.
			This score can be used to select the n_features features with the highest values
			for the test chi-squared statistic from X, which must contain only non-negative
			integer feature values such as booleans or frequencies (e.g., term counts in
			document classification), relative to the classes.
			
			Parameters:
			----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary class target_names. ( n_samples, ).
			
			
			Return:
			--------
			(float, float) chi2, pvalue
		
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			( stats, p_vals ) = self.score_function( X, y )
			return ( stats, p_vals )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'chi_square( self, X: ndarray, y: ndarray ) -> ( ndarray, ndarray)'
			error = ErrorDialog( exception )
			error.show( )
		
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: int=0.2, random: int=42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		'''

			Purpose:
			_______


			Parameters:
			---------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary class target_names. ( n_samples, ).
			size (int): The size of the testing data set
			random (int): A random seed.


			Returns:
			________
			tuple ( np.ndarray, np.ndarray, np.ndarray, np.ndarray )
			ex. ( X_train, X_test, y_train, y_test )


		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = ('split_data( self, X: ndarray, y: ndarray, size: int=0.2, '
			                    'random: int=42 ) -> ( ndarray, ndarray, ndarray, ndarray )')
			error = ErrorDialog( exception )
			error.show( )
	
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
	
	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels from input features using the regression model.


			Parameters:
			---------
			X (np.ndarray | pd.DataFrame):
			Input features.


			Returns:
			--------
			np.ndarray:

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'project( self, X: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Returns the coefficient of determination R^2 of the prediction.
			The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
			((y_true - y_pred) ** 2).sum() and v is the total sum of squares
			((y_true - y_true.mean()) ** 2).sum().

			The best possible score is 1.0 and it can be negative
			(because the model can be arbitrarily worse). A constant model that always predicts
			the expected value of y, disregarding the input features, would get a R^2 score of 0.0.


			Parameters:
			-----------
			X (np.ndarray ): Input features.
			y (np.ndarray ): True binary class labels.

			Returns:
			--------
			Dict[ str, float]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			X_training, X_testing, y_training, y_testing = self.split_data( X, y )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.accuracy = accuracy_score( y, y_pred )
			_metrics = \
			{
				'Training Score': self.training_score,
				'Testing Score': self.testing_score,
				'Accuracy Score': self.accuracy,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'project( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Apply the CCA transformation.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).

			Return:
			-------
			(np.ndarray, np.ndarray): Transformed X and Y.

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

class SelectPercent( Selector ):
	"""

		Purpose:
		---------


	"""
	model: sf.SelectPercentile
	prediction: Optional[ np.ndarray ]
	percent: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	score_function: Optional[ callable ]
	accuracy: Optional[ float ]
	chi2_score: Optional[ float ]
	
	def __init__( self, score_func: callable=sf.chi2, percent: int=10 ) -> None:
		"""

			Purpose:
			---------
			Initialize SelectBest.

			score_func:

		"""
		super( ).__init__( )
		self.percent = percent
		self.score_function: score_func
		self.model = sf.SelectPercentile( score_func=self.score_function, percentile=self.percent )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members

		'''
		return [ 'percent',
		         'model',
		         'prediction',
		         'transformed_data',
		         'split_data',
		         'score_function',
		         'train',
		         'project',
		         'train_transform',
		         'transform',
		         'chi_square' ]
	
	def chi_square( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ float, float ]:
		'''

			Purpose:
			-------
			Compute chi-squared stats between each non-negative feature and class.
			This score can be used to select the n_features features with the highest values
			for the test chi-squared statistic from X, which must contain only non-negative
			integer feature values such as booleans or frequencies (e.g., term counts in
			document classification), relative to the classes.

			Parameters:
			----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary class target_names. ( n_samples, ).


			Return:
			--------
			(float, float) chi2, pvalue

		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			( stats, p_vals ) = self.score_function( X, y )
			return ( stats, p_vals )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'chi_square( self, X: ndarray, y: ndarray ) -> ( ndarray, ndarray)'
			error = ErrorDialog( exception )
			error.show( )
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: int=0.2, random: int=42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		'''

			Purpose:
			_______


			Parameters:
			---------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary class target_names. ( n_samples, ).
			size (int): The size of the testing data set
			random (int): A random seed.


			Returns:
			________
			tuple ( np.ndarray, np.ndarray, np.ndarray, np.ndarray )
			ex. ( X_train, X_test, y_train, y_test )


		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = ('split_data( self, X: ndarray, y: ndarray, size: int=0.2, '
			                    'random: int=42 ) -> ( ndarray, ndarray, ndarray, ndarray )')
			error = ErrorDialog( exception )
			error.show( )
	
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
	
	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels from input features using the regression model.


			Parameters:
			---------
			X (np.ndarray | pd.DataFrame):
			Input features.


			Returns:
			--------
			np.ndarray:

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'project( self, X: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Returns the coefficient of determination R^2 of the prediction.
			The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
			((y_true - y_pred) ** 2).sum() and v is the total sum of squares
			((y_true - y_true.mean()) ** 2).sum().

			The best possible score is 1.0 and it can be negative
			(because the model can be arbitrarily worse). A constant model that always predicts
			the expected value of y, disregarding the input features, would get a R^2 score of 0.0.


			Parameters:
			-----------
			X (np.ndarray ): Input features.
			y (np.ndarray ): True binary class labels.

			Returns:
			--------
			Dict[ str, float]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			X_training, X_testing, y_training, y_testing = self.split_data( X, y )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.accuracy = accuracy_score( y, y_pred )
			_metrics = \
				{
						'Training Score': self.training_score,
						'Testing Score': self.testing_score,
						'Accuracy Score': self.accuracy,
				}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'project( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Apply the Sequential Back Selection transformation.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).

			Return:
			-------
			(np.ndarray, np.ndarray): Transformed X.

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

class SBS( Selector ):
	'''
	
		Purpose:
		--------
	
	
	'''
	scoring: Optional[ callable ]
	prediction: Optional[ np.ndarray ]
	classifier: Optional[ Classifier ]
	random_state: Optional[ int ]
	test_size: Optional[ float ]
	k_features: Optional[ int ]
	accuracy: Optional[ float ]
	
	def __init__( self, classifier: Classifier, k_features: int,
			scoring: callable=accuracy_score, test_size: float=0.25, random_state: int=1 ):
		'''

			Purpose:
			--------
			Sequential Back Selection (SBS) Contrstructor

		'''
		super( ).__init__( )
		self.scoring = scoring
		self.classifier = clone( classifier )
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state
		self.prediction = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members

		'''
		return [ 'k',
		         'model',
		         'prediction',
		         'transformed_data',
		         'split_data',
		         'train',
		         'transform',
		         '_calc_score'
		         'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: int=0.2, random: int=42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		'''

			Purpose:
			_______
			Splits the dataset into to traing and testing splits


			Parameters:
			---------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary class target_names. ( n_samples, ).
			size (int): The size of the testing data set
			random (int): A random seed.


			Returns:
			________
			tuple ( np.ndarray, np.ndarray, np.ndarray, np.ndarray )
			ex. ( X_train, X_test, y_train, y_test )


		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return ( X_train, X_test, y_train, y_test )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = ('split_data( self, X: ndarray, y: ndarray, size: int=0.2, '
			                    'random: int=42 ) -> ( ndarray, ndarray, ndarray, ndarray )')
			error = ErrorDialog( exception )
			error.show( )
	
	def train( self, X: np.ndarray, y: np.ndarray ) ->  object | None:
		'''

			Purpose:
			--------
			Fits the data to the model

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).
			
			
			Return:
			---------
			object: The Sequential Back Selection (SBS) trained model.
			

		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=self.test_size,
					random_state=self.random_state )
			dim = X_train.shape[ 1 ]
			self.indices_ = tuple( range( dim ) )
			self.subsets_ = [ self.indices_ ]
			score = self._calc_score( X_train, y_train, X_test, y_test, self.indices_ )
			self.scores_ = [ score ]
			while dim > self.k_features:
				scores = [ ]
				subsets = [ ]
				for p in combinations( self.indices_, r=dim - 1 ):
					score = self._calc_score( X_train, y_train, X_test, y_test, p )
					scores.append( score )
					subsets.append( p )
					best = np.argmax( scores )
					self.indices_ = subsets[ best ]
					self.subsets_.append( self.indices_ )
					dim -= 1
					self.scores_.append( scores[ best ] )
					self.k_score_ = self.scores_[ -1 ]
					return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> object'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Apply the Sequential Back Selection transformation.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).

			Return:
			-------
			(np.ndarray, np.ndarray): Transformed X .

		"""
		try:
			throw_if( 'X', X )
			return X[ :, self.indices_ ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def _calc_score( self, X_train: np.ndarray, y_train: np.ndarray,
			X_test: np.ndarray, y_test: np.ndarray, indices: List[ int ] ) -> float:
		'''

			Purpose:
			--------
			Instance method used to calculate the model's performance


			Parameters:
			-----------
			X_train (np.ndarray, np.ndarray):  Transformed X
			y_train (np.ndarray, ):  Transformed y
			X_test (np.ndarray, np.ndarray):  Transformed X
			y_test (np.ndarray, ):  Transformed y
			indices (List[ int ]): List of integer indices

			Return:
			-------
			float: The resulting score representing performance.

		'''
		try:
			throw_if( 'X_train', X_train )
			throw_if( 'y_train', y_train )
			throw_if( 'X_train', X_train )
			throw_if( 'y_train', y_train )
			throw_if( 'indices', indices )
			self.estimator.fit( X_train[ :, indices ], y_train )
			y_pred = self.estimator.predict( X_test[ :, indices ] )
			score = self.scoring( y_test, y_pred )
			return score
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = ('_calc_score( self, X_train: ndarray, y_train: ndarray, '
			                    'X_test: ndarray, y_test: ndarray, indices: List[ int ]) ->float')
			error = ErrorDialog( exception )
			error.show( )

class RFE( Selector ):
	"""

		Purpose:
		---------
		Recursive Feature Elimination (RFE) Given an external estimator that assigns weights
		to features (e.g., the coefficients of a linear model), recursive feature elimination (RFE)
		is to select features by recursively considering smaller and smaller sets of features.
		
		First, the estimator is trained on the initial set of features and the importance of each
		feature is obtained either through a coef_ attribute or
		through a feature_importances_ attribute. Then, the least important features are pruned
		from current set of features. That procedure is recursively repeated on the pruned set
		until the desired number of features to select is eventually reached.

	"""
	model: Optional[ sf.RFE ]
	prediction: Optional[ np.ndarray ]
	classifier: Optional[ NearestNeighbor ]
	transformed_data: Optional[ np.ndarray ]
	n_features_to_select: Optional[ int ]
	verbose: Optional[ int ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, k_features: int=None, verbose: int=0  ) -> None:
		"""

			Purpose:
			---------
			Initialize Recursive Feature Elimination (RFE)


		"""
		super( ).__init__( )
		self.n_features_to_select = k_features
		self.classifier = NearestNeighbor( )
		self.verbose = verbose
		self.model = sf.RFE( estimator=self.classifier,
			n_features_to_select=self.n_features_to_select, verbose=self.verbose )
		self.prediction = None
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members

		'''
		return [ 'classifier',
		         'n_features_to_select',
		         'verbose',
		         'transformed_data',
		         'features_in',
		         'ranking',
		         'split_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	@property
	def features_in( self ) -> int:
		'''

			Returns
			-------
			ndarray of shape (n_features,)
			The number of features selected features.

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	@property
	def ranking( self ) -> np.ndarray :
		'''

			Returns
			-------
			An array of shape [n_features]
			The feature ranking, such that ranking_[i] corresponds to the ranking
			position of the i-th feature. Selected (i.e., estimated best)
			features are assigned rank 1
			

		'''
		if self.model.ranking_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.ranking_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: int = 0.2, random: int = 42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		'''

			Purpose:
			_______


			Parameters:
			---------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary class target_names. ( n_samples, ).
			size (int): The size of the testing data set
			random (int): A random seed.


			Returns:
			________
			tuple ( np.ndarray, np.ndarray, np.ndarray, np.ndarray )
			ex. ( X_train, X_test, y_train, y_test )


		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = ('split_data( self, X: ndarray, y: ndarray, size: int=0.2, '
			                    'random: int=42 ) -> ( ndarray, ndarray, ndarray, ndarray )')
			error = ErrorDialog( exception )
			error.show( )
	
	def train( self, X: np.ndarray ) -> sf.VarianceThreshold | None:
		"""

			Purpose:
			---------
			Fits the RFE model.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'fit( self, X: np.ndarray ) -> object | None'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels from input features using the regression model.


			Parameters:
			---------
			X (np.ndarray | pd.DataFrame):
			Input features.


			Returns:
			--------
			np.ndarray:

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'project( self, X: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Returns the coefficient of determination R^2 of the prediction.
			The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
			((y_true - y_pred) ** 2).sum() and v is the total sum of squares
			((y_true - y_true.mean()) ** 2).sum().

			The best possible score is 1.0 and it can be negative
			(because the model can be arbitrarily worse). A constant model that always predicts
			the expected value of y, disregarding the input features, would get a R^2 score of 0.0.


			Parameters:
			-----------
			X (np.ndarray ): Input features.
			y (np.ndarray ): True binary class labels.

			Returns:
			--------
			Dict[ str, float]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			X_training, X_testing, y_training, y_testing = self.split_data( X, y )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.accuracy = accuracy_score( y, y_pred )
			_metrics = \
			{
				'Training Score': self.training_score,
				'Testing Score': self.testing_score,
				'Accuracy Score': self.accuracy,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Apply RFE transformation.

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
			exception.cause = 'RFE'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit and transform the stores using recursive feature elimination.

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
			exception.cause = 'RFE'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
