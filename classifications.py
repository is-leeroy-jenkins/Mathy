'''
	******************************************************************************************
	  Assembly:                mathy
	  Filename:                classifications.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022
	
	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="classifications.py" company="Terry D. Eppler">
	
		 mathy Models
	
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
		classifications.py
	</summary>
	******************************************************************************************
'''
from __future__ import annotations
from typing import Dict
from typing import Optional, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.ensemble as ske
import sklearn.linear_model as skc
import sklearn.neighbors as skn
import sklearn.neural_network as snn
import sklearn.svm as skv
import sklearn.tree as skd
from matplotlib.colors import ListedColormap
from sklearn.base import ClassifierMixin
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, r2_score,
                             mean_squared_error, mean_absolute_error,
                             explained_variance_score, median_absolute_error)
from sklearn.preprocessing import Binarizer
from boogr import Error, ErrorDialog

def throw_if( name: str, value: object ):
	if not value:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Classifier( ):
	"""

		Purpose:
		---------
		Abstract base class that defines the interface for all linerar_model wrappers.

	"""
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self ):
		pass
	
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
	
	def project( self, X: np.ndarray, y: np.ndarray=None  ) -> np.ndarray:
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
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
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""

			Purpose:
			---------
			Evaluate the model using multiple performance metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
				y (np.ndarray): Ground truth target_names.

			Returns:
			-----------
				dict: Dictionary containing multiple evaluation metrics.

		"""
		raise NotImplementedError

class Perceptron( Classifier ):
	"""


			Purpose:
			---------
			The Perceptron is a simple classification algorithm suitable for
			large scale learning. By default:
				It does not require a learning rate.
				It is not regularized (penalized).
				It updates its model only on mistakes.

			The last characteristic implies that the Perceptron is slightly faster to train than
			SGD with the hinge loss and that the resulting models are sparser. In fact, the
			Perceptron is a wrapper around the SGDClassifier class using a perceptron loss and a
			constant learning rate.

	"""
	perceptron_model: skc.Perceptron
	prediction: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	alpha: Optional[ float ]
	max_iter: Optional[ int ]
	shuffle: Optional[ bool ]
	penalty: Optional[ str ]
	
	def __init__( self, reg: float=0.0001, iters: int=1000, shuffle: bool=True, penalty='l2' ) -> None:
		"""

			Purpose:
			---------
			Initialize the PerceptronClassifier linerar_model.


			Parameters:
			----------
			max_iter (int): Maximum number of iterations.
			Default is 1000.

		"""
		super( ).__init__( )
		self.alpha = reg
		self.max_iter = iters
		self.shuffle = shuffle
		self.penalty = penalty
		self.perceptron_model = skc.Perceptron( alpha=self.alpha, max_iter=self.max_iter,
			shuffle=self.shuffle, penalty=self.penalty, )
		self.prediction = None
		self.decision = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
		         'max_iter',
		         'random_state',
		         'accuracy',
		         'mean_absolute_error',
		         'mean_squared_error',
		         'r_mean_squared_error',
		         'r2_score',
		         'penalty',
		         'alpha',
		         'explained_variance_score',
		         'median_absolute_error',
		         'train',
		         'project',
		         'score',
		         'analyze',
		         'create_heatmap',
		         'weights',
		         'decision_function',
		         'weights',
		         'iterations' ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.perceptron_model.coef_ is None:
			raise AttributeError( 'The Perceptron data is untrained.' )
		else:
			return self.perceptron_model.coef_
	
	@property
	def iterations( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ is ndarray of shape (n_classes, )
			Actual number of iterations for all classes.
			If binary or multinomial, it returns only 1 element. For liblinear solver,
			only the maximum number of iteration across all classes is given.

		'''
		if self.perceptron_model.n_iter_ is None:
			raise AttributeError( 'The model iterations have not been initialized!' )
		else:
			return self.perceptron_model.n_iter_
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict confidence scores for samples. The confidence score for a sample is proportional
			to the signed distance of that sample to the hyperplane.

			Parameters
			----------
			X (np.ndarray) of shape (n_samples, n_features)
			The data matrix for which we want to get the confidence scores.

			Returns
			-------
			np.ndarray of shape (n_samples,) or (n_samples, n_classes)
			Confidence scores per (n_samples, n_classes) combination. In the binary case,
			confidence score for self.classes_[1] where >0 means this class would be predicted.

		"""
		try:
			throw_if( 'X', X )
			self.decision = self.perceptron_model.decision_function( X )
			return self.decision
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'decision_function( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Perceptron | None:
		"""

			Purpose:
			---------
			Fit the PerceptronClassifier linerar_model.

			Parameters:
			---------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary class target_names.

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.perceptron_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None  ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict binary class target_names using the PerceptronClassifier.

			Parameters:
			----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			---------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.perceptron_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""


			Purpose:
			---------
			Compute accuracy of the PerceptronClassifier classifier.

			Parameters:
			---------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			---------
			float

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.perceptron_model.predict( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""


			Purpose:
			-----------
			Evaluate classifier performance using standard classification metrics.

			Parameters:
			---------
			X (np.ndarray): Input feature_names of shape (n_samples, n_features).
			y (np.ndarray): Ground truth class target_names.

			Returns:
			---------
			dict: Dictionary of evaluation metrics including:
			- Accuracy (float)
			- Precision (float)
			- Recall (float)
			- F1 Score (float)
			- ROC AUC (float)
			- Matthews Corrcoef (float)
			- Confusion Matrix (List[List[int]])

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.perceptron_model.predict( X )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			---------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
				y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			---------
				None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.perceptron_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, anno=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''

			Purpose:
			--------
			Visualize how well it separates the different sample

			:param X:
			:type X: np.ndarray
			:param y:
			:type y: np.ndarray
			:param test_idx:
			:type test_idx: int
			:param resolution:
			:type resolution: float
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min, x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T, y )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[idx ],
					marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter( X_test[ :, 0 ], X_test[:, 1 ], c='none', edgecolor='black',
						alpha=1.0, linewidth=1, marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = ('visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, '
								'resolution=0.02 )')
			error = ErrorDialog( exception )
			error.show( )

class LinearRegression( Classifier ):
	"""
	
		Purpose:
		--------
		Wrapper class for sklearn.linear_model.LinearRegression to enable its use in binary
		classification tasks. This includes conversion of continuous outputs to binary labels
		via thresholding.
		
		
		Parameters:
		----------
		threshold (float, optional):
		Threshold above which predictions are considered class 1 (default: 0.5).
		**kwargs:
		Additional keyword arguments passed to sklearn's LinearRegression.
		
		
		Attributes:
		---------
		model (SklearnLinearRegression):
		Underlying scikit-learn linear regression model.
		threshold (float):
		Threshold for classification decision boundary.
	
	"""
	linear_model: skc.LinearRegression
	binarizer: Binarizer
	y_prediction: Optional[ np.ndarray ]
	threshold: float
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	alpha: Optional[ float ]
	
	def __init__( self, threshold: float=0.5 ) -> None:
		super( ).__init__( )
		self.threshold = threshold
		self.linear_model = skc.LinearRegression( )
		self.prediction = None
		self.probability = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
		         'max_iter',
		         'random_state',
		         'accuracy',
		         'mean_absolute_error',
		         'mean_squared_error',
		         'r_mean_squared_error',
		         'r2_score',
		         'penalty',
		         'alpha',
		         'explained_variance_score',
		         'median_absolute_error',
		         'train',
		         'project',
		         'score',
		         'analyze',
		         'create_heatmap',
		         'weights',
		         'density_function',
		         'weights', ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.linear_model.coef_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
		else:
			return self.linear_model.coef_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LinearRegression:
		"""
		
			Purpose:
			--------
			Train the linear regression model on input features and binary targets.
			
			
			Parameters:
			-----------
			X (np.ndarray | pd.DataFrame):
			Input features.
			y (np.ndarray | pd.Series):
			Binary class labels (0 or 1).
			
			
			Returns:
			None
			
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.linear_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LinearModel'
			exception.method = 'train'
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
			Predicted class labels (0 or 1).
			
		"""
		try:
			throw_if( 'X', X )
			self.predictions = self.linear_model.predict( X )
			self.binarizer = Binarizer( threshold=self.threshold )
			_shape = self.predictions.reshape( -1, 1 )
			return self.binarizer.fit_transform( _shape ).astype( int ).flatten( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LinearModel'
			exception.method = 'predict'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			Parameters:
			-----------
			X (np.ndarray ): Input features.
			y (np.ndarray ): True binary class labels.
			
			Returns:
			--------
			float: Accuracy score (0.0 to 1.0).
		
		"""
		try:
			self.y_prediction = self.predict( X )
			self.accuracy = accuracy_score( y, self.y_prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LinearModel'
			exception.method = 'score'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""
	
			Purpose:
			-----------
			Evaluate the classifier using multiple classification metrics.
	
			Parameters:
			-----------
			X (np.ndarray): Input feature_names of shape (n_samples, n_features).
			y (np.ndarray): True target_names of shape (n_samples,).
	
			Returns:
			-----------
			dict: Dictionary containing:
			- Accuracy (float)
			- Precision (float)
			- Recall (float)
			- F1 Score (float)
			- ROC AUC (float)
			- Matthews Corrcoef (float)
			- Confusion Matrix (List[List[int]])
	
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.linear_model.predict( X )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LinearModel'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
	
	
			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.
	
			Parameters:
			---------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
				y (np.ndarray): True class target vector of shape ( n_samples, ).
	
			Returns:
			---------
				None
	
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.linear_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, anno=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LinearModel'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''
	
			Purpose:
			--------
			Visualize how well it separates the different sample
	
			:param X:
			:type X: np.ndarray
			:param y:
			:type y: np.ndarray
			:param test_idx:
			:type test_idx: int
			:param resolution:
			:type resolution: float
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min,
				x2_max, resolution ) )
			lab = self.project( np.ndarray( [ xx1.ravel( ), xx2.ravel( ) ] ).T, y )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[ idx ],
					marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c='none', edgecolor='black',
						alpha=1.0, linewidth=1, marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LinearModel'
			exception.method = ('visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, '
								'resolution=0.02 )')
			error = ErrorDialog( exception )
			error.show( )

class LogisticRegression( Classifier ):
	"""

		Purpose:
		--------
		Logistic regression, despite its name, is a linear model for classification rather
		than regression. Logistic regression is also known in the literature as logit regression,
		maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model,
		the probabilities describing the possible outcomes of a single trial are modeled
		using a logistic function.

		This class implements regularized logistic regression using the ‘liblinear’ library,
		‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ solvers. Note that alpha is
		applied by default. It can handle both dense and sparse input. Use C-ordered arrays or
		CSR matrices containing 64-bit floats for optimal performance;
		any other input format will be converted (and copied). The ‘newton-cg’, ‘sag’, and
		‘lbfgs’ solvers support only L2 alpha with primal formulation, or no
		alpha. The ‘liblinear’ solver supports both L1 and L2 alpha,
		with a dual formulation only for the L2 alpha. The Elastic-Net alpha
		is only supported by the ‘saga’ solver.

	"""
	logistic_model: skc.LogisticRegression
	prediction: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	random_state: int
	penalty: str
	multi_class: str
	alpha: float
	max_iter: int
	solver: str
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, C: float=1.0, penalty: str='l2', iters: int=100,
			multi_class: str='multinomial', solver: str='lbfgs' ) -> None:
		"""

			Purpose:
			--------
			Initialize the Logistic Regression linerar_model.

			Parameters:
			-----------
				max (int): Maximum number of iterations. Default is 1000.
				solver (str): Algorithm to use in optimization. Default is 'lbfgs'.

		"""
		super( ).__init__( )
		self.alpha = C
		self.penalty = penalty
		self.max_iter = iters
		self.multi_class = multi_class
		self.solver = solver
		self.logistic_model = skc.LogisticRegression( C=self.alpha, max_iter=self.max_iter,
			multi_class=self.multi_class, solver=self.solver, penalty=self.penalty )
		self.prediction = None
		self.decision = None
		self.probability = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
		         'accuracy',
		         'penalty',
		         'solver',
		         'multi_class',
		         'random_state',
		         'alpha',
		         'max_iter',
		         'mean_absolute_error',
		         'mean_squared_error',
		         'predict_probabilty',
		         'r_mean_squared_error',
		         'r2_score',
		         'explained_variance_score',
		         'decision_function',
		         'median_absolute_error',
		         'train',
		         'project',
		         'score',
		         'analyze',
		         'weights',
		         'iterations',
		         'labels' ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.logistic_model.coef_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
		else:
			return self.logistic_model.coef_
	
	@property
	def labels( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.logistic_model.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.logistic_model.classes_
	
	@property
	def iterations( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ is ndarray of shape (n_classes, )
			Actual number of iterations for all classes.
			If binary or multinomial, it returns only 1 element. For liblinear solver,
			only the maximum number of iteration across all classes is given.

		'''
		if self.logistic_model.n_iter_ is None:
			raise AttributeError( 'The model iterations have not been initialized!' )
		else:
			return self.logistic_model.n_iter_
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict confidence scores for samples. The confidence score for a sample is proportional
			to the signed distance of that sample to the hyperplane.

			Parameters
			----------
			X (np.ndarray) of shape (n_samples, n_features)
			The data matrix for which we want to get the confidence scores.

			Returns
			-------
			np.ndarray of shape (n_samples,) or (n_samples, n_classes)
			Confidence scores per (n_samples, n_classes) combination. In the binary case,
			confidence score for self.classes_[1] where >0 means this class would be predicted.

		"""
		try:
			throw_if( 'X', X )
			self.decision = self.logistic_model.decision_function( X )
			return self.decision
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'decision_function( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
		
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Probability estimates. The returned estimates for all classes are ordered
			by the label of classes. For a multi_class problem, if multi_class is
			set to be “multinomial” the softmax function is used to find the
			predicted probability of each class. Else use a one-vs-rest approach,
			i.e. calculate the probability of each class assuming it to be positive
			using the logistic function and normalize these values across all the classes.

			Parameters
			----------
			X (np.ndarray) of shape (n_samples, n_features)
			Vector to be scored, where n_samples is the number of samples
			and n_features is the number of features.

			Returns
			-------
			np.ndarray of shape (n_samples,) or (n_samples, n_classes)
			Returns the probability of the sample for each class in the model,
			where classes are ordered as they are in self.classes_.

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.logistic_model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LogisticRegression | None:
		"""

			Purpose:
			-----------
			Fit the logistic regression linerar_model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.logistic_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None  ) -> np.ndarray:
		"""

			Purpose:
			-----------
			Predict class target_names using the logistic regression linerar_model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ). IGNORED

			Returns:
			-----------
			np.ndarray: Predicted class target_names.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.logistic_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			-----------
			Compute classification accuracy.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				float: Accuracy accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.logistic_model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluate the classifier using multiple classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Input feature_names of shape (n_samples, n_features).
			y (np.ndarray): True target_names of shape (n_samples,).

			Returns:
			-----------
			dict: Dictionary containing:
			- Accuracy (float)
			- Precision (float)
			- Recall (float)
			- F1 Score (float)
			- ROC AUC (float)
			- Matthews Corrcoef (float)
			- Confusion Matrix (List[List[int]])

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.logistic_model.predict( X )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
				y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.logistic_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

class Ridge( Classifier ):
	"""

		Purpose:
		--------
		This classifier first converts binary targets to {-1, 1} and then treats the problem as a
		regression task, optimizing the same objective as above. The predicted class corresponds
		to the sign of the regressor’s prediction. For multiclass classification, the problem is
		treated as multi-output regression, and the predicted class corresponds to the output
		with the highest value.

		It might seem questionable to use a (penalized) Least Squares loss to fit a classification
		model instead of the more traditional logistic or hinge losses. However, in practice,
		all those models can lead to similar cross-validation scores in terms of accuracy
		or precision/recall, while the penalized least squares loss used by the RidgeClassifier
		allows for a very different choice of the numerical solvers with
		distinct computational performance profiles.

	"""
	ridge_classifier: skc.RidgeClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	alpha: Optional[ float ]
	solver: Optional[ str ]
	
	def __init__( self, alpha: float=1.0, solver: str='auto', size: int=1000, rando: int=42 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the Ridge Classifierlinerar_model.

			Parameters:
			-----------
			- alpha (float): Regularization strength. Default is 1.0.
			- solver (str): Solver to use. Default is 'auto'.
			- max (int): max iterations
			- rando (int): random seed

		"""
		super( ).__init__( )
		self.alpha = alpha
		self.solver = solver
		self.max_iter = size
		self.random_state = rando
		self.ridge_classifier = skc.RidgeClassifier( alpha=self.alpha, solver=self.solver,
			max_iter=self.max_iter, random_state=self.random_state )
		self.prediction = None
		self.probability = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_iter',
				 'random_state',
				 'accuracy',
				 'alpha',
				 'solver',
				 'ridge_classifier',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'decision_function',
				 'analyze',
				 'create_heatmap',
				 'weights',
		         'iterations',
		         'features' ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.ridge_classifier.coef_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
		else:
			return self.ridge_classifier.coef_
		
	@property
	def iterations( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ is ndarray of shape (n_classes, )
			Actual number of iterations for all classes.
			If binary or multinomial, it returns only 1 element. For liblinear solver,
			only the maximum number of iteration across all classes is given.

		'''
		if self.ridge_classifier.n_iter_ is None:
			raise AttributeError( 'The model iterations have not been initialized!' )
		else:
			return self.ridge_classifier.n_iter_
	
	@property
	def features( self ) -> np.ndarray:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.ridge_classifier.n_features_in_ is None:
			raise AttributeError( 'The model features have not been initialized!' )
		else:
			return self.ridge_classifier.n_features_in_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Ridge | None:
		"""


			Purpose:
			-----------
			Fit the RidgeRegressor regression linerar_model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
				Pipeline

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.ridge_classifier.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Project target target_names using the RidgeRegressor linerar_model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): IGNORED

			Returns:
			-----------
			np.ndarray: Predicted target names.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.ridge_classifier.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""


			Purpose:
			-----------
			Compute the R-squared accuracy for the Ridge model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				float: R-squared accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.ridge_classifier.predict( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict | None:
		"""

			Purpose:
			-----------
				Evaluates the Ridge model
				using multiple metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				dict: Evaluation metrics including MAE, RMSE, R², etc.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.ridge_classifier.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict confidence scores for samples. The confidence score for a sample is proportional
			to the signed distance of that sample to the hyperplane.

			Parameters
			----------
			X (np.ndarray) of shape (n_samples, n_features)
			The data matrix for which we want to get the confidence scores.

			Returns
			-------
			np.ndarray of shape (n_samples,) or (n_samples, n_classes)
			Confidence scores per (n_samples, n_classes) combination. In the binary case,
			confidence score for self.classes_[1] where >0 means this class would be predicted.

		"""
		try:
			throw_if( 'X', X )
			self.decision = self.ridge_classifier.decision_function( X )
			return self.decision
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'decision_function( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
			
	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot predicted vs actual target_names.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.ridge_classifier.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''
	
			Purpose:
			--------
			Visualize how well it separates the different sample
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).
			
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			# setup marker generator and color map
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			
			# plot the decision surface
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
				np.arange( x2_min, x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ),
											xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			
			# plot class examples
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[
					idx ], marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				
				# plot all examples
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c='none', edgecolor='black',
						alpha=1.0, linewidth=1, marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class Lasso( Classifier ):
	"""
	
		Purpose:
		---------
		Wrapper class for sklearn.linear_model.Lasso to enable
		its use in binary classification tasks.
	
		Parameters:
		------------
		threshold (float, optional):
		Threshold above which predictions are considered class 1 (default: 0.5).
		
		Attributes:
		-----------
		model (SklearnLasso):
		Underlying scikit-learn Lasso model.
		threshold (float):
		Threshold for classification decision boundary.
	
	"""
	lasso_model: skc.Lasso
	threshold: float
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	alpha: Optional[ float ]

	def __init__( self, threshold: float=0.5 ) -> None:
		super( ).__init__( )
		self.threshold = threshold
		self.lasso_model = skc.Lasso( threshold=self.threshold )
		self.prediction = None
		self.probability = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_iter',
				 'random_state',
				 'accuracy',
				 'loss',
				 'regularization',
				 'alpha',
				 'sgd_classifier',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'weights',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'weights',
		         'iterations',
		         'features' ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.lasso_model.coef_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
		else:
			return self.lasso_model.coef_
	
	@property
	def iterations( self ) -> np.ndarray:
		'''
	
			Returns
			-------
			classes_ is ndarray of shape (n_classes, )
			Actual number of iterations for all classes.
			If binary or multinomial, it returns only 1 element. For liblinear solver,
			only the maximum number of iteration across all classes is given.
	
		'''
		if self.lasso_model.n_iter_ is None:
			raise AttributeError( 'The model iterations have not been initialized!' )
		else:
			return self.lasso_model.n_iter_
	
	@property
	def features( self ) -> np.ndarray:
		'''
	
			Returns
			-------
			n_features_in_
			The number of features seen during training
	
		'''
		if self.lasso_model.n_features_in_ is None:
			raise AttributeError( 'The model features have not been trained!' )
		else:
			return self.lasso_model.n_features_in_
			
	def train( self, X: np.ndarray, y: np.ndarray ) -> Lasso | None:
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.lasso_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> LassoModel'
			error = ErrorDialog( exception )
			error.show()
	
	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Purpose:
			-------
			Predict class labels from input features using the Lasso model.
		
			Parameters:
			----------
			X (np.ndarray | pd.DataFrame): Input features.
		
			Returns:
			--------
			np.ndarray: Predicted class labels (0 or 1).
			
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.lasso_model.predict( X )
			self.binarizer = Binarizer( threshold=self.threshold )
			_shape = self.prediction.reshape( -1, 1 )
			return self.binarizer.fit_transform( _shape ).astype( int ).flatten( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		try:
			self.prediction = self.project( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""
	
			Purpose:
			-----------
			Evaluate the model using multiple regression metrics.
	
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Ground truth target_names.
	
			Returns:
			-----------
			dict: Dictionary of MAE, MSE, RMSE, R², etc.
	
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.lasso_model.predict( X )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )

	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
	
			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.
	
			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
				y (np.ndarray): True class target vector of shape ( n_samples, ).
	
			Returns:
			-----------
				None
	
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.lasso_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

class GradientDescent( Classifier ):
	"""

		Purpose:
		--------
		Linear classifiers (SVM, logistic regression, etc.) with Stochastic Gradient Descent (SGD)
		training.  This estimator implements regularized linear models with stochastic
		gradient descent learning:
		
		the gradient of the loss is estimated each sample at a time and the model is updated along
		the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch
		(online/out-of-core) learning via the partial_fit method. For best results using the
		default learning rate schedule, the stores should have zero mean and unit variance.

		This implementation works with stores represented as dense or sparse arrays of floating point
		 values for the feature_names. The model it fits can be controlled with the loss parameter;
		 by default, it fits a linear support vector machine (SVM).

		The regularizer is a penalty added to the loss function that shrinks model parameters
		towards the zero vector using either the squared Euclidean norm L2 or the absolute norm
		L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value
		because of the regularizer, the update is truncated to 0.0 to allow for learning sparse
		 models and achieve online feature selection.

	"""
	sgd_model: skc.SGDClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_iter: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	loss: Optional[ str ]
	regularization: Optional[ Any ]
	alpha: Optional[ float ]
	
	def __init__( self, loss: str='log_loss', size: int=5, reg: str='l2', alpha: float=0.0001 ) -> None:
		"""

			Purpose:
			-----------
			Initialize the SGDClassifier linerar_model.

			Parameters:
			-----------
			loss (str): Loss function to use. Defaults to 'hinge'.
			reg (str): Regularization function to use. Default is 'l2'.
			max (int): Maximum number of passes over the df. Default is 1000.

		"""
		super( ).__init__( )
		self.loss = loss
		self.max_iter = size
		self.regularization = reg
		self.alpha = alpha
		self.sgd_model = skc.SGDClassifier( loss=self.loss, max_iter=self.max_iter,
			penalty=self.regularization, alpha=self.alpha )
		self.prediction = None
		self.probability = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_iter',
				 'random_state',
				 'accuracy',
				 'loss',
				 'regularization',
				 'alpha',
				 'sgd_classifier',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'weights',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'weights',
		         'iterations' ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.sgd_model.coef_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
		else:
			return self.sgd_model.coef_
		
	@property
	def iterations( self ) -> np.ndarray:
		'''
	
			Returns
			-------
			classes_ is ndarray of shape (n_classes, )
			Actual number of iterations for all classes.
			If binary or multinomial, it returns only 1 element. For liblinear solver,
			only the maximum number of iteration across all classes is given.
	
		'''
		if self.sgd_model.n_iter_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.sgd_model.n_iter_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientDescent | None:
		"""

			Purpose:
			-----------
			Fit the SGD classifier linerar_model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
				Pipeline

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.sgd_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			-----------
			Predict class target_names using the SGD classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			-----------
				np.ndarray: Predicted class target_names.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.sgd_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			-----------
			Compute R^2 accuracy for the SGDRegressor.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
			float: R^2 accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.sgd_model.predict( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluate the classifier using standard metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
			dict: Dictionary containing:
			- Accuracy (float)
			- Precision (float)
			- Recall (float)
			- F1 Score (float)
			- ROC AUC (float)
			- Matthews Corrcoef (float)
			- Confusion Matrix (List[List[int]])

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.sgd_model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = (''
								'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict confidence scores for samples. The confidence score for a sample is
			proportional
			to the signed distance of that sample to the hyperplane.

			Parameters
			----------
			X (np.ndarray) of shape (n_samples, n_features)
			The data matrix for which we want to get the confidence scores.

			Returns
			-------
			np.ndarray of shape (n_samples,) or (n_samples, n_classes)
			Confidence scores per (n_samples, n_classes) combination. In the binary case,
			confidence score for self.classes_[1] where >0 means this class would be predicted.

		"""
		try:
			throw_if( 'X', X )
			self.decision = self.sgd_model.decision_function( X )
			return self.decision
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'decision_function( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""
	
			Purpose:
			---------
			Probability estimates. The returned estimates for all classes are ordered
			by the label of classes. For a multi_class problem, if multi_class is
			set to be “multinomial” the softmax function is used to find the
			predicted probability of each class. Else use a one-vs-rest approach,
			i.e. calculate the probability of each class assuming it to be positive
			using the logistic function and normalize these values across all the classes.
	
			Parameters
			----------
			X (np.ndarray) of shape (n_samples, n_features)
			Vector to be scored, where n_samples is the number of samples
			and n_features is the number of features.
	
			Returns
			-------
			np.ndarray of shape (n_samples,) or (n_samples, n_classes)
			Returns the probability of the sample for each class in the model,
			where classes are ordered as they are in self.classes_.
	
		"""
		try:
			throw_if( 'X', X )
			self.probability = self.sgd_classifer.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	   
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.sgd_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''
	
			Purpose:
			--------
			Visualize how well it separates the different sample
	
			:param X:
			:type X: np.ndarray
			:param y:
			:type y: np.ndarray
			:param test_idx:
			:type test_idx: int
			:param resolution:
			:type resolution: float
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min,
				x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[ idx ],
					marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c='none',
						edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='Test')
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class NearestNeighbor( Classifier ):
	"""

		Purpose:
		--------
		The principle behind the k-nearest neighbor methods is to find a predefined number of
		training samples closest in distance to the new point, and predict the label from these.
		The number of samples can be a user-defined constant (k-nearest neighbor rate),
		or vary based on the local density of points (radius-based neighbor rate).
		The distance can, in general, be any metric measure: standard Euclidean distance is the
		most common choice. Neighbors-based methods are known as non-generalizing
		machine rate methods, since they simply “remember” all of its training df
		(possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).

	"""
	neighbor_model: skn.KNeighborsClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	n_neighbors: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	algorithm: Any
	metric: str
	
	def __init__( self, num: int=5, algorithm: str='auto', metric: str='minkowski' ) -> None:
		"""


			Purpose:
			-----------
			Initialize the KNeighborsClassifier linerar_model.

			Attributes:
			-----------
				linerar_model (KNeighborsClassifier): Internal non-parametric classifier.
					Parameters:
						num (int): Number of neighbors to use. Default is 5.

		"""
		super( ).__init__( )
		self.n_neighbors = num
		self.algorithm = algorithm
		self.metric = metric
		self.neighbor_model = skn.KNeighborsClassifier( n_neighbors=self.n_neighbors,
			algorithm=self.algorithm, metric=self.metric )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'accuracy',
				 'n_neigbors',
				 'algorithm',
				 'metric',
				 'neighbor_classifier',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'predict_probabilty',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
				 'labels' ]
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> NearestNeighbor | None:
		"""

			Purpose:
			--------
			Fit the KNN classifier linerar_model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.neighbor_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Predict class target_names using the KNN classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			-----------
			np.ndarray: Predicted class target_names.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.neighbor_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Return probability estimates for the test stores X.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.neighbor_model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultilayerClassifier'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""


			Purpose:
			-----------
			Compute classification accuracy for k-NN.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				float: Accuracy accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.neighbor_model.predict( X )
			return accuracy_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict | None:
		"""


			Purpose:
			-----------
			Evaluate classification performance using various metrics.


			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				dict: Dictionary containing:
					- Accuracy (float)
					- Precision (float)
					- Recall (float)
					- F1 Score (float)
					- ROC AUC (float)
					- Matthews Corrcoef (float)
					- Confusion Matrix (List[List[int]])

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.neighbor_model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborClassifier'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
				y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.neighbor_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbotClassifier'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''

			Purpose:
			--------
			Visualize how well it separates the different sample

			:param X:
			:type X: np.ndarray
			:param y:
			:type y: np.ndarray
			:param test_idx:
			:type test_idx: int
			:param resolution:
			:type resolution: float
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min, x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[ idx ],
					marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter( X_test[ :, 0 ], X_test[:, 1 ], c='none',
						edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborClassifier'
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class DecisionTree( Classifier ):
	'''

		Purpose:
		--------
		Decision Trees (DTs) are a non-parametric supervised learning method used for
		classification. The goal is to create a model that predicts the value of a
		target variable by learning simple decision rules inferred from the stores feature_names.

		A tree can be seen as a piecewise constant approximation. Decision trees learn from stores
		to approximate a sine curve with a set of if-then-else decision rules.
		The deeper the tree, the more complex the decision rules and the fitter the model.

	'''
	decision_model: skd.DecisionTreeClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	classifier: Optional[ Any ]
	splitter: Optional[ str ]
	
	def __init__( self, criterion='gini', splitter='best', depth=3, rando: int=42 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the KNeighborsClassifier linerar_model.

		"""
		super( ).__init__( )
		self.criterion = criterion
		self.splitter = splitter
		self.max_depth = depth
		self.random_state = rando
		self.decision_model = skd.DecisionTreeClassifier( criterion=self.criterion,
			splitter=self.splitter, max_depth=self.max_depth, random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'accuracy',
				 'criterion',
				 'splitter',
				 'dt_classifier',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap' ]
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> DecisionTree | None:
		"""

			Purpose:
			--------
			Fit the KNN classifier linerar_model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.decision_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Predict class target_names using the KNN classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			-----------
				np.ndarray: Predicted class target_names.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.decision_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Return probability estimates for the test stores X.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.decision_model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeClassifier'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""


			Purpose:
			-----------
			Compute classification accuracy for k-NN.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				float: Accuracy accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.decision_model.predict( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluate classification performance using various metrics.


			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				dict: Dictionary containing:
					- Accuracy (float)
					- Precision (float)
					- Recall (float)
					- F1 Score (float)
					- ROC AUC (float)
					- Matthews Corrcoef (float)
					- Confusion Matrix (List[List[int]])

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.decision_model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeClassifier'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.decision_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''

			Purpose:
			--------
			Visualize how well it separates the different sample

			Parmeters:
			----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).
			
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min, x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[idx ],
					marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter(
						X_test[ :, 0 ], X_test[:, 1 ], c='none', edgecolor='black',
						alpha=1.0, linewidth=1, marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ''
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class RandomForest( Classifier ):
	"""

		Purpose:
		--------
		In random forests, each tree in the ensemble is built from a sample drawn with replacement
		(i.e., a bootstrap sample) from the training set.

		Furthermore, when splitting each node during the construction of a tree,
		the best split is found either from all input feature_names or a random subset of
		size max_features.

		The purpose of these two sources of randomness is to decrease the variance
		of the forest estimator. Individual decision trees typically exhibit high variance
		and tend to overfit. The injected randomness in forests yield decision trees with
		decoupled prediction errors. By taking an average of those predictions,
		errors can cancel out. Random forests achieve a reduced variance
		by combining diverse trees, sometimes at the cost of a slight increase in bias.
		The variance reduction is often significant hence yielding an overall better model.

	"""
	n_estimators: int
	criterion: Optional[ Any ]
	forest_model: ske.RandomForestClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ Any ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, est: int=10, crit: Any='gini', size: Any=None, rando: int=42 ) -> None:
		"""

			Purpose:
			-----------
			Initializes the RandomForestClassifier.

		"""
		super( ).__init__( )
		self.n_estimators = est
		self.criterion = crit
		self.max_depth = size
		self.random_state = rando
		self.forest_model = ske.RandomForestClassifier( n_estimators=self.n_estimators,
			criterion=self.criterion, max_depth=self.max_depth, random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'accuracy',
				 'n_estimators',
				 'max_depth',
				 'criterior',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'labels' ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.forest_model.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.forest_model.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> RandomForest | None:
		"""

			Purpose:
			-----------
			Fit the classifier.


			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
				Pipeline


		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.forest_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			-------
			Predict class target_names
			using the SGD classifier.

			Parameters:
			----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			---------
			np.ndarray: Predicted class target_names.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.forest_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Return probability estimates for the test stores X.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.forest_model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			-----------
			Compute R^2 accuracy for the SGDRegressor.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				float: R^2 accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.forest_model.predict( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluate the Lasso model using multiple regression metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				dict: Dictionary of MAE, RMSE, R², etc.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.forest_model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'VAR': self.explained_variance_score,
					'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.forest_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''

			Purpose:
			--------
			Visualize how well it separates the different sample

			:param X:
			:type X: np.ndarray
			:param y:
			:type y: np.ndarray
			:param test_idx:
			:type test_idx: int
			:param resolution:
			:type resolution: float
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min, x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[idx ],
					marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter(  X_test[ :, 0 ], X_test[ :, 1 ], c='none', edgecolor='black',
						alpha=1.0, linewidth=1, marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ''
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class GradientBoost( Classifier ):
	"""

		Purpose:
		--------
		Gradient Boosting Classifier builds an additive model in a forward stage-wise fashion;
		it allows for the optimization  of arbitrary differentiable loss functions.
		In each stage n_classes_ regression trees are  fit on the negative gradient of the binomial
		or multinomial deviance loss function. Binary classification is a special case where
		only a single regression tree is induced.

		The feature_names are always randomly permuted at each split. Therefore, the best found
		split
		may vary, even with the same training stores and max_features=n_features, if the improvement
		of the criterion is identical for several splits enumerated during the search of the best
		split. To obtain a deterministic behaviour during fitting, rando has to be fixed.

	"""
	gradient_model: ske.GradientBoostingClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, lss: str='deviance', rate: int=0.1, est: int=100,
			size: int=3, rando: int=42 ) -> None:
		"""

			Purpose:
			________
			Initialize the GradientBoostingClassifier.

			Parameters:
			___________
			lss: str
			rate: int
			estimators: int
			max: int
			rando: int

		"""
		super( ).__init__( )
		self.loss = lss
		self.learning_rate = rate
		self.n_estimators = est
		self.max_depth = size
		self.random_state = rando
		self.gradient_model = ske.GradientBoostingClassifier( loss=self.loss,
			learning_rate=self.learning_rate, n_estimators=self.n_estimators,
			max_depth=self.max_depth, random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'accuracy',
				 'loss',
				 'learning_rate',
				 'n_estimators',
				 'gradient_boost_classifier',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
				 'labels' ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.gradient_model.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.gradient_model.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoost | None:
		"""

			Purpose:
			________
			Fit the model to the training df.

			Parameters:
			__________
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			Pipeline

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.gradient_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoostClassifier'
			exception.method = ('train( self, X: np.ndarray, y: np.ndarray ) -> '
								'GradientBoostingClassifier')
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			________
			Predict class target_names.

			Parameters:
			__________
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			________
			np.ndarray: Predicted target_names.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.gradient_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoostClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Return probability estimates for the test stores X.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.gradient_model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoostingClassifier'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			_______
			Compute classification accuracy.

			Parameters:
			__________
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			________
			float: Accuracy accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.gradient_model.predict( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoostingClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			--------
			Evaluate classifier using multiple metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
				Dict[str, float]: Evaluation scores.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.gradient_model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoostingClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Display the confusion matrix.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.gradient_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoostingClassifier'
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''

			Purpose:
			--------
			Visualize how well it separates the different sample

			:param X:
			:type X: np.ndarray
			:param y:
			:type y: np.ndarray
			:param test_idx:
			:type test_idx: int
			:param resolution:
			:type resolution: float
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min,
				x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[idx ],
					marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
			if test_idx:
				X_test, y_test = X[ test_idx, : ], y[ test_idx ]
				plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c='none', edgecolor='black',
					alpha=1.0, linewidth=1,  marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ''
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class AdaptiveBoost( Classifier ):
	"""

		Purpose:
		---------
		An Boost classifier is a meta-estimator that begins by fitting a classifier
		on the original dataset and then fits additional copies of the classifier on the
		same dataset but where the weights of incorrectly classified instances are
		adjusted such that subsequent classifiers focus more on difficult cases.

	"""
	adaptive_model = ske.AdaBoostClassifier
	prediction: Optional[ np.ndarray ]
	n_estimators: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	X_scaled: Optional[ pd.DataFrame ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	estimator: Optional[ Any ]
	learning_rate: Optional[ float ]
	
	def __init__( self, num: int=100, learning: float=1.0 ) -> None:
		"""

			Initialize the Random Forest Classifier.

		"""
		super( ).__init__( )
		self.estimator = 'AdaBoostClassifier'
		self.n_estimators = num
		self.learning_rate = learning
		self.adaptive_model = ske.AdaBoostClassifier( estimator=self.estimator,
			n_estimators=self.n_estimators, learning_rate=self.learning_rate )
		self.X_scaled = None
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'accuracy',
				 'X_scaled',
				 'n_estimators',
				 'learning_rate',
				 'ada_boost_classifier',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'errors',
		         'weights',
		         'labels' ]
	
	@property
	def errors( self ) -> np.ndarray | None:
		if self.adaptive_model.estimator_errors_ is None:
			raise AttributeError( 'The model errors have not been initialized!' )
		else:
			return self.adaptive_model.estimator_errors_
	   
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.adaptive_model.estimator_weights_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
		else:
			return self.adaptive_model.estimator_weights_
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.adaptive_model.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.adaptive_model.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> AdaptiveBoost | None:
		"""

			Purpose:
			_______
			Fit the classifier.

			Parameters:
			_________
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			Pipeline

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.adaptive_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Predict class target_names
			using the SGD classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			-----------
			np.ndarray: Predicted class target_names.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.adaptive_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
			float: R^2 accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.adaptive_model.predict( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				dict: Dictionary of MAE, RMSE, R², etc.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.adaptive_model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'VAR': self.explained_variance_score,
					'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.adaptive_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''

			Purpose:
			--------
			Visualize how well it separates the different sample

			:param X:
			:type X: np.ndarray
			:param y:
			:type y: np.ndarray
			:param test_idx:
			:type test_idx: int
			:param resolution:
			:type resolution: float
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min, x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[idx ],
					marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c='none', edgecolor='black',
						alpha=1.0, linewidth=1, marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ''
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
class BaggingModel( Classifier ):
	"""

		Purpose:
		--------
		 Bagging methods form a class of algorithms which build several instances of a black-box
		 estimator on random subsets of the original training set and then aggregate their
		 individual predictions to form a final prediction. These methods are used as a way
		 to reduce the variance of a base estimator (e.g., a decision tree), by introducing
		 randomization into its construction procedure and then making an ensemble out of it.
		 In many cases, bagging methods constitute a very simple way to improve with respect
		 to a single model, without making it necessary to adapt the underlying base algorithm.
		 As they provide a way to reduce overfitting, bagging methods work best with strong and
		 complex models (e.g., fully developed decision trees), in contrast with boosting methods
		 which usually work best with weak models (e.g., shallow decision trees).

	"""
	bagging_classifier: ske.BaggingClassifier
	prediction: Optional[ np.ndarray ]
	max_features: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	base_estimator: Optional[ Any ]
	n_estimators: Optional[ int ]
	
	def __init__( self, base: object=None, num: int=10, size: int=1, rando: int=42 ) -> None:
		"""

			Initialize the BaggingClassifier.

		"""
		super( ).__init__( )
		self.base_estimator = base
		self.n_estimators = num
		self.max_features = size
		self.random_state = rando
		self.bagging_classifier = ske.BaggingClassifier( estimator=self.base_estimator,
			n_estimators=self.n_estimators, max_features=self.max_features,
			random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'accuracy',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'labels' ]
		
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.bagging_classifier.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.bagging_classifier.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BaggingModel | None:
		"""

			Purpose:
			--------
			 Fit the classifier.

			Parameters:
			----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-------
			Pipeline

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.bagging_classifier.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Predict class target_names
			using the SGD classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			-----------
				np.ndarray: Predicted class target_names.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.bagging_classifier.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				float: R^2 accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.bagging_classifier.predict( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
			dict: Dictionary of MAE, RMSE, R², etc.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, square=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MAE': self.mean_absolute_error,
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Plot confusion matrix
			for classifier predictions.

			Parameters:
			------------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
				y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			---------
				None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.bagging_classifier.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, index=None, resolution=0.02 ):
		'''

			Purpose:
			--------
			Visualize how well it separates the different sample

			:param X:
			:type X: np.ndarray
			:param y:
			:type y: np.ndarray
			:param index:
			:type index: int
			:param resolution:
			:type resolution: float
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min, x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[ idx ],
					marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				if index:
					X_test, y_test = X[ index, : ], y[ index ]
					plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c='none', edgecolor='black',
						alpha=1.0, linewidth=1, marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ''
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class VotingModel( Classifier ):
	"""

		Purpose:
		--------
		The idea behind the VotingClassifier is to combine conceptually different machine rate
		classifiers and use a majority vote or the average predicted probabilities (soft vote)
		to predict the class target_names. Such a classifier can be useful for a set of equally
		well
		performing model in order to balance out their individual weaknesses.

	"""
	voting_model: ske.VotingClassifier
	prediction: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	estimators: List[ (str, object) ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	estimators: List[ (str, object) ]
	vote: str
	
	def __init__( self, estimators: List[ ( str, object ) ], vote='hard' ) -> None:
		"""

			Initialize the RandomForestClassifier.

		"""
		super( ).__init__( )
		self.estimators = estimators
		self.voting = vote
		self.voting_model = ske.VotingClassifier( estimators=self.estimators, voting=self.voting )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'accuracy',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'labels' ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.voting_model.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.voting_model.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> VotingModel | None:
		"""

			Purpose:
			---------
			Fit the classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			Pipeline

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.voting_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Predict class target_names
			using the SGD classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			-----------
				np.ndarray: Predicted class target_names.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.voting_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				float: R^2 accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.voting_model.predict( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				dict: Dictionary of MAE, RMSE, R², etc.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.voting_model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MAE': self.mean_absolute_error,
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.voting_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''

			Purpose:
			--------
			Visualize how well it separates the different sample

			:param X:
			:type X: np.ndarray
			:param y:
			:type y: np.ndarray
			:param test_idx:
			:type test_idx: int
			:param resolution:
			:type resolution: float
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min, x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8,
					c=colors[ idx ], marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c='none',
						edgecolor='black', alpha=1.0, linewidth=1,
						marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ''
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
   
class StackingModel( Classifier ):
	"""

		Purpose:
		-------
		Stack of estimators with a final classifier. Stacked generalization consists in stacking
		the
		output of individual estimator and use a classifier to compute the final prediction.
		Stacking allows to use the strength of each individual estimator by using their output
		as input of a final estimator. Note that estimators_ are fitted on the full X while
		final_estimator_ is trained using cross-validated predictions of the base
		estimators using cross_val_predict.

	"""
	stacking_model: ske.StackingClassifier
	estimators: List[ Tuple[ str, ClassifierMixin ] ]
	final_estimator: Optional[ ClassifierMixin ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, est: List[ Tuple[ str, ClassifierMixin ] ], final: ClassifierMixin=None ) -> None:
		"""

			Initialize the RandomForestClassifier.

		"""
		super( ).__init__( )
		self.estimators = est
		self.final_estimator = final
		self.stacking_model = ske.StackingClassifier( estimators=self.estimators,
			final_estimator=self.final_estimator )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'accuracy',
				 'final_estimator',
				 'estimators',
				 'stacking_classifier',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'labels' ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.stacking_model.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.stacking_model.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> StackingModel | None:
		"""

			Purpose:
			---------
				Fit the classifier.

			Parameters:
			----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
				Pipeline

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.stacking_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Predict class target_names
			using the SGD classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			-----------
			np.ndarray: Predicted class target_names.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.stacking_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				float: R^2 accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.stacking_model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Stack Classifier model
			using multiple regression metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
				dict: Dictionary of MAE, RMSE, R², etc.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.stacking_model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MAE': self.mean_absolute_error,
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			-----------
			None
			

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.stacking_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''

			Purpose:
			--------
			Visualize how well it separates the different sample

			:param X:
			:type X: np.ndarray
			:param y:
			:type y: np.ndarray
			:param test_idx:
			:type test_idx: int
			:param resolution:
			:type resolution: float
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min, x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[idx ],
					marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c='none', edgecolor='black',
						alpha=1.0, linewidth=1, marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ''
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )


class SupportVector( Classifier ):
	"""

		Support Vector Classifier (SVC).The implementation is based on libsvm. The fit time scales
		at least quadratically with the number of samples and may be impractical beyond tens of
		thousands of samples.

	"""
	svc_model: skv.SVC
	multiclass: Optional[ str ]
	regulation: Optional[ float ]
	penalty: Optional[ str ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, multi: str='ovr', C: float=1.0, penalty: str='l2', degree: int=3 ) -> None:
		"""
		
			Purpose:
			---------
			Initialize the SVC model.
	
			:param kernel: Kernel type to be used in the algorithm.
			:type kernel: str
			:param C: Regularization parameter.
			:type C: float
			
		"""
		super( ).__init__( )
		self.multiclass = multi
		self.regulation = C
		self.penalty = penalty
		self.degree = degree
		self.svc_model = skv.SVC( multi_class=self.multiclass, C=self.regulation,
			random_state=self.random_state, penalty=self.penalty, degree=self.degree )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'accuracy',
				 'svc_classifier',
				 'kernel',
				 'regulation',
				 'degree',
				 'mean_absolute_error',
				 'mean_squared_error',
				 'r_mean_squared_error',
				 'r2_score',
				 'explained_variance_score',
				 'median_absolute_error',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'vectors',
		         'weights' ]
	
	@property
	def vectors( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			np.ndarray - array-like, shape = [n_SV, n_features]

		'''
		if self.svc_model.support_vectors_ is None:
			raise AttributeError( 'The models support vectors are uninitialized!' )
		else:
			return self.svc_model.support_vectors_
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.svc_model.coef_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
		else:
			return self.svc_model.coef_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> SupportVector | None:
		"""
		
			Purpose:
			---------
			Fit the SVC model to the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.svc_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""
			
			Purpose:
			--------
			Predict class target_names for the input feature_names.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.svc_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = ('train( self, X: np.ndarray, y: np.ndarray ) -> '
								'SupportVectorClassifier')
			error = ErrorDialog( exception )
			error.show( )
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Return probability estimates for the test stores X.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.svc_model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y_true: np.ndarray ) -> float | None:
		"""
		
			Purpose:
			---------
			Evaluate the model using accuracy score.
	

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).
			
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y_true', y_true )
			self.prediction = self.svc_model.predict( X )
			self.accuracy = accuracy_score( y_true, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'score( self, X: np.ndarray, y_true: np.ndarray ) -> float '
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y_true: np.ndarray ) -> str | None:
		"""
		
			Purpose:
			----------
			Generate classification report.
	

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).
			
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y_true', y_true )
			self.prediction = self.svc_model.predict( X )
			return classification_report( y_true, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'score( self, X: np.ndarray, y_true: np.ndarray ) -> float '
			error = ErrorDialog( exception )
			error.show( )
	
	def create_heatmap( self, X: np.ndarray, y_true: np.ndarray ) -> None:
		"""
		
			Purpose:
			---------
			Generate and display a confusion matrix.
	

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y_true (np.ndarray): True class target vector of shape ( n_samples, ).
			
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y_true', y_true )
			self.prediction = self.svc_model.predict( X )
			cm = confusion_matrix( y_true, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Projected' )
			plt.ylabel( 'Observed' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'create_heatmap( self, X: np.ndarray, y_true: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''

			Purpose:
			--------
			Visualize how well it separates the different sample

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Optional target array  of shape ( n_samples, ).
			test_idx: Opional[ int ]
			resolution: Optional[ float ]

		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			# setup marker generator and color map
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			
			# plot the decision surface
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min, x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			
			# plot class examples
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[idx ],
					marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				
				# plot all examples
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c='none', edgecolor='black',
						alpha=1.0, linewidth=1, marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ''
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class MultiLayerPerceptron( Classifier ):
	"""

		Purpose:
		-----------
		This model optimizes the squared error using LBFGS or stochastic gradient descent.

		Activation function for the hidden layers:
		- ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
		- ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
		- ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
		- ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

		The solver for weight optimization:
		- ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
		- ‘sgd’ refers to stochastic gradient descent.
		- ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma and Diederik

	"""
	multilayer_model: snn.MLPClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	hidden_layers: tuple[ int, ... ]
	activation_function: str
	solver: str
	alpha: float
	learning_rate: Any
	
	def __init__( self, hidden=(100,), activation='relu', solver='adam', alpha=0.0001,
			learning: str = 'constant', rando: int = 42 ) -> None:
		super( ).__init__( )
		self.hidden_layers = hidden
		self.activation_function = activation
		self.learning_rate = learning
		self.solver = solver
		self.alpha = alpha
		self.random_state = rando
		self.multilayer_model = snn.MLPClassifier( hidden_layer_sizes=self.hidden_layers,
			activation=self.activation_function, solver=self.solver, alpha=self.alpha,
			learning_rate=self.learning_rate, random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
		         'probability',
		         'max_depth',
		         'random_state',
		         'accuracy',
		         'mean_absolute_error',
		         'mean_squared_error',
		         'r_mean_squared_error',
		         'r2_score',
		         'explained_variance_score',
		         'median_absolute_error',
		         'train',
		         'project',
		         'score',
		         'analyze',
		         'create_heatmap',
		         'predict_probability',
		         'weights',
		         'classes',
		         'loss' ]
	
	@property
	def loss( self ) -> float:
		if self.multilayer_model.loss_ is None:
			raise AttributeError( 'The model loss has not been initialized!' )
		else:
			return self.multilayer_model.loss_
	
	@property
	def classes( self ) -> np.ndarray:
		if self.multilayer_model.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized' )
		else:
			return self.multilayer_model.classes_
	
	@property
	def weights( self ) -> np.ndarray:
		'''

			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.multilayer_model.coefs_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
		else:
			return self.multilayer_model.coefs_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> MultiLayerPerceptron | None:
		"""

			Purpose:
			-----------
			Fits all pipeline steps to the text df.

			Parameters:
			-----------
				X (np.ndarray): Input feature matrix.
				y (Optional[np.ndarray]): Optional target array.

			Returns:
			--------
				Pipeline

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.multilayer_model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerClassifier'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray = None ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Predict the class labels for the provided stores.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.multilayer_model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultilayerRegression'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Return probability estimates for the test stores X.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.multilayer_model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultilayerClassifier'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			-----------
			Compute the R^2 accuracy of the model on the given test df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True values.

			Returns:
			-----------
			float: R-squared accuracy.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.multilayer_model.predict( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultilayerRegression'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluate the model using multiple regression metrics.


			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Ground truth target_names.

			Returns:
			-----------
			dict: Dictionary of MAE, MSE, RMSE, R², etc.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.multilayer_model.predict( X )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerClassifier'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			---------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			---------
				None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.multilayer_model.predict( X )
			cm = confusion_matrix( y, self.prediction )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
			plt.xlabel( 'Predicted' )
			plt.ylabel( 'Actual' )
			plt.title( 'Confusion Matrix' )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
		'''

			Purpose:
			--------
			Visualize how well it separates the different sample

			:param X:
			:type X: np.ndarray
			:param y:
			:type y: np.ndarray
			:param test_idx:
			:type test_idx: int
			:param resolution:
			:type resolution: float
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )
			# plot the decision surface
			x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
			x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
			xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ), np.arange( x2_min, x2_max, resolution ) )
			lab = self.project( np.array( [ xx1.ravel( ),
			                                xx2.ravel( ) ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			for idx, cl in enumerate( np.unique( y ) ):
				plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[
					idx ], marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )
				if test_idx:
					X_test, y_test = X[ test_idx, : ], y[ test_idx ]
					plt.scatter(
						X_test[ :, 0 ], X_test[
							:, 1 ], c='none', edgecolor='black', alpha=1.0, linewidth=1,
						marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ''
			exception.method = 'visualize( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )


