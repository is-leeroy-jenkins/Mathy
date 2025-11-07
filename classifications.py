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
from sklearn.model_selection import train_test_split as split
from matplotlib import markers
from matplotlib.colors import ListedColormap
import seaborn as sns
from seaborn import colors
from sklearn.base import ClassifierMixin
from sklearn.metrics import ( recall_score, precision_score, confusion_matrix, classification_report,
                             auc, average_precision_score, balanced_accuracy_score,
                             ConfusionMatrixDisplay, accuracy_score, top_k_accuracy_score, f1_score,
                             hinge_loss, log_loss, mean_squared_error, root_mean_squared_error,
                             mean_absolute_error, median_absolute_error)
from sklearn.preprocessing import Binarizer
from boogr import Error, ErrorDialog

def throw_if( name: str, value: object ):
	if value is None:
		raise Exception( f'Argument "{name}" cannot be empty!' )
	


class Classifier( ):
	"""

		Purpose:
		---------
		Abstract base class that defines the interface for all linerar_model wrappers.

	"""
	max_iter: Optional[ int ]
	random_state: Optional[ int ]
	learning_rate: Optional[ float ]
	random_state: Optional[ int ]
	binarizer: Optional[ Binarizer ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	misclass: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	markers: Optional[ List[ str ] ]
	
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
			( np.ndarray, np.ndarray, np.ndarray, np.ndarray, ) | None ):
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
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
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			---------
			Evaluate the model using multiple performance metrics.

			Area Under Curve - AUC,
			Average Precision Score - APS,
			F1 Score - F1S,
			Hinge Loss - HLS,
			Log Loss - LLS,
			Top-K Accuracy - TKA
			
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
	model: skc.Perceptron
	binarizer: Optional[ Binarizer ]
	prediction: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	alpha: Optional[ float ]
	max_iter: Optional[ int ]
	shuffle: Optional[ bool ]
	penalty: Optional[ str ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float=0.001, eta: float=1.0, iters: int=1000,
			shuffle: bool=False, penalty=None ) -> None:
		"""

			Purpose:
			---------
			Initialize the Perceptron linerar model.


			Parameters:
			----------
			max_iter (int): Maximum number of iterations.
			Default is 1000.

		"""
		super( ).__init__( )
		self.binarizer = Binarizer( threshold=0.5 )
		self.alpha = alpha
		self.max_iter = iters
		self.shuffle = shuffle
		self.penalty = penalty
		self.learning_rate = eta
		self.model = skc.Perceptron( alpha=self.alpha, max_iter=self.max_iter,
			shuffle=self.shuffle, eta0=self.learning_rate, penalty=self.penalty )
		self.decision = None
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'model',
				 'prediction',
		         'misclass',
		         'max_iter',
		         'random_state',
		         'decision',
		         'train',
		         'project',
		         'score',
		         'analyze',
		         'penalty',
		         'shuffle',
		         'alpha',
		         'create_heatmap',
		         'weights',
		         'decision_function',
		         'weights',
		         'iterations',
		         'testing_score',
		         'training_score', ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape ( n_features, ) or ( n_targets, n_features )

		'''
		if self.model.coef_ is None:
			raise AttributeError( 'The Perceptron data is untrained.' )
		else:
			return self.model.coef_
	
	@property
	def iterations( self ) -> np.ndarray:
		'''

			Returns
			-------
			n_iter_ is a np.ndarray of shape ( n_classes, ).
			Represents the actual number of iterations for all classes.
			If binary or multinomial, it returns only 1 element. For liblinear solver,
			only the maximum number of iteration across all classes is given.

		'''
		if self.model.n_iter_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_iter_
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.classes_
	
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
			X_train, X_test, y_train, y_test = split( X, y, test_size=size,
				random_state=random )
			return ( X_train, X_test, y_train, y_test )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
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
			self.decision = self.model.decision_function( X )
			return self.decision
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
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
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
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
			exception.cause = 'Perceptron'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score ,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			_dataframe = pd.DataFrame( _metrics  )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Any:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlations' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def confusion_graph( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			plt.figure( figsize=( 8, 6 ) )
			cm = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( cm ).plot( )
			plt.tight_layout( )
			plt.title( 'Confusion Matrix' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def region_plot( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=10 ):
		'''
		
		'''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			markers = ('o', 's', '^', 'v', '<')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			# plot the decision surface
			X_training, X_testing, y_training, y_testing = self.split_data( X, y )
			cmap = ListedColormap( colors[ :len( np.unique( y_testing ) ) ] )
			x1_min, x1_max = X_testing.iloc[ :, 0 ].min( ) - 1, X_testing.iloc[ :, 0 ].max( ) + 1
			x2_min, x2_max = X_testing.iloc[ :, 1 ].min( ) - 1, X_testing.iloc[ :, 1 ].max( ) + 1
			first = np.arange( x1_min, x1_max, resolution )
			second = np.arange( x2_min, x2_max, resolution )
			xx1, xx2 = np.meshgrid( first, second, copy=False )
			lab = self.project( np.array( [ xx1.ravel(), xx2.ravel() ] ).T )
			lab = lab.reshape( xx1.shape )
			plt.contourf( xx1, xx2, lab, alpha=0.3, cmap=cmap )
			plt.xlim( xx1.min( ), xx1.max( ) )
			plt.ylim( xx2.min( ), xx2.max( ) )
			# plot class examples
			for idx, cl in enumerate( np.unique( y_testing ) ):
				plt.scatter( x=X_testing[ y_testing == cl, 0 ], y=X_testing[ y_testing == cl, 1 ], alpha=0.8,
					label=f'Class {cl}', edgecolor='black' )
			# highlight test examples
			if test_idx:
				# plot all examples
				X_test, y_test = X_testing[ test_idx, : ], y_testing[ test_idx ]
				plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ],
					c='none', edgecolor='black', alpha=1.0,
					linewidth=1, marker='o',
					s=100, label='Test set' )
			
			plt.grid( visible=True )
			plt.legend( loc='best' )
			plt.tight_layout( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
class LeastSquares( Classifier ):
	"""
	
		Purpose:
		--------
		Least Squares Regression fits a linear model with coefficients w = (w1, …, wp)
		to minimize the residual sum of squares between the observed targets
		in the dataset, and the targets predicted by the linear approximation.
		
		Parameters:
		----------
		threshold (float, optional):
		Threshold above which predictions are considered class 1 (default: 0.5).
		
	"""
	model: skc.LinearRegression
	binarizer: Optional[ Binarizer ]
	prediction: Optional[ np.ndarray ]
	misclass: Optional[ float ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.model = skc.LinearRegression( )
		self.binarizer = Binarizer( threshold=0.5 )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'model',
				 'prediction',
		         'train',
		         'project',
		         'score',
		         'analyze',
		         'weights',
		         'scatter_plot',
		         'testing_score',
		         'training_score', ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			np.ndarray - shape (n_features, ) or (n_targets, n_features)
			Estimated coefficients for the linear regression problem.
			If multiple targets are passed during the fit (y 2D),
			this is a 2D array of shape (n_targets, n_features), while if only one target
			is passed, this is a 1D array of length n_features.

		'''
		if self.model.coef_ is None:
			raise AttributeError( 'The weights have not been initialized!!' )
		else:
			return self.model.coef_
	
	@property
	def features_in( self ) -> np.ndarray:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
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
			X_train, X_test, y_train, y_test = split( X, y, test_size=size,
				random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LeastSquares | None:
		"""
		
			Purpose:
			--------
			Train the linear regression model on input features and binary targets.
			
			
			Parameters:
			-----------
			X (np.ndarray | pd.DataFrame):
			Input features.
			y (np.ndarray | pd.Series):
			
			
			
			Returns:
			None
			
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
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
			
		"""
		try:
			throw_if( 'X', X )
			y_pred = self.model.predict( X )
			self.binarizer = Binarizer( threshold=0 )
			_shape = y_pred.reshape( -1, 1 )
			self.prediction = self.binarizer.fit_transform( _shape ).astype( int ).flatten( )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'predict'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			The coefficient of determination is defined as the residual
			sum of squares ((y_true - y_pred)** 2).sum() and the total
			sum of squares ((y_true - y_true.mean()) ** 2).sum().
			The best possible score is 1.0 and it can be negative (because
			the model can be arbitrarily worse). A constant model that
			always predicts the expected value of y, disregarding the
			input features, would get a score of 0.0.
			
			- Training Score
			- Testing Score
			- Weights
			- Features in
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Any:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 10, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
		"""

			Purpose:
			-----------
			Plot scatter diagram for regression predictions.

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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
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
	model: skc.LogisticRegression
	binarizer: Optional[ Binarizer ]
	prediction: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	random_state: Optional[ int ]
	penalty: str
	multi_class: str
	C: float
	max_iter: int
	solver: str
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, C: float=1.0, penalty: str='l2', iters: int=1000,
			multiclass: str='multinomial', solver: str='lbfgs' ) -> None:
		"""

			Purpose:
			--------
			Initialize the Logistic Regression linerar_model.

			Parameters:
			-----------
			iters (int): Maximum number of iterations. Default is 1000.
			solver (str): Algorithm to use in optimization. Default is 'lbfgs'.

		"""
		super( ).__init__( )
		self.binarizer = Binarizer( threshold=0.5 )
		self.C = C
		self.penalty = penalty
		self.max_iter = iters
		self.multi_class = multiclass
		self.solver = solver
		self.model = skc.LogisticRegression( C=self.C, max_iter=self.max_iter,
			multi_class=self.multi_class, solver=self.solver, penalty=self.penalty )
		self.prediction = None
		self.decision = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
		         'penalty',
		         'solver',
		         'multi_class',
		         'random_state',
		         'max_iter',
		         'predict_probabilty',
		         'decision_function',
		         'train',
		         'project',
		         'score',
		         'analyze',
		         'weights',
		         'iterations',
		         'labels',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.model.coef_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.coef_
	
	@property
	def labels( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def iterations( self ) -> int:
		'''

			Returns
			-------
			classes_ is ndarray of shape (n_classes, )
			Actual number of iterations for all classes.
			If binary or multinomial, it returns only 1 element. For liblinear solver,
			only the maximum number of iteration across all classes is given.

		'''
		if self.model.n_iter_ is None:
			raise AttributeError( 'The model has not been trained!' )
		else:
			return self.model.n_iter_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: int=0.2 ) -> ( np.ndarray, np.ndarray, np.ndarray, np.ndarray ):
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
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=42 )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
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
			self.decision = self.model.decision_function( X )
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
			self.probability = self.model.predict_proba( X )
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
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )

	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
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
			_prediction = self.model.predict( X )
			self.binarizer = Binarizer( threshold=0 )
			_shape = _prediction.reshape( -1, 1 )
			self.prediction = self.binarizer.fit_transform( _shape ).astype( int ).flatten( )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'project'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 10, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
	model: skc.RidgeClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	alpha: Optional[ float ]
	solver: Optional[ str ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float=1.0, solver: str='auto', iters: int=1000, rando: int=42 ) -> None:
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
		self.max_iter = iters
		self.random_state = rando
		self.model = skc.RidgeClassifier( alpha=self.alpha, solver=self.solver,
			max_iter=self.max_iter, random_state=self.random_state )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_iter',
				 'random_state',
				 'alpha',
				 'solver',
				 'model',
				 'train',
				 'project',
				 'score',
				 'decision_function',
				 'analyze',
				 'create_heatmap',
				 'weights',
		         'iterations',
		         'features',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.model.coef_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.coef_
		
	@property
	def labels( self ) -> int:
		'''

			Returns
			-------
			classes_ is ndarray of shape (n_classes, )
			Actual number of lab for all classes.
			If binary or multinomial, it returns only 1 element. For liblinear solver,
			only the maximum number of iteration across all classes is given.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def features( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
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
			X_train, X_test, y_train, y_test = split( X, y, test_size=size,
				random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 10, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
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
			self.decision = self.model.decision_function( X )
			return self.decision
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'decision_function( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
			
class Lasso( Classifier ):
	"""
	
		Purpose:
		---------
		Linear Model trained with L1 prior as regularizer
	
		Parameters:
		------------
		alpha (float, optional):
		Threshold above which predictions are considered class 1 (default: 0.1).
		
		Attributes:
		-----------
		model (SklearnLasso):
		Underlying scikit-learn Lasso model.
		threshold (float):
		Threshold for classification decision boundary.
	
	"""
	model: skc.Lasso
	prediction: Optional[ np.ndarray ]
	binarizer: Optional[ Binarizer ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	selection: Optional[ str ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]

	def __init__( self, alpha: float=1.0, iters: int=500,
			rando: int=42, threshold: float=0.5, selection: str='random' ) -> None:
		super( ).__init__( )
		self.alpha = alpha
		self.max_iter = iters
		self.random_state = rando
		self.threshold = threshold
		self.selection = selection
		self.binarizer = Binarizer( threshold=self.threshold )
		self.model = skc.Lasso( alpha=self.alpha, max_iter=self.max_iter,
			random_state=self.random_state, selection=self.selection )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_iter',
				 'random_state',
				 'regularization',
				 'alpha',
		         'selection',
		         'threshold',
				 'model',
				 'weights',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'weights',
		         'iterations',
		         'features',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.model.coef_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.coef_
	
	@property
	def iterations( self ) -> int:
		'''
	
			Returns
			-------
			n_iter_ (int) is ndarray of shape ( n_classes, )
			Represents the number of iterations run by the coordinate descent solver
			to reach the specified tolerance.
	
		'''
		if self.model.n_iter_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		'''
	
			Returns
			-------
			n_features_in_
			The number of features seen during training
	
		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
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
			X_train, X_test, y_train, y_test = split( X, y, test_size=size,
				random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Lasso | None:
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> LassoModel'
			error = ErrorDialog( exception )
			error.show()

	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
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
			_prediction = self.model.predict( X )
			self.binarizer = Binarizer( threshold=0 )
			_shape = _prediction.reshape( -1, 1 )
			self.prediction = self.binarizer.fit_transform( _shape ).astype( int ).flatten( )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}, marker='o' )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
	model: skc.SGDClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_iter: Optional[ int ]
	random_state: Optional[ int ]
	loss: Optional[ str ]
	learning_rate: Optional[ str ]
	average: Optional[ bool ]
	regularization: Optional[ Any ]
	alpha: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, loss: str='hinge', iters: int=100,
			reg: str='l2', alpha: float=0.00001, ave: bool=True, rate: str='optimal' ) -> None:
		"""

			Purpose:
			-----------
			Initialize the SGDClassifier linerar_model.

			Parameters:
			-----------
			loss (str): Loss function to use. Defaults to 'hinge'.
			reg (str): Regularization function to use. Default is 'l2'.
			max (int): Maximum number of passes over the df. Default is 10000.

		"""
		super( ).__init__( )
		self.loss = loss
		self.learning_rate = rate
		self.max_iter = iters
		self.regularization = reg
		self.alpha = alpha
		self.average = ave
		self.model = skc.SGDClassifier( loss=self.loss, max_iter=self.max_iter,
			penalty=self.regularization, alpha=self.alpha,
			average=self.average, learning_rate=self.learning_rate )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_iter',
				 'random_state',
				 'loss',
				 'regularization',
				 'alpha',
		         'average',
				 'model',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'weights',
		         'iterations',
		         'labels',
		         'features',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.model.coef_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.coef_
		
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
		if self.model.n_iter_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_iter_
	
	@property
	def labels( self ) -> int:
		'''

			Returns
			-------
			classes_ is ndarray of shape (n_classes, )
			Actual number of label for all classes.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def features( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
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
			exception.cause = 'GradientDescent'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
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
			exception.cause = 'GradientDescent'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[str,float ]')
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
			self.decision = self.model.decision_function( X )
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
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
	model: skn.KNeighborsClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	n_neighbors: Optional[ int ]
	leaf_size: Optional[ int ]
	power: Optional[ int ]
	algorithm: Any
	metric: str
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, num: int=5, algorithm: str='auto',
			power: int=1, metric: str='minkowski', leafs: int=30 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the KNeighborsClassifier linerar_model.

	
			Parameters:
			---------
			neighbors (int): Number of neighbors to use. Default is 5.

		"""
		super( ).__init__( )
		self.n_neighbors = num
		self.algorithm = algorithm
		self.metric = metric
		self.leaf_size = leafs
		self.power = power
		self.model = skn.KNeighborsClassifier( n_neighbors=self.n_neighbors, p=self.power,
			algorithm=self.algorithm, metric=self.metric, leaf_size=self.leaf_size )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'n_neigbors',
				 'algorithm',
				 'metric',
				 'model',
				 'predict_probabilty',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
				 'labels',
		         'features_in',
		         'samplers',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		'''

			Returns
			-------
			ndarray of shape (n_features,)
			The number of features seen during training.

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	@property
	def samples( self ) -> int:
		'''

			Returns
			-------
			n_samples_fit_ (int)
			The number of samples fit during training.

		'''
		if self.model.n_samples_fit_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_samples_fit_
		
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
			exception.cause = 'NearestNeighbor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
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
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
	model: skd.DecisionTreeClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	min_split: Optional[ float ]
	random_state: Optional[ int ]
	hinge_loss: Optional[ float ]
	classifier: Optional[ Any ]
	splitter: Optional[ str ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, criterion='gini', splitter='best', depth=5, split: float=0.8, rando: int=42 ) -> None:
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
		self.min_split = split
		self.model = skd.DecisionTreeClassifier( criterion=self.criterion,
			splitter=self.splitter, max_depth=self.max_depth,
			min_samples_split=self.min_split, random_state=self.random_state )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
		         'min_split',
				 'random_state',
				 'criterion',
				 'splitter',
				 'model',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'labels',
		         'features_in',
		         'feature_importances',
		         'outputs',
		         'tree',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		'''

			Returns
			-------
			ndarray of shape (n_features,)
			The number of features seen during training.

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	@property
	def feature_importances( self ) -> np.ndarray:
		'''

			Returns
			-------
			ndarray of shape (n_features,)
			The impurity-based feature importances.

		'''
		if self.model.feature_importances_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.feature_importances_
	
	@property
	def outputs( self ) -> int:
		'''

			Returns
			-------
			n_outputs_ (int):
			The number of outputs after training.

		'''
		if self.model.n_outputs_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_outputs_
	
	@property
	def tree( self ) -> int:
		'''

			Returns
			-------
			tree_ (int):
			The underlying Tree object.

		'''
		if self.model.tree_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.tree_
	
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
			exception.cause = 'DecisionTree'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
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
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
	model: ske.RandomForestClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ Any ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, num: int=10, criterion: Any='gini', depth: Any=None, rando: int=42 ) -> None:
		"""

			Purpose:
			-----------
			Initializes the RandomForestClassifier.

		"""
		super( ).__init__( )
		self.n_estimators = num
		self.criterion = criterion
		self.max_depth = depth
		self.random_state = rando
		self.model = ske.RandomForestClassifier( n_estimators=self.n_estimators,
			criterion=self.criterion, max_depth=self.max_depth, random_state=self.random_state )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'n_estimators',
				 'max_depth',
				 'criterior',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'labels',
		         'features_in',
		         'feature_importances',
		         'outputs',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		'''

			Returns
			-------
			ndarray of shape (n_features,)
			The number of features seen during training.

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	@property
	def feature_importances( self ) -> np.ndarray:
		'''

			Returns
			-------
			ndarray of shape (n_features,)
			The impurity-based feature importances.

		'''
		if self.model.feature_importances_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.feature_importances_
	
	@property
	def outputs( self ) -> int:
		'''

			Returns
			-------
			n_outputs_ (int):
			The number of outputs after training.

		'''
		if self.model.n_outputs_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_outputs_
		
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
			exception.cause = 'RandomForest'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
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
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
		split may vary, even with the same training stores and max_features=n_features,
		if the improvement of the criterion is identical for several splits enumerated
		during the search of the best split. To obtain a deterministic behaviour during fitting,
		rando has to be fixed.

	"""
	model: ske.GradientBoostingClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	criterion: Optional[ str ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, lss: str='log_loss', rate: int=0.1, est: int=100,
			depth: int=3, rando: int=42, criterion: str='squared_error' ) -> None:
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
		self.max_depth = depth
		self.random_state = rando
		self.criterion = criterion
		self.model = ske.GradientBoostingClassifier( loss=self.loss,
			learning_rate=self.learning_rate, n_estimators=self.n_estimators,
			max_depth=self.max_depth, random_state=self.random_state, criterion=self.criterion )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'loss',
				 'learning_rate',
				 'n_estimators',
				 'model',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
				 'labels',
		         'features_in',
		         'trees',
		         'feature_importances'
		         'estimators',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def trees( self ) -> int:
		'''

			Returns
			-------
			n_trees_per_iteration_ - (int):
			Number of trees per iteration

		'''
		if self.model.n_trees_per_iteration_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.n_trees_per_iteration_
	
	@property
	def features_in( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	@property
	def feature_importances( self ) -> int:
		'''

			Returns
			-------
			feature_importances_ (int):
			The impurity-based feature importances.

		'''
		if self.model.feature_importances_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.feature_importances_

	@property
	def estimators( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.estimators_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.estimators_
	
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
			X_train, X_test, y_train, y_test = split( X, y, test_size=size,
				random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = ('train( self, X: np.ndarray, y: np.ndarray ) -> '
								'GradientBoost')
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
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
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
	model = ske.AdaBoostClassifier
	prediction: Optional[ np.ndarray ]
	n_estimators: Optional[ int ]
	random_state: Optional[ int ]
	recall: Optional[ float ]
	X_scaled: Optional[ pd.DataFrame ]
	estimator: Optional[ Any ]
	learning_rate: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, num: int=100, learning: float=1.0 ) -> None:
		"""

			Initialize the Random Forest Classifier.

		"""
		super( ).__init__( )
		self.estimator = None
		self.n_estimators = num
		self.learning_rate = learning
		self.model = ske.AdaBoostClassifier( estimator=self.estimator,
			n_estimators=self.n_estimators, learning_rate=self.learning_rate )
		self.X_scaled = None
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'X_scaled',
				 'n_estimators',
				 'learning_rate',
				 'model',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'estimator_errors',
		         'estimator_weights',
		         'labels',
		         'features_in',
		         'importances',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def estimator_errors( self ) -> np.ndarray | None:
		if self.model.estimator_errors_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.estimator_errors_
	   
	@property
	def estimator_weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.model.estimator_weights_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.estimator_weights_

	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	@property
	def estimators( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.estimators_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.estimators_
	
	@property
	def feature_importances( self ) -> int:
		'''

			Returns
			-------
			feature_importances_ (int):
			The impurity-based feature importances.

		'''
		if self.model.feature_importances_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.feature_importances_
	
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
			exception.cause = 'AdaptiveBoost'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
	model: ske.BaggingClassifier
	prediction: Optional[ np.ndarray ]
	max_features: Optional[ int ]
	random_state: Optional[ int ]
	hinge_loss: Optional[ float ]
	base_estimator: Optional[ Any ]
	n_estimators: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, base: object=None, num: int=10, size: int=1, rando: int=42 ) -> None:
		"""

			Initialize the BaggingClassifier.

		"""
		super( ).__init__( )
		self.base_estimator = base
		self.n_estimators = num
		self.max_features = size
		self.random_state = rando
		self.model = ske.BaggingClassifier( estimator=self.base_estimator,
			n_estimators=self.n_estimators, max_features=self.max_features,
			random_state=self.random_state )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.misclass = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'labels',
		         'features',
		         'estimators',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def features( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	@property
	def estimators( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.estimators_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.estimators_
	
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
			exception.cause = 'BaggingModel'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

class VotingModel( Classifier ):
	"""

		Purpose:
		--------
		The idea behind the VotingModel is to combine conceptually different machine rate
		classifiers and use a majority vote or the average predicted probabilities (soft vote)
		to predict the class target_names. Such a classifier can be useful for a set of equally
		well performing model in order to balance out their individual weaknesses.

	"""
	model: ske.VotingClassifier
	prediction: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	estimators: List[ (str, object) ]
	vote: str
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, estimators: List[ ( str, object ) ], vote='hard' ) -> None:
		"""

			Initialize the RandomForestClassifier.

		"""
		super( ).__init__( )
		self.estimators = estimators
		self.voting = vote
		self.model = ske.VotingClassifier( estimators=self.estimators, voting=self.voting )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'model',
				 'prediction',
				 'max_depth',
				 'random_state',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'labels',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def features( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	@property
	def estimators( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.estimators_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.estimators_
	
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
			exception.cause = 'VotingModel'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.misclass = ( y != y_pred ).sum( )
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Mis-Classifications': self.misclass,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

class StackingModel( Classifier ):
	"""

		Purpose:
		-------
		Stack of estimators with a final classifier. Stacked generalization consists in stacking
		the output of individual estimator and use a classifier to compute the final prediction.
		
		Stacking allows to use the strength of each individual estimator by using their output
		as input of a final estimator. Note that estimators_ are fitted on the full X while
		final_estimator_ is trained using cross-validated predictions of the base
		estimators using cross_val_predict.

	"""
	model: ske.StackingClassifier
	estimators: List[ Tuple[ str, ClassifierMixin ] ]
	final_estimator: Optional[ ClassifierMixin ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, est: List[ Tuple[ str, ClassifierMixin ] ], final: ClassifierMixin=None ) -> None:
		"""

			Initialize the RandomForestClassifier.

		"""
		super( ).__init__( )
		self.estimators = est
		self.final_estimator = final
		self.model = ske.StackingClassifier( estimators=self.estimators,
			final_estimator=self.final_estimator )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'final_estimator',
				 'estimators',
				 'model',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'labels',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def features( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	@property
	def estimators( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.estimators_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.estimators_
	
	@property
	def final( self ) -> str:
		'''

			Returns
			-------
			final_estimator
			The classifier fit on the output of estimators_ and responsible for final predictions

		'''
		if self.model.final_estimator_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.final_estimator_
	
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
			exception.cause = 'StackingModel'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.misclass = ( y != y_pred ).sum( )
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Mis-Classifications': self.misclass,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
								'float ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

class SupportVector( Classifier ):
	"""

		Support Vector Classifier (SVC).The implementation is based on libsvm. The fit time scales
		at least quadratically with the number of samples and may be impractical beyond tens of
		thousands of samples.

	"""
	model: skv.SVC
	kernel: Optional[ str ]
	multiclass: Optional[ str ]
	regulation: Optional[ float ]
	penalty: Optional[ str ]
	prediction: Optional[ np.ndarray ]
	misclass: Optional[ float ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self,  C: float=1.0, kernel: str='rbf', degree: int=3 ) -> None:
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
		self.regulation = C
		self.kernel = kernel
		self.degree = degree
		self.model = skv.SVC( C=self.regulation, kernel=self.kernel, degree=self.degree )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'model',
				 'kernel',
				 'regulation',
				 'degree',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'vectors',
		         'weights'
		         'supports',
		         'labels',
		         'iterations',
		         'features',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def vectors( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			np.ndarray - array-like, shape = [n_SV, n_features]

		'''
		if self.model.support_vectors_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.support_vectors_
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.model.coef_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.coef_
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def iterations( self ) -> int:
		'''

			Returns
			-------
			n_iter_ (int) is ndarray of shape ( n_classes, )
			Represents the number of iterations run by the coordinate descent solver
			to reach the specified tolerance.

		'''
		if self.model.n_iter_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	@property
	def supports( self ) -> int:
		'''

			Returns
			-------
			n_support_
			The number of support vectors per class.

		'''
		if self.model.n_support_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_support_
	
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
			exception.cause = 'SupportVector'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = ('train( self, X: np.ndarray, y: np.ndarray ) -> SupportVector')
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
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray )  -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'score( self, X: np.ndarray, y_true: np.ndarray ) -> float '
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'score( self, X: np.ndarray, y_true: np.ndarray ) -> float '
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
	model: snn.MLPClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	hidden_layers: tuple[ int, ... ]
	activation_function: str
	solver: str
	alpha: float
	learning_rate: Any
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, hidden=( 100, ), activation='logistic', solver='lbfgs', alpha=0.0001,
			learning: str='constant', rando: int=42 ) -> None:
		super( ).__init__( )
		self.hidden_layers = hidden
		self.activation_function = activation
		self.learning_rate = learning
		self.solver = solver
		self.alpha = alpha
		self.random_state = rando
		self.model = snn.MLPClassifier( hidden_layer_sizes=self.hidden_layers,
			activation=self.activation_function, solver=self.solver, alpha=self.alpha,
			learning_rate=self.learning_rate, random_state=self.random_state )
		self.prediction = None
		self.probability = None
		self.precision = 0.0
		self.misclass = 0.0
		self.balanced_accuracy = 0.0
		self.accuracy = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'model',
		         'hidden_layers',
		         'activation_function',
		         'learning_rate',
		         'solver',
		         'alpha',
				 'prediction',
		         'probability',
		         'max_depth',
		         'random_state',
		         'train',
		         'project',
		         'score',
		         'analyze',
		         'create_heatmap',
		         'predict_probability',
		         'weights',
		         'labels',
		         'loss',
		         'outputs',
		         'layers'
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def loss( self ) -> float:
		if self.model.loss_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.loss_
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.classes_
	
	@property
	def weights( self ) -> np.ndarray:
		'''

			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

		'''
		if self.model.coefs_ is None:
			raise AttributeError( 'The weights have not been initialized!' )
		else:
			return self.model.coefs_
	
	
	@property
	def layers( self ) -> np.ndarray:
		'''

			Returns
			-------
			n_layer_ (int)
			Number of layers.

		'''
		if self.model.n_layers_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.n_layers_
	
	@property
	def outputs( self ) -> np.ndarray:
		'''

			Returns
			-------
			n_layer_ (int)
			Number of outputs.

		'''
		if self.model.n_outputs_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.n_outputs_
	
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
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
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
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute the classification accuracy of the model.
			
			F1-Score - F1 Score
			Precision - Prescision Score
			Accuracy - Accuracy Score
			Recall - Recall Score
			
			
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
			self.precision = precision_score( y, y_pred, average=None )
			self.accuracy = accuracy_score( y, y_pred )
			self.recall = recall_score( y, y_pred, average=None )
			self.balanced_accuracy = balanced_accuracy_score( y, y_pred )
			self.f1_score = f1_score( y, y_pred, average=None )
			_metrics = \
			{
				'Training Score': self.training_score,
	            'Testing Score': self.testing_score,
				'Precision Score': self.precision,
				'Accuracy Score': self.accuracy,
				'Recall Score': self.recall,
				'Balanced Accuracy': self.balanced_accuracy,
				'F Score': self.f1_score,
			}
			_dataframe = pd.DataFrame( _metrics )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
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
			- Mean Squared Error (float)
			- Root Mean Squared Error (float)
			- Mean Absolute Error (float)
			- Median Absolute Error (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			data = pd.DataFrame( X )
			corr = data.corr( method='pearson' )
			plt.figure( figsize=( 8, 6 ) )
			sns.heatmap( corr, fmt='.1%', cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ):
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
			_mrk = ( 'o', 's', '^', 'v', '<' )
			_clr = ( 'red', 'blue', 'lightgreen', 'gray', 'cyan' )
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'} )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )



