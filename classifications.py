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
from sklearn.metrics import (recall_score, precision_score, confusion_matrix, classification_report,
                             auc, roc_curve, average_precision_score, balanced_accuracy_score,
                             ConfusionMatrixDisplay, accuracy_score, top_k_accuracy_score, f1_score,
                             hinge_loss, log_loss, mean_squared_error, root_mean_squared_error,
                             mean_absolute_error, median_absolute_error)
from sklearn.preprocessing import Binarizer
from boogr import Error

def throw_if( name: str, value: object ):
	if value is None:
		raise Exception( f'Argument "{name}" cannot be empty!' )


class Classifier( ):
	"""

		Purpose:
		---------
		Abstract base class that defines the interface for all classification wrappers.

	"""
	max_iter: Optional[ int ]
	random_state: Optional[ int ]
	learning_rate: Optional[ Any ]
	binarizer: Optional[ Binarizer ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	misclass: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	markers: Optional[ List[ str ] ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize shared wrapper state.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
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
		self.binarizer = None
		self.prediction = None
		self.probability = None
		self.decision = None
		self.misclass = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.balanced_accuracy = 0.0
		self.f1_score = 0.0
		self.training_score = None
		self.testing_score = None
		self.classification_report = None
		self.confusion_matrix_values = None
	
	def split_data( self, X: np.ndarray,
			y: np.ndarray ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ] | None:
		"""

			Purpose:
			---------
			Split feature and target arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			--------
			tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ] | None

		"""
		raise NotImplementedError
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
			Fit the classifier to the training data.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			--------
			object | None

		"""
		raise NotImplementedError
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Generate predicted class labels from the trained classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for
				signature consistency.

			Returns:
			--------
			np.ndarray

		"""
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Compute scalar summary metrics for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame | None

		"""
		raise NotImplementedError
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Perform tabular or diagnostic analysis for the classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame | None

		"""
		raise NotImplementedError
	
	def classification_scores( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Compute stable scalar classification metrics for UI display using
			the supplied evaluation data.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.testing_score = float( self.model.score( X, y ) )
			
			if self.training_score is None:
				self.training_score = self.testing_score
			
			self.misclass = float( np.sum( y != y_pred ) )
			self.precision = float(
				precision_score( y, y_pred, average='weighted', zero_division=0 )
			)
			self.accuracy = float( accuracy_score( y, y_pred ) )
			self.recall = float(
				recall_score( y, y_pred, average='weighted', zero_division=0 )
			)
			self.balanced_accuracy = float( balanced_accuracy_score( y, y_pred ) )
			self.f1_score = float(
				f1_score( y, y_pred, average='weighted', zero_division=0 )
			)
			self.classification_report = classification_report(
				y,
				y_pred,
				output_dict=True,
				zero_division=0
			)
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			
			_metrics = {
					'Training Score': self.training_score,
					'Testing Score': self.testing_score,
					'Mis-Classifications': self.misclass,
					'Precision Score': self.precision,
					'Accuracy Score': self.accuracy,
					'Recall Score': self.recall,
					'Balanced Accuracy': self.balanced_accuracy,
					'F Score': self.f1_score,
			}
			
			return pd.DataFrame( [ _metrics ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Classifier'
			exception.method = 'classification_scores( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def correlation_heatmap( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Render a correlation heatmap for the supplied feature matrix.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			df_data = pd.DataFrame( X )
			df_corr = df_data.corr( method='pearson' )
			plt.figure( figsize=(8, 6) )
			sns.heatmap( df_corr, cmap='coolwarm', annot=True )
			plt.tight_layout( )
			plt.title( 'Correlation Matrix' )
			plt.grid( False )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Classifier'
			exception.method = 'correlation_heatmap( self, X: np.ndarray ) -> None'
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Render a confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( self.confusion_matrix_values ).plot( values_format='d' )
			plt.title( 'Confusion Matrix' )
			plt.grid( False )
			plt.tight_layout( )
			return self.confusion_matrix_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Classifier'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	_classification_scores = classification_scores
	_correlation_heatmap = correlation_heatmap


class Perceptron( Classifier ):
	"""

		Purpose:
		---------
		Perceptron functionality behind the Mathy
		classification interface used by the surrounding module.

	"""
	model: skc.Perceptron
	binarizer: Optional[ Binarizer ]
	prediction: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	random_state: Optional[ int ]
	alpha: Optional[ float ]
	max_iter: Optional[ int ]
	shuffle: Optional[ bool ]
	penalty: Optional[ str ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float=0.001, eta: float=1.0, iters: int=1000,
			shuffle: bool=False, penalty: Optional[ str ] = None,
			random: int=42 ) -> None:
		"""

			Purpose:
			---------
			Initialize the Perceptron classifier wrapper.

			Parameters:
			-----------
			alpha (float): Regularization strength.
			eta (float): Initial learning-rate value passed as eta0.
			iters (int): Maximum number of iterations.
			shuffle (bool): Whether to shuffle the training data after each epoch.
			penalty (Optional[ str ]): Penalty term applied during fitting.
			random (int): Random seed.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.binarizer = Binarizer( threshold=0.5 )
		self.alpha = alpha
		self.max_iter = iters
		self.shuffle = shuffle
		self.penalty = penalty
		self.learning_rate = eta
		self.random_state = random
		self.validate_configuration( )
		self.model = skc.Perceptron(
			alpha=self.alpha,
			max_iter=self.max_iter,
			shuffle=self.shuffle,
			eta0=self.learning_rate,
			penalty=self.penalty,
			random_state=self.random_state
		)
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
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
		         'confusion_matrix',
		         'scatter_plot',
		         'region_plot',
		         'weights',
		         'decision_function',
		         'iterations',
		         'labels',
		         'testing_score',
		         'training_score' ]
	
	def validate_configuration( self ) -> None:
		"""

			Purpose:
			---------
			Validate wrapper configuration before estimator construction.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		try:
			_valid_penalties = { None, 'l2', 'l1', 'elasticnet' }
			
			if self.penalty not in _valid_penalties:
				raise ValueError( f'Unsupported penalty: {self.penalty}' )
			
			if self.alpha is None or self.alpha < 0.0:
				raise ValueError( 'Argument "alpha" must be greater than or equal to zero.' )
			
			if self.learning_rate is None or self.learning_rate <= 0.0:
				raise ValueError( 'Argument "eta" must be greater than zero.' )
			
			if self.max_iter is None or self.max_iter < 1:
				raise ValueError( 'Argument "iters" must be greater than zero.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'validate_configuration( self ) -> None'
			raise exception
	
	@property
	def weights( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted feature weights.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The Perceptron data is untrained.' )
		return self.model.coef_
	
	@property
	def iterations( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the number of completed fitting iterations.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float=0.2, random: int=42 ) -> Tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			---------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return split( X, y, test_size=size, random_state=random )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = (
					'split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, '
					'random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]'
			)
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Compute decision scores for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Perceptron | None:
		"""

			Purpose:
			---------
			Fit the Perceptron classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			--------
			Perceptron | None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.training_score = float( self.model.score( X, y ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Perceptron | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict class labels for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for
				signature consistency.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Return tabular classifier metrics for analysis.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			df_metrics = self.score( X, y )
			self.testing_score = float( self.model.score( X, y ) )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Render a confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( self.confusion_matrix_values ).plot( values_format='d' )
			plt.title( 'Confusion Matrix' )
			plt.grid( False )
			plt.tight_layout( )
			return self.confusion_matrix_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot(
				x=y,
				y=y_pred,
				scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' }
			)
			plt.plot(
				[ y.min( ), y.max( ) ],
				[ y.min( ), y.max( ) ],
				'k--',
				label='Perfect Prediction'
			)
			plt.text(
				x=y.min( ),
				y=y.max( ) * 0.95,
				s=_text,
				fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 )
			)
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
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception


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
	model: skc.SGDClassifier
	binarizer: Optional[ Binarizer ]
	prediction: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	random_state: Optional[ int ]
	alpha: Optional[ float ]
	max_iter: Optional[ int ]
	shuffle: Optional[ bool ]
	penalty: Optional[ str ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float=0.0001, eta: float=0.01, iters: int=1000,
			shuffle: bool=False, penalty: Optional[ str ] = 'l2',
			random: int=42 ) -> None:
		"""

			Purpose:
			---------
			Initialize the Least Squares classification wrapper.

			Parameters:
			-----------
			alpha (float): Regularization strength.
			eta (float): Initial learning-rate value passed as eta0.
			iters (int): Maximum number of iterations.
			shuffle (bool): Whether to shuffle the training data after each epoch.
			penalty (Optional[ str ]): Penalty term applied during fitting.
			random (int): Random seed.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.binarizer = Binarizer( threshold=0.5 )
		self.alpha = alpha
		self.max_iter = iters
		self.shuffle = shuffle
		self.penalty = penalty
		self.learning_rate = eta
		self.random_state = random
		self.validate_configuration( )
		self.model = skc.SGDClassifier( loss='perceptron', alpha=self.alpha, max_iter=self.max_iter,
			shuffle=self.shuffle, eta0=self.learning_rate, learning_rate='constant',
			penalty=self.penalty, random_state=self.random_state )
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
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
		         'confusion_matrix',
		         'scatter_plot',
		         'weights',
		         'decision_function',
		         'iterations',
		         'labels',
		         'testing_score',
		         'training_score' ]
	
	def validate_configuration( self ) -> None:
		"""

			Purpose:
			---------
			Validate wrapper configuration before estimator construction.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		try:
			_valid_penalties = { None, 'l2', 'l1', 'elasticnet' }
			
			if self.penalty not in _valid_penalties:
				raise ValueError( f'Unsupported penalty: {self.penalty}' )
			
			if self.alpha is None or self.alpha < 0.0:
				raise ValueError( 'Argument "alpha" must be greater than or equal to zero.' )
			
			if self.learning_rate is None or self.learning_rate <= 0.0:
				raise ValueError( 'Argument "eta" must be greater than zero.' )
			
			if self.max_iter is None or self.max_iter < 1:
				raise ValueError( 'Argument "iters" must be greater than zero.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'validate_configuration( self ) -> None'
			raise exception
	
	@property
	def weights( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted feature weights.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def iterations( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the number of completed fitting iterations.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float=0.2, random: int=42 ) -> Tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			---------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return split( X, y, test_size=size, random_state=random )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = (
					'split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, '
					'random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]'
			)
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Compute decision scores for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.decision = self.model.decision_function( X )
			return self.decision
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'decision_function( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LeastSquares | None:
		"""

			Purpose:
			---------
			Fit the Least Squares classification wrapper.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			--------
			LeastSquares | None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.training_score = float( self.model.score( X, y ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> LeastSquares | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict class labels for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for
				signature consistency.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Return tabular classifier metrics for analysis.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			df_metrics = self.score( X, y )
			self.testing_score = float( self.model.score( X, y ) )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Render a confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( self.confusion_matrix_values ).plot( values_format='d' )
			plt.title( 'Confusion Matrix' )
			plt.grid( False )
			plt.tight_layout( )
			return self.confusion_matrix_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
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
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception


class LogisticRegression( Classifier ):
	"""

		Purpose:
		--------
		Wrap scikit-learn LogisticRegression behind the Mathy classification
		interface used by the Streamlit application. This implementation
		supports binary and multiclass classification, tabular metric analysis,
		confusion matrix rendering, and binary ROC visualization.

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
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	
	def __init__( self, C: float=1.0, penalty: str='l2', iters: int=1000,
			multiclass: str='multinomial', solver: str='lbfgs',
			random: int=42 ) -> None:
		"""

			Purpose:
			--------
			Initialize the Logistic Regression classifier.

			Parameters:
			-----------
			C (float): Inverse regularization strength.
			penalty (str): Penalty norm retained for wrapper compatibility.
			iters (int): Maximum number of optimization iterations.
			multiclass (str): Requested multiclass mode retained for wrapper
				compatibility.
			solver (str): Optimization solver.
			random (int): Random seed.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.C = C
		self.random_state = random
		self.penalty = penalty
		self.max_iter = iters
		self.multi_class = multiclass
		self.solver = solver
		self.validate_configuration( )
		self.model = skc.LogisticRegression( C=self.C, max_iter=self.max_iter,
			solver=self.solver, random_state=self.random_state )
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			--------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'prediction',
		         'penalty',
		         'solver',
		         'multi_class',
		         'random_state',
		         'max_iter',
		         'predict_probability',
		         'decision_function',
		         'train',
		         'project',
		         'score',
		         'analyze',
		         'confusion_matrix',
		         'roc_curve',
		         'weights',
		         'iterations',
		         'labels',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score' ]
	
	def validate_configuration( self ) -> None:
		"""

			Purpose:
			--------
			Validate wrapper-level settings before estimator construction.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		try:
			supported_solvers = { 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag',
			                      'saga' }
			
			if self.solver not in supported_solvers:
				raise ValueError( f'Unsupported solver: {self.solver}' )
			
			if self.penalty not in { 'l1', 'l2', 'elasticnet', 'none', None }:
				raise ValueError( f'Unsupported penalty setting: {self.penalty}' )
			
			if self.penalty == 'l1' and self.solver not in { 'liblinear', 'saga' }:
				raise ValueError( 'penalty="l1" requires solver "liblinear" or "saga".' )
			
			if self.penalty == 'elasticnet' and self.solver != 'saga':
				raise ValueError( 'penalty="elasticnet" requires solver "saga".' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'validate_configuration( self ) -> None'
			raise exception
	
	@property
	def weights( self ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return fitted coefficient weights.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def iterations( self ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return the number of completed optimization iterations.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_iter_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			--------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size,
				random_state=random, )
			return X_train, X_test, y_train, y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = (
					'split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, '
					'random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]'
			)
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Compute confidence scores for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class-probability estimates for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LogisticRegression | None:
		"""

			Purpose:
			--------
			Fit the logistic regression model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			LogisticRegression | None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.training_score = float( self.model.score( X, y ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> LogisticRegression | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels from input features.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for
				signature consistency.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): True class labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Return tabular classifier metrics for Streamlit display.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			df_metrics = self.score( X, y )
			self.testing_score = float( self.model.score( X, y ) )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Render a confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( self.confusion_matrix_values ).plot( values_format='d' )
			plt.title( 'Confusion Matrix' )
			plt.grid( False )
			plt.tight_layout( )
			return self.confusion_matrix_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""

			Purpose:
			--------
			Render a binary ROC curve using predicted class probabilities.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth binary labels.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, float ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			classes = np.unique( y )
			if len( classes ) != 2:
				raise ValueError( 'roc_curve is supported only for binary classification.' )
			
			probability = self.predict_probability( X )
			fpr, tpr, _ = roc_curve( y, probability[ :, 1 ], pos_label=classes[ 1 ] )
			roc_auc = auc( fpr, tpr )
			
			plt.plot( fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})' )
			plt.plot( [ 0, 1 ], [ 0, 1 ], linestyle='--', label='Random Guess' )
			plt.xlim( [ -0.01, 1.01 ] )
			plt.ylim( [ -0.01, 1.01 ] )
			plt.xlabel( 'False Positive Rate' )
			plt.ylabel( 'True Positive Rate' )
			plt.title( 'Receiver Operating Characteristic' )
			plt.legend( loc='lower right' )
			plt.grid( visible=True )
			plt.tight_layout( )
			return fpr, tpr, roc_auc
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot(
				x=y,
				y=y_pred,
				scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' }
			)
			plt.plot(
				[ y.min( ), y.max( ) ],
				[ y.min( ), y.max( ) ],
				'k--',
				label='Perfect Prediction'
			)
			plt.text(
				x=y.min( ),
				y=y.max( ) * 0.95,
				s=_text,
				fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 )
			)
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
			raise exception
			
	
class Ridge( Classifier ):
	"""

		Purpose:
		--------
		A classifier that first converts binary targets to {-1, 1} and then treats the problem as a
		regression task, optimizing the same objective as above. The predicted class corresponds
		to the sign of the regressor’s prediction. For multiclass classification, the problem is
		treated as multi-output regression, and the predicted class corresponds to the output
		with the highest value. It might seem questionable to use a (penalized) Least Squares loss to fit a classification
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
	random_state: Optional[ int ]
	alpha: Optional[ float ]
	solver: Optional[ str ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
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
			--------
			Initialize the Ridge classifier.

			Parameters:
			-----------
			alpha (float): Regularization strength.
			solver (str): Solver used by RidgeClassifier.
			iters (int): Maximum number of iterations where supported by the solver.
			rando (int): Random seed.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.alpha = alpha
		self.solver = solver
		self.max_iter = iters
		self.random_state = rando
		self.model = skc.RidgeClassifier(
			alpha=self.alpha,
			solver=self.solver,
			max_iter=self.max_iter,
			random_state=self.random_state
		)
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			--------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
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
		         'weights',
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
		"""

			Purpose:
			--------
			Return fitted coefficient weights.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if self.model.coef_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if self.model.classes_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			--------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, 
			random: int=42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		"""

			Purpose:
			--------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			tuple

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split(
				X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Ridge | None:
		"""

			Purpose:
			--------
			Fit the Ridge classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			Ridge | None

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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for signature consistency.

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): True class labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Render a correlation heatmap for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Compute decision scores for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ np.min( y ), np.max( y ) ], [ np.min( y ), np.max( y ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=np.min( y ), y=np.max( y ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
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
			raise exception
			
			
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
		model (Lasso):
		Underlying scikit-learn Lasso model.
		threshold (float):
		Threshold for classification decision boundary.
	
	"""
	model: skc.Lasso
	prediction: Optional[ np.ndarray ]
	binarizer: Optional[ Binarizer ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	random_state: Optional[ int ]
	selection: Optional[ str ]
	alpha: Optional[ float ]
	threshold: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float=1.0, iters: int=500, rando: int=42, threshold: float=0.5, 
			selection: str='random' ) -> None:
		"""

			Purpose:
			---------
			Initialize the thresholded Lasso wrapper.

			Parameters:
			-----------
			alpha (float): Regularization strength.
			iters (int): Maximum number of coordinate-descent iterations.
			rando (int): Random seed.
			threshold (float): Threshold used to convert regression output to class labels.
			selection (str): Coordinate update order.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.alpha = alpha
		self.max_iter = iters
		self.random_state = rando
		self.threshold = threshold
		self.selection = selection
		self.binarizer = Binarizer( threshold=self.threshold )
		self.model = skc.Lasso( alpha=self.alpha, max_iter=self.max_iter,
			random_state=self.random_state, selection=self.selection )
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'prediction',
		         'max_iter',
		         'random_state',
		         'alpha',
		         'selection',
		         'threshold',
		         'model',
		         'weights',
		         'train',
		         'project',
		         'score',
		         'analyze',
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
		"""

			Purpose:
			---------
			Return fitted regression coefficients.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if self.model.coef_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def iterations( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of coordinate-descent iterations used during fitting.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if self.model.n_iter_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, 
			random: int=42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		"""

			Purpose:
			---------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			tuple

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Lasso | None:
		"""

			Purpose:
			---------
			Fit the Lasso regression model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Binary target vector encoded numerically.

			Returns:
			--------
			Lasso | None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Lasso'
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Convert continuous regression predictions to binary class labels.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for signature consistency.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			_prediction = self.model.predict( X )
			_shape = _prediction.reshape( -1, 1 )
			self.prediction = self.binarizer.fit_transform( _shape ).astype( int ).flatten( )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Compute scalar summary classification metrics for thresholded predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): True binary class labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Render a correlation heatmap for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.
			y (np.ndarray): Ground-truth binary labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Plot observed labels against continuous regression estimates.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): True class labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={
					'color': 'red' }, marker='o' )
			plt.plot( [ np.min( y ), np.max( y ) ], [ np.min( y ), np.max( y ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=np.min( y ), y=np.max( y ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
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
			raise exception
			

class GradientDescent( Classifier ):
	"""

		Purpose:
		--------
		Linear classifiers (SVM, logistic regression, etc.) with Stochastic Gradient Descent (SGD)
		training.  This estimator implements regularized linear models with stochastic
		gradient descent learning:
		
		The gradient of the loss is estimated each sample at a time and the model is updated along
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
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, loss: str='hinge', iters: int=100, reg: str='l2', alpha: float=0.00001,
			ave: bool=True, rate: str='optimal' ) -> None:
		"""

			Purpose:
			---------
			Initialize the SGD classifier wrapper.

			Parameters:
			-----------
			loss (str): Loss function used by SGDClassifier.
			iters (int): Maximum number of iterations.
			reg (str): Regularization penalty.
			alpha (float): Regularization strength.
			ave (bool): Whether to use averaged SGD weights.
			rate (str): Learning-rate schedule.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.loss = loss
		self.learning_rate = rate
		self.max_iter = iters
		self.regularization = reg
		self.alpha = alpha
		self.average = ave
		self.model = skc.SGDClassifier( loss=self.loss, max_iter=self.max_iter, 
			penalty=self.regularization, alpha=self.alpha, average=self.average,
			learning_rate=self.learning_rate )
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
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
		         'weights',
		         'iterations',
		         'labels',
		         'features',
		         'decision_function',
		         'predict_probability',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def weights( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted linear coefficients.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if self.model.coef_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def iterations( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the number of completed fitting iterations.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if self.model.n_iter_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if self.model.classes_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> (np.ndarray, np.ndarray, np.ndarray,  np.ndarray):
		"""

			Purpose:
			---------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Target vector.
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			tuple

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientDescent | None:
		"""

			Purpose:
			---------
			Fit the SGD classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Target labels.

			Returns:
			--------
			GradientDescent | None

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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict class labels for the provided features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for signature consistency.

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Render a correlation heatmap for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Compute decision scores for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class-probability estimates when the selected loss supports them.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			if self.loss not in ('log_loss', 'modified_huber'):
				raise ValueError(
					'predict_probability requires loss="log_loss" or loss="modified_huber".'
				)
			
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): True class labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ np.min( y ), np.max( y ) ], [ np.min( y ), np.max( y ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=np.min( y ), y=np.max( y ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
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
			raise exception
			

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
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	
	def __init__( self, num: int=5, algorithm: str='auto',
			power: int=2, metric: str='minkowski', leafs: int=30 ) -> None:
		"""

			Purpose:
			--------
			Initialize the K-Nearest Neighbors classifier.

			Parameters:
			-----------
			num (int): Number of neighbors.
			algorithm (str): Neighbor search algorithm.
			power (int): Power parameter for the Minkowski metric.
			metric (str): Distance metric.
			leafs (int): Leaf size for BallTree or KDTree.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.n_neighbors = num
		self.algorithm = algorithm
		self.power = power
		self.metric = metric
		self.leaf_size = leafs
		self.validate_configuration( )
		self.model = skn.KNeighborsClassifier( n_neighbors=self.n_neighbors, algorithm=self.algorithm,
			p=self.power, metric=self.metric, leaf_size=self.leaf_size )
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			--------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'model',
		         'prediction',
		         'probability',
		         'n_neighbors',
		         'leaf_size',
		         'power',
		         'algorithm',
		         'metric',
		         'train',
		         'project',
		         'predict_probability',
		         'score',
		         'analyze',
		         'confusion_matrix',
		         'roc_curve',
		         'scatter_plot',
		         'labels',
		         'features_in',
		         'samples',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score' ]
	
	def validate_configuration( self ) -> None:
		"""

			Purpose:
			--------
			Validate wrapper configuration before estimator construction.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		try:
			_valid_algorithms = { 'auto', 'ball_tree', 'kd_tree', 'brute' }
			_valid_metrics = { 'minkowski', 'euclidean', 'manhattan', 'chebyshev', 'hamming',
					'canberra', 'braycurtis', 'cityblock', 'cosine', 'l1', 'l2',
					'nan_euclidean', 'mahalanobis', 'seuclidean' }
			if self.algorithm not in _valid_algorithms:
				raise ValueError( f'Unsupported algorithm: {self.algorithm}' )
			
			if self.metric not in _valid_metrics:
				raise ValueError( f'Unsupported metric: {self.metric}' )
			
			if self.n_neighbors is None or self.n_neighbors < 1:
				raise ValueError( 'Argument "num" must be greater than zero.' )
			
			if self.leaf_size is None or self.leaf_size < 1:
				raise ValueError( 'Argument "leafs" must be greater than zero.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'validate_configuration( self ) -> None'
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		"""

			Purpose:
			--------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def samples( self ) -> int:
		"""

			Purpose:
			--------
			Return the number of samples fit during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if not hasattr( self.model, 'n_samples_fit_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_samples_fit_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			--------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random, )
			return X_train, X_test, y_train, y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = (
					'split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, '
					'random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]'
			)
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> NearestNeighbor | None:
		"""

			Purpose:
			--------
			Fit the K-Nearest Neighbors classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			NearestNeighbor | None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.training_score = float( self.model.score( X, y ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> NearestNeighbor | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels using the K-Nearest Neighbors classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for
				signature consistency.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray'
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class-probability estimates for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): True class labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Return tabular classifier metrics for Streamlit display.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			df_metrics = self.score( X, y )
			self.testing_score = float( self.model.score( X, y ) )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Render a confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( self.confusion_matrix_values ).plot( values_format='d' )
			plt.title( 'Confusion Matrix' )
			plt.grid( False )
			plt.tight_layout( )
			return self.confusion_matrix_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""

			Purpose:
			--------
			Render a binary ROC curve using predicted class probabilities.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth binary labels.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, float ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			classes = np.unique( y )
			if len( classes ) != 2:
				raise ValueError( 'roc_curve is supported only for binary classification.' )
			
			probability = self.predict_probability( X )
			fpr, tpr, _ = roc_curve( y, probability[ :, 1 ], pos_label=classes[ 1 ] )
			roc_auc = auc( fpr, tpr )
			
			plt.plot( fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})' )
			plt.plot( [ 0, 1 ], [ 0, 1 ], linestyle='--', label='Random Guess' )
			plt.xlim( [ -0.01, 1.01 ] )
			plt.ylim( [ -0.01, 1.01 ] )
			plt.xlabel( 'False Positive Rate' )
			plt.ylabel( 'True Positive Rate' )
			plt.title( 'Receiver Operating Characteristic' )
			plt.legend( loc='lower right' )
			plt.grid( visible=True )
			plt.tight_layout( )
			return fpr, tpr, roc_auc
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_trn = self.training_score if self.training_score is not None else 0.0
			_tst = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot(
				x=y,
				y=y_pred,
				scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' }
			)
			plt.plot(
				[ np.min( y ), np.max( y ) ],
				[ np.min( y ), np.max( y ) ],
				'k--',
				label='Perfect Prediction'
			)
			plt.text(
				x=np.min( y ),
				y=np.max( y ) * 0.95,
				s=_text,
				fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 )
			)
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
			raise exception


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
	random_state: Optional[ int ]
	criterion: Optional[ str ]
	splitter: Optional[ str ]
	max_depth: Optional[ int ]
	min_samples_split: Optional[ int ]
	min_samples_leaf: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	
	def __init__( self, criterion: str='gini', splitter: str='best',
			depth: Optional[ int ] = None, min_split: int=2,
			min_leaf: int=1, random: int=42 ) -> None:
		"""

			Purpose:
			---------
			Initialize the Decision Tree classifier wrapper.

			Parameters:
			-----------
			criterion (str): Split quality criterion.
			splitter (str): Strategy used to choose the split at each node.
			depth (Optional[ int ]): Maximum tree depth.
			min_split (int): Minimum number of samples required to split an internal node.
			min_leaf (int): Minimum number of samples required at a leaf node.
			random (int): Random seed.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.criterion = criterion
		self.splitter = splitter
		self.max_depth = depth
		self.min_samples_split = min_split
		self.min_samples_leaf = min_leaf
		self.random_state = random
		self.validate_configuration( )
		self.model = skd.DecisionTreeClassifier( criterion=self.criterion, splitter=self.splitter,
			max_depth=self.max_depth,  min_samples_split=self.min_samples_split,
			min_samples_leaf=self.min_samples_leaf, random_state=self.random_state )
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'prediction',
		         'probability',
		         'random_state',
		         'criterion',
		         'splitter',
		         'max_depth',
		         'min_samples_split',
		         'min_samples_leaf',
		         'model',
		         'train',
		         'project',
		         'predict_probability',
		         'score',
		         'analyze',
		         'confusion_matrix',
		         'roc_curve',
		         'scatter_plot',
		         'labels',
		         'features_in',
		         'feature_importances',
		         'classes_count',
		         'outputs',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score' ]
	
	def validate_configuration( self ) -> None:
		"""

			Purpose:
			---------
			Validate wrapper configuration before estimator construction.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		try:
			_valid_criteria = { 'gini', 'entropy', 'log_loss' }
			_valid_splitters = { 'best', 'random' }
			
			if self.criterion not in _valid_criteria:
				raise ValueError( f'Unsupported criterion: {self.criterion}' )
			
			if self.splitter not in _valid_splitters:
				raise ValueError( f'Unsupported splitter: {self.splitter}' )
			
			if self.min_samples_split is None or self.min_samples_split < 2:
				raise ValueError( 'Argument "min_split" must be greater than or equal to two.' )
			
			if self.min_samples_leaf is None or self.min_samples_leaf < 1:
				raise ValueError( 'Argument "min_leaf" must be greater than or equal to one.' )
			
			if self.max_depth is not None and self.max_depth < 1:
				raise ValueError( 'Argument "depth" must be greater than zero when specified.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'validate_configuration( self ) -> None'
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def feature_importances( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return impurity-based feature importances.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'feature_importances_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.feature_importances_
	
	@property
	def classes_count( self ) -> np.ndarray | int:
		"""

			Purpose:
			---------
			Return the number of classes after training.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray | int

		"""
		if not hasattr( self.model, 'n_classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_classes_
	
	@property
	def outputs( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of outputs after training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if not hasattr( self.model, 'n_outputs_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_outputs_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float=0.2, random: int=42 ) -> Tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			---------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return split( X, y, test_size=size, random_state=random )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = (
					'split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, '
					'random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]'
			)
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> DecisionTree | None:
		"""

			Purpose:
			---------
			Fit the Decision Tree classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			DecisionTree | None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.training_score = float( self.model.score( X, y ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> DecisionTree | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict class labels for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for
				signature consistency.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class-probability estimates for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Return tabular classifier metrics for analysis.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			df_metrics = self.score( X, y )
			self.testing_score = float( self.model.score( X, y ) )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Render a confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( self.confusion_matrix_values ).plot( values_format='d' )
			plt.title( 'Confusion Matrix' )
			plt.grid( False )
			plt.tight_layout( )
			return self.confusion_matrix_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""

			Purpose:
			---------
			Render a binary ROC curve using predicted class probabilities.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth binary labels.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, float ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			classes = np.unique( y )
			if len( classes ) != 2:
				raise ValueError( 'roc_curve is supported only for binary classification.' )
			
			probability = self.predict_probability( X )
			fpr, tpr, _ = roc_curve( y, probability[ :, 1 ], pos_label=classes[ 1 ] )
			roc_auc = auc( fpr, tpr )
			
			plt.plot( fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})' )
			plt.plot( [ 0, 1 ], [ 0, 1 ], linestyle='--', label='Random Guess' )
			plt.xlim( [ -0.01, 1.01 ] )
			plt.ylim( [ -0.01, 1.01 ] )
			plt.xlabel( 'False Positive Rate' )
			plt.ylabel( 'True Positive Rate' )
			plt.title( 'Receiver Operating Characteristic' )
			plt.legend( loc='lower right' )
			plt.grid( visible=True )
			plt.tight_layout( )
			return fpr, tpr, roc_auc
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot(
				x=y,
				y=y_pred,
				scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' }
			)
			plt.plot(
				[ y.min( ), y.max( ) ],
				[ y.min( ), y.max( ) ],
				'k--',
				label='Perfect Prediction'
			)
			plt.text(
				x=y.min( ),
				y=y.max( ) * 0.95,
				s=_text,
				fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 )
			)
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
			raise exception
			

class RandomForest( Classifier ):
	"""

		Purpose:
		--------
		In random forests, each tree in the ensemble is built from a sample drawn with replacement
		(i.e., a bootstrap sample) from the training set.

		Furthermore, when plitting each node during the construction of a tree,
		the best split is found either from all input feature_names or a random subset of
		size max_features.s

		The purpose of these two sources of randomness is to decrease the variance
		of the forest estimator. Individual decision trees typically exhibit high variance
		and tend to overfit. The injected randomness in forests yield decision trees with
		decoupled prediction errors. By taking an average of those predictions,
		errors can cancel out. Random forests achieve a reduced variance
		by combining diverse trees, sometimes at the cost of a slight increase in bias.
		The variance reduction is often significant hence yielding an overall better model.

	"""
	model: ske.RandomForestClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	random_state: Optional[ int ]
	n_estimators: Optional[ int ]
	max_depth: Optional[ int ]
	criterion: Optional[ str ]
	n_jobs: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	
	def __init__( self, estimators: int=100, depth: Optional[ int ] = None,
			criterion: str='gini', jobs: int=-1, random: int=42 ) -> None:
		"""

			Purpose:
			---------
			Initialize the Random Forest classifier wrapper.

			Parameters:
			-----------
			estimators (int): Number of trees in the forest.
			depth (Optional[ int ]): Maximum tree depth.
			criterion (str): Split quality criterion.
			jobs (int): Number of parallel worker processes.
			random (int): Random seed.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.n_estimators = estimators
		self.max_depth = depth
		self.criterion = criterion
		self.n_jobs = jobs
		self.random_state = random
		self.validate_configuration( )
		self.model = ske.RandomForestClassifier( n_estimators=self.n_estimators,
			max_depth=self.max_depth, criterion=self.criterion, n_jobs=self.n_jobs,
			random_state=self.random_state )
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'prediction',
		         'probability',
		         'random_state',
		         'n_estimators',
		         'max_depth',
		         'criterion',
		         'n_jobs',
		         'model',
		         'train',
		         'project',
		         'predict_probability',
		         'score',
		         'analyze',
		         'confusion_matrix',
		         'roc_curve',
		         'scatter_plot',
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
		         'training_score' ]
	
	def validate_configuration( self ) -> None:
		"""

			Purpose:
			---------
			Validate wrapper configuration before estimator construction.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		try:
			_valid = { 'gini', 'entropy', 'log_loss' }
			if self.criterion not in _valid:
				raise ValueError( f'Unsupported criterion: {self.criterion}' )
			
			if self.n_estimators is None or self.n_estimators < 1:
				raise ValueError( 'Argument "estimators" must be greater than zero.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'validate_configuration( self ) -> None'
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def feature_importances( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return impurity-based feature importances.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'feature_importances_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.feature_importances_
	
	@property
	def outputs( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of outputs after training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if not hasattr( self.model, 'n_outputs_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_outputs_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			---------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return X_train, X_test, y_train, y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = (
					'split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, '
					'random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]'
			)
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> RandomForest | None:
		"""

			Purpose:
			---------
			Fit the Random Forest classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			RandomForest | None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.training_score = float( self.model.score( X, y ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> RandomForest | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict class labels for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for
				signature consistency.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class-probability estimates for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Return tabular classifier metrics for Streamlit display.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			df_metrics = self.score( X, y )
			self.testing_score = float( self.model.score( X, y ) )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Render a confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( self.confusion_matrix_values ).plot( values_format='d' )
			plt.title( 'Confusion Matrix' )
			plt.grid( False )
			plt.tight_layout( )
			return self.confusion_matrix_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""

			Purpose:
			---------
			Render a binary ROC curve using predicted class probabilities.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth binary labels.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, float ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			classes = np.unique( y )
			if len( classes ) != 2:
				raise ValueError( 'roc_curve is supported only for binary classification.' )
			
			probability = self.predict_probability( X )
			fpr, tpr, _ = roc_curve( y, probability[ :, 1 ], pos_label=classes[ 1 ] )
			roc_auc = auc( fpr, tpr )
			
			plt.plot( fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})' )
			plt.plot( [ 0, 1 ], [ 0, 1 ], linestyle='--', label='Random Guess' )
			plt.xlim( [ -0.01, 1.01 ] )
			plt.ylim( [ -0.01, 1.01 ] )
			plt.xlabel( 'False Positive Rate' )
			plt.ylabel( 'True Positive Rate' )
			plt.title( 'Receiver Operating Characteristic' )
			plt.legend( loc='lower right' )
			plt.grid( visible=True )
			plt.tight_layout( )
			return fpr, tpr, roc_auc
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_trn = self.training_score if self.training_score is not None else 0.0
			_tst = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot(
				x=y,
				y=y_pred,
				scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' }
			)
			plt.plot(
				[ y.min( ), y.max( ) ],
				[ y.min( ), y.max( ) ],
				'k--',
				label='Perfect Prediction'
			)
			plt.text(
				x=y.min( ),
				y=y.max( ) * 0.95,
				s=_text,
				fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 )
			)
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
			raise exception
			

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
	random_state: Optional[ int ]
	n_estimators: Optional[ int ]
	learning_rate: Optional[ float ]
	max_depth: Optional[ int ]
	criterion: Optional[ str ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	
	def __init__( self, estimators: int=100, rate: float=0.1, depth: int=3,
			criterion: str='friedman_mse', random: int=42 ) -> None:
		"""

			Purpose:
			--------
			Initialize the Gradient Boosting classifier wrapper.

			Parameters:
			-----------
			estimators (int): Number of boosting stages.
			rate (float): Learning rate applied to each boosting stage.
			depth (int): Maximum depth of each regression tree estimator.
			criterion (str): Split quality criterion used by the individual trees.
			random (int): Random seed.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.n_estimators = estimators
		self.learning_rate = rate
		self.max_depth = depth
		self.criterion = criterion
		self.random_state = random
		self.validate_configuration( )
		self.model = ske.GradientBoostingClassifier(
			n_estimators=self.n_estimators,
			learning_rate=self.learning_rate,
			max_depth=self.max_depth,
			criterion=self.criterion,
			random_state=self.random_state
		)
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			--------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'prediction',
		         'probability',
		         'random_state',
		         'n_estimators',
		         'learning_rate',
		         'max_depth',
		         'criterion',
		         'model',
		         'train',
		         'project',
		         'predict_probability',
		         'score',
		         'analyze',
		         'confusion_matrix',
		         'roc_curve',
		         'scatter_plot',
		         'labels',
		         'features_in',
		         'feature_importances',
		         'outputs',
		         'stages',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score' ]
	
	def validate_configuration( self ) -> None:
		"""

			Purpose:
			--------
			Validate wrapper configuration before estimator construction.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		try:
			_valid = { 'friedman_mse', 'squared_error' }
			if self.criterion not in _valid:
				raise ValueError( f'Unsupported criterion: {self.criterion}' )
			
			if self.n_estimators is None or self.n_estimators < 1:
				raise ValueError( 'Argument "estimators" must be greater than zero.' )
			
			if self.learning_rate is None or self.learning_rate <= 0.0:
				raise ValueError( 'Argument "rate" must be greater than zero.' )
			
			if self.max_depth is None or self.max_depth < 1:
				raise ValueError( 'Argument "depth" must be greater than zero.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'validate_configuration( self ) -> None'
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		"""

			Purpose:
			--------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def feature_importances( self ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return impurity-based feature importances.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'feature_importances_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.feature_importances_
	
	@property
	def outputs( self ) -> int:
		"""

			Purpose:
			--------
			Return the number of outputs after training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if not hasattr( self.model, 'n_trees_per_iteration_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_trees_per_iteration_
	
	@property
	def stages( self ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return the staged estimators array.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'estimators_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimators_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			--------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return X_train, X_test, y_train, y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = (
					'split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, '
					'random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]'
			)
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoost | None:
		"""

			Purpose:
			--------
			Fit the Gradient Boosting classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			GradientBoost | None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.training_score = float( self.model.score( X, y ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoost | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for
				signature consistency.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class-probability estimates for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Return tabular classifier metrics for Streamlit display.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			df_metrics = self.score( X, y )
			self.testing_score = float( self.model.score( X, y ) )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Render a confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( self.confusion_matrix_values ).plot( values_format='d' )
			plt.title( 'Confusion Matrix' )
			plt.grid( False )
			plt.tight_layout( )
			return self.confusion_matrix_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""

			Purpose:
			--------
			Render a binary ROC curve using predicted class probabilities.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth binary labels.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, float ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			classes = np.unique( y )
			if len( classes ) != 2:
				raise ValueError( 'roc_curve is supported only for binary classification.' )
			
			probability = self.predict_probability( X )
			fpr, tpr, _ = roc_curve( y, probability[ :, 1 ], pos_label=classes[ 1 ] )
			roc_auc = auc( fpr, tpr )
			
			plt.plot( fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})' )
			plt.plot( [ 0, 1 ], [ 0, 1 ], linestyle='--', label='Random Guess' )
			plt.xlim( [ -0.01, 1.01 ] )
			plt.ylim( [ -0.01, 1.01 ] )
			plt.xlabel( 'False Positive Rate' )
			plt.ylabel( 'True Positive Rate' )
			plt.title( 'Receiver Operating Characteristic' )
			plt.legend( loc='lower right' )
			plt.grid( visible=True )
			plt.tight_layout( )
			return fpr, tpr, roc_auc
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_trn = self.training_score if self.training_score is not None else 0.0
			_tst = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
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
			raise exception
			

class AdaptiveBoost( Classifier ):
	"""

		Purpose:
		---------
		A Boost classifier is a meta-estimator that begins by fitting a classifier
		on the original dataset and then fits additional copies of the classifier on the
		same dataset but where the weights of incorrectly classified instances are
		adjusted such that subsequent classifiers focus more on difficult cases.

	"""
	model: ske.AdaBoostClassifier
	base_estimator: object
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	random_state: Optional[ int ]
	n_estimators: Optional[ int ]
	learning_rate: Optional[ float ]
	algorithm: Optional[ str ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	
	def __init__( self, base: object = None, estimators: int=50,
			rate: float=1.0, algorithm: str='SAMME',
			random: int=42 ) -> None:
		"""

			Purpose:
			---------
			Initialize the AdaBoost classifier wrapper.

			Parameters:
			-----------
			base (object): Base estimator used inside the ensemble. If None,
				scikit-learn uses its default DecisionTreeClassifier.
			estimators (int): Number of boosting stages.
			rate (float): Learning rate applied to each boosting stage.
			algorithm (str): Discrete boosting algorithm identifier retained
				for wrapper compatibility.
			random (int): Random seed.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.base_estimator = base
		self.n_estimators = estimators
		self.learning_rate = rate
		self.algorithm = algorithm
		self.random_state = random
		self.validate_configuration( )
		self.model = ske.AdaBoostClassifier( estimator=self.base_estimator,
			n_estimators=self.n_estimators, learning_rate=self.learning_rate,
			random_state=self.random_state 	)
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'prediction',
		         'probability',
		         'random_state',
		         'base_estimator',
		         'n_estimators',
		         'learning_rate',
		         'algorithm',
		         'model',
		         'train',
		         'project',
		         'predict_probability',
		         'score',
		         'analyze',
		         'confusion_matrix',
		         'roc_curve',
		         'scatter_plot',
		         'labels',
		         'features_in',
		         'feature_importances',
		         'estimators',
		         'estimator_weights',
		         'estimator_errors',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score' ]
	
	def validate_configuration( self ) -> None:
		"""

			Purpose:
			---------
			Validate wrapper configuration before estimator construction.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		try:
			if self.n_estimators is None or self.n_estimators < 1:
				raise ValueError( 'Argument "estimators" must be greater than zero.' )
			
			if self.learning_rate is None or self.learning_rate <= 0.0:
				raise ValueError( 'Argument "rate" must be greater than zero.' )
			
			if self.algorithm not in { 'SAMME', 'deprecated', None }:
				raise ValueError( f'Unsupported algorithm setting: {self.algorithm}' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'validate_configuration( self ) -> None'
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def feature_importances( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return impurity-based feature importances when available.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'feature_importances_' ):
			raise AttributeError( 'Feature importances are not available for the trained model.' )
		return self.model.feature_importances_
	
	@property
	def estimators( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted boosting estimators.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'estimators_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return np.array( self.model.estimators_, dtype=object )
	
	@property
	def estimator_weights( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted estimator weights.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'estimator_weights_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimator_weights_
	
	@property
	def estimator_errors( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted estimator errors.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'estimator_errors_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimator_errors_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, 
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			---------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return split( X, y, test_size=size, random_state=random )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = (
					'split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, '
					'random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]'
			)
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> AdaptiveBoost | None:
		"""

			Purpose:
			---------
			Fit the AdaBoost classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			AdaptiveBoost | None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.training_score = float( self.model.score( X, y ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> AdaptiveBoost | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict class labels for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for
				signature consistency.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class-probability estimates for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Return tabular classifier metrics for analysis.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			df_metrics = self.score( X, y )
			self.testing_score = float( self.model.score( X, y ) )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Render a confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( self.confusion_matrix_values ).plot( values_format='d' )
			plt.title( 'Confusion Matrix' )
			plt.grid( False )
			plt.tight_layout( )
			return self.confusion_matrix_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""

			Purpose:
			---------
			Render a binary ROC curve using predicted class probabilities.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth binary labels.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, float ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			classes = np.unique( y )
			if len( classes ) != 2:
				raise ValueError( 'roc_curve is supported only for binary classification.' )
			
			probability = self.predict_probability( X )
			fpr, tpr, _ = roc_curve( y, probability[ :, 1 ], pos_label=classes[ 1 ] )
			roc_auc = auc( fpr, tpr )
			
			plt.plot( fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})' )
			plt.plot( [ 0, 1 ], [ 0, 1 ], linestyle='--', label='Random Guess' )
			plt.xlim( [ -0.01, 1.01 ] )
			plt.ylim( [ -0.01, 1.01 ] )
			plt.xlabel( 'False Positive Rate' )
			plt.ylabel( 'True Positive Rate' )
			plt.title( 'Receiver Operating Characteristic' )
			plt.legend( loc='lower right' )
			plt.grid( visible=True )
			plt.tight_layout( )
			return fpr, tpr, roc_auc
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
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
			raise exception
			

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
	base_estimator: object
	n_estimators: int
	max_features: int | float
	random_state: int
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	
	def __init__( self, base: object = None, num: int=10, max: int | float = 1.0, rando: int=42 ) -> None:
		"""

			Purpose:
			--------
			Initialize the Bagging classifier.

			Parameters:
			-----------
			base (object): Base estimator used inside the ensemble. If None,
				scikit-learn uses its default DecisionTreeClassifier.
			num (int): Number of estimators in the ensemble.
			max (int | float): Number or fraction of features drawn for each base estimator.
			rando (int): Random seed used by supported routines.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.base_estimator = base
		self.n_estimators = num
		self.max_features = max
		self.random_state = rando
		self.validate_configuration( )
		self.model = ske.BaggingClassifier( estimator=self.base_estimator, n_estimators=self.n_estimators,
			max_features=self.max_features, random_state=self.random_state )
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			--------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'prediction',
		         'probability',
		         'random_state',
		         'base_estimator',
		         'n_estimators',
		         'max_features',
		         'train',
		         'project',
		         'predict_probability',
		         'score',
		         'analyze',
		         'confusion_matrix',
		         'roc_curve',
		         'scatter_plot',
		         'labels',
		         'features',
		         'estimators',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score' ]
	
	def validate_configuration( self ) -> None:
		"""

			Purpose:
			--------
			Validate wrapper configuration before estimator construction.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		try:
			if self.n_estimators is None or self.n_estimators < 1:
				raise ValueError( 'Argument "num" must be greater than zero.' )
			
			if self.max_features is None:
				raise ValueError( 'Argument "max" cannot be empty!' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'validate_configuration( self ) -> None'
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.classes_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			--------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def estimators( self ) -> List[ Any ]:
		"""

			Purpose:
			--------
			Return fitted bagging estimators.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ Any ]

		"""
		if not hasattr( self.model, 'estimators_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimators_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float=0.2, random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			--------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return X_train, X_test, y_train, y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = (
					'split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, '
					'random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]'
			)
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BaggingModel | None:
		"""

			Purpose:
			--------
			Fit the Bagging classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			BaggingModel | None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.training_score = float( self.model.score( X, y ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> BaggingModel | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for signature consistency.

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class-probability estimates for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Return tabular classifier metrics for Streamlit display.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			df_metrics = self.score( X, y )
			self.testing_score = float( self.model.score( X, y ) )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Render a confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( self.confusion_matrix_values ).plot( values_format='d' )
			plt.title( 'Confusion Matrix' )
			plt.grid( False )
			plt.tight_layout( )
			return self.confusion_matrix_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""

			Purpose:
			--------
			Render a binary ROC curve using predicted class probabilities.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth binary labels.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, float ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			classes = np.unique( y )
			if len( classes ) != 2:
				raise ValueError( 'roc_curve is supported only for binary classification.' )
			
			probability = self.predict_probability( X )
			fpr, tpr, _ = roc_curve( y, probability[ :, 1 ], pos_label=classes[ 1 ] )
			roc_auc = auc( fpr, tpr )
			
			plt.plot( fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})' )
			plt.plot( [ 0, 1 ], [ 0, 1 ], linestyle='--', label='Random Guess' )
			plt.xlim( [ -0.01, 1.01 ] )
			plt.ylim( [ -0.01, 1.01 ] )
			plt.xlabel( 'False Positive Rate' )
			plt.ylabel( 'True Positive Rate' )
			plt.title( 'Receiver Operating Characteristic' )
			plt.legend( loc='lower right' )
			plt.grid( visible=True )
			plt.tight_layout( )
			return fpr, tpr, roc_auc
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_trn = self.training_score if self.training_score is not None else 0.0
			_tst = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
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
			raise exception
			

class VotingModel( Classifier ):
	"""

		Purpose:
		--------
		The Voting Model is to combine conceptually different machine rate
		classifiers and use a majority vote or the average predicted probabilities (soft vote)
		to predict the class target_names. Such a classifier can be useful for a set of equally
		well performing model in order to balance out their individual weaknesses.

	"""
	model: ske.VotingClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	random_state: Optional[ int ]
	estimator_list: List[ Tuple[ str, object ] ]
	voting: str
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, estimators: List[ Tuple[ str, object ] ], vote: str='hard' ) -> None:
		"""

			Purpose:
			--------
			Initialize the Voting classifier.

			Parameters:
			-----------
			estimators (List[ Tuple[ str, object ] ]): Named base estimators.
			vote (str): Voting method, either 'hard' or 'soft'.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.estimator_list = estimators
		self.voting = vote
		self.model = ske.VotingClassifier( estimators=self.estimator_list,
			voting=self.voting )
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			--------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'model',
		         'prediction',
		         'probability',
		         'voting',
		         'estimator_list',
		         'train',
		         'project',
		         'predict_probability',
		         'score',
		         'analyze',
		         'scatter_plot',
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
		"""

			Purpose:
			--------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		return self.model.classes_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			--------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def estimators( self ) -> List[ Any ]:
		"""

			Purpose:
			--------
			Return fitted voting estimators.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ Any ]

		"""
		if self.model.estimators_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimators_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float=0.2, random: int=42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		"""

			Purpose:
			--------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			tuple

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> VotingModel | None:
		"""

			Purpose:
			--------
			Fit the Voting classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			VotingModel | None

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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for signature consistency.

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class-probability estimates when soft voting is enabled.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			if self.voting != 'soft':
				raise ValueError( 'predict_probability requires voting="soft".' )
			
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Render a correlation heatmap for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text,
				fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
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
			raise exception
			

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
	estimator_list: List[ Tuple[ str, ClassifierMixin ] ]
	final_estimator: Optional[ ClassifierMixin ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, est: List[ Tuple[ str, ClassifierMixin ] ], final: ClassifierMixin=None ) -> None:
		"""

			Purpose:
			--------
			Initialize the Stacking classifier.

			Parameters:
			-----------
			est (List[ Tuple[ str, ClassifierMixin ] ]): Named base estimators.
			final (ClassifierMixin): Final estimator fit on stacked predictions.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.estimator_list = est
		self.final_estimator = final
		self.model = ske.StackingClassifier(
			estimators=self.estimator_list,
			final_estimator=self.final_estimator
		)
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			--------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'prediction',
		         'probability',
		         'final_estimator',
		         'estimator_list',
		         'model',
		         'train',
		         'project',
		         'predict_probability',
		         'score',
		         'analyze',
		         'scatter_plot',
		         'labels',
		         'features',
		         'estimators',
		         'final',
		         'accuracy',
		         'precision',
		         'balanced_accuracy',
		         'f1_score',
		         'recall',
		         'testing_score',
		         'training_score', ]
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		return self.model.classes_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			--------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def estimators( self ) -> List[ Any ]:
		"""

			Purpose:
			--------
			Return fitted base estimators.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ Any ]

		"""
		if self.model.estimators_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimators_
	
	@property
	def final( self ) -> Any:
		"""

			Purpose:
			--------
			Return the fitted final estimator.

			Parameters:
			-----------
			None

			Returns:
			--------
			Any

		"""
		if self.model.final_estimator_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.final_estimator_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float=0.2, random: int=42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		"""

			Purpose:
			--------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			tuple

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> StackingModel | None:
		"""

			Purpose:
			--------
			Fit the Stacking classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			StackingModel | None

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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Predict class labels for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for signature consistency.

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Return class-probability estimates for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			--------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Render a correlation heatmap for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ],
				'k--', label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text,
				fontsize=8, bbox=dict( facecolor='white', alpha=0.7 ) )
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
			raise exception
			

class SupportVector( Classifier ):
	"""

		Support Vector Classifier (SVC) is asupervised machine learning algorithm used primarily
		for classification, though it also handles regression. It works by finding an optimal
		"hyperplane"—a decision boundary—that maximizes the margin (distance) between different
		data classes, which improves prediction accuracy and generalization to new datais
		based on libsvm. The fit time scales at least quadratically with the number of samples
		and may be impractical beyond tens of thousands of samples.

	"""
	model: skv.SVC
	kernel: Optional[ str ]
	regulation: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	misclass: Optional[ float ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	degree: int
	
	def __init__( self, C: float=1.0, kernel: str='rbf', degree: int=3, random: int=42 ) -> None:
		"""

			Purpose:
			---------
			Initialize the support vector classifier wrapper.

			Parameters:
			-----------
			C (float): Regularization parameter.
			kernel (str): Kernel type.
			degree (int): Polynomial degree used when kernel='poly'.
			random (int): Random seed used when probability estimation is enabled.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.regulation = C
		self.kernel = kernel
		self.degree = degree
		self.random_state = random
		self.validate_configuration( )
		self.model = skv.SVC( C=self.regulation, kernel=self.kernel, degree=self.degree,
			probability=True, random_state=self.random_state )
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'prediction',
		         'probability',
		         'decision',
		         'random_state',
		         'model',
		         'kernel',
		         'regulation',
		         'degree',
		         'train',
		         'project',
		         'predict_probability',
		         'decision_function',
		         'score',
		         'analyze',
		         'confusion_matrix',
		         'roc_curve',
		         'scatter_plot',
		         'vectors',
		         'weights',
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
		         'training_score' ]
	
	def validate_configuration( self ) -> None:
		"""

			Purpose:
			---------
			Validate wrapper configuration before estimator construction.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		try:
			_valid_kernels = { 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' }
			if self.kernel not in _valid_kernels:
				raise ValueError( f'Unsupported kernel: {self.kernel}' )
			
			if self.kernel != 'poly' and self.degree is None:
				self.degree = 3
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'validate_configuration( self ) -> None'
			raise exception
	
	@property
	def vectors( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted support vectors.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'support_vectors_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.support_vectors_
	
	@property
	def weights( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted linear coefficients when a linear kernel is used.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if self.kernel != 'linear':
			raise AttributeError( 'The weights are only available when kernel="linear".' )
		
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.classes_
	
	@property
	def iterations( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the number of optimization iterations.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during training.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def supports( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the number of support vectors per class.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if not hasattr( self.model, 'n_support_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_support_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, 
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			---------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Test-set proportion.
			random (int): Random seed.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return X_train, X_test, y_train, y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = (
					'split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, '
					'random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]'
			)
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> SupportVector | None:
		"""

			Purpose:
			---------
			Fit the support vector classifier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			SupportVector | None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.training_score = float( self.model.score( X, y ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> SupportVector | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict class labels for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[ np.ndarray ]): Ignored optional argument preserved for
				signature consistency.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Compute decision scores for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.decision = self.model.decision_function( X )
			return self.decision
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'decision_function( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class-probability estimates for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Return tabular classifier metrics for Streamlit display.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			df_metrics = self.score( X, y )
			self.testing_score = float( self.model.score( X, y ) )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Render a confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			self.confusion_matrix_values = confusion_matrix( y, y_pred )
			ConfusionMatrixDisplay( self.confusion_matrix_values ).plot( values_format='d' )
			plt.title( 'Confusion Matrix' )
			plt.grid( False )
			plt.tight_layout( )
			return self.confusion_matrix_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""

			Purpose:
			---------
			Render a binary ROC curve using predicted class probabilities.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth binary labels.

			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray, float ]

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			classes = np.unique( y )
			if len( classes ) != 2:
				raise ValueError( 'roc_curve is supported only for binary classification.' )
			
			probability = self.predict_probability( X )
			fpr, tpr, _ = roc_curve( y, probability[ :, 1 ], pos_label=classes[ 1 ] )
			roc_auc = auc( fpr, tpr )
			
			plt.plot( fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})' )
			plt.plot( [ 0, 1 ], [ 0, 1 ], linestyle='--', label='Random Guess' )
			plt.xlim( [ -0.01, 1.01 ] )
			plt.ylim( [ -0.01, 1.01 ] )
			plt.xlabel( 'False Positive Rate' )
			plt.ylabel( 'True Positive Rate' )
			plt.title( 'Receiver Operating Characteristic' )
			plt.legend( loc='lower right' )
			plt.grid( visible=True )
			plt.tight_layout( )
			return fpr, tpr, roc_auc
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_trn = self.training_score if self.training_score is not None else 0.0
			_tst = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot(
				x=y,
				y=y_pred,
				scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' }
			)
			plt.plot(
				[ y.min( ), y.max( ) ],
				[ y.min( ), y.max( ) ],
				'k--',
				label='Perfect Prediction'
			)
			plt.text(
				x=y.min( ),
				y=y.max( ) * 0.95,
				s=_text,
				fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 )
			)
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
			raise exception
			

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
	random_state: Optional[ int ]
	hidden_layers: Tuple[ int, ... ]
	activation_function: str
	solver: str
	alpha: float
	learning_rate: Any
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	balanced_accuracy: Optional[ float ]
	recall: Optional[ float ]
	f1_score: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, hidden=(100,), activation='logistic', solver='lbfgs', alpha=0.0001,
			learning: str='constant', rando: int=42 ) -> None:
		"""

			Purpose:
			---------
			Initialize the multilayer perceptron classifier.

			Parameters:
			-----------
			hidden (Tuple[ int, ... ]): Hidden layer sizes.
			activation (str): Activation function for the hidden layers.
			solver (str): Weight optimization solver.
			alpha (float): L2 regularization strength.
			learning (str): Learning-rate schedule when solver='sgd'.
			rando (int): Random seed.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.hidden_layers = hidden
		self.activation_function = activation
		self.learning_rate = learning
		self.solver = solver
		self.alpha = alpha
		self.random_state = rando
		self.model = snn.MLPClassifier( hidden_layer_sizes=self.hidden_layers,
			activation=self.activation_function, solver=self.solver,
			alpha=self.alpha, learning_rate=self.learning_rate, random_state=self.random_state )
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Provide a list of strings representing class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]

		"""
		return [ 'prediction',
		         'probability',
		         'random_state',
		         'hidden_layers',
		         'activation_function',
		         'solver',
		         'alpha',
		         'learning_rate',
		         'model',
		         'train',
		         'project',
		         'predict_probability',
		         'score',
		         'analyze',
		         'labels',
		         'weights',
		         'layers',
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
		"""

			Purpose:
			---------
			Return class labels known to the classifier.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray

		"""
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		return self.model.classes_
	
	@property
	def weights( self ) -> List[ np.ndarray ]:
		"""

			Purpose:
			---------
			Return fitted layer weight matrices.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ np.ndarray ]

		"""
		if self.model.coefs_ is None:
			raise AttributeError( 'The weights have not been initialized!' )
		return self.model.coefs_
	
	@property
	def layers( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of fitted network layers.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if self.model.n_layers_ is None:
			raise AttributeError( 'The data has not been trained!' )
		return self.model.n_layers_
	
	@property
	def outputs( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of output units.

			Parameters:
			-----------
			None

			Returns:
			--------
			int

		"""
		if self.model.n_outputs_ is None:
			raise AttributeError( 'The data has not been trained!' )
		return self.model.n_outputs_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, 
			random: int=42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		"""

			Purpose:
			---------
			Split input arrays into training and testing subsets.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Target vector.
			size (float): Test set proportion.
			random (int): Random seed.

			Returns:
			--------
			tuple

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return split( X, y, test_size=size, random_state=random )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> MultiLayerPerceptron | None:
		"""

			Purpose:
			---------
			Fit the multilayer perceptron classifier.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.
			y (np.ndarray): Target labels.

			Returns:
			--------
			MultiLayerPerceptron | None

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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict class labels for the provided features.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.
			y (np.ndarray): Ignored.

			Returns:
			--------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return class-probability estimates for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.

			Returns:
			--------
			np.ndarray

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
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Compute scalar summary classification metrics.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			pd.DataFrame

		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Render a correlation heatmap for the supplied features.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): Ground-truth labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Plot observed labels against predicted labels.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): True class labels.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ np.min( y ), np.max( y ) ], [ np.min( y ), np.max( y ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=np.min( y ), y=np.max( y ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
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
			raise exception
			



