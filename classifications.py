"""******************************************************************************************
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
    Provides classification wrappers and diagnostics for Mathy modeling workflows. The
    module centralizes linear classifiers, tree classifiers, ensemble classifiers,
    nearest-neighbor classifiers, support-vector classifiers, neural-network classifiers,
    train/test splitting, prediction, scoring, probability estimation, confusion matrices,
    ROC calculations, and exploratory plotting behind a consistent wrapper interface.
</summary>
******************************************************************************************
"""
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
from boogr import Error, Logger

def throw_if( name: str, value: object ):
	"""Validate a required argument.

		Purpose:
		    Raises an exception when a required argument is missing so classifier methods fail before
		    downstream sklearn operations receive invalid input.

		Args:
		    name: Argument name used in the validation error message.
		    value: Argument value checked for missing state.

		Raises:
		    Exception: Raised when `value` is `None`.
	"""
	if value is None:
		raise Exception( f'Argument "{name}" cannot be empty!' )

class Classifier( ):
	"""Classifier classifier wrapper.

		Purpose:
		    Defines the shared interface and diagnostic state for Mathy classification wrappers,
		    including training, prediction, scoring, reporting, confusion-matrix generation, and
		    exploratory visualization contracts used by concrete classifier implementations.

		Attributes:
		    max_iter: Max iter value maintained by the Classifier wrapper.
		    random_state: Random state value maintained by the Classifier wrapper.
		    learning_rate: Learning rate value maintained by the Classifier wrapper.
		    binarizer: Binarizer value maintained by the Classifier wrapper.
		    prediction: Prediction value maintained by the Classifier wrapper.
		    probability: Probability value maintained by the Classifier wrapper.
		    decision: Decision value maintained by the Classifier wrapper.
		    misclass: Misclass value maintained by the Classifier wrapper.
		    accuracy: Accuracy value maintained by the Classifier wrapper.
		    precision: Precision value maintained by the Classifier wrapper.
		    recall: Recall value maintained by the Classifier wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the Classifier wrapper.
		    f1_score: F1 score value maintained by the Classifier wrapper.
		    training_score: Training score value maintained by the Classifier wrapper.
		    testing_score: Testing score value maintained by the Classifier wrapper.
		    classification_report: Classification report value maintained by the Classifier wrapper.
		    confusion_matrix_values: Confusion matrix values value maintained by the Classifier wrapper.
		    markers: Markers value maintained by the Classifier wrapper.
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
		"""Initialize Classifier.

				Purpose:
				    Initializes the Classifier wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.
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
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		raise NotImplementedError
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		raise NotImplementedError
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		raise NotImplementedError
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		raise NotImplementedError
	
	def classification_scores( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Calculate classification metrics.

				Purpose:
				    Builds a classification metrics dataframe from predictions and target labels, including
				    accuracy, precision, recall, F1, and related diagnostic measures.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			self.classification_report = classification_report( y, y_pred, output_dict=True,
				zero_division=0 )
			
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
			exception.method = 'classification_scores( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def correlation_heatmap( self, X: np.ndarray ) -> None:
		"""Render a correlation heatmap.

				Purpose:
				    Renders a correlation heatmap for numeric feature data to support exploratory analysis
				    before classification modeling.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Calculate the confusion matrix.

				Purpose:
				    Computes and stores the confusion matrix for classifier predictions against supplied target
				    labels.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Confusion-matrix values for predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'confusion_matrix( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	_classification_scores = classification_scores
	_correlation_heatmap = correlation_heatmap

class Perceptron( Classifier ):
	"""Perceptron classifier wrapper.

		Purpose:
		    Wraps sklearn.linear_model.Perceptron for linear classification with configurable
		    regularization, learning rate, iteration count, shuffling behavior, and random state while
		    exposing consistent Mathy training and evaluation methods.

		Attributes:
		    model: Model value maintained by the Perceptron wrapper.
		    binarizer: Binarizer value maintained by the Perceptron wrapper.
		    prediction: Prediction value maintained by the Perceptron wrapper.
		    decision: Decision value maintained by the Perceptron wrapper.
		    random_state: Random state value maintained by the Perceptron wrapper.
		    alpha: Alpha value maintained by the Perceptron wrapper.
		    max_iter: Max iter value maintained by the Perceptron wrapper.
		    shuffle: Shuffle value maintained by the Perceptron wrapper.
		    penalty: Penalty value maintained by the Perceptron wrapper.
		    training_score: Training score value maintained by the Perceptron wrapper.
		    testing_score: Testing score value maintained by the Perceptron wrapper.
		    classification_report: Classification report value maintained by the Perceptron wrapper.
		    confusion_matrix_values: Confusion matrix values value maintained by the Perceptron wrapper.
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
	
	def __init__( self, alpha: float = 0.001, eta: float = 1.0, iters: int = 1000,
			shuffle: bool = False, penalty: Optional[ str ] = None,
			random: int = 42 ) -> None:
		"""Initialize Perceptron.

				Purpose:
				    Initializes the Perceptron wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    alpha: Regularization strength or learning-rate parameter assigned to the estimator.
				    eta: Learning-rate parameter assigned to the estimator.
				    iters: Maximum iteration count assigned to the estimator.
				    shuffle: Flag indicating whether samples are shuffled during estimator training.
				    penalty: Regularization penalty assigned to the estimator.
				    random: Random seed used for reproducible partitioning or estimator behavior.
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
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Validate classifier configuration.

				Purpose:
				    Validates classifier configuration values before model training so unsupported options fail
				    early with explicit errors.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	@property
	def weights( self ) -> np.ndarray:
		"""Return weights metadata.

				Purpose:
				    Returns fitted coefficient weights from the underlying classifier for model inspection and
				    downstream diagnostics.

				Returns:
				    np.ndarray: Fitted coefficient array from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The Perceptron data is untrained.' )
		return self.model.coef_
	
	@property
	def iterations( self ) -> np.ndarray:
		"""Return iterations metadata.

				Purpose:
				    Returns fitted iteration-count metadata from the underlying classifier for convergence
				    inspection.

				Returns:
				    np.ndarray: Fitted iteration metadata from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> Tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return split( X, y, test_size=size, random_state=random )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate decision scores.

				Purpose:
				    Calculates classifier decision scores, margins, or distances from the fitted estimator for
				    ranking, thresholding, and diagnostic workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Decision-score array for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Perceptron | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    Perceptron | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> Perceptron | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification diagnostics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'analyze( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Calculate the confusion matrix.

				Purpose:
				    Computes and stores the confusion matrix for classifier predictions against supplied target
				    labels.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Confusion-matrix values for predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'confusion_matrix( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class LeastSquares( Classifier ):
	"""LeastSquares classifier wrapper.

		Purpose:
		    Wraps sklearn.linear_model.PassiveAggressiveClassifier-style least-squares classification
		    behavior for linear margin-based classification with configurable regularization, learning
		    rate, iteration count, and random state.

		Attributes:
		    model: Model value maintained by the LeastSquares wrapper.
		    binarizer: Binarizer value maintained by the LeastSquares wrapper.
		    prediction: Prediction value maintained by the LeastSquares wrapper.
		    decision: Decision value maintained by the LeastSquares wrapper.
		    random_state: Random state value maintained by the LeastSquares wrapper.
		    alpha: Alpha value maintained by the LeastSquares wrapper.
		    max_iter: Max iter value maintained by the LeastSquares wrapper.
		    shuffle: Shuffle value maintained by the LeastSquares wrapper.
		    penalty: Penalty value maintained by the LeastSquares wrapper.
		    probability: Probability value maintained by the LeastSquares wrapper.
		    training_score: Training score value maintained by the LeastSquares wrapper.
		    testing_score: Testing score value maintained by the LeastSquares wrapper.
		    classification_report: Classification report value maintained by the LeastSquares wrapper.
		    confusion_matrix_values: Confusion matrix values value maintained by the LeastSquares wrapper.
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
	probability: Optional[ np.ndarray ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix_values: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float = 0.0001, eta: float = 0.01, iters: int = 1000,
			shuffle: bool = False, penalty: Optional[ str ] = 'l2',
			random: int = 42 ) -> None:
		"""Initialize LeastSquares.

				Purpose:
				    Initializes the LeastSquares wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    alpha: Regularization strength or learning-rate parameter assigned to the estimator.
				    eta: Learning-rate parameter assigned to the estimator.
				    iters: Maximum iteration count assigned to the estimator.
				    shuffle: Flag indicating whether samples are shuffled during estimator training.
				    penalty: Regularization penalty assigned to the estimator.
				    random: Random seed used for reproducible partitioning or estimator behavior.
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
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Validate classifier configuration.

				Purpose:
				    Validates classifier configuration values before model training so unsupported options fail
				    early with explicit errors.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	@property
	def weights( self ) -> np.ndarray:
		"""Return weights metadata.

				Purpose:
				    Returns fitted coefficient weights from the underlying classifier for model inspection and
				    downstream diagnostics.

				Returns:
				    np.ndarray: Fitted coefficient array from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def iterations( self ) -> np.ndarray:
		"""Return iterations metadata.

				Purpose:
				    Returns fitted iteration-count metadata from the underlying classifier for convergence
				    inspection.

				Returns:
				    np.ndarray: Fitted iteration metadata from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> Tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return split( X, y, test_size=size, random_state=random )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate decision scores.

				Purpose:
				    Calculates classifier decision scores, margins, or distances from the fitted estimator for
				    ranking, thresholding, and diagnostic workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Decision-score array for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LeastSquares | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    LeastSquares | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> LeastSquares | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification diagnostics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'analyze( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Calculate the confusion matrix.

				Purpose:
				    Computes and stores the confusion matrix for classifier predictions against supplied target
				    labels.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Confusion-matrix values for predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'confusion_matrix( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class LogisticRegression( Classifier ):
	"""LogisticRegression classifier wrapper.

		Purpose:
		    Wraps sklearn.linear_model.LogisticRegression for probabilistic linear classification with
		    configurable penalty, inverse regularization strength, solver, multiclass handling,
		    iteration count, and random state.

		Attributes:
		    model: Model value maintained by the LogisticRegression wrapper.
		    binarizer: Binarizer value maintained by the LogisticRegression wrapper.
		    prediction: Prediction value maintained by the LogisticRegression wrapper.
		    decision: Decision value maintained by the LogisticRegression wrapper.
		    probability: Probability value maintained by the LogisticRegression wrapper.
		    transformed_data: Transformed data value maintained by the LogisticRegression wrapper.
		    random_state: Random state value maintained by the LogisticRegression wrapper.
		    penalty: Penalty value maintained by the LogisticRegression wrapper.
		    multi_class: Multi class value maintained by the LogisticRegression wrapper.
		    C: C value maintained by the LogisticRegression wrapper.
		    max_iter: Max iter value maintained by the LogisticRegression wrapper.
		    solver: Solver value maintained by the LogisticRegression wrapper.
		    accuracy: Accuracy value maintained by the LogisticRegression wrapper.
		    precision: Precision value maintained by the LogisticRegression wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the LogisticRegression wrapper.
		    recall: Recall value maintained by the LogisticRegression wrapper.
		    f1_score: F1 score value maintained by the LogisticRegression wrapper.
		    training_score: Training score value maintained by the LogisticRegression wrapper.
		    testing_score: Testing score value maintained by the LogisticRegression wrapper.
		    classification_report: Classification report value maintained by the LogisticRegression wrapper.
		    confusion_matrix_values: Confusion matrix values value maintained by the LogisticRegression wrapper.
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
	
	def __init__( self, C: float = 1.0, penalty: str = 'l2', iters: int = 1000,
			multiclass: str = 'multinomial', solver: str = 'lbfgs',
			random: int = 42 ) -> None:
		"""Initialize LogisticRegression.

				Purpose:
				    Initializes the LogisticRegression wrapper by assigning configuration values, constructing
				    the underlying sklearn estimator when applicable, and preparing runtime state used by
				    training, prediction, scoring, and diagnostics.

				Args:
				    C: Inverse regularization strength assigned to the estimator.
				    penalty: Regularization penalty assigned to the estimator.
				    iters: Maximum iteration count assigned to the estimator.
				    multiclass: Multiclass handling strategy assigned to the estimator.
				    solver: Optimization solver assigned to the estimator.
				    random: Random seed used for reproducible partitioning or estimator behavior.
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
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Validate classifier configuration.

				Purpose:
				    Validates classifier configuration values before model training so unsupported options fail
				    early with explicit errors.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	@property
	def weights( self ) -> np.ndarray:
		"""Return weights metadata.

				Purpose:
				    Returns fitted coefficient weights from the underlying classifier for model inspection and
				    downstream diagnostics.

				Returns:
				    np.ndarray: Fitted coefficient array from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def iterations( self ) -> np.ndarray:
		"""Return iterations metadata.

				Purpose:
				    Returns fitted iteration-count metadata from the underlying classifier for convergence
				    inspection.

				Returns:
				    np.ndarray: Fitted iteration metadata from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_iter_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate decision scores.

				Purpose:
				    Calculates classifier decision scores, margins, or distances from the fitted estimator for
				    ranking, thresholding, and diagnostic workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Decision-score array for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LogisticRegression | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    LogisticRegression | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> LogisticRegression | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification diagnostics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'analyze( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Calculate the confusion matrix.

				Purpose:
				    Computes and stores the confusion matrix for classifier predictions against supplied target
				    labels.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Confusion-matrix values for predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'confusion_matrix( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""Calculate ROC curve values.

				Purpose:
				    Computes receiver-operating-characteristic arrays and area-under-curve values from
				    classifier probabilities or decision scores.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    Tuple[np.ndarray, np.ndarray, float]: False-positive-rate array, true-positive-rate array, and area-under-curve value.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'roc_curve( self, *args ) -> Tuple[np.ndarray, np.ndarray, float]'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class Ridge( Classifier ):
	"""Ridge classifier wrapper.

		Purpose:
		    Ridge Classifier for linear classification with L2 regularization,
		    solver selection, iteration control, fitted coefficient access, and Mathy scoring and
		    visualization helpers.

		Attributes:
		    model: Model value maintained by the Ridge wrapper.
		    prediction: Prediction value maintained by the Ridge wrapper.
		    probability: Probability value maintained by the Ridge wrapper.
		    decision: Decision value maintained by the Ridge wrapper.
		    random_state: Random state value maintained by the Ridge wrapper.
		    alpha: Alpha value maintained by the Ridge wrapper.
		    solver: Solver value maintained by the Ridge wrapper.
		    accuracy: Accuracy value maintained by the Ridge wrapper.
		    precision: Precision value maintained by the Ridge wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the Ridge wrapper.
		    recall: Recall value maintained by the Ridge wrapper.
		    f1_score: F1 score value maintained by the Ridge wrapper.
		    training_score: Training score value maintained by the Ridge wrapper.
		    testing_score: Testing score value maintained by the Ridge wrapper.
		    classification_report: Classification report value maintained by the Ridge wrapper.
		    confusion_matrix: Confusion matrix value maintained by the Ridge wrapper.
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
	
	def __init__( self, alpha: float = 1.0, solver: str = 'auto', iters: int = 1000,
			rando: int = 42 ) -> None:
		"""Initialize Ridge.

				Purpose:
				    Initializes the Ridge wrapper by assigning configuration values, constructing the underlying
				    sklearn estimator when applicable, and preparing runtime state used by training, prediction,
				    scoring, and diagnostics.

				Args:
				    alpha: Regularization strength or learning-rate parameter assigned to the estimator.
				    solver: Optimization solver assigned to the estimator.
				    iters: Maximum iteration count assigned to the estimator.
				    rando: Random seed assigned to the estimator.
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
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Return weights metadata.

				Purpose:
				    Returns fitted coefficient weights from the underlying classifier for model inspection and
				    downstream diagnostics.

				Returns:
				    np.ndarray: Fitted coefficient array from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.coef_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.classes_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    (np.ndarray, np.ndarray, np.ndarray, np.ndarray): Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'split_data( self, *args ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray)'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Ridge | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    Ridge | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> Ridge | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'analyze( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate decision scores.

				Purpose:
				    Calculates classifier decision scores, margins, or distances from the fitted estimator for
				    ranking, thresholding, and diagnostic workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Decision-score array for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class Lasso( Classifier ):
	"""Lasso classifier wrapper.

		Purpose:
		    Wraps sklearn.linear_model.Lasso-derived classification behavior by thresholding continuous
		    projections into class predictions while exposing coefficient inspection, splitting,
		    scoring, and visualization utilities.

		Attributes:
		    model: Model value maintained by the Lasso wrapper.
		    prediction: Prediction value maintained by the Lasso wrapper.
		    binarizer: Binarizer value maintained by the Lasso wrapper.
		    probability: Probability value maintained by the Lasso wrapper.
		    decision: Decision value maintained by the Lasso wrapper.
		    random_state: Random state value maintained by the Lasso wrapper.
		    selection: Selection value maintained by the Lasso wrapper.
		    alpha: Alpha value maintained by the Lasso wrapper.
		    threshold: Threshold value maintained by the Lasso wrapper.
		    accuracy: Accuracy value maintained by the Lasso wrapper.
		    precision: Precision value maintained by the Lasso wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the Lasso wrapper.
		    recall: Recall value maintained by the Lasso wrapper.
		    f1_score: F1 score value maintained by the Lasso wrapper.
		    training_score: Training score value maintained by the Lasso wrapper.
		    testing_score: Testing score value maintained by the Lasso wrapper.
		    classification_report: Classification report value maintained by the Lasso wrapper.
		    confusion_matrix: Confusion matrix value maintained by the Lasso wrapper.
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
	
	def __init__( self, alpha: float = 1.0, iters: int = 500, rando: int = 42,
			threshold: float = 0.5,
			selection: str = 'random' ) -> None:
		"""Initialize Lasso.

				Purpose:
				    Initializes the Lasso wrapper by assigning configuration values, constructing the underlying
				    sklearn estimator when applicable, and preparing runtime state used by training, prediction,
				    scoring, and diagnostics.

				Args:
				    alpha: Regularization strength or learning-rate parameter assigned to the estimator.
				    iters: Maximum iteration count assigned to the estimator.
				    rando: Random seed assigned to the estimator.
				    threshold: Decision threshold used to convert continuous outputs into class labels.
				    selection: Coordinate-selection strategy assigned to the estimator.
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
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Return weights metadata.

				Purpose:
				    Returns fitted coefficient weights from the underlying classifier for model inspection and
				    downstream diagnostics.

				Returns:
				    np.ndarray: Fitted coefficient array from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.coef_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def iterations( self ) -> int:
		"""Return iterations metadata.

				Purpose:
				    Returns fitted iteration-count metadata from the underlying classifier for convergence
				    inspection.

				Returns:
				    int: Fitted iteration metadata from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.n_iter_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    (np.ndarray, np.ndarray, np.ndarray, np.ndarray): Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'split_data( self, *args ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray)'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Lasso | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    Lasso | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> Lasso | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'analyze( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class GradientDescent( Classifier ):
	"""GradientDescent classifier wrapper.

		Purpose:
		    Wraps sklearn.linear_model.SGDClassifier for stochastic-gradient linear classification with
		    configurable loss, regularization, averaging, learning-rate schedule, and iteration control.

		Attributes:
		    model: Model value maintained by the GradientDescent wrapper.
		    prediction: Prediction value maintained by the GradientDescent wrapper.
		    probability: Probability value maintained by the GradientDescent wrapper.
		    decision: Decision value maintained by the GradientDescent wrapper.
		    max_iter: Max iter value maintained by the GradientDescent wrapper.
		    random_state: Random state value maintained by the GradientDescent wrapper.
		    loss: Loss value maintained by the GradientDescent wrapper.
		    learning_rate: Learning rate value maintained by the GradientDescent wrapper.
		    average: Average value maintained by the GradientDescent wrapper.
		    regularization: Regularization value maintained by the GradientDescent wrapper.
		    alpha: Alpha value maintained by the GradientDescent wrapper.
		    accuracy: Accuracy value maintained by the GradientDescent wrapper.
		    precision: Precision value maintained by the GradientDescent wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the GradientDescent wrapper.
		    recall: Recall value maintained by the GradientDescent wrapper.
		    f1_score: F1 score value maintained by the GradientDescent wrapper.
		    training_score: Training score value maintained by the GradientDescent wrapper.
		    testing_score: Testing score value maintained by the GradientDescent wrapper.
		    classification_report: Classification report value maintained by the GradientDescent wrapper.
		    confusion_matrix: Confusion matrix value maintained by the GradientDescent wrapper.
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
	
	def __init__( self, loss: str = 'hinge', iters: int = 100, reg: str = 'l2',
			alpha: float = 0.00001,
			ave: bool = True, rate: str = 'optimal' ) -> None:
		"""Initialize GradientDescent.

				Purpose:
				    Initializes the GradientDescent wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    loss: Loss function assigned to the estimator.
				    iters: Maximum iteration count assigned to the estimator.
				    reg: Regularization penalty assigned to the estimator.
				    alpha: Regularization strength or learning-rate parameter assigned to the estimator.
				    ave: Flag controlling averaged stochastic-gradient behavior.
				    rate: Learning-rate or boosting-rate value assigned to the estimator.
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
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Return weights metadata.

				Purpose:
				    Returns fitted coefficient weights from the underlying classifier for model inspection and
				    downstream diagnostics.

				Returns:
				    np.ndarray: Fitted coefficient array from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.coef_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def iterations( self ) -> np.ndarray:
		"""Return iterations metadata.

				Purpose:
				    Returns fitted iteration-count metadata from the underlying classifier for convergence
				    inspection.

				Returns:
				    np.ndarray: Fitted iteration metadata from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.n_iter_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.classes_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    (np.ndarray, np.ndarray, np.ndarray, np.ndarray): Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'split_data( self, *args ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray)'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientDescent | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    GradientDescent | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> GradientDescent | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'analyze( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate decision scores.

				Purpose:
				    Calculates classifier decision scores, margins, or distances from the fitted estimator for
				    ranking, thresholding, and diagnostic workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Decision-score array for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class NearestNeighbor( Classifier ):
	"""NearestNeighbor classifier wrapper.

		Purpose:
		    Wraps sklearn.neighbors.KNeighborsClassifier for instance-based classification using
		    configurable neighbor count, search algorithm, leaf size, distance metric, and Minkowski
		    power.

		Attributes:
		    model: Model value maintained by the NearestNeighbor wrapper.
		    prediction: Prediction value maintained by the NearestNeighbor wrapper.
		    probability: Probability value maintained by the NearestNeighbor wrapper.
		    n_neighbors: N neighbors value maintained by the NearestNeighbor wrapper.
		    leaf_size: Leaf size value maintained by the NearestNeighbor wrapper.
		    power: Power value maintained by the NearestNeighbor wrapper.
		    algorithm: Algorithm value maintained by the NearestNeighbor wrapper.
		    metric: Metric value maintained by the NearestNeighbor wrapper.
		    accuracy: Accuracy value maintained by the NearestNeighbor wrapper.
		    precision: Precision value maintained by the NearestNeighbor wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the NearestNeighbor wrapper.
		    recall: Recall value maintained by the NearestNeighbor wrapper.
		    f1_score: F1 score value maintained by the NearestNeighbor wrapper.
		    training_score: Training score value maintained by the NearestNeighbor wrapper.
		    testing_score: Testing score value maintained by the NearestNeighbor wrapper.
		    classification_report: Classification report value maintained by the NearestNeighbor wrapper.
		    confusion_matrix_values: Confusion matrix values value maintained by the NearestNeighbor wrapper.
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
	
	def __init__( self, num: int = 5, algorithm: str = 'auto',
			power: int = 2, metric: str = 'minkowski', leafs: int = 30 ) -> None:
		"""Initialize NearestNeighbor.

				Purpose:
				    Initializes the NearestNeighbor wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    num: Number of neighbors, estimators, or model components assigned to the wrapper.
				    algorithm: Algorithm option assigned to the estimator.
				    power: Minkowski distance power assigned to the estimator.
				    metric: Distance metric assigned to the estimator.
				    leafs: Leaf-size parameter assigned to the nearest-neighbor estimator.
		"""
		super( ).__init__( )
		self.n_neighbors = num
		self.algorithm = algorithm
		self.power = power
		self.metric = metric
		self.leaf_size = leafs
		self.validate_configuration( )
		self.model = skn.KNeighborsClassifier( n_neighbors=self.n_neighbors,
			algorithm=self.algorithm,
			p=self.power, metric=self.metric, leaf_size=self.leaf_size )
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Validate classifier configuration.

				Purpose:
				    Validates classifier configuration values before model training so unsupported options fail
				    early with explicit errors.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		"""Return features in metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def samples( self ) -> int:
		"""Return samples metadata.

				Purpose:
				    Returns fitted sample-count metadata recorded by the underlying nearest-neighbor classifier.

				Returns:
				    int: Sample-count metadata learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_samples_fit_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_samples_fit_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> NearestNeighbor | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    NearestNeighbor | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> NearestNeighbor | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification diagnostics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'analyze( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Calculate the confusion matrix.

				Purpose:
				    Computes and stores the confusion matrix for classifier predictions against supplied target
				    labels.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Confusion-matrix values for predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'confusion_matrix( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""Calculate ROC curve values.

				Purpose:
				    Computes receiver-operating-characteristic arrays and area-under-curve values from
				    classifier probabilities or decision scores.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    Tuple[np.ndarray, np.ndarray, float]: False-positive-rate array, true-positive-rate array, and area-under-curve value.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'roc_curve( self, *args ) -> Tuple[np.ndarray, np.ndarray, float]'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class DecisionTree( Classifier ):
	"""DecisionTree classifier wrapper.

		Purpose:
		    Wraps sklearn.tree.DecisionTreeClassifier for tree-based classification with configurable
		    split criterion, splitter strategy, maximum depth, minimum split size, minimum leaf size,
		    and random state.

		Attributes:
		    model: Model value maintained by the DecisionTree wrapper.
		    prediction: Prediction value maintained by the DecisionTree wrapper.
		    probability: Probability value maintained by the DecisionTree wrapper.
		    random_state: Random state value maintained by the DecisionTree wrapper.
		    criterion: Criterion value maintained by the DecisionTree wrapper.
		    splitter: Splitter value maintained by the DecisionTree wrapper.
		    max_depth: Max depth value maintained by the DecisionTree wrapper.
		    min_samples_split: Min samples split value maintained by the DecisionTree wrapper.
		    min_samples_leaf: Min samples leaf value maintained by the DecisionTree wrapper.
		    accuracy: Accuracy value maintained by the DecisionTree wrapper.
		    precision: Precision value maintained by the DecisionTree wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the DecisionTree wrapper.
		    recall: Recall value maintained by the DecisionTree wrapper.
		    f1_score: F1 score value maintained by the DecisionTree wrapper.
		    training_score: Training score value maintained by the DecisionTree wrapper.
		    testing_score: Testing score value maintained by the DecisionTree wrapper.
		    classification_report: Classification report value maintained by the DecisionTree wrapper.
		    confusion_matrix_values: Confusion matrix values value maintained by the DecisionTree wrapper.
	"""
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
	
	def __init__( self, criterion: str = 'gini', splitter: str = 'best',
			depth: Optional[ int ] = None, min_split: int = 2,
			min_leaf: int = 1, random: int = 42 ) -> None:
		"""Initialize DecisionTree.

				Purpose:
				    Initializes the DecisionTree wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    criterion: Split or impurity criterion assigned to the estimator.
				    splitter: Tree split strategy assigned to the estimator.
				    depth: Maximum model depth assigned to the estimator.
				    min_split: Minimum sample count required to split an internal tree node.
				    min_leaf: Minimum sample count required at a tree leaf node.
				    random: Random seed used for reproducible partitioning or estimator behavior.
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
			max_depth=self.max_depth, min_samples_split=self.min_samples_split,
			min_samples_leaf=self.min_samples_leaf, random_state=self.random_state )
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Validate classifier configuration.

				Purpose:
				    Validates classifier configuration values before model training so unsupported options fail
				    early with explicit errors.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		"""Return features in metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def feature_importances( self ) -> np.ndarray:
		"""Return feature importances metadata.

				Purpose:
				    Returns impurity-based feature-importance values learned by the underlying tree or ensemble
				    classifier.

				Returns:
				    np.ndarray: Feature-importance array learned by the fitted classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'feature_importances_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.feature_importances_
	
	@property
	def classes_count( self ) -> np.ndarray | int:
		"""Return classes count metadata.

				Purpose:
				    Returns the number of samples associated with learned classes or tree outputs, as exposed by
				    the fitted classifier.

				Returns:
				    np.ndarray | int: Class-count metadata exposed by the fitted classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_classes_
	
	@property
	def outputs( self ) -> int:
		"""Return outputs metadata.

				Purpose:
				    Returns the number of classifier outputs learned during fitting.

				Returns:
				    int: Output-count metadata exposed by the fitted classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_outputs_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_outputs_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> Tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return split( X, y, test_size=size, random_state=random )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> DecisionTree | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    DecisionTree | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> DecisionTree | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification diagnostics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'analyze( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Calculate the confusion matrix.

				Purpose:
				    Computes and stores the confusion matrix for classifier predictions against supplied target
				    labels.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Confusion-matrix values for predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'confusion_matrix( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""Calculate ROC curve values.

				Purpose:
				    Computes receiver-operating-characteristic arrays and area-under-curve values from
				    classifier probabilities or decision scores.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    Tuple[np.ndarray, np.ndarray, float]: False-positive-rate array, true-positive-rate array, and area-under-curve value.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'roc_curve( self, *args ) -> Tuple[np.ndarray, np.ndarray, float]'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class RandomForest( Classifier ):
	"""RandomForest classifier wrapper.

		Purpose:
		    Wraps sklearn.ensemble.RandomForestClassifier for ensemble tree classification with
		    configurable estimator count, depth, split criterion, parallelism, and random state.

		Attributes:
		    model: Model value maintained by the RandomForest wrapper.
		    prediction: Prediction value maintained by the RandomForest wrapper.
		    probability: Probability value maintained by the RandomForest wrapper.
		    decision: Decision value maintained by the RandomForest wrapper.
		    random_state: Random state value maintained by the RandomForest wrapper.
		    n_estimators: N estimators value maintained by the RandomForest wrapper.
		    max_depth: Max depth value maintained by the RandomForest wrapper.
		    criterion: Criterion value maintained by the RandomForest wrapper.
		    n_jobs: N jobs value maintained by the RandomForest wrapper.
		    accuracy: Accuracy value maintained by the RandomForest wrapper.
		    precision: Precision value maintained by the RandomForest wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the RandomForest wrapper.
		    recall: Recall value maintained by the RandomForest wrapper.
		    f1_score: F1 score value maintained by the RandomForest wrapper.
		    training_score: Training score value maintained by the RandomForest wrapper.
		    testing_score: Testing score value maintained by the RandomForest wrapper.
		    classification_report: Classification report value maintained by the RandomForest wrapper.
		    confusion_matrix_values: Confusion matrix values value maintained by the RandomForest wrapper.
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
	
	def __init__( self, estimators: int = 100, depth: Optional[ int ] = None,
			criterion: str = 'gini', jobs: int = -1, random: int = 42 ) -> None:
		"""Initialize RandomForest.

				Purpose:
				    Initializes the RandomForest wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    estimators: Named estimator collection or estimator count assigned to the ensemble wrapper.
				    depth: Maximum model depth assigned to the estimator.
				    criterion: Split or impurity criterion assigned to the estimator.
				    jobs: Parallel worker count assigned to the estimator.
				    random: Random seed used for reproducible partitioning or estimator behavior.
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
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Validate classifier configuration.

				Purpose:
				    Validates classifier configuration values before model training so unsupported options fail
				    early with explicit errors.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		"""Return features in metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def feature_importances( self ) -> np.ndarray:
		"""Return feature importances metadata.

				Purpose:
				    Returns impurity-based feature-importance values learned by the underlying tree or ensemble
				    classifier.

				Returns:
				    np.ndarray: Feature-importance array learned by the fitted classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'feature_importances_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.feature_importances_
	
	@property
	def outputs( self ) -> int:
		"""Return outputs metadata.

				Purpose:
				    Returns the number of classifier outputs learned during fitting.

				Returns:
				    int: Output-count metadata exposed by the fitted classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_outputs_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_outputs_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> RandomForest | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    RandomForest | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> RandomForest | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification diagnostics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'analyze( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Calculate the confusion matrix.

				Purpose:
				    Computes and stores the confusion matrix for classifier predictions against supplied target
				    labels.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Confusion-matrix values for predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'confusion_matrix( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""Calculate ROC curve values.

				Purpose:
				    Computes receiver-operating-characteristic arrays and area-under-curve values from
				    classifier probabilities or decision scores.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    Tuple[np.ndarray, np.ndarray, float]: False-positive-rate array, true-positive-rate array, and area-under-curve value.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'roc_curve( self, *args ) -> Tuple[np.ndarray, np.ndarray, float]'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class GradientBoost( Classifier ):
	"""GradientBoost classifier wrapper.

		Purpose:
		    Wraps sklearn.ensemble.GradientBoostingClassifier for staged additive tree classification
		    with configurable estimator count, learning rate, tree depth, criterion, and random state.

		Attributes:
		    model: Model value maintained by the GradientBoost wrapper.
		    prediction: Prediction value maintained by the GradientBoost wrapper.
		    probability: Probability value maintained by the GradientBoost wrapper.
		    random_state: Random state value maintained by the GradientBoost wrapper.
		    n_estimators: N estimators value maintained by the GradientBoost wrapper.
		    learning_rate: Learning rate value maintained by the GradientBoost wrapper.
		    max_depth: Max depth value maintained by the GradientBoost wrapper.
		    criterion: Criterion value maintained by the GradientBoost wrapper.
		    accuracy: Accuracy value maintained by the GradientBoost wrapper.
		    precision: Precision value maintained by the GradientBoost wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the GradientBoost wrapper.
		    recall: Recall value maintained by the GradientBoost wrapper.
		    f1_score: F1 score value maintained by the GradientBoost wrapper.
		    training_score: Training score value maintained by the GradientBoost wrapper.
		    testing_score: Testing score value maintained by the GradientBoost wrapper.
		    classification_report: Classification report value maintained by the GradientBoost wrapper.
		    confusion_matrix_values: Confusion matrix values value maintained by the GradientBoost wrapper.
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
	
	def __init__( self, estimators: int = 100, rate: float = 0.1, depth: int = 3,
			criterion: str = 'friedman_mse', random: int = 42 ) -> None:
		"""Initialize GradientBoost.

				Purpose:
				    Initializes the GradientBoost wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    estimators: Named estimator collection or estimator count assigned to the ensemble wrapper.
				    rate: Learning-rate or boosting-rate value assigned to the estimator.
				    depth: Maximum model depth assigned to the estimator.
				    criterion: Split or impurity criterion assigned to the estimator.
				    random: Random seed used for reproducible partitioning or estimator behavior.
		"""
		super( ).__init__( )
		self.n_estimators = estimators
		self.learning_rate = rate
		self.max_depth = depth
		self.criterion = criterion
		self.random_state = random
		self.validate_configuration( )
		self.model = ske.GradientBoostingClassifier( n_estimators=self.n_estimators,
			learning_rate=self.learning_rate, max_depth=self.max_depth,
			criterion=self.criterion, random_state=self.random_state )
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Validate classifier configuration.

				Purpose:
				    Validates classifier configuration values before model training so unsupported options fail
				    early with explicit errors.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		"""Return features in metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def feature_importances( self ) -> np.ndarray:
		"""Return feature importances metadata.

				Purpose:
				    Returns impurity-based feature-importance values learned by the underlying tree or ensemble
				    classifier.

				Returns:
				    np.ndarray: Feature-importance array learned by the fitted classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'feature_importances_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.feature_importances_
	
	@property
	def outputs( self ) -> int:
		"""Return outputs metadata.

				Purpose:
				    Returns the number of classifier outputs learned during fitting.

				Returns:
				    int: Output-count metadata exposed by the fitted classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_trees_per_iteration_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_trees_per_iteration_
	
	@property
	def stages( self ) -> np.ndarray:
		"""Return stages metadata.

				Purpose:
				    Returns staged decision-function metadata exposed by the fitted gradient-boosting
				    classifier.

				Returns:
				    np.ndarray: Staged estimator or staged decision metadata exposed by the fitted classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'estimators_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimators_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoost | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    GradientBoost | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> GradientBoost | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification diagnostics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'analyze( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Calculate the confusion matrix.

				Purpose:
				    Computes and stores the confusion matrix for classifier predictions against supplied target
				    labels.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Confusion-matrix values for predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'confusion_matrix( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""Calculate ROC curve values.

				Purpose:
				    Computes receiver-operating-characteristic arrays and area-under-curve values from
				    classifier probabilities or decision scores.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    Tuple[np.ndarray, np.ndarray, float]: False-positive-rate array, true-positive-rate array, and area-under-curve value.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'roc_curve( self, *args ) -> Tuple[np.ndarray, np.ndarray, float]'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class AdaptiveBoost( Classifier ):
	"""AdaptiveBoost classifier wrapper.

		Purpose:
		    Wraps sklearn.ensemble.AdaBoostClassifier for adaptive ensemble classification with
		    configurable base estimator, estimator count, learning rate, algorithm, and random state.

		Attributes:
		    model: Model value maintained by the AdaptiveBoost wrapper.
		    base_estimator: Base estimator value maintained by the AdaptiveBoost wrapper.
		    prediction: Prediction value maintained by the AdaptiveBoost wrapper.
		    probability: Probability value maintained by the AdaptiveBoost wrapper.
		    random_state: Random state value maintained by the AdaptiveBoost wrapper.
		    n_estimators: N estimators value maintained by the AdaptiveBoost wrapper.
		    learning_rate: Learning rate value maintained by the AdaptiveBoost wrapper.
		    algorithm: Algorithm value maintained by the AdaptiveBoost wrapper.
		    accuracy: Accuracy value maintained by the AdaptiveBoost wrapper.
		    precision: Precision value maintained by the AdaptiveBoost wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the AdaptiveBoost wrapper.
		    recall: Recall value maintained by the AdaptiveBoost wrapper.
		    f1_score: F1 score value maintained by the AdaptiveBoost wrapper.
		    training_score: Training score value maintained by the AdaptiveBoost wrapper.
		    testing_score: Testing score value maintained by the AdaptiveBoost wrapper.
		    classification_report: Classification report value maintained by the AdaptiveBoost wrapper.
		    confusion_matrix_values: Confusion matrix values value maintained by the AdaptiveBoost wrapper.
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
	
	def __init__( self, base: object = None, estimators: int = 50,
			rate: float = 1.0, algorithm: str = 'SAMME',
			random: int = 42 ) -> None:
		"""Initialize AdaptiveBoost.

				Purpose:
				    Initializes the AdaptiveBoost wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    base: Base estimator supplied to the ensemble wrapper.
				    estimators: Named estimator collection or estimator count assigned to the ensemble wrapper.
				    rate: Learning-rate or boosting-rate value assigned to the estimator.
				    algorithm: Algorithm option assigned to the estimator.
				    random: Random seed used for reproducible partitioning or estimator behavior.
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
			random_state=self.random_state )
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Validate classifier configuration.

				Purpose:
				    Validates classifier configuration values before model training so unsupported options fail
				    early with explicit errors.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.classes_
	
	@property
	def features_in( self ) -> int:
		"""Return features in metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def feature_importances( self ) -> np.ndarray:
		"""Return feature importances metadata.

				Purpose:
				    Returns impurity-based feature-importance values learned by the underlying tree or ensemble
				    classifier.

				Returns:
				    np.ndarray: Feature-importance array learned by the fitted classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'feature_importances_' ):
			raise AttributeError( 'Feature importances are not available for the trained model.' )
		return self.model.feature_importances_
	
	@property
	def estimators( self ) -> np.ndarray:
		"""Return estimators metadata.

				Purpose:
				    Returns fitted base estimators retained by the ensemble classifier.

				Returns:
				    np.ndarray: Fitted base estimators retained by the ensemble classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'estimators_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return np.array( self.model.estimators_, dtype=object )
	
	@property
	def estimator_weights( self ) -> np.ndarray:
		"""Return estimator weights metadata.

				Purpose:
				    Returns learned AdaBoost estimator weights assigned during ensemble fitting.

				Returns:
				    np.ndarray: AdaBoost estimator weights learned during fitting.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'estimator_weights_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimator_weights_
	
	@property
	def estimator_errors( self ) -> np.ndarray:
		"""Return estimator errors metadata.

				Purpose:
				    Returns learned AdaBoost estimator errors assigned during ensemble fitting.

				Returns:
				    np.ndarray: AdaBoost estimator errors learned during fitting.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'estimator_errors_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimator_errors_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return split( X, y, test_size=size, random_state=random )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> AdaptiveBoost | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    AdaptiveBoost | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> AdaptiveBoost | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification diagnostics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'analyze( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Calculate the confusion matrix.

				Purpose:
				    Computes and stores the confusion matrix for classifier predictions against supplied target
				    labels.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Confusion-matrix values for predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'confusion_matrix( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""Calculate ROC curve values.

				Purpose:
				    Computes receiver-operating-characteristic arrays and area-under-curve values from
				    classifier probabilities or decision scores.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    Tuple[np.ndarray, np.ndarray, float]: False-positive-rate array, true-positive-rate array, and area-under-curve value.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'roc_curve( self, *args ) -> Tuple[np.ndarray, np.ndarray, float]'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class BaggingModel( Classifier ):
	"""BaggingModel classifier wrapper.

		Purpose:
		    Wraps sklearn.ensemble.BaggingClassifier for bootstrap-aggregation classification with
		    configurable base estimator, estimator count, sample fraction, and random state.

		Attributes:
		    model: Model value maintained by the BaggingModel wrapper.
		    base_estimator: Base estimator value maintained by the BaggingModel wrapper.
		    n_estimators: N estimators value maintained by the BaggingModel wrapper.
		    max_features: Max features value maintained by the BaggingModel wrapper.
		    random_state: Random state value maintained by the BaggingModel wrapper.
		    prediction: Prediction value maintained by the BaggingModel wrapper.
		    probability: Probability value maintained by the BaggingModel wrapper.
		    accuracy: Accuracy value maintained by the BaggingModel wrapper.
		    precision: Precision value maintained by the BaggingModel wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the BaggingModel wrapper.
		    recall: Recall value maintained by the BaggingModel wrapper.
		    f1_score: F1 score value maintained by the BaggingModel wrapper.
		    training_score: Training score value maintained by the BaggingModel wrapper.
		    testing_score: Testing score value maintained by the BaggingModel wrapper.
		    classification_report: Classification report value maintained by the BaggingModel wrapper.
		    confusion_matrix_values: Confusion matrix values value maintained by the BaggingModel wrapper.
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
	
	def __init__( self, base: object = None, num: int = 10, max: int | float = 1.0,
			rando: int = 42 ) -> None:
		"""Initialize BaggingModel.

				Purpose:
				    Initializes the BaggingModel wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    base: Base estimator supplied to the ensemble wrapper.
				    num: Number of neighbors, estimators, or model components assigned to the wrapper.
				    max: Maximum sample fraction or count assigned to the bagging estimator.
				    rando: Random seed assigned to the estimator.
		"""
		super( ).__init__( )
		self.base_estimator = base
		self.n_estimators = num
		self.max_features = max
		self.random_state = rando
		self.validate_configuration( )
		self.model = ske.BaggingClassifier( estimator=self.base_estimator,
			n_estimators=self.n_estimators,
			max_features=self.max_features, random_state=self.random_state )
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Validate classifier configuration.

				Purpose:
				    Validates classifier configuration values before model training so unsupported options fail
				    early with explicit errors.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.classes_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def estimators( self ) -> List[ Any ]:
		"""Return estimators metadata.

				Purpose:
				    Returns fitted base estimators retained by the ensemble classifier.

				Returns:
				    List[Any]: Fitted base estimators retained by the ensemble classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'estimators_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimators_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> Tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BaggingModel | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    BaggingModel | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> BaggingModel | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification diagnostics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'analyze( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Calculate the confusion matrix.

				Purpose:
				    Computes and stores the confusion matrix for classifier predictions against supplied target
				    labels.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Confusion-matrix values for predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'confusion_matrix( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""Calculate ROC curve values.

				Purpose:
				    Computes receiver-operating-characteristic arrays and area-under-curve values from
				    classifier probabilities or decision scores.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    Tuple[np.ndarray, np.ndarray, float]: False-positive-rate array, true-positive-rate array, and area-under-curve value.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'roc_curve( self, *args ) -> Tuple[np.ndarray, np.ndarray, float]'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class VotingModel( Classifier ):
	"""VotingModel classifier wrapper.

		Purpose:
		    Wraps sklearn.ensemble.VotingClassifier for hard or soft ensemble voting across named base
		    estimators while exposing common Mathy training, prediction, scoring, and visualization
		    methods.

		Attributes:
		    model: Model value maintained by the VotingModel wrapper.
		    prediction: Prediction value maintained by the VotingModel wrapper.
		    probability: Probability value maintained by the VotingModel wrapper.
		    random_state: Random state value maintained by the VotingModel wrapper.
		    estimator_list: Estimator list value maintained by the VotingModel wrapper.
		    voting: Voting value maintained by the VotingModel wrapper.
		    accuracy: Accuracy value maintained by the VotingModel wrapper.
		    precision: Precision value maintained by the VotingModel wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the VotingModel wrapper.
		    recall: Recall value maintained by the VotingModel wrapper.
		    f1_score: F1 score value maintained by the VotingModel wrapper.
		    training_score: Training score value maintained by the VotingModel wrapper.
		    testing_score: Testing score value maintained by the VotingModel wrapper.
		    classification_report: Classification report value maintained by the VotingModel wrapper.
		    confusion_matrix: Confusion matrix value maintained by the VotingModel wrapper.
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
	
	def __init__( self, estimators: List[ Tuple[ str, object ] ], vote: str = 'hard' ) -> None:
		"""Initialize VotingModel.

				Purpose:
				    Initializes the VotingModel wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    estimators: Named estimator collection or estimator count assigned to the ensemble wrapper.
				    vote: Voting strategy assigned to the voting ensemble.
		"""
		super( ).__init__( )
		self.estimator_list = estimators
		self.voting = vote
		self.model = ske.VotingClassifier( estimators=self.estimator_list,
			voting=self.voting )
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		return self.model.classes_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def estimators( self ) -> List[ Any ]:
		"""Return estimators metadata.

				Purpose:
				    Returns fitted base estimators retained by the ensemble classifier.

				Returns:
				    List[Any]: Fitted base estimators retained by the ensemble classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.estimators_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimators_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> (np.ndarray, np.ndarray, np.ndarray,
	                                                  np.ndarray):
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    (np.ndarray, np.ndarray, np.ndarray, np.ndarray): Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'split_data( self, *args ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray)'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> VotingModel | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    VotingModel | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> VotingModel | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'analyze( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class StackingModel( Classifier ):
	"""StackingModel classifier wrapper.

		Purpose:
		    Wraps sklearn.ensemble.StackingClassifier for stacked ensemble classification using named
		    base estimators and an optional final estimator for meta-level prediction.

		Attributes:
		    model: Model value maintained by the StackingModel wrapper.
		    estimator_list: Estimator list value maintained by the StackingModel wrapper.
		    final_estimator: Final estimator value maintained by the StackingModel wrapper.
		    prediction: Prediction value maintained by the StackingModel wrapper.
		    probability: Probability value maintained by the StackingModel wrapper.
		    accuracy: Accuracy value maintained by the StackingModel wrapper.
		    precision: Precision value maintained by the StackingModel wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the StackingModel wrapper.
		    recall: Recall value maintained by the StackingModel wrapper.
		    f1_score: F1 score value maintained by the StackingModel wrapper.
		    training_score: Training score value maintained by the StackingModel wrapper.
		    testing_score: Testing score value maintained by the StackingModel wrapper.
		    classification_report: Classification report value maintained by the StackingModel wrapper.
		    confusion_matrix: Confusion matrix value maintained by the StackingModel wrapper.
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
	
	def __init__( self, est: List[ Tuple[ str, ClassifierMixin ] ],
			final: ClassifierMixin = None ) -> None:
		"""Initialize StackingModel.

				Purpose:
				    Initializes the StackingModel wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    est: Named base estimators assigned to the stacking ensemble.
				    final: Final estimator assigned to the stacking ensemble.
		"""
		super( ).__init__( )
		self.estimator_list = est
		self.final_estimator = final
		self.model = ske.StackingClassifier(
			estimators=self.estimator_list,
			final_estimator=self.final_estimator
		)
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		return self.model.classes_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def estimators( self ) -> List[ Any ]:
		"""Return estimators metadata.

				Purpose:
				    Returns fitted base estimators retained by the ensemble classifier.

				Returns:
				    List[Any]: Fitted base estimators retained by the ensemble classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.estimators_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.estimators_
	
	@property
	def final( self ) -> Any:
		"""Return final metadata.

				Purpose:
				    Returns the fitted final estimator used by the stacking classifier.

				Returns:
				    Any: Final estimator used by the stacking classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.final_estimator_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.final_estimator_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> (np.ndarray, np.ndarray, np.ndarray,
	                                                  np.ndarray):
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    (np.ndarray, np.ndarray, np.ndarray, np.ndarray): Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'split_data( self, *args ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray)'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> StackingModel | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    StackingModel | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> StackingModel | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'analyze( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class SupportVector( Classifier ):
	"""SupportVector classifier wrapper.

		Purpose:
		    Wraps sklearn.svm.SVC for support-vector classification with configurable penalty parameter,
		    kernel, polynomial degree, probability estimation, and random state.

		Attributes:
		    model: Model value maintained by the SupportVector wrapper.
		    kernel: Kernel value maintained by the SupportVector wrapper.
		    regulation: Regulation value maintained by the SupportVector wrapper.
		    prediction: Prediction value maintained by the SupportVector wrapper.
		    misclass: Misclass value maintained by the SupportVector wrapper.
		    probability: Probability value maintained by the SupportVector wrapper.
		    decision: Decision value maintained by the SupportVector wrapper.
		    random_state: Random state value maintained by the SupportVector wrapper.
		    accuracy: Accuracy value maintained by the SupportVector wrapper.
		    precision: Precision value maintained by the SupportVector wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the SupportVector wrapper.
		    recall: Recall value maintained by the SupportVector wrapper.
		    f1_score: F1 score value maintained by the SupportVector wrapper.
		    training_score: Training score value maintained by the SupportVector wrapper.
		    testing_score: Testing score value maintained by the SupportVector wrapper.
		    classification_report: Classification report value maintained by the SupportVector wrapper.
		    confusion_matrix_values: Confusion matrix values value maintained by the SupportVector wrapper.
		    degree: Degree value maintained by the SupportVector wrapper.
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
	
	def __init__( self, C: float = 1.0, kernel: str = 'rbf', degree: int = 3,
			random: int = 42 ) -> None:
		"""Initialize SupportVector.

				Purpose:
				    Initializes the SupportVector wrapper by assigning configuration values, constructing the
				    underlying sklearn estimator when applicable, and preparing runtime state used by training,
				    prediction, scoring, and diagnostics.

				Args:
				    C: Inverse regularization strength assigned to the estimator.
				    kernel: Kernel function assigned to the support-vector classifier.
				    degree: Polynomial kernel degree assigned to the support-vector classifier.
				    random: Random seed used for reproducible partitioning or estimator behavior.
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
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Validate classifier configuration.

				Purpose:
				    Validates classifier configuration values before model training so unsupported options fail
				    early with explicit errors.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	@property
	def vectors( self ) -> np.ndarray:
		"""Return vectors metadata.

				Purpose:
				    Returns support vectors learned by the fitted support-vector classifier.

				Returns:
				    np.ndarray: Support vectors learned by the fitted support-vector classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'support_vectors_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.support_vectors_
	
	@property
	def weights( self ) -> np.ndarray:
		"""Return weights metadata.

				Purpose:
				    Returns fitted coefficient weights from the underlying classifier for model inspection and
				    downstream diagnostics.

				Returns:
				    np.ndarray: Fitted coefficient array from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.kernel != 'linear':
			raise AttributeError( 'The weights are only available when kernel="linear".' )
		
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.coef_
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'classes_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.classes_
	
	@property
	def iterations( self ) -> np.ndarray:
		"""Return iterations metadata.

				Purpose:
				    Returns fitted iteration-count metadata from the underlying classifier for convergence
				    inspection.

				Returns:
				    np.ndarray: Fitted iteration metadata from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.

				Purpose:
				    Returns the number of input features observed by the underlying classifier during fitting.

				Returns:
				    int: Input-feature count learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def supports( self ) -> np.ndarray:
		"""Return supports metadata.

				Purpose:
				    Returns support-vector indices learned by the fitted support-vector classifier.

				Returns:
				    np.ndarray: Support-vector indices learned by the fitted support-vector classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_support_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_support_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> SupportVector | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    SupportVector | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> SupportVector | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def decision_function( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate decision scores.

				Purpose:
				    Calculates classifier decision scores, margins, or distances from the fitted estimator for
				    ranking, thresholding, and diagnostic workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Decision-score array for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification diagnostics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'analyze( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Calculate the confusion matrix.

				Purpose:
				    Computes and stores the confusion matrix for classifier predictions against supplied target
				    labels.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Confusion-matrix values for predicted and actual labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'confusion_matrix( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def roc_curve( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray, float ]:
		"""Calculate ROC curve values.

				Purpose:
				    Computes receiver-operating-characteristic arrays and area-under-curve values from
				    classifier probabilities or decision scores.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    Tuple[np.ndarray, np.ndarray, float]: False-positive-rate array, true-positive-rate array, and area-under-curve value.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'roc_curve( self, *args ) -> Tuple[np.ndarray, np.ndarray, float]'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class MultiLayerPerceptron( Classifier ):
	"""MultiLayerPerceptron classifier wrapper.

		Purpose:
		    Multi-Layer Perceptron Classifier for feed-forward neural-network classification
		    with configurable hidden layers, activation, solver, regularization, learning-rate behavior,
		    and random state.

		Attributes:
		    model: Model value maintained by the MultiLayerPerceptron wrapper.
		    prediction: Prediction value maintained by the MultiLayerPerceptron wrapper.
		    probability: Probability value maintained by the MultiLayerPerceptron wrapper.
		    random_state: Random state value maintained by the MultiLayerPerceptron wrapper.
		    hidden_layers: Hidden layers value maintained by the MultiLayerPerceptron wrapper.
		    activation_function: Activation function value maintained by the MultiLayerPerceptron wrapper.
		    solver: Solver value maintained by the MultiLayerPerceptron wrapper.
		    alpha: Alpha value maintained by the MultiLayerPerceptron wrapper.
		    learning_rate: Learning rate value maintained by the MultiLayerPerceptron wrapper.
		    accuracy: Accuracy value maintained by the MultiLayerPerceptron wrapper.
		    precision: Precision value maintained by the MultiLayerPerceptron wrapper.
		    balanced_accuracy: Balanced accuracy value maintained by the MultiLayerPerceptron wrapper.
		    recall: Recall value maintained by the MultiLayerPerceptron wrapper.
		    f1_score: F1 score value maintained by the MultiLayerPerceptron wrapper.
		    training_score: Training score value maintained by the MultiLayerPerceptron wrapper.
		    testing_score: Testing score value maintained by the MultiLayerPerceptron wrapper.
		    classification_report: Classification report value maintained by the MultiLayerPerceptron wrapper.
		    confusion_matrix: Confusion matrix value maintained by the MultiLayerPerceptron wrapper.
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
			learning: str = 'constant', rando: int = 42 ) -> None:
		"""Initialize MultiLayerPerceptron.

				Purpose:
				    Initializes the MultiLayerPerceptron wrapper by assigning configuration values, constructing
				    the underlying sklearn estimator when applicable, and preparing runtime state used by
				    training, prediction, scoring, and diagnostics.

				Args:
				    hidden: Hidden-layer sizes assigned to the neural-network classifier.
				    activation: Activation function assigned to the neural-network classifier.
				    solver: Optimization solver assigned to the estimator.
				    alpha: Regularization strength or learning-rate parameter assigned to the estimator.
				    learning: Learning-rate schedule assigned to the neural-network classifier.
				    rando: Random seed assigned to the estimator.
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
		"""List public members.

				Purpose:
				    Returns the stable public members exposed by the wrapper for interactive discovery, notebook
				    exploration, and IDE inspection.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
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
		"""Return labels metadata.

				Purpose:
				    Returns class labels learned by the underlying classifier during training.

				Returns:
				    np.ndarray: Class-label array learned by the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.classes_ is None:
			raise AttributeError( 'The data has not been trained!' )
		return self.model.classes_
	
	@property
	def weights( self ) -> List[ np.ndarray ]:
		"""Return weights metadata.

				Purpose:
				    Returns fitted coefficient weights from the underlying classifier for model inspection and
				    downstream diagnostics.

				Returns:
				    List[np.ndarray]: Fitted coefficient array from the underlying classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.coefs_ is None:
			raise AttributeError( 'The weights have not been initialized!' )
		return self.model.coefs_
	
	@property
	def layers( self ) -> int:
		"""Return layers metadata.

				Purpose:
				    Returns the number of neural-network layers learned by the fitted multilayer perceptron.

				Returns:
				    int: Neural-network layer count learned by the fitted classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.n_layers_ is None:
			raise AttributeError( 'The data has not been trained!' )
		return self.model.n_layers_
	
	@property
	def outputs( self ) -> int:
		"""Return outputs metadata.

				Purpose:
				    Returns the number of classifier outputs learned during fitting.

				Returns:
				    int: Output-count metadata exposed by the fitted classifier.

				Raises:
				    Exception: Raised when validation fails or required fitted metadata is unavailable.
		"""
		if self.model.n_outputs_ is None:
			raise AttributeError( 'The data has not been trained!' )
		return self.model.n_outputs_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
		"""Split feature and target data.

				Purpose:
				    Creates reproducible training and testing partitions from feature and target arrays for
				    classifier fitting, scoring, and evaluation.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.
				    size: Testing-set proportion used for train/test partitioning.
				    random: Random seed used for reproducible partitioning or estimator behavior.

				Returns:
				    (np.ndarray, np.ndarray, np.ndarray, np.ndarray): Training features, testing features, training labels, and testing labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return split( X, y, test_size=size, random_state=random )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'split_data( self, *args ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray)'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> MultiLayerPerceptron | None:
		"""Train the classifier.

				Purpose:
				    Fits the underlying sklearn classifier to aligned feature and target arrays and returns the
				    current wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    MultiLayerPerceptron | None: Fitted classifier wrapper instance.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'train( self, *args ) -> MultiLayerPerceptron | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: np.ndarray = None ) -> np.ndarray:
		"""Generate classifier predictions.

				Purpose:
				    Generates class predictions from the fitted classifier and stores the predicted labels on
				    the wrapper for later diagnostics.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    np.ndarray: Predicted class labels for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def predict_probability( self, X: np.ndarray ) -> np.ndarray:
		"""Calculate class probabilities.

				Purpose:
				    Calculates class-membership probabilities from the fitted classifier and stores the
				    probability matrix for later metric and curve calculations.

				Args:
				    X: Feature matrix used by the classifier workflow.

				Returns:
				    np.ndarray: Class-probability matrix for the supplied feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score classifier performance.

				Purpose:
				    Evaluates classifier performance against supplied features and labels, updates score-related
				    state, and returns a tabular metrics summary.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Returns:
				    pd.DataFrame: Classification metrics dataframe produced from supplied features and labels.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			return self.classification_scores( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Analyze classifier performance.

				Purpose:
				    Computes classification diagnostics for supplied features and labels, including predictions,
				    metrics, reports, and confusion-matrix values when available.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
		"""
		try:
			throw_if( 'y', y )
			self._correlation_heatmap( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'analyze( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Render a classifier scatter plot.

				Purpose:
				    Renders a two-dimensional scatter plot of feature values colored by class labels for
				    exploratory classifier review.

				Args:
				    X: Feature matrix used by the classifier workflow.
				    y: Target-label array aligned to the feature matrix.

				Raises:
				    Error: Raised when validation, estimator execution, metric calculation, or plotting fails.
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
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception
			



