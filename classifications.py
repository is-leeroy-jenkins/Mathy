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
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	median_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
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
	prediction: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	recall: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	alpha: Optional[ float ]
	max_iter: Optional[ int ]
	shuffle: Optional[ bool ]
	penalty: Optional[ str ]
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	
	def __init__( self, alpha: float=0.0001, iters: int=1000, shuffle: bool=True, penalty='l2' ) -> None:
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
		self.alpha = alpha
		self.max_iter = iters
		self.shuffle = shuffle
		self.penalty = penalty
		self.model = skc.Perceptron( alpha=self.alpha, max_iter=self.max_iter,
			shuffle=self.shuffle, penalty=self.penalty, )
		self.decision = None
		self.prediction = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'model',
				 'prediction',
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
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'mean_squared_error',
		         'root_mean_squared_error',
		         'median_absolute_error',
		         'mean_abosolute_error']
	
	@property
	def weights( self ) -> np.ndarray:
		'''
			
			Returns
			-------
			Weights assigned to the features.
			ndarray of shape (n_features,) or (n_targets, n_features)

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
			classes_ is ndarray of shape (n_classes, )
			Actual number of iterations for all classes.
			If binary or multinomial, it returns only 1 element. For liblinear solver,
			only the maximum number of iteration across all classes is given.

		'''
		if self.model.n_iter_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_iter_
	
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': float( f'{self.mean_squared_error:.2f}' ),
				'RMSE': float( f'{ self.root_mean_squared_error:.2f}' ),
				'MEAE': float( f'{self.mean_absolute_error:.2f}' ),
				'MDAE': float( f'{self.median_absolute_error:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Preceptron Classification: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( visible=True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Perceptron'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
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
	model: skc.LinearRegression
	binarizer: Binarizer
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ np.ndarray ]
	recall: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	alpha: Optional[ float ]
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.model = skc.LinearRegression( )
		self.prediction = None
		self.probability = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
	
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
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.coef_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LinearRegression | None:
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
			exception.cause = 'LinearRegression'
			exception.method = 'train'
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
			exception.cause = 'LinearRegression'
			exception.method = 'predict'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Linear Regression: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LinearRegression'
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
	prediction: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	recall: Optional[ float ]
	mean_squared_error: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	f1_score: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	median_absolute_error: Optional[ float ]
	random_state: int
	penalty: str
	multi_class: str
	alpha: float
	max_iter: int
	solver: str
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, C: float=1.0, penalty: str='l2', iters: int=100,
			multi_class: str='ovr', solver: str='lbfgs' ) -> None:
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
		self.model = skc.LogisticRegression( C=self.alpha, max_iter=self.max_iter,
			multi_class=self.multi_class, solver=self.solver, penalty=self.penalty )
		self.prediction = None
		self.decision = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
	
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
		         'alpha',
		         'max_iter',
		         'predict_probabilty',
		         'decision_function',
		         'train',
		         'project',
		         'score',
		         'analyze',
		         'weights',
		         'iterations',
		         'labels'
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray:
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
			exception.cause = 'LogisticRegression'
			exception.method = 'project'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LogisticRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			plt.scatter( y, self.prediction, alpha=0.5  )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Logistic Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
	model: skc.RidgeClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	alpha: Optional[ float ]
	solver: Optional[ str ]
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
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
		self.model = skc.RidgeClassifier( alpha=self.alpha, solver=self.solver,
			max_iter=self.max_iter, random_state=self.random_state )
		self.prediction = None
		self.probability = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
		
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
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
	def features( self ) -> np.ndarray:
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=( 8, 6 ) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Ridge Regression: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	alpha: Optional[ float ]
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]

	def __init__( self, alpha: float=1.0, iters: int=500, rando: int=42, threshold: float=0.5 ) -> None:
		super( ).__init__( )
		self.alpha = alpha
		self.max_iter = iters
		self.random_state = rando
		self.threshold = threshold
		self.model = skc.Lasso( alpha=self.alpha, max_iter=self.max_iter,
			random_state=self.random_state )
		self.prediction = None
		self.probability = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
	
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
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
	def features( self ) -> np.ndarray:
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
			y_pred = self.model.predict( X )
			self.binarizer = Binarizer( threshold=self.threshold )
			_shape = y_pred.reshape( 1, -1 )
			self.prediction = self.binarizer.fit_transform( _shape ).astype( int ).flatten( )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
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
			Evaluate classifier performance using standard classification metrics.

			Parameters:
			---------
			X (np.ndarray): Input feature_names of shape (n_samples, n_features).
			y (np.ndarray): Ground truth class target_names.

			Returns:
			---------
			dict: Dictionary of evaluation metrics including:
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Lasso Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),
			            X.max( ) ], [ y.min( ),
			                          y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	loss: Optional[ str ]
	regularization: Optional[ Any ]
	alpha: Optional[ float ]
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
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
		self.model = skc.SGDClassifier( loss=self.loss, max_iter=self.max_iter,
			penalty=self.regularization, alpha=self.alpha )
		self.prediction = None
		self.probability = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
		
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
				 'model',
				 'weights',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap',
		         'weights',
		         'iterations',
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
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
			Evaluate classifier performance using standard classification metrics.

			Parameters:
			---------
			X (np.ndarray): Input feature_names of shape (n_samples, n_features).
			y (np.ndarray): Ground truth class target_names.

			Returns:
			---------
			dict: Dictionary of evaluation metrics including:
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Gradient Descent Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),
			            X.max( ) ], [ y.min( ),
			                          y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
	recall: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	algorithm: Any
	metric: str
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, neighbors: int=5, algorithm: str='auto', metric: str='minkowski' ) -> None:
		"""


			Purpose:
			-----------
			Initialize the KNeighborsClassifier linerar_model.

			Attributes:
			-----------
				linerar_model (KNeighborsClassifier): Internal non-parametric classifier.
					Parameters:
						neighbors (int): Number of neighbors to use. Default is 5.

		"""
		super( ).__init__( )
		self.n_neighbors = neighbors
		self.algorithm = algorithm
		self.metric = metric
		self.model = skn.KNeighborsClassifier( n_neighbors=self.n_neighbors,
			algorithm=self.algorithm, metric=self.metric )
		self.prediction = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
		
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
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
			self.prediction = self.model.predict( X )
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
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultilayerClassifier'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborClassifier'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborClassifier'
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'K-Nearest Neighbor Regression: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	classifier: Optional[ Any ]
	splitter: Optional[ str ]
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
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
		self.model = skd.DecisionTreeClassifier( criterion=self.criterion,
			splitter=self.splitter, max_depth=self.max_depth, random_state=self.random_state )
		self.prediction = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
		
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-------
			Provides a list of strings representing class members

		'''
		return [ 'prediction',
				 'max_depth',
				 'random_state',
				 'criterion',
				 'splitter',
				 'model',
				 'train',
				 'project',
				 'score',
				 'analyze',
				 'create_heatmap'
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
			self.prediction = self.model.predict( X )
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
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeClassifier'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
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
			Evaluate classifier performance using standard classification metrics.

			Parameters:
			---------
			X (np.ndarray): Input feature_names of shape (n_samples, n_features).
			y (np.ndarray): Ground truth class target_names.

			Returns:
			---------
			dict: Dictionary of evaluation metrics including:
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeClassifier'
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Decision Tree Regression: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
	precision: Optional[ float ]
	recall: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
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
		self.model = ske.RandomForestClassifier( n_estimators=self.n_estimators,
			criterion=self.criterion, max_depth=self.max_depth, random_state=self.random_state )
		self.prediction = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
		
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
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
	def features( self ) -> np.ndarray:
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Random Forest Regression: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ),  y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
		split
		may vary, even with the same training stores and max_features=n_features, if the improvement
		of the criterion is identical for several splits enumerated during the search of the best
		split. To obtain a deterministic behaviour during fitting, rando has to be fixed.

	"""
	model: ske.GradientBoostingClassifier
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
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
		self.model = ske.GradientBoostingClassifier( loss=self.loss,
			learning_rate=self.learning_rate, n_estimators=self.n_estimators,
			max_depth=self.max_depth, random_state=self.random_state )
		self.prediction = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
		
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
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Gradient Boost Regression: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ),  y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	f1_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	X_scaled: Optional[ pd.DataFrame ]
	estimator: Optional[ Any ]
	learning_rate: Optional[ float ]
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, num: int=100, learning: float=1.0 ) -> None:
		"""

			Initialize the Random Forest Classifier.

		"""
		super( ).__init__( )
		self.estimator = 'AdaBoostClassifier'
		self.n_estimators = num
		self.learning_rate = learning
		self.model = ske.AdaBoostClassifier( estimator=self.estimator,
			n_estimators=self.n_estimators, learning_rate=self.learning_rate )
		self.X_scaled = None
		self.prediction = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
		
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
		         'errors',
		         'weights',
		         'labels',
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
	@property
	def errors( self ) -> np.ndarray | None:
		if self.model.estimator_errors_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.estimator_errors_
	   
	@property
	def weights( self ) -> np.ndarray:
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
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.classes_
	
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Adaptive Boost Regression: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ),  y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
	recall: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	base_estimator: Optional[ Any ]
	n_estimators: Optional[ int ]
	mean_squared_error: Optional[ float ]
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
		self.precision = 0.0
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
	
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
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
		
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingClassifier'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Bagging Regression: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ),  y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
		The idea behind the VotingClassifier is to combine conceptually different machine rate
		classifiers and use a majority vote or the average predicted probabilities (soft vote)
		to predict the class target_names. Such a classifier can be useful for a set of equally
		well
		performing model in order to balance out their individual weaknesses.

	"""
	model: ske.VotingClassifier
	prediction: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	estimators: List[ (str, object) ]
	vote: str
	mean_squared_error: Optional[ float ]
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
		self.precision = 0.0
		self.area_under_curve = 0.0
		self.recall_score = 0.0
		self.f1_score = 0.0
		self.average_precision_score = 0.0
		self.top_k_accuracy = 0.0
		self.log_loss = 0.0
		self.hinge_loss = 0.0
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
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Voting Regression: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ),  y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
		the
		output of individual estimator and use a classifier to compute the final prediction.
		Stacking allows to use the strength of each individual estimator by using their output
		as input of a final estimator. Note that estimators_ are fitted on the full X while
		final_estimator_ is trained using cross-validated predictions of the base
		estimators using cross_val_predict.

	"""
	model: ske.StackingClassifier
	estimators: List[ Tuple[ str, ClassifierMixin ] ]
	final_estimator: Optional[ ClassifierMixin ]
	prediction: Optional[ np.ndarray ]
	recall: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
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
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
		
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
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingClassifier'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Stacking Regression: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ),  y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
	multiclass: Optional[ str ]
	regulation: Optional[ float ]
	penalty: Optional[ str ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
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
		self.model = skv.SVC( multi_class=self.multiclass, C=self.regulation,
			random_state=self.random_state, penalty=self.penalty, degree=self.degree )
		self.prediction = None
		self.recall = 0.0
		self.accuracy = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.f1_score = 0.0
		self.recall = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.median_absolute_error = 0.0
		self.mean_absolute_error = 0.0
		
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
		         'weights',
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
			self.probability = self.model.predict_proba( X )
			return self.probability
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'score( self, X: np.ndarray, y_true: np.ndarray ) -> float '
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Support Vector Regression: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ),  y.max( ) ], 'r--' )
			plt.grid( visible=True )
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
	recall: Optional[ float ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	average_precision: Optional[ float ]
	f1_score: Optional[ float ]
	hinge_loss: Optional[ float ]
	median_absolute_error: Optional[ float ]
	hidden_layers: tuple[ int, ... ]
	activation_function: str
	solver: str
	alpha: float
	learning_rate: Any
	mean_squared_error: Optional[ float ]
	classification_report: Optional[ Dict[ str, Any ] ]
	confusion_matrix: Optional[ np.ndarray ]
	
	def __init__( self, hidden=( 100, ), activation='relu', solver='adam', alpha=0.0001,
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
		self.precision = 0.0
		self.area_under_curve = 0.0
		self.recall_score = 0.0
		self.f1_score = 0.0
		self.average_precision_score = 0.0
		self.top_k_accuracy = 0.0
		self.log_loss = 0.0
		self.hinge_loss = 0.0
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
		         'classes',
		         'loss',
		         'precision',
		         'accuracy',
		         'f1_score',
		         'recall',
		         'area_under_curve',
		         'average_precision',
		         'top_k_accuracy',
		         'log_loss',
		         'hinge_loss' ]
	
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
			raise AttributeError( 'The model data has not been trained!' )
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
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.coefs_
	
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
			exception.cause = 'MultilayerClassifier'
			exception.method = 'predict_probability( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
			self.prediction = self.project( X, y  )
			self.precision = precision_score( y, self.prediction, average=None )
			self.accuracy = accuracy_score( y, self.prediction )
			self.recall = recall_score( y, self.prediction, average=None )
			self.f1_score = f1_score( y, self.prediction, average=None )
			return \
			{
				'F1': float( f'{self.f1_score:.2f}' ),
				'PRE': float( f'{self.precision:.2f}' ),
				'ACC': float( f'{self.accuracy:.2f}' ),
				'REC': float( f'{self.recall:.2f}' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
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
			- Accuracy Scoe (float)
			- Area Under the Curve (float)
			- Average Precision Score (float)
			- Top-K Accuracy Score (float)
			- Hinge-Loss (float)
			- Logarithmic-Loss (float)

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X, )
			self.mean_squared_error = mean_squared_error( y, self.prediction)
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error,
				'MEAE': self.mean_absolute_error,
				'MDAE': self.median_absolute_error,
			}
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
			self.prediction = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			plt.scatter( y, self.prediction, alpha=0.5 )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Multi-Layer Perceptron: Observed vs Projected' )
			plt.plot( [ X.min( ), X.max( ) ], [ y.min( ),  y.max( ) ], 'r--' )
			plt.grid( visible=True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )



