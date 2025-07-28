'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                classifications.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="classifications.py" company="Terry D. Eppler">

     Mathy Models

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
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import ClassifierMixin
import sklearn.ensemble as ske
import sklearn.linear_model as skc
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
	r2_score, mean_squared_error, mean_absolute_error,
	explained_variance_score, median_absolute_error
)
import sklearn.neighbors as skn
import sklearn.neural_network as snn
from sklearn.pipeline import Pipeline
import sklearn.svm as skv
import sklearn.tree as skd

from booger import Error, ErrorDialog


class Model(  ):
	"""

		Purpose:
		---------
		Abstract base class that defines the interface for all linerar_model wrappers.

	"""
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


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Generate predictions from  the trained linerar_model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).

			Returns:
			-----------
				np.ndarray: Predicted labels or class labels.

		"""
		raise NotImplementedError


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			---------
			Compute the core metric (e.g., R²) of the model on test df.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): True target labels.

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
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth labels.

			Returns:
			-----------
				dict: Dictionary containing multiple evaluation metrics.

		"""
		raise NotImplementedError

class PerceptronClassifier( Model ):
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
	perceptron_classifier: skc.Perceptron
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	alpha: Optional[ float ]
	max_iter: Optional[ int ]
	shuffle: Optional[ bool ]
	l1_ratio: Optional[ float ]


	def __init__( self, reg: float=0.0001, max: int=1000, shuffle: bool=True, l1: float=0.15 ) -> None:
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
		self.max_iter = max
		self.shuffle = shuffle
		self.l1_ratio = l1
		self.perceptron_classifier = skc.Perceptron( alpha=self.alpha, max_iter=self.max_iter,
			shuffle=self.shuffle, l1_ratio=self.l1_ratio, )
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
		return [ 'prediction', 'max_depth', 'random_state', 'accuracy',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> PerceptronClassifier | None:
		"""

			Purpose:
			---------
			Fit the PerceptronClassifier linerar_model.

			Parameters:
			---------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Binary class labels.

			Returns:
			--------
			object

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.perceptron_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict binary class labels using the PerceptronClassifier.

			Parameters:
			---------
			X (pd.DataFrame): Feature matrix.

			Returns:
			---------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.perceptron_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Test feature_names.
				y (np.ndarray): True class labels.

			Returns:
			---------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.perceptron_classifier.predict( X )
				self.accuracy = accuracy_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
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
				y (np.ndarray): Ground truth class labels.

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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.perceptron_classifier.predict( X )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared=False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared=False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
			---------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.perceptron_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
				plt.xlabel( 'Predicted' )
				plt.ylabel( 'Actual' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=colors[ idx ],
						marker=markers[ idx ], label=f'Class {cl}', edgecolor='black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c='none',
							edgecolor='black', alpha=1.0, linewidth=1,
							marker='o', s=100, label='Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

class MultiLayerClassifier( Model ):
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
	multilayer_classifier: snn.MLPClassifier
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	hidden_layers: tuple[ int, int ]
	activation_function: str
	solver: str
	alpha: float
	learning_rate: Any
	random_state: int



	def __init__( self, hidden=( 100, ), activation='relu', solver='adam',
	              alpha = 0.0001, learning: str='constant', rando: int=42 ) -> None:
		super( ).__init__( )
		self.hidden_layers = hidden
		self.activation_function = activation
		self.learning_rate = learning
		self.solver = solver
		self.alpha = alpha
		self.random_state = rando
		self.multilayer_classifier = snn.MLPClassifier( hidden_layer_sizes=self.hidden_layers,
			activation=self.activation_function, solver=self.solver, alpha=self.alpha,
			learning_rate=self.learning_rate, random_state=self.random_state )
		self.pipeline = Pipeline( steps=list( hidden ) )
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
		return [ 'prediction', 'max_depth', 'random_state', 'accuracy',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> MultiLayerClassifier | None:
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.multilayer_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultiLayerClassifier'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			-----------
			Applies all transformations in the pipeline to the text df.

			Parameters:
			-----------
				X (np.ndarray): Input feature matrix.

			Returns:
			-----------
				np.ndarray: Transformed feature matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.multilayer_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegression'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			-----------
				Compute the R^2 accuracy of the model on the given test df.

			Parameters:
			-----------
				X (np.ndarray): Test feature_names.
				y (np.ndarray): True values.

			Returns:
			-----------
				float: R-squared accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.multilayer_classifier.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluate the model using multiple regression metrics.


			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth labels.

			Returns:
			-----------
				dict: Dictionary of MAE, MSE, RMSE, R², etc.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.multilayer_classifier.predict( X )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared = False )
				return \
				{
						'MAE': self.mean_absolute_error,
						'MSE': self.mean_squared_error,
						'RMSE': self.r_mean_squared_error,
						'R2': self.r2_score,
						'Explained Variance': self.explained_variance_score,
						'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
			---------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.multilayer_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
				plt.xlabel( 'Predicted' )
				plt.ylabel( 'Actual' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultiLayerClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx=None, resolution=0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class RidgeClassifier( Model ):
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

	def __init__( self, alpha: float=1.0, solver: str='auto', max: int=1000,
	              rando: int=42 ) -> None:
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
		self.max_iter = max
		self.random_state = rando
		self.ridge_classifier = skc.RidgeClassifier( alpha=self.alpha,
			solver=self.solver, max_iter=self.max_iter, random_state=self.random_state )
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
		return [ 'prediction', 'max_iter', 'random_state', 'accuracy',
		         'alpha', 'solver', 'ridge_classifier',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> RidgeClassifier | None:
		"""


			Purpose:
			-----------
			Fit the RidgeRegressor regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.ridge_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			-----------
			Project target labels using the RidgeRegressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted target labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.ridge_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
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
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth labels.

			Returns:
			-----------
				float: R-squared accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ridge_classifier.predict( X )
				self.accuracy = accuracy_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
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
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				dict: Evaluation metrics including MAE, RMSE, R², etc.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ridge_classifier.predict( X )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared=False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared = False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot predicted vs actual labels.

			Parameters:
			-----------
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ridge_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
				plt.xlabel( 'Predicted' )
				plt.ylabel( 'Actual' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx = None,
	                           resolution = 0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class StochasticGradientClassifier( Model ):
	"""

		Purpose:
		--------
		Linear classifiers (SVM, logistic regression, etc.) with SGD training. This estimator
		implements regularized linear models with stochastic gradient descent (SGD) learning:
		the gradient of the loss is estimated each sample at a time and the model is updated along
		the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch
		(online/out-of-core) learning via the partial_fit method. For best results using the
		default learning rate schedule, the data should have zero mean and unit variance.

		This implementation works with data represented as dense or sparse arrays of floating point
		 values for the feature_names. The model it fits can be controlled with the loss parameter;
		 by default, it fits a linear support vector machine (SVM).

		The regularizer is a penalty added to the loss function that shrinks model parameters
		towards the zero vector using either the squared euclidean norm L2 or the absolute norm
		L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value
		because of the regularizer, the update is truncated to 0.0 to allow for learning sparse
		 models and achieve online feature selection.

	"""
	stochastic_classifier: skc.SGDClassifier
	prediction: Optional[ np.ndarray ]
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


	def __init__( self, loss: str='hinge', max: int=5, reg: str='l2', alpha: float=0.0001 ) -> None:
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
		self.max_iter = max
		self.regularization = reg
		self.alpha = alpha
		self.stochastic_classifier = skc.SGDClassifier( loss=self.loss,
			max_iter=self.max_iter, penalty=self.regularization, alpha=self.alpha )
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
		return [ 'prediction', 'max_iter', 'random_state', 'accuracy',
		         'loss', 'regularization', 'alpha', 'stochastic_classifier',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> StochasticGradientClassifier | None:
		"""

			Purpose:
			-----------
			Fit the SGD classifier linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.stochastic_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

				Purpose:
				-----------
					Predict class labels using the SGD classifier.

				Parameters:
				-----------
					X (pd.DataFrame): Feature matrix.

				Returns:
				-----------
					np.ndarray: Predicted class labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.stochastic_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
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
			X (np.ndarray): Test feature_names.
			y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
			float: R^2 accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stochastic_classifier.predict( X )
				self.accuracy = accuracy_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluate the classifier using standard metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape (n_samples, n_features).
			y (np.ndarray): True class labels of shape (n_samples,).

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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stochastic_classifier.predict( X )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared=False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
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
			X (np.ndarray): Input feature_names.
			y (np.ndarray): True class labels.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stochastic_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
				plt.xlabel( 'Predicted' )
				plt.ylabel( 'Actual' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticGradientClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx = None,
	                           resolution = 0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class NearestNeighborClassifier( Model ):
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
	neighbor_classifier: skn.KNeighborsClassifier
	prediction: Optional[ np.ndarray ]
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


	def __init__( self, num: int=5, algorithm: Any='auto', metric: str='minkowski' ) -> None:
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
		self.neighbor_classifier = skn.KNeighborsClassifier( n_neighbors=self.n_neighbors,
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
		return [ 'prediction', 'max_depth', 'random_state', 'accuracy',
		         'n_neigbors', 'algorithm', 'metric', 'neighbor_classifier',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]

	def train( self, X: np.ndarray, y: np.ndarray ) -> NearestNeighborClassifier | None:
		"""

			Purpose:
			--------
			Fit the KNN classifier linerar_model.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (np.ndarray): Class labels.

			Returns:
			-------
			None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.neighbor_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			-----------
			Predict class labels using the KNN classifier.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted class labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.neighbor_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""


			Purpose:
			-----------
			Compute classification accuracy for k-NN.

			Parameters:
			-----------
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth labels.

			Returns:
			-----------
				float: Accuracy accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.neighbor_classifier.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict | None:
		"""


			Purpose:
			-----------
			Evaluate classification performance using various metrics.


			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).
				y (np.ndarray): True class labels of shape (n_samples,).

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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.neighbor_classifier.predict( X )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared = False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
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
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.neighbor_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
				plt.xlabel( 'Predicted' )
				plt.ylabel( 'Actual' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighbotClassification'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx = None,
	                           resolution = 0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class DecisionTreeClassifier( Model ):
	'''

		Purpose:
		--------
		Decision Trees (DTs) are a non-parametric supervised learning method used for
		classification. The goal is to create a model that predicts the value of a
		target variable by learning simple decision rules inferred from the data feature_names.

		A tree can be seen as a piecewise constant approximation. Decision trees learn from data
		to approximate a sine curve with a set of if-then-else decision rules.
		The deeper the tree, the more complex the decision rules and the fitter the model.

	'''
	dt_classifier: skd.DecisionTreeClassifier
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
		self.dt_classifier = skd.DecisionTreeClassifier( criterion=self.criterion,
			splitter=self.splitter, max_depth=self.max_depth, random_state =self.random_state )
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
		return [ 'prediction', 'max_depth', 'random_state', 'accuracy',
		         'criterion', 'splitter', 'dt_classifier',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> DecisionTreeClassifier | None:
		"""

			Purpose:
			--------
			Fit the KNN classifier linerar_model.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (np.ndarray): Class labels.

			Returns:
			-------
			None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.dt_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			-----------
			Predict class labels using the KNN classifier.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted class labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.dt_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""


			Purpose:
			-----------
			Compute classification accuracy for k-NN.

			Parameters:
			-----------
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth labels.

			Returns:
			-----------
				float: Accuracy accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.dt_classifier.predict( X )
				self.accuracy = accuracy_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeClassifier'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""


			Purpose:
			-----------
			Evaluate classification performance using various metrics.


			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).
				y (np.ndarray): True class labels of shape (n_samples,).

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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.dt_classifier.predict( X )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared=False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared = False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.dt_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
				plt.xlabel( 'Predicted' )
				plt.ylabel( 'Actual' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx = None,
	                           resolution = 0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class RandomForestClassifier( Model ):
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
	random_forest_classifier: ske.RandomForestClassifier
	prediction: Optional[ np.ndarray ]
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


	def __init__( self, est: int=10, crit: Any='gini', max: Any=None, rando: int=42 ) -> None:
		"""

			Purpose:
			-----------
			Initializes the RandomForestClassifier.

		"""
		super( ).__init__( )
		self.n_estimators = est
		self.criterion = crit
		self.max_depth = max
		self.random_state = rando
		self.random_forest_classifier = ske.RandomForestClassifier( n_estimators=self.n_estimators,
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
		return [ 'prediction', 'max_depth', 'random_state', 'accuracy', 'n_estimators',
		         'max_depth', 'criterior',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> RandomForestClassifier | None:
		"""

			Purpose:
			-----------
			Fit the classifier.


			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline


		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.random_forest_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

				Purpose:
				-------
				Predict class labels
				using the SGD classifier.

				Parameters:
				----------
					X (pd.DataFrame): Feature matrix.

				Returns:
				---------
					np.ndarray: Predicted class labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.random_forest_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
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
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				float: R^2 accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.random_forest_classifier.predict( X )
				self.accuracy = accuracy_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluate the Lasso model using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				dict: Dictionary of MAE, RMSE, R², etc.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.random_forest_classifier.predict( X )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared=False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared=False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.random_forest_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
				plt.xlabel( 'Predicted' )
				plt.ylabel( 'Actual' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx = None,
	                           resolution = 0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class GradientBoostingClassifier( Model ):
	"""

		Purpose:
		--------
		Gradient Boosting Classifier builds an additive model in a forward stage-wise fashion;
		it allows for the optimization  of arbitrary differentiable loss functions.
		In each stage n_classes_ regression trees are  fit on the negative gradient of the binomial
		or multinomial deviance loss function. Binary classification is a special case where
		only a single regression tree is induced.

		The feature_names are always randomly permuted at each split. Therefore, the best found split
		may vary, even with the same training data and max_features=n_features, if the improvement
		of the criterion is identical for several splits enumerated during the search of the best
		split. To obtain a deterministic behaviour during fitting, rando has to be fixed.

	"""
	gradient_boost_classifier: ske.GradientBoostingClassifier
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]


	def __init__( self, lss: str='deviance', rate: int=0.1,
	              est: int=100, max: int=3, rando: int=42 ) -> None:
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
		self.max_depth = max
		self.random_state = rando
		self.gradient_boost_classifier = ske.GradientBoostingClassifier( loss=self.loss,
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
		return [ 'prediction', 'max_depth', 'random_state', 'accuracy',
		         'loss', 'learning_rate', 'n_estimators', 'gradient_boost_classifier',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoostingClassifier | None:
		"""

			Purpose:
			________
				Fit the model to the training df.

			Parameters:
			__________
				X (np.ndarray): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.gradient_boost_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'GradientBoostClassification'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoostingClassifier'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			________
				Predict class labels.

			Parameters:
			__________
				X (np.ndarray): Feature matrix.

			Returns:
			________
				np.ndarray: Predicted labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.gradient_boost_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'GradientBoostClassification'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			_______
				Compute classification accuracy.

			Parameters:
			__________
				X (np.ndarray): Features.
				y (np.ndarray): Ground truth labels.

			Returns:
			________
				float: Accuracy accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.gradient_boost_classifier.predict( X )
				self.accuracy = accuracy_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'GradientBoostingClassifier'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			--------
				Evaluate classifier using multiple metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix.
				y (np.ndarray): Ground truth labels.

			Returns:
			--------
				Dict[str, float]: Evaluation scores.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.gradient_boost_classifier.predict( X )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared=False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth labels.

		"""
		self.prediction = self.gradient_boost_classifier.predict( X )
		cm = confusion_matrix( y, self.prediction )
		sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
		plt.xlabel( 'Predicted' )
		plt.ylabel( 'Actual' )
		plt.title( 'Confusion Matrix' )
		plt.tight_layout( )
		plt.show( )

	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx = None,
	                           resolution = 0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class AdaBoostClassifier( Model ):
	"""

		Purpose:
		---------
		An Boost classifier is a meta-estimator that begins by fitting a classifier
		on the original dataset and then fits additional copies of the classifier on the
		same dataset but where the weights of incorrectly classified instances are
		adjusted such that subsequent classifiers focus more on difficult cases.

	"""
	ada_boost_classifier = ske.AdaBoostClassifier
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
		self.ada_boost_classifier = ske.AdaBoostClassifier( estimator=self.estimator,
			n_estimators=self.n_estimators, learning_rate=self.learning_rate  )
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
		return [ 'prediction', 'max_depth', 'random_state', 'accuracy',
		         'X_scaled', 'n_estimators', 'learning_rate', 'ada_boost_classifier',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> AdaBoostClassifier | None:
		"""

			Purpose:
			_______
				Fit the classifier.

			Parameters:
			_________
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.ada_boost_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Predict class labels
			using the SGD classifier.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
			np.ndarray: Predicted class labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.ada_boost_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				float: R^2 accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ada_boost_classifier.predict( X )
				self.accuracy = accuracy_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				dict: Dictionary of MAE, RMSE, R², etc.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ada_boost_classifier.predict( X )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared = False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ada_boost_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
				plt.xlabel( 'Predicted' )
				plt.ylabel( 'Actual' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx = None,
	                           resolution = 0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class BaggingClassifier( Model ):
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

	def __init__( self, base: object=None, num: int=10, max: int=1, rando: int=42 ) -> None:
		"""

			Initialize the BaggingClassifier.

		"""
		super( ).__init__( )
		self.base_estimator = base
		self.n_estimators = num
		self.max_features = max
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
		return [ 'prediction', 'max_depth', 'random_state', 'accuracy',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> BaggingClassifier | None:
		"""

			Purpose:
			--------
			 Fit the classifier.

			Parameters:
			----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			-------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.bagging_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Predict class labels
			using the SGD classifier.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted class labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.bagging_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				float: R^2 accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.bagging_classifier.predict( X )
				self.accuracy = accuracy_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				dict: Dictionary of MAE, RMSE, R², etc.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					square=False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared=False )
				return \
					{
							'MAE': self.mean_absolute_error,
							'MSE': self.mean_squared_error,
							'RMSE': self.r_mean_squared_error,
							'R2': self.r2_score,
							'Explained Variance': self.explained_variance_score,
							'Median Absolute Error': self.median_absolute_error,
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
			---------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.bagging_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
				plt.xlabel( 'Predicted' )
				plt.ylabel( 'Actual' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx = None,
	                           resolution = 0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class VotingClassifier( Model ):
	"""

		Purpose:
		--------
		The idea behind the VotingClassifier is to combine conceptually different machine rate
		classifiers and use a majority vote or the average predicted probabilities (soft vote)
		to predict the class labels. Such a classifier can be useful for a set of equally well
		performing model in order to balance out their individual weaknesses.

	"""
	voting_classifier: ske.VotingClassifier
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


	def __init__( self, estimators: List[ (str, object) ], vote='hard' ) -> None:
		"""

			Initialize the RandomForestClassifier.

		"""
		super( ).__init__( )
		self.estimators = estimators
		self.voting = vote
		self.voting_classifier = ske.VotingClassifier( estimators=self.estimators,
			voting=self.voting )
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
		return [ 'prediction', 'max_depth', 'random_state', 'accuracy',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> VotingClassifier | None:
		"""

			Purpose:
			---------
				Fit the classifier.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.voting_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Predict class labels
			using the SGD classifier.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted class labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.voting_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				float: R^2 accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.voting_classifier.predict( X )
				self.accuracy = accuracy_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				dict: Dictionary of MAE, RMSE, R², etc.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.voting_classifier.predict( X )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared = False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.voting_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
				plt.xlabel( 'Predicted' )
				plt.ylabel( 'Actual' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx = None,
	                           resolution = 0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class StackingClassifier( Model ):
	"""

		Purpose:
		-------
		Stack of estimators with a final classifier. Stacked generalization consists in stacking the
		output of individual estimator and use a classifier to compute the final prediction.
		Stacking allows to use the strength of each individual estimator by using their output
		as input of a final estimator. Note that estimators_ are fitted on the full X while
		final_estimator_ is trained using cross-validated predictions of the base
		estimators using cross_val_predict.

	"""
	stacking_classifier: ske.StackingClassifier
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

	def __init__( self, est: List[ Tuple[ str, ClassifierMixin ] ],
	              final: Optional[ ClassifierMixin ]=None ) -> None:
		"""

			Initialize the RandomForestClassifier.

		"""
		super( ).__init__( )
		self.estimators = est
		self.final_estimator = final
		self.stacking_classifier = ske.StackingClassifier( estimators=self.estimators,
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
		return [ 'prediction', 'accuracy', 'final_estimator', 'estimators', 'stacking_classifier',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> StackingClassifier | None:
		"""

			Purpose:
			---------
				Fit the classifier.

			Parameters:
			----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.stacking_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Predict class labels
			using the SGD classifier.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
			np.ndarray: Predicted class labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.stacking_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
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
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				float: R^2 accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stacking_classifier.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Stack Classifier model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target labels.

			Returns:
			-----------
				dict: Dictionary of MAE, RMSE, R², etc.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stacking_classifier.predict( X )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared=False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingClassifier'
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
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stacking_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
				plt.xlabel( 'Predicted' )
				plt.ylabel( 'Actual' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx = None,
	                           resolution = 0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

class SupportVectorClassifier:
	"""

		Support Vector Classification (SVC).

	"""
	svc_classifier: skv.SVC
	kernel: str
	regulation: float
	degree: int
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]

	def __init__( self, kernel: str='rbf', C: float=1.0, degree: int=3 ) -> None:
		"""
		
			Purpose:
			---------			
			Initialize the SVC model.
	
			:param kernel: Kernel type to be used in the algorithm.
			:type kernel: str
			:param C: Regularization parameter.
			:type C: float
			
		"""
		self.kernel = kernel
		self.regulation = C
		self.degree = degree
		self.svc_classifier = skv.SVC( kernel=self.kernel, C=self.regulation,
			random_state=self.random_state, degree=self.degree )
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
		return [ 'prediction', 'max_depth', 'random_state', 'accuracy',
		         'svc_classifier', 'kernel', 'regulation', 'degree',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_heatmap' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> SupportVectorClassifier | None:
		"""
		
			Purpose:
			---------
			Fit the SVC model to the data.
	
			:param X: Input feature_names.
			:type X: np.ndarray
			:param y: Target labels.
			:type y: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.svc_classifier.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SupportVectorClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
			
			Purpose:
			--------
			Predict class labels for the input feature_names.
	
			:param X: Input feature_names.
			:type X: np.ndarray
			:return: Predicted class labels.
			:rtype: np.ndarray
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.svc_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SupportVectorClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> SupportVectorClassifier'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y_true: np.ndarray ) -> float | None:
		"""
		
			Purpose:
			---------
			Evaluate the model using accuracy score.
	
			:param X: Input feature_names.
			:type X: np.ndarray
			:param y_true: True labels.
			:type y_true: np.ndarray
			:return: Accuracy score.
			:rtype: float
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" cannot be None')
			elif y_true is None:
				raise Exception( 'The argument "y_true" cannot be None')
			else:
				self.prediction = self.svc_classifier.predict( X )
				self.accuracy = accuracy_score( y_true, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SupportVectorClassifier'
			exception.method = 'score( self, X: np.ndarray, y_true: np.ndarray ) -> float '
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y_true: np.ndarray ) -> str | None:
		"""
		
			Purpose:
			----------
			Generate classification report.
	
			:param self:
			:type self:
			:param X: Input feature_names.
			:type X: np.ndarray
			:param y_true: True labels.
			:type y_true: np.ndarray
			:return: Classification report.
			:rtype: str
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" cannot be None' )
			elif y_true is None:
				raise Exception( 'The argument "y_true" cannot be None' )
			else:
				self.prediction = self.svc_classifier.predict( X )
				return classification_report( y_true, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SupportVectorClassifier'
			exception.method = 'score( self, X: np.ndarray, y_true: np.ndarray ) -> float '
			error = ErrorDialog( exception )
			error.show( )


	def create_heatmap( self, X: np.ndarray, y_true: np.ndarray ) -> None:
		"""
		
			Purpose:
			---------
			Generate and display a confusion matrix.
	
			:param self:
			:type self:
			:param X: Input feature_names.
			:type X: np.ndarray
			:param y_true: True labels.
			:type y_true: np.ndarray
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" cannot be None' )
			elif y_true is None:
				raise Exception( 'The argument "y_true" cannot be None' )
			else:
				self.prediction = self.svc_classifier.predict( X )
				cm = confusion_matrix( y_true, self.prediction )
				sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
				plt.xlabel( 'Projected' )
				plt.ylabel( 'Observed' )
				plt.title( 'Confusion Matrix' )
				plt.tight_layout( )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SupportVectorClassifier'
			exception.method = 'create_heatmap( self, X: np.ndarray, y_true: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def plot_decision_regions( self, X: np.ndarray, y: np.ndarray, test_idx = None,
	                           resolution = 0.02 ):
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
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				# setup marker generator and color map
				markers = ('o', 's', '^', 'v', '<')
				colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
				cmap = ListedColormap( colors[ :len( np.unique( y ) ) ] )

				# plot the decision surface
				x1_min, x1_max = X[ :, 0 ].min( ) - 1, X[ :, 0 ].max( ) + 1
				x2_min, x2_max = X[ :, 1 ].min( ) - 1, X[ :, 1 ].max( ) + 1
				xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution ),
					np.arange( x2_min, x2_max, resolution ) )
				lab = self.project( np.array( [ xx1.ravel( ), xx2.ravel( ) ] ).T )
				lab = lab.reshape( xx1.shape )
				plt.contourf( xx1, xx2, lab, alpha = 0.3, cmap = cmap )
				plt.xlim( xx1.min( ), xx1.max( ) )
				plt.ylim( xx2.min( ), xx2.max( ) )

				# plot class examples
				for idx, cl in enumerate( np.unique( y ) ):
					plt.scatter( x = X[ y == cl, 0 ], y = X[ y == cl, 1 ], alpha = 0.8,
						c = colors[ idx ],
						marker = markers[ idx ], label = f'Class {cl}', edgecolor = 'black' )

					# plot all examples
					if test_idx:
						X_test, y_test = X[ test_idx, : ], y[ test_idx ]
						plt.scatter( X_test[ :, 0 ], X_test[ :, 1 ], c = 'none',
							edgecolor = 'black', alpha = 1.0, linewidth = 1,
							marker = 'o', s = 100, label = 'Test set' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'plot_decision_regions( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )

