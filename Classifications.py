'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                Classifications.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="Classifications.py" company="Terry D. Eppler">

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
	Classifications.py
</summary>
******************************************************************************************
'''
from __future__ import annotations

from typing import Dict
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
	RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
	BaggingClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import (
	RidgeClassifier, SGDClassifier, Perceptron,
)
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, precision_score, f1_score, roc_auc_score,
                             matthews_corrcoef, recall_score, classification_report)
from sklearn.metrics import (
	r2_score, mean_squared_error, mean_absolute_error,
	explained_variance_score, median_absolute_error
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, MinMaxScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Booger import Error, ErrorDialog

class Model( BaseModel ):
	"""

		Purpose:
		---------
		Abstract base class that defines the interface for all linerar_model wrappers.

	"""

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )
		self.pipeline = None

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

	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Generate predictions from  the trained linerar_model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).

			Returns:
			-----------
				np.ndarray: Predicted target_values or class labels.

		"""
		raise NotImplementedError

	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			---------
			Compute the core metric (e.g., R²) of the model on test df.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): True target target_values.

			Returns:
			-----------
				float: Score value (e.g., R² for regressors).

		"""
		raise NotImplementedError

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Purpose:
			---------
			Evaluate the model using multiple performance metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target_values.

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

	def __init__( self, reg: float = 0.0001, max: int = 1000, mix: bool = True ) -> None:
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
		self.perceptron_classifier: Perceptron = Perceptron( alpha = reg, max_iter = max,
			shuffle = mix )
		self.prediction: np.array = None
		self.mean_absolute_error: float = 0.0
		self.mean_squared_error: float = 0.0
		self.r_mean_squared_error: float = 0.0
		self.r2_score: float = 0.0
		self.explained_variance_score: float = 0.0
		self.median_absolute_error: float = 0.0
		self.accuracy: float = 0.0

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
			Fit the PerceptronClassifier linerar_model.

			Parameters:
			---------
			X (pd.DataFrame): Feature matrix.
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
				X (np.ndarray): Test features.
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

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict | None:
		"""


			Purpose:
			-----------
			Evaluate classifier performance using standard classification metrics.

			Parameters:
			---------
				X (np.ndarray): Input features of shape (n_samples, n_features).
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
				X (np.ndarray): Input features.
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
				ConfusionMatrixDisplay( confusion_matrix = cm )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class MultilayerClassification( Model ):
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

	def __init__( self, hidden: tuple = (100,), activation = 'relu', solver = 'adam',
	              alpha = 0.0001,
	              learning: str = 'constant', rando: int = 42 ) -> None:
		super( ).__init__( )
		self.hidden_layers = hidden
		self.activation_function = activation
		self.learning_rate = learning
		self.solver = solver
		self.alpha = alpha
		self.random_state = rando
		self.multilayer_classifier: MLPClassifier = MLPClassifier( hidden_layer_sizes = hidden,
			activation = activation, solver = solver, alpha = alpha, learning_rate = learning,
			random_state = 42 )
		self.pipeline: Pipeline = Pipeline( steps = list( hidden ) )
		self.prediction: np.array = None
		self.accuracy: float = 0.0
		self.mean_absolute_error: float = 0.0
		self.mean_squared_error: float = 0.0
		self.r_mean_squared_error: float = 0.0
		self.r2_score: float = 0.0
		self.explained_variance_score: float = 0.0
		self.median_absolute_error: float = 0.0

	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
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
			exception.cause = 'MultilayerRegression'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )

	def transform( self, X: np.ndarray ) -> np.ndarray | None:
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
				self.prediction = self.pipeline.transform( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegression'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray | None:
		"""


			Purpose:
			-----------
			Fits and transforms all pipeline steps on the text df.

			Parameters:
			-----------
				X (np.ndarray): Input feature matrix.
				y (Optional[np.ndarray]): Optional target array.

			Returns:
			-----------
				np.ndarray: Transformed feature matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.multilayer_classifier.fit_transform( X, y )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegression'
			exception.method = ('fit_transform( self, X: np.ndarray, y: '
			                    'Optional[ np.ndarray ]=None ) -> np.ndarray')
			error = ErrorDialog( exception )
			error.show( )

	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			-----------
				Compute the R^2 accuracy of the model on the given test df.

			Parameters:
			-----------
				X (np.ndarray): Test features.
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

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict | None:
		"""

			Purpose:
			-----------
			Evaluate the model using multiple regression metrics.


			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target_values.

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
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average = 'binary' )
				self.recall = mean_squared_error( y, self.prediction, average = 'binary' )
				self.f1_score = f1_score( y, self.prediction, average = 'binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
					{
							'Accuracy': self.accuracy,
							'Precision': self.precision,
							'Recall': self.recall,
							'F1 Score': self.f1_score,
							'ROC AUC': self.roc_auc_score,
							'Correlation Coeff': self.correlation_coefficient
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerClassification'
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
				X (np.ndarray): Input features.
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
				ConfusionMatrixDisplay( confusion_matrix = cm )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class RidgeClassification( Model ):
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
		or precision/recall, while the penalized least squares loss used by the RidgeClassification
		allows for a very different choice of the numerical solvers with
		distinct computational performance profiles.

	"""

	def __init__( self, alpha: float = 1.0, solver: str = 'auto', max: int = 1000,
	              rando: int = 42 ) -> None:
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
		self.alpha: float = alpha
		self.solver: str = solver
		self.max_iter: int = max
		self.random_state: int = rando
		self.ridge_classifier: RidgeClassifier = RidgeClassifier( alpha = self.alpha,
			solver = self.solver, max_iter = self.max_iter, random_state = self.random_state )
		self.prediction: np.array = None
		self.accuracy: float = 0.0
		self.precision: float = 0.0
		self.recall: float = 0.0
		self.f1_score: float = 0.0
		self.roc_auc_score: float = 0.0
		self.correlation_coefficient: float = 0.0

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
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
				self.ridge_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )

	def project( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Project target target_values using the RidgeRegressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted target target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.ridge_regressor.predict( X )
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target_values.

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
				self.prediction = self.ridge_regressor.predict( X )
				return r2_score( y, self.prediction )
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
				y (np.ndarray): Ground truth target target_values.

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
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average = 'binary' )
				self.recall = mean_squared_error( y, self.prediction, average = 'binary' )
				self.f1_score = f1_score( y, self.prediction, average = 'binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
					{
							'Accuracy': self.accuracy,
							'Precision': self.precision,
							'Recall': self.recall,
							'F1 Score': self.f1_score,
							'ROC AUC': self.roc_auc_score,
							'Correlation Coeff': self.correlation_coefficient
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
			Plot predicted vs actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

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
				self.prediction = self.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Ridge Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StochasticDescentClassification( Model ):
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
		 values for the features. The model it fits can be controlled with the loss parameter;
		 by default, it fits a linear support vector machine (SVM).

		The regularizer is a penalty added to the loss function that shrinks model parameters
		towards the zero vector using either the squared euclidean norm L2 or the absolute norm
		L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value
		because of the regularizer, the update is truncated to 0.0 to allow for learning sparse
		 models and achieve online feature selection.

	"""

	def __init__( self, loss: str = 'hinge', max: int = 5, reg: str = 'l2' ) -> None:
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
		self.loss: str = loss
		self.max_iter: int = max
		self.regularization: str = reg
		self.stochastic_classifier: SGDClassifier = SGDClassifier( loss = self.loss,
			max_iter = self.max_iter, penalty = self.regularization )
		self.prediction: np.array = None
		self.mean_absolute_error: float = 0.0
		self.mean_squared_error: float = 0.0
		self.r_mean_squared_error: float = 0.0
		self.r2_score: float = 0.0
		self.explained_variance_score: float = 0.0
		self.median_absolute_error: float = 0.0

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
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
			X (np.ndarray): Test features.
			y (np.ndarray): Ground truth target target_values.

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
				return r2_score( y, self.prediction )
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
			X (np.ndarray): Input features.
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
				ConfusionMatrixDisplay( confusion_matrix = cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class NearestNeighborClassification( Model ):
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

	def __init__( self, num: int = 5 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the KNeighborsClassifier linerar_model.

			Attributes:
			-----------
				linerar_model (KNeighborsClassifier): Internal non-parametric classifier.
					Parameters:
						n_neighbors (int): Number of neighbors to use. Default is 5.

		"""
		super( ).__init__( )
		self.n_neighbors: int = num
		self.neighbor_classifier: KNeighborsClassifier = KNeighborsClassifier(
			n_neighbors = self.n_neighbors )
		self.prediction: np.array = None
		self.score: float = 0.0
		self.mean_absolute_error: float = 0.0
		self.mean_squared_error: float = 0.0
		self.r_mean_squared_error: float = 0.0
		self.r2_score: float = 0.0
		self.explained_variance_score: float = 0.0
		self.median_absolute_error: float = 0.0

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
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
				X (np.ndarray): Test features.
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
				return accuracy_score( y, self.neighbor_classifier.predict( X ) )
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
				X (np.ndarray): Input features.
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
				ConfusionMatrixDisplay( confusion_matrix = cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class DecisionTreeClassification( Model ):
	'''

		Purpose:
		--------
		Decision Trees (DTs) are a non-parametric supervised learning method used for
		classification. The goal is to create a model that predicts the value of a
		target variable by learning simple decision rules inferred from the data features.

		A tree can be seen as a piecewise constant approximation. Decision trees learn from data
		to approximate a sine curve with a set of if-then-else decision rules.
		The deeper the tree, the more complex the decision rules and the fitter the model.

	'''

	def __init__( self, criterion = 'gini', splitter = 'best', depth = 3, rando: int = 42 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the KNeighborsClassifier linerar_model.

			Attributes:
			-----------
				linerar_model (KNeighborsClassifier): Internal non-parametric classifier.
					Parameters:
						n_neighbors (int): Number of neighbors to use. Default is 5.

		"""
		super( ).__init__( )
		self.criterion: str = criterion
		self.splitter: str = splitter
		self.max_depth: int = depth
		self.random_state = rando
		self.dt_classifier: DecisionTreeClassifier = DecisionTreeClassifier(
			criterion = self.criterion,
			splitter = self.splitter, max_depth = self.max_depth )
		self.prediction: np.array = None
		self.score: float = 0.0
		self.mean_absolute_error: float = 0.0
		self.mean_squared_error: float = 0.0
		self.r_mean_squared_error: float = 0.0
		self.r2_score: float = 0.0
		self.explained_variance_score: float = 0.0
		self.median_absolute_error: float = 0.0

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
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
			exception.cause = 'DecisionTreeClassification'
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
			exception.cause = 'DecisionTreeClassification'
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
				X (np.ndarray): Test features.
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
				return accuracy_score( y, self.dt_classifier.predict( X ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeClassification'
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
			exception.cause = 'DecisionTreeClassification'
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
				X (np.ndarray): Input features.
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
				ConfusionMatrixDisplay( confusion_matrix = cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class RandomForestClassification( Model ):
	"""

		Purpose:
		--------
		In random forests, each tree in the ensemble is built from a sample drawn with replacement
		(i.e., a bootstrap sample) from the training set.

		Furthermore, when splitting each node during the construction of a tree,
		the best split is found either from all input features or a random subset of
		size max_features.

		The purpose of these two sources of randomness is to decrease the variance
		of the forest estimator. Individual decision trees typically exhibit high variance
		and tend to overfit. The injected randomness in forests yield decision trees with
		decoupled prediction errors. By taking an average of those predictions,
		errors can cancel out. Random forests achieve a reduced variance
		by combining diverse trees, sometimes at the cost of a slight increase in bias.
		The variance reduction is often significant hence yielding an overall better model.

	"""

	def __init__( self, est: int = 10, crit: str = 'gini', max: int = 3, rando: int = 42 ) -> None:
		"""

			Purpose:
			-----------
			Initializes the RandomForestClassification.

		"""
		super( ).__init__( )
		self.n_estimators: int = est
		self.criterion: str = crit
		self.max_depth: int = max
		self.random_state: int = rando
		self.random_forest_classifier: RandomForestClassifier = RandomForestClassifier(
			n_estimators = est,
			criterion = crit, random_state = rando )
		self.prediction: np.array = None
		self.accuracy: float = 0.0
		self.precision: float = 0.0
		self.recall: float = 0.0
		self.f1_score: float = 0.0
		self.roc_auc_score: float = 0.0
		self.correlation_coefficient: float = 0.0

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
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
			exception.cause = 'RandomForestClassification'
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
			exception.cause = 'RandomForestClassification'
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

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
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassification'
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
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

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
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average = 'binary' )
				self.recall = mean_squared_error( y, self.prediction, average = 'binary' )
				self.f1_score = f1_score( y, self.prediction, average = 'binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
					{
							'Accuracy': self.accuracy,
							'Precision': self.precision,
							'Recall': self.recall,
							'F1 Score': self.f1_score,
							'ROC AUC': self.roc_auc_score,
							'Correlation Coeff': self.correlation_coefficient
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassification'
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
				X (np.ndarray): Input features.
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
				ConfusionMatrixDisplay( confusion_matrix = cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class GradientBoostingClassification( Model ):
	"""

		Purpose:
		--------
		Gradient Boosting Classifier builds an additive model in a forward stage-wise fashion;
		it allows for the optimization  of arbitrary differentiable loss functions.
		In each stage n_classes_ regression trees are  fit on the negative gradient of the binomial
		or multinomial deviance loss function. Binary classification is a special case where
		only a single regression tree is induced.

		The features are always randomly permuted at each split. Therefore, the best found split
		may vary, even with the same training data and max_features=n_features, if the improvement
		of the criterion is identical for several splits enumerated during the search of the best
		split. To obtain a deterministic behaviour during fitting, rando has to be fixed.

	"""

	def __init__( self, lss: str = 'deviance', rate: int = 0.1,
	              est: int = 100, max: int = 3, rando: int = 42 ) -> None:
		"""

			Purpose:
			________
			Initialize the GradientBoostingClassification.

			Parameters:
			___________
			lss: str
			rate: int
			estimators: int
			max: int
			rando: int

		"""
		super( ).__init__( )
		self.loss: str = lss
		self.learning_rate: float = rate
		self.n_estimators: int = est
		self.max_depth: int = max
		self.random_state: int = rando
		self.gradient_boost_classifier = GradientBoostingClassifier( loss = lss,
			learning_rate = rate,
			n_estimators = est, max_depth = max, random_state = rando )
		self.prediction: np.array = None
		self.accuracy: float = 0.0
		self.precision: float = 0.0
		self.recall: float = 0.0
		self.f1_score: float = 0.0
		self.roc_auc_score: float = 0.0
		self.correlation_coefficient: float = 0.0
		self.median_absolute_error: float = 0.0

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
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
		self.gradient_boost_classifier.fit( X, y )
		return self

	def project( self, X: np.ndarray ) -> np.ndarray:
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
		self.prediction = self.gradient_boost_classifier.predict( X )
		return self.prediction

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
		self.prediction = self.gradient_boost_classifier.predict( X )
		self.accuracy = accuracy_score( y, self.prediction )
		return self.accuracy

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
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
		self.prediction = self.gradient_boost_classifier.predict( X )
		return \
			{
					"Accuracy": accuracy_score( y, self.prediction ),
					"Precision": precision_score( y, self.prediction, average = 'binary' ),
					"Recall": recall_score( y, self.prediction, average = 'binary' ),
					"F1 Score": f1_score( y, self.prediction, average = 'binary' ),
					"ROC AUC": roc_auc_score( y, self.prediction ),
					"Matthews Corrcoef": matthews_corrcoef( y, self.prediction )
			}

	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
				Display the confusion matrix.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth labels.

		"""
		self.prediction = self.gradient_boost_classifier.predict( X )
		cm = confusion_matrix( y, self.prediction )
		ConfusionMatrixDisplay( confusion_matrix = cm ).plot( cmap = 'Oranges' )
		plt.title( "Gradient Boosting Classifier Confusion Matrix" )
		plt.grid( False )
		plt.show( )


class AdaBoostClassification( Model ):
	"""

		Purpose:
		---------
		An Boost classifier is a meta-estimator that begins by fitting a classifier
		on the original dataset and then fits additional copies of the classifier on the
		same dataset but where the weights of incorrectly classified instances are
		adjusted such that subsequent classifiers focus more on difficult cases.

	"""

	def __init__( self, est: int = 100, max: int = 3 ) -> None:
		"""

			Initialize the Random Forest Classifier.

		"""
		super( ).__init__( )
		self.scaler_type = 'AdaBoostClassifier'
		self.max_depth: int = max
		self.n_estimators: int = est
		self.ada_boost_classifier: AdaBoostClassifier = AdaBoostClassifier( n_estimators = est )
		self.prediction: np.array = None
		self.accuracy: float = 0.0
		self.precision: float = 0.0
		self.recall: float = 0.0
		self.f1_score: float = 0.0
		self.roc_auc_score: float = 0.0
		self.correlation_coefficient: float = 0.0

	def scale( self ) -> None:
		"""

			Purpose:
			-----------
			Scale numeric features using selected scaler.


		"""
		if self.scaler_type is None:
			return

		scaler = \
			{
					'standard': StandardScaler( ),
					'minmax': MinMaxScaler( )
			}.get( self.scaler_type )

		if scaler is None:
			raise ValueError( 'Scaler must be standard or minmax.' )

		scaled_array = scaler.fit_transform( self.X[ self.numeric_features ] )
		scaled_df = pd.DataFrame( scaled_array, columns = self.numeric_features,
			index = self.X.index )

		# Combine scaled numeric with untouched categorical
		self.X_scaled = pd.concat( [ scaled_df, self.X[ self.categorical_features ] ], axis = 1 )

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
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
			exception.cause = 'AdaBoostClassification'
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
			exception.cause = 'AdaBoostClassification'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

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
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassification'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

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
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average = 'binary' )
				self.recall = mean_squared_error( y, self.prediction, average = 'binary' )
				self.f1_score = f1_score( y, self.prediction, average = 'binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
					{
							'Accuracy': self.accuracy,
							'Precision': self.precision,
							'Recall': self.recall,
							'F1 Score': self.f1_score,
							'ROC AUC': self.roc_auc_score,
							'Correlation Coeff': self.correlation_coefficient
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassification'
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
				X (np.ndarray): Input features.
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
				ConfusionMatrixDisplay( confusion_matrix = cm ).create_graph( )
				plt.title( 'ADA Boost Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BaggingClassification( Model ):
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

	def __init__( self, base: object = None, num: int = 10, max: int = 1, rando: int = 42 ) -> None:
		"""

			Initialize the BaggingClassification.

		"""
		super( ).__init__( )
		self.base_estimator: object = base
		self.n_estimators: int = num
		self.max_features: int = max
		self.random_state: int = rando
		self.bagging_classifier: BaggingClassifier = BaggingClassifier( n_estimators = num,
			max_features = self.max, random_state = rando )
		self.prediction: np.array = None
		self.accuracy: float = 0.0
		self.precision: float = 0.0
		self.recall: float = 0.0
		self.f1_score: float = 0.0
		self.roc_auc_score: float = 0.0
		self.correlation_coefficient: float = 0.0

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
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
			exception.cause = 'BaggingClassification'
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
			exception.cause = 'BaggingClassification'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

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
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassification'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

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
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average = 'binary' )
				self.recall = mean_squared_error( y, self.prediction, average = 'binary' )
				self.f1_score = f1_score( y, self.prediction, average = 'binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
					{
							'Accuracy': self.accuracy,
							'Precision': self.precision,
							'Recall': self.recall,
							'F1 Score': self.f1_score,
							'ROC AUC': self.roc_auc_score,
							'Correlation Coeff': self.correlation_coefficient
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassification'
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
				X (np.ndarray): Input features.
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
				ConfusionMatrixDisplay( confusion_matrix = cm ).create_graph( )
				plt.title( 'Bagging Classifier Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class VotingClassification( Model ):
	"""

		Purpose:
		--------
		The idea behind the VotingClassification is to combine conceptually different machine rate
		classifiers and use a majority vote or the average predicted probabilities (soft vote)
		to predict the class labels. Such a classifier can be useful for a set of equally well
		performing model in order to balance out their individual weaknesses.

	"""

	def __init__( self, estimators: List[ (str, object) ], vote = 'hard' ) -> None:
		"""

			Initialize the RandomForestClassification.

		"""
		super( ).__init__( )
		self.estimators: List[ (str, object) ] = estimators
		self.voting: str = vote
		self.voting_classifier: VotingClassifier = VotingClassifier( estimators = estimators,
			voting = vote )
		self.prediction: np.array = None
		self.accuracy: float = 0.0
		self.precision: float = 0.0
		self.recall: float = 0.0
		self.f1_score: float = 0.0
		self.roc_auc_score: float = 0.0
		self.correlation_coefficient: float = 0.0

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
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
			exception.cause = 'VotingClassification'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )

	def project( self, X: np.ndarray ) -> np.ndarray:
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
			exception.cause = 'VotingClassification'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

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
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassification'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

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
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average = 'binary' )
				self.recall = mean_squared_error( y, self.prediction, average = 'binary' )
				self.f1_score = f1_score( y, self.prediction, average = 'binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
					{
							'Accuracy': self.accuracy,
							'Precision': self.precision,
							'Recall': self.recall,
							'F1 Score': self.f1_score,
							'ROC AUC': self.roc_auc_score,
							'Correlation Coeff': self.correlation_coefficient
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassification'
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
				X (np.ndarray): Input features.
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
				ConfusionMatrixDisplay( confusion_matrix = cm ).create_graph( )
				plt.title( 'Voting Classifer Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StackClassification( Model ):
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

	def __init__( self, est: List[ Tuple[ str, ClassifierMixin ] ],
	              final: Optional[ ClassifierMixin ] = None ) -> None:
		"""

			Initialize the RandomForestClassification.

		"""
		super( ).__init__( )
		self.estimators: List[ Tuple[ str, ClassifierMixin ] ] = est
		self.final_estimator: ClassifierMixin = final
		self.stacking_classifier: StackingClassifier = StackingClassifier( estimators = est,
			final_estimator = final )
		self.prediction: np.array = None
		self.accuracy: float = 0.0
		self.precision: float = 0.0
		self.recall: float = 0.0
		self.f1_score: float = 0.0
		self.roc_auc_score: float = 0.0
		self.correlation_coefficient: float = 0.0

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
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
			exception.cause = 'StackClassification'
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
			exception.cause = 'StackClassification'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

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
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackClassification'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Evaluate the Stack Classifier model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

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
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average = 'binary' )
				self.recall = mean_squared_error( y, self.prediction, average = 'binary' )
				self.f1_score = f1_score( y, self.prediction, average = 'binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
					{
							'Accuracy': self.accuracy,
							'Precision': self.precision,
							'Recall': self.recall,
							'F1 Score': self.f1_score,
							'ROC AUC': self.roc_auc_score,
							'Correlation Coeff': self.correlation_coefficient
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackClassification'
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
				X (np.ndarray): Input features.
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
				ConfusionMatrixDisplay( confusion_matrix = cm ).create_graph( )
				plt.title( 'Stacking Classifer Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class SupportVectorClassification:
	"""

		Wrapper for sklearn's Support Vector Classification (SVC).

	"""

	def __init__( self, kernel: str = 'rbf', C: float = 1.0 ) -> None:
		"""
		Initialize the SVC model.

		:param kernel: Kernel type to be used in the algorithm.
		:type kernel: str
		:param C: Regularization parameter.
		:type C: float
		"""
		self.svc_model = SVC( kernel=kernel, C=C )
		self.prediction: np.array = None
		self.accuracy: float = 0.0
		self.precision: float = 0.0
		self.recall: float = 0.0
		self.f1_score: float = 0.0
		self.roc_auc_score: float = 0.0
		self.correlation_coefficient: float = 0.0


	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		Fit the SVC model to the data.

		:param X: Input features.
		:type X: np.ndarray
		:param y: Target labels.
		:type y: np.ndarray
		"""
		self.svc_model.fit( X, y )


	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
		Predict class labels for the input features.

		:param X: Input features.
		:type X: np.ndarray
		:return: Predicted class labels.
		:rtype: np.ndarray
		"""
		return self.svc_model.predict( X )


	def evaluate( self, X: np.ndarray, y_true: np.ndarray ) -> float:
		"""
		Evaluate the model using accuracy score.

		:param X: Input features.
		:type X: np.ndarray
		:param y_true: True labels.
		:type y_true: np.ndarray
		:return: Accuracy score.
		:rtype: float
		"""
		y_pred = self.svc_model.predict( X )
		return accuracy_score( y_true, y_pred )

	def analyze( self, X: np.ndarray, y_true: np.ndarray ) -> str:
		"""
		Generate classification report.

		:param self:
		:type self:
		:param X: Input features.
		:type X: np.ndarray
		:param y_true: True labels.
		:type y_true: np.ndarray
		:return: Classification report.
		:rtype: str
		"""
		y_pred = self.predict( X )
		return classification_report( y_true, y_pred )

	def create_matrix( self, X: np.ndarray, y_true: np.ndarray ) -> None:
		"""
		Generate and display a confusion matrix.

		:param self:
		:type self:
		:param X: Input features.
		:type X: np.ndarray
		:param y_true: True labels.
		:type y_true: np.ndarray
		"""
		y_pred = self.predict( X )
		cm = confusion_matrix( y_true, y_pred )
		sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues' )
		plt.xlabel( "Predicted" )
		plt.ylabel( "Actual" )
		plt.title( "Confusion Matrix" )
		plt.tight_layout( )
		plt.show( )