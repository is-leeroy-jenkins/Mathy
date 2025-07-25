'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                regressions.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="regressions.py" company="Terry D. Eppler">

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
	regressions.py
</summary>
******************************************************************************************
'''
from __future__ import annotations

from argparse import ArgumentError
from typing import Dict
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
	RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
	BaggingRegressor, VotingRegressor, StackingRegressor)
from sklearn.linear_model import (
	LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
	BayesianRidge, SGDRegressor, )
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, precision_score, f1_score, roc_auc_score,
                             matthews_corrcoef)
from sklearn.metrics import (
	r2_score, mean_squared_error, mean_absolute_error,
	explained_variance_score, median_absolute_error
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from booger import Error, ErrorDialog

class Model( BaseModel ):
	"""

		Purpose:
		---------
		Abstract base class that defines the interface for all linerar_model wrappers.

	"""
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )

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
				np.ndarray: Predicted target_values or class labels.

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
				y (np.ndarray): True target target_values.

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

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				dict: Dictionary containing multiple evaluation metrics.

		"""
		raise NotImplementedError


class MultilayerRegression( Model ):
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
	multilayer_regressor: MLPRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	random_state: Optional[ int ]
	alpha: Optional[ float ]
	learning: str
	activation_function: str
	solver: str
	hidden_layers: tuple


	def __init__( self, hidden: tuple=(100,), activ='relu', solver='adam',
	              alpha=0.0001, learning: str='constant', rando: int=42 ) -> None:
		super( ).__init__( )
		self.hidden_layers = hidden
		self.activation_function = activ
		self.learning = learning
		self.solver = solver
		self.alpha = alpha
		self.random_state = rando
		self.multilayer_regressor = MLPRegressor( hidden_layer_sizes=hidden,
			activation=activ, solver=solver, alpha=alpha, learning_rate=learning,
			random_state=42 )
		self.pipeline = Pipeline( steps=list( hidden ) )
		self.prediction: np.array = None
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
		return [ 'prediction', 'accuracy', 'learning', 'activation_function',
		         'hidden_layers', 'random_state', 'alpha', 'max_depth', 'mean_absolute_error',
		         'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> MultilayerRegression | None:
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
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.multilayer_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegression'
			exception.method = 'train( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> Pipeline'
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
				self.prediction = self.multilayer_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
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
				self.prediction = self.multilayer_regressor.predict( X )
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
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
				{
					"MAE": mean_absolute_error( y, self.prediction ),
					"MSE": mean_squared_error( y, self.prediction ),
					"RMSE": mean_squared_error( y, self.prediction, squared = False ),
					"R2": r2_score( y, self.prediction ),
					"Explained Variance": explained_variance_score( y, self.prediction ),
					"Median Absolute Error": median_absolute_error( y, self.prediction )
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
				Plot actual vs predicted target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True target target_values.

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
				plt.title( 'MLP: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class LinearRegressor( Model ):
	"""

		Purpose:
		-----------
		Ordinary Least Squares Regression fits a linear model with coefficients to minimize the
		residual sum of squares between the observed targets in the dataset, and the targets
		predicted by the linear approximation. The coefficient estimates for Ordinary Least Squares
		rely on the independence of the features.

		When features are correlated and the columns of the design matrix have an approximately
		linear dependence, the design matrix becomes close to singular and as a result,
		the least-squares estimate becomes highly sensitive to random errors in the observed target,
		producing a large variance. This situation of multicollinearity can arise, for example,
		when data are collected without an experimental design.

	"""
	linear_regressor: LinearRegression
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self ) -> None:
		"""

			Purpose:
			-----------
			Initialize the Linear Regression linerar_model.

			Parameters:
			-----------
			fit_intercept (bool): Whether to include an intercept term. Default is True.
			copy_X (bool): Whether to copy the feature matrix. Default is True.

		"""
		super( ).__init__( )
		self.linear_regressor = LinearRegression( fit_intercept=True,
			copy_X=True )
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
		return [ 'prediction', 'accuracy', 'learning_rate',
		         'n_estimators', 'random_state', 'loss', 'max_depth', 'mean_absolute_error',
		         'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]

	def train( self, X: np.ndarray, y: np.ndarray ) -> LinearRegressor | None:
		"""

			Purpose:
			-----------
			Fit the OLS regression linerar_model.

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
				self.linear_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			-----------
			Predict target target_values using the OLS linerar_model.

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
				self.prediction = self.linerar_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""


			Purpose:
			-----------
			Compute the R-squared accuracy of the OLS model.

			Parameters:
			-----------
			X (np.ndarray): Test features.
			y (np.ndarray): True target target_values.

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
				self.prediction = self.linerar_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
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
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
				{
					"MAE": mean_absolute_error( y, self.prediction ),
					"MSE": mean_squared_error( y, self.prediction ),
					"RMSE": mean_squared_error( y, self.prediction, squared = False ),
					"R2": r2_score( y, self.prediction ),
					"Explained Variance": explained_variance_score( y, self.prediction ),
					"Median Absolute Error": median_absolute_error( y, self.prediction )
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot actual vs predicted target_values.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): True target target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.linerar_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Linear Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class RidgeRegression( Model ):
	"""

		Purpose:
		--------
		Solves a regression model where the loss function is the linear least squares function and
		alpha is given by the l2-norm. Also known as Ridge Regression
		or Tikhonov alpha. This estimator has built-in support for
		multi-variate regression (i.e., when y is a 2d-array of shape (n_samples, n_targets))

		The complexity parameter  controls the amount of shrinkage: the larger the value of alpha,
		the greater the amount of shrinkage and thus the coefficients become
		more robust to collinearity.

		The algorithm used to fit the model is coordinate descent. To avoid unnecessary memory
		duplication the X argument of the fit method should be directly passed as a
		Fortran-contiguous numpy array. Regularization improves the conditioning of the problem
		and reduces the variance of the estimates. Larger values specify stronger alpha.
		Alpha corresponds to 1 / (2C) in other linear models such as LogisticRegression or LinearSVC.
		If an array is passed, penalties are assumed to be specific to the targets.

	"""
	ridge_regressor: Ridge
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	random_state: int
	learning_rate: float
	alpha: float
	max_iter: int
	solver: str


	def __init__( self, alpha: float=1.0, solver: str='auto', max: int=1000,
	              rando: int=42 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the RidgeRegressor linerar_model.

			Attributes:
			-----------
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.

		"""
		super( ).__init__( )
		self.alpha = alpha
		self.solver = solver
		self.max_iter = max
		self.random_state = rando
		self.ridge_regressor = Ridge( alpha=self.alpha, solver=self.solver,
			max_iter=self.max_iter, random_state=self.random_state )
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
		return [ 'prediction', 'accuracy', 'alpha',
		         'solver', 'random_state', 'max_iter', 'mean_absolute_error',
		         'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> RidgeRegression | None:
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
			exception.cause = 'RidgeRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
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
			exception.cause = 'RidgeRegression'
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
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
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
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
				{
					"MAE": mean_absolute_error( y, self.prediction ),
					"MSE": mean_squared_error( y, self.prediction ),
					"RMSE": mean_squared_error( y, self.prediction, squared = False ),
					"R2": r2_score( y, self.prediction ),
					"Explained Variance": explained_variance_score( y, self.prediction ),
					"Median Absolute Error": median_absolute_error( y, self.prediction )
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegression'
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
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class LassoRegression( Model ):
	"""

		Purpose:
		--------
		Linear Model trained with L1 for the regularizer. Regularization improves the
		conditioning of the problem and reduces the variance of the estimates. Larger values
		specify stronger alpha. Technically the Lasso model is optimizing the same
		objective function as the Elastic Net with l1_ratio=1.0 (no L2 penalty).
		The algorithm used to fit the model is coordinate descent.

		To avoid unnecessary memory duplication the X argument of the fit method should be directly
		passed as a Fortran-contiguous numpy array. Regularization improves the conditioning of the
		problem and reduces the variance of the estimates. Larger values specify stronger
		alpha. Alpha corresponds to 1 / (2C) in other linear models such as
		LogisticRegression or LinearSVC. If an array is passed, penalties are assumed to be
		specific to the targets. Hence they must correspond in number.

	"""
	lasso_regressor: Lasso
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	random_state: int
	learning_rate: float
	alpha: float
	max_iter: int
	solver: str


	def __init__( self, alph: float=1.0, max: int=500, rando: int=42 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the LassoRegression linerar_model.
		"""
		super( ).__init__( )
		self.alpha: float = alph
		self.max_iter: int = max
		self.random_state: int = rando
		self.lasso_regressor = Lasso( alpha=self.alpha, max_iter=self.max_iter,
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
		return [ 'prediction', 'accuracy',
		         'random_state', 'alpha', 'max_iter', 'mean_absolute_error',
		         'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			--------
			Fit the LassoRegression.

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
				self.lasso_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )

	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			-----------
			Predict target target_values using the LassoRegression linerar_model.


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
				self.prediction = self.lasso_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""


			Purpose:
			-----------
			Compute R^2 accuracy for the Lasso model.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target_values.

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
				self.prediction = self.lasso_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegression'
			exception.method = 'accuracy(self, X: np.ndarray, y: np.ndarray) -> float'
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
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = f1_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
				{
					"MAE": mean_absolute_error( y, self.prediction ),
					"MSE": mean_squared_error( y, self.prediction ),
					"RMSE": mean_squared_error( y, self.prediction, squared = False ),
					"R2": r2_score( y, self.prediction ),
					"Explained Variance": explained_variance_score( y, self.prediction ),
					"Median Absolute Error": median_absolute_error( y, self.prediction )
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )

	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot actual vs. predicted target_values.

			Parameters:
			-----------
				X (np.ndarray): Input feature matrix.
				y (np.ndarray): Ground truth target_values.

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
				plt.title( 'Lasso Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class ElasticNetRegression( Model ):
	"""

		Purpose:
		--------
		ElasticNet is a linear regression model trained with both and -norm regularization of the
		coefficients. This combination allows for learning a sparse model where few of the weights
		are non-zero like Lasso, while still maintaining the regularization properties of Ridge.
		We control the convex combination of and using the l1_ratio parameter.

		Elastic-net is useful when there are multiple features that are correlated with one another.
		Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

	"""
	elasticnet_regressor: ElasticNet
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	random_state: int
	ratio: float
	alpha: float
	max_iter: int
	selection: str


	def __init__( self, alpha: float=1.0, ratio: float=0.5, max: int=200,
	              rando: int=42, select: str='random' ) -> None:
		"""

			Purpose:
			-----------
			Initialize the ElasticNet Regressor linerar_model.


			Parameters:
			----------
			alpha (float): Overall alpha strength. Default is 1.0.
			ratio (float): Mixing parameter (0 = RidgeRegressor, 1 = LassoRegression). Default is 0.5.
			max (int): Maximum number of iterations. Default is 200.
			rando (int): Number of random iterations. Default is 42.
			select (str): selection

		"""
		super( ).__init__( )
		self.alpha = alpha
		self.ratio = ratio
		self.random_state = rando
		self.selection = select
		self.max_iter = max
		self.elasticnet_regressor = ElasticNet( alpha=self.alpha,
			l1_ratio=self.ratio, random_state=self.random_state,
			max_iter=self.max_iter, selection=self.selection )
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
		return [ 'prediction', 'accuracy', 'alpha',
		         'ratio', 'random_state', 'selection', 'max_iter', 'mean_absolute_error',
		         'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]

	def train( self, X: np.ndarray, y: np.ndarray ) -> ElasticNetRegression | None:
		"""


			Purpose:
			-----------
			Fit the ElasticNetRegression regression linerar_model.

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
				self.elasticnet_model.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			-----------
			Predict target target_values using the ElasticNetRegression linerar_model.

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
				self.prediction = self.elasticnet_model.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""


			Purpose:
			-----------
			Compute R^2 accuracy on the test set.

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
			else:
				self.prediction = self.elasticnet_model.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""


			Purpose:
			-----------
			Evaluate model performance using regression metrics.


			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				dict: Evaluation metrics.

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
				self.r2_score = r2_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
				{
					"MAE": mean_absolute_error( y, self.prediction ),
					"MSE": mean_squared_error( y, self.prediction ),
					"RMSE": mean_squared_error( y, self.prediction, squared = False ),
					"R2": r2_score( y, self.prediction ),
					"Explained Variance": explained_variance_score( y, self.prediction ),
					"Median Absolute Error": median_absolute_error( y, self.prediction )
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot actual vs. predicted regression output.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True target target_values.

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
				plt.title( 'ElasticNet Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class LogisticRegressor( Model ):
	"""

		Purpose:
		--------
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
	logistic_regressor: LogisticRegression
	prediction: Optional[ np.ndarray ]
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
	alpha: float
	max_iter: int
	solver: str

	def __init__( self, c: float=1.0, penalty: str='l2', max: int=1000,
	              solver: str='lbfgs' ) -> None:
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
		self.alpha = c
		self.penalty = penalty
		self.max_iter = max
		self.solver = solver
		self.logistic_regressor = LogisticRegression( C=self.alpha,
			max_iter=self.max_iter, solver=self.solver, penalty=self.penalty )
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
		return [ 'prediction', 'accuracy', 'penalty',
		         'solver', 'random_state', 'alpha', 'max_iter','mean_absolute_error',
		         'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			-----------
			Fit the logistic regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target class labels.

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
				self.logistic_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LogisticRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			-----------
			Predict class labels using the logistic regression linerar_model.

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
				self.prediction = self.logistic_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LogisticRegressor'
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
				X (np.ndarray): Test features.
				y (np.ndarray): True class labels.

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
				self.prediction = self.logistic_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LogisticRegressor'
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
				X (np.ndarray): Input features of shape (n_samples, n_features).
				y (np.ndarray): True labels of shape (n_samples,).

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
				self.r2_score = f1_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
				{
						"MAE": mean_absolute_error( y, self.prediction ),
						"MSE": mean_squared_error( y, self.prediction ),
						"RMSE": mean_squared_error( y, self.prediction, squared = False ),
						"R2": r2_score( y, self.prediction ),
						"Explained Variance": explained_variance_score( y, self.prediction ),
						"Median Absolute Error": median_absolute_error( y, self.prediction )
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LogisticRegressor'
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
				self.prediction = self.logistic_regressor.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix = cm )
				plt.title( 'Logistic Regression Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LogisticRegressor'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BayesianRidgeRegression( Model ):
	"""

		Purpose:
		--------
		Bayesian regression techniques can be used to include alpha parameters in the
		estimation procedure: the alpha parameter is not set in a hard sense
		but tuned to the df at hand. This can be done by introducing uninformative priors over
		the hyper parameters of the model. The alpha used in Ridge regression and
		classification is equivalent to finding a maximum a posteriori estimation under a
		Gaussian prior over the coefficients with precision . Instead of setting lambda manually,
		it is possible to treat it as a random variable to be estimated from the df.

		This implementation is based on the algorithm described in Appendix A of (Tipping, 2001)
		where updates of the alpha parameters are done as suggested in (MacKay, 1992).
		Note that according to A New View of Automatic Relevance Determination
		(Wipf and Nagarajan, 2008) these update rules do not guarantee that the marginal likelihood
		is increasing between two consecutive iterations of the optimization.

	"""
	bayesian_ridge_regressor: BayesianRidge
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	scale_alpha: float
	shape_lambda: float
	shape_alpha: float
	max_iter: int
	shape_lambda: float
	scale_lambda: float


	def __init__( self, max: int=300, shape_alpha: float=1e-06,
	              scale_alpha: float=1e-06, shape_lambda: float=1e-06,
	              scale_lambda: float=1e-06 ) -> None:
		"""

			Purpose:
			-----------
				Initializes the BayesianRidgeRegression.

		"""
		super( ).__init__( )
		self.max_iter = max
		self.shape_alpha = shape_alpha
		self.scale_alpha = scale_alpha
		self.shape_lambda = shape_lambda
		self.scale_lambda = scale_lambda
		self.bayesian_ridge_regressor = BayesianRidge( alpha_1=self.shape_alpha, alpha_2=self.scale_alpha,
			lambda_1=self.shape_lambda, lambda_2=self.scale_lambda )
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
		return [ 'prediction', 'accuracy', 'shape_alpha', 'scale_alpha',
		         'shape_lambda', 'random_state', 'scale_lambda', 'max_iter','mean_absolute_error',
		         'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			-----------
				Fit the Bayesian RidgeRegressor
				regression linerar_model.

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
				self.bayesian_ridge_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRidgeRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			-----------
				Predicts target target_values
				using the Bayesian linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.bayesian_ridge_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRidgeRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			-----------
				Compute the R^2 accuracy
				of the model on test df.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): True target_values.

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
				self.prediction = self.bayesian_ridge_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRidgeRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluate the Bayesian model with regression metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): True target target_values.

			Returns:
			-----------
				dict: Dictionary of evaluation metrics.

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
					squared=False )
				self.r2_score = f1_score( y, self.prediction, average='binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
				{
					"MAE": mean_absolute_error( y, self.prediction ),
					"MSE": mean_squared_error( y, self.prediction ),
					"RMSE": mean_squared_error( y, self.prediction, squared = False ),
					"R2": r2_score( y, self.prediction ),
					"Explained Variance": explained_variance_score( y, self.prediction ),
					"Median Absolute Error": median_absolute_error( y, self.prediction )
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRidgeRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot predicted vs. actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True target target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.bayesian_ridge_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Bayesian-Ridge Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRidgeRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StochasticDescentRegression( Model ):
	"""

		Purpose:
		--------
		Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to discriminative
		rate of linear classifiers under convex loss functions such as
		(linear) Support VectorStore Machines and Logistic Regression. Even though SGD has been around
		in the machine rate community for a long time, it has received a considerable amount
		of attention just recently in the context of large-scale rate.

		SGD has been successfully applied to large-scale and sparse machine rate problems
		often encountered in text classification and natural language processing.
		Given that the df is sparse, the classifiers in this module easily scale to problems
		with more than 10^5 training examples and more than 10^5 features.

		The regularizer is a penalty added to the loss function that shrinks model parameters
		towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1
		or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value
		because of the regularizer, the update is truncated to 0.0 to allow for learning sparse
		models and achieve online feature selection.

		This implementation works with data represented as dense numpy arrays of floating point
		values for the features.

	"""
	stochastic_regressor = SGDRegressor
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	random_state: int
	penalty: str
	loss: str
	max_iter: int
	penalty: str

	def __init__( self, loss: str='hinge', max: int=5, penalty: str='l2' ) -> None:
		"""

			Purpose:
			-----------
			Initialize the SGDRegressor

			Parameters:
			-----------
			- alpha (float): Regulation
			- reg (str): Regularization term. Default is 'l2'.
			- max (int): Maximum number of passes. Default is 1000.

		"""
		super( ).__init__( )
		self.loss = loss
		self.max_iter = max
		self.penalty = penalty
		self.stochastic_regressor = SGDRegressor( loss=self.loss,
			max_iter=self.max_iter, penalty=self.penalty )
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
		return [ 'prediction', 'accuracy', 'penalty',
		         'max_iter', 'random_state', 'loss', 'mean_absolute_error',
		         'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]

	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			-----------
			Fit the SGD regressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target target_values.

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
				self.stochastic_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticDescentRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			-----------
			Predict target_values using the SGD regressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.stochastic_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticDescentRegression'
			exception.method = ''
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
				self.prediction = self.stochastic_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
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
			Evaluate regression model performance.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True target target_values.

			Returns:
			-----------
				dict: Evaluation metrics dictionary.

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
				self.r2_score = f1_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
				{
					"MAE": mean_absolute_error( y, self.prediction ),
					"MSE": mean_squared_error( y, self.prediction ),
					"RMSE": mean_squared_error( y, self.prediction, squared = False ),
					"R2": r2_score( y, self.prediction ),
					"Explained Variance": explained_variance_score( y, self.prediction ),
					"Median Absolute Error": median_absolute_error( y, self.prediction )
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticDescentRegression'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot predicted vs. actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True target target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stochastic_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Stochastic Gradient Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticDescentRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class NearestNeighborRegression( Model ):
	"""

		Purpose:
		--------
		The principle behind k-nearest neighbor methods is to find a predefined number of
		training samples closest in distance to the new point, and predict the label from these.
		The number of samples can be a user-defined constant (k-nearest neighbor rate),
		or vary based on the local density of points (radius-based neighbor rate).
		The distance can, in general, be any metric measure: standard Euclidean distance is the
		most common choice. Neighbors-based methods are known as non-generalizing
		machine rate methods, since they simply “remember” all of its training df
		(possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).

	"""
	neighbor_regressor: KNeighborsRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	n_neighbors: int
	algorithm: str
	power: float
	metric: str


	def __init__( self, num: int=5, algo: str='auto', p: float=2.0, metric: str='minkowski' ) -> None:
		"""


			Purpose:
			-----------
			Initialize the linerar_model (KNeighborsRegressor): Internal non-parametric regressor.

			Parameters:
			-----------
					Parameters:
						num: Number of neighbors to use. Default is 5.

		"""
		super( ).__init__( )
		self.n_neighbors = num
		self.algorithm = algo
		self.power = p
		self.metric = metric
		self.neighbor_regressor = KNeighborsRegressor( n_neighbors=self.n_neighbors,
			algorithm=self.algorithm, p=self.power, metric=self.metric )
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
		return [ 'prediction', 'accuracy', 'algorithm',
		         'n_neighbors', 'random_state', 'power', 'mean_absolute_error',
		         'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]



	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""


			Purpose:
			-----------
			Fit the KNN regressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target target_values.

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
				self.neighbor_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			-----------
			Predict target_values using the KNN regressor.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.neighbor_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			-----------
			Compute R^2 accuracy for k-NN regressor.

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
				self.prediction = self.neighbor_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluate k-NN regression performance with multiple metrics.


			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): True target target_values.

			Returns:
			-----------
				dict: Dictionary of evaluation scores.

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
				self.r2_score = f1_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
				{
					"MAE": mean_absolute_error( y, self.prediction ),
					"MSE": mean_squared_error( y, self.prediction ),
					"RMSE": mean_squared_error( y, self.prediction, squared = False ),
					"R2": r2_score( y, self.prediction ),
					"Explained Variance": explained_variance_score( y, self.prediction ),
					"Median Absolute Error": median_absolute_error( y, self.prediction )
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegression'
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
				self.prediction = self.neighbor_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Nearest-Neighbor Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class DecisionTreeRegression( Model ):
	'''

		Purpose:
		--------
		Decision Trees (DTs) are a non-parametric supervised learning method used for
		regression. The goal is to create a model that predicts the value of a
		target variable by learning simple decision rules inferred from the data features.

		A tree can be seen as a piecewise constant approximation. Decision trees learn from data
		to approximate a sine curve with a set of if-then-else decision rules.
		The deeper the tree, the more complex the decision rules and the fitter the model.

	'''
	dt_regressor: DecisionTreeRegression
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	criterion: str
	splitter: str
	max_depth: int
	random_state: int


	def __init__( self, criterion='squared_error', splitter='best', depth=3,
	              rando: int=42 ) -> None:
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
		self.dt_regresssor = DecisionTreeRegressor( criterion=self.criterion,
			splitter=self.splitter, max_depth=self.max_depth, random_state=rando )
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
		return [ 'prediction', 'accuracy', 'criterion',
		         'splitter', 'random_state', 'max_depth', 'mean_absolute_error',
		         'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> DecisionTreeRegression | None:
		"""


			Purpose:
			-----------
			Fit the Decision-Tree regressor linerar_model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Target target_values.

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
				self.dt_regresssor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			-----------
			Predict target_values using the KNN regressor.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.dt_regresssor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			-----------
			Compute R^2 accuracy for k-NN regressor.

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
				self.prediction = self.dt_regresssor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluate k-NN regression performance with multiple metrics.


			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): True target target_values.

			Returns:
			-----------
				dict: Dictionary of evaluation scores.

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
				self.r2_score = f1_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
					{
							"MAE": mean_absolute_error( y, self.prediction ),
							"MSE": mean_squared_error( y, self.prediction ),
							"RMSE": mean_squared_error( y, self.prediction, squared = False ),
							"R2": r2_score( y, self.prediction ),
							"Explained Variance": explained_variance_score( y, self.prediction ),
							"Median Absolute Error": median_absolute_error( y, self.prediction )
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeRegression'
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
				self.prediction = self.dt_regresssor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Decision Tree Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class RandomForestRegression( Model ):
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
		some errors can cancel out. Random forests achieve a reduced variance
		by combining diverse trees, sometimes at the cost of a slight increase in bias.
		The variance reduction is often significant hence yielding an overall better model.

	"""
	random_forest_regressor: RandomForestRegressor
	n_estimators: int
	random_state: int
	max_depth: int
	criterion: str
	learning_rate: float
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, est: int=10, crit: str='gini', max: int=3, rando: int=42 ) -> None:
		"""

			Purpose:
			-----------
			Initialize the RidgeRegressor linerar_model.

			Parameters:
			-----------
			alpha (float): Regularization strength. Default is 1.0.
			solver (str): Solver to use. Default is 'gini'.
			max (int): maximum iterations
			rando (int): random seed value

		"""
		super( ).__init__( )
		self.n_estimators = est
		self.criterion = crit
		self.max_depth = max
		self.random_state = rando
		self.random_forest_regressor = RandomForestRegressor(
			n_estimators=est, criterion=crit, random_state=rando )
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
		return [ 'prediction', 'accuracy', 'criterion',
		         'n_estimators', 'random_state', 'loss', 'max_depth',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]


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
				self.random_forest_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
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
				self.prediction = self.random_forest_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegression'
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
				self.prediction = self.random_forest_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			-----------
			Evaluates the Ridge model using multiple metrics.

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
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = f1_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
					{
							"MAE": mean_absolute_error( y, self.prediction ),
							"MSE": mean_squared_error( y, self.prediction ),
							"RMSE": mean_squared_error( y, self.prediction, squared = False ),
							"R2": r2_score( y, self.prediction ),
							"Explained Variance": explained_variance_score( y, self.prediction ),
							"Median Absolute Error": median_absolute_error( y, self.prediction )
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegression'
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
				self.prediction = self.random_forest_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Random Forest Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class GradientBoostingRegression( Model ):
	"""

		Purpose:
		--------
		Gradient Boosting builds an additive model in a forward stage-wise fashion;
		it allows for the optimization  of arbitrary differentiable loss functions.
		In each stage n_classes_ regression trees are  fit on the negative gradient of the binomial
		or multinomial deviance loss function. Binary classification is a special case where
		only a single regression tree is induced.

	"""
	gradient_boost_regressor: GradientBoostingRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	loss: str
	learning_rate: float
	n_estimators: int
	max_detpth: int


	def __init__( self, lss: str='deviance', rate: float=0.1,
	              est: int=100, max: int=3, rando: int=42 ) -> None:
		"""

			Purpose:
			_______
				Initialize the GradientBoostingRegression.

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
		self.gradient_boost_regressor = GradientBoostingRegressor( loss=lss, learning_rate=rate,
			n_estimators=est, max_depth=max, random_state=rando )
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
		return [ 'prediction', 'accuracy', 'learning_rate',
		         'n_estimators', 'random_state', 'loss', 'max_depth',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoostingRegression | None:
		"""

			Purpose:
			_______
				Fit the gradient boosting model.

			Parameters:
			___________
				X (np.ndarray): Training features.
				y (np.ndarray): Training targets.

			Returns:
			--------
				Pipeline

		"""
		self.gradient_boost_regressor.fit( X, y )
		return self


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			_________
			Predict regression targets.

			Parameters:
			_________
			X (np.ndarray): Feature matrix.

			Returns:
			________
			np.ndarray: Predicted target values.

		"""
		self.prediction = self.gradient_boost_regressor.predict( X )
		return self.prediction


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			________
			Compute the coefficient of determination R².

			Parameters:
			___________
			X (np.ndarray): Test features.
			y (np.ndarray): True target values.

			Returns:
			_______
			float: R² accuracy.

		"""
		self.prediction = self.gradient_boost_regressor.predict( X )
		self.accuracy = r2_score( y, self.prediction )
		return self.accuracy


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			_______
			Evaluate performance using standard regression metrics.

			Parameters:
			__________
			X (np.ndarray): Input features.
			y (np.ndarray): Ground truth target values.

			Returns:
			________
			Dict[str, float]: Evaluation metrics.

		"""
		self.prediction = self.gradient_boost_regressor.predict( X )
		return \
		{
			'MAE': mean_absolute_error( y, self.prediction ),
			'MSE': mean_squared_error( y, self.prediction ),
			'RMSE': mean_squared_error( y, self.prediction, squared = False ),
			'R2': r2_score( y, self.prediction ),
			'Explained Variance': explained_variance_score( y, self.prediction ),
			'Median Absolute Error': median_absolute_error( y, self.prediction )
		}


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			________
			Plot predicted vs actual target values.

			Parameters:
			___________
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Ground truth target values.

		"""
		self.prediction = self.gradient_boost_regressor.predict( X )
		plt.scatter( y, self.prediction, alpha = 0.6 )
		plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
		plt.xlabel( 'Observed' )
		plt.ylabel( 'Projected' )
		plt.title( 'Gradient-Boosting Regression: Observed vs Projected' )
		plt.grid( True )
		plt.show( )


class AdaBoostRegression( Model ):
	"""

		Purpose:
		---------
		An AdaBoost [1] regressor is a meta-estimator that begins by fitting a regressor on the
		original dataset and then fits additional copies of the regressor on the same dataset but
		where the weights of instances are adjusted according to the error of the current prediction.

		The core principle of Boost Regression is to fit a sequence of weak learners
		(i.e., models that are only slightly better than random guessing,
		such as small decision trees) on repeatedly modified versions of the df.
		The predictions from all of them are then combined through a weighted
		majority vote (or sum) to produce the final prediction.

	"""
	ada_boost_regressor: AdaBoostRegressor
	n_estimators: int
	random_state: int
	loss: str
	learning_rate: float
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, est: int=50, rando: int=42, loss: str='linear', learning: float=1.0 ) -> None:
		"""

			Purpose:
			--------
			Initialize the Ridge Regression Model.


			Parameters:
			----------
			estimators (int): The number of estimators used. Default is 100.
			max (int): The maximum number of iterations. Default is '3'.

		"""
		super( ).__init__( )
		self.n_estimators = est
		self.random_state = rando
		self.loss = loss
		self.learning_rate = learning
		self.ada_boost_regressor = AdaBoostRegressor( n_estimators=est, random_state=rando,
			loss=loss, learning_rate=learning )
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
		return [ 'prediction', 'kernel', 'accuracy',
		         'n_estimators', 'random_state', 'loss', 'learning_rate',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> AdaBoostRegression | None:
		"""

			Purpose:
			--------
				Fit the RidgeRegressor regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.

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
				self.ada_boost_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Project target target_values
			using the RidgeRegressor linerar_model.

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
				self.prediction = self.ada_boost_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			--------
			Compute the R-squared
			accuracy for the Ridge model.

			Parameters:
			----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			--------
				float: R-squared accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ada_boost_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

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
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = f1_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
					{
							"MAE": mean_absolute_error( y, self.prediction ),
							"MSE": mean_squared_error( y, self.prediction ),
							"RMSE": mean_squared_error( y, self.prediction, squared = False ),
							"R2": r2_score( y, self.prediction ),
							"Explained Variance": explained_variance_score( y, self.prediction ),
							"Median Absolute Error": median_absolute_error( y, self.prediction )
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )

	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot predicted vs
			actual target_values.

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
				self.prediction = self.ada_boost_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'ADA Boost Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BaggingRegression( Model ):
	"""

		Purpose:
		--------
		 Bagging methods form a class of algorithms which build several instances of a black-box
		 estimator on random subsets of the original training set and then aggregate their
		 individual predictions to form a final prediction. These methods are used as a way
		 to reduce the variance of a base estimator (e.g., a decision tree), by introducing
		 randomization into its construction procedure and then making an ensemble out of it.

		 Bagging methods constitute a very simple way to improve with respect
		 to a single model, without making it necessary to adapt the underlying base algorithm.
		 As they provide a way to reduce overfitting, bagging methods work best with strong and
		 complex models (e.g., fully developed decision trees), in contrast with boosting methods
		 which usually work best with weak models (e.g., shallow decision trees).

	"""
	bagging_estimator: BaggingRegressor
	base_estimator: object
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, base: object=None, num: int=10, max: int=1, rando: int=42 ) -> None:
		"""

			Purpose:
			--------
			Initialize the RidgeRegressor linerar_model.

			Parameters:
			-----------
			alpha (float): Regularization strength. Default is 1.0.
			solver (str): Solver to use. Default is 'auto'.

		"""
		super( ).__init__( )
		self.base_estimator = base
		self.n_estimators = num
		self.max_features = max
		self.random_state = rando
		self.bagging_regressor = BaggingRegressor( max_features=max, random_state=rando )
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
		return [ 'prediction', 'base_estimator', 'n_estimators', 'max_features', 'accuracy',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph', 'random_state' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			--------
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
				self.bagging_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )

	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Project target target_values
			using the RidgeRegressor linerar_model.

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
				self.prediction = self.bagging_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Compute the R-squared
			accuracy for the Ridge model.

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
				self.prediction = self.bagging_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

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
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = f1_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
					{
							"MAE": mean_absolute_error( y, self.prediction ),
							"MSE": mean_squared_error( y, self.prediction ),
							"RMSE": mean_squared_error( y, self.prediction, squared = False ),
							"R2": r2_score( y, self.prediction ),
							"Explained Variance": explained_variance_score( y, self.prediction ),
							"Median Absolute Error": median_absolute_error( y, self.prediction )
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )

	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot predicted vs
			actual target_values.

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
				self.prediction = self.bagging_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Bagging Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class VotingRegression( Model ):
	"""

		Purpose:
		--------

		Prediction voting regressor for unfitted estimators. A voting regressor is an ensemble
		meta-estimator that fits several base regressors, each on the whole dataset.
		Then it averages the individual predictions to form a final prediction.

	"""
	voting_regressor: VotingRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]

	def __init__( self, est: List[ (str, object) ], vot = 'hard' ) -> None:
		"""

			Purpose:
			--------
			Initialize the RidgeRegressor linerar_model.

			Parameters:
			-----------
			est "estimators" - List[ ( str, object ) ]:
			vot "voting" - (str): Solver to use. Default is 'hard'.

		"""
		super( ).__init__( )
		self.estimators: List[ (str, object) ] = est
		self.voting = vot
		self.voting_regressor = VotingRegressor( estimators = est )
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
		return [ 'prediction', 'kernel', 'C', 'epsilon', 'accuracy',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> VotingRegression | None:
		"""

			Purpose:
			--------
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
				self.voting_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )

	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Project target target_values
			using the RidgeRegressor linerar_model.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.

			Returns:
			-------
			np.ndarray: Predicted target target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.voting_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Compute the R-squared
			accuracy for the Ridge model.

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
				self.prediction = self.voting_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )

	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

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
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = f1_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
					{
							"MAE": mean_absolute_error( y, self.prediction ),
							"MSE": mean_squared_error( y, self.prediction ),
							"RMSE": mean_squared_error( y, self.prediction, squared = False ),
							"R2": r2_score( y, self.prediction ),
							"Explained Variance": explained_variance_score( y, self.prediction ),
							"Median Absolute Error": median_absolute_error( y, self.prediction )
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )

	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot predicted vs
			actual target_values.

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
				self.prediction = self.voting_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Voting Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

class StackRegression( Model ):
	"""

			Purpose:
			--------
			Stack of estimators with a final regressor. Stacked generalization consists in stacking
			the output of individual estimator and use a regressor to compute the final prediction.
			Stacking allows to use the strength of each individual estimator by using
			their output as input of a final estimator. Note that estimators_ are fitted on the
			full X while final_estimator_ is trained using cross-validated predictions of
			the base estimators using cross_val_predict.

	"""
	stacking_regressor: StackingRegressor
	final_estimator: ClassifierMixin
	estimators: List[ Tuple[ str, ClassifierMixin ] ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, est: List[ Tuple[ str, ClassifierMixin ] ],
	              final: ClassifierMixin=None ) -> None:
		"""

			Purpose:
			--------
			Initialize the RidgeRegressor linerar_model.

			Parameters:
			----------
			estimators - List[ Tuple[ str, ClassifierMixin ] ]:
			Base estimators which will be stacked together.
			Each element of the list is defined as a tuple of string (i.e. name) and an estimator
			instance. An estimator can be set to ‘drop’ using set_params. The type of estimator is
			generally expected to be a classifier. However, one can pass a regressor for some
			use case (e.g. ordinal regression).

			final - ClassifierMixin, default=None
			A classifier which will be used to combine the base estimators.

		"""
		super( ).__init__( )
		self.estimators = est
		self.final_estimator = final
		self.stacking_regressor = StackingRegressor( estimators=est )
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
		return [ 'prediction', 'estimators', 'final_estimator', 'accuracy',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> StackRegression | None:
		"""

			Purpose:
			---------
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
				self.stacking_regressor.fit( X, y )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Project target target_values
			using the RidgeRegressor linerar_model.

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
				self.prediction = self.stacking_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Compute the R-squared
			accuracy for the Ridge model.

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
				self.prediction = self.stacking_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

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
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = f1_score( y, self.prediction, average = 'binary' )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
					{
							"MAE": mean_absolute_error( y, self.prediction ),
							"MSE": mean_squared_error( y, self.prediction ),
							"RMSE": mean_squared_error( y, self.prediction, squared = False ),
							"R2": r2_score( y, self.prediction ),
							"Explained Variance": explained_variance_score( y, self.prediction ),
							"Median Absolute Error": median_absolute_error( y, self.prediction )
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Plot predicted vs
			actual target_values.

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
				self.prediction = self.stacking_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Stacking Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class SupportVectorRegression:
	"""
	Wrapper for sklearn's Support Vector Regression (SVR).
	"""
	svr_model: SVR
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	kernel: str
	C: float
	epsilon: float


	def __init__( self, kernel: str='rbf', c: float=1.0, epsilon: float=0.1 ) -> None:
		"""

			Purpose:
			---------
			Initialize the SVR model.

			:param kernel: Kernel type to be used in the algorithm.
			:type kernel: str
			:param c: Regularization parameter.
			:type c: float
			:param epsilon: Epsilon in the epsilon-SVR model.
			:type epsilon: float

		"""
		self.kernel = kernel
		self.C = c
		self.epsilon = epsilon
		self.svr_model = SVR( kernel=kernel, C=c, epsilon=epsilon )
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
		return [ 'prediction', 'kernel', 'C', 'epsilon', 'accuracy',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error',
		         'train', 'project', 'score', 'analyze', 'create_graph' ]


	def train( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Fit the SVR model to the data.

			:param X: Input features.
			:type X: np.ndarray
			:param y: Target values.
			:type y: np.ndarray

		"""
		self.svr_model.fit( X, y )


	def project( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Predict target values for the input features.

			:param y:
			:type y:
			:param y:
			:type y:
			:param X: Input features.
			:type X: np.ndarray
			:return: Predicted target values.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				return self.svr_model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SupportVectorRegression'
			exception.method = 'project( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

			Purpose:
			--------

			Compute the R-squared
			accuracy for the Ridge model.

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
				self.prediction = self.svr_model.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SupportVectorRegression'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			---------
			Print detailed regression metrics.

			:param X: Input features.
			:type X: np.ndarray
			:param y: Ground truth values.
			:type y: np.ndarray

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
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				return \
					{
							"MAE": mean_absolute_error( y, self.prediction ),
							"MSE": mean_squared_error( y, self.prediction ),
							"RMSE": mean_squared_error( y, self.prediction, squared = False ),
							"R2": r2_score( y, self.prediction ),
							"Explained Variance": explained_variance_score( y, self.prediction ),
							"Median Absolute Error": median_absolute_error( y, self.prediction )
					}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Visualize the true vs predicted values for regression.

			:param X: Input features.
			:type X: np.ndarray
			:param y: True target values.
			:type y: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				y_pred = self.svr_model.predict( X )
				plt.scatter( y, y_pred, color = 'blue', edgecolor = 'k' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.xlabel( "True Values" )
				plt.ylabel( "Predicted Values" )
				plt.title( "SVR: True vs Predicted" )
				plt.grid( True )
				plt.tight_layout( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )