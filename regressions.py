"""******************************************************************************************
  Assembly:                mathy
  Filename:                regressions.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="regressions.py" company="Terry D. Eppler">

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
    Provides regression model wrappers for Mathy modeling workflows. The module centralizes
    linear, regularized linear, Bayesian, stochastic-gradient, nearest-neighbor, tree,
    ensemble, support-vector, Gaussian-process, and neural-network regressors behind a
    consistent split, train, predict, score, analyze, and diagnostic plotting interface.
</summary>
******************************************************************************************
"""
from __future__ import annotations
from boogr import Error, Logger
from typing import Dict
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.ensemble as ske
import sklearn.linear_model as skl
import sklearn.neighbors as skn
import sklearn.svm as skv
import sklearn.tree as skd
from sklearn.base import ClassifierMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split as split
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import Binarizer
import sklearn.neural_network as skm
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             root_mean_squared_error,
                             explained_variance_score, median_absolute_error, max_error,
                             accuracy_score, )

def throw_if( name: str, value: object ):
	if value is None:
		raise Exception( f'Argument "{name}" cannot be empty!' )

class Regression( ):
	"""Regression regression wrapper.

		Purpose:
		    Defines the shared regression wrapper contract and common evaluation-state fields used by concrete Mathy regressor implementations.

		Attributes:
		    max_iter: Maximum number of estimator iterations.
		    random_state: Random seed or random-state configuration used by the estimator.
		    learning_rate: Estimator learning-rate configuration.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    max_error: Most recent maximum residual error metric.
		    training_score: Most recent estimator score on the training split.
		    testing_score: Most recent estimator score on the testing split.
	"""
	
	max_iter: Optional[ int ]
	random_state: Optional[ int ]
	learning_rate: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	max_error: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the Regression wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.
		"""
		self.max_iter = None
		self.random_state = None
		self.learning_rate = None
		self.prediction = None
		self.mean_absolute_error = None
		self.mean_squared_error = None
		self.root_mean_squared_error = None
		self.r2_score = None
		self.explained_variance_score = None
		self.max_error = None
		self.training_score = None
		self.testing_score = None
	
	def split_data( self, X: np.ndarray,
			y: np.ndarray ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ] | None:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    NotImplementedError: Raised when the abstract interface method is called directly.
		"""
		raise NotImplementedError
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""Train.

				Purpose:
				    Fits the underlying Regression regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    object | None: Fitted wrapper instance.

				Raises:
				    NotImplementedError: Raised when the abstract interface method is called directly.
		"""
		raise NotImplementedError
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted Regression regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    NotImplementedError: Raised when the abstract interface method is called directly.
		"""
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted Regression regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    float | pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    NotImplementedError: Raised when the abstract interface method is called directly.
		"""
		raise NotImplementedError
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted Regression model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    Dict[str, float] | pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    NotImplementedError: Raised when the abstract interface method is called directly.
		"""
		raise NotImplementedError

class LeastSquares( Regression ):
	"""LeastSquares regression wrapper.

		Purpose:
		    Wraps sklearn.linear_model.LinearRegression for ordinary least-squares regression, coefficient inspection, prediction, scoring, metric analysis, and diagnostic scatter plotting.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    max_error: Most recent maximum residual error metric.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    fit_intercept: Flag indicating whether the estimator fits an intercept term.
		    copy_X: Flag indicating whether estimator input data is copied during fitting.
		    tol: Optimization tolerance passed to the estimator.
		    n_jobs: Number of parallel worker jobs used by the estimator.
		    positive: Flag constraining coefficients to positive values when supported.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	
	model: skl.LinearRegression
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	fit_intercept: bool
	copy_X: bool
	tol: float
	n_jobs: Optional[ int ]
	positive: bool
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, fit: bool = True, copy: bool = True, tol: float = 1e-6,
			jobs: Optional[ int ] = None, positive: bool = False ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the LeastSquares wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    fit: Flag indicating whether the estimator fits an intercept term.
				    copy: Flag indicating whether input feature data is copied during fitting.
				    tol: Numerical tolerance used by estimator optimization or convergence checks.
				    jobs: Number of parallel worker jobs used by the estimator.
				    positive: Flag constraining fitted coefficients to positive values when supported.
		"""
		super( ).__init__( )
		self.fit_intercept = fit
		self.copy_X = copy
		self.tol = tol
		self.n_jobs = jobs
		self.positive = positive
		self.model = skl.LinearRegression(
			fit_intercept=self.fit_intercept,
			copy_X=self.copy_X,
			tol=self.tol,
			n_jobs=self.n_jobs,
			positive=self.positive
		)
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the LeastSquares wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'fit_intercept',
				'copy_X',
				'tol',
				'n_jobs',
				'positive',
				'weights',
				'intercept',
				'features',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'max_error',
				'training_score',
				'testing_score',
				'split_data',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot'
		]
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.

				Purpose:
				    Returns fitted `intercept` metadata from the underlying LeastSquares estimator after training.

				Returns:
				    np.ndarray | float | None: Fitted `intercept` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.

				Purpose:
				    Returns fitted `weights` metadata from the underlying LeastSquares estimator after training.

				Returns:
				    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.coef_
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying LeastSquares estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LeastSquares | None:
		"""Train.

				Purpose:
				    Fits the underlying LeastSquares regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    LeastSquares | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'train( self, *args ) -> LeastSquares | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted LeastSquares regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted LeastSquares regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted LeastSquares model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [
								'Training Score',
								'Testing Score',
								'R-Squared',
								'MAE',
								'MSE',
								'RMSE',
								'EVS',
								'MAX'
						],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score,
								self.mean_absolute_error,
								self.mean_squared_error,
								self.root_mean_squared_error,
								self.explained_variance_score,
								self.max_error
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted LeastSquares regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = (
					f'Training Score = {_training:.1%}\n'
					f'Testing Score = {_testing:.1%}\n'
			)
			
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
			exception.cause = 'LeastSquares'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class Ridge( Regression ):
	"""Ridge regression wrapper.

		Purpose:
		    Wraps sklearn.linear_model.Ridge for L2-regularized linear regression with coefficient inspection, prediction, scoring, metric analysis, and diagnostic scatter plotting.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    alpha: Regularization strength or loss parameter passed to the estimator.
		    fit_intercept: Flag indicating whether the estimator fits an intercept term.
		    copy_X: Flag indicating whether estimator input data is copied during fitting.
		    max_iter: Maximum number of estimator iterations.
		    tol: Optimization tolerance passed to the estimator.
		    solver: Solver selected for estimator optimization.
		    positive: Flag constraining coefficients to positive values when supported.
		    random_state: Random seed or random-state configuration used by the estimator.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: skl.Ridge
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	alpha: float
	fit_intercept: bool
	copy_X: bool
	max_iter: Optional[ int ]
	tol: float
	solver: str
	positive: bool
	random_state: Optional[ int ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float = 1.0, fit: bool = True, copy: bool = True,
			iters: Optional[ int ] = None, tol: float = 1e-4, solver: str = 'auto',
			positive: bool = False, rando: Optional[ int ] = None ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the Ridge wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    alpha: Regularization strength, loss parameter, or model-specific alpha value.
				    fit: Flag indicating whether the estimator fits an intercept term.
				    copy: Flag indicating whether input feature data is copied during fitting.
				    iters: Maximum number of optimization iterations.
				    tol: Numerical tolerance used by estimator optimization or convergence checks.
				    solver: Optimization solver used by the estimator.
				    positive: Flag constraining fitted coefficients to positive values when supported.
				    rando: Random-state seed passed to the underlying estimator.
		"""
		super( ).__init__( )
		self.alpha = alpha
		self.fit_intercept = fit
		self.copy_X = copy
		self.max_iter = iters
		self.tol = tol
		self.solver = solver
		self.positive = positive
		self.random_state = rando
		self.model = skl.Ridge(
			alpha=self.alpha,
			fit_intercept=self.fit_intercept,
			copy_X=self.copy_X,
			max_iter=self.max_iter,
			tol=self.tol,
			solver=self.solver,
			positive=self.positive,
			random_state=self.random_state
		)
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the Ridge wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'alpha',
				'fit_intercept',
				'copy_X',
				'max_iter',
				'tol',
				'solver',
				'positive',
				'random_state',
				'weights',
				'intercept',
				'features',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'training_score',
				'testing_score'
		]
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.

				Purpose:
				    Returns fitted `intercept` metadata from the underlying Ridge estimator after training.

				Returns:
				    np.ndarray | float | None: Fitted `intercept` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.

				Purpose:
				    Returns fitted `weights` metadata from the underlying Ridge estimator after training.

				Returns:
				    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying Ridge estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Ridge | None:
		"""Train.

				Purpose:
				    Fits the underlying Ridge regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    Ridge | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'train( self, *args ) -> Ridge | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted Ridge regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted Ridge regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted Ridge model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [
								'Training Score',
								'Testing Score',
								'R-Squared',
								'MAE',
								'MSE',
								'RMSE',
								'EVS',
								'Median AE',
								'MAX'
						],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score,
								self.mean_absolute_error,
								self.mean_squared_error,
								self.root_mean_squared_error,
								self.explained_variance_score,
								self.median_absolute_error,
								self.max_error
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted Ridge regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = (
					f'Training Score = {_training:.1%}\n'
					f'Testing Score = {_testing:.1%}\n'
			)
			
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
			exception.cause = 'Ridge'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class Lasso( Regression ):
	"""Lasso regression wrapper.

		Purpose:
		    Wraps sklearn.linear_model.Lasso for L1-regularized linear regression with sparse coefficient estimation, prediction, scoring, metric analysis, and diagnostic scatter plotting.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    alpha: Regularization strength or loss parameter passed to the estimator.
		    fit_intercept: Flag indicating whether the estimator fits an intercept term.
		    precompute: Estimator configuration or runtime state retained by the wrapper.
		    copy_X: Flag indicating whether estimator input data is copied during fitting.
		    max_iter: Maximum number of estimator iterations.
		    tol: Optimization tolerance passed to the estimator.
		    warm_start: Estimator configuration or runtime state retained by the wrapper.
		    positive: Flag constraining coefficients to positive values when supported.
		    random_state: Random seed or random-state configuration used by the estimator.
		    selection: Coordinate-selection strategy used by compatible estimators.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: skl.Lasso
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	alpha: float
	fit_intercept: bool
	precompute: bool
	copy_X: bool
	max_iter: int
	tol: float
	warm_start: bool
	positive: bool
	random_state: Optional[ int ]
	selection: str
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float = 0.01, fit: bool = True, precompute: bool = False,
			copy: bool = True, iters: int = 1000, tol: float = 1e-4, warm: bool = False,
			positive: bool = False, rando: Optional[ int ] = None,
			select: str = 'cyclic', selection: Optional[ str ] = None ) -> None:
		"""Initialize the Lasso regression wrapper.

				Purpose:
				    Configures an L1-regularized linear regression estimator for sparse coefficient
				    estimation, continuous-target prediction, model scoring, residual analysis, and
				    coefficient diagnostics. Supports the existing `select` parameter and the
				    application-facing `selection` parameter for coordinate-descent feature selection.

				Args:
				    alpha: Regularization strength applied to the L1 penalty.
				    fit: Flag indicating whether the estimator fits an intercept.
				    precompute: Flag controlling Gram-matrix precomputation.
				    copy: Flag indicating whether the input feature matrix is copied.
				    iters: Maximum number of coordinate-descent iterations.
				    tol: Optimization tolerance used to determine convergence.
				    warm: Flag indicating whether the previous fitted solution is reused.
				    positive: Flag constraining fitted coefficients to nonnegative values.
				    rando: Random seed used when random coordinate selection is enabled.
				    select: Coordinate-selection strategy used when `selection` is not supplied.
				    selection: Coordinate-selection strategy supplied by the application.
		"""
		super( ).__init__( )
		self.alpha = alpha
		self.fit_intercept = fit
		self.precompute = precompute
		self.copy_X = copy
		self.max_iter = iters
		self.tol = tol
		self.warm_start = warm
		self.positive = positive
		self.random_state = rando
		self.selection = selection if selection is not None else select
		self.model = skl.Lasso(
			alpha=self.alpha,
			fit_intercept=self.fit_intercept,
			precompute=self.precompute,
			copy_X=self.copy_X,
			max_iter=self.max_iter,
			tol=self.tol,
			warm_start=self.warm_start,
			positive=self.positive,
			random_state=self.random_state,
			selection=self.selection
		)
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the Lasso wrapper for
				    interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'alpha',
				'fit_intercept',
				'precompute',
				'copy_X',
				'max_iter',
				'tol',
				'warm_start',
				'positive',
				'random_state',
				'selection',
				'weights',
				'intercept',
				'features',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'training_score',
				'testing_score'
		]
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.

				Purpose:
				    Returns fitted `intercept` metadata from the underlying Lasso estimator after training.

				Returns:
				    np.ndarray | float | None: Fitted `intercept` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.

				Purpose:
				    Returns fitted `weights` metadata from the underlying Lasso estimator after training.

				Returns:
				    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying Lasso estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Lasso | None:
		"""Train.

				Purpose:
				    Fits the underlying Lasso regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    Lasso | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'train( self, *args ) -> Lasso | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted Lasso regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted Lasso regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted Lasso model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [
								'Training Score',
								'Testing Score',
								'R-Squared',
								'MAE',
								'MSE',
								'RMSE',
								'EVS',
								'Median AE',
								'MAX'
						],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score,
								self.mean_absolute_error,
								self.mean_squared_error,
								self.root_mean_squared_error,
								self.explained_variance_score,
								self.median_absolute_error,
								self.max_error
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted Lasso regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
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
			exception.cause = 'Lasso'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class ElasticNet( Regression ):
	"""ElasticNet regression wrapper.

		Purpose:
		    Wraps sklearn.linear_model.ElasticNet for combined L1 and L2 regularized linear regression with prediction, scoring, metric analysis, and diagnostic scatter plotting.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    alpha: Regularization strength or loss parameter passed to the estimator.
		    l1_ratio: Elastic-net mixing parameter.
		    fit_intercept: Flag indicating whether the estimator fits an intercept term.
		    precompute: Estimator configuration or runtime state retained by the wrapper.
		    max_iter: Maximum number of estimator iterations.
		    copy_X: Flag indicating whether estimator input data is copied during fitting.
		    tol: Optimization tolerance passed to the estimator.
		    warm_start: Estimator configuration or runtime state retained by the wrapper.
		    positive: Flag constraining coefficients to positive values when supported.
		    random_state: Random seed or random-state configuration used by the estimator.
		    selection: Coordinate-selection strategy used by compatible estimators.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: skl.ElasticNet
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	alpha: float
	l1_ratio: float
	fit_intercept: bool
	precompute: bool
	max_iter: int
	copy_X: bool
	tol: float
	warm_start: bool
	positive: bool
	random_state: Optional[ int ]
	selection: str
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float = 1.0, ratio: float = 0.5, fit: bool = True,
			precompute: bool = False, iters: int = 1000, copy: bool = True,
			tol: float = 1e-4, warm: bool = False, positive: bool = False,
			rando: Optional[ int ] = None, select: str = 'cyclic' ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the ElasticNet wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    alpha: Regularization strength, loss parameter, or model-specific alpha value.
				    ratio: Elastic-net L1/L2 mixing ratio.
				    fit: Flag indicating whether the estimator fits an intercept term.
				    precompute: Flag or setting controlling Gram-matrix precomputation.
				    iters: Maximum number of optimization iterations.
				    copy: Flag indicating whether input feature data is copied during fitting.
				    tol: Numerical tolerance used by estimator optimization or convergence checks.
				    warm: Flag indicating whether previous estimator state is reused across fits.
				    positive: Flag constraining fitted coefficients to positive values when supported.
				    rando: Random-state seed passed to the underlying estimator.
				    select: Coordinate-selection strategy used during optimization.
		"""
		super( ).__init__( )
		self.alpha = alpha
		self.l1_ratio = ratio
		self.fit_intercept = fit
		self.precompute = precompute
		self.max_iter = iters
		self.copy_X = copy
		self.tol = tol
		self.warm_start = warm
		self.positive = positive
		self.random_state = rando
		self.selection = select
		self.model = skl.ElasticNet(
			alpha=self.alpha,
			l1_ratio=self.l1_ratio,
			fit_intercept=self.fit_intercept,
			precompute=self.precompute,
			max_iter=self.max_iter,
			copy_X=self.copy_X,
			tol=self.tol,
			warm_start=self.warm_start,
			positive=self.positive,
			random_state=self.random_state,
			selection=self.selection
		)
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the ElasticNet wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'alpha',
				'l1_ratio',
				'fit_intercept',
				'precompute',
				'max_iter',
				'copy_X',
				'tol',
				'warm_start',
				'positive',
				'random_state',
				'selection',
				'weights',
				'intercept',
				'features',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'training_score',
				'testing_score'
		]
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.

				Purpose:
				    Returns fitted `intercept` metadata from the underlying ElasticNet estimator after training.

				Returns:
				    np.ndarray | float | None: Fitted `intercept` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.

				Purpose:
				    Returns fitted `weights` metadata from the underlying ElasticNet estimator after training.

				Returns:
				    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying ElasticNet estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> ElasticNet | None:
		"""Train.

				Purpose:
				    Fits the underlying ElasticNet regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    ElasticNet | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'train( self, *args ) -> ElasticNet | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted ElasticNet regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted ElasticNet regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted ElasticNet model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [
								'Training Score',
								'Testing Score',
								'R-Squared',
								'MAE',
								'MSE',
								'RMSE',
								'EVS',
								'Median AE',
								'MAX'
						],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score,
								self.mean_absolute_error,
								self.mean_squared_error,
								self.root_mean_squared_error,
								self.explained_variance_score,
								self.median_absolute_error,
								self.max_error
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted ElasticNet regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
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
			exception.cause = 'ElasticNet'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class LeastAngle( Regression ):
	"""LeastAngle regression wrapper.

		Purpose:
		    Wraps sklearn.linear_model.Lars for least-angle regression with coefficient-path-oriented linear modeling, prediction, scoring, metric analysis, and diagnostic scatter plotting.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    n_nonzero_coefs: Estimator configuration or runtime state retained by the wrapper.
		    fit_intercept: Flag indicating whether the estimator fits an intercept term.
		    precompute: Estimator configuration or runtime state retained by the wrapper.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
	"""
	model: skl.Lars
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	n_nonzero_coefs: int
	fit_intercept: bool
	precompute: bool
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, coeffs: int = 500, fit: bool = True, precompute: bool = True ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the LeastAngle wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    coeffs: Maximum number of coefficients or active variables used by the estimator.
				    fit: Flag indicating whether the estimator fits an intercept term.
				    precompute: Flag or setting controlling Gram-matrix precomputation.
		"""
		super( ).__init__( )
		self.fit_intercept = fit
		self.n_nonzero_coefs = coeffs
		self.precompute = precompute
		self.model = skl.Lars(
			fit_intercept=self.fit_intercept,
			precompute=self.precompute,
			n_nonzero_coefs=self.n_nonzero_coefs
		)
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the LeastAngle wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'fit_intercept',
				'precompute',
				'n_nonzero_coefs',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'training_score',
				'testing_score'
		]
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LeastAngle | None:
		"""Train.

				Purpose:
				    Fits the underlying LeastAngle regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    LeastAngle | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'train( self, *args ) -> LeastAngle | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted LeastAngle regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted LeastAngle regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.r2_score = r2_score( y, self.prediction )
			
			_metrics = {
					'Training Score': self.training_score,
					'Testing Score': self.testing_score,
					'R-Squared Score': self.r2_score,
			}
			
			idx = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=idx )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted LeastAngle model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			
			_metrics = {
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.root_mean_squared_error,
					'EVS': self.explained_variance_score,
					'MAX': self.max_error,
			}
			
			_data = pd.Series( _metrics )
			df_metrics = pd.DataFrame( _data )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted LeastAngle regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
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
			exception.cause = 'LeastAngle'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class BayesianRidge( Regression ):
	"""BayesianRidge regression wrapper.

		Purpose:
		    Wraps sklearn.linear_model.BayesianRidge for probabilistic linear regression with Bayesian regularization, prediction, scoring, metric analysis, and diagnostic scatter plotting.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    scale_alpha: Estimator configuration or runtime state retained by the wrapper.
		    shape_lambda: Estimator configuration or runtime state retained by the wrapper.
		    shape_alpha: Estimator configuration or runtime state retained by the wrapper.
		    max_iter: Maximum number of estimator iterations.
		    scale_lambda: Estimator configuration or runtime state retained by the wrapper.
		    tol: Optimization tolerance passed to the estimator.
		    alpha_init: Estimator configuration or runtime state retained by the wrapper.
		    lambda_init: Estimator configuration or runtime state retained by the wrapper.
		    compute_score: Estimator configuration or runtime state retained by the wrapper.
		    fit_intercept: Flag indicating whether the estimator fits an intercept term.
		    copy_X: Flag indicating whether estimator input data is copied during fitting.
		    verbose: Estimator configuration or runtime state retained by the wrapper.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: skl.BayesianRidge
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	scale_alpha: float
	shape_lambda: float
	shape_alpha: float
	max_iter: int
	scale_lambda: float
	tol: float
	alpha_init: Optional[ float ]
	lambda_init: Optional[ float ]
	compute_score: bool
	fit_intercept: bool
	copy_X: bool
	verbose: bool
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, max: int = 300, shape_alpha: float = 1e-06, scale_alpha: float = 1e-06,
			shape_lambda: float = 1e-06, scale_lambda: float = 1e-06, tol: float = 1e-3,
			alpha_init: Optional[ float ] = None, lambda_init: Optional[ float ] = None,
			compute_score: bool = False, fit: bool = True, copy: bool = True,
			verbose: bool = False ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the BayesianRidge wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    max: Configuration value passed to the underlying regression estimator.
				    shape_alpha: Shape parameter for the Bayesian alpha prior.
				    scale_alpha: Scale parameter for the Bayesian alpha prior.
				    shape_lambda: Shape parameter for the Bayesian lambda prior.
				    scale_lambda: Scale parameter for the Bayesian lambda prior.
				    tol: Numerical tolerance used by estimator optimization or convergence checks.
				    alpha_init: Initial alpha value for Bayesian regression.
				    lambda_init: Initial lambda value for Bayesian regression.
				    compute_score: Flag indicating whether Bayesian scores are computed during fitting.
				    fit: Flag indicating whether the estimator fits an intercept term.
				    copy: Flag indicating whether input feature data is copied during fitting.
				    verbose: Verbosity flag or level passed to the estimator.
		"""
		super( ).__init__( )
		self.max_iter = max
		self.shape_alpha = shape_alpha
		self.scale_alpha = scale_alpha
		self.shape_lambda = shape_lambda
		self.scale_lambda = scale_lambda
		self.tol = tol
		self.alpha_init = alpha_init
		self.lambda_init = lambda_init
		self.compute_score = compute_score
		self.fit_intercept = fit
		self.copy_X = copy
		self.verbose = verbose
		self.model = skl.BayesianRidge(
			max_iter=self.max_iter,
			tol=self.tol,
			alpha_1=self.shape_alpha,
			alpha_2=self.scale_alpha,
			lambda_1=self.shape_lambda,
			lambda_2=self.scale_lambda,
			alpha_init=self.alpha_init,
			lambda_init=self.lambda_init,
			compute_score=self.compute_score,
			fit_intercept=self.fit_intercept,
			copy_X=self.copy_X,
			verbose=self.verbose
		)
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the BayesianRidge wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'shape_alpha',
				'scale_alpha',
				'shape_lambda',
				'scale_lambda',
				'max_iter',
				'tol',
				'alpha_init',
				'lambda_init',
				'compute_score',
				'fit_intercept',
				'copy_X',
				'verbose',
				'weights',
				'intercept',
				'features',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'training_score',
				'testing_score'
		]
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.

				Purpose:
				    Returns fitted `intercept` metadata from the underlying BayesianRidge estimator after training.

				Returns:
				    np.ndarray | float | None: Fitted `intercept` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.

				Purpose:
				    Returns fitted `weights` metadata from the underlying BayesianRidge estimator after training.

				Returns:
				    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying BayesianRidge estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BayesianRidge | None:
		"""Train.

				Purpose:
				    Fits the underlying BayesianRidge regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    BayesianRidge | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'train( self, *args ) -> BayesianRidge | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted BayesianRidge regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted BayesianRidge regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted BayesianRidge model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [
							'Training Score',
							'Testing Score',
							'R-Squared',
							'MAE',
							'MSE',
							'RMSE',
							'EVS',
							'Median AE',
							'MAX'
					],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score,
							self.mean_absolute_error,
							self.mean_squared_error,
							self.root_mean_squared_error,
							self.explained_variance_score,
							self.median_absolute_error,
							self.max_error
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted BayesianRidge regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' } )
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
			exception.cause = 'BayesianRidge'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class GradientDescent( Regression ):
	"""GradientDescent regression wrapper.

		Purpose:
		    Wraps sklearn.linear_model.SGDRegressor for stochastic-gradient regression with configurable loss, penalty, learning-rate, convergence, and early-stopping behavior.

		Attributes:
		    random_state: Random seed or random-state configuration used by the estimator.
		    loss: Estimator configuration or runtime state retained by the wrapper.
		    max_iter: Maximum number of estimator iterations.
		    penalty: Estimator configuration or runtime state retained by the wrapper.
		    learning_rate: Estimator learning-rate configuration.
		    l1_ratio: Elastic-net mixing parameter.
		    alpha: Regularization strength or loss parameter passed to the estimator.
		    fit_intercept: Flag indicating whether the estimator fits an intercept term.
		    tol: Optimization tolerance passed to the estimator.
		    shuffle: Estimator configuration or runtime state retained by the wrapper.
		    verbose: Estimator configuration or runtime state retained by the wrapper.
		    epsilon: Estimator configuration or runtime state retained by the wrapper.
		    eta0: Estimator configuration or runtime state retained by the wrapper.
		    power_t: Estimator configuration or runtime state retained by the wrapper.
		    early_stopping: Estimator configuration or runtime state retained by the wrapper.
		    validation_fraction: Estimator configuration or runtime state retained by the wrapper.
		    n_iter_no_change: Estimator configuration or runtime state retained by the wrapper.
		    warm_start: Estimator configuration or runtime state retained by the wrapper.
		    average: Estimator configuration or runtime state retained by the wrapper.
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	random_state: Optional[ int ]
	loss: Optional[ str ]
	max_iter: Optional[ int ]
	penalty: Optional[ str ]
	learning_rate: Optional[ str ]
	l1_ratio: Optional[ float ]
	alpha: Optional[ float ]
	fit_intercept: bool
	tol: float
	shuffle: bool
	verbose: int
	epsilon: float
	eta0: float
	power_t: float
	early_stopping: bool
	validation_fraction: float
	n_iter_no_change: int
	warm_start: bool
	average: bool
	model: skl.SGDRegressor
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, loss: str = 'squared_error', iters: int = 1000, penalty: str = 'l2',
			alpha: float = 0.0001, rando: Optional[ int ] = 42,
			learning_rate: str = 'invscaling', l1_ratio: float = 0.15,
			fit: bool = True, tol: float = 1e-3, shuffle: bool = True, verbose: int = 0,
			epsilon: float = 0.1, eta0: float = 0.01, power_t: float = 0.25,
			early_stopping: bool = False, validation_fraction: float = 0.1,
			n_iter_no_change: int = 5, warm: bool = False, average: bool = False ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the GradientDescent wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    loss: Loss function used by the estimator.
				    iters: Maximum number of optimization iterations.
				    penalty: Regularization parameter used by the estimator.
				    alpha: Regularization strength, loss parameter, or model-specific alpha value.
				    rando: Random-state seed passed to the underlying estimator.
				    learning_rate: Learning-rate schedule used by the estimator.
				    l1_ratio: Elastic-net mixing ratio for L1 and L2 penalties.
				    fit: Flag indicating whether the estimator fits an intercept term.
				    tol: Numerical tolerance used by estimator optimization or convergence checks.
				    shuffle: Flag indicating whether training samples are shuffled between epochs.
				    verbose: Verbosity flag or level passed to the estimator.
				    epsilon: Epsilon-insensitive loss parameter or robustness threshold.
				    eta0: Initial learning-rate value.
				    power_t: Exponent used by inverse-scaling learning-rate schedules.
				    early_stopping: Flag enabling validation-based early stopping.
				    validation_fraction: Training-data fraction reserved for early-stopping validation.
				    n_iter_no_change: Number of iterations with no improvement before early stopping.
				    warm: Flag indicating whether previous estimator state is reused across fits.
				    average: Flag enabling averaged stochastic-gradient coefficients.
		"""
		super( ).__init__( )
		self.random_state = rando
		self.loss = loss
		self.max_iter = iters
		self.penalty = penalty
		self.learning_rate = learning_rate
		self.l1_ratio = l1_ratio
		self.alpha = alpha
		self.fit_intercept = fit
		self.tol = tol
		self.shuffle = shuffle
		self.verbose = verbose
		self.epsilon = epsilon
		self.eta0 = eta0
		self.power_t = power_t
		self.early_stopping = early_stopping
		self.validation_fraction = validation_fraction
		self.n_iter_no_change = n_iter_no_change
		self.warm_start = warm
		self.average = average
		self.model = skl.SGDRegressor( loss=self.loss, penalty=self.penalty, alpha=self.alpha,
			l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept, max_iter=self.max_iter,
			tol=self.tol, shuffle=self.shuffle, verbose=self.verbose, epsilon=self.epsilon,
			random_state=self.random_state, learning_rate=self.learning_rate,
			eta0=self.eta0, power_t=self.power_t, early_stopping=self.early_stopping,
			validation_fraction=self.validation_fraction, n_iter_no_change=self.n_iter_no_change,
			warm_start=self.warm_start, average=self.average )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the GradientDescent wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'penalty',
				'max_iter',
				'random_state',
				'loss',
				'learning_rate',
				'l1_ratio',
				'alpha',
				'fit_intercept',
				'tol',
				'shuffle',
				'verbose',
				'epsilon',
				'eta0',
				'power_t',
				'early_stopping',
				'validation_fraction',
				'n_iter_no_change',
				'warm_start',
				'average',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'weights',
				'intercept',
				'features',
				'iterations',
				'training_score',
				'testing_score'
		]
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.

				Purpose:
				    Returns fitted `weights` metadata from the underlying GradientDescent estimator after training.

				Returns:
				    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.

				Purpose:
				    Returns fitted `intercept` metadata from the underlying GradientDescent estimator after training.

				Returns:
				    np.ndarray | float | None: Fitted `intercept` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying GradientDescent estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def iterations( self ) -> int | None:
		"""Iterations.

				Purpose:
				    Returns fitted `iterations` metadata from the underlying GradientDescent estimator after training.

				Returns:
				    int | None: Fitted `iterations` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 't_' ):
			raise AttributeError( 'The model has not been initialized!' )
		return self.model.t_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientDescent | None:
		"""Train.

				Purpose:
				    Fits the underlying GradientDescent regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    GradientDescent | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'train( self, *args ) -> GradientDescent | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted GradientDescent regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted GradientDescent regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted GradientDescent model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [
							'Training Score',
							'Testing Score',
							'R-Squared',
							'MAE',
							'MSE',
							'RMSE',
							'EVS',
							'Median AE',
							'MAX'
					],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score,
							self.mean_absolute_error,
							self.mean_squared_error,
							self.root_mean_squared_error,
							self.explained_variance_score,
							self.median_absolute_error,
							self.max_error
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted GradientDescent regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' } )
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
			exception.cause = 'GradientDescent'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class NearestNeighbor( Regression ):
	"""NearestNeighbor regression wrapper.

		Purpose:
		    Wraps sklearn.neighbors.KNeighborsRegressor for neighbor-based continuous-value prediction, model scoring, metric analysis, and diagnostic scatter plotting.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    n_neighbors: Estimator configuration or runtime state retained by the wrapper.
		    weights: Optional estimator or neighbor weights.
		    algorithm: Estimator configuration or runtime state retained by the wrapper.
		    leaf_size: Estimator configuration or runtime state retained by the wrapper.
		    power: Estimator configuration or runtime state retained by the wrapper.
		    metric: Estimator configuration or runtime state retained by the wrapper.
		    metric_params: Estimator configuration or runtime state retained by the wrapper.
		    n_jobs: Number of parallel worker jobs used by the estimator.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: skn.KNeighborsRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	n_neighbors: int
	weights: str
	algorithm: str
	leaf_size: int
	power: float
	metric: str
	metric_params: Optional[ Dict[ str, object ] ]
	n_jobs: Optional[ int ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, num: int = 5, weight: str = 'uniform', algo: str = 'auto',
			leaf: int = 30, power: float = 2.0, metric: str = 'minkowski',
			metric_params: Optional[ Dict[ str, object ] ] = None,
			jobs: Optional[ int ] = None ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the NearestNeighbor wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    num: Number of neighbors, estimators, or model-specific components.
				    weight: Neighbor weighting strategy.
				    algo: Neighbor-search algorithm.
				    leaf: Leaf-size or minimum leaf configuration.
				    power: Power parameter for Minkowski-style distance metrics.
				    metric: Distance metric used by neighbor estimators.
				    metric_params: Additional metric-specific keyword parameters.
				    jobs: Number of parallel worker jobs used by the estimator.
		"""
		super( ).__init__( )
		self.n_neighbors = num
		self.weights = weight
		self.algorithm = algo
		self.leaf_size = leaf
		self.power = power
		self.metric = metric
		self.metric_params = metric_params
		self.n_jobs = jobs
		self.model = skn.KNeighborsRegressor( n_neighbors=self.n_neighbors, weights=self.weights,
			algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.power, metric=self.metric,
			metric_params=self.metric_params, n_jobs=self.n_jobs )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the NearestNeighbor wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'weights',
				'algorithm',
				'n_neighbors',
				'leaf_size',
				'power',
				'metric',
				'metric_params',
				'n_jobs',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'training_score',
				'testing_score',
				'features'
		]
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying NearestNeighbor estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> NearestNeighbor | None:
		"""Train.

				Purpose:
				    Fits the underlying NearestNeighbor regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    NearestNeighbor | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'train( self, *args ) -> NearestNeighbor | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted NearestNeighbor regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted NearestNeighbor regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted NearestNeighbor model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [
								'Training Score',
								'Testing Score',
								'R-Squared',
								'MAE',
								'MSE',
								'RMSE',
								'EVS',
								'Median AE',
								'MAX'
						],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score,
								self.mean_absolute_error,
								self.mean_squared_error,
								self.root_mean_squared_error,
								self.explained_variance_score,
								self.median_absolute_error,
								self.max_error
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted NearestNeighbor regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' } )
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
			exception.cause = 'NearestNeighbor'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class DecisionTree( Regression ):
	"""DecisionTree regression wrapper.

		Purpose:
		    Wraps sklearn.tree.DecisionTreeRegressor for non-parametric tree-based regression, prediction, scoring, metric analysis, and diagnostic scatter plotting.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    criterion: Split or impurity criterion used by tree-based estimators.
		    splitter: Tree splitter strategy.
		    max_depth: Maximum tree depth.
		    random_state: Random seed or random-state configuration used by the estimator.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
	"""
	model: skd.DecisionTreeRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	criterion: str
	splitter: str
	max_depth: int
	random_state: int
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, criterion: str = 'squared_error', splitter: str = 'best',
			depth: int = 3, rando: int = 42 ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the DecisionTree wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    criterion: Tree split or ensemble loss criterion.
				    splitter: Tree split strategy.
				    depth: Maximum tree depth.
				    rando: Random-state seed passed to the underlying estimator.
		"""
		super( ).__init__( )
		self.criterion = criterion
		self.splitter = splitter
		self.max_depth = depth
		self.random_state = rando
		self.model = skd.DecisionTreeRegressor( criterion=self.criterion, splitter=self.splitter,
			max_depth=self.max_depth, random_state=self.random_state )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the DecisionTree wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'criterion',
				'splitter',
				'random_state',
				'max_depth',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'training_score',
				'testing_score'
		]
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> DecisionTree | None:
		"""Train.

				Purpose:
				    Fits the underlying DecisionTree regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    DecisionTree | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
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
			exception.method = 'train( self, *args ) -> DecisionTree | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted DecisionTree regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted DecisionTree regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.r2_score = r2_score( y, self.prediction )
			_metrics = {
					'Training Score': self.training_score,
					'Testing Score': self.testing_score,
					'R-Squared Score': self.r2_score,
			}
			idx = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=idx )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted DecisionTree model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			_metrics = {
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.root_mean_squared_error,
					'EVS': self.explained_variance_score,
					'MAX': self.max_error,
			}
			_data = pd.Series( _metrics )
			df_metrics = pd.DataFrame( _data )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted DecisionTree regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' } )
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
			exception.cause = 'DecisionTree'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class ExtraTreesModel( Regression ):
	"""ExtraTreesModel regression wrapper.

		Purpose:
		    Wraps sklearn.ensemble.ExtraTreesRegressor for randomized ensemble tree regression, feature metadata inspection, prediction, scoring, and metric analysis.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    n_estimators: Number of ensemble estimators.
		    criterion: Split or impurity criterion used by tree-based estimators.
		    max_depth: Maximum tree depth.
		    min_samples_split: Estimator configuration or runtime state retained by the wrapper.
		    min_samples_leaf: Estimator configuration or runtime state retained by the wrapper.
		    min_weight_fraction_leaf: Estimator configuration or runtime state retained by the wrapper.
		    max_features: Estimator configuration or runtime state retained by the wrapper.
		    max_leaf_nodes: Estimator configuration or runtime state retained by the wrapper.
		    min_impurity_decrease: Estimator configuration or runtime state retained by the wrapper.
		    bootstrap: Estimator configuration or runtime state retained by the wrapper.
		    oob_score: Estimator configuration or runtime state retained by the wrapper.
		    n_jobs: Number of parallel worker jobs used by the estimator.
		    random_state: Random seed or random-state configuration used by the estimator.
		    verbose: Estimator configuration or runtime state retained by the wrapper.
		    warm_start: Estimator configuration or runtime state retained by the wrapper.
		    ccp_alpha: Estimator configuration or runtime state retained by the wrapper.
		    max_samples: Estimator configuration or runtime state retained by the wrapper.
		    monotonic_cst: Estimator configuration or runtime state retained by the wrapper.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: ske.ExtraTreesRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	n_estimators: int
	criterion: str
	max_depth: Optional[ int ]
	min_samples_split: int | float
	min_samples_leaf: int | float
	min_weight_fraction_leaf: float
	max_features: int | float | str | None
	max_leaf_nodes: Optional[ int ]
	min_impurity_decrease: float
	bootstrap: bool
	oob_score: bool
	n_jobs: Optional[ int ]
	random_state: Optional[ int ]
	verbose: int
	warm_start: bool
	ccp_alpha: float
	max_samples: Optional[ int | float ]
	monotonic_cst: Optional[ object ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, estimators: int = 100, criterion: str = 'squared_error',
			depth: Optional[ int ] = None, split: int | float = 2, leaf: int | float = 1,
			weight_fraction: float = 0.0, features: int | float | str | None = 1.0,
			leaf_nodes: Optional[ int ] = None, impurity: float = 0.0,
			bootstrap: bool = False, oob_score: bool = False, jobs: Optional[ int ] = None,
			rando: Optional[ int ] = 42, verbose: int = 0, warm: bool = False,
			ccp_alpha: float = 0.0, samples: Optional[ int | float ] = None,
			monotonic: Optional[ object ] = None ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the ExtraTreesModel wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    estimators: Number of estimators in an ensemble.
				    criterion: Tree split or ensemble loss criterion.
				    depth: Maximum tree depth.
				    split: Minimum samples required to split an internal tree node.
				    leaf: Leaf-size or minimum leaf configuration.
				    weight_fraction: Minimum weighted fraction required at a leaf node.
				    features: Maximum feature-selection setting used by the estimator.
				    leaf_nodes: Maximum number of leaf nodes.
				    impurity: Minimum impurity decrease required for a split.
				    bootstrap: Flag enabling bootstrap sampling.
				    oob_score: Flag enabling out-of-bag scoring.
				    jobs: Number of parallel worker jobs used by the estimator.
				    rando: Random-state seed passed to the underlying estimator.
				    verbose: Verbosity flag or level passed to the estimator.
				    warm: Flag indicating whether previous estimator state is reused across fits.
				    ccp_alpha: Complexity parameter used for minimal cost-complexity pruning.
				    samples: Maximum sample setting used by ensemble estimators.
				    monotonic: Monotonic constraint configuration for compatible estimators.
		"""
		super( ).__init__( )
		self.n_estimators = estimators
		self.criterion = criterion
		self.max_depth = depth
		self.min_samples_split = split
		self.min_samples_leaf = leaf
		self.min_weight_fraction_leaf = weight_fraction
		self.max_features = features
		self.max_leaf_nodes = leaf_nodes
		self.min_impurity_decrease = impurity
		self.bootstrap = bootstrap
		self.oob_score = oob_score
		self.n_jobs = jobs
		self.random_state = rando
		self.verbose = verbose
		self.warm_start = warm
		self.ccp_alpha = ccp_alpha
		self.max_samples = samples
		self.monotonic_cst = monotonic
		self.model = ske.ExtraTreesRegressor( n_estimators=self.n_estimators,
			criterion=self.criterion,
			max_depth=self.max_depth, min_samples_split=self.min_samples_split,
			min_samples_leaf=self.min_samples_leaf,
			min_weight_fraction_leaf=self.min_weight_fraction_leaf,
			max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes,
			min_impurity_decrease=self.min_impurity_decrease, bootstrap=self.bootstrap,
			oob_score=self.oob_score, n_jobs=self.n_jobs, random_state=self.random_state,
			verbose=self.verbose, warm_start=self.warm_start, ccp_alpha=self.ccp_alpha,
			max_samples=self.max_samples, monotonic_cst=self.monotonic_cst )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the ExtraTreesModel wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'n_estimators',
				'criterion',
				'max_depth',
				'min_samples_split',
				'min_samples_leaf',
				'min_weight_fraction_leaf',
				'max_features',
				'max_leaf_nodes',
				'min_impurity_decrease',
				'bootstrap',
				'oob_score',
				'n_jobs',
				'random_state',
				'verbose',
				'warm_start',
				'ccp_alpha',
				'max_samples',
				'monotonic_cst',
				'features',
				'training_score',
				'testing_score',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'split_data',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot'
		]
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying ExtraTreesModel estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> ExtraTreesModel | None:
		"""Train.

				Purpose:
				    Fits the underlying ExtraTreesModel regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    ExtraTreesModel | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'train( self, *args ) -> ExtraTreesModel | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted ExtraTreesModel regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted ExtraTreesModel regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted ExtraTreesModel model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [
							'Training Score',
							'Testing Score',
							'R-Squared',
							'MAE',
							'MSE',
							'RMSE',
							'EVS',
							'Median AE',
							'MAX'
					],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score,
							self.mean_absolute_error,
							self.mean_squared_error,
							self.root_mean_squared_error,
							self.explained_variance_score,
							self.median_absolute_error,
							self.max_error
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted ExtraTreesModel regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' } )
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
			exception.cause = 'ExtraTreesModel'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class RandomForest( Regression ):
	"""RandomForest regression wrapper.

		Purpose:
		    Wraps sklearn.ensemble.RandomForestRegressor for bagged ensemble tree regression, feature metadata inspection, prediction, scoring, and metric analysis.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    n_estimators: Number of ensemble estimators.
		    criterion: Split or impurity criterion used by tree-based estimators.
		    max_depth: Maximum tree depth.
		    min_samples_split: Estimator configuration or runtime state retained by the wrapper.
		    min_samples_leaf: Estimator configuration or runtime state retained by the wrapper.
		    min_weight_fraction_leaf: Estimator configuration or runtime state retained by the wrapper.
		    max_features: Estimator configuration or runtime state retained by the wrapper.
		    max_leaf_nodes: Estimator configuration or runtime state retained by the wrapper.
		    min_impurity_decrease: Estimator configuration or runtime state retained by the wrapper.
		    bootstrap: Estimator configuration or runtime state retained by the wrapper.
		    oob_score: Estimator configuration or runtime state retained by the wrapper.
		    n_jobs: Number of parallel worker jobs used by the estimator.
		    random_state: Random seed or random-state configuration used by the estimator.
		    verbose: Estimator configuration or runtime state retained by the wrapper.
		    warm_start: Estimator configuration or runtime state retained by the wrapper.
		    ccp_alpha: Estimator configuration or runtime state retained by the wrapper.
		    max_samples: Estimator configuration or runtime state retained by the wrapper.
		    monotonic_cst: Estimator configuration or runtime state retained by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: ske.RandomForestRegressor
	n_estimators: int
	criterion: str
	max_depth: Optional[ int ]
	min_samples_split: int | float
	min_samples_leaf: int | float
	min_weight_fraction_leaf: float
	max_features: int | float | str | None
	max_leaf_nodes: Optional[ int ]
	min_impurity_decrease: float
	bootstrap: bool
	oob_score: bool
	n_jobs: Optional[ int ]
	random_state: Optional[ int ]
	verbose: int
	warm_start: bool
	ccp_alpha: float
	max_samples: Optional[ int | float ]
	monotonic_cst: Optional[ object ]
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, estimators: int = 100, criterion: str = 'squared_error',
			depth: Optional[ int ] = None, split: int | float = 2, leaf: int | float = 1,
			weight_fraction: float = 0.0, features: int | float | str | None = 1.0,
			leaf_nodes: Optional[ int ] = None, impurity: float = 0.0,
			bootstrap: bool = True, oob_score: bool = False, jobs: Optional[ int ] = None,
			rando: Optional[ int ] = 42, verbose: int = 0, warm: bool = False,
			ccp_alpha: float = 0.0, samples: Optional[ int | float ] = None,
			monotonic: Optional[ object ] = None ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the RandomForest wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    estimators: Number of estimators in an ensemble.
				    criterion: Tree split or ensemble loss criterion.
				    depth: Maximum tree depth.
				    split: Minimum samples required to split an internal tree node.
				    leaf: Leaf-size or minimum leaf configuration.
				    weight_fraction: Minimum weighted fraction required at a leaf node.
				    features: Maximum feature-selection setting used by the estimator.
				    leaf_nodes: Maximum number of leaf nodes.
				    impurity: Minimum impurity decrease required for a split.
				    bootstrap: Flag enabling bootstrap sampling.
				    oob_score: Flag enabling out-of-bag scoring.
				    jobs: Number of parallel worker jobs used by the estimator.
				    rando: Random-state seed passed to the underlying estimator.
				    verbose: Verbosity flag or level passed to the estimator.
				    warm: Flag indicating whether previous estimator state is reused across fits.
				    ccp_alpha: Complexity parameter used for minimal cost-complexity pruning.
				    samples: Maximum sample setting used by ensemble estimators.
				    monotonic: Monotonic constraint configuration for compatible estimators.
		"""
		super( ).__init__( )
		self.n_estimators = estimators
		self.criterion = criterion
		self.max_depth = depth
		self.min_samples_split = split
		self.min_samples_leaf = leaf
		self.min_weight_fraction_leaf = weight_fraction
		self.max_features = features
		self.max_leaf_nodes = leaf_nodes
		self.min_impurity_decrease = impurity
		self.bootstrap = bootstrap
		self.oob_score = oob_score
		self.n_jobs = jobs
		self.random_state = rando
		self.verbose = verbose
		self.warm_start = warm
		self.ccp_alpha = ccp_alpha
		self.max_samples = samples
		self.monotonic_cst = monotonic
		self.model = ske.RandomForestRegressor( n_estimators=self.n_estimators,
			criterion=self.criterion, max_depth=self.max_depth,
			min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
			min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_features=self.max_features,
			max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease,
			bootstrap=self.bootstrap, oob_score=self.oob_score, n_jobs=self.n_jobs,
			random_state=self.random_state, verbose=self.verbose, warm_start=self.warm_start,
			ccp_alpha=self.ccp_alpha, max_samples=self.max_samples,
			monotonic_cst=self.monotonic_cst )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the RandomForest wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'n_estimators',
				'criterion',
				'max_depth',
				'min_samples_split',
				'min_samples_leaf',
				'min_weight_fraction_leaf',
				'max_features',
				'max_leaf_nodes',
				'min_impurity_decrease',
				'bootstrap',
				'oob_score',
				'n_jobs',
				'random_state',
				'verbose',
				'warm_start',
				'ccp_alpha',
				'max_samples',
				'monotonic_cst',
				'features',
				'training_score',
				'testing_score',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'split_data',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot'
		]
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying RandomForest estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> RandomForest | None:
		"""Train.

				Purpose:
				    Fits the underlying RandomForest regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    RandomForest | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'train( self, *args ) -> RandomForest | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted RandomForest regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted RandomForest regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted RandomForest model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [
							'Training Score',
							'Testing Score',
							'R-Squared',
							'MAE',
							'MSE',
							'RMSE',
							'EVS',
							'Median AE',
							'MAX'
					],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score,
							self.mean_absolute_error,
							self.mean_squared_error,
							self.root_mean_squared_error,
							self.explained_variance_score,
							self.median_absolute_error,
							self.max_error
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted RandomForest regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
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
			exception.cause = 'RandomForest'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class GradientBoost( Regression ):
	"""GradientBoost regression wrapper.

		Purpose:
		    Wraps sklearn.ensemble.GradientBoostingRegressor for stage-wise boosted tree regression, prediction, scoring, metric analysis, and diagnostic scatter plotting.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    loss: Estimator configuration or runtime state retained by the wrapper.
		    learning_rate: Estimator learning-rate configuration.
		    n_estimators: Number of ensemble estimators.
		    subsample: Estimator configuration or runtime state retained by the wrapper.
		    criterion: Split or impurity criterion used by tree-based estimators.
		    min_samples_split: Estimator configuration or runtime state retained by the wrapper.
		    min_samples_leaf: Estimator configuration or runtime state retained by the wrapper.
		    min_weight_fraction_leaf: Estimator configuration or runtime state retained by the wrapper.
		    max_depth: Maximum tree depth.
		    min_impurity_decrease: Estimator configuration or runtime state retained by the wrapper.
		    init: Estimator configuration or runtime state retained by the wrapper.
		    random_state: Random seed or random-state configuration used by the estimator.
		    max_features: Estimator configuration or runtime state retained by the wrapper.
		    alpha: Regularization strength or loss parameter passed to the estimator.
		    verbose: Estimator configuration or runtime state retained by the wrapper.
		    max_leaf_nodes: Estimator configuration or runtime state retained by the wrapper.
		    warm_start: Estimator configuration or runtime state retained by the wrapper.
		    validation_fraction: Estimator configuration or runtime state retained by the wrapper.
		    n_iter_no_change: Estimator configuration or runtime state retained by the wrapper.
		    tol: Optimization tolerance passed to the estimator.
		    ccp_alpha: Estimator configuration or runtime state retained by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: ske.GradientBoostingRegressor
	loss: str
	learning_rate: float
	n_estimators: int
	subsample: float
	criterion: str
	min_samples_split: int | float
	min_samples_leaf: int | float
	min_weight_fraction_leaf: float
	max_depth: Optional[ int ]
	min_impurity_decrease: float
	init: Optional[ object ]
	random_state: Optional[ int ]
	max_features: int | float | str | None
	alpha: float
	verbose: int
	max_leaf_nodes: Optional[ int ]
	warm_start: bool
	validation_fraction: float
	n_iter_no_change: Optional[ int ]
	tol: float
	ccp_alpha: float
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, loss: str = 'squared_error', rate: float = 0.1,
			estimators: int = 100, subsample: float = 1.0, criterion: str = 'friedman_mse',
			split: int | float = 2, leaf: int | float = 1, weight_fraction: float = 0.0,
			depth: Optional[ int ] = 3, impurity: float = 0.0, init: Optional[ object ] = None,
			rando: Optional[ int ] = 42, features: int | float | str | None = None,
			alpha: float = 0.9, verbose: int = 0, leaf_nodes: Optional[ int ] = None,
			warm: bool = False, validation_fraction: float = 0.1, no_change: Optional[ int ] = None,
			tol: float = 1e-4, ccp_alpha: float = 0.0 ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the GradientBoost wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    loss: Loss function used by the estimator.
				    rate: Learning rate used by boosting estimators.
				    estimators: Number of estimators in an ensemble.
				    subsample: Subsample fraction used by boosting estimators.
				    criterion: Tree split or ensemble loss criterion.
				    split: Minimum samples required to split an internal tree node.
				    leaf: Leaf-size or minimum leaf configuration.
				    weight_fraction: Minimum weighted fraction required at a leaf node.
				    depth: Maximum tree depth.
				    impurity: Minimum impurity decrease required for a split.
				    init: Initial estimator used by boosting.
				    rando: Random-state seed passed to the underlying estimator.
				    features: Maximum feature-selection setting used by the estimator.
				    alpha: Regularization strength, loss parameter, or model-specific alpha value.
				    verbose: Verbosity flag or level passed to the estimator.
				    leaf_nodes: Maximum number of leaf nodes.
				    warm: Flag indicating whether previous estimator state is reused across fits.
				    validation_fraction: Training-data fraction reserved for early-stopping validation.
				    no_change: Number of iterations with no improvement before stopping.
				    tol: Numerical tolerance used by estimator optimization or convergence checks.
				    ccp_alpha: Complexity parameter used for minimal cost-complexity pruning.
		"""
		super( ).__init__( )
		self.loss = loss
		self.learning_rate = rate
		self.n_estimators = estimators
		self.subsample = subsample
		self.criterion = criterion
		self.min_samples_split = split
		self.min_samples_leaf = leaf
		self.min_weight_fraction_leaf = weight_fraction
		self.max_depth = depth
		self.min_impurity_decrease = impurity
		self.init = init
		self.random_state = rando
		self.max_features = features
		self.alpha = alpha
		self.verbose = verbose
		self.max_leaf_nodes = leaf_nodes
		self.warm_start = warm
		self.validation_fraction = validation_fraction
		self.n_iter_no_change = no_change
		self.tol = tol
		self.ccp_alpha = ccp_alpha
		self.model = ske.GradientBoostingRegressor( loss=self.loss,
			learning_rate=self.learning_rate,
			n_estimators=self.n_estimators, subsample=self.subsample, criterion=self.criterion,
			min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
			min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_depth=self.max_depth,
			min_impurity_decrease=self.min_impurity_decrease, init=self.init,
			random_state=self.random_state, max_features=self.max_features, alpha=self.alpha,
			verbose=self.verbose, max_leaf_nodes=self.max_leaf_nodes, warm_start=self.warm_start,
			validation_fraction=self.validation_fraction, n_iter_no_change=self.n_iter_no_change,
			tol=self.tol, ccp_alpha=self.ccp_alpha )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the GradientBoost wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'loss',
				'learning_rate',
				'n_estimators',
				'subsample',
				'criterion',
				'min_samples_split',
				'min_samples_leaf',
				'min_weight_fraction_leaf',
				'max_depth',
				'min_impurity_decrease',
				'init',
				'random_state',
				'max_features',
				'alpha',
				'verbose',
				'max_leaf_nodes',
				'warm_start',
				'validation_fraction',
				'n_iter_no_change',
				'tol',
				'ccp_alpha',
				'features',
				'training_score',
				'testing_score',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'split_data',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot'
		]
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying GradientBoost estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoost | None:
		"""Train.

				Purpose:
				    Fits the underlying GradientBoost regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    GradientBoost | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'train( self, *args ) -> GradientBoost | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted GradientBoost regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted GradientBoost regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted GradientBoost model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [
							'Training Score',
							'Testing Score',
							'R-Squared',
							'MAE',
							'MSE',
							'RMSE',
							'EVS',
							'Median AE',
							'MAX'
					],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score,
							self.mean_absolute_error,
							self.mean_squared_error,
							self.root_mean_squared_error,
							self.explained_variance_score,
							self.median_absolute_error,
							self.max_error
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted GradientBoost regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' } )
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

class AdaptiveBoost( Regression ):
	"""AdaptiveBoost regression wrapper.

		Purpose:
		    Wraps sklearn.ensemble.AdaBoostRegressor for adaptive boosted regression using a configurable base estimator and loss function.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    estimator: Estimator configuration or runtime state retained by the wrapper.
		    n_estimators: Number of ensemble estimators.
		    learning_rate: Estimator learning-rate configuration.
		    loss: Estimator configuration or runtime state retained by the wrapper.
		    random_state: Random seed or random-state configuration used by the estimator.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: ske.AdaBoostRegressor
	estimator: Optional[ object ]
	n_estimators: int
	learning_rate: float
	loss: str
	random_state: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, estimator: Optional[ object ] = None, estimators: int = 50,
			rate: float = 1.0, loss: str = 'linear',
			rando: Optional[ int ] = 42 ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the AdaptiveBoost wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    estimator: Base estimator used by an ensemble wrapper.
				    estimators: Number of estimators in an ensemble.
				    rate: Learning rate used by boosting estimators.
				    loss: Loss function used by the estimator.
				    rando: Random-state seed passed to the underlying estimator.
		"""
		super( ).__init__( )
		self.estimator = estimator
		self.n_estimators = estimators
		self.learning_rate = rate
		self.loss = loss
		self.random_state = rando
		self.model = ske.AdaBoostRegressor( estimator=self.estimator,
			n_estimators=self.n_estimators,
			learning_rate=self.learning_rate, loss=self.loss, random_state=self.random_state )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the AdaptiveBoost wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'estimator',
				'n_estimators',
				'learning_rate',
				'loss',
				'random_state',
				'base_estimator',
				'features',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'training_score',
				'testing_score'
		]
	
	@property
	def base_estimator( self ) -> object | None:
		"""Base estimator.

				Purpose:
				    Returns fitted `base_estimator` metadata from the underlying AdaptiveBoost estimator after training.

				Returns:
				    object | None: Fitted `base_estimator` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if hasattr( self.model, 'estimator_' ):
			return self.model.estimator_
		raise AttributeError( 'The model has not been trained!' )
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying AdaptiveBoost estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> AdaptiveBoost | None:
		"""Train.

				Purpose:
				    Fits the underlying AdaptiveBoost regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    AdaptiveBoost | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'train( self, *args ) -> AdaptiveBoost | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted AdaptiveBoost regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted AdaptiveBoost regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted AdaptiveBoost model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [
							'Training Score',
							'Testing Score',
							'R-Squared',
							'MAE',
							'MSE',
							'RMSE',
							'EVS',
							'Median AE',
							'MAX'
					],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score,
							self.mean_absolute_error,
							self.mean_squared_error,
							self.root_mean_squared_error,
							self.explained_variance_score,
							self.median_absolute_error,
							self.max_error
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted AdaptiveBoost regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
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

class BaggingModel( Regression ):
	"""BaggingModel regression wrapper.

		Purpose:
		    Wraps sklearn.ensemble.BaggingRegressor for bootstrap-aggregated regression over configurable base estimators, samples, features, and parallel execution.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    estimator: Estimator configuration or runtime state retained by the wrapper.
		    n_estimators: Number of ensemble estimators.
		    max_samples: Estimator configuration or runtime state retained by the wrapper.
		    max_features: Estimator configuration or runtime state retained by the wrapper.
		    bootstrap: Estimator configuration or runtime state retained by the wrapper.
		    bootstrap_features: Estimator configuration or runtime state retained by the wrapper.
		    oob_score: Estimator configuration or runtime state retained by the wrapper.
		    warm_start: Estimator configuration or runtime state retained by the wrapper.
		    n_jobs: Number of parallel worker jobs used by the estimator.
		    random_state: Random seed or random-state configuration used by the estimator.
		    verbose: Estimator configuration or runtime state retained by the wrapper.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: ske.BaggingRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	estimator: Optional[ object ]
	n_estimators: int
	max_samples: Optional[ int | float ]
	max_features: int | float
	bootstrap: bool
	bootstrap_features: bool
	oob_score: bool
	warm_start: bool
	n_jobs: Optional[ int ]
	random_state: Optional[ int ]
	verbose: int
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, estimator: Optional[ object ] = None, num: int = 10,
			samples: Optional[ int | float ] = None, features: int | float = 1.0,
			bootstrap: bool = True, bootstrap_features: bool = False,
			oob_score: bool = False, warm: bool = False, jobs: Optional[ int ] = None,
			rando: Optional[ int ] = None, verbose: int = 0 ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the BaggingModel wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    estimator: Base estimator used by an ensemble wrapper.
				    num: Number of neighbors, estimators, or model-specific components.
				    samples: Maximum sample setting used by ensemble estimators.
				    features: Maximum feature-selection setting used by the estimator.
				    bootstrap: Flag enabling bootstrap sampling.
				    bootstrap_features: Configuration value passed to the underlying regression estimator.
				    oob_score: Flag enabling out-of-bag scoring.
				    warm: Flag indicating whether previous estimator state is reused across fits.
				    jobs: Number of parallel worker jobs used by the estimator.
				    rando: Random-state seed passed to the underlying estimator.
				    verbose: Verbosity flag or level passed to the estimator.
		"""
		super( ).__init__( )
		self.estimator = estimator
		self.n_estimators = num
		self.max_samples = samples
		self.max_features = features
		self.bootstrap = bootstrap
		self.bootstrap_features = bootstrap_features
		self.oob_score = oob_score
		self.warm_start = warm
		self.n_jobs = jobs
		self.random_state = rando
		self.verbose = verbose
		self.model = ske.BaggingRegressor( estimator=self.estimator, n_estimators=self.n_estimators,
			max_samples=self.max_samples, max_features=self.max_features, bootstrap=self.bootstrap,
			bootstrap_features=self.bootstrap_features, oob_score=self.oob_score,
			warm_start=self.warm_start, n_jobs=self.n_jobs, random_state=self.random_state,
			verbose=self.verbose )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the BaggingModel wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'estimator',
				'n_estimators',
				'max_samples',
				'max_features',
				'bootstrap',
				'bootstrap_features',
				'oob_score',
				'warm_start',
				'n_jobs',
				'random_state',
				'verbose',
				'base_estimator',
				'features',
				'training_score',
				'testing_score',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'split_data',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot'
		]
	
	@property
	def base_estimator( self ) -> object | None:
		"""Base estimator.

				Purpose:
				    Returns fitted `base_estimator` metadata from the underlying BaggingModel estimator after training.

				Returns:
				    object | None: Fitted `base_estimator` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if hasattr( self.model, 'estimator_' ):
			return self.model.estimator_
		raise AttributeError( 'The model has not been trained!' )
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying BaggingModel estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BaggingModel | None:
		"""Train.

				Purpose:
				    Fits the underlying BaggingModel regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    BaggingModel | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'train( self, *args ) -> BaggingModel | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted BaggingModel regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted BaggingModel regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( {
					'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [
							self.training_score if self.training_score is not None else np.nan,
							self.testing_score if self.testing_score is not None else np.nan,
							self.r2_score
					]
			} )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted BaggingModel model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [
								'Training Score',
								'Testing Score',
								'R-Squared',
								'MAE',
								'MSE',
								'RMSE',
								'EVS',
								'Median AE',
								'MAX'
						],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score,
								self.mean_absolute_error,
								self.mean_squared_error,
								self.root_mean_squared_error,
								self.explained_variance_score,
								self.median_absolute_error,
								self.max_error
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted BaggingModel regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' } )
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

class VotingModel( Regression ):
	"""VotingModel regression wrapper.

		Purpose:
		    Wraps sklearn.ensemble.VotingRegressor for averaging predictions from multiple named regression estimators with optional weights.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    estimators: Named estimator collection used by ensemble wrappers.
		    weights: Optional estimator or neighbor weights.
		    n_jobs: Number of parallel worker jobs used by the estimator.
		    verbose: Estimator configuration or runtime state retained by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: ske.VotingRegressor
	estimators: List[ tuple[ str, object ] ]
	weights: Optional[ List[ float ] ]
	n_jobs: Optional[ int ]
	verbose: bool
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, est: Optional[ List[ tuple[ str, object ] ] ] = None,
			weights: Optional[ List[ float ] ] = None, jobs: Optional[ int ] = None,
			verbose: bool = False ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the VotingModel wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    est: Named estimator list used by voting or stacking wrappers.
				    weights: Estimator weights used during voting or averaging.
				    jobs: Number of parallel worker jobs used by the estimator.
				    verbose: Verbosity flag or level passed to the estimator.
		"""
		super( ).__init__( )
		self.estimators = est if est is not None else [ ('least_squares', skl.LinearRegression( )),
		                                                ('ridge', skl.Ridge( )),
		                                                ('nearest_neighbor',
		                                                 skn.KNeighborsRegressor( )) ]
		self.weights = weights
		self.n_jobs = jobs
		self.verbose = verbose
		self.model = ske.VotingRegressor( estimators=self.estimators, weights=self.weights,
			n_jobs=self.n_jobs, verbose=self.verbose )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the VotingModel wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'estimators',
				'weights',
				'n_jobs',
				'verbose',
				'features',
				'named_estimators',
				'training_score',
				'testing_score',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'split_data',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot'
		]
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying VotingModel estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def named_estimators( self ) -> Dict[ str, object ] | None:
		"""Named estimators.

				Purpose:
				    Returns fitted `named_estimators` metadata from the underlying VotingModel estimator after training.

				Returns:
				    Dict[str, object] | None: Fitted `named_estimators` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'named_estimators_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.named_estimators_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> VotingModel | None:
		"""Train.

				Purpose:
				    Fits the underlying VotingModel regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    VotingModel | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'train( self, *args ) -> VotingModel | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted VotingModel regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted VotingModel regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted VotingModel model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [
								'Training Score',
								'Testing Score',
								'R-Squared',
								'MAE',
								'MSE',
								'RMSE',
								'EVS',
								'Median AE',
								'MAX'
						],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score,
								self.mean_absolute_error,
								self.mean_squared_error,
								self.root_mean_squared_error,
								self.explained_variance_score,
								self.median_absolute_error,
								self.max_error
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted VotingModel regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 },
				line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ],
				'k--', label='Perfect Prediction' )
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
			exception.cause = 'VotingModel'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class StackingModel( Regression ):
	"""StackingModel regression wrapper.

		Purpose:
		    Wraps sklearn.ensemble.StackingRegressor for stacked generalization using base regressors, a final estimator, cross-validation, and optional passthrough features.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    estimators: Named estimator collection used by ensemble wrappers.
		    final_estimator: Final estimator used by stacked regression.
		    cv: Estimator configuration or runtime state retained by the wrapper.
		    n_jobs: Number of parallel worker jobs used by the estimator.
		    passthrough: Estimator configuration or runtime state retained by the wrapper.
		    verbose: Estimator configuration or runtime state retained by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: ske.StackingRegressor
	estimators: List[ tuple[ str, object ] ]
	final_estimator: object | None
	cv: int | None
	n_jobs: int | None
	passthrough: bool
	verbose: int
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, est: Optional[ List[ tuple[ str, object ] ] ] = None, final: object = None,
			cv: int = None, jobs: int = None, passthrough: bool = False, verbose: int = 0 ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the StackingModel wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    est: Named estimator list used by voting or stacking wrappers.
				    final: Final estimator used by stacked regression.
				    cv: Cross-validation configuration used by stacked regression.
				    jobs: Number of parallel worker jobs used by the estimator.
				    passthrough: Flag indicating whether original features are passed to the final estimator.
				    verbose: Verbosity flag or level passed to the estimator.
		"""
		super( ).__init__( )
		self.estimators = est if est is not None else [ ('least_squares', skl.LinearRegression( )),
		                                                ('ridge', skl.Ridge( )),
		                                                ('nearest_neighbor',
		                                                 skn.KNeighborsRegressor( )) ]
		self.final_estimator = final
		self.cv = cv
		self.n_jobs = jobs
		self.passthrough = passthrough
		self.verbose = verbose
		self.model = ske.StackingRegressor( estimators=self.estimators,
			final_estimator=self.final_estimator, cv=self.cv, n_jobs=self.n_jobs,
			passthrough=self.passthrough )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the StackingModel wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'estimators',
				'final_estimator',
				'cv',
				'n_jobs',
				'passthrough',
				'verbose',
				'features',
				'estimator_list',
				'final',
				'training_score',
				'testing_score',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'split_data',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot'
		]
	
	@property
	def estimator_list( self ) -> List[ object ] | None:
		"""Estimator list.

				Purpose:
				    Returns fitted `estimator_list` metadata from the underlying StackingModel estimator after training.

				Returns:
				    List[object] | None: Fitted `estimator_list` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'estimators_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.estimators_
	
	@property
	def final( self ) -> object | None:
		"""Final.

				Purpose:
				    Returns fitted `final` metadata from the underlying StackingModel estimator after training.

				Returns:
				    object | None: Fitted `final` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'final_estimator_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.final_estimator_
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying StackingModel estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return (self.X_train, self.X_test, self.y_train, self.y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> StackingModel | None:
		"""Train.

				Purpose:
				    Fits the underlying StackingModel regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    StackingModel | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'train( self, *args ) -> StackingModel | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted StackingModel regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted StackingModel regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted StackingModel model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [
								'Training Score',
								'Testing Score',
								'R-Squared',
								'MAE',
								'MSE',
								'RMSE',
								'EVS',
								'Median AE',
								'MAX'
						],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score,
								self.mean_absolute_error,
								self.mean_squared_error,
								self.root_mean_squared_error,
								self.explained_variance_score,
								self.median_absolute_error,
								self.max_error
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted StackingModel regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
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
			exception.cause = 'StackingModel'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class SupportVector( Regression ):
	"""SupportVector regression wrapper.

		Purpose:
		    Wraps sklearn.svm.SVR for kernel-based support-vector regression with configurable kernel, regularization, tolerance, epsilon, and cache behavior.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    kernel: Estimator configuration or runtime state retained by the wrapper.
		    degree: Estimator configuration or runtime state retained by the wrapper.
		    gamma: Estimator configuration or runtime state retained by the wrapper.
		    coef0: Estimator configuration or runtime state retained by the wrapper.
		    tol: Optimization tolerance passed to the estimator.
		    penalty: Estimator configuration or runtime state retained by the wrapper.
		    epsilon: Estimator configuration or runtime state retained by the wrapper.
		    shrinking: Estimator configuration or runtime state retained by the wrapper.
		    cache_size: Estimator configuration or runtime state retained by the wrapper.
		    verbose: Estimator configuration or runtime state retained by the wrapper.
		    max_iter: Maximum number of estimator iterations.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
		    X_train: Estimator configuration or runtime state retained by the wrapper.
		    X_test: Estimator configuration or runtime state retained by the wrapper.
		    y_train: Estimator configuration or runtime state retained by the wrapper.
		    y_test: Estimator configuration or runtime state retained by the wrapper.
	"""
	model: skv.SVR
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	kernel: str
	degree: int
	gamma: str | float
	coef0: float
	tol: float
	penalty: float
	epsilon: float
	shrinking: bool
	cache_size: float
	verbose: bool
	max_iter: int
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, kernel: str = 'rbf', degree: int = 3, gamma: str | float = 'scale',
			coef0: float = 0.0, tol: float = 1e-3, penalty: float = 1.0,
			epsilon: float = 0.1, shrinking: bool = True, cache: float = 200.0,
			verbose: bool = False, iters: int = -1 ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the SupportVector wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    kernel: Kernel configuration used by kernel-based estimators.
				    degree: Polynomial degree used by kernel-based estimators.
				    gamma: Kernel coefficient used by support-vector regression.
				    coef0: Independent kernel coefficient used by polynomial or sigmoid kernels.
				    tol: Numerical tolerance used by estimator optimization or convergence checks.
				    penalty: Regularization parameter used by the estimator.
				    epsilon: Epsilon-insensitive loss parameter or robustness threshold.
				    shrinking: Flag enabling the shrinking heuristic for support-vector optimization.
				    cache: Kernel cache size in megabytes.
				    verbose: Verbosity flag or level passed to the estimator.
				    iters: Maximum number of optimization iterations.
		"""
		super( ).__init__( )
		self.kernel = kernel
		self.degree = degree
		self.gamma = gamma
		self.coef0 = coef0
		self.tol = tol
		self.penalty = penalty
		self.epsilon = epsilon
		self.shrinking = shrinking
		self.cache_size = cache
		self.verbose = verbose
		self.max_iter = iters
		self.model = skv.SVR( kernel=self.kernel, degree=self.degree, gamma=self.gamma,
			coef0=self.coef0, tol=self.tol, C=self.penalty, epsilon=self.epsilon,
			shrinking=self.shrinking, cache_size=self.cache_size, verbose=self.verbose,
			max_iter=self.max_iter )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the SupportVector wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'kernel',
				'degree',
				'gamma',
				'coef0',
				'tol',
				'penalty',
				'epsilon',
				'shrinking',
				'cache_size',
				'verbose',
				'max_iter',
				'features',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'training_score',
				'testing_score'
		]
	
	@property
	def features( self ) -> int:
		"""Features.

				Purpose:
				    Returns fitted `features` metadata from the underlying SupportVector estimator after training.

				Returns:
				    int: Fitted `features` metadata from the underlying estimator.

				Raises:
				    AttributeError: Raised when fitted estimator metadata is unavailable.
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> SupportVector | None:
		"""Train.

				Purpose:
				    Fits the underlying SupportVector regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    SupportVector | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'train( self, *args ) -> SupportVector | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted SupportVector regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted SupportVector regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame(
				{
						'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted SupportVector model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			
			df_metrics = pd.DataFrame(
				{
						'Metric': [
								'Training Score',
								'Testing Score',
								'R-Squared',
								'MAE',
								'MSE',
								'RMSE',
								'EVS',
								'Median AE',
								'MAX'
						],
						'Value': [
								self.training_score if self.training_score is not None else np.nan,
								self.testing_score if self.testing_score is not None else np.nan,
								self.r2_score,
								self.mean_absolute_error,
								self.mean_squared_error,
								self.root_mean_squared_error,
								self.explained_variance_score,
								self.median_absolute_error,
								self.max_error
						]
				}
			)
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted SupportVector regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
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
			exception.cause = 'SupportVector'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class GaussianProcess( Regression ):
	"""GaussianProcess regression wrapper.

		Purpose:
		    Wraps sklearn.gaussian_process.GaussianProcessRegressor for kernel-based probabilistic regression with optimizer, alpha, restart, and normalization configuration.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    kernel: Estimator configuration or runtime state retained by the wrapper.
		    alpha: Regularization strength or loss parameter passed to the estimator.
		    optimizer: Estimator configuration or runtime state retained by the wrapper.
		    n_restarts_optimizer: Estimator configuration or runtime state retained by the wrapper.
		    normalize_y: Estimator configuration or runtime state retained by the wrapper.
		    copy_X_train: Estimator configuration or runtime state retained by the wrapper.
		    random_state: Random seed or random-state configuration used by the estimator.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    max_error: Most recent maximum residual error metric.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
	"""
	model: GaussianProcessRegressor
	kernel: object | None
	alpha: float | np.ndarray
	optimizer: str | object | None
	n_restarts_optimizer: int
	normalize_y: bool
	copy_X_train: bool
	random_state: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, kernel: object = None, alpha: float = 1e-10,
			optimizer: str = 'fmin_l_bfgs_b',
			restarts: int = 0, normalize: bool = False, copy: bool = True,
			rando: int = None ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the GaussianProcess wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    kernel: Kernel configuration used by kernel-based estimators.
				    alpha: Regularization strength, loss parameter, or model-specific alpha value.
				    optimizer: Optimizer used by Gaussian-process regression.
				    restarts: Number of optimizer restarts.
				    normalize: Flag indicating whether targets are normalized before fitting.
				    copy: Flag indicating whether input feature data is copied during fitting.
				    rando: Random-state seed passed to the underlying estimator.
		"""
		super( ).__init__( )
		self.kernel = kernel
		self.alpha = alpha
		self.optimizer = optimizer
		self.n_restarts_optimizer = restarts
		self.normalize_y = normalize
		self.copy_X_train = copy
		self.random_state = rando
		self.model = GaussianProcessRegressor( kernel=self.kernel, alpha=self.alpha,
			optimizer=self.optimizer, n_restarts_optimizer=self.n_restarts_optimizer,
			normalize_y=self.normalize_y, copy_X_train=self.copy_X_train,
			random_state=self.random_state )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the GaussianProcess wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'kernel',
				'alpha',
				'optimizer',
				'n_restarts_optimizer',
				'normalize_y',
				'copy_X_train',
				'random_state',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'training_score',
				'testing_score'
		]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size,
				random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GaussianProcess | None:
		"""Train.

				Purpose:
				    Fits the underlying GaussianProcess regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    GaussianProcess | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'train( self, *args ) -> GaussianProcess | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted GaussianProcess regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted GaussianProcess regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.r2_score = r2_score( y, self.prediction )
			
			_metrics = { 'Training Score': self.training_score, 'Testing Score': self.testing_score,
			             'R-Squared Score': self.r2_score, }
			
			idx = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=idx )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted GaussianProcess model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			
			_metrics = \
				{
						'MAE': self.mean_absolute_error,
						'MSE': self.mean_squared_error,
						'RMSE': self.root_mean_squared_error,
						'EVS': self.explained_variance_score,
						'MAX': self.max_error,
				}
			
			_data = pd.Series( _metrics )
			df_metrics = pd.DataFrame( _data )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted GaussianProcess regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
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
			exception.cause = 'GaussianProcess'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class MultiLayerPerceptron( Regression ):
	"""MultiLayerPerceptron regression wrapper.

		Purpose:
		    Wraps sklearn.neural_network.MLPRegressor for feed-forward neural-network regression with configurable architecture, activation, solver, regularization, and learning behavior.

		Attributes:
		    model: Underlying sklearn regression estimator managed by the wrapper.
		    prediction: Most recent prediction array returned by the fitted estimator.
		    transformed_data: Estimator configuration or runtime state retained by the wrapper.
		    mean_absolute_error: Most recent mean absolute error metric.
		    mean_squared_error: Most recent mean squared error metric.
		    root_mean_squared_error: Most recent root mean squared error metric.
		    r2_score: Most recent coefficient-of-determination metric.
		    explained_variance_score: Most recent explained-variance metric.
		    median_absolute_error: Estimator configuration or runtime state retained by the wrapper.
		    random_state: Random seed or random-state configuration used by the estimator.
		    alpha: Regularization strength or loss parameter passed to the estimator.
		    learning: Estimator configuration or runtime state retained by the wrapper.
		    activation_function: Estimator configuration or runtime state retained by the wrapper.
		    solver: Solver selected for estimator optimization.
		    hidden_layers: Estimator configuration or runtime state retained by the wrapper.
		    testing_score: Most recent estimator score on the testing split.
		    training_score: Most recent estimator score on the training split.
	"""
	model: skm.MLPRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	random_state: Optional[ int ]
	alpha: Optional[ float ]
	learning: str
	activation_function: str
	solver: str
	hidden_layers: tuple
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, hidden: tuple = (100,), activ: str = 'relu', solver: str = 'adam',
			alpha: float = 0.0001, learning: str = 'constant', rando: int = 42 ) -> None:
		"""Initialize instance.

				Purpose:
				    Initializes the MultiLayerPerceptron wrapper with estimator configuration, runtime state, cached prediction fields, and regression metric fields required by training and evaluation workflows.

				Args:
				    hidden: Hidden-layer architecture for neural-network regression.
				    activ: Activation function used by neural-network regression.
				    solver: Optimization solver used by the estimator.
				    alpha: Regularization strength, loss parameter, or model-specific alpha value.
				    learning: Learning-rate schedule used by neural-network regression.
				    rando: Random-state seed passed to the underlying estimator.
		"""
		super( ).__init__( )
		self.hidden_layers = hidden
		self.activation_function = activ
		self.solver = solver
		self.alpha = alpha
		self.learning = learning
		self.random_state = rando
		self.model = skm.MLPRegressor(
			hidden_layer_sizes=self.hidden_layers,
			activation=self.activation_function,
			solver=self.solver,
			alpha=self.alpha,
			learning_rate=self.learning,
			random_state=self.random_state
		)
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.

				Purpose:
				    Returns the stable public API surface exposed by the MultiLayerPerceptron wrapper for interactive inspection and documentation generation.

				Returns:
				    List[str]: Public member names exposed by the wrapper.
		"""
		return [
				'model',
				'prediction',
				'hidden_layers',
				'activation_function',
				'solver',
				'alpha',
				'learning',
				'random_state',
				'mean_absolute_error',
				'mean_squared_error',
				'root_mean_squared_error',
				'r2_score',
				'explained_variance_score',
				'median_absolute_error',
				'max_error',
				'train',
				'project',
				'score',
				'analyze',
				'scatter_plot',
				'training_score',
				'testing_score'
		]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.

				Purpose:
				    Splits aligned feature and target arrays into training and testing partitions using sklearn.model_selection.train_test_split with wrapper defaults.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.
				    size: Testing-set proportion used by train/test splitting.
				    random: Random seed used by train/test splitting.

				Returns:
				    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training features, testing features, training targets, and testing targets.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size,
				random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> MultiLayerPerceptron | None:
		"""Train.

				Purpose:
				    Fits the underlying MultiLayerPerceptron regression estimator to feature and target arrays and returns the wrapper for chained modeling workflows.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    MultiLayerPerceptron | None: Fitted wrapper instance.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
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
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.

				Purpose:
				    Generates continuous target predictions from the fitted MultiLayerPerceptron regression estimator and caches the prediction array on the wrapper.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.

				Returns:
				    np.ndarray | None: Predicted continuous target values generated by the fitted estimator.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.

				Purpose:
				    Evaluates the fitted MultiLayerPerceptron regression estimator with its primary scoring behavior and returns the score in the wrapper reporting format.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing the primary model score and evaluation fields.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.r2_score = r2_score( y, self.prediction )
			
			_metrics = {
					'Training Score': self.training_score,
					'Testing Score': self.testing_score,
					'R-Squared Score': self.r2_score,
			}
			
			idx = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=idx )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.

				Purpose:
				    Computes regression evaluation metrics for the fitted MultiLayerPerceptron model, including error, explained-variance, and coefficient-of-determination measures.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Returns:
				    pd.DataFrame | None: Dataframe containing regression metrics computed from observed and predicted target values.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			
			_metrics = {
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.root_mean_squared_error,
					'EVS': self.explained_variance_score,
					'MAX': self.max_error,
			}
			
			_data = pd.Series( _metrics )
			df_metrics = pd.DataFrame( _data )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.

				Purpose:
				    Renders a diagnostic scatter plot comparing observed target values against predictions from the fitted MultiLayerPerceptron regression estimator.

				Args:
				    X: Feature matrix used for fitting, prediction, splitting, scoring, or metric evaluation.
				    y: Target vector aligned to the rows of `X`.

				Raises:
				    Error: Raised when validation or wrapped regression estimator execution fails.
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
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
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception
			
