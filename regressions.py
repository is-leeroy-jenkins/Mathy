"""
	******************************************************************************************
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
	        regressions.py
	</summary>
	******************************************************************************************
"""
from __future__ import annotations
from boogr import Error, ErrorDialog
from typing import Dict
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble as ske
import sklearn.linear_model as skl
import sklearn.neighbors as skn
import sklearn.neural_network as skn
import sklearn.svm as skv
import sklearn.tree as skd
from sklearn.base import ClassifierMixin
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             explained_variance_score, median_absolute_error, max_error,
                             accuracy_score, )

def throw_if( name: str, value: object ):
	if value is None:
		raise Exception( f'Argument "{name}" cannot be empty!' )

class Regression:
	"""

    Purpose:
    ---------
    Abstract base class that defines the interface for all linerar_model wrappers.

    """
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self ):
		pass
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""
	
	        Purpose:
	        ---------
	        Fits the model to the training stores
	
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
	        Predictions using the trained model.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
	
	        Returns:
	        -----------
	        np.ndarray: Predicted target_names or class target_names.

        """
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
	
	        Purpose:
	        ---------
	        Return the core regression metric (e.g., R²).
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        -----------
	        float: Score value (e.g., R² for regressors).

        """
		raise NotImplementedError
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

	        Purpose:
	        ---------
	        
				Mean Absolute Error - MAE
				Mean Squared Error - MSE
				Root MSE - RSME
				R-squared - R2
				Explained Varriance Score - EVS
				Max_Error = MAX
			
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	        
	        Returns:
	        -----------
	        dict: Dictionary containing multiple evaluation metrics.

        """
		raise NotImplementedError


class LinearRegression( Regression ):
	"""

	    Purpose:
	    -----------
	    Ordinary Least Squares Regression fits a linear model with coefficients to minimize the
	    residual sum of squares between the observed targets in the dataset, and the targets
	    predicted by the linear approximation. The coefficient estimates for Ordinary Least Squares
	    rely on the independence of the feature_names.
	
	    When feature_names are correlated and the n_features of the design matrix have an approximately
	    linear dependence, the design matrix becomes close to singular and as a result,
	    the least-squares estimate becomes highly sensitive to random errors in the observed target,
	    producing a large variance. This situation of multicollinearity can arise, for example,
	    when stores are collected without an experimental design.

    """
	
	model: skl.LinearRegression
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, fit: bool=True, copy: bool=True ) -> None:
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
		self.fit_intercept = fit
		self.copy_X = copy
		self.model = skl.LinearRegression( fit_intercept=self.fit_intercept, copy_X=self.copy_X )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""
	
	        Purpose:
	        -------
	        Provides a list of strings representing class members

        """
		return [ 'prediction', 'accuracy', 'learning_rate', 'n_estimators', 'random_state',
		         'weights', 'max_depth', 'mean_absolute_error', 'mean_squared_error',
		         'r_mean_squared_error', 'r2_score', 'explained_variance_score', 'weights',
		         'max_error', 'train', 'project', 'score', 'analyze', 'create_scatter',
		         'weights', 'features' ]
	
	@property
	def weights( self ) -> np.ndarray | None:
		if self.model.coef_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.coef_
	
	@property
	def features( self ) -> np.ndarray:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LinearRegression | None:
		"""
	
	        Purpose:
	        -----------
	        Fit the OLS regression linerar_model.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
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
			exception.cause = "LinearRegressor"
			exception.method = "train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline"
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

	
	        Purpose:
	        -----------
	        Predict target target_names using the OLS linerar_model.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): Input feature matrix.
	
	        Returns:
	        -----------
	        np.ndarray: Predicted target target_names.

        """
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LinearRegression'
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
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
	        Returns:
	        -----------
	        float: R-squared accuracy.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = accuracy_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LinearRegression'
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
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
	        Returns:
	        -----------
	        dict: Dictionary of MAE, MSE, RMSE, R², etc.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'EVS': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LinearRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

	        Purpose:
	        -----------
	        Plot actual vs predicted target_names.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): Input feature matrix.
	        y ( n_samples, ): True target target_names.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( "Observed" )
			plt.ylabel( "Projected" )
			plt.title( "Linear Regression: Observed vs Projected" )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LinearRegression'
			exception.method = ("create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None")
			error = ErrorDialog( exception )
			error.show( )

class Ridge( Regression ):
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
	model: skl.Ridge
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, alpha: float=1.0, solver: str='auto', iters: int=1000, rando: int=42, ) -> None:
		"""


	        Purpose:
	        -----------
	        Initialize the RidgeRegressor linerar_model.
	
	        Attributes:
	        -----------
	        linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
	        
	        Parameters:
	        -----------
	        alpha (float): Regularization strength. Default is 1.0.
	        solver (str): Solver to use. Default is 'auto'.

        """
		super( ).__init__( )
		self.alpha = alpha
		self.solver = solver
		self.max_iter = iters
		self.random_state = rando
		self.model = skl.Ridge( alpha=self.alpha, solver=self.solver,
			max_iter=self.max_iter, random_state=self.random_state, )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""

        Purpose:
        -------
        Provides a list of strings representing class members

        """
		return [ 'prediction', 'accuracy', 'alpha', 'solver', 'random_state', 'max_iter',
		         'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
		         'r2_score', 'explained_variance_score', 'median_absolute_error', 'train',
		         'project', 'score', 'analyze', 'create_scatter', ]
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Ridge | None:
		"""
	
	
	        Purpose:
	        -----------
	        Fit the RidgeRegressor regression linerar_model.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
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
			exception.cause = 'Ridge'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Ridge'
			error = ErrorDialog( exception )
			error.show( )
	
	@property
	def weights( self ) -> np.ndarray | None:
		if self.model.coef_ is None:
			raise AttributeError( 'The classification data is untrained.' )
		else:
			return self.model.coef_
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

	
	        Purpose:
	        -----------
	        Project target target_names using the RidgeRegressor linerar_model.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	
	        Returns:
	        -----------
	        np.ndarray: Predicted target target_names.

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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

	
	        Purpose:
	        -----------
	        Compute the R-squared accuracy for the Ridge model.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
	        Returns:
	        -----------
	        float: R-squared accuracy.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
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
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
	        Returns:
	        -----------
	        dict: Evaluation metrics including MAE, RMSE, R², etc.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'EVS': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = "Ridge"
			exception.method = "analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict"
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


	        Purpose:
	        -----------
	        Plot predicted vs actual target_names.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
	        Returns:
	        -----------
	        None

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Ridge Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )

class Lasso( Regression ):
	"""
	
	    Purpose:
	    --------
	    Linear Classifier trained with L1 for the regularizer. Regularization improves the
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
	
	model: skl.Lasso
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, alpha: float=1.0, iters: int=500, rando: int=42 ) -> None:
		"""

	        Purpose:
	        -----------
	        Initialize the LassoRegression linerar_model.
	        
        """
		super( ).__init__( )
		self.alpha = alpha
		self.max_iter = iters
		self.random_state = rando
		self.model = skl.Lasso( alpha=self.alpha, max_iter=self.max_iter,
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
		"""
	
	        Purpose:
	        -------
	        Provides a list of strings representing class members

        """
		return [ 'prediction', 'accuracy', 'random_state', 'alpha', 'max_iter',
			'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error',
			'r2_score', 'explained_variance_score', 'median_absolute_error', 'train',
			'project', 'score', 'analyze', 'create_scatter', 'weights' ]
	
	@property
	def weights( self ) -> np.ndarray | None:
		if self.model.coef_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
		else:
			return self.model.coef_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Lasso | None:
		"""

	        Purpose:
	        --------
	        Fit the LassoRegression.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
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
			exception.cause = 'Lasso'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> self'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

	        Purpose:
	        -----------
	        Predict target target_names using the LassoRegression linerar_model.
	
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	
	        Returns:
	        -----------
	        np.ndarray: Predicted target target_names.

        """
		try:
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
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
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
	        Returns:
	        -----------
	        float: R^2 accuracy.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
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
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
	        Returns:
	        -----------
	        dict: Dictionary of MAE, RMSE, R², etc.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'EVS': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LassoRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


	        Purpose:
	        -----------
	        Plot actual vs. predicted target_names.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X.reshape( 1, -1 ) )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Lasso Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )

class ElasticNet( Regression ):
	"""

	    Purpose:
	    --------
	    ElasticNet is a linear regression model trained with both L1 and L2-norm regularization of the
	    coefficients. This combination allows for learning a sparse model where few of the weights
	    are non-zero like Lasso, while still maintaining the regularization properties of Ridge.
	    We control the convex combination of and using the l1_ratio parameter.
	
	    Elastic-net is useful when there are multiple feature_names that are correlated with one another.
	    Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

    """
	model: skl.ElasticNet
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, alpha: float=1.0, ratio: float=0.5, max: int=200,
		rando: int=None, select: str='random', ) -> None:
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
		self.model = skl.ElasticNet( alpha=self.alpha, l1_ratio=self.ratio,
			random_state=self.random_state, max_iter=self.max_iter, selection=self.selection, )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""

        Purpose:
        -------
        Provides a list of strings representing class members

        """
		return [ 'prediction', 'accuracy', 'alpha', 'ratio', 'random_state',
			'selection', 'max_iter', 'mean_absolute_error', 'mean_squared_error',
			'r_mean_squared_error', 'r2_score', 'explained_variance_score',
			'median_absolute_error', 'train', 'project', 'score', 'analyze', 'create_scatter', ]
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> ElasticNet | None:
		"""

	
	        Purpose:
	        -----------
	        Fit the ElasticNetRegressor regression linerar_model.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
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
			exception.cause = 'ElasticNet'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> self'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
	
	
	        Purpose:
	        -----------
	        Predict target target_names using the ElasticNetRegressor linerar_model.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
	        Returns:
	        -----------
	        np.ndarray: Predicted target target_names.

        """
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
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
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
	        Returns:
	        -----------
	        float: R^2 accuracy.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
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
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.
	
	        Returns:
	        -----------
	        dict: Evaluation metrics.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
	
	        Purpose:
	        -----------
	        Plot actual vs. predicted regression output.
	
	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'ElasticNet Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNetRegressor'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )

class LeastAngle( Regression ):
	"""

	    Purpose:
	    --------
	    Least-angle regression (LARS) is a regression algorithm for high-dimensional stores.
	    LARS is similar to forward stepwise regression. At each step, it finds the feature most
	    correlated with the target. When there are multiple features having equal correlation,
	    instead of continuing along the same feature, it proceeds in a direction equiangular
	    between the features.

    """
	
	model: skl.Lars
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	eps: float
	n_nonzero_coefs: int
	fit_intercept: bool
	normalize: bool
	precompute: bool
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, coeffs: int=500, fit: bool=True, normal: bool=True,
		precompute: bool=True, ) -> None:
		"""

	        Purpose:
	        --------
	        Initialize the Least Angle Regression model.
	
	        Parameters:
	        -----------
	        coeffs (int): Maximum number of non-zero coefficients. 500 default
	        fit (bool): fit the intercept
	        normal (bool): Normalize
	        precompute (bool): Precompute coefficients

        """
		super( ).__init__( )
		self.fit_intercept = fit
		self.normalize = normal
		self.nonzero_coefficients = coeffs
		self.precompute = precompute
		self.model = skl.Lars( fit_intercept=self.fit_intercept, normalize=self.normalize,
			precompute=self.precompute, n_nonzero_coefs=self.nonzero_coefficients, )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""
	
	        Purpose:
	        -------
	        Provides a list of strings representing class members

        """
		return [  'model', 'prediction', 'accuracy', 'fit_intercept', 'normalize',
			'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error', 'r2_score',
			'explained_variance_score', 'median_absolute_error', 'train', 'project',
			'score', 'analyze', 'plot_scatter', ]
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LeastAngle | None:
		"""

        Purpose:
        -----------
        Fit the least angle
         regression linerar_model.

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
			exception.cause = 'LeastAngle'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

        Purpose:
        -----------
        Predict class target_names using the least angle regression linerar_model.

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
			exception.cause = 'LeastAngle'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

        Purpose:
        -----------
        Compute regression accuracy.

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
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

        Purpose:
        -----------
        Evaluate the regression using multiple classification metrics.

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
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
		
		def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
			"""

	        Purpose:
	        -----------
	        Plot predicted vs. actual target_names.

	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).

	        """
			try:
				throw_if( 'X', X )
				throw_if( 'y', y )
				self.prediction = self.model.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Least-Angle Regression: Observed vs Projected' )
				plt.plot( [ X.min( ), X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
			except Exception as e:
				exception = Error( e )
				exception.module = 'mathy'
				exception.cause = 'LeastAngle'
				exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
				error = ErrorDialog( exception )
				error.show( )

class BayesianRidge( Regression ):
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
	model: skl.BayesianRidge
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, max: int=300, shape_alpha: float=1e-06, scale_alpha: float=1e-06, 
		shape_lambda: float=1e-06, scale_lambda: float=1e-06, ) -> None:
		"""

        Purpose:
        -----------
                Initializes the BayesianRidgeRegressor.

        """
		super( ).__init__( )
		self.max_iter = max
		self.shape_alpha = shape_alpha
		self.scale_alpha = scale_alpha
		self.shape_lambda = shape_lambda
		self.scale_lambda = scale_lambda
		self.model = skl.BayesianRidge( alpha_1=self.shape_alpha,
			alpha_2=self.scale_alpha, lambda_1=self.shape_lambda, lambda_2=self.scale_lambda, )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""

        Purpose:
        -------
        Provides a list of strings representing class members

        """
		return [  'model', 'prediction', 'accuracy', 'shape_alpha', 'scale_alpha', 'shape_lambda',
			'random_state', 'scale_lambda', 'max_iter', 'mean_absolute_error', 
			'mean_squared_error', 'r_mean_squared_error', 'r2_score', 'explained_variance_score', 
			'median_absolute_error', 'train', 'project', 'score', 'analyze', 'plot_scatter', ]
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BayesianRidge | None:
		"""

        Purpose:
        -----------
                Fit the Bayesian RidgeRegressor
                regression linerar_model.

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
			exception.cause = 'BayesianRidge'
			exception.method = "train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline"
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

        Purpose:
        -----------
        Predicts target target_names using the Bayesian linerar_model.

        Parameters:
        -----------
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

        Returns:
        -----------
                np.ndarray: Predicted target_names.

        """
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = "project( self, X: np.ndarray ) -> np.ndarray"
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
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        Returns:
        -----------
                float: R^2 accuracy.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		'''
	
	        Purpose:
	        -----------
	        Evaluate the Bayesian model with regression metrics.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        -----------
	                dict: Dictionary of evaluation metrics.

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

        Purpose:
        -----------
        Plot predicted vs. actual target_names.

        Parameters:
        -----------
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Bayesian-Ridge Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )

class GradientDescent( Regression ):
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
    with more than 10^5 training examples and more than 10^5 feature_names.

    The regularizer is a penalty added to the loss function that shrinks model parameters
    towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1
    or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value
    because of the regularizer, the update is truncated to 0.0 to allow for learning sparse
    models and achieve online feature selection.

    This implementation works with stores represented as dense numpy arrays of floating point
    values for the feature_names.

    """	
	model = skl.SGDRegressor
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, loss: str='squared_loss', iters: int=1000, penalty: str='l2', 
		alpha: float=0.0001, rando: int=42, ) -> None:
		'''
	
	        Purpose:
	        -----------
	        Initialize the SGDRegressor
	
	        Parameters:
	        -----------
	        - alpha (float): Regulation
	        - reg (str): Regularization term. Default is 'l2'.
	        - max (int): Maximum number of passes. Default is 1000.

        '''
		super( ).__init__( )
		self.loss = loss
		self.max_iter = iters
		self.alpha = alpha
		self.random_state = rando
		self.penalty = penalty
		self.model = skl.SGDRegressor( loss=self.loss, max_iter=self.max_iter,
			penalty=self.penalty )
		self.prediction = None
		self.accuracy = 0.0
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""

        Purpose:
        -------
        Provides a list of strings representing class members

        """
		return [ 'model', 'prediction', 'accuracy', 'penalty', 'max_iter', 'random_state', 'loss',
			'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error', 'r2_score',
			'explained_variance_score', 'median_absolute_error', 'train', 'project', 'score', 
			'analyze', 'create_scatter', 'weights', 'labels' ]
	
	@property
	def weights( self ) -> np.ndarray | None:
		if self.model.coef_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
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
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.model.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientDescent | None:
		"""

        Purpose:
        -----------
        Fit the SGD regressor linerar_model.

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
			exception.cause = 'GradientDescent'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

        Purpose:
        -----------
        Predict target_names using the SGD regressor linerar_model.

        Parameters:
        -----------
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

        Returns:
        -----------
                np.ndarray: Predicted target_names.

        """
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		'''

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

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		'''

	        Purpose:
	        -----------
	        Evaluate regression model performance.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        -----------
	        dict: Evaluation metrics dictionary.

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		'''

	        Purpose:
	        -----------
	        Plot predicted vs. actual target_names.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Gradient Descent: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )

class NearestNeighbor( Regression ):
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
	
	model: skn.KNeighborsRegressor
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, num: int=5, algo: str='auto', power: float=2.0, metric: str='minkowski' ) -> None:
		"""

	
	        Purpose:
	        -----------
	        Initialize the linerar_model (KNeighborsRegressor): Internal non-parametric regressor.
	
	        Parameters:
	        -----------
	        num: Number of neighbors to use. Default is 5.

        """
		super( ).__init__( )
		self.n_neighbors = num
		self.algorithm = algo
		self.power = power
		self.metric = metric
		self.model = skn.KNeighborsRegressor( n_neighbors=self.n_neighbors,
			algorithm=self.algorithm, p=self.power, metric=self.metric, )
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
		return [ 'prediction', 'accuracy', 'algorithm', 'n_neighbors', 'random_state', 'power', 
			'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error', 'r2_score', 
			'explained_variance_score', 'median_absolute_error', 'train', 'project', 
			'score', 'analyze', 'create_scatter', 'labels' ]
	
	@property
	def labels( self ) -> np.ndarray | None:
		if self.model.classes_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
		else:
			return self.model.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> NearestNeighbor | None:
		"""


        Purpose:
        -----------
        Fit the KNN regressor linerar_model.

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
			exception.cause = 'NearestNeighbor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		'''

	        Purpose:
	        -----------
	        Predict target_names using the KNN regressor.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	
	        Returns:
	        -----------
	        np.ndarray: Predicted target_names.

        '''
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = "project( self, X: np.ndarray ) -> np.ndarray"
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

	        Purpose:
	        -----------
	        Compute R^2 accuracy for k-NN regressor.
	
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
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		'''
	
	        Purpose:
	        -----------
	        Evaluate k-NN regression performance with multiple metrics.
	
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        -----------
	        dict: Dictionary of evaluation scores.

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
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
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Nearest-Neighbor Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )

class DecisionTree( Regression ):
	'''

	    Purpose:
	    --------
	    Decision Trees (DTs) are a non-parametric supervised learning method used for
	    regression. The goal is to create a model that predicts the value of a
	    target variable by learning simple decision rules inferred from the stores feature_names.
	
	    A tree can be seen as a piecewise constant approximation. Decision trees learn from stores
	    to approximate a sine curve with a set of if-then-else decision rules.
	    The deeper the tree, the more complex the decision rules and the fitter the model.

    '''
	
	model: skd.DecisionTreeRegressor
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, criterion='squared_error', splitter='best', depth=3, rando: int=42 ) -> None:
		'''


        Purpose:
        -----------
        Initialize the KNeighborsClassifier linerar_model.

        '''
		super( ).__init__( )
		self.criterion = criterion
		self.splitter = splitter
		self.max_depth = depth
		self.random_state = rando
		self.model = skd.DecisionTreeRegressor( criterion=self.criterion,
			splitter=self.splitter, max_depth=self.max_depth, random_state=self.random_state, )
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
		return [ 'prediction', 'accuracy', 'criterion', 'splitter', 'random_state', 'max_depth',
			'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error', 'model',
			'r2_score', 'explained_variance_score', 'median_absolute_error', 'train',
			'project', 'score', 'analyze', 'create_scatter', ]
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> DecisionTree | None:
		'''
	
	
	        Purpose:
	        -----------
	        Fit the Decision-Tree regressor linerar_model.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        --------
	        self

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		'''
	
	        Purpose:
	        -----------
	        Predict target_names using the KNN regressor.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	
	        Returns:
	        -----------
	        np.ndarray: Predicted target_names.

        '''
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		'''
	
	        Purpose:
	        -----------
	        Compute R^2 accuracy for k-NN regressor.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        -----------
	        float: R-squared accuracy.

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		'''
	
	        Purpose:
	        -----------
	        Evaluate k-NN regression performance with multiple metrics.
	
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        -----------
	        dict: Dictionary of evaluation scores.

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		'''
	
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

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Decision Tree Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )

class RandomForest( Regression ):
	'''

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
	    some errors can cancel out. Random forests achieve a reduced variance
	    by combining diverse trees, sometimes at the cost of a slight increase in bias.
	    The variance reduction is often significant hence yielding an overall better model.

    '''
	
	model: ske.RandomForestRegressor
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, est: int=10, crit: str='gini', max: int=3, rando: int=42 ) -> None:
		'''
	
	        Purpose:
	        -----------
	        Initialize the RidgeRegressor linerar_model.
	
	        Parameters:
	        -----------
	        alpha (float): Regularization strength. Default is 1.0.
	        solver (str): Solver to use. Default is 'gini'.
	        max (int): maximum iterations
	        rando (int): random seed value

        '''
		super( ).__init__( )
		self.n_estimators = est
		self.criterion = crit
		self.max_depth = max
		self.random_state = rando
		self.model = ske.RandomForestRegressor( n_estimators=self.n_estimators,
			criterion=self.criterion, random_state=self.random_state, )
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
		return [ 'model',  'prediction', 'accuracy', 'criterion', 'n_estimators', 'random_state',
			'loss', 'max_depth', 'mean_absolute_error', 'mean_squared_error',
			'r_mean_squared_error', 'r2_score', 'explained_variance_score',
			'median_absolute_error', 'train', 'project', 'score', 'analyze', 'create_scatter', ]
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> RandomForest | None:
		'''
	
	        Purpose:
	        -----------
	        Fit the RidgeRegressor regression linerar_model.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        --------
	        self

        '''
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
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

        Purpose:
        -----------
        Project target target_names using the RidgeRegressor linerar_model.

        Parameters:
        -----------
                X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

        Returns:
        -----------
                np.ndarray: Predicted target target_names.

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
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
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
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        Returns:
        -----------
                dict: Evaluation metrics including MAE, RMSE, R², etc.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
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
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Random Forest Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )

class GradientBoost( Regression ):
	"""

    Purpose:
    --------
    Gradient Boosting builds an additive model in a forward stage-wise fashion;
    it allows for the optimization  of arbitrary differentiable loss functions.
    In each stage n_classes_ regression trees are  fit on the negative gradient of the binomial
    or multinomial deviance loss function. Binary classification is a special case where
    only a single regression tree is induced.

    """
	
	model: ske.GradientBoostingRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	loss: Optional[ str ]
	learning_rate: Optional[ float ]
	random_state: Optional[ int ]
	n_estimators: Optional[ int ]
	max_detpth: Optional[ int ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, lss: str='deviance', rate: float=0.1, est: int=100, max: int=3, rando: int=42, ) -> None:
		"""

        Purpose:
        _______
                Initialize the GradientBoostingRegressor.

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
		self.model = ske.GradientBoostingRegressor( loss=self.loss,
			learning_rate=self.learning_rate, n_estimators=self.n_estimators, 
			max_depth=self.max_depth, random_state=self.random_state, )
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
		return [ 'model', 'prediction', 'accuracy', 'learning_rate', 'n_estimators', 'random_state',
			'loss', 'max_depth', 'mean_absolute_error', 'mean_squared_error', 
			'r_mean_squared_error', 'r2_score', 'explained_variance_score', 
			'median_absolute_error', 'train', 'project', 'score', 'analyze', 'create_scatter', ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.model.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoost | None:
		"""

        Purpose:
        _______
                Fit the gradient boosting model.

        Parameters:
        ___________
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
			exception.cause = 'GradientBoost'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

        Purpose:
        _________
        Predict regression targets.

        Parameters:
        _________
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

        Returns:
        ________
        np.ndarray: Predicted target values.

        """
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		'''

        Purpose:
        ________
        Compute the coefficient of determination R².

        Parameters:
        ___________
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        Returns:
        _______
        float: R² accuracy.

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

        Purpose:
        _______
        Evaluate performance using standard regression metrics.

        Parameters:
        ___________
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        Returns:
        ________
        Dict[str, float]: Evaluation metrics.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

        Purpose:
        ________
        Plot predicted vs actual target values.

        Parameters:
        ___________
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction, alpha=0.6 )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Gradient-Boosting Regression: Observed vs Projected' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )

class AdaptiveBoost( Regression ):
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
	
	model: ske.AdaBoostRegressor
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, est: int=50, rando: int=42, loss: str='linear', learning: float=1.0, ) -> None:
		"""

        Purpose:
        --------
        Initialize the Ridge Regression Classifier.


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
		self.model = ske.AdaBoostRegressor( n_estimators=self.n_estimators,
			random_state=self.random_state, loss=self.loss, learning_rate=self.learning_rate, )
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
		return [ 'model', 'prediction', 'kernel', 'accuracy', 'n_estimators', 'random_state', 'loss',
			'learning_rate', 'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error', 
			'r2_score', 'explained_variance_score', 'median_absolute_error', 'train', 
			'project', 'score', 'analyze', 'create_scatter', ]
	
	@property
	def errors( self ) -> np.ndarray | None:
		if self.model.estimator_errors_ is None:
			raise AttributeError( 'The model errors have not been initialized!' )
		else:
			return self.model.estimator_errors_
	
	@property
	def weights( self ) -> np.ndarray | None:
		if self.model.estimator_weights_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
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
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.model.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> AdaptiveBoost | None:
		"""

        Purpose:
        --------
                Fit the RidgeRegressor regression linerar_model.

        Parameters:
        ___________
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
			exception.cause = 'GradientBoost'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		'''

	        Project target target_names
	        using the RidgeRegressor linerar_model.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	
	        Returns:
	        -----------
	        np.ndarray: Predicted target target_names.

        '''
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

        Purpose:
        --------
        Compute the R-squared
        accuracy for the Ridge model.

        Parameters:
        ----------
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        Returns:
        --------
                float: R-squared accuracy.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

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
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		'''

	        Plot predicted vs
	        actual target_names.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        -----------
	        None

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'ADA Boost: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )

class BaggingModel( Regression ):
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
	
	model: ske.BaggingRegressor
	base_estimator: object
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
		self.bagging_regressor = ske.BaggingRegressor( estimator=self.base_estimator, 
			max_features=self.max_features, random_state=self.random_state, )
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
		return [ 'model', 'prediction', 'base_estimator', 'n_estimators', 'max_features',
			'accuracy', 'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error', 
			'r2_score', 'explained_variance_score', 'median_absolute_error', 'train', 
			'project', 'score', 'analyze', 'create_scatter', 'random_state', 'labels' ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.bagging_regressor.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.bagging_regressor.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BaggingModel | None:
		"""

        Purpose:
        --------
        Fit the RidgeRegressor regression linerar_model.

        Parameters:
        ___________
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        Returns:
        --------
        self

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.bagging_regressor.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

        Project target target_names
        using the RidgeRegressor linerar_model.

        Parameters:
        -----------
                X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

        Returns:
        -----------
                np.ndarray: Predicted target target_names.

        """
		try:
			throw_if( 'X', X )
			self.prediction = self.bagging_regressor.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

        Compute the R-squared
        accuracy for the Ridge model.

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
			self.prediction = self.bagging_regressor.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

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
			self.prediction = self.bagging_regressor.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

        Plot predicted vs
        actual target_names.

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
			self.prediction = self.bagging_regressor.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Bagging Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )

class VotingModel( Regression ):
	"""

    Purpose:
    --------

    Prediction voting regressor for unfitted estimators. A voting regressor is an ensemble
    meta-estimator that fits several base regressors, each on the whole dataset.
    Then it averages the individual predictions to form a final prediction.

    """
	
	model: ske.VotingRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	estimators: List[ (str, object) ]
	
	def __init__( self, est: List[ (str, object) ] ) -> None:
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
		self.estimators = est
		self.model = ske.VotingRegressor( estimators=self.estimators )
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
		return [ 'model', 'prediction', 'kernel', 'C', 'epsilon', 'accuracy', 'mean_absolute_error',
			'mean_squared_error', 'r_mean_squared_error', 'r2_score', 'explained_variance_score', 
			'median_absolute_error', 'train', 'project', 'score', 'analyze', 'create_scatter',
		    'labels' ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.model.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> VotingModel | None:
		'''
	
	        Purpose:
	        --------
	        Fit the RidgeRegressor regression linerar_model.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        --------
	        self

        '''
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
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		'''

	        Purpose:
	        ---------
	        Project target target_names
	        using the RidgeRegressor linerar_model.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	
	        Returns:
	        -------
	        np.ndarray: Predicted target target_names.

        '''
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		'''
	
	        Compute the R-squared
	        accuracy for the Ridge model.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        -----------
                float: R-squared accuracy.

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

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
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

        Plot predicted vs
        actual target_names.

        Parameters:
        -----------
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        Returns:
        -----------
                None

        """
		try:
			if X is None:
				raise Exception( "The argument 'X' is required!" )
			elif y is None:
				raise Exception( "The argument 'y' is required!" )
			else:
				throw_if( 'X', X )
				throw_if( 'y', y )
				self.prediction = self.model.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Voting Regression: Observed vs Projected' )
				plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = ("create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None")
			error = ErrorDialog( exception )
			error.show( )

class StackingModel( Regression ):
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
	model: ske.StackingRegressor
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, est: List[
		Tuple[ str, ClassifierMixin ] ], final: ClassifierMixin = None ) -> None:
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
		self.model = ske.StackingRegressor( estimators=self.estimators,
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
		"""
	
	        Purpose:
	        -------
	        Provides a list of strings representing class members

        """
		return [ 'model', 'prediction', 'estimators', 'final_estimator', 'accuracy',
			'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error', 'r2_score', 
			'explained_variance_score', 'median_absolute_error', 'train', 'project', 
			'score', 'analyze', 'create_scatter', 'labels' ]
	
	@property
	def labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			classes_ ndarray of shape (n_classes, )
			A list of class labels known to the classifier.

		'''
		if self.model.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized!' )
		else:
			return self.model.classes_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> StackingModel | None:
		"""

	        Purpose:
	        ---------
	        Fit the RidgeRegressor regression linerar_model.
	
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
			exception.cause = 'StackingModel'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
	
	        Project target target_names
	        using the RidgeRegressor linerar_model.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	
	        Returns:
	        -----------
	        np.ndarray: Predicted target target_names.

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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""
	
	        Compute the R-squared
	        accuracy for the Ridge model.
	
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
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

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
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

        Purpose:
        ---------
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
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Stacking Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )

class SupportVector( Regression ):
	"""
    Wrapper for sklearn's Support Vector Regression (SVR).
    """
	
	model: skv.SVR
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	kernel: str
	regularization: float
	epsilon: float
	
	def __init__( self, kernel: str='rbf', C: float=1.0, epsilon: float=0.1 ) -> None:
		'''

	        Purpose:
	        ---------
	        Initialize the SVR model.
	
	        :param kernel: Kernel type to be used in the algorithm.
	        :type kernel: str
	        :param C: Regularization parameter.
	        :type C: float
	        :param epsilon: Epsilon in the epsilon-SVR model.
	        :type epsilon: float

        '''
		super( ).__init__( )
		self.kernel = kernel
		self.regularization = C
		self.epsilon = epsilon
		self.model = skv.SVR( kernel=self.kernel, C=self.regulation, epsilon=self.epsilon )
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
		return [ 'model', 'prediction', 'kernel', 'regulation', 'epsilon', 'accuracy',
			'mean_absolute_error', 'mean_squared_error', 'r_mean_squared_error', 'r2_score', 
			'explained_variance_score', 'median_absolute_error', 'train', 'project', 
			'score', 'analyze', 'create_scatter', ]
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> SupportVector | None:
		"""

        Purpose:
        --------
        Fit the SVR model to the stores.

        Parameters:
        ___________
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
			exception.cause = 'SupportVector'
			exception.method = 'project( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray | None:
		"""

        Purpose:
        --------
        Predict target values for the input feature_names.

        Parameters:
        ___________
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return self.model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'project( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		'''
	
	        Purpose:
	        --------
	
	        Compute the R-squared
	        accuracy for the Ridge model.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).
	
	        Returns:
	        -----------
	        float: R-squared accuracy.

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

        Purpose:
        ---------
        Print detailed regression metrics.

        Parameters:
        ___________
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

        Purpose:
        --------
        Visualize the true vs predicted values for regression.

        Parameters:
        ___________
        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
        y (np.ndarray): True class target vector of shape ( n_samples, ).

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction, color="blue", edgecolor="k" )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.xlabel( 'True Values' )
			plt.ylabel( 'Predicted Values' )
			plt.title( 'SVR: True vs Predicted' )
			plt.grid( True )
			plt.tight_layout( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )

class MultiLayerPerceptron( Regression ):
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
	model: skn.MLPRegressor
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, hidden: tuple = (100,), activ='relu', solver='adam', alpha=0.0001,
			learning: str='constant', rando: int=42, ) -> None:
		super( ).__init__( )
		self.hidden_layers = hidden
		self.activation_function = activ
		self.learning = learning
		self.solver = solver
		self.alpha = alpha
		self.random_state = rando
		self.model = skn.MLPRegressor( hidden_layer_sizes=self.hidden_layers,
			activation=self.activation_function, solver=self.solver, alpha=self.alpha,
			learning_rate=self.learning, random_state=self.random_state, )
		self.prediction = None
	
	def __dir__( self ) -> List[ str ]:
		"""

	        Purpose:
	        -------
	        Provides a list of strings representing class members

        """
		return [ 'prediction',
		         'model',
		         'accuracy',
		         'learning',
		         'activation_function',
		         'hidden_layers',
		         'random_state',
		         'alpha',
		         'max_depth',
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
		         'create_scatter',
		         'loss',
		         'classes',
		         'weights' ]
	
	@property
	def loss( self ) -> float:
		if self.model.loss_ is None:
			raise AttributeError( 'The model loss has not been initialized!' )
		else:
			return self.model.loss_
	
	@property
	def classes( self ) -> np.ndarray:
		if self.model.classes_ is None:
			raise AttributeError( 'The model labels have not been initialized' )
		else:
			return self.model.classes_
	
	@property
	def weights( self ) -> np.ndarray:
		if self.model.coefs_ is None:
			raise AttributeError( 'The model weights have not been initialized!' )
		else:
			return self.model.coefs_
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> MultiLayerPerceptron | None:
		"""

	        Purpose:
	        -----------
	        Fits all pipeline steps to the text df.

	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.

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
			exception.cause = ""
			exception.method = (
					'train( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> Pipeline')
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
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ""
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
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.

	        Returns:
	        -----------
	        float: R-squared accuracy.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			self.accuracy = r2_score( y, self.prediction )
			return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ""
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
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.

	        Returns:
	        -----------
	        dict: Dictionary of MAE, MSE, RMSE, R², etc.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ""
			exception.method = "analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict"
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

	        Purpose:
	        -----------
	        Plot actual vs predicted target_names.

	        Parameters:
	        -----------
	        X ( n_samples, n_features ): np.ndarray - feature matrix.
	        y ( n_samples, ): np.ndarray - target vector.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( "Observed" )
			plt.ylabel( "Projected" )
			plt.title( "MultiLayerPerceptron: Observed vs Projected" )
			plt.plot( [ X.min( ),  X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ""
			exception.method = ("create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None")
			error = ErrorDialog( exception )
			error.show( )

class GaussianProcess( Regression ):
	'''

	    Purpose:
	    --------
	    Wraps sklearn's GaussianProcessRegressor to provide a clean interface
	    for model training, prediction, and performance evaluation.

    '''
	
	model: gpr.GaussianProcessRegressor
	alpha: Optional[ float ]
	normalize: Optional[ bool ]
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	median_absolute_error: Optional[ float ]
	explained_variance_score: Optional[ float ]
	r2_score: Optional[ float ]
	alpha: Optional[ float ]
	
	def __init__( self, alpha: float=1e-10, normalize_y: bool=True ) -> None:
		"""

        Purpose:
        --------
        Initializes the Gaussian Process Regressor with a default RBF kernel.

        Parameters:
        -----------
        kernel (sklearn.gaussian_process.kernels.Kernel): Kernel to use.
        alpha (float): Value added to the diagonal of the kernel matrix.
        normalize_y (bool): Whether to normalize the target values.

        Returns:
        --------
        None

        """
		super( ).__init__( )
		self.normalize = normalize_y
		self.alpha = alpha
		self.kernel = C( 1.0, (1e-3, 1e3) ) * RBF( 1.0, (1e-2, 1e2) )
		self.model = GaussianProcess( kernel=self.kernel, alpha=alpha, normalize_y=normalize_y )
		
		def __dir__( self ) -> List[ str ]:
			"""

		        Purpose:
		        -------
		        Provides a list of strings representing class members

	        """
			return [ 'prediction',
			         'model',
			         'accuracy',
			         'alpha',
			         'normalize',
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
			         'create_scatter', ]
		
	def train( self, X: np.ndarray, y: np.ndarray ) -> GaussianProcess | None:
		"""

        Purpose:
        --------
        Fit the Gaussian Process Regressor to the training data.

        Parameters:
        -----------
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target vector of shape (n_samples,).

        Returns:
        --------
        self

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.train( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'train'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		'''

	        Purpose:
	        --------
	        Predict using the trained model.
	
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
	
	        Returns:
	        --------
	        np.ndarray: Predicted values.

        '''
		try:
			throw_if( 'X', X )
			self.prediction = self.model.project( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = "project"
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | None:
		"""

        Purpose:
        --------
        Compute R² score for the model on the test data.

        Parameters:
        -----------
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): True target values.

        Returns:
        --------
        float: R² score.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return self.model.score( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'score'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

        Purpose:
        --------
        Compute regression metrics: MSE, RMSE, MAE, R², Median AE, Explained Variance.

        Parameters:
        -----------
        X (np.ndarray): Feature matrix.
        y (np.ndarray): True target values.

        Returns:
        --------
        Dict[str, float]: Dictionary of metrics.

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.project( X )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			return \
			{
				'MSE': self.mean_squared_error,
				'RMSE': self.r_mean_squared_error,
				'R2': self.r2_score,
				'VAR': self.explained_variance_score,
				'MAE': self.mean_absolute_error,
				'MAX': self.max_error
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'analyze'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None:
		'''

	        Purpose:
	        -----------
	        Plot predicted vs. actual target_names.

	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True class target vector of shape ( n_samples, ).

        '''
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			plt.scatter( y, self.prediction )
			plt.xlabel( 'Observed' )
			plt.ylabel( 'Projected' )
			plt.title( 'Guassian Process Regression: Observed vs Projected' )
			plt.plot( [ X.min( ),X.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
			plt.grid( True )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = ('create_scatter( self, X: np.ndarray, y: np.ndarray ) -> None')
			error = ErrorDialog( exception )
			error.show( )