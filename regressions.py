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
from boogr import Error
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
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error,
                             explained_variance_score, median_absolute_error, max_error,
                             accuracy_score, )


def throw_if( name: str, value: object ):
	if value is None:
		raise Exception( f'Argument "{name}" cannot be empty!' )


class Regression( ):
	"""

    Purpose:
    ---------
    Abstract base class that defines the interface for all linerar_model wrappers.

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
		"""
		
	        Purpose:
	        --------
	        Initialize the common regression state shared by derived wrappers.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        None
		
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
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		raise NotImplementedError
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""
		
	        Purpose:
	        --------
	        Fit the regression model to the training data.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        object | None:
	            The trained wrapper instance or another implementation-specific result.
		
        """
		raise NotImplementedError
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Generate regression predictions using the trained model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None:
	            Predicted continuous target values.
		
        """
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted regressor using its primary scoring behavior.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        float | pd.DataFrame | None:
	            A primary score or implementation-specific scoring summary.
		
        """
		raise NotImplementedError
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        Dict[str, float] | pd.DataFrame | None:
	            A metrics summary containing regression evaluation results.
		
        """
		raise NotImplementedError


class LeastSquares( Regression ):
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
		"""
		
	        Purpose:
	        --------
	        Initialize the Ordinary Least Squares regression wrapper.
		
	        Parameters:
	        -----------
	        fit (bool): Specifies whether to calculate the intercept for the model.
	        copy (bool): Specifies whether to copy the input feature matrix.
	        tol (float): Solver precision used by sparse least-squares routines.
	        jobs (Optional[int]): Number of parallel jobs used when supported by sklearn.
	        positive (bool): Specifies whether coefficients are constrained to be positive.
		
	        Returns:
	        --------
	        None
		
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the intercept term learned by the linear model.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | float | None: The learned intercept term.
		
        """
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Return the learned regression coefficients.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | None: The learned coefficient vector or matrix.
		
        """
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.coef_
	
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
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LeastSquares | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Ordinary Least Squares regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        LeastSquares | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> LeastSquares'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained OLS model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
        """
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute regression scores for the fitted OLS model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing score values.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted OLS model using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        None
		
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
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
	
		
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
		"""
		
	        Purpose:
	        --------
	        Initialize the Ridge regression wrapper.
		
	        Parameters:
	        -----------
	        alpha (float): Regularization strength.
	        fit (bool): Specifies whether to calculate the intercept for the model.
	        copy (bool): Specifies whether to copy the input feature matrix.
	        iters (Optional[int]): Maximum number of iterations for iterative solvers.
	        tol (float): Precision of the solution for iterative solvers.
	        solver (str): Solver used by the Ridge estimator.
	        positive (bool): Specifies whether to constrain coefficients to be positive.
	        rando (Optional[int]): Random seed used by supported stochastic solvers.
		
	        Returns:
	        --------
	        None
		
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the intercept term learned by the Ridge model.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | float | None: The learned intercept term.
		
        """
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Return the learned Ridge regression coefficients.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | None: The learned coefficient vector or matrix.
		
        """
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
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
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
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
	        Fit the Ridge regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        Ridge | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Ridge'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Ridge model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted Ridge model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Ridge model using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        None
		
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
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			
			
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
			select: str = 'cyclic' ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Lasso regression wrapper.
		
	        Parameters:
	        -----------
	        alpha (float): Regularization strength for the L1 penalty.
	        fit (bool): Specifies whether to calculate the intercept for the model.
	        precompute (bool): Specifies whether to use a precomputed Gram matrix.
	        copy (bool): Specifies whether to copy the input feature matrix.
	        iters (int): Maximum number of iterations.
	        tol (float): Tolerance used by the coordinate descent optimizer.
	        warm (bool): Specifies whether to reuse the previous solution as initialization.
	        positive (bool): Specifies whether coefficients are constrained to be positive.
	        rando (Optional[int]): Random seed used when selection is random.
	        select (str): Coefficient update strategy used by the optimizer.
		
	        Returns:
	        --------
	        None
		
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
		self.selection = select
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the intercept term learned by the Lasso model.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | float | None: The learned intercept term.
		
        """
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Return the learned Lasso regression coefficients.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | None: The learned coefficient vector or matrix.
		
        """
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
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
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Lasso | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Lasso regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        Lasso | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Lasso'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Lasso model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
        """
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted Lasso model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Lasso model using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'Lasso'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			
			
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
		"""
		
	        Purpose:
	        --------
	        Initialize the Elastic Net regression wrapper.
		
	        Parameters:
	        -----------
	        alpha (float): Overall regularization strength.
	        ratio (float): Mixing parameter between L1 and L2 penalties.
	        fit (bool): Specifies whether to calculate the intercept for the model.
	        precompute (bool): Specifies whether to use a precomputed Gram matrix.
	        iters (int): Maximum number of iterations.
	        copy (bool): Specifies whether to copy the input feature matrix.
	        tol (float): Tolerance used by the coordinate descent optimizer.
	        warm (bool): Specifies whether to reuse the previous solution as initialization.
	        positive (bool): Specifies whether coefficients are constrained to be positive.
	        rando (Optional[int]): Random seed used when selection is random.
	        select (str): Coefficient update strategy used by the optimizer.
		
	        Returns:
	        --------
	        None
		
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the intercept term learned by the Elastic Net model.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | float | None: The learned intercept term.
		
        """
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Return the learned Elastic Net regression coefficients.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | None: The learned coefficient vector or matrix.
		
        """
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
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
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> ElasticNet | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Elastic Net regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        ElasticNet | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> ElasticNet'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Elastic Net model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted Elastic Net model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Elastic Net model using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'ElasticNet'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			

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
	
	def __init__( self, coeffs: int = 500, fit: bool=True, precompute: bool=True ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Least Angle Regression wrapper.
		
	        Parameters:
	        -----------
	        coeffs (int): Maximum number of non-zero coefficients.
	        fit (bool): Specifies whether to calculate the intercept.
	        precompute (bool): Specifies whether to use a precomputed Gram matrix.
		
	        Returns:
	        --------
	        None
		
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
			size: float=0.2, random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
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
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LeastAngle | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Least Angle Regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        LeastAngle | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> LeastAngle'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Least Angle Regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted LARS model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted LARS model using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			

class BayesianRidge( Regression ):
	"""
	
	    Purpose:
	    --------
	    Bayesian regression techniques can be used to include alpha parameters in the
	    estimation procedure: the alpha parameter is not set in a hard sense
	    but tuned to the df at hand. This can be done by introducing uninformative priors over
	    the hyperparameters of the model. The alpha used in Ridge regression and
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
		"""
		
	        Purpose:
	        --------
	        Initialize the Bayesian Ridge regression wrapper.
		
	        Parameters:
	        -----------
	        max (int): Maximum number of iterations.
	        shape_alpha (float): Shape parameter for the alpha Gamma prior.
	        scale_alpha (float): Rate parameter for the alpha Gamma prior.
	        shape_lambda (float): Shape parameter for the lambda Gamma prior.
	        scale_lambda (float): Rate parameter for the lambda Gamma prior.
	        tol (float): Tolerance used by the optimizer.
	        alpha_init (Optional[float]): Initial value for alpha.
	        lambda_init (Optional[float]): Initial value for lambda.
	        compute_score (bool): Specifies whether to compute the log marginal likelihood.
	        fit (bool): Specifies whether to calculate the intercept for the model.
	        copy (bool): Specifies whether to copy the input feature matrix.
	        verbose (bool): Specifies whether to emit solver progress messages.
		
	        Returns:
	        --------
	        None
		
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the intercept term learned by the Bayesian Ridge model.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | float | None: The learned intercept term.
		
        """
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Return the learned Bayesian Ridge regression coefficients.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | None: The learned coefficient vector or matrix.
		
        """
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
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
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BayesianRidge | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Bayesian Ridge regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        BayesianRidge | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> BayesianRidge'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Bayesian Ridge model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
        """
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted Bayesian Ridge model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Bayesian Ridge model using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.cause = 'BayesianRidge'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'BayesianRidge'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
	
			
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
		"""
		
	        Purpose:
	        --------
	        Initialize the stochastic gradient descent regression wrapper.
		
	        Parameters:
	        -----------
	        loss (str): Loss function optimized by SGD.
	        iters (int): Maximum number of passes over the training data.
	        penalty (str): Regularization penalty applied to the coefficients.
	        alpha (float): Regularization strength.
	        rando (Optional[int]): Random seed used by supported routines.
	        learning_rate (str): Learning-rate schedule used by the optimizer.
	        l1_ratio (float): Elastic Net mixing parameter when penalty is elasticnet.
	        fit (bool): Specifies whether to calculate the intercept for the model.
	        tol (float): Stopping tolerance.
	        shuffle (bool): Specifies whether to shuffle training data after each epoch.
	        verbose (int): Verbosity level for the estimator.
	        epsilon (float): Epsilon used by Huber and epsilon-insensitive losses.
	        eta0 (float): Initial learning rate for schedules that use it.
	        power_t (float): Exponent used by the invscaling learning-rate schedule.
	        early_stopping (bool): Specifies whether to use early stopping.
	        validation_fraction (float): Proportion of training data used for validation.
	        n_iter_no_change (int): Early-stopping patience in epochs.
	        warm (bool): Specifies whether to reuse the previous solution as initialization.
	        average (bool): Specifies whether to use averaged SGD weights.
		
	        Returns:
	        --------
	        None
		
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
		self.model = skl.SGDRegressor(
			loss=self.loss,
			penalty=self.penalty,
			alpha=self.alpha,
			l1_ratio=self.l1_ratio,
			fit_intercept=self.fit_intercept,
			max_iter=self.max_iter,
			tol=self.tol,
			shuffle=self.shuffle,
			verbose=self.verbose,
			epsilon=self.epsilon,
			random_state=self.random_state,
			learning_rate=self.learning_rate,
			eta0=self.eta0,
			power_t=self.power_t,
			early_stopping=self.early_stopping,
			validation_fraction=self.validation_fraction,
			n_iter_no_change=self.n_iter_no_change,
			warm_start=self.warm_start,
			average=self.average
		)
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the learned SGD regression coefficients.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | None: The learned coefficient vector.
		
        """
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""
		
	        Purpose:
	        --------
	        Return the intercept term learned by the SGD regressor.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | float | None: The learned intercept term.
		
        """
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
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
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def iterations( self ) -> int | None:
		"""
		
	        Purpose:
	        --------
	        Return the number of weight updates performed by the estimator.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        int | None: The number of effective iterations completed by the optimizer.
		
        """
		if not hasattr( self.model, 't_' ):
			raise AttributeError( 'The model has not been initialized!' )
		return self.model.t_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientDescent | None:
		"""
		
	        Purpose:
	        --------
	        Fit the stochastic gradient descent regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        GradientDescent | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> GradientDescent'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained SGD regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted SGD model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.cause = 'GradientDescent'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted SGD model using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.cause = 'GradientDescent'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'GradientDescent'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			

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
		"""
		
	        Purpose:
	        --------
	        Initialize the K-Nearest Neighbors regression wrapper.
		
	        Parameters:
	        -----------
	        num (int): Number of neighbors used during prediction.
	        weight (str): Weighting strategy used for neighbor contributions.
	        algo (str): Neighbor-search algorithm.
	        leaf (int): Leaf size passed to tree-based search algorithms.
	        power (float): Power parameter for the Minkowski metric.
	        metric (str): Distance metric used to search for nearest neighbors.
	        metric_params (Optional[Dict[str, object]]): Additional metric keyword arguments.
	        jobs (Optional[int]): Number of parallel jobs used for neighbor searches.
		
	        Returns:
	        --------
	        None
		
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
		self.model = skn.KNeighborsRegressor(
			n_neighbors=self.n_neighbors,
			weights=self.weights,
			algorithm=self.algorithm,
			leaf_size=self.leaf_size,
			p=self.power,
			metric=self.metric,
			metric_params=self.metric_params,
			n_jobs=self.n_jobs
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the number of features seen during training.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> NearestNeighbor | None:
		"""
		
	        Purpose:
	        --------
	        Fit the K-Nearest Neighbors regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        NearestNeighbor | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> NearestNeighbor'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained KNN regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted KNN regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.cause = 'NearestNeighbor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted KNN regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'NearestNeighbor'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			

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
		"""
		
	        Purpose:
	        --------
	        Initialize the Decision Tree regression wrapper.
		
	        Parameters:
	        -----------
	        criterion (str): Function used to measure the quality of a split.
	        splitter (str): Strategy used to choose the split at each node.
	        depth (int): Maximum depth of the decision tree.
	        rando (int): Random seed used by supported routines.
		
	        Returns:
	        --------
	        None
		
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
			size: float=0.2, random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> DecisionTree | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Decision Tree regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        DecisionTree | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> DecisionTree'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Decision Tree regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted Decision Tree regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Decision Tree regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'DecisionTree'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception


class ExtraTreesModel( Regression ):
	"""
	
	    Purpose:
	    --------
	    Extremely Randomized Trees regression wrapper around sklearn.ensemble.ExtraTreesRegressor.
	    This implementation preserves the existing Mathy wrapper contract while aligning the
	    constructor surface to the current scikit-learn API and correcting score and
	    evaluation flow.

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
		"""
		
	        Purpose:
	        --------
	        Initialize the Extremely Randomized Trees regression wrapper.
		
	        Parameters:
	        -----------
	        estimators (int): Number of trees in the ensemble.
	        criterion (str): Function used to measure split quality.
	        depth (Optional[int]): Maximum depth of each tree.
	        split (int | float): Minimum number or proportion of samples required to split a node.
	        leaf (int | float): Minimum number or proportion of samples required at a leaf node.
	        weight_fraction (float): Minimum weighted fraction of samples required at a leaf node.
	        features (int | float | str | None): Number of features considered when looking for the best split.
	        leaf_nodes (Optional[int]): Maximum number of leaf nodes.
	        impurity (float): Minimum impurity decrease required to split a node.
	        bootstrap (bool): Specifies whether bootstrap samples are used when building trees.
	        oob_score (bool): Specifies whether to use out-of-bag samples to estimate generalization.
	        jobs (Optional[int]): Number of parallel jobs used by fit and predict.
	        rando (Optional[int]): Random seed used by the ensemble.
	        verbose (int): Verbosity level for fit and predict.
	        warm (bool): Specifies whether to reuse the previous solution and add more estimators.
	        ccp_alpha (float): Complexity parameter used for Minimal Cost-Complexity Pruning.
	        samples (Optional[int | float]): Number or proportion of samples drawn when bootstrap is True.
	        monotonic (Optional[object]): Monotonicity constraints applied to each feature.
		
	        Returns:
	        --------
	        None
		
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
		self.model = ske.ExtraTreesRegressor(
			n_estimators=self.n_estimators,
			criterion=self.criterion,
			max_depth=self.max_depth,
			min_samples_split=self.min_samples_split,
			min_samples_leaf=self.min_samples_leaf,
			min_weight_fraction_leaf=self.min_weight_fraction_leaf,
			max_features=self.max_features,
			max_leaf_nodes=self.max_leaf_nodes,
			min_impurity_decrease=self.min_impurity_decrease,
			bootstrap=self.bootstrap,
			oob_score=self.oob_score,
			n_jobs=self.n_jobs,
			random_state=self.random_state,
			verbose=self.verbose,
			warm_start=self.warm_start,
			ccp_alpha=self.ccp_alpha,
			max_samples=self.max_samples,
			monotonic_cst=self.monotonic_cst
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the number of features seen during training.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> ExtraTreesModel | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Extremely Randomized Trees regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        ExtraTreesModel | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> ExtraTreesModel'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Extremely Randomized Trees regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
        """
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted Extremely Randomized Trees regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.cause = 'ExtraTreesModel'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Extremely Randomized Trees regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.cause = 'ExtraTreesModel'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'ExtraTreesModel'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
		
		
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
		"""
		
	        Purpose:
	        --------
	        Initialize the Random Forest regression wrapper.
		
	        Parameters:
	        -----------
	        estimators (int): Number of trees in the forest.
	        criterion (str): Function used to measure split quality.
	        depth (Optional[int]): Maximum depth of each tree.
	        split (int | float): Minimum number or proportion of samples required to split a node.
	        leaf (int | float): Minimum number or proportion of samples required at a leaf node.
	        weight_fraction (float): Minimum weighted fraction of samples required at a leaf node.
	        features (int | float | str | None): Number of features considered when looking for the best split.
	        leaf_nodes (Optional[int]): Maximum number of leaf nodes.
	        impurity (float): Minimum impurity decrease required to split a node.
	        bootstrap (bool): Specifies whether bootstrap samples are used when building trees.
	        oob_score (bool): Specifies whether to use out-of-bag samples to estimate generalization.
	        jobs (Optional[int]): Number of parallel jobs used by fit and predict.
	        rando (Optional[int]): Random seed used by the ensemble.
	        verbose (int): Verbosity level for fit and predict.
	        warm (bool): Specifies whether to reuse the previous solution and add more estimators.
	        ccp_alpha (float): Complexity parameter used for Minimal Cost-Complexity Pruning.
	        samples (Optional[int | float]): Number or proportion of samples drawn when bootstrap is True.
	        monotonic (Optional[object]): Monotonicity constraints applied to each feature.
		
	        Returns:
	        --------
	        None
		
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
		self.model = ske.RandomForestRegressor(
			n_estimators=self.n_estimators,
			criterion=self.criterion,
			max_depth=self.max_depth,
			min_samples_split=self.min_samples_split,
			min_samples_leaf=self.min_samples_leaf,
			min_weight_fraction_leaf=self.min_weight_fraction_leaf,
			max_features=self.max_features,
			max_leaf_nodes=self.max_leaf_nodes,
			min_impurity_decrease=self.min_impurity_decrease,
			bootstrap=self.bootstrap,
			oob_score=self.oob_score,
			n_jobs=self.n_jobs,
			random_state=self.random_state,
			verbose=self.verbose,
			warm_start=self.warm_start,
			ccp_alpha=self.ccp_alpha,
			max_samples=self.max_samples,
			monotonic_cst=self.monotonic_cst
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the number of features seen during training.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> RandomForest | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Random Forest regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        RandomForest | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> RandomForest'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Random Forest regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-Squared scores for the fitted Random Forest regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-Squared scores.
		
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
			exception.cause = 'RandomForest'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Random Forest regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.cause = 'RandomForest'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'RandomForest'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			

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
			estimators: int = 100, subsample: float = 1.0,
			criterion: str = 'friedman_mse', split: int | float = 2,
			leaf: int | float = 1, weight_fraction: float = 0.0,
			depth: Optional[ int ] = 3, impurity: float = 0.0,
			init: Optional[ object ] = None, rando: Optional[ int ] = 42,
			features: int | float | str | None = None, alpha: float = 0.9,
			verbose: int = 0, leaf_nodes: Optional[ int ] = None,
			warm: bool = False, validation_fraction: float = 0.1,
			no_change: Optional[ int ] = None, tol: float = 1e-4,
			ccp_alpha: float = 0.0 ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Gradient Boosting regression wrapper.
		
	        Parameters:
	        -----------
	        loss (str): Loss function optimized during boosting.
	        rate (float): Learning rate applied to each boosting stage.
	        estimators (int): Number of boosting stages to perform.
	        subsample (float): Fraction of samples used for fitting each base learner.
	        criterion (str): Function used to measure split quality in the base trees.
	        split (int | float): Minimum number or proportion of samples required to split a node.
	        leaf (int | float): Minimum number or proportion of samples required at a leaf node.
	        weight_fraction (float): Minimum weighted fraction of samples required at a leaf node.
	        depth (Optional[int]): Maximum depth of the individual regression trees.
	        impurity (float): Minimum impurity decrease required to split a node.
	        init (Optional[object]): Initial estimator used to compute the first predictions.
	        rando (Optional[int]): Random seed used by supported routines.
	        features (int | float | str | None): Number of features considered for the best split.
	        alpha (float): Quantile or Huber loss alpha parameter where applicable.
	        verbose (int): Verbosity level for fit and predict.
	        leaf_nodes (Optional[int]): Maximum number of leaf nodes per tree.
	        warm (bool): Specifies whether to reuse the previous solution and add estimators.
	        validation_fraction (float): Proportion of training data used for validation when early stopping.
	        no_change (Optional[int]): Early-stopping patience in iterations.
	        tol (float): Tolerance for early stopping.
	        ccp_alpha (float): Complexity parameter used for Minimal Cost-Complexity Pruning.
		
	        Returns:
	        --------
	        None
		
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
		self.model = ske.GradientBoostingRegressor(
			loss=self.loss,
			learning_rate=self.learning_rate,
			n_estimators=self.n_estimators,
			subsample=self.subsample,
			criterion=self.criterion,
			min_samples_split=self.min_samples_split,
			min_samples_leaf=self.min_samples_leaf,
			min_weight_fraction_leaf=self.min_weight_fraction_leaf,
			max_depth=self.max_depth,
			min_impurity_decrease=self.min_impurity_decrease,
			init=self.init,
			random_state=self.random_state,
			max_features=self.max_features,
			alpha=self.alpha,
			verbose=self.verbose,
			max_leaf_nodes=self.max_leaf_nodes,
			warm_start=self.warm_start,
			validation_fraction=self.validation_fraction,
			n_iter_no_change=self.n_iter_no_change,
			tol=self.tol,
			ccp_alpha=self.ccp_alpha
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the number of features seen during training.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoost | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Gradient Boosting regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        GradientBoost | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoost'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Gradient Boosting regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-Squared scores for the fitted Gradient Boosting regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-Squared scores.
		
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
			exception.cause = 'GradientBoost'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Gradient Boosting regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.cause = 'GradientBoost'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'GradientBoost'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			

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
		"""
		
	        Purpose:
	        --------
	        Initialize the AdaBoost regression wrapper.
		
	        Parameters:
	        -----------
	        estimator (Optional[object]): Base regressor used for each boosting stage.
	        estimators (int): Number of weak learners used in the ensemble.
	        rate (float): Learning rate applied to each boosting stage.
	        loss (str): Loss function used when updating sample weights.
	        rando (Optional[int]): Random seed used by supported routines.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.estimator = estimator
		self.n_estimators = estimators
		self.learning_rate = rate
		self.loss = loss
		self.random_state = rando
		self.model = ske.AdaBoostRegressor(
			estimator=self.estimator,
			n_estimators=self.n_estimators,
			learning_rate=self.learning_rate,
			loss=self.loss,
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the fitted base estimator template used by the AdaBoost ensemble.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        object | None: The fitted base estimator template.
		
        """
		if hasattr( self.model, 'estimator_' ):
			return self.model.estimator_
		raise AttributeError( 'The model has not been trained!' )
	
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
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> AdaptiveBoost | None:
		"""
		
	        Purpose:
	        --------
	        Fit the AdaBoost regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        AdaptiveBoost | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> AdaptiveBoost'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained AdaBoost regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-Squared scores for the fitted AdaBoost regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-Squared scores.
		
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
			exception.cause = 'AdaptiveBoost'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted AdaBoost regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.cause = 'AdaptiveBoost'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'AdaptiveBoost'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			

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
		"""
		
	        Purpose:
	        --------
	        Initialize the bagging regression wrapper.
		
	        Parameters:
	        -----------
	        estimator (Optional[object]): Base regressor fit on each sampled subset.
	        num (int): Number of base estimators in the ensemble.
	        samples (Optional[int | float]): Number or proportion of samples drawn for each estimator.
	        features (int | float): Number or proportion of features drawn for each estimator.
	        bootstrap (bool): Specifies whether samples are drawn with replacement.
	        bootstrap_features (bool): Specifies whether features are drawn with replacement.
	        oob_score (bool): Specifies whether to use out-of-bag samples to estimate generalization.
	        warm (bool): Specifies whether to reuse the previous fitted ensemble and add estimators.
	        jobs (Optional[int]): Number of parallel jobs used by fit and predict.
	        rando (Optional[int]): Random seed used by the ensemble sampler.
	        verbose (int): Verbosity level for fit and predict.
		
	        Returns:
	        --------
	        None
		
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
		self.model = ske.BaggingRegressor(
			estimator=self.estimator,
			n_estimators=self.n_estimators,
			max_samples=self.max_samples,
			max_features=self.max_features,
			bootstrap=self.bootstrap,
			bootstrap_features=self.bootstrap_features,
			oob_score=self.oob_score,
			warm_start=self.warm_start,
			n_jobs=self.n_jobs,
			random_state=self.random_state,
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the fitted base estimator template used by the bagging ensemble.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        object | None: The fitted base estimator template.
		
        """
		if hasattr( self.model, 'estimator_' ):
			return self.model.estimator_
		raise AttributeError( 'The model has not been trained!' )
	
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
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BaggingModel | None:
		"""
		
	        Purpose:
	        --------
	        Fit the bagging regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        BaggingModel | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> BaggingModel'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained bagging regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted bagging regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.cause = 'BaggingModel'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted bagging regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'BaggingModel'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			

class VotingModel( Regression ):
	"""
	
	    Purpose:
	    --------
	
	    Prediction voting regressor for unfitted estimators. A voting regressor is an ensemble
	    meta-estimator that fits several base regressors, each on the whole dataset.
	    Then it averages the individual predictions to form a final prediction.

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
		"""
		
	        Purpose:
	        --------
	        Initialize the voting regression wrapper.
		
	        Parameters:
	        -----------
	        est (Optional[List[tuple[str, object]]]): A list of named regressors included
	            in the voting ensemble.
	        weights (Optional[List[float]]): Optional weights applied to each regressor.
	        jobs (Optional[int]): Number of parallel jobs used by fit.
	        verbose (bool): Specifies whether to emit timing information while fitting.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.estimators = est if est is not None else [
				('least_squares', skl.LinearRegression( )),
				('ridge', skl.Ridge( )),
				('nearest_neighbor', skn.KNeighborsRegressor( ))
		]
		self.weights = weights
		self.n_jobs = jobs
		self.verbose = verbose
		self.model = ske.VotingRegressor(
			estimators=self.estimators,
			weights=self.weights,
			n_jobs=self.n_jobs,
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the number of features seen during training.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def named_estimators( self ) -> Dict[ str, object ] | None:
		"""
		
	        Purpose:
	        --------
	        Return the fitted estimators by name.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        Dict[str, object] | None: A dictionary of fitted estimators keyed by name.
		
        """
		if not hasattr( self.model, 'named_estimators_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.named_estimators_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
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
	        Fit the voting regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        VotingModel | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> VotingModel'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained voting regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-Squared scores for the fitted voting regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-Squared scores.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted voting regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'VotingModel'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			

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
	
	def __init__( self, est: Optional[ List[ tuple[ str, object ] ] ] = None,
			final: object = None, cv: int = None, jobs: int = None,
			passthrough: bool = False, verbose: int = 0 ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Stacking regression wrapper.
		
	        Parameters:
	        -----------
	        est (Optional[List[tuple[str, object]]]): Named base regressors included in the ensemble.
	        final (object): Final regressor trained on stacked predictions.
	        cv (int): Cross-validation splitting strategy used to generate stacked predictions.
	        jobs (int): Number of parallel jobs used by fit.
	        passthrough (bool): Specifies whether original features are concatenated with predictions.
	        verbose (int): Verbosity level for fit.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.estimators = est if est is not None else [
				('least_squares', skl.LinearRegression( )),
				('ridge', skl.Ridge( )),
				('nearest_neighbor', skn.KNeighborsRegressor( ))
		]
		self.final_estimator = final
		self.cv = cv
		self.n_jobs = jobs
		self.passthrough = passthrough
		self.verbose = verbose
		self.model = ske.StackingRegressor(
			estimators=self.estimators,
			final_estimator=self.final_estimator,
			cv=self.cv,
			n_jobs=self.n_jobs,
			passthrough=self.passthrough
		)
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return fitted base estimators.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[object] | None: The fitted base estimators.
		
        """
		if not hasattr( self.model, 'estimators_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.estimators_
	
	@property
	def final( self ) -> object | None:
		"""
		
	        Purpose:
	        --------
	        Return the fitted final estimator.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        object | None: The fitted final estimator.
		
        """
		if not hasattr( self.model, 'final_estimator_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.final_estimator_
	
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
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return (self.X_train, self.X_test, self.y_train, self.y_test)
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
	        Fit the Stacking regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        StackingModel | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> StackingModel'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Stacking regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted Stacking regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Stacking regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'StackingModel'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			

class SupportVector( Regression ):
	"""
	
      Provides Support Vector Regression (SVR) functionality.
      
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
		"""
		
	        Purpose:
	        --------
	        Initialize the Support Vector regression wrapper.
		
	        Parameters:
	        -----------
	        kernel (str): Kernel type used by the regressor.
	        degree (int): Degree used by the polynomial kernel.
	        gamma (str | float): Kernel coefficient for rbf, poly, and sigmoid kernels.
	        coef0 (float): Independent term used by poly and sigmoid kernels.
	        tol (float): Tolerance for the stopping criterion.
	        penalty (float): Regularization parameter C.
	        epsilon (float): Epsilon tube within which no penalty is associated in the loss.
	        shrinking (bool): Specifies whether to use the shrinking heuristic.
	        cache (float): Kernel cache size in megabytes.
	        verbose (bool): Specifies whether to enable verbose libsvm output.
	        iters (int): Hard limit on iterations within the solver.
		
	        Returns:
	        --------
	        None
		
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
		self.model = skv.SVR(
			kernel=self.kernel,
			degree=self.degree,
			gamma=self.gamma,
			coef0=self.coef0,
			tol=self.tol,
			C=self.penalty,
			epsilon=self.epsilon,
			shrinking=self.shrinking,
			cache_size=self.cache_size,
			verbose=self.verbose,
			max_iter=self.max_iter
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
		"""
		
	        Purpose:
	        --------
	        Return the number of features seen during training.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        int: The number of fitted input features.
		
        """
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> SupportVector | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Support Vector regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        SupportVector | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> SupportVector'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Support Vector regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
        """
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted Support Vector regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Support Vector regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.cause = 'SupportVector'
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			

class GaussianProcess( Regression ):
	'''

	    Purpose:
	    --------
	    Allows prediction without prior fitting (based on the GP prior)
		provides an additional method sample_y(X), which evaluates samples
		drawn from the GPR (prior or posterior) at given inputs
		exposes a method log_marginal_likelihood(theta), which can be used externally
		for other ways of selecting hyperparameters, e.g., via Markov chain Monte Carlo.

    '''
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
	
	def __init__( self, kernel: object=None, alpha: float=1e-10,
			optimizer: str='fmin_l_bfgs_b', restarts: int=0,
			normalize: bool=False, copy: bool=True, rando: int=None ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Gaussian Process regression wrapper.
		
	        Parameters:
	        -----------
	        kernel (object): Covariance kernel used by the Gaussian process.
	        alpha (float): Value added to the kernel diagonal during fitting.
	        optimizer (str): Optimizer used to tune kernel hyperparameters.
	        restarts (int): Number of optimizer restarts.
	        normalize (bool): Specifies whether to normalize the target values.
	        copy (bool): Specifies whether to persist a copy of the training data.
	        rando (int): Random seed used by supported routines.
		
	        Returns:
	        --------
	        None
		
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
	
	def split_data( self, X: np.ndarray, y: np.ndarray,  size: float=0.2,
			random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
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
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GaussianProcess | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Gaussian Process regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        GaussianProcess | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> GaussianProcess'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained Gaussian Process regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
        """
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted Gaussian Process regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Gaussian Process regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.cause = 'GaussianProcess'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			
			
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
		"""
		
	        Purpose:
	        --------
	        Initialize the Multi-Layer Perceptron regression wrapper.
		
	        Parameters:
	        -----------
	        hidden (tuple): Sizes of the hidden layers.
	        activ (str): Activation function for the hidden layers.
	        solver (str): Solver used for weight optimization.
	        alpha (float): L2 regularization strength.
	        learning (str): Learning-rate schedule for stochastic optimizers.
	        rando (int): Random seed used by supported routines.
		
	        Returns:
	        --------
	        None
		
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
		"""
		
	        Purpose:
	        --------
	        Provide a list of strings representing the class members.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        List[str]: A list of member names exposed by the wrapper.
		
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
	
	def split_data( self, X: np.ndarray, y: np.ndarray,
			size: float = 0.2, random: int = 42 ) -> tuple[
		np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
	        Purpose:
	        --------
	        Split feature and target arrays into training and testing subsets.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
	        size (float): Proportion of the dataset reserved for testing.
	        random (int): Random seed used by the splitter.
		
	        Returns:
	        --------
	        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	            A tuple containing ( X_train, X_test, y_train, y_test ).
		
        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split(
				X,
				y,
				test_size=size,
				random_state=random
			)
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> MultiLayerPerceptron | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Multi-Layer Perceptron regression model.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        MultiLayerPerceptron | None: The trained wrapper instance.
		
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
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> MultiLayerPerceptron'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Predict continuous target values using the trained MLP regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
		
	        Returns:
	        --------
	        np.ndarray | None: Predicted target values.
		
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
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Compute training, testing, and R-squared scores for the fitted MLP regressor.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing training, testing, and R-squared scores.
		
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
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted MLP regressor using multiple regression metrics.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): Target vector of shape ( n_samples, ).
		
	        Returns:
	        --------
	        pd.DataFrame | None: A DataFrame containing regression evaluation metrics.
		
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
	        Purpose:
	        --------
	        Plot observed values against predicted values for regression analysis.
		
	        Parameters:
	        -----------
	        X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
	        y (np.ndarray): True target vector of shape ( n_samples, ).
		
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
			exception.method = 'scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None'
			raise exception
			
