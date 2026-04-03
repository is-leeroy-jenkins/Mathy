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
	
	def __init__( self, fit: bool = True, copy: bool = True ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Ordinary Least Squares regression wrapper.
		
	        Parameters:
	        -----------
	        fit (bool): Specifies whether to calculate the intercept for the model.
	        copy (bool): Specifies whether to copy the input feature matrix.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.fit_intercept = fit
		self.copy_X = copy
		self.model = skl.LinearRegression( fit_intercept=self.fit_intercept, copy_X=self.copy_X )
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
				'copy_X',
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
		else:
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
		else:
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
		else:
			return self.model.n_features_in_
	
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
			X_train, X_test, y_train, y_test = split( X, y, test_size=size,
				random_state=random )
			return (X_train, X_test, y_train, y_test)
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
	        Compute training, testing, and R-squared scores for the fitted OLS model.
		
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
					'Training': self.training_score,
					'Testing': self.testing_score,
					'R-Squared': self.r2_score,
			}
			
			_index = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=_index )
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
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
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
	alpha: float
	solver: str
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, alpha: float=1.0, solver: str='auto', iters: int=1000, rando: int=42 ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Ridge regression wrapper.
		
	        Parameters:
	        -----------
	        alpha (float): Regularization strength.
	        solver (str): Solver used by the Ridge estimator.
	        iters (int): Maximum number of iterations for iterative solvers.
	        rando (int): Random seed used by supported solvers.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.alpha = alpha
		self.solver = solver
		self.max_iter = iters
		self.random_state = rando
		self.model = skl.Ridge( alpha=self.alpha, solver=self.solver, max_iter=self.max_iter,
			random_state=self.random_state )
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
				'alpha',
				'solver',
				'random_state',
				'max_iter',
				'weights',
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
		else:
			return self.model.coef_
	
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
			X_train, X_test, y_train, y_test = split( X, y,
				test_size=size, random_state=random )
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
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.r2_score = r2_score( y, self.prediction )
			
			_metrics = {
					'Training Score': self.training_score,
					'Testing Score': self.testing_score,
					'R-Squared Score': self.r2_score,
			}
			
			_index = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=_index )
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
	alpha: float
	max_iter: int
	random_state: int
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, alpha: float=0.01, iters: int=500, rando: int=42 ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Lasso regression wrapper.
		
	        Parameters:
	        -----------
	        alpha (float): Regularization strength for the L1 penalty.
	        iters (int): Maximum number of iterations.
	        rando (int): Random seed used by supported routines.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.alpha = alpha
		self.max_iter = iters
		self.random_state = rando
		self.model = skl.Lasso(
			alpha=self.alpha,
			max_iter=self.max_iter,
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
				'alpha',
				'max_iter',
				'random_state',
				'weights',
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
		else:
			return self.model.coef_
	
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
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.r2_score = r2_score( y, self.prediction )
			
			_metrics = { 'Training Score': self.training_score, 'Testing Score': self.testing_score,
					'R-Squared Score': self.r2_score, }
			
			_index = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=_index )
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
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' })
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
	random_state: Optional[ int ]
	ratio: float
	alpha: float
	max_iter: int
	selection: str
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, alpha: float=1.0, ratio: float=0.5, max: int=200,
			rando: int=None, select: str='random' ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the ElasticNet regression wrapper.
		
	        Parameters:
	        -----------
	        alpha (float): Overall regularization strength.
	        ratio (float): Mixing parameter where 0.0 trends toward Ridge and 1.0 equals Lasso.
	        max (int): Maximum number of iterations.
	        rando (int): Random seed used by supported routines.
	        select (str): Coefficient update strategy used by the optimizer.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.alpha = alpha
		self.ratio = ratio
		self.random_state = rando
		self.selection = select
		self.max_iter = max
		self.model = skl.ElasticNet( alpha=self.alpha, l1_ratio=self.ratio,
			random_state=self.random_state, max_iter=self.max_iter, selection=self.selection )
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
				'alpha',
				'ratio',
				'random_state',
				'selection',
				'max_iter',
				'weights',
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
	def weights( self ) -> np.ndarray | None:
		"""
		
	        Purpose:
	        --------
	        Return the learned ElasticNet regression coefficients.
		
	        Parameters:
	        -----------
	        None
		
	        Returns:
	        --------
	        np.ndarray | None: The learned coefficient vector or matrix.
		
        """
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		else:
			return self.model.coef_
	
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
			exception.cause = 'ElasticNet'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> ElasticNet | None:
		"""
		
	        Purpose:
	        --------
	        Fit the ElasticNet regression model.
		
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
	        Predict continuous target values using the trained ElasticNet model.
		
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
	        Compute training, testing, and R-squared scores for the fitted ElasticNet model.
		
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
			exception.cause = 'ElasticNet'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted ElasticNet model using multiple regression metrics.
		
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
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, max: int = 300, shape_alpha: float = 1e-06, scale_alpha: float = 1e-06,
			shape_lambda: float = 1e-06, scale_lambda: float = 1e-06 ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Bayesian Ridge regression wrapper.
		
	        Parameters:
	        -----------
	        max (int): Maximum number of iterations.
	        shape_alpha (float): Shape parameter for the alpha Gamma prior.
	        scale_alpha (float): Scale parameter for the alpha Gamma prior.
	        shape_lambda (float): Shape parameter for the lambda Gamma prior.
	        scale_lambda (float): Scale parameter for the lambda Gamma prior.
		
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
		self.model = skl.BayesianRidge(
			max_iter=self.max_iter,
			alpha_1=self.shape_alpha,
			alpha_2=self.scale_alpha,
			lambda_1=self.shape_lambda,
			lambda_2=self.scale_lambda
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
				'shape_alpha',
				'scale_alpha',
				'shape_lambda',
				'scale_lambda',
				'max_iter',
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
			size: float = 0.2, random: int = 42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
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
	model: skl.SGDRegressor
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, loss: str = 'squared_error', iters: int = 1000, penalty: str = 'l2',
			alpha: float = 0.0001, rando: int = 42, learning_rate: str = 'optimal',
			l1_ratio: float = 0.15 ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the stochastic gradient descent regression wrapper.
		
	        Parameters:
	        -----------
	        loss (str): Loss function optimized by SGD.
	        iters (int): Maximum number of training iterations.
	        penalty (str): Regularization penalty applied to the coefficients.
	        alpha (float): Regularization strength.
	        rando (int): Random seed used by supported routines.
	        learning_rate (str): Learning-rate schedule used by the optimizer.
	        l1_ratio (float): Elastic Net mixing parameter when penalty is elasticnet.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.loss = loss
		self.max_iter = iters
		self.alpha = alpha
		self.random_state = rando
		self.penalty = penalty
		self.learning_rate = learning_rate
		self.l1_ratio = l1_ratio
		self.model = skl.SGDRegressor( loss=self.loss, max_iter=self.max_iter, penalty=self.penalty,
			alpha=self.alpha, random_state=self.random_state, learning_rate=self.learning_rate,
			l1_ratio=self.l1_ratio )
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
				'penalty',
				'max_iter',
				'random_state',
				'loss',
				'learning_rate',
				'l1_ratio',
				'alpha',
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
		else:
			return self.model.coef_
	
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
		else:
			return self.model.t_
	
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
	        Compute training, testing, and R-squared scores for the fitted SGD regressor.
		
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
			exception.cause = 'GradientDescent'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted SGD regressor using multiple regression metrics.
		
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
	n_neighbors: int
	algorithm: str
	power: float
	metric: str
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, num: int = 5, algo: str = 'auto', power: float = 2.0,
			metric: str = 'minkowski' ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the K-Nearest Neighbors regression wrapper.
		
	        Parameters:
	        -----------
	        num (int): Number of neighbors used during prediction.
	        algo (str): Neighbor-search algorithm.
	        power (float): Power parameter for the Minkowski metric.
	        metric (str): Distance metric used to search for nearest neighbors.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.n_neighbors = num
		self.algorithm = algo
		self.power = power
		self.metric = metric
		self.model = skn.KNeighborsRegressor( n_neighbors=self.n_neighbors, algorithm=self.algorithm,
			p=self.power, metric=self.metric )
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
				'algorithm',
				'n_neighbors',
				'power',
				'metric',
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
	max_depth: int | None
	criterion: str
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
	n_jobs: int | None
	
	def __init__( self, estimators: int = 100, criterion: str = 'squared_error',
			depth: int = None, rando: int = 42, jobs: int = None ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Random Forest regression wrapper.
		
	        Parameters:
	        -----------
	        estimators (int): Number of trees in the forest.
	        criterion (str): Function used to measure split quality.
	        depth (int): Maximum depth of each tree.
	        rando (int): Random seed used by supported routines.
	        jobs (int): Number of parallel jobs used during fit and prediction.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.n_estimators = estimators
		self.criterion = criterion
		self.max_depth = depth
		self.random_state = rando
		self.n_jobs = jobs
		self.model = ske.RandomForestRegressor( n_estimators=self.n_estimators, criterion=self.criterion,
			max_depth=self.max_depth, random_state=self.random_state, n_jobs=self.n_jobs )
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
				'n_estimators',
				'random_state',
				'max_depth',
				'criterion',
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
	        Compute training, testing, and R-squared scores for the fitted Random Forest regressor.
		
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
	criterion: str
	max_depth: int
	random_state: int
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
	
	def __init__( self, loss: str = 'squared_error', rate: float = 0.1,
			estimators: int = 100, criterion: str = 'friedman_mse',
			depth: int = 3, rando: int = 42 ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Gradient Boosting regression wrapper.
		
	        Parameters:
	        -----------
	        loss (str): Loss function optimized during boosting.
	        rate (float): Learning rate applied to each boosting stage.
	        estimators (int): Number of boosting stages to perform.
	        criterion (str): Function used to measure split quality in the base trees.
	        depth (int): Maximum depth of the individual regression trees.
	        rando (int): Random seed used by supported routines.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.loss = loss
		self.learning_rate = rate
		self.n_estimators = estimators
		self.criterion = criterion
		self.max_depth = depth
		self.random_state = rando
		self.model = ske.GradientBoostingRegressor( loss=self.loss, learning_rate=self.learning_rate,
			n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth,
			random_state=self.random_state )
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
				'loss',
				'learning_rate',
				'n_estimators',
				'criterion',
				'max_depth',
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
	        Compute training, testing, and R-squared scores for the fitted Gradient Boosting regressor.
		
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
	model: ske.GradientBoostingRegressor
	loss: str
	learning_rate: float
	n_estimators: int
	criterion: str
	max_depth: int
	random_state: int
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
	
	def __init__( self, loss: str = 'squared_error', rate: float = 0.1,
			estimators: int = 100, criterion: str = 'friedman_mse',
			depth: int = 3, rando: int = 42 ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Gradient Boosting regression wrapper.
		
	        Parameters:
	        -----------
	        loss (str): Loss function optimized during boosting.
	        rate (float): Learning rate applied to each boosting stage.
	        estimators (int): Number of boosting stages to perform.
	        criterion (str): Function used to measure split quality in the base trees.
	        depth (int): Maximum depth of the individual regression trees.
	        rando (int): Random seed used by supported routines.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.loss = loss
		self.learning_rate = rate
		self.n_estimators = estimators
		self.criterion = criterion
		self.max_depth = depth
		self.random_state = rando
		self.model = ske.GradientBoostingRegressor( loss=self.loss, learning_rate=self.learning_rate,
			n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth,
			random_state=self.random_state )
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
				'loss',
				'learning_rate',
				'n_estimators',
				'criterion',
				'max_depth',
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
	        Compute training, testing, and R-squared scores for the fitted Gradient Boosting regressor.
		
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
			exception.cause = 'GradientBoost'
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
	base_estimator: object
	n_estimators: int
	max_features: int | float
	random_state: int
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
	
	def __init__( self, base: object = None, num: int = 10, max: int = 1, rando: int = 42 ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Bagging regression wrapper.
		
	        Parameters:
	        -----------
	        base (object): Base estimator used inside the ensemble.
	        num (int): Number of estimators in the ensemble.
	        max (int): Number or fraction of features drawn for each base estimator.
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
		self.model = ske.BaggingRegressor( estimator=self.base_estimator, n_estimators=self.n_estimators,
			max_features=self.max_features, random_state=self.random_state )
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
				'base_estimator',
				'n_estimators',
				'max_features',
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
			exception.cause = 'BaggingModel'
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BaggingModel | None:
		"""
		
	        Purpose:
	        --------
	        Fit the Bagging regression model.
		
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
	        Predict continuous target values using the trained Bagging regressor.
		
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
	        Compute training, testing, and R-squared scores for the fitted Bagging regressor.
		
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
			exception.cause = 'BaggingModel'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""
		
	        Purpose:
	        --------
	        Evaluate the fitted Bagging regressor using multiple regression metrics.
		
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
	estimators: List[ tuple[ str, object ] ]
	weights: Optional[ List[ float ] ]
	n_jobs: Optional[ int ]
	verbose: bool
	
	def __init__( self, est: List[ tuple[ str, object ] ],
			weights: List[ float ]=None, jobs: int=None, verbose: bool=False ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Voting regression wrapper.
		
	        Parameters:
	        -----------
	        est (List[tuple[str, object]]): Named base regressors included in the ensemble.
	        weights (List[float]): Optional weights applied to the individual regressor predictions.
	        jobs (int): Number of parallel jobs used during fit.
	        verbose (bool): Specifies whether to print progress messages during fitting.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.estimators = est
		self.weights = weights
		self.n_jobs = jobs
		self.verbose = verbose
		self.model = ske.VotingRegressor( estimators=self.estimators, weights=self.weights,
			n_jobs=self.n_jobs, verbose=self.verbose )
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
				'estimators',
				'weights',
				'n_jobs',
				'verbose',
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
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
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
	        Fit the Voting regression model.
		
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
	        Predict continuous target values using the trained Voting regressor.
		
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
	        Compute training, testing, and R-squared scores for the fitted Voting regressor.
		
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
			
			_metrics = {  'Training Score': self.training_score, 'Testing Score': self.testing_score,
					'R-Squared Score': self.r2_score, }
			
			idx = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=idx )
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
	        Evaluate the fitted Voting regressor using multiple regression metrics.
		
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
	
	def __init__( self, est: List[ tuple[ str, object ] ], final: object = None,
			cv: int = None, jobs: int = None, passthrough: bool = False,
			verbose: int = 0 ) -> None:
		"""
		
	        Purpose:
	        --------
	        Initialize the Stacking regression wrapper.
		
	        Parameters:
	        -----------
	        est (List[tuple[str, object]]): Named base regressors included in the ensemble.
	        final (object): Final regressor trained on stacked predictions.
	        cv (int): Cross-validation splitting strategy used to generate stacked predictions.
	        jobs (int): Number of parallel jobs used during fitting.
	        passthrough (bool): Specifies whether to concatenate the original features to the
	            stacked predictions for the final estimator.
	        verbose (int): Verbosity level used during fitting.
		
	        Returns:
	        --------
	        None
		
        """
		super( ).__init__( )
		self.estimators = est
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
			passthrough=self.passthrough,
			verbose=self.verbose
		)
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
				'estimators',
				'final_estimator',
				'cv',
				'n_jobs',
				'passthrough',
				'verbose',
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
	probability: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	kernel: str
	C: float
	epsilon: float
	
	def __init__( self, kernel: str='rbf', C: float=1.0, epsilon: float=0.1, gamma: float='' ) -> None:
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
		self.C = C
		self.epsilon = epsilon
		self.model = skv.SVR( kernel=self.kernel, C=self.C, epsilon=self.epsilon )
		self.prediction = None
		self.probability = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
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
		         'kernel',
		         'C',
		         'epsilon',
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
		         'scatter_plot',
		         'training_score',
		         'training_score' ]
	
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
			exception.cause = ''
			exception.method = 'split_data( self, X: np.ndarray, y: np.ndarray )'
			raise exception
			
	
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
			raise exception
			
	
	def project( self, X: np.ndarray  ) -> np.ndarray | None:
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
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'project( self, X: np.ndarray, y: np.ndarray ) -> float'
			raise exception
			
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
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
			self.prediction = self.project( X )
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.r2_score = r2_score( y, self.prediction )
			
			_metrics = \
				{
						'Training Score': self.training_score,
						'Testing Score': self.testing_score,
						'R-Squared Score': self.r2_score,
				}
			
			idx = range( len( _metrics.items( ) ) )
			_dataframe = pd.DataFrame( _metrics, index=idx )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ''
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
			
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
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
	        DataFrame

        """
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.r_mean_squared_error = root_mean_squared_error( y, self.prediction )
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
			_dataframe = pd.DataFrame( _data )
			return _dataframe
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = ''
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> DataFrame'
			raise exception
			

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
			_mrk = ('o', 's', '^', 'v', '<')
			_clr = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			_cmap = ListedColormap( _clr[ :len( np.unique( y ) ) ] )
			_trn = self.training_score
			_tst = self.testing_score
			_text = f'Training Score = {_trn:.1%}\nTesting Score = {_tst:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--', label='Perfect Prediction' )
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
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
			
