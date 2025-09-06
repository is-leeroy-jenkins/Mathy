'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                forecasting.py
  Author:                  Terry D. Eppler
  Created:                 08-31-2025

  Last Modified By:        Terry D. Eppler
  Last Modified On:        08-31-2025
******************************************************************************************
<copyright file="forecasting.py" company="Terry D. Eppler">

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
	forecasting.py
</summary>
******************************************************************************************
'''
from __future__ import annotations
from typing import Optional, Dict
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import statsmodels.tsa.arima.model as am
import statsmodels.api as sm
from sklearn.metrics import (
	mean_squared_error,
	mean_absolute_error,
	median_absolute_error,
	explained_variance_score,
	r2_score
)
from booger import Error, ErrorDialog

def throw_if( name: str, value: object ):
	if not value:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class LaggedTimeSeries( ):
	"""
    
        Purpose:
        --------
        Wraps statsmodels.OLS for univariate time-series forecasting using lag features.

	"""
	
	model: Optional[ sm.regression.linear_model.RegressionResultsWrapper ]
	lag: int
	prediction: Optional[ np.ndarray ]
	X_train: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	
	def __init__( self, lag: int = 5 ) -> None:
		"""
    
            Purpose:
            --------
            Initializes the wrapper and sets lag order.
    
            Parameters:
            -----------
            lag (int): Number of lagged time-steps to use as predictors.
    
            Returns:
            --------
            None

		"""
		try:
			self.lag = lag
			self.model = None
			self.prediction = None
			self.X_train = None
			self.y_train = None
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinaryLeastSquares'
			exception.method = '__init__'
			error = ErrorDialog( exception )
			error.show( )
	
	def _lag_transform( self, series: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
		"""

		Purpose:
		--------
		Constructs lagged feature matrix and target vector.

		Parameters:
		-----------
		series (np.ndarray): 1D array of time-series values.

		Returns:
		--------
		Tuple[ X (np.ndarray), y (np.ndarray) ]

		"""
		n = len( series )
		X = np.array( [ series[ i - self.lag:i ] for i in range( self.lag, n ) ] )
		y = series[ self.lag: ]
		return X, y
	
	def train( self, series: np.ndarray ) -> LaggedTimeSeries | None:
		"""

		Purpose:
		--------
		Transform univariate series into lagged features and fit OLS model.

		Parameters:
		-----------
		series (np.ndarray): 1D time-series array.

		Returns:
		--------
		self

		"""
		try:
			throw_if( 'series', series )
			X, y = self._lag_transform( series )
			X = sm.add_constant( X )
			self.X_train = X
			self.y_train = y
			ols = sm.OLS( y, X )
			self.model = ols.fit( )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinaryLeastSquares'
			exception.method = 'train'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, n_steps: int=1 ) -> np.ndarray | None:
		"""

            Purpose:
            --------
            Forecasts future values using recursive prediction.
    
            Parameters:
            -----------
            n_steps (int): Number of time steps to predict ahead.
    
            Returns:
            --------
            np.ndarray: Array of predicted values.

		"""
		try:
			throw_if( 'X_train', self.X_train )
			last_window = self.X_train[ -1, 1: ].copy( )  # drop constant column
			preds = [ ]
			
			for _ in range( n_steps ):
				X_input = sm.add_constant( last_window.reshape( 1, -1 ) )
				y_pred = self.model.predict( X_input )[ 0 ]
				preds.append( y_pred )
				last_window = np.roll( last_window, -1 )
				last_window[ -1 ] = y_pred
			
			self.prediction = np.array( preds )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinaryLeastSquares'
			exception.method = 'project'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self ) -> float | None:
		"""

            Purpose:
            --------
            Returns R² on training set.
    
            Returns:
            --------
            float: R² coefficient of determination.

		"""
		try:
			throw_if( 'X_train', self.X_train )
			throw_if( 'y_train', self.y_train )
			return self.model.rsquared
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinaryLeastSquares'
			exception.method = 'score'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self ) -> Dict[ str, float ] | None:
		"""

            Purpose:
            --------
            Computes standard regression evaluation metrics.
    
            Returns:
            --------
            Dict[str, float]: Dictionary of metric names and values.

		"""
		try:
			throw_if( 'X_train', self.X_train )
			throw_if( 'y_train', self.y_train )
			y_pred = self.model.predict( self.X_train )
			return {
					'MSE': mean_squared_error( self.y_train, y_pred ),
					'RMSE': np.sqrt( mean_squared_error( self.y_train, y_pred ) ),
					'MAE': mean_absolute_error( self.y_train, y_pred ),
					'MedianAE': median_absolute_error( self.y_train, y_pred ),
					'R2': r2_score( self.y_train, y_pred ),
					'ExplainedVariance': explained_variance_score( self.y_train, y_pred )
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinaryLeastSquares'
			exception.method = 'analyze'
			error = ErrorDialog( exception )
			error.show( )

class ExpandingWindowSplitter:
	"""

        Purpose:
        --------
        Custom expanding-window time series cross-validator. Compatible with statsmodels.
        Each split yields a growing training set and fixed-size test set.

    """
	
	initial_window: int
	test_window: int
	max_splits: Optional[ int ]
	
	def __init__( self, initial_window: int = 30, test_window: int = 10,
	              max_splits: Optional[ int ] = None ) -> None:
		"""
    
            Purpose:
            --------
            Initializes the expanding window splitter.
    
            Parameters:
            -----------
            initial_window (int): Minimum number of observations in the training set.
            test_window (int): Number of observations in each test split.
            max_splits (Optional[int]): Maximum number of splits to generate.
    
            Returns:
            --------
            None

        """
		try:
			self.initial_window = initial_window
			self.test_window = test_window
			self.max_splits = max_splits
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExpandingWindowSplitter'
			exception.method = '__init__'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, series: np.ndarray ) -> Generator[Tuple[ np.ndarray, np.ndarray ], None, None ]:
		"""
    
                Purpose:
                --------
                Yields train/test index pairs for expanding cross-validation.
        
                Parameters:
                -----------
                series (np.ndarray): 1D time-series array.
        
                Returns:
                --------
                Generator[ Tuple[train_indices, test_indices] ]

        """
		try:
			throw_if( 'series', series )
			n = len( series )
			start = self.initial_window
			count = 0
			
			while (start + self.test_window) <= n:
				train_idx = np.arange( 0, start )
				test_idx = np.arange( start, start + self.test_window )
				yield train_idx, test_idx
				start += self.test_window
				count += 1
				if self.max_splits and count >= self.max_splits:
					break
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExpandingWindowSplitter'
			exception.method = 'split'
			error = ErrorDialog( exception )
			error.show( )
	
	def get_n_splits( self, series: np.ndarray ) -> int | None:
		"""
    
            Purpose:
            --------
            Returns the number of valid train/test pairs possible.
    
            Parameters:
            -----------
            series (np.ndarray): 1D time-series data.
    
            Returns:
            --------
            int: Number of splits

        """
		try:
			throw_if( 'series', series )
			n = len( series )
			splits = (n - self.initial_window) // self.test_window
			if self.max_splits:
				return min( splits, self.max_splits )
			return splits
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExpandingWindowSplitter'
			exception.method = 'get_n_splits'
			error = ErrorDialog( exception )
			error.show( )
	
	def visualize( self, series: np.ndarray ) -> None:
		"""

            Purpose:
            --------
            Visualizes the train/test split structure across time.
    
            Parameters:
            -----------
            series (np.ndarray): 1D time-series array.
    
            Returns:
            --------
            None

        """
		try:
			throw_if( 'series', series )
			n_splits = self.get_n_splits( series )
			fig, ax = plt.subplots( n_splits, 1, figsize = (10, 2 * n_splits), sharex = True )
			
			for i, (train, test) in enumerate( self.split( series ) ):
				ax[ i ].scatter( train, [ i + 0.5 ] * len( train ), c = "blue",
					label = "Train", marker = "|" )
				ax[ i ].scatter( test, [ i + 0.5 ] * len( test ), c = "orange",
					label = "Test", marker = "|" )
				ax[ i ].set_ylabel( f"Split {i + 1}" )
				ax[ i ].legend( loc = 'upper right' )
			
			plt.xlabel( "Time Step Index" )
			plt.suptitle( "Expanding Window Cross-Validation" )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExpandingWindowSplitter'
			exception.method = 'visualize'
			error = ErrorDialog( exception )
			error.show( )

class ArimaModel( ):
	"""

        Purpose:
        --------
        Wrapper class for statsmodels ARIMA model for univariate time-series forecasting.

    """
	order: tuple[ int, int, int ]
	model: Optional[ ARIMA ]
	results: Optional[ ARIMAResults ]
	prediction: Optional[ np.ndarray ]
	train_data: Optional[ np.ndarray ]
	
	def __init__( self, order: Tuple[ int, int, int ] = (1, 0, 0) ) -> None:
		"""

            Purpose:
            --------
            Initialize ARIMA model with a given (p,d,q) order.
    
            Parameters:
            -----------
            order (tuple): The (p,d,q) order of the model (AR, I, MA).
    
            Returns:
            --------
            None

        """
		try:
			self.order = order
			self.model = None
			self.results = None
			self.prediction = None
			self.train_data = None
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ArimaModel'
			exception.method = '__init__'
			error = ErrorDialog( exception )
			error.show( )
	
	def train( self, series: np.ndarray ) -> ArimaModel | None:
		"""

        Purpose:
        --------
        Fit ARIMA model to univariate time-series data.

        Parameters:
        -----------
        series (np.ndarray): 1D time-series array.

        Returns:
        --------
        self

        """
		try:
			throw_if( 'series', series )
			self.train_data = series
			self.model = am.ARIMA( series, order = self.order )
			self.results = self.model.fit( )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ArimaModel'
			exception.method = 'train'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, n_steps: int = 1 ) -> np.ndarray | None:
		"""

            Purpose:
            --------
            Forecast n future time steps ahead.
    
            Parameters:
            -----------
            n_steps (int): Number of steps to forecast ahead.
    
            Returns:
            --------
            np.ndarray: Forecasted values.

        """
		try:
			throw_if( 'results', self.results )
			forecast = self.results.forecast( steps = n_steps )
			self.prediction = forecast
			return forecast
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ArimaModel'
			exception.method = 'project'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self ) -> float | None:
		"""
    
            Purpose:
            --------
            Returns R² score on training data.
    
            Returns:
            --------
            float: R² score.

        """
		try:
			throw_if( 'train_data', self.train_data )
			y_pred = self.results.fittedvalues
			return r2_score( self.train_data[ self.order[ 1 ]: ], y_pred )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ArimaModel'
			exception.method = 'score'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self ) -> Dict[ str, float ] | None:
		"""

            Purpose:
            --------
            Evaluate ARIMA fit using common metrics.
    
            Returns:
            --------
            Dict[str, float]: MSE, RMSE, MAE, R², Explained Variance, Median AE.

        """
		try:
			throw_if( 'train_data', self.train_data )
			y_true = self.train_data[ self.order[ 1 ]: ]
			y_pred = self.results.fittedvalues
			return \
				{
						'MSE': mean_squared_error( y_true, y_pred ),
						'RMSE': np.sqrt( mean_squared_error( y_true, y_pred ) ),
						'MAE': mean_absolute_error( y_true, y_pred ),
						'MedianAE': median_absolute_error( y_true, y_pred ),
						'R2': r2_score( y_true, y_pred ),
						'ExplainedVariance': explained_variance_score( y_true, y_pred )
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ArimaModel'
			exception.method = 'analyze'
			error = ErrorDialog( exception )
			error.show( )

class SarimaModel( ):
	"""

        Purpose:
        --------
        Wrapper for seasonal ARIMA (SARIMA) models using statsmodels' SARIMAX engine.

    """
	
	order: Tuple[ int, int, int ]
	seasonal_order: Tuple[ int, int, int, int ]
	model: Optional[ SARIMAX ]
	results: Optional[ SARIMAXResults ]
	train_data: Optional[ np.ndarray ]
	prediction: Optional[ np.ndarray ]
	
	def __init__( self, order: Tuple[ int, int, int ] = (1, 1, 1),
	              seasonal_order: Tuple[ int, int, int, int ] = (0, 0, 0, 0) ) -> None:
		"""
    
            Purpose:
            --------
            Initializes SARIMA model with ARIMA and seasonal components.
    
            Parameters:
            -----------
            order (Tuple[int, int, int]): (p,d,q) non-seasonal parameters.
            seasonal_order (Tuple[int, int, int, int]): (P,D,Q,s) seasonal parameters.
    
            Returns:
            --------
            None

        """
		try:
			self.order = order
			self.seasonal_order = seasonal_order
			self.model = None
			self.results = None
			self.train_data = None
			self.prediction = None
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SARIMAWrapper'
			exception.method = '__init__'
			error = ErrorDialog( exception )
			error.show( )
	
	def train( self, series: np.ndarray ) -> SarimaModel | None:
		"""

            Purpose:
            --------
            Fits a SARIMA model to a univariate series.
    
            Parameters:
            -----------
            series (np.ndarray): 1D time-series array.
    
            Returns:
            --------
            self

        """
		try:
			throw_if( 'series', series )
			self.train_data = series
			self.model = SARIMAX(
				endog = series,
				order = self.order,
				seasonal_order = self.seasonal_order,
				enforce_stationarity = False,
				enforce_invertibility = False
			)
			self.results = self.model.fit( disp = False )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SarimaModel'
			exception.method = 'train'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, n_steps: int = 1 ) -> np.ndarray | None:
		"""

        Purpose:
        --------
        Forecast future time steps using SARIMA.

        Parameters:
        -----------
        n_steps (int): Number of periods to forecast.

        Returns:
        --------
        np.ndarray: Predicted future values.

        """
		try:
			throw_if( 'results', self.results )
			self.prediction = self.results.forecast( steps = n_steps )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SarimaModel'
			exception.method = 'project'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self ) -> float | None:
		"""
    
            Purpose:
            --------
            Returns R² score on in-sample fitted values.
    
            Returns:
            --------
            float: R² coefficient of determination.

        """
		try:
			throw_if( 'train_data', self.train_data )
			y_true = self.train_data[ self.order[ 1 ]: ]
			y_pred = self.results.fittedvalues[ self.order[ 1 ]: ]
			return r2_score( y_true, y_pred )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SarimaModel'
			exception.method = 'score'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self ) -> Dict[ str, float ] | None:
		"""

            Purpose:
            --------
            Evaluates SARIMA in-sample accuracy using standard regression metrics.
    
            Returns:
            --------
            Dict[str, float]: Dictionary of MSE, RMSE, MAE, R², etc.

        """
		try:
			throw_if( 'train_data', self.train_data )
			y_true = self.train_data[ self.order[ 1 ]: ]
			y_pred = self.results.fittedvalues[ self.order[ 1 ]: ]
			return \
				{
						'MSE': mean_squared_error( y_true, y_pred ),
						'RMSE': np.sqrt( mean_squared_error( y_true, y_pred ) ),
						'MAE': mean_absolute_error( y_true, y_pred ),
						'MedianAE': median_absolute_error( y_true, y_pred ),
						'R2': r2_score( y_true, y_pred ),
						'ExplainedVariance': explained_variance_score( y_true, y_pred )
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SarimaModel'
			exception.method = 'analyze'
			error = ErrorDialog( exception )
			error.show( )
