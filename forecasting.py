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
from boogr import Error
from typing import Optional, Dict, Generator, Tuple
import numpy as np
import statsmodels.tsa.statespace.sarimax as st
import statsmodels.tsa.arima.model as am
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             median_absolute_error, explained_variance_score, r2_score)
from statsmodels.regression.linear_model import RegressionResultsWrapper


def throw_if( name: str, value: object ):
    if not value:
        raise Exception( f'Argument "{name}" cannot be empty!' )


class TimeSeries( ):
	'''
	
		Purpose:
		--------
		Base class for time-series objects
		
	'''
	training_data: Optional[ np.ndarray ]
	tranining_values: Optional[ np.ndarray ]
	prediction: Optional[ np.ndarray ]
	
	def __init__( self ):
		self.training_data = None
		self.tranining_values = None
		self.prediction = None

		
class LaggingSeries( TimeSeries ):
    """
    
        Purpose:
        --------
        Wraps statsmodels.OLS for univariate time-series forecasting using lag features.

	"""
    model: Optional[ RegressionResultsWrapper ]
    lag: int
    prediction: Optional[ np.ndarray ]
    training_data: Optional[ np.ndarray ]
    training_values: Optional[ np.ndarray ]
    design_matrix: Optional[ np.ndarray ]
    
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
	    self.lag = lag
	    self.model = None
	    self.prediction = None
	    self.training_data = None
	    self.training_values = None
	    self.design_matrix = None
    
    def lag_transform( self, series: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
	    """
	
			Purpose:
			--------
			Constructs lagged feature matrix and target vector.
	
			Parameters:
			-----------
			series (np.ndarray): 1D array of time-series values.
	
			Returns:
			--------
			Tuple[ np.ndarray, np.ndarray ]: Lagged predictors and target vector.

		"""
	    try:
		    throw_if( 'series', series )
		    if len( series ) <= self.lag:
			    raise ValueError( f'Argument "series" must contain more than {self.lag} observations.')
		    
		    values = np.asarray( series, dtype=float ).reshape( -1 )
		    n = len( values )
		    self.training_data = np.array([ values[ i - self.lag:i ] for i in range( self.lag, n )],
			    dtype=float )
		    self.training_values = values[ self.lag: ]
		    return self.training_data, self.training_values
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'LaggingSeries'
		    exception.method = 'lag_transform'
		    raise exception
    
    def train( self, series: np.ndarray ) -> LaggingSeries | None:
	    """
	
			Purpose:
			--------
			Transform univariate series into lagged features and fit OLS model.
	
			Parameters:
			-----------
			series (np.ndarray): 1D time-series array.
	
			Returns:
			--------
			LaggingSeries: Current instance.

		"""
	    try:
		    throw_if( 'series', series )
		    x_data, y_data = self.lag_transform( series )
		    self.design_matrix = sm.add_constant( x_data, has_constant='add' )
		    self.model = sm.OLS( y_data, self.design_matrix ).fit( )
		    return self
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'LaggingSeries'
		    exception.method = 'train'
		    raise exception
    
    def project( self, n_steps: int = 1 ) -> np.ndarray:
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
		    throw_if( 'n_steps', n_steps )
		    throw_if( 'training_data', self.training_data )
		    throw_if( 'model', self.model )
		    if n_steps < 1:
			    raise ValueError( 'Argument "n_steps" must be greater than zero.' )
		    
		    last_window = self.training_data[ -1 ].astype( float ).copy( )
		    preds = [ ]
		    for _ in range( n_steps ):
			    x_input = sm.add_constant( last_window.reshape( 1, -1 ), has_constant='add' )
			    next_value = float( self.model.predict( x_input )[ 0 ] )
			    preds.append( next_value )
			    last_window = np.roll( last_window, -1 )
			    last_window[ -1 ] = next_value
		    
		    self.prediction = np.array( preds, dtype=float )
		    return self.prediction
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'LaggingSeries'
		    exception.method = 'project'
		    raise exception
    
    def score( self ) -> float | None:
	    """

			Purpose:
			--------
			Returns R² on the training set.
	
			Parameters:
			-----------
			None
	
			Returns:
			--------
			float: R² coefficient of determination.

		"""
	    try:
		    throw_if( 'model', self.model )
		    return float( self.model.rsquared )
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'LaggingSeries'
		    exception.method = 'score'
		    raise exception
    
    def analyze( self ) -> Dict[ str, float ] | None:
	    """

			Purpose:
			--------
			Computes standard regression evaluation metrics.
	
			Parameters:
			-----------
			None
	
			Returns:
			--------
			Dict[ str, float ]: Dictionary of metric names and values.

		"""
	    try:
		    throw_if( 'training_values', self.training_values )
		    throw_if( 'design_matrix', self.design_matrix )
		    throw_if( 'model', self.model )
		    self.prediction = np.asarray( self.model.predict( self.design_matrix ), dtype=float )
		    return { 'MSE': mean_squared_error( self.training_values, self.prediction ),
				    'RMSE': np.sqrt( mean_squared_error( self.training_values, self.prediction ) ),
				    'MAE': mean_absolute_error( self.training_values, self.prediction ),
				    'MedianAE': median_absolute_error( self.training_values, self.prediction ),
				    'R2': r2_score( self.training_values, self.prediction ),
				    'ExplainedVariance': explained_variance_score( self.training_values,
					    self.prediction ) }
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'LaggingSeries'
		    exception.method = 'analyze'
		    raise exception
            

class ExpandingWindow( ):
    """

        Purpose:
        --------
        Custom expanding-window time series cross-validator. Compatible with statsmodels.
        Each split yields a growing training set and fixed-size test set.

    """
    initial_window: int
    test_window: int
    max_splits: Optional[ int ]
    
    def __init__( self, initial: int=30, windows: int=10, splits: int=None ) -> None:
	    """
	
			Purpose:
			--------
			Initializes the expanding window splitter.
	
			Parameters:
			-----------
			initial (int): Minimum number of observations in the training set.
			windows (int): Number of observations in each test split.
			splits (Optional[int]): Maximum number of splits to generate.
	
			Returns:
			--------
			None

		"""
	    self.initial_window = initial
	    self.test_window = windows
	    self.max_splits = splits
    
    def split( self, series: np.ndarray ) -> Generator[ Tuple[ np.ndarray, np.ndarray ], None, None ]:
	    """
	
			Purpose:
			--------
			Yields train/test index pairs for expanding cross-validation.
		
			Parameters:
			-----------
			series (np.ndarray): 1D time-series array.
		
			Returns:
			--------
			Generator[ Tuple[ np.ndarray, np.ndarray ], None, None ]: Train/test index pairs.

		"""
	    try:
		    throw_if( 'series', series )
		    if self.initial_window < 1:
			    raise ValueError( 'Argument "initial" must be greater than zero.' )
		    if self.test_window < 1:
			    raise ValueError( 'Argument "windows" must be greater than zero.' )
		    if self.max_splits is not None and self.max_splits < 1:
			    raise ValueError( 'Argument "splits" must be greater than zero when provided.' )
		    
		    values = np.asarray( series ).reshape( -1 )
		    n_obs = len( values )
		    mess = 'Argument "series" must contain more observations than the initial window.'
		    if self.initial_window >= n_obs:
			    raise ValueError( mess )
		    split_count = 0
		    start = self.initial_window
		    while (start + self.test_window) <= n_obs:
			    train_idx = np.arange( 0, start )
			    test_idx = np.arange( start, start + self.test_window )
			    yield train_idx, test_idx
			    start += self.test_window
			    split_count += 1
			    if self.max_splits is not None and split_count >= self.max_splits:
				    break
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'ExpandingWindow'
		    exception.method = 'split'
		    raise exception
    
    def get_splits( self, series: np.ndarray ) -> list[ Tuple[ np.ndarray, np.ndarray ] ]:
	    """
	
			Purpose:
			--------
			Materializes and returns all expanding-window splits.
	
			Parameters:
			-----------
			series (np.ndarray): 1D time-series array.
	
			Returns:
			--------
			list[ Tuple[ np.ndarray, np.ndarray ] ]: List of train/test index pairs.

		"""
	    try:
		    throw_if( 'series', series )
		    return list( self.split( series ) )
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'ExpandingWindow'
		    exception.method = 'get_splits'
		    raise exception
    
    def visualize( self, series: np.ndarray ) -> None:
	    """
	
			Purpose:
			--------
			Visualizes each expanding-window train/test split using line plots.
	
			Parameters:
			-----------
			series (np.ndarray): 1D time-series array.
	
			Returns:
			--------
			None

		"""
	    try:
		    throw_if( 'series', series )
		    values = np.asarray( series, dtype=float ).reshape( -1 )
		    splits = self.get_splits( values )
		    if not splits:
			    raise ValueError( 'No train/test splits were generated for visualization.' )
		    
		    n_splits = len( splits )
		    fig, axes = plt.subplots( n_splits, 1, figsize=(10, 2.5 * n_splits) )
		    if n_splits == 1:
			    axes = [ axes ]
		    
		    x_axis = np.arange( len( values ) )
		    for i, (train_idx, test_idx) in enumerate( splits ):
			    axis = axes[ i ]
			    axis.plot( x_axis[ train_idx ], values[ train_idx ], label='Train' )
			    axis.plot( x_axis[ test_idx ], values[ test_idx ], label='Test' )
			    axis.set_title( f'Split {i + 1}' )
			    axis.legend( )
		    
		    plt.tight_layout( )
		    plt.show( )
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'ExpandingWindow'
		    exception.method = 'visualize'
		    raise exception


class ARIMA( TimeSeries ):
	"""

		Purpose:
		--------
		Autoregressive Integrated Moving Average (ARIMA)
		This model is the basic interface for ARIMA-type models, including those with exogenous
		regressors and those with seasonal components. The most general form of the model is
		SARIMAX(p, d, q)x(P, D, Q, s). It also allows all specialized cases, including
		autoregressive models: AR(p)
		moving average models: MA(q)
		mixed autoregressive moving average models: ARMA(p, q)
		integration models: ARIMA(p, d, q)
		seasonal models: SARIMA(P, D, Q, s)
		regression with errors that follow one of the above ARIMA-type models

	"""
	order: Tuple[ int, int, int ]
	model: Optional[ am.ARIMA ]
	results: Optional[ am.ARIMAResults ]
	prediction: Optional[ np.ndarray ]
	train_data: Optional[ np.ndarray ]
	
	def __init__( self, order: Tuple[ int, int, int ]=( 1, 0, 0 ) ) -> None:
		"""

			Purpose:
			--------
			Initialize ARIMA model with a given (p,d,q) order.
	
			Parameters:
			-----------
			order (Tuple[ int, int, int ]): The (p,d,q) order of the model (AR, I, MA).
	
			Returns:
			--------
			None

		"""
		self.order = order
		self.model = None
		self.results = None
		self.prediction = None
		self.train_data = None
	
	def train( self, series: np.ndarray ) -> ARIMA | None:
		"""

			Purpose:
			--------
			Fit ARIMA model to univariate time-series data.
	
			Parameters:
			-----------
			series (np.ndarray): 1D time-series array.
	
			Returns:
			--------
			ARIMA: Current instance.

		"""
		try:
			throw_if( 'series', series )
			values = np.asarray( series, dtype=float ).reshape( -1 )
			if len( values ) <= max( self.order[ 0 ], self.order[ 2 ], 1 ):
				msg = 'Argument "series" doesnt contain enough observations for the selected ARIMA'
				raise ValueError( msg )
			
			self.train_data = values
			self.model = am.ARIMA( endog=self.train_data, order=self.order )
			self.results = self.model.fit( )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ARIMA'
			exception.method = 'train'
			raise exception
	
	def project( self, n_steps: int=1 ) -> np.ndarray:
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
			throw_if( 'n_steps', n_steps )
			if n_steps < 1:
				raise ValueError( 'Argument "n_steps" must be greater than zero.' )
			
			forecast = np.asarray( self.results.forecast( steps=n_steps ), dtype=float )
			self.prediction = forecast
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ARIMA'
			exception.method = 'project'
			raise exception
	
	def score( self ) -> float | None:
		"""
	
			Purpose:
			--------
			Returns R² score on training data.
	
			Parameters:
			-----------
			None
	
			Returns:
			--------
			float: R² score.

		"""
		try:
			throw_if( 'train_data', self.train_data )
			throw_if( 'results', self.results )
			y_pred = np.asarray( self.results.fittedvalues, dtype=float ).reshape( -1 )
			y_true = np.asarray( self.train_data, dtype=float ).reshape( -1 )
			if len( y_pred ) == 0:
				raise ValueError( 'ARIMA fitted values are empty.' )
			
			y_true = y_true[ -len( y_pred ): ]
			return float( r2_score( y_true, y_pred ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ARIMA'
			exception.method = 'score'
			raise exception
	
	def analyze( self ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			--------
			Evaluate ARIMA fit using common metrics.
	
			Parameters:
			-----------
			None
	
			Returns:
			--------
			Dict[ str, float ]: MSE, RMSE, MAE, R2, Explained Variance, Median AE.

		"""
		try:
			throw_if( 'train_data', self.train_data )
			throw_if( 'results', self.results )
			y_pred = np.asarray( self.results.fittedvalues, dtype=float ).reshape( -1 )
			y_true = np.asarray( self.train_data, dtype=float ).reshape( -1 )
			if len( y_pred ) == 0:
				raise ValueError( 'ARIMA fitted values are empty.' )
			
			y_true = y_true[ -len( y_pred ): ]
			return {
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
			exception.cause = 'ARIMA'
			exception.method = 'analyze'
			raise exception
            

class SARIMA( TimeSeries ):
    """

        Purpose:
        --------
        Wrapper for seasonal ARIMA (SARIMA)
        models using statsmodels' SARIMAX engine.

    """
    order: Tuple[ int, int, int ]
    seasonal_order: Tuple[ int, int, int, int ]
    model: Optional[ st.SARIMAX ]
    results: Optional[ st.SARIMAXResults ]
    training_data: Optional[ np.ndarray ]
    prediction: Optional[ np.ndarray ]
    
    def __init__( self, order: Tuple[ int, int, int ]=( 1, 1, 1 ),
		    seasonal: Tuple[ int, int, int, int ]=( 0, 0, 0, 0 ) ) -> None:
	    """
	
			Purpose:
			--------
			Initializes SARIMA model with ARIMA and seasonal components.
	
			Parameters:
			-----------
			order (Tuple[ int, int, int ]): (p,d,q) non-seasonal parameters.
			seasonal (Tuple[ int, int, int, int ]): (P,D,Q,s) seasonal parameters.
	
			Returns:
			--------
			None

		"""
	    self.order = order
	    self.seasonal_order = seasonal
	    self.model = None
	    self.results = None
	    self.training_data = None
	    self.prediction = None
    
    def train( self, series: np.ndarray ) -> SARIMA | None:
	    """

			Purpose:
			--------
			Fits a SARIMA model to a univariate series.
	
			Parameters:
			-----------
			series (np.ndarray): 1D time-series array.
	
			Returns:
			--------
			SARIMA: Current instance.

		"""
	    try:
		    throw_if( 'series', series )
		    values = np.asarray( series, dtype=float ).reshape( -1 )
		    min_obs = max( self.order[ 0 ] + self.order[ 1 ] + self.order[ 2 ],
			    self.seasonal_order[ 0 ] + self.seasonal_order[ 1 ] + self.seasonal_order[ 2 ], 1 )
		    if len( values ) <= min_obs:
			    msg = 'Argument "series" doesnt contain enough observations for the selected SARIMA'
			    raise ValueError( msg )
		    
		    self.training_data = values
		    self.model = st.SARIMAX( endog=self.training_data, order=self.order,
			    seasonal_order=self.seasonal_order, enforce_stationarity=False,
			    enforce_invertibility=False )
		    self.results = self.model.fit( disp=False )
		    return self
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'SARIMA'
		    exception.method = 'train'
		    raise exception
    
    def project( self, n_steps: int = 1 ) -> np.ndarray:
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
		    throw_if( 'n_steps', n_steps )
		    if n_steps < 1:
			    raise ValueError( 'Argument "n_steps" must be greater than zero.' )
		    
		    forecast = np.asarray( self.results.forecast( steps=n_steps ), dtype=float )
		    self.prediction = forecast
		    return self.prediction
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'SARIMA'
		    exception.method = 'project'
		    raise exception
    
    def score( self ) -> float | None:
	    """
	
			Purpose:
			--------
			Returns R² score on in-sample fitted values.
	
			Parameters:
			-----------
			None
	
			Returns:
			--------
			float: R² coefficient of determination.

		"""
	    try:
		    throw_if( 'training_data', self.training_data )
		    throw_if( 'results', self.results )
		    y_pred = np.asarray( self.results.fittedvalues, dtype=float ).reshape( -1 )
		    y_true = np.asarray( self.training_data, dtype=float ).reshape( -1 )
		    if len( y_pred ) == 0:
			    raise ValueError( 'SARIMA fitted values are empty.' )
		    
		    y_true = y_true[ -len( y_pred ): ]
		    return float( r2_score( y_true, y_pred ) )
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'SARIMA'
		    exception.method = 'score'
		    raise exception
    
    def analyze( self ) -> Dict[ str, float ] | None:
	    """

			Purpose:
			--------
			Evaluates SARIMA in-sample accuracy using standard regression metrics.
	
			Parameters:
			-----------
			None
	
			Returns:
			--------
			Dict[ str, float ]: Dictionary of MSE, RMSE, MAE, R², and explained variance.

		"""
	    try:
		    throw_if( 'training_data', self.training_data )
		    throw_if( 'results', self.results )
		    y_pred = np.asarray( self.results.fittedvalues, dtype=float ).reshape( -1 )
		    y_true = np.asarray( self.training_data, dtype=float ).reshape( -1 )
		    if len( y_pred ) == 0:
			    raise ValueError( 'SARIMA fitted values are empty.' )
		    
		    y_true = y_true[ -len( y_pred ): ]
		    return {
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
		    exception.cause = 'SARIMA'
		    exception.method = 'analyze'
		    raise exception
            
