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
from typing import Optional, Dict, Generator, Tuple
import numpy as np
import statsmodels.tsa.statespace.sarimax as st
import statsmodels.tsa.arima.model as am
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             median_absolute_error, explained_variance_score, r2_score)
from statsmodels.regression.linear_model import RegressionResultsWrapper

from boogr import Error, ErrorDialog

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
    
    model: Optional[ sm.regression.linear_model.RegressionResultsWrapper ]
    lag: int
    prediction: Optional[ np.ndarray ]
    training_data: Optional[ np.ndarray ]
    training_values: Optional[ np.ndarray ]
    
    def __init__( self, lag: int=5 ) -> None:
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
        try:
	        throw_if( 'series', series )
	        n = len( series )
	        self.training_data = np.array( [ series[ i - self.lag:i ] for i in range( self.lag, n ) ] )
	        self.training_values = series[ self.lag: ]
	        return ( self.training_data, self.training_values )
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'LaggingSeries'
            exception.method = 'train'
            error = ErrorDialog( exception )
            error.show( )
    
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
			self

		"""
        try:
            throw_if( 'series', series )
            X, y = self._lag_transform( series )
            X = sm.add_constant( X )
            self.model = sm.OLS( y, X )
            return self
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'LaggingSeries'
            exception.method = 'train'
            error = ErrorDialog( exception )
            error.show( )
    
    def project( self, n_steps: int=1 ) -> np.ndarray:
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
            throw_if( 'X_train', self.training_data )
            last_window = self.training_data[ -1, 1: ].copy( )
            preds = [ ]
            for _ in range( n_steps ):
                X_input = sm.add_constant( last_window.reshape( 1, -1 ) )
                self.prediction = self.model.predict( X_input )[ 0 ]
                preds.append( self.prediction )
                last_window = np.roll( last_window, -1 )
                last_window[ -1 ] = self.prediction
            
            self.prediction = np.array( preds )
            return self.prediction
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'LaggingSeries'
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
            throw_if( 'training_data', self.training_data )
            throw_if( 'training_values', self.training_values )
            return self.model.rsquared
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'LaggingSeries'
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
            throw_if( 'training_data', self.training_data )
            throw_if( 'training_values', self.training_values )
            self.prediction = self.model.predict( self.training_data )
            return \
	        {
		        'MSE': mean_squared_error( self.training_values, self.prediction ),
		        'RMSE': np.sqrt( mean_squared_error( self.training_values, self.prediction ) ),
		        'MAE': mean_absolute_error( self.training_values, self.prediction ),
		        'MedianAE': median_absolute_error( self.training_values, self.prediction ),
		        'R2': r2_score( self.training_values, self.prediction ),
		        'ExplainedVariance': explained_variance_score( self.training_values, self.prediction )
	        }
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'LaggingSeries'
            exception.method = 'analyze'
            error = ErrorDialog( exception )
            error.show( )

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
            initial_window (int): Minimum number of observations in the training set.
            test_window (int): Number of observations in each test split.
            max_splits (Optional[int]): Maximum number of splits to generate.
    
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
                Generator[ Tuple[train_indices, test_indices] ]

        """
        try:
            throw_if( 'series', series )
            count = 0
            n = len( series )
            start = self.initial_window
            while ( start + self.test_window ) <= n:
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
            exception.cause = 'ExpandingWindow'
            exception.method = ( 'split( self, series: np.ndarray ) -> '
                                'Generator[ Tuple[ np.ndarray, np.ndarray ], None, None]' )
            error = ErrorDialog( exception )
            error.show( )
    
    def get_splits( self, series: np.ndarray ) -> int | None:
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
            splits = ( n - self.initial_window ) // self.test_window
            if self.max_splits:
                return min( splits, self.max_splits )
            return splits
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'ExpandingWindow'
            exception.method = 'get_splits'
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
            n_splits = self.get_splits( series )
            fig, ax = plt.subplots( n_splits, 1, figsize=( 10, 2 * n_splits ), sharex=True )
            for i, (train, test) in enumerate( self.split( series ) ):
                ax[ i ].scatter( train, [ i + 0.5 ] * len( train ), c='blue', label='Train', marker='|' )
                ax[i ].scatter( test, [ i + 0.5 ] * len( test ), c='orange', label='Test', marker='|' )
                ax[ i ].set_ylabel( f'Split {i + 1}' )
                ax[ i ].legend( loc='upper right' )
            
            plt.xlabel( 'Time Step Index' )
            plt.suptitle( 'Expanding Window Cross-Validation' )
            plt.tight_layout( )
            plt.show( )
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'ExpandingWindow'
            exception.method = 'visualize'
            error = ErrorDialog( exception )
            error.show( )

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
    order: tuple[ int, int, int ]
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
            order (tuple): The (p,d,q) order of the model (AR, I, MA).
    
            Returns:
            --------
            None

        """
        self.order = order
        self.model = am.ARIMA( order=self.order )
        self.results = None
        self.prediction = None
        self.train_data = None
    
    def train( self, series: np.ndarray ) -> am.ARIMAResults | None:
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
            self.model = am.ARIMA( series, order=self.order )
            self.results = self.model.fit( )
            return self
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'ARIMA'
            exception.method = 'train'
            error = ErrorDialog( exception )
            error.show( )
    
    def project( self, n_steps: int = 1 ) -> np.ndarray:
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
            forecast = self.results.forecast( steps=n_steps )
            self.prediction = forecast
            return forecast
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'ARIMA'
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
            exception.cause = 'ARIMA'
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
            exception.cause = 'ARIMA'
            exception.method = 'analyze'
            error = ErrorDialog( exception )
            error.show( )

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
            order (Tuple[int, int, int]): (p,d,q) non-seasonal parameters.
            seasonal_order (Tuple[int, int, int, int]): (P,D,Q,s) seasonal parameters.
    
            Returns:
            --------
            None

        """
        self.order = order
        self.seasonal_order = seasonal
        self.model = st.SARIMAX( order=self.order, seasonal_order=self.seasonal_order )
        self.results = None
        self.training_data = None
        self.prediction = None
    
    def train( self, series: np.ndarray ) -> st.SARIMAXResults | None:
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
            self.training_data = series
            self.model = st.SARIMAX( endog=series, order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False )
            self.results = self.model.fit( disp=False )
            return self
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'SARIMA'
            exception.method = 'train'
            error = ErrorDialog( exception )
            error.show( )
    
    def project( self, n_steps: int=1 ) -> np.ndarray:
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
            self.prediction = self.results.forecast( steps=n_steps )
            return self.prediction
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'SARIMA'
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
            throw_if( 'train_data', self.training_data )
            y_true = self.training_data[ self.order[ 1 ]: ]
            y_pred = self.results.fittedvalues[ self.order[ 1 ]: ]
            return r2_score( y_true, y_pred )
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'SARIMA'
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
            throw_if( 'train_data', self.training_data )
            y_true = self.training_data[ self.order[ 1 ]: ]
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
            exception.cause = 'SARIMA'
            exception.method = 'analyze'
            error = ErrorDialog( exception )
            error.show( )
