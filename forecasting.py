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
from sklearn.metrics import (mean_squared_error, mean_absolute_error, max_error,
                             median_absolute_error, explained_variance_score, r2_score)
from statsmodels.regression.linear_model import RegressionResultsWrapper
import sklearn.ensemble as ske
import sklearn.linear_model as skl


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


class ExpandingWindow( ):
	"""

		Purpose:
		--------
		Custom expanding-window time series cross-validator. Compatible with statsmodels.
		Each split yields a growing training set and fixed-size test set. Expanding window
		cross-validation (or forward-chaining) is a time series validation technique where the
		training set grows over time, incorporating more historical data in each subsequent fold
		while testing on the following period. It ensures temporal order is maintained, preventing
		data leakage, and is ideal for scenarios with limited data.

	"""
	initial_window: int
	test_window: int
	max_splits: Optional[ int ]
	n_splits: Optional[ int ]
	max_train_size: Optional[ int ]
	gap: int
	
	def __init__( self, initial: int = 30, windows: int = 10, splits: int | None = None,
			n_splits: int | None = None, max_train_size: int | None = None,
			test_size: int | None = None, gap: int = 0 ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the expanding-window cross-validator. The legacy parameters
			`initial`, `windows`, and `splits` are preserved for drop-in compatibility.
			The sklearn-like parameters `n_splits`, `max_train_size`, `test_size`, and
			`gap` are also supported.
		
			Parameters:
			-----------
			initial (int): Minimum number of observations in the first training window.
			windows (int): Legacy name for test window size.
			splits (int | None): Legacy name for the maximum number of splits.
			n_splits (int | None): Maximum number of splits to generate.
			max_train_size (int | None): Optional rolling cap on the training window size.
			test_size (int | None): Size of each test window. Overrides `windows` when set.
			gap (int): Number of observations between each train and test partition.
		
			Returns:
			--------
			None
		
		"""
		self.initial_window = int( initial )
		self.test_window = int( test_size ) if test_size is not None else int( windows )
		self.max_splits = int( splits ) if splits is not None else None
		self.n_splits = int( n_splits ) if n_splits is not None else self.max_splits
		self.max_train_size = int( max_train_size ) if max_train_size is not None else None
		self.gap = int( gap )
	
	def split( self, series: np.ndarray, y: np.ndarray = None,
			groups: np.ndarray = None ) -> Generator[ Tuple[ np.ndarray, np.ndarray ], None, None ]:
		"""
		
			Purpose:
			--------
			Yield expanding train/test index pairs for a one-dimensional time-series.
			The signature accepts `y` and `groups` for sklearn-style compatibility,
			although they are not used.
		
			Parameters:
			-----------
			series (np.ndarray): One-dimensional time-series array.
			y (Optional[np.ndarray]): Unused. Present for API compatibility.
			groups (Optional[np.ndarray]): Unused. Present for API compatibility.
		
			Returns:
			--------
			Generator[ Tuple[ np.ndarray, np.ndarray ], None, None ]:
				Train/test index pairs.
		
		"""
		try:
			throw_if( 'series', series )
			values = np.asarray( series ).reshape( -1 )
			n_obs = len( values )
			
			if self.initial_window < 1:
				raise ValueError( 'Argument "initial" must be greater than zero.' )
			
			if self.test_window < 1:
				raise ValueError( 'Argument "test_size" must be greater than zero.' )
			
			if self.gap < 0:
				raise ValueError( 'Argument "gap" cannot be negative.' )
			
			if self.max_train_size is not None and self.max_train_size < 1:
				raise ValueError( 'Argument "max_train_size" must be greater than zero.' )
			
			if self.n_splits is not None and self.n_splits < 1:
				raise ValueError( 'Argument "n_splits" must be greater than zero.' )
			
			if self.initial_window + self.gap + self.test_window > n_obs:
				message = (
						'Argument "series" does not contain enough observations for the '
						'requested initial window, gap, and test size.'
				)
				raise ValueError( message )
			
			split_count = 0
			train_stop = self.initial_window
			
			while (train_stop + self.gap + self.test_window) <= n_obs:
				if self.max_train_size is None:
					train_start = 0
				else:
					train_start = max( 0, train_stop - self.max_train_size )
				
				test_start = train_stop + self.gap
				test_stop = test_start + self.test_window
				
				train_idx = np.arange( train_start, train_stop, dtype=int )
				test_idx = np.arange( test_start, test_stop, dtype=int )
				
				if len( train_idx ) == 0 or len( test_idx ) == 0:
					break
				
				yield train_idx, test_idx
				
				split_count += 1
				if self.n_splits is not None and split_count >= self.n_splits:
					break
				
				train_stop = test_stop
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
			Materialize and return all expanding-window train/test splits.
		
			Parameters:
			-----------
			series (np.ndarray): One-dimensional time-series array.
		
			Returns:
			--------
			list[ Tuple[ np.ndarray, np.ndarray ] ]:
				List of train/test index pairs.
		
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
	
	def get_n_splits( self, series: Optional[ np.ndarray ] = None,
			y: Optional[ np.ndarray ] = None, groups: Optional[ np.ndarray ] = None ) -> int:
		"""
		
			Purpose:
			--------
			Return the number of splits. When a series is provided, compute the actual
			number of realizable splits from the data. Otherwise, return the configured
			maximum when available.
		
			Parameters:
			-----------
			series (Optional[np.ndarray]): Optional one-dimensional time-series array.
			y (Optional[np.ndarray]): Unused. Present for API compatibility.
			groups (Optional[np.ndarray]): Unused. Present for API compatibility.
		
			Returns:
			--------
			int:
				Number of splits.
		
		"""
		try:
			if series is None:
				if self.n_splits is not None:
					return int( self.n_splits )
				
				raise ValueError(
					'Argument "series" is required when the number of splits is not '
					'explicitly configured.'
				)
			
			throw_if( 'series', series )
			return len( list( self.split( series ) ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExpandingWindow'
			exception.method = 'get_n_splits'
			raise exception
	
	def visualize( self, series: np.ndarray ) -> plt.Figure | None:
		"""
		
			Purpose:
			--------
			Build and return a matplotlib figure showing each train/test split.
			Returning the figure avoids the Streamlit blank-figure issue caused by
			calling plt.show() internally.
		
			Parameters:
			-----------
			series (np.ndarray): One-dimensional time-series array.
		
			Returns:
			--------
			plt.Figure | None:
				Figure containing the split visualization.
		
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
				axis.grid( True, alpha=0.25 )
			
			plt.tight_layout( )
			return fig
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExpandingWindow'
			exception.method = 'visualize'
			raise exception


class TimeSeriesSpliter( ):
	"""
	
		Purpose:
		--------
		Provide a time-series cross-validator that mirrors the current sklearn
		TimeSeriesSplit behavior and returning train/test index pairs for ordered data.
	
	"""
	n_splits: int
	max_train_size: Optional[ int ]
	test_size: Optional[ int ]
	gap: int
	
	def __init__( self, splits: int = 5, max_train_size: Optional[ int ] = None,
			test_size: Optional[ int ] = None, gap: int = 0 ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the time-series cross-validator.
		
			Parameters:
			-----------
			splits (int): Number of splits to generate.
			max_train_size (Optional[int]): Optional cap on the training window size.
			test_size (Optional[int]): Optional fixed size for each test window.
			gap (int): Number of observations excluded between train and test windows.
		
			Returns:
			--------
			None
		
		"""
		self.n_splits = int( splits )
		self.max_train_size = max_train_size
		self.test_size = test_size
		self.gap = int( gap )
	
	def split( self, series: np.ndarray, y: np.ndarray = None,
			groups: np.ndarray = None ) -> Generator[ Tuple[ np.ndarray, np.ndarray ], None, None ]:
		"""
		
			Purpose:
			--------
			Yield expanding train/test index pairs for an ordered one-dimensional
			time-series using sklearn-like temporal cross-validation semantics.
		
			Parameters:
			-----------
			series (np.ndarray): One-dimensional ordered time-series array.
			y (Optional[np.ndarray]): Unused placeholder for API compatibility.
			groups (Optional[np.ndarray]): Unused placeholder for API compatibility.
		
			Returns:
			--------
			Generator[Tuple[np.ndarray, np.ndarray], None, None]:
				A generator of train/test index pairs.
		
		"""
		try:
			throw_if( 'series', series )
			
			values = np.asarray( series ).reshape( -1 )
			n_samples = len( values )
			
			if self.n_splits < 2:
				raise ValueError( 'Argument "splits" must be at least 2.' )
			
			if self.gap < 0:
				raise ValueError( 'Argument "gap" cannot be negative.' )
			
			if self.max_train_size is not None and self.max_train_size < 1:
				raise ValueError( 'Argument "max_train_size" must be greater than zero.' )
			
			if self.test_size is None:
				computed_test_size = n_samples // (self.n_splits + 1)
			else:
				computed_test_size = int( self.test_size )
			
			if computed_test_size < 1:
				raise ValueError( 'Argument "test_size" must be greater than zero.' )
			
			required = self.n_splits * computed_test_size + self.gap
			if required >= n_samples:
				raise ValueError(
					'The series does not contain enough observations for the '
					'requested number of splits, test size, and gap.'
				)
			
			test_starts = range( n_samples - (self.n_splits * computed_test_size),
				n_samples, computed_test_size )
			
			for test_start in test_starts:
				train_end = test_start - self.gap
				
				if train_end <= 0:
					raise ValueError(
						'The requested gap leaves no observations available for training.'
					)
				
				if self.max_train_size is not None and self.max_train_size < train_end:
					train_start = train_end - self.max_train_size
				else:
					train_start = 0
				
				train_index = np.arange( train_start, train_end, dtype=int )
				test_index = np.arange( test_start,
					min( test_start + computed_test_size, n_samples ), dtype=int )
				
				if len( train_index ) == 0 or len( test_index ) == 0:
					continue
				
				yield train_index, test_index
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TimeSeriesSpliter'
			exception.method = 'split'
			raise exception
	
	def get_n_splits( self, series: np.ndarray = None,
			y: np.ndarray = None, groups: np.ndarray = None ) -> int | None:
		"""
		
			Purpose:
			--------
			Return the configured number of time-series splits.
		
			Parameters:
			-----------
			series (Optional[np.ndarray]): Unused placeholder for API compatibility.
			y (Optional[np.ndarray]): Unused placeholder for API compatibility.
			groups (Optional[np.ndarray]): Unused placeholder for API compatibility.
		
			Returns:
			--------
			int | None:
				The configured number of splits.
		
		"""
		try:
			return self.n_splits
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TimeSeriesSpliter'
			exception.method = 'get_n_splits'
			raise exception
	
	def get_splits( self, series: np.ndarray ) -> list[ Tuple[ np.ndarray, np.ndarray ] ]:
		"""
		
			Purpose:
			--------
			Materialize all train/test index pairs for the supplied series.
		
			Parameters:
			-----------
			series (np.ndarray): One-dimensional ordered time-series array.
		
			Returns:
			--------
			list[Tuple[np.ndarray, np.ndarray]]:
				A list of train/test index pairs.
		
		"""
		try:
			throw_if( 'series', series )
			return list( self.split( series ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TimeSeriesSpliter'
			exception.method = 'get_splits'
			raise exception
	
	def visualize( self, series: np.ndarray ) -> plt.Figure | None:
		"""
		
			Purpose:
			--------
			Build and return a matplotlib figure showing each train/test split.
		
			Parameters:
			-----------
			series (np.ndarray): One-dimensional ordered time-series array.
		
			Returns:
			--------
			plt.Figure | None:
				A matplotlib figure containing the split visualization.
		
		"""
		try:
			throw_if( 'series', series )
			values = np.asarray( series, dtype=float ).reshape( -1 )
			splits = self.get_splits( values )
			
			if not splits:
				raise ValueError( 'No time-series splits were generated.' )
			
			fig, axes = plt.subplots( len( splits ), 1, figsize=(10, 2.5 * len( splits )) )
			
			if len( splits ) == 1:
				axes = [ axes ]
			
			x_axis = np.arange( len( values ) )
			
			for i, (train_idx, test_idx) in enumerate( splits ):
				axis = axes[ i ]
				axis.plot( x_axis[ train_idx ], values[ train_idx ], label='Train' )
				axis.plot( x_axis[ test_idx ], values[ test_idx ], label='Test' )
				axis.set_title( f'Split {i + 1}' )
				axis.grid( True, alpha=0.25 )
				axis.legend( )
			
			plt.tight_layout( )
			return fig
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TimeSeriesSpliter'
			exception.method = 'visualize'
			raise exception


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


class LagBoostingSeries( TimeSeries ):
	"""
	
		Purpose:
		--------
		Univariate time-series forecasting by transforming the series into lagged
		supervised-learning features.
	
	"""
	model: Optional[ ske.HistGradientBoostingRegressor ]
	lag: int
	loss: str
	quantile: Optional[ float ]
	learning_rate: float
	max_iter: int
	max_leaf_nodes: Optional[ int ]
	max_depth: Optional[ int ]
	min_samples_leaf: int
	l2_regularization: float
	max_features: float
	max_bins: int
	monotonic_cst: Optional[ object ]
	interaction_cst: Optional[ object ]
	warm_start: bool
	early_stopping: str | bool
	scoring: str
	validation_fraction: float
	n_iter_no_change: int
	tol: float
	verbose: int
	random_state: Optional[ int ]
	training_values: Optional[ np.ndarray ]
	fitted_values: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	
	def __init__( self, lag: int = 12, loss: str = 'squared_error',
			quantile: Optional[ float ] = None, rate: float = 0.1,
			iters: int = 100, leaf_nodes: Optional[ int ] = 31,
			depth: Optional[ int ] = None, leaf: int = 20,
			regularization: float = 0.0, features: float = 1.0,
			bins: int = 255, monotonic: Optional[ object ] = None,
			interaction: Optional[ object ] = None, warm: bool = False,
			stopping: str | bool = 'auto', scoring: str = 'loss',
			validation: float = 0.1, no_change: int = 10,
			tol: float = 1e-7, verbose: int = 0,
			rando: Optional[ int ] = None ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the lagged boosting time-series forecaster.
		
			Parameters:
			-----------
			lag (int): Number of lag observations used as predictors.
			loss (str): Boosting loss function.
			quantile (Optional[float]): Quantile level used when loss is quantile.
			rate (float): Learning rate applied to each boosting stage.
			iters (int): Maximum number of boosting iterations.
			leaf_nodes (Optional[int]): Maximum number of leaf nodes per tree.
			depth (Optional[int]): Maximum depth of each tree.
			leaf (int): Minimum samples per leaf.
			regularization (float): L2 regularization strength.
			features (float): Proportion of features sampled per split.
			bins (int): Maximum number of bins used for histogram binning.
			monotonic (Optional[object]): Monotonic constraints for features.
			interaction (Optional[object]): Interaction constraints for features.
			warm (bool): Reuse the solution of the previous call to fit.
			stopping (str | bool): Early-stopping strategy.
			scoring (str): Scoring method for early stopping.
			validation (float): Validation fraction used for early stopping.
			no_change (int): Early-stopping patience.
			tol (float): Numerical tolerance for early stopping.
			verbose (int): Verbosity level.
			rando (Optional[int]): Random seed.
		
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.lag = lag
		self.loss = loss
		self.quantile = quantile
		self.learning_rate = rate
		self.max_iter = iters
		self.max_leaf_nodes = leaf_nodes
		self.max_depth = depth
		self.min_samples_leaf = leaf
		self.l2_regularization = regularization
		self.max_features = features
		self.max_bins = bins
		self.monotonic_cst = monotonic
		self.interaction_cst = interaction
		self.warm_start = warm
		self.early_stopping = stopping
		self.scoring = scoring
		self.validation_fraction = validation
		self.n_iter_no_change = no_change
		self.tol = tol
		self.verbose = verbose
		self.random_state = rando
		self.model = ske.HistGradientBoostingRegressor( loss=self.loss, quantile=self.quantile,
			learning_rate=self.learning_rate, max_iter=self.max_iter,
			max_leaf_nodes=self.max_leaf_nodes, max_depth=self.max_depth,
			min_samples_leaf=self.min_samples_leaf, l2_regularization=self.l2_regularization,
			max_features=self.max_features, max_bins=self.max_bins,
			monotonic_cst=self.monotonic_cst, interaction_cst=self.interaction_cst,
			warm_start=self.warm_start, early_stopping=self.early_stopping, scoring=self.scoring,
			validation_fraction=self.validation_fraction, n_iter_no_change=self.n_iter_no_change,
			tol=self.tol, verbose=self.verbose, random_state=self.random_state )
		self.training_data = None
		self.training_values = None
		self.prediction = None
		self.fitted_values = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
	
	def lag_transform( self, series: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
			--------
			Construct lagged predictors and aligned target values from a
			one-dimensional time-series.
		
			Parameters:
			-----------
			series (np.ndarray): One-dimensional time-series values.
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray]:
				Lagged predictor matrix and target vector.
		
		"""
		try:
			throw_if( 'series', series )
			
			if self.lag < 1:
				raise ValueError( 'Argument "lag" must be greater than zero.' )
			
			values = np.asarray( series, dtype=float ).reshape( -1 )
			if len( values ) <= self.lag:
				raise ValueError(
					f'Argument "series" must contain more than {self.lag} observations.'
				)
			
			self.training_data = np.array(
				[ values[ i - self.lag:i ] for i in range( self.lag, len( values ) ) ],
				dtype=float )
			
			self.training_values = values[ self.lag: ]
			return self.training_data, self.training_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LagBoostingSeries'
			exception.method = 'lag_transform'
			raise exception
	
	def train( self, series: np.ndarray ) -> LagBoostingSeries | None:
		"""
		
			Purpose:
			--------
			Transform the series into lagged features and fit the boosting model.
		
			Parameters:
			-----------
			series (np.ndarray): One-dimensional time-series values.
		
			Returns:
			--------
			LagBoostingSeries | None:
				The trained wrapper instance.
		
		"""
		try:
			throw_if( 'series', series )
			
			if self.loss == 'quantile' and self.quantile is None:
				raise ValueError(
					'Argument "quantile" is required when loss is "quantile".'
				)
			
			x_data, y_data = self.lag_transform( series )
			self.model.fit( x_data, y_data )
			self.fitted_values = self.model.predict( x_data )
			self.prediction = None
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LagBoostingSeries'
			exception.method = 'train'
			raise exception
	
	def project( self, n_steps: int = 1 ) -> np.ndarray | None:
		"""
		
			Purpose:
			--------
			Forecast future values using recursive lag-based prediction.
		
			Parameters:
			-----------
			n_steps (int): Number of future observations to forecast.
		
			Returns:
			--------
			np.ndarray | None:
				Forecasted values.
		
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
				next_value = float( self.model.predict( last_window.reshape( 1, -1 ) )[ 0 ] )
				preds.append( next_value )
				last_window = np.roll( last_window, -1 )
				last_window[ -1 ] = next_value
			
			self.prediction = np.array( preds, dtype=float )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LagBoostingSeries'
			exception.method = 'project'
			raise exception
	
	def score( self ) -> float | None:
		"""
		
			Purpose:
			--------
			Return the in-sample coefficient of determination for the fitted model.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			float | None:
				In-sample R-squared score.
		
		"""
		try:
			throw_if( 'training_data', self.training_data )
			throw_if( 'training_values', self.training_values )
			return float( self.model.score( self.training_data, self.training_values ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LagBoostingSeries'
			exception.method = 'score'
			raise exception
	
	def analyze( self ) -> Dict[ str, float ] | None:
		"""
		
			Purpose:
			--------
			Compute standard regression diagnostics on the fitted in-sample values.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			Dict[str, float] | None:
				Dictionary of metric names and values.
		
		"""
		try:
			throw_if( 'training_values', self.training_values )
			
			if self.fitted_values is None:
				throw_if( 'training_data', self.training_data )
				self.fitted_values = self.model.predict( self.training_data )
			
			self.mean_absolute_error = mean_absolute_error( self.training_values,
				self.fitted_values )
			
			self.mean_squared_error = mean_squared_error( self.training_values,
				self.fitted_values )
			
			self.root_mean_squared_error = float( np.sqrt( self.mean_squared_error ) )
			self.r2_score = r2_score( self.training_values, self.fitted_values )
			self.explained_variance_score = explained_variance_score( self.training_values,
				self.fitted_values )
			
			self.median_absolute_error = median_absolute_error(
				self.training_values,
				self.fitted_values
			)
			self.max_error = max_error(
				self.training_values,
				self.fitted_values
			)
			
			return {
					'MAE': float( self.mean_absolute_error ),
					'MSE': float( self.mean_squared_error ),
					'RMSE': float( self.root_mean_squared_error ),
					'R2': float( self.r2_score ),
					'EVS': float( self.explained_variance_score ),
					'MedianAE': float( self.median_absolute_error ),
					'MAX': float( self.max_error )
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LagBoostingSeries'
			exception.method = 'analyze'
			raise exception


class LagQuantileSeries( TimeSeries ):
	"""
	
		Purpose:
		--------
		Univariate time-series forecasting by transforming the series into lagged supervised-learning
		features and fitting a conditional quantile model.
	
	"""
	model: Optional[ skl.QuantileRegressor ]
	lag: int
	quantile: float
	alpha: float
	fit_intercept: bool
	solver: str
	solver_options: Optional[ Dict[ str, object ] ]
	training_values: Optional[ np.ndarray ]
	fitted_values: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	
	def __init__( self, lag: int = 12, quantile: float = 0.5, alpha: float = 1.0,
			fit: bool = True, solver: str = 'highs',
			solver_options: Optional[ Dict[ str, object ] ] = None ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the lagged quantile-regression forecaster.
		
			Parameters:
			-----------
			lag (int): Number of lag observations used as predictors.
			quantile (float): Target conditional quantile in the open interval (0, 1).
			alpha (float): L1 regularization strength.
			fit (bool): Specifies whether to fit an intercept.
			solver (str): Linear-programming solver used by QuantileRegressor.
			solver_options (Optional[Dict[str, object]]): Additional solver options.
		
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.lag = lag
		self.quantile = quantile
		self.alpha = alpha
		self.fit_intercept = fit
		self.solver = solver
		self.solver_options = solver_options
		self.model = skl.QuantileRegressor(
			quantile=self.quantile,
			alpha=self.alpha,
			fit_intercept=self.fit_intercept,
			solver=self.solver,
			solver_options=self.solver_options
		)
		self.training_data = None
		self.training_values = None
		self.prediction = None
		self.fitted_values = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
	
	def lag_transform( self, series: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
			--------
			Construct lagged predictors and aligned target values from a
			one-dimensional time-series.
		
			Parameters:
			-----------
			series (np.ndarray): One-dimensional time-series values.
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray]:
				Lagged predictor matrix and target vector.
		
		"""
		try:
			throw_if( 'series', series )
			
			if self.lag < 1:
				raise ValueError( 'Argument "lag" must be greater than zero.' )
			
			values = np.asarray( series, dtype=float ).reshape( -1 )
			if len( values ) <= self.lag:
				raise ValueError(
					f'Argument "series" must contain more than {self.lag} observations.'
				)
			
			self.training_data = np.array(
				[ values[ i - self.lag:i ] for i in range( self.lag, len( values ) ) ],
				dtype=float
			)
			self.training_values = values[ self.lag: ]
			return self.training_data, self.training_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LagQuantileSeries'
			exception.method = 'lag_transform'
			raise exception
	
	def train( self, series: np.ndarray ) -> LagQuantileSeries | None:
		"""
		
			Purpose:
			--------
			Transform the series into lagged features and fit the quantile model.
		
			Parameters:
			-----------
			series (np.ndarray): One-dimensional time-series values.
		
			Returns:
			--------
			LagQuantileSeries | None:
				The trained wrapper instance.
		
		"""
		try:
			throw_if( 'series', series )
			
			if self.quantile <= 0.0 or self.quantile >= 1.0:
				raise ValueError(
					'Argument "quantile" must be strictly between 0 and 1.'
				)
			
			if self.alpha < 0.0:
				raise ValueError( 'Argument "alpha" cannot be negative.' )
			
			x_data, y_data = self.lag_transform( series )
			self.model.fit( x_data, y_data )
			self.fitted_values = self.model.predict( x_data )
			self.prediction = None
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LagQuantileSeries'
			exception.method = 'train'
			raise exception
	
	def project( self, n_steps: int = 1 ) -> np.ndarray | None:
		"""
		
			Purpose:
			--------
			Forecast future values using recursive lag-based quantile prediction.
		
			Parameters:
			-----------
			n_steps (int): Number of future observations to forecast.
		
			Returns:
			--------
			np.ndarray | None:
				Forecasted quantile values.
		
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
				next_value = float( self.model.predict( last_window.reshape( 1, -1 ) )[ 0 ] )
				preds.append( next_value )
				last_window = np.roll( last_window, -1 )
				last_window[ -1 ] = next_value
			
			self.prediction = np.array( preds, dtype=float )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LagQuantileSeries'
			exception.method = 'project'
			raise exception
	
	def score( self ) -> float | None:
		"""
		
			Purpose:
			--------
			Return the in-sample coefficient of determination for the fitted model.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			float | None:
				In-sample R-squared score.
		
		"""
		try:
			throw_if( 'training_data', self.training_data )
			throw_if( 'training_values', self.training_values )
			return float( self.model.score( self.training_data, self.training_values ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LagQuantileSeries'
			exception.method = 'score'
			raise exception
	
	def analyze( self ) -> Dict[ str, float ] | None:
		"""
		
			Purpose:
			--------
			Compute standard regression diagnostics on the fitted in-sample values.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			Dict[str, float] | None:
				Dictionary of metric names and values.
		
		"""
		try:
			throw_if( 'training_values', self.training_values )
			
			if self.fitted_values is None:
				throw_if( 'training_data', self.training_data )
				self.fitted_values = self.model.predict( self.training_data )
			
			self.mean_absolute_error = mean_absolute_error(
				self.training_values,
				self.fitted_values
			)
			self.mean_squared_error = mean_squared_error(
				self.training_values,
				self.fitted_values
			)
			self.root_mean_squared_error = float(
				np.sqrt( self.mean_squared_error )
			)
			self.r2_score = r2_score(
				self.training_values,
				self.fitted_values
			)
			self.explained_variance_score = explained_variance_score(
				self.training_values,
				self.fitted_values
			)
			self.median_absolute_error = median_absolute_error(
				self.training_values,
				self.fitted_values
			)
			self.max_error = max_error(
				self.training_values,
				self.fitted_values
			)
			
			return {
					'MAE': float( self.mean_absolute_error ),
					'MSE': float( self.mean_squared_error ),
					'RMSE': float( self.root_mean_squared_error ),
					'R2': float( self.r2_score ),
					'EVS': float( self.explained_variance_score ),
					'MedianAE': float( self.median_absolute_error ),
					'MAX': float( self.max_error )
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LagQuantileSeries'
			exception.method = 'analyze'
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


            
