'''
    ******************************************************************************************
      Assembly:                mathy
      Filename:                data.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
    ******************************************************************************************
    <copyright file="data.py" company="Terry D. Eppler">

             mathy Data

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
        data.py
    </summary>
******************************************************************************************
'''
from argparse import ArgumentError
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as split
from scalers import Scaler, NormalScaler, StandardScaler, MinMaxScaler
import scipy.stats as stats
from typing import Dict, Optional, List, Any
from boogr import Error
from encoders import Encoder, LabelEncoder, TargetEncoder, OrdinalEncoder, OneHotEncoder


def throw_if( name: str, value: object ):
    if value is None:
        raise ValueError( f'Argument "{name}" cannot be empty!' )


def entropy( y: np.ndarray ) -> float | None:
    """
        
        Purpose:
        --------
        Calculate the entropy of a label distribution.
    
        Entropy is a measure of the impurity or disorder in a set of labels.
        It is calculated as the sum over all classes:
        H(y) = -Σ (p_i * log2(p_i))
    
        Parameters
        ----------
        y : np.ndarray
        Array of class labels.
    
        Returns
        -------
        float
        Entropy value (non-negative scalar).
    
    """
    try:
        throw_if( 'y', y )
        unique, counts = np.unique( y, return_counts=True )
        probs = counts / len( y )
        _entropy = -np.sum( probs * np.log2( probs + 1e-9 ) )
        return _entropy
    except Exception as e:
        exception = Error( e )
        exception.module = 'mathy'
        exception.cause = 'data'
        exception.method = 'entropy( y: np.ndarray ) -> float '
        raise exception
 
        
def information_gain( X_column: np.ndarray, y: np.ndarray, threshold: float ) -> float | None:
    """
    
        Purpose:
        ---------
        Compute the information gain from splitting the data at a given threshold.
        Information gain quantifies the reduction in entropy after a dataset is split.
        It is calculated as:
        IG = H(parent) - [ (N_left / N_total) * H(left) + (N_right / N_total) * H(right) ]
    
        Parameters
        ----------
        X_column : np.ndarray - A single feature column (1D array) of predictor values.
        y : np.ndarray - Array of target labels.
        threshold : float -Threshold value to split the feature.
    
        Returns
        -------
        float
        Information gain value (higher is better). Returns 0 if no valid split is found.
    
    """
    try:
        parent_entropy = entropy( y )
        left_idx = X_column <= threshold
        right_idx = X_column > threshold
        if sum( left_idx ) == 0 or sum( right_idx ) == 0:
            return 0  # Avoid splits with empty subset
        left_entropy = entropy( y[ left_idx ] )
        right_entropy = entropy( y[ right_idx ] )
        num_left, num_right = sum( left_idx ), sum( right_idx )
        weighted_entropy = (num_left / len( y )) * left_entropy + (
                num_right / len( y )) * right_entropy
        _information = parent_entropy - weighted_entropy
        return _information
    except Exception as e:
        exception = Error( e )
        exception.module = 'mathy'
        exception.cause = 'data'
        exception.method = 'information_gain( X_column: np.ndarray, y: np.ndarray, threshold: float ) -> float'
        raise exception


def best_split( X: np.ndarray, y: np.ndarray, number: int=10 ) -> Tuple[ int, float ] | None:
	'''

			Purpose:
			-------
			Identify the best feature and threshold to split data for maximum information gain.

			Parameters
			----------
			X : np.ndarray - 2D array of shape (n_samples, n_features) representing the input features.
			y : np.ndarray - 1D array of target labels.
			number : int, optional - Number of equally spaced thresholds to
							test per feature (default is 10).

			Returns
			-------
			tuple (best_feature_index: int, best_threshold: float) — the feature and threshold that
			yield the highest information gain. If no split improves entropy, returns (None, None).

	'''
	try:
		throw_if( 'X', X )
		throw_if( 'y', y )
		best_gain = 0.0
		best_feature = None
		best_threshold = None
		for feature in range( X.shape[ 1 ] ):
			thresholds = np.linspace( X[ :, feature ].min( ), X[ :, feature ].max( ), number )
			for t in thresholds:
				gain = information_gain( X[ :, feature ], y, t )
				if gain > best_gain:
					best_gain = gain
					best_feature = feature
					best_threshold = t
		
		return (best_feature, best_threshold)
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = ('best_split( X: np.ndarray, y: np.ndarray, number: int=10 ) -> Tuple')
		raise exception


def gini_impurity( p: float ) -> float | None:
	'''

		Purpose:
		_______
		Gini impurity for a Bernoulli variable with success probability p.

		Parameters:
		_________
		p (float): Probability in [0, 1].

		Returns:
		_______
		float | None: Gini impurity, or None on error.

	'''
	try:
		throw_if( 'p', p )
		if p < 0 or p > 1:
			raise Exception( 'Argument "p" must be in [0, 1]' )
		_impurity = 1.0 - (p ** 2) - ((1.0 - p) ** 2)
		return _impurity
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'gini_impurity( p: float ) -> float'
		raise exception


def decision_tree_stump( X: np.ndarray, y: np.ndarray, num_thresholds: int = 10 ):
	"""
		
		Purpose:
		--------
		Build a one-level decision tree (stump) for classification.
	
		This function creates a decision rule using a single feature and threshold
		that maximizes information gain. The leaf nodes are labeled with the
		most common class in each partition (left and right).
	
		Parameters
		----------
		X : np.ndarray
		2D array of shape (n_samples, n_features) containing normalized feature values.
		
		y : np.ndarray
		1D array of integer class labels.
		
		num_thresholds : int, optional
		Number of threshold points to evaluate per feature (default is 10).
	
		Returns
		-------
		dict
			A dictionary representing the decision tree stump with keys:
			- 'feature': int, index of selected feature
			- 'threshold': float, value used to split
			- 'left_label': int, majority class for left split
			- 'right_label': int, majority class for right split
		
	"""
	try:
		field, depth = best_split( X, y, num_thresholds )
		if field is None or depth is None:
			return None
		
		left_idx = X[ :, field ] <= depth
		right_idx = X[ :, field ] > depth
		left_label = np.bincount( y[ left_idx ] ).argmax( )
		right_label = np.bincount( y[ right_idx ] ).argmax( )
		return \
		{
				'feature': field,
				'threshold': depth,
				'left_label': left_label,
				'right_label': right_label
		}
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'decision_tree_stump( X: np.ndarray, y: np.ndarray, num_thresholds: int=10 )'
		raise exception


def euclidian_distance( X: np.ndarray, centroids: np.ndarray ) -> np.ndarray:
    """
    
        Purpose:
        -----------
        Compute the Euclidean distance from each data point to each cluster centroid.
    
        This function is used in the E-step (Expectation step) of K-Means to determine
        the "closeness" of each point to every centroid. The result is used to assign
        each point to the nearest cluster.
    
        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the dataset.
        centroids : np.ndarray
            A 2D array of shape (k, n_features) representing the current centroids of the clusters.
    
        Returns
        -------
        np.ndarray
            A 2D array of shape (n_samples, k) where the element at [i, j] is the
            Euclidean distance between sample i and centroid j.
            
    """
    try:
        throw_if( 'X', X )
        throw_if( 'centroids', centroids )
        _distances = np.zeros( (X.shape[ 0 ], centroids.shape[ 0 ]) )
        for i in range( centroids.shape[ 0 ] ):
            _distances[ :, i ] = np.linalg.norm( X - centroids[ i ], axis=1 )
        return _distances
    except Exception as e:
        exception = Error( e )
        exception.module = 'mathy'
        exception.cause = 'data'
        exception.method = 'euclidian_distance( X: np.ndarray, centroids: np.ndarray ) -> np.ndarray'
        raise exception


def k_means( X: np.ndarray, k: int, iters=10 ) -> Tuple[ np.ndarray, np.ndarray ] | None:
	"""
	
		Purpose:
		--------
		Perform K-Means clustering using a manual implementation.
	
		This function clusters the input data into `k` clusters by minimizing the within-cluster
		variance (inertia). It follows the standard K-Means iterative process:
			1. Initialize centroids randomly from data points.
			2. Assign each data point to the nearest centroid.
			3. Recalculate centroids as the mean of assigned points.
			4. Repeat until convergence or maximum iterations reached.
	
		Parameters
		----------
		X : np.ndarray
			A 2D array of shape (n_samples, n_features) representing the input data.
		k : int
			The number of clusters to find.
		iters : int, optional
			Maximum number of iterations to perform (default is 10).
	
		Returns
		-------
		tuple
			labels : np.ndarray
				Array of shape (n_samples,) where each value is the assigned cluster index (0 to
				k-1).
			centroids : np.ndarray
				Array of shape (k, n_features) representing the final cluster centers.
	
		Notes
		-----
		- This implementation uses Euclidean distance.
		- Initial centroids are selected randomly, so results may vary unless a random seed is set.
		- No convergence tolerance is used; it only checks for exact centroid stability.
	
	"""
	try:
		throw_if( 'X', X )
		centroids = X[ np.random.choice( X.shape[ 0 ], k, replace=False ) ]
		labels = np.zeros( X.shape[ 0 ], dtype=int )
		for _ in range( iters ):
			distances = euclidian_distance( X, centroids )
			labels = np.argmin( distances, axis=1 )
			new_centroids = np.array(
				[
						X[ labels == i ].mean( axis=0 ) if np.any( labels == i ) else centroids[ i ]
						for i in range( k )
				]
			)
			if np.allclose( centroids, new_centroids ):
				centroids = new_centroids
				break
			
			centroids = new_centroids
		
		return labels, centroids
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = ('k_means( X: np.ndarray, k: int, max_iters=10 ) -> Tuple')
		raise exception
		
def misclassification_error( p: float ) -> float | None:
    '''

        Purpose:
        ________
        Misclassification error for Bernoulli (1 - max class probability).

        Parameters:
        ________
        p (float): Probability in [0, 1].

        Returns:
        ________
        float | None: Error rate, or None on error.

    '''
    try:
        throw_if( 'p', p )
        _errors = 1 - np.max( [ p, 1 - p ] )
        return _errors
    except Exception as e:
        exception = Error( e )
        exception.module = 'mathy'
        exception.cause = 'data'
        exception.method = 'misclassification_error( p: float ) -> float'
        raise exception

   
def sigmoid( z: float ) -> float | None:
    '''

        Purpose:
        _________
        While the logit function maps the probability to a real-number range, we can consider the
        inverse of this function to map the real-number range back to a [0, 1] range for the
        probability p. This inverse of the logit function is typically called the logistic
        sigmoid function,
        which is sometimes simply abbreviated to sigmoid function due to its characteristic S-shape

        Parameters:
        _________
        z (float): Real-valued input.

        Returns:
        _________
        float | None: σ(z), or None on error.

    '''
    try:
        throw_if( 'z', z )
        z = float( np.clip( z, -709, 709 ) )
        _input = 1.0 / (1.0 + np.exp( -z ) )
        return _input
    except Exception as e:
        exception = Error( e )
        exception.module = 'mathy'
        exception.cause = 'data'
        exception.method = 'sigmoid( z: float ) -> float'
        raise exception


class DataSource( ):
	"""

		Purpose:
		-----------
		Utility class for preparing machine rate datasets from a pandas DataFrame.

		Members:
		------------
		dataframe: pd.DataFrame
		stores: np.ndarray
		n_samples: int
		n_features: int
		target: str
		test_size: float
		random_state: int
		feature_names: list
		target_names
		categorical_columns
		numeric_columns: list
		X_training: pd.DataFrame
		y_training
		X_testing
		y_testing

	"""
	data: pd.DataFrame
	size: float
	seed: int
	scaler: Optional[ Scaler ]
	label_encoder: Optional[ OrdinalEncoder ]
	target_encoder: Optional[ TargetEncoder ]
	target: Optional[ str ]
	targets: Optional[ np.ndarray ]
	n_samples: Optional[ int ]
	n_features: Optional[ int ]
	percentiles: Optional[ List[ float ] ]
	scaling_factor: Optional[ int ]
	feature_names: Optional[ List[ str ] ]
	target_names: Optional[ np.ndarray ]
	categorical_columns: Optional[ List[ str ] ]
	numeric_columns: Optional[ List[ str ] ]
	numeric_data: Optional[ pd.DataFrame ]
	categorical_data: Optional[ pd.DataFrame ]
	datatuple: Optional[ List[ Tuple[ str, Encoder, List[ str ] ] ] ]
	numeric_metrics: Optional[ pd.DataFrame ]
	pivot_table: Optional[ pd.DataFrame ]
	mean_standard_error: Optional[ pd.DataFrame ]
	average: Optional[ pd.Series ]
	kurtosis: Optional[ pd.Series ]
	skew: Optional[ pd.Series ]
	variance: Optional[ pd.Series ]
	covariance: Optional[ pd.DataFrame ]
	standard_deviation: Optional[ pd.Series ]
	column_transformer: Optional[ ColumnTransformer ]
	X_training: Optional[ pd.DataFrame ]
	X_testing: Optional[ pd.DataFrame ]
	y_training: Optional[ pd.Series ]
	y_testing: Optional[ pd.Series ]
	
	def __init__( self, df: pd.DataFrame, target: str, size: float = 0.25, rando: int = 42 ):
		"""

			Purpose:
			-----------
			Initialize and split the dataset.

			Parameters:
			-----------
			df (pd.DataFrame): Source dataframe.
			target (str): Name of the target column.
			size (float): Test set proportion.
			rando (int): Random seed for reproducibility.

			Returns:
			-----------
				None

		"""
		throw_if( 'df', df )
		throw_if( 'target', target )
		self.data = df.copy( )
		self.size = size
		self.seed = rando
		if target not in df.columns:
			raise ArgumentError( None, f'target "{target}" not in dataframe' )
		
		self.target = target
		self.feature_names = [ c for c in self.data.columns if c != target ]
		self.numeric_columns = self.data.select_dtypes( include=[ 'number' ] ).columns.tolist( )
		self.categorical_columns = self.data.select_dtypes(
			include=[ 'object', 'category', ] ).columns.tolist( )
		self.n_samples = self.data.shape[ 0 ]
		self.n_features = len( self.feature_names )
		self.targets = self.data[ target ].values
		self.target_names = np.array( sorted( np.unique( self.data[ target ].to_numpy( ) ) ) )
		self.numeric_data = self.data[ self.numeric_columns ].copy( )
		self.categorical_data = self.data[ self.categorical_columns ].copy( )
		self.skew = self.data.skew( axis=0, numeric_only=True )
		self.variance = self.data.var( axis=0, ddof=1, numeric_only=True )
		self.kurtosis = self.data.kurt( axis=0, numeric_only=True )
		self.average = self.data.mean( axis=0, numeric_only=True )
		self.mean_standard_error = self.data.sem( axis=0, ddof=1, numeric_only=True )
		self.standard_deviation = self.data.std( axis=0, ddof=1, numeric_only=True )
		self.covariance = self.data.cov( ddof=1, numeric_only=True )
		self.numeric_metrics = self.data[ self.numeric_columns ].describe(
			percentiles=[ .05, .1, .25, .3, .5, .70, .8, .95 ] )
		self.datatuple = [ ]
		self.scaler = None
		self.label_encoder = None
		self.target_encoder = None
		self.pivot_table = None
		self.column_transformer = None
		self.X_training, self.X_testing, self.y_training, self.y_testing = split(
			self.data[ self.feature_names ], self.data[ self.target ], test_size=self.size,
			random_state=self.seed )
	
	def __dir__( self ):
		'''

			Purpose:
			-----------
			This function retuns a list of strings (members of the class)

		'''
		return [ 'data',
		         'target',
		         'size',
		         'seed',
		         'scaler',
		         'n_samples',
		         'n_features',
		         'targets',
		         'target_names',
		         'feature_names',
		         'label_encoder',
		         'target_encoder',
		         'categorical_columns',
		         'numeric_columns',
		         'pivot_table',
		         'mean_standard_error',
		         'average',
		         'kurtosis',
		         'skew',
		         'variance',
		         'covariance',
		         'numeric_data',
		         'numeric_metrics',
		         'standard_deviation',
		         'categorical_data',
		         'column_transformer',
		         'datatuple',
		         'X_training',
		         'X_testing',
		         'y_training',
		         'y_testing',
		         # Methods
		         'export_excel',
		         'create_heatmap',
		         'transform_columns',
		         'maxminize',
		         'normalize',
		         'standardize',
		         'encode_targets',
		         'encode_labels',
		         'encode_features',
		         'create_pivot',
		         'create_histogram', ]
	
	def transform_columns( self, name: str, encoder: Encoder, columns: List[ str ] ) -> pd.DataFrame:
		"""

			Purpose:
			-----------
			Add a (name, transformer, columns) triple and fit/transform X using ColumnTransformer.

			Parameters:
			-----------
			name (str): Transformer name.
			encoder (Preprocessor): Transformer implementing fit/transform.
			columns (list[str]): Column names to transform.

			Returns:
			-----------
			pd.DataFrame

		"""
		try:
			throw_if( 'name', name )
			throw_if( 'encoder', encoder )
			throw_if( 'columns', columns )
			self.datatuple.append( (name, encoder, columns) )
			self.column_transformer = ColumnTransformer(
				transformers=self.datatuple,
				remainder='passthrough'
			)
			values = self.data[ self.feature_names ]
			transformed = self.column_transformer.fit_transform( values )
			passthrough = [ c for c in self.feature_names if c not in columns ]
			feature_names = columns + passthrough
			df_transformed = pd.DataFrame(
				transformed,
				columns=feature_names,
				index=self.data.index
			)
			df_transformed[ self.target ] = self.data[ self.target ].values
			self.data = df_transformed
			self.feature_names = [ c for c in self.data.columns if c != self.target ]
			self.numeric_columns = self.data.select_dtypes( include=[ 'number' ] ).columns.tolist( )
			self.categorical_columns = self.data.select_dtypes(
				include=[ 'object', 'category', ] ).columns.tolist( )
			self.numeric_data = self.data[ self.numeric_columns ].copy( )
			self.categorical_data = self.data[ self.categorical_columns ].copy( )
			self.targets = self.data[ self.target ].values
			self.X_training, self.X_testing, self.y_training, self.y_testing = split(
				self.data[ self.feature_names ],
				self.data[ self.target ],
				test_size=self.size,
				random_state=self.seed
			)
			return self.data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = ('transform_columns( self, name: str, encoder: Encoder, '
			                    'columns: List[ str ] ) -> pd.DataFrame')
			raise exception
	
	def standardize( self ) -> pd.DataFrame:
		"""

			Purpose:
			-----------
			Instance method that converts numeric data values
			into a standardized form (ie, subtracting the average
			and diidiving by the standard deviation).


			Returns:
			-----------
			pd.DataFrame

		"""
		try:
			self.scaler = StandardScaler( )
			standard_data = self.scaler.train_transform( self.numeric_data )
			return standard_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'standardize( self ) -> pd.DataFrame'
			raise exception
	
	def maxminize( self ) -> pd.DataFrame:
		"""

			Purpose:
			-----------
			Instance method that converts numeric data values
			into a standardized form (ie, subtracting the average
			and diidiving by the standard deviation).


			Returns:
			-----------
			pd.DataFrame

		"""
		try:
			self.scaler = MinMaxScaler( )
			standardized_data = self.scaler.train_transform( self.numeric_data )
			return standardized_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'standardize( self ) -> pd.DataFrame'
			raise exception
	
	def normalize( self ) -> pd.DataFrame:
		"""

			Purpose:
			-----------
			Instance method that converts numeric data values
			into a standardized form (ie, subtracting the average
			and diidiving by the standard deviation).


			Returns:
			-----------
			pd.DataFrame

		"""
		try:
			self.scaler = NormalScaler( )
			normalized_data = self.scaler.train_transform( self.numeric_data )
			return normalized_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'standardize( self ) -> pd.DataFrame'
			raise exception
	
	def encode_labels( self, col: str ) -> np.ndarray:
		"""

			Purpose:
			-----------
			Instance method that converts numeric data values
			into a standardized form (ie, subtracting the average
			and diidiving by the standard deviation).


			Returns:
			-----------
			pd.DataFrame

		"""
		try:
			throw_if( 'col', col )
			self.label_encoder = LabelEncoder( )
			y = self.data[ col ].values
			labels = self.label_encoder.train_transform( y )
			self.data[ col ] = labels
			if col == self.target:
				self.targets = self.data[ self.target ].values
			if col in self.numeric_columns:
				self.numeric_data = self.data[ self.numeric_columns ].copy( )
			if col in self.categorical_columns:
				self.categorical_data = self.data[ self.categorical_columns ].copy( )
			return labels
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'encode_labels( self, col: str ) -> np.ndarray'
			raise exception
	
	def encode_features( self ) -> pd.DataFrame:
		"""

			Purpose:
			-----------
			Instance method that encodes feature columns while leaving
			the target column unchanged.


			Returns:
			-----------
			pd.DataFrame

		"""
		try:
			features = [ ]
			self.label_encoder = OrdinalEncoder( )
			for col in self.data.columns:
				if col != self.target:
					features.append( col )
			
			values = self.data[ features ].values
			encoded_features = self.label_encoder.train_transform( values )
			df_encoded = pd.DataFrame(
				encoded_features,
				columns=features,
				index=self.data.index
			)
			df_encoded[ self.target ] = self.data[ self.target ].values
			self.data = df_encoded
			self.feature_names = [ c for c in self.data.columns if c != self.target ]
			self.numeric_columns = self.data.select_dtypes( include=[ 'number' ] ).columns.tolist( )
			self.categorical_columns = self.data.select_dtypes(
				include=[ 'object', 'category', ] ).columns.tolist( )
			self.numeric_data = self.data[ self.numeric_columns ].copy( )
			self.categorical_data = self.data[ self.categorical_columns ].copy( )
			self.targets = self.data[ self.target ].values
			self.X_training, self.X_testing, self.y_training, self.y_testing = split(
				self.data[ self.feature_names ],
				self.data[ self.target ],
				test_size=self.size,
				random_state=self.seed
			)
			return self.data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'encode_features( self ) -> pd.DataFrame'
			raise exception
	
	def encode_targets( self ) -> np.ndarray:
		"""

			Purpose:
			-----------
			Instance method that encodes the target column
			and writes the result back to the dataframe.


			Returns:
			-----------
			np.ndarray

		"""
		try:
			self.label_encoder = LabelEncoder( )
			encoded_targets = self.label_encoder.train_transform( self.targets )
			self.data[ self.target ] = encoded_targets
			self.targets = self.data[ self.target ].values
			self.target_names = np.array( sorted( np.unique( self.targets ) ) )
			self.X_training, self.X_testing, self.y_training, self.y_testing = split(
				self.data[ self.feature_names ],
				self.data[ self.target ],
				test_size=self.size,
				random_state=self.seed
			)
			return encoded_targets
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'encode_targets( self ) -> np.ndarray'
			raise exception
	
	def create_pivot( self, cols: List, vals: List, idx: List ) -> pd.DataFrame:
		'''

			Purpose:
			_______
			Create a spreadsheet-style pivot table as a DataFrame.

			Parameters:
			__________
			df (pd.DataFrame): Source dataframe.
			cols (list): Columns to use for columns axis of pivot.
			vals (list): Value columns to aggregate.
			idx (list): Columns to use as row index of pivot.

			Returns:
			________
			pd.DataFrame | None: Pivot table or None on error.

		'''
		try:
			throw_if( 'cols', cols )
			throw_if( 'vals', vals )
			throw_if( 'idx', idx )
			self.pivot_table = self.data.pivot_table( index=idx, columns=cols, values=vals,
				dropna=True, margins=True )
			return self.pivot_table
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'create_pivot( self ) -> pd.DataFrame '
			raise exception
	
	def export_excel( self, filepath: str = None ) -> None:
		'''

			Purpose:
			--------
			Exports dataframe to an Excel file.


			:param filepath:
			:type filepath:
			:return:
			:rtype:
		'''
		try:
			throw_if( 'filepath', filepath )
			self.data.to_excel( filepath )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'export_excel( self, filepath: str=None ) -> None'
			raise exception
	
	def create_histogram( self ) -> None:
		'''

			Purpose:
			________

			Method to create histogram of numeric n_features.

		'''
		try:
			plt.figure( figsize=(8, 6) )
			data = self.data.sum( axis=0, numeric_only=True )
			sns.histplot( data, kde=True, legend=True, line_kws={ 'color': 'red' } )
			plt.title( 'Distributions' )
			plt.xlabel( 'Mean' )
			plt.ylabel( 'Frequency' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'create_histogram( self )'
			raise exception
	
	def create_heatmap( self, numeric: bool = True ) -> None:
		'''

			Purpose:
			--------
			Method to show the pearson-correlation analysis of the dataset.
			
			
		'''
		try:
			correlations = self.data.corr( 'pearson', numeric_only=numeric )
			plt.figure( figsize=(8, 6) )
			sns.heatmap( correlations, cmap='coolwarm', annot=True )
			plt.title( 'Correlations' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'create_heatmap( self )'
			raise exception
	
	def safe_numeric_series( self, df: pd.DataFrame, col: str ) -> np.ndarray:
		"""
		
			Purpose:
			________
			Convert a DataFrame column to a clean numeric NumPy array, dropping any
			non-numeric or missing values.
		
			Parameters:
			___________
			df : pd.DataFrame
				Source DataFrame containing the column.
			col : str
				Name of the column to convert.
		
			Returns:
			________
			np.ndarray
				One-dimensional array of float values with NaNs removed.
				
		"""
		v = pd.to_numeric( df[ col ], errors="coerce" ).dropna( ).values.astype( float )
		return v

	def create_profile( self, df: pd.DataFrame, cols: List[ str ] ) -> pd.DataFrame:
		"""
		
			Purpose:
			________
			Compute an extended descriptive statistics profile for a set of numeric
			columns, including tails, dispersion measures, and simple outlier rates.
		
			Parameters:
			___________
			df : pd.DataFrame
				DataFrame containing the numeric columns.
			cols : List[str]
				List of column names to profile.
		
			Returns:
			________
			pd.DataFrame
				DataFrame with one row per feature and many descriptive statistics
				columns (mean, std, quantiles, skew, kurtosis, outlier rates, etc.).
				
		"""
		rows: List[ Dict[ str, Any ] ] = [ ]
		n = df.shape[ 0 ]
		percentiles = [ 0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100 ]
		
		for c in cols:
			v = self.safe_numeric_series( df, c )
			non_missing = int( np.isfinite( v ).sum( ) )
			missing = int( n - non_missing )
			if non_missing == 0:
				continue
			
			q_vals = np.nanpercentile( v, percentiles )
			q = dict( zip( percentiles, q_vals ) )
			
			mean = float( np.nanmean( v ) )
			std = float( np.nanstd( v, ddof=0 ) )
			var = float( np.nanvar( v, ddof=0 ) )
			med = float( np.nanmedian( v ) )
			mad = float( np.nanmedian( np.abs( v - med ) ) )
			iqr = float( q[ 75 ] - q[ 25 ] )
			rng = float( q[ 100 ] - q[ 0 ] )
			
			skew = float( stats.skew( v ) ) if v.size >= 3 else 0.0
			kurt = float( stats.kurtosis( v ) ) if v.size >= 4 else 0.0
			zero_pct = float( (v == 0).mean( ) * 100.0 )
			
			lo = q[ 25 ] - 1.5 * iqr
			hi = q[ 75 ] + 1.5 * iqr
			out_iqr = float( ((v < lo) | (v > hi)).mean( ) * 100.0 )
			
			z = (v - mean) / (std + 1e-12)
			out_z3 = float( (np.abs( z ) > 3.0).mean( ) * 100.0 )
			
			normal_p: Optional[ float ] = None
			try:
				if 8 <= v.size <= 5000:
					_, p = stats.shapiro( v )
					normal_p = float( p )
				elif v.size > 5000:
					_, p = stats.normaltest( v[ :5000 ] )
					normal_p = float( p )
				elif v.size >= 8:
					_, p = stats.normaltest( v )
					normal_p = float( p )
			except Exception:
				normal_p = None
			
			rows.append(
				{
						"feature": c,
						"count": int( v.size ),
						"missing_pct": float( (missing / n) * 100.0 ) if n else 0.0,
						"mean": mean,
						"std": std,
						"var": var,
						"min": float( q[ 0 ] ),
						"p01": float( q[ 1 ] ),
						"p05": float( q[ 5 ] ),
						"p10": float( q[ 10 ] ),
						"q1": float( q[ 25 ] ),
						"median": float( q[ 50 ] ),
						"q3": float( q[ 75 ] ),
						"p90": float( q[ 90 ] ),
						"p95": float( q[ 95 ] ),
						"p99": float( q[ 99 ] ),
						"max": float( q[ 100 ] ),
						"iqr": iqr,
						"range": rng,
						"mad": mad,
						"skew": skew,
						"kurtosis": kurt,
						"zero_pct": zero_pct,
						"outlier_iqr_pct": out_iqr,
						"outlier_z3_pct": out_z3,
						"normality_p": normal_p,
				}
			)
		
		out = pd.DataFrame( rows )
		if out.empty:
			return out
		return out.sort_values(
			[ "missing_pct", "outlier_iqr_pct" ], ascending=[ True, False ]
		)
            
