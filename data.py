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
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union, Sequence
from pandas.core.common import random_state
from pandas.core.reshape import pivot
from sklearn.model_selection import train_test_split as split
from sklearn.covariance import empirical_covariance
from sklearn.compose import ColumnTransformer
import sklearn.decomposition as sd
import sklearn.feature_selection as sf
from sklearn.feature_selection import chi2
from torch.backends.opt_einsum import strategy
from enums import Scaler
from sklearn.metrics import silhouette_score
from sklearn.cross_decomposition import CCA
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from pydantic import BaseModel, Field, validator
from boogr import Error, ErrorDialog
from encoders import Encoder

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
        error = ErrorDialog( exception )
        error.show( )

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
        exception.method = ('information_gain( X_column: np.ndarray, y: np.ndarray, '
                            'threshold: float ) -> float')
        error = ErrorDialog( exception )
        error.show( )

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
		best_gain = 0
		best_feature = None
		best_threshold = None
		for feature in range( X.shape[ 1 ] ):
			thresholds = np.linspace( X[ :, feature ].min( ), X[ :, feature ].max( ), number )
			for t in thresholds:
				gain = information_gain( X[ :, feature ], y, number )
				if gain > best_gain:
					best_gain, best_feature, best_threshold = gain, feature, number
					return (best_feature, best_threshold)
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = ('best_split( X: np.ndarray, y: np.ndarray, number: int=10 ) -> Tuple')
		error = ErrorDialog( exception )
		error.show( )

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
        _impurity = 1.0 - max( p, 1.0 - p )
        return _impurity
    except Exception as e:
        exception = Error( e )
        exception.module = 'mathy'
        exception.cause = 'data'
        exception.method = 'gini_impurity( p: float ) -> float'
        error = ErrorDialog( exception )
        error.show( )

def decision_tree_stump( X: np.ndarray, y: np.ndarray,  num_thresholds: int=10 ):
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
    field, depth = best_split( X, y, num_thresholds )
    if field is None:
        return None
    
    left_idx = X[ :, field ] <= depth
    right_idx = X[ :, field ] > depth
    left_label = np.bincount( y[ left_idx ] ).argmax( )
    right_label = np.bincount( y[ right_idx ] ).argmax( )
    return \
    {
	    'feature': field,
	    'threshold': num_thresholds,
	    'left_label': left_label,
	    'right_label': right_label
    }

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
        error = ErrorDialog( exception )
        error.show( )

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
		centroids = X[ np.random.choice( X.shape[ 0 ], k, replace=False ) ]
		for _ in range( iters ):
			distances = euclidian_distance( X, centroids )
			labels = np.argmin( distances, axis=1 )
			new_centroids = np.array( [ X[ labels == i ].mean( axis=0 ) for i in range( k ) ] )
			if np.all( centroids == new_centroids ):
				break
			centroids = new_centroids
			return labels, centroids
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = ('k_means( X: np.ndarray, k: int, max_iters=10 ) -> Tuple')
		error = ErrorDialog( exception )
		error.show( )

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
        error = ErrorDialog( exception )
        error.show( )

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
        error = ErrorDialog( exception )
        error.show( )

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
    dataframe: pd.DataFrame
    size: float
    state: int
    data: Optional[ np.ndarray ]
    targets: Optional[ np.ndarray ]
    n_samples: Optional[ int ]
    n_features: Optional[ int ]
    scaling_factor: Optional[ int ]
    feature_names: Optional[ List[ str ] ]
    target_names: Optional[ np.ndarray ]
    categorical_columns: Optional[ List[ str ] ]
    numeric_columns: Optional[ List[ str ] ]
    numeric_data: Optional[ pd.DataFrame ]
    X_train: Optional[ np.ndarray ]
    X_test: Optional[ np.ndarray ]
    y_train: Optional[ np.ndarray ]
    y_test: Optional[ np.ndarray ]
    transtuple: Optional[ List[ Tuple[ str, Encoder, List[ str ] ] ] ]
    numeric_metrics: Optional[ pd.DataFrame ]
    categorical_metrics: Optional[ pd.DataFrame ]
    pivot_table: Optional[ pd.DataFrame ]
    mean_standard_error: Optional[ pd.DataFrame ]
    average: Optional[ pd.Series ]
    kurtosis: Optional[ pd.Series ]
    skew: Optional[ pd.Series ]
    variance: Optional[ pd.Series ]
    standard_deviation: Optional[ pd.Series ]
    column_transformer: Optional[ ColumnTransformer ]
    
    def __init__( self, df: pd.DataFrame, target: str, size: float=0.25, rando: int=42 ):
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
        self.dataframe = df.copy( )
        self.size = size
        self.state = rando
        if target not in df.columns:
            raise ArgumentError( None, f'target "{target}" not in dataframe' )
        self.feature_names = list( self.dataframe.columns )
        self.numeric_columns = self.dataframe.select_dtypes( include=[ 'number' ] ).columns.tolist( )
        self.categorical_columns = self.dataframe.select_dtypes( include=[ 'object', 'category' ] ).columns.tolist( )
        self.data = self.dataframe.values
        self.n_samples = self.dataframe.shape[ 0 ]
        self.n_features = self.dataframe.shape[ 1 ]
        self.targets = df[ target ].to_numpy( )
        self.target_names = np.array( sorted( np.unique( self.targets ) ) )
        self.X_train = split( self.data, self.targets, test_size=self.size, random_state=self.state, stratify=None )[ 0 ]
        self.X_test = split( self.data, self.targets, test_size=self.size, random_state=self.state, stratify=None )[ 1 ]
        self.y_train = split( self.data, self.targets, test_size=self.size, random_state=self.state, stratify=None )[ 2 ]
        self.y_test = split( self.data, self.targets, test_size=self.size, random_state=self.state, stratify=None )[ 3 ]
        self.numeric_data = df.select_dtypes( include='number' ).copy( )
        self.skew = self.numeric_data.skew( axis=0, numeric_only=True )
        self.variance = self.numeric_data.var( axis=0, ddof=1, numeric_only=True )
        self.kurtosis = self.numeric_data.kurt( axis=0, numeric_only=True )
        self.average = self.numeric_data.mean( axis=0, numeric_only=True )
        self.mean_standard_error = self.numeric_data.sem( axis=0, ddof=1, numeric_only=True )
        self.standard_deviation = self.numeric_data.std( axis=0, ddof=1, numeric_only=True )
        self.transtuple: List[ Tuple[ str, Encoder, list[ str ] ] ] = [ ]
        self.numeric_metrics = None
        self.categorical_metrics = None
        self.pivot_table = None
        self.column_transformer = None
    
    def __dir__( self ):
        '''

            Purpose:
            -----------
            This function retuns a list of strings (members of the class)

        '''
        return [ 'dataframe', 'n_samples', 'n_features', 'target_names', 'feature_names',
                 'test_size', 'random_state', 'categorical_metrics', 'categorical_columns',
                 'transtuple', 'numeric_metrics', 'numeric', 'pivot_table', 'calculate_statistics',
                 'numeric_columns', 'mean_standard_error', 'training_data', 'testing_data',
                 'training_values', 'testing_values', 'data', 'target', 'scale_down', 'scale_values',
                 'average', 'kurtosis', 'variance', 'y_testing', 'transform_columns',
                 'create_pivot_table', 'standard_deviation', 'export_excel', 'plot_correlations',
                 'transform_columns', 'calculate_numeric_statistics',
                 'calculate_categorical_statistics', 'create_pivot_table', 'plot_histogram', ]
    
    def transform_columns( self, name: str, encoder: Encoder, columns: List[ str ] ) -> None:
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
            None

        """
        try:
            throw_if( 'name', name )
            throw_if( 'encoder', encoder )
            throw_if( 'columns', columns )
            self.transtuple.append( ( name, encoder, columns ) )
            self.column_transformer = ColumnTransformer( transformers=self.transtuple,
	            remainder='passthrough' )
            self.data = self.dataframe[ self.feature_names ]
            _ = self.column_transformer.fit_transform( self.data )
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'DataSource'
            exception.method = ('transform_columns( self, name: str, encoder: object, n_features: '
                                'List[ str ] )')
            error = ErrorDialog( exception )
            error.show( )
    
    def calculate_numeric_statistics( self ) -> pd.DataFrame:
	    """

			Purpose:
			-----------
			Method calculating descriptive statistics for the datasets numeric n_features.

			Returns:
			-----------
			pd.DataFrame

		"""
	    try:
		    percentiles = [ .05, .1, .25, .3, .5, .75, .8, .9, .95 ]
		    self.numeric_metrics = self.dataframe.describe( percentiles, include=[ np.number ] )
		    return self.numeric_metrics
	    except Exception as e:
		    exception = Error( e )
		    exception.module = 'mathy'
		    exception.cause = 'DataSource'
		    exception.method = 'calculate_numeric_statistics( self ) -> pd.DataFrame'
		    error = ErrorDialog( exception )
		    error.show( )
    
    def calculate_categorical_statistics( self ) -> pd.DataFrame:
        """

            Purpose:
            -----------
            Method calculating descriptive statistics for the datasets categorical n_features.

            Returns:
            -----------
            pd.DataFrame

        """
        try:
            self.categorical_metrics = self.dataframe.describe( include=[ object ] )
            return self.categorical_metrics
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'DataSource'
            exception.method = 'calculate_categorical_statistics( self ) -> pd.DataFrame '
            error = ErrorDialog( exception )
            error.show( )
    
    def create_pivot_table( self, df: pd.DataFrame, cols: List, vals: List, idx: List ) -> pd.DataFrame:
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
            throw_if( 'df', df )
            throw_if( 'cols', cols )
            throw_if( 'vals', vals )
            throw_if( 'idx', idx )
            self.dataframe = df.copy( )
            self.pivot_table = df.pivot_table( index=idx, columns=cols, values=vals,
	            dropna=True, margins=True )
            return self.pivot_table
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'DataSource'
            exception.method = 'create_pivot_table( self ) -> pd.DataFrame '
            error = ErrorDialog( exception )
            error.show( )
    
    def export_excel( self, filepath: str=None ) -> None:
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
            self.dataframe.to_excel( filepath )
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'DataSource'
            exception.method = 'export_excel( self, filepath: str=None ) -> None'
            error = ErrorDialog( exception )
            error.show( )
    
    def plot_histogram( self ):
        '''

            Purpose:
            ________

            Method to create histogram of numeric n_features.

        '''
        try:
            _col_means = self.dataframe.select_dtypes( 'number' ).mean( axis=0 )
            plt.figure( figsize=( 10, 6 ) )
            sns.histplot( _col_means, bins=20, kde=True )
            plt.title( 'Column Means' )
            plt.xlabel( 'Average' )
            plt.ylabel( 'Frequency' )
            plt.show( )
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'stores'
            exception.method = 'show_histogram( self )'
            error = ErrorDialog( exception )
            error.show( )
    
    def plot_correlations( self, numeric: bool=True ):
        '''

            Purpose:
            --------
            Method to show the pearson-correlation analysis of the dataset.
        '''
        try:
            _correlation = self.dataframe.corr( 'pearson', numeric_only=numeric )
            plt.figure( figsize=( 10, 6 ) )
            sns.heatmap( _correlation, cmap='coolwarm', annot=True )
            plt.title( 'Correlation Analysis' )
            plt.show( )
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'stores'
            exception.method = 'show_correlation_analysis( self )'
            error = ErrorDialog( exception )
            error.show( )
    
    def scale_down( self, amount: int ):
        """

            Purpose:
            --------
            Divides all numeric columns in the DataFrame by 1000 and rounds to 2 decimal places.

            Parameters:
            ---------
            df (pd.DataFrame): The input DataFrame with numeric columns to be scaled.
            amount (int):  The scaling factor ex. 1000000 converts values into millions

            Returns:
            --------
            pd.DataFrame: The transformed DataFrame.

        """
        try:
            throw_if( 'amount', amount )
            self.scaling_factor = amount
            _num = self.dataframe.select_dtypes( include='number' ).columns
            self.dataframe[ _num ] = self.dataframe[ _num ].div( self.scaling_factor ).round( 2 )
            return self.dataframe
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'Data'
            exception.method = 'scale_values( df, include )'
            error = ErrorDialog( exception )
            error.show( )
    
    def scale_values( self, df: pd.DataFrame, amount: int ):
        """
        
            Purpose:
            --------
            Divides all numeric columns in the DataFrame by 1000 and rounds to 2 decimal places.
        
            Parameters:
            ---------
            df (pd.DataFrame): The input DataFrame with numeric columns to be scaled.
            amount (int):  The scaling factor ex. 1000000 converts values into millions
        
            Returns:
            --------
            pd.DataFrame: The transformed DataFrame.
        
        """
        try:
            throw_if( 'df', df )
            throw_if( 'amount', amount )
            self.scaling_factor = amount
            numeric_cols = df.select_dtypes( include='number' ).columns
            df[ numeric_cols ] = df[ numeric_cols ].div( self.scaling_factor ).round( 2 )
            return df
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'Data'
            exception.method = 'scale_values( df, include )'
            error = ErrorDialog( exception )
            error.show( )

class VarianceThreshold( ):
    """

        Purpose:
        ---------
        VarianceThreshold is a simple baseline approach to feature selection. It removes all
        feature_names whose variance doesn’t meet some threshold. By default, it removes all
        zero-variance feature_names, i.e. feature_names that have the same value in all samples.

    """
    model: sf.VarianceThreshold
    transformed_data: Optional[ np.ndarray ]
    threshold: Optional[ float ]
    
    def __init__( self, thresh: float=0.0 ) -> None:
        """

            Purpose:
            ---------
            Initialize VarianceThreshold.

            :param threshold: Features with variance below this are removed.
            :type threshold: float
            
        """
        self.threshold = thresh
        self.model = sf.VarianceThreshold( threshold=self.threshold )
        self.transformed_data = None
    
    def __dir__( self ):
	    '''
		    
		    Returns
		    -------
			A list of strings representing class members
			
	    '''
	    return [ 'threshold', 'model', 'transformed_data',
	             'train', 'transform', 'train_transform' ]
	    
    def train( self, X: np.ndarray ) -> sf.VarianceThreshold | None:
        """

            Purpose:
            ---------
            Fit the variance threshold model.

            Parameters:
            -----------
            X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
            y (np.ndarray): Target vector w/shape ( n_samples, ).

        """
        try:
            throw_if( 'X', X )
            self.model.fit( X )
            return self
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'VarianceThreshold'
            exception.method = 'fit( self, X: np.ndarray ) -> object | None'
            error = ErrorDialog( exception )
            error.show( )
    
    def transform( self, X: np.ndarray ) -> np.ndarray:
        """
    
            Purpose:
            ---------
            Apply variance threshold selection.

            Parameters:
            -----------
            X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).

        """
        try:
            throw_if( 'X', X )
            self.transformed_data = self.model.transform( X )
            return self.transformed_data
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'VarianceThreshold'
            exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
            error = ErrorDialog( exception )
            error.show( )
    
    def train_transform( self, X: np.ndarray ) -> np.ndarray:
        """
    
            Purpose:
            ---------
            Fit and transform the stores using variance thresholding.

            Parameters:
            -----------
            X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
            
            
            Return:
            -------
            np.ndarray

    
        """
        try:
            throw_if( 'X', X )
            self.transformed_data = self.model.fit_transform( X )
            return self.transformed_data
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'VarianceThreshold'
            exception.method = ''
            error = ErrorDialog( exception )
            error.show( )

class CorrelationAnalysis( ):
    """

        Canonical Correlation Analysis (CCA) extracts the ‘directions of covariance’,
        i.e. the components of each datasets that explain the most shared variance
        between both datasets.

    """
    analysis: Optional[ CCA ]
    n_components: Optional[ int ]
    scale: Optional[ bool ]
    max_iter: Optional[ int ]
    transformed_data: Optional[ Tuple[ np.ndarray, np.ndarray ] ]
    
    def __init__( self, num: int=2, scale: bool=True, size: int=500 ) -> None:
        """

            Purpose:
            ---------
            Initialize CCA.
            
            Parameters:
            -----------
            num (int): Number of components to extract.
            scale (bool): Whether to scale the correlation analysis.
            max (int): The maximum number of components to extract.
            
        """
        self.scale = scale
        self.n_components = num
        self.max_iter = size
        self.analysis = CCA( n_components=self.n_components, scale=self.scale, max_iter=self.max_iter )
        self.transformed_data = None
    
    def __dir__( self ):
	    '''
	    
		    Returns
		    -------
			Returns a list of strings representing class members.
			
	    '''
	    return [ 'analysis', 'n_components', 'max_iter', 'analysis',
	             'transformed_data', 'train', 'transform', 'train_transform' ]
	    
    def train( self, X: np.ndarray, y: np.ndarray ) -> CCA:
        """

            Purpose:
            ---------
            Fit the CCA model to X and Y.

            Parameters:
            -----------
            X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
            y (np.ndarray): Target vector w/shape ( n_samples, ).

            Return:
            -------
            CCA or None
            
        """
        try:
            throw_if( 'X', X )
            throw_if( 'y', y )
            self.analysis.fit( X, y )
            return self
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'CorrelationAnalysis'
            exception.method = 'train( self, X: np.ndarray, Y: np.ndarray ) -> object'
            error = ErrorDialog( exception )
            error.show( )
    
    def transform( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
        """

            Purpose:
            ---------
            Apply the CCA transformation.

            Parameters:
            -----------
            X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
            y (np.ndarray): Target vector w/shape ( n_samples, ).

            Return:
            -------
            (np.ndarray, np.ndarray): Transformed X and Y.
            
        """
        try:
            throw_if( 'X', X )
            throw_if( 'y', y )
            self.transformed_data = self.analysis.transform( X, y )
            return self.transformed_data
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'CorrelationAnalysis'
            exception.method = 'transform( self, X: np.ndarray, Y: np.ndarray ) -> tuple'
            error = ErrorDialog( exception )
            error.show( )
    
    def train_transform( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
        """

            Purpose:
            ---------
            Fit and transform with CCA.

            Parameters:
            -----------
            X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
            y (np.ndarray): Target vector w/shape ( n_samples, ).


        """
        try:
            throw_if( 'X', X )
            throw_if( 'y', y )
            self.transformed_data = self.analysis.fit( X, y ).transform( X, y )
            return self.transformed_data
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'CorrelationAnalysis'
            exception.method = 'train_transform( self, X: np.ndarray, Y: np.ndarray ) -> tuple'
            error = ErrorDialog( exception )
            error.show( )

class ComponentAnalysis( ):
    """

        Purpose:
        ---------
        Principal Component Analysis (PCA). Linear dimensionality reduction using
        Singular Value Decomposition of the stores to project it to a lower dimensional space.
        The input stores is centered but not scaled for each feature before applying the SVD.
        It uses the LAPACK implementation of the full SVD or a randomized truncated SVD
        by the method of Halko et al. 2009, depending on the shape of the input stores and
        the number of components to extract.

    """
    model: sd.PCA
    svd_solver: Optional[ str ]
    n_components: Optional[ int ]
    transformed_data: Optional[ np.ndarray ]
    
    def __init__( self, num: int=2, solver: str='auto' ) -> None:
        """

            Purpose:
            ---------
            Initialize PCA.

            :param num: Number of components.
            :type num: int
            
            :param solver: The solver used by the model
            :type solver: str

        """
        self.n_components = num
        self.svd_solver = solver
        self.model = sd.PCA( n_components=self.n_components, svd_solver=self.svd_solver )
        self.transformed_data = None
    
    def __dir__( self ):
	    '''
		    
		    Returns
		    -------
			A list of strings representing class members.
			
	    '''
	    return [ 'component_analysis', 'svd_solver', 'n_components', 'transformed_data',
	             'train', 'transform', 'train_transform' ]
    
    def train( self, X: np.ndarray ) -> sd.PCA:
        """

            Purpose:
            ---------
            Fit PCA to the input stores.

            Parameters:
            -----------
            X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).\

            Return:
            -------
            sd.PCA

        """
        try:
            throw_if( 'X', X )
            self.model.fit( X )
            return self
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'ComponentAnalysis'
            exception.method = 'def fit( self, X: np.ndarray ) -> ComponentAnalysis'
            error = ErrorDialog( exception )
            error.show( )
    
    def transform( self, X: np.ndarray ) -> np.ndarray:
        """

            Purpose:
            ---------
            Apply PCA transformation.

            Parameters:
            -----------
            X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
            y (np.ndarray): Target vector w/shape ( n_samples, ).
            
            Return:
            --------
            np.ndarray


        """
        try:
            throw_if( 'X', X )
            self.transformed_data = self.model.transform( X )
            return self.transformed_data
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'ComponentAnalysis'
            exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
            error = ErrorDialog( exception )
            error.show( )
    
    def train_transform( self, X: np.ndarray ) -> np.ndarray:
        """

            Purpose:
            ---------
            Fit PCA and transform input stores.

            Parameters:
            -----------
            X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
            
            Return:
            -------
            np.ndarray


        """
        try:
            throw_if( 'X', X )
            self.transformed_data = self.model.fit_transform( X )
            return self.transformed_data
        except Exception as e:
            exception = Error( e )
            exception.module = 'mathy'
            exception.cause = 'ComponentAnalysis'
            exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
            error = ErrorDialog( exception )
            error.show( )

class SelectBest( ):
	"""

		Purpose:
		---------
		

	"""
	model: sf.SelectKBest
	transformed_data: Optional[ np.ndarray ]
	threshold: Optional[ float ]
	
	def __init__( self, k: int=3 ) -> None:
		"""

			Purpose:
			---------
			Initialize SelectBest.

			:param threshold: Features with variance below this are removed.
			:type threshold: float

		"""
		self.k = k
		self.model = sf.SelectKBest( score_func=chi2, k=self.k )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members

		'''
		return [ 'k',
		         'model',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def train( self, X: np.ndarray ) -> sf.VarianceThreshold | None:
		"""

			Purpose:
			---------
			Fit the model.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'fit( self, X: np.ndarray ) -> object | None'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Apply selection.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit and transform the data.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).


			Return:
			-------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = \
				''
			error = ErrorDialog( exception )
			error.show( )

class SelectPercentage( ):
	"""

		Purpose:
		---------


	"""
	model: sf.SelectPercentile
	transformed_data: Optional[ np.ndarray ]
	threshold: Optional[ float ]
	
	def __init__( self, percent: int=10 ) -> None:
		"""

			Purpose:
			---------
			Initialize SelectBest.

			:param threshold: Features with variance below this are removed.
			:type threshold: float

		"""
		self.percent = percent
		self.model = sf.SelectPercentile( percentile=self.percent )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings representing class members

		'''
		return [ 'k',
		         'model',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def train( self, X: np.ndarray ) -> sf.SelectPercentile | None:
		"""

			Purpose:
			---------
			Fit the model.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
			y (np.ndarray): Target vector w/shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercentage'
			exception.method = 'fit( self, X: np.ndarray ) -> object | None'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Apply selection.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercentage'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit and transform the data.

			Parameters:
			-----------
			X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).


			Return:
			-------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercentage'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
