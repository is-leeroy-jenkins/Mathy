'''
	******************************************************************************************
	  Assembly:                Name
	  Filename:                features.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022
	
	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="features.py" company="Terry D. Eppler">
	
	     features.py
	     Copyright ©  2022  Terry Eppler
	
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
	features.py
	</summary>
	******************************************************************************************
'''
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import sklearn.cross_decomposition as sd
import sklearn.decomposition as sd
from sklearn.decomposition import PCA as PrincipalComponentAnalysis
from sklearn.cross_decomposition import CCA as CanonicalCorrelationAnalysis
import sklearn.feature_selection as sf
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from itertools import combinations
from sklearn.model_selection import train_test_split as split
from classifications import Classifier, NearestNeighbor
from boogr import Error

def throw_if( name: str, value: object ):
    if value is None:
        raise ValueError( f'Argument "{name}" cannot be empty!' )

class Selector( ):
	'''
		
		Purpose:
		--------
		Base class for implementing feature selection functionality
		
		
	'''
	markers: Optional[ List[ str ] ]
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	
	def __init__( self ) -> None:
		"""
		
			Purpose:
			--------
			Initialize shared selector state and plotting markers used by
			concrete selector implementations.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
		
		"""
		self.markers = [ '.',
		                 'o',
		                 'v',
		                 '^',
		                 '<',
		                 '>',
		                 '1',
		                 '2',
		                 '3',
		                 '4',
		                 '8',
		                 's',
		                 'p',
		                 'P',
		                 '*',
		                 'h',
		                 'H',
		                 '+',
		                 'x',
		                 'X',
		                 'd',
		                 'D' ]
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
	
	def split_data( self, X: np.ndarray, y: np.ndarray ) -> tuple:
		"""
		
			Purpose:
			--------
			Split the specified feature matrix and target vector into training
			and testing subsets.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
		
			Returns:
			--------
			tuple:
				A four-item tuple in the form
				( X_train, X_test, y_train, y_test ).
		
		"""
		raise NotImplementedError
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
		"""
		
			Purpose:
			--------
			Fit the underlying selector or decomposition model using the
			specified feature matrix and optional target vector.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector of shape
				( n_samples, ). Required only for supervised selectors or
				supervised decomposition models.
		
			Returns:
			--------
			object | None:
				The fitted selector instance or None if not implemented by
				the concrete subclass.
		
		"""
		raise NotImplementedError
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Project or predict from the specified feature matrix using a fitted
			model when the wrapped estimator supports that operation.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector. This argument is
				preserved for subclass compatibility where needed.
		
			Returns:
			--------
			np.ndarray:
				The projected or predicted output produced by the fitted model.
		
		"""
		raise NotImplementedError
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
		"""
		
			Purpose:
			--------
			Apply the fitted selector or decomposition model to transform the
			specified feature matrix into a reduced or alternative feature space.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				subclass compatibility where required by the wrapped estimator.
		
			Returns:
			--------
			object | None:
				The transformed data produced by the fitted model.
		
		"""
		raise NotImplementedError
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
		"""
		
			Purpose:
			--------
			Fit the underlying selector or decomposition model and immediately
			transform the specified feature matrix.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector of shape
				( n_samples, ). Required only for supervised selectors or
				supervised decomposition models.
		
			Returns:
			--------
			object | None:
				The transformed data produced by the fitted model.
		
		"""
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""
		
			Purpose:
			--------
			Compute metrics or summary information for the fitted selector or
			decomposition model using the specified feature matrix and target
			vector.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
		
			Returns:
			--------
			Dict[str, float] | None:
				A metrics dictionary or None if the concrete subclass does not
				expose scoring information.
		
		"""
		raise NotImplementedError


class VarianceThreshold( Selector ):
	"""

		Purpose:
		---------
		VarianceThreshold is a simple baseline approach to feature selection. It removes all
		feature_names whose variance doesn’t meet some threshold. By default, it removes all
		zero-variance feature_names, i.e. feature_names that have the same value in all samples.

	"""
	model: sf.VarianceThreshold
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	threshold: Optional[ float ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, thresh: float = 0.0 ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the VarianceThreshold selector.
		
			Parameters:
			-----------
			thresh (float): Features with training-set variance lower than this
				threshold are removed.
		
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.threshold = thresh
		self.model = sf.VarianceThreshold( threshold=self.threshold )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return a list of strings representing class members.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[str]:
				A list of member names.
		
		"""
		return [ 'threshold',
		         'model',
		         'prediction',
		         'transformed_data',
		         'split_data',
		         'train',
		         'project',
		         'score',
		         'transform',
		         'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
			--------
			Split the specified feature matrix and target vector into training
			and testing subsets.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Proportion of the dataset to include in the test split.
			random (int): Random seed used by the splitter.
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
				A four-item tuple in the form
				( X_train, X_test, y_train, y_test ).
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = ('split_data( self, X: np.ndarray, y: np.ndarray, '
			                    'size: float=0.2, random: int=42 ) -> '
			                    'Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]')
			raise exception
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> sf.VarianceThreshold | None:
		"""
		
			Purpose:
			--------
			Fit the variance-threshold selector using the specified feature matrix.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by this selector.
		
			Returns:
			--------
			sf.VarianceThreshold | None:
				The fitted wrapper instance.
		
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Project the specified feature matrix into the retained feature space.
			For this selector, projection is equivalent to transformation.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by this selector.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only retained features.
		
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Return selector summary metrics describing the effect of variance
			thresholding on the specified feature matrix.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by this selector.
		
			Returns:
			--------
			pd.DataFrame:
				A one-row dataframe containing threshold and retained-feature
				metrics for the fitted selector.
		
		"""
		try:
			throw_if( 'X', X )
			if self.transformed_data is None:
				self.transformed_data = self.transform( X )
			_support = self.model.get_support( )
			_original_count = int( X.shape[ 1 ] )
			_retained_count = int( self.transformed_data.shape[ 1 ] )
			_removed_count = int( _original_count - _retained_count )
			_metrics = \
			{
				'Threshold': [ self.threshold ],
				'Original Features': [ _original_count ],
				'Retained Features': [ _retained_count ],
				'Removed Features': [ _removed_count ],
				'Selection Ratio': [ _retained_count / _original_count if _original_count else 0.0 ],
				'Removed Ratio': [ _removed_count / _original_count if _original_count else 0.0 ],
				'Minimum Retained Variance': [ float( np.min( self.model.variances_[ _support ] ) )
						if np.any( _support ) else np.nan ],
				'Maximum Retained Variance': [ float( np.max( self.model.variances_[ _support ] ) )
						if np.any( _support ) else np.nan ],
			}
			return pd.DataFrame( _metrics )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = ('score( self, X: np.ndarray, '
			                    'y: Optional[ np.ndarray ]=None ) -> pd.DataFrame')
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Apply variance-threshold feature selection to the specified feature
			matrix using the fitted selector.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by this selector.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only retained features.
		
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Fit the variance-threshold selector and immediately transform the
			specified feature matrix.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by this selector.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only retained features.
		
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = ('train_transform( self, X: np.ndarray, '
			                    'y: Optional[ np.ndarray ]=None ) -> np.ndarray')
			raise exception
			

class CCA( Selector ):
	"""

		Canonical Correlation Analysis (CCA) extracts the ‘directions of covariance’,
		i.e. the components of each data sets that explain the most shared variance
		between both datasets.

	"""
	model: Optional[ object ]
	prediction: Optional[ np.ndarray ]
	n_components: Optional[ int ]
	scale: Optional[ bool ]
	max_iter: Optional[ int ]
	transformed_data: Optional[ Tuple[ np.ndarray, np.ndarray ] ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, num: int = 2, scale: bool = True, size: int = 500 ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the Canonical Correlation Analysis wrapper.
		
			Parameters:
			-----------
			num (int): Number of canonical components to extract.
			scale (bool): Specifies whether X and y should be scaled before
				fitting the model.
			size (int): Maximum number of iterations used by the underlying
				solver.
		
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.scale = scale
		self.n_components = num
		self.max_iter = size
		self.model = CanonicalCorrelationAnalysis( n_components=self.n_components, scale=self.scale,
			max_iter=self.max_iter )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return a list of strings representing class members.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[str]:
				A list of member names.
		
		"""
		return [ 'model',
		         'n_components',
		         'scale',
		         'max_iter',
		         'prediction',
		         'transformed_data',
		         'split_data',
		         'train',
		         'project',
		         'score',
		         'transform',
		         'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
			--------
			Split the specified predictor matrix and target matrix into training
			and testing subsets.
		
			Parameters:
			-----------
			X (np.ndarray): Predictor matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target matrix or vector of shape
				( n_samples, ) or ( n_samples, n_targets ).
			size (float): Proportion of the dataset to include in the test split.
			random (int): Random seed used by the splitter.
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
				A four-item tuple in the form
				( X_train, X_test, y_train, y_test ).
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = ('split_data( self, X: np.ndarray, y: np.ndarray, '
			                    'size: float=0.2, random: int=42 ) -> '
			                    'Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]')
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Predict target values for the specified predictor matrix using the
			fitted CCA model.
		
			Parameters:
			-----------
			X (np.ndarray): Predictor matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector or matrix preserved
				for interface compatibility. It is not used by prediction.
		
			Returns:
			--------
			np.ndarray:
				Predicted target values of shape ( n_samples, ) or
				( n_samples, n_targets ).
		
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Compute summary metrics for the fitted CCA model, including training
			and testing coefficient-of-determination scores and output dimensions.
		
			Parameters:
			-----------
			X (np.ndarray): Predictor matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target matrix or vector of shape
				( n_samples, ) or ( n_samples, n_targets ).
		
			Returns:
			--------
			pd.DataFrame:
				A one-row dataframe containing model summary metrics.
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_training, X_testing, y_training, y_testing = self.split_data( X, y )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			y_pred = self.project( X )
			_target_count = int( y.shape[ 1 ] ) if len( y.shape ) > 1 else 1
			_prediction_count = int( y_pred.shape[ 1 ] ) if len( y_pred.shape ) > 1 else 1
			_metrics = \
			{
				'Components': [ self.n_components ],
				'Training Score': [ float( self.training_score ) ],
				'Testing Score': [ float( self.testing_score ) ],
				'Predictor Count': [ int( X.shape[ 1 ] ) ],
				'Target Count': [ _target_count ],
				'Prediction Count': [ _prediction_count ],
			}
			df_metrics = pd.DataFrame( _metrics )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame'
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""
		
			Purpose:
			--------
			Fit the CCA model to the specified predictor and target data.
		
			Parameters:
			-----------
			X (np.ndarray): Predictor matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target matrix or vector of shape
				( n_samples, ) or ( n_samples, n_targets ).
		
			Returns:
			--------
			object | None:
				The fitted wrapper instance.
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> object | None'
			raise exception
	
	def transform( self, X: np.ndarray,
			y: Optional[ np.ndarray ] = None ) -> Tuple[  np.ndarray, np.ndarray ] | np.ndarray:
		"""
		
			Purpose:
			--------
			Apply the fitted CCA transformation to the specified predictor matrix
			and, when provided, the target matrix.
		
			Parameters:
			-----------
			X (np.ndarray): Predictor matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target matrix of shape
				( n_samples, n_targets ).
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray] | np.ndarray:
				The transformed predictor matrix when y is omitted, or a tuple
				containing transformed X and transformed y when y is provided.
		
		"""
		try:
			throw_if( 'X', X )
			if y is None:
				self.transformed_data = self.model.transform( X )
			else:
				self.transformed_data = self.model.transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = ('transform( self, X: np.ndarray, '
			                    'y: Optional[ np.ndarray ]=None )')
			raise exception
	
	def train_transform( self, X: np.ndarray, y: np.ndarray ) -> Tuple[
		                                                             np.ndarray, np.ndarray ] | np.ndarray:
		"""
		
			Purpose:
			--------
			Fit the CCA model and immediately apply the learned dimension
			reduction to the specified training data.
		
			Parameters:
			-----------
			X (np.ndarray): Predictor matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target matrix or vector of shape
				( n_samples, ) or ( n_samples, n_targets ).
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray] | np.ndarray:
				The transformed predictor matrix and transformed target matrix
				returned by the fitted CCA model.
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'train_transform( self, X: np.ndarray, y: np.ndarray )'
			raise exception
			

class PCA( Selector ):
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
	model: Optional[ object ]
	prediction: Optional[ np.ndarray ]
	svd_solver: Optional[ str ]
	n_components: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, num: int=2, solver: str='auto' ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the PCA wrapper.
		
			Parameters:
			-----------
			num (int): Number of principal components to retain.
			solver (str): Singular value decomposition solver used by the
				underlying PCA implementation.
		
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.n_components = num
		self.svd_solver = solver
		self.model = PrincipalComponentAnalysis( n_components=self.n_components,
			svd_solver=self.svd_solver )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return a list of strings representing class members.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[str]:
				A list of member names.
		
		"""
		return [ 'model',
		         'prediction',
		         'svd_solver',
		         'n_components',
		         'transformed_data',
		         'split_data',
		         'train',
		         'project',
		         'score',
		         'transform',
		         'train_transform' ]
	
	def split_data( self, X: np.ndarray,  y: np.ndarray,  size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
			--------
			Split the specified feature matrix and target vector into training
			and testing subsets.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Proportion of the dataset to include in the test split.
			random (int): Random seed used by the splitter.
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
				A four-item tuple in the form
				( X_train, X_test, y_train, y_test ).
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = ('split_data( self, X: np.ndarray, y: np.ndarray, '
			                    'size: float=0.2, random: int=42 ) -> '
			                    'Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]')
			raise exception
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
		"""
		
			Purpose:
			--------
			Fit the PCA model to the specified feature matrix.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by PCA.
		
			Returns:
			--------
			object | None:
				The fitted wrapper instance.
		
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Project the specified feature matrix into principal component space.
			For this wrapper, projection is equivalent to transformation.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by PCA.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix in principal component space.
		
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Return PCA summary metrics describing dimensionality reduction and
			explained variance for the fitted model.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by PCA.
		
			Returns:
			--------
			pd.DataFrame:
				A one-row dataframe containing PCA summary metrics.
		
		"""
		try:
			throw_if( 'X', X )
			if self.transformed_data is None:
				self.transformed_data = self.transform( X )
			_original_count = int( X.shape[ 1 ] )
			_component_count = int( self.transformed_data.shape[ 1 ] )
			_total_explained = float( np.sum( self.model.explained_variance_ratio_ ) ) \
				if hasattr( self.model, 'explained_variance_ratio_' ) else np.nan
			_metrics = \
			{
					'Original Features': [ _original_count ],
					'Components': [ _component_count ],
					'Explained Variance Total': [ _total_explained ],
					'Largest Component Variance': [
							float( np.max( self.model.explained_variance_ratio_ ) )
							if hasattr( self.model, 'explained_variance_ratio_' )
							else np.nan ],
					'Smallest Component Variance': [
							float( np.min( self.model.explained_variance_ratio_ ) )
							if hasattr( self.model, 'explained_variance_ratio_' )
							else np.nan ],
					'Solver': [ self.svd_solver ],
			}
			df_metrics = pd.DataFrame( _metrics )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Apply the fitted PCA transformation to the specified feature matrix.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by PCA.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix in principal component space.
		
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Fit the PCA model and immediately transform the specified feature
			matrix into principal component space.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by PCA.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix in principal component space.
		
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = ('train_transform( self, X: np.ndarray, '
			                    'y: Optional[ np.ndarray ]=None ) -> np.ndarray')
			raise exception
			

class SelectBest( Selector ):
	"""

		Purpose:
		---------
		A univariate feature selection works by selecting the best features based on univariate
		statistical tests. Removes all but the 'k' highest scoring features
		

	"""
	model: Optional[ object ]
	prediction: Optional[ np.ndarray ]
	score_function: Optional[ object ]
	n_features: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, score_func: object=sf.chi2, num: int=10 ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the SelectKBest wrapper.
		
			Parameters:
			-----------
			score_func (object): Univariate scoring function used to rank
				features.
			num (int): Number of top-ranked features to retain.
		
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.score_function = score_func
		self.n_features = num
		self.model = sf.SelectKBest( score_func=self.score_function, k=self.n_features )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return a list of strings representing class members.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[str]:
				A list of member names.
		
		"""
		return [ 'model',
		         'prediction',
		         'score_function',
		         'n_features',
		         'transformed_data',
		         'split_data',
		         'chi_square',
		         'train',
		         'project',
		         'score',
		         'transform',
		         'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
			--------
			Split the specified feature matrix and target vector into training
			and testing subsets.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Proportion of the dataset to include in the test split.
			random (int): Random seed used by the splitter.
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
				A four-item tuple in the form
				( X_train, X_test, y_train, y_test ).
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = ('split_data( self, X: np.ndarray, y: np.ndarray, '
			                    'size: float=0.2, random: int=42 ) -> '
			                    'Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]')
			raise exception
	
	def chi_square( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ] | None:
		"""
		
			Purpose:
			--------
			Compute chi-square scores and p-values for the specified feature
			matrix and target vector.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray] | None:
				A two-item tuple containing chi-square statistics and p-values.
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return sf.chi2( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'chi_square( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""
		
			Purpose:
			--------
			Fit the SelectKBest selector using the specified feature matrix and
			optional target vector.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Target vector of shape ( n_samples, ).
				This is generally required for supervised scoring functions such
				as chi-square.
		
			Returns:
			--------
			object | None:
				The fitted wrapper instance.
		
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Project the specified feature matrix into the retained feature space.
			For this selector, projection is equivalent to transformation.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used during transformation.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only the selected
				features.
		
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Return selector summary metrics describing feature ranking and
			selection results for the fitted SelectKBest model.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility.
		
			Returns:
			--------
			pd.DataFrame:
				A dataframe containing feature scores, p-values, and selection
				indicators.
		
		"""
		try:
			throw_if( 'X', X )
			if self.transformed_data is None:
				self.transformed_data = self.transform( X )
			_support = self.model.get_support( )
			_scores = self.model.scores_
			_pvalues = self.model.pvalues_ if hasattr( self.model, 'pvalues_' ) else None
			df_scores = pd.DataFrame(
			{
					'Feature': np.arange( 0, X.shape[ 1 ] ),
					'Score': _scores,
					'PValue': _pvalues if _pvalues is not None else np.full(
						X.shape[ 1 ], np.nan ),
					'Selected': _support,
			} )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Apply the fitted SelectKBest selector to the specified feature
			matrix.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used during transformation.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only the selected
				features.
		
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Fit the SelectKBest selector and immediately transform the specified
			feature matrix into the reduced feature space.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Target vector of shape ( n_samples, ).
				This is generally required for supervised scoring functions such
				as chi-square.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only the selected
				features.
		
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = ('train_transform( self, X: np.ndarray, '
			                    'y: Optional[ np.ndarray ]=None ) -> np.ndarray')
			raise exception
			

class SelectPercent( Selector ):
	"""

		Purpose:
		---------
		A univariate feature selection works by selecting the best features based on univariate
		statistical tests. It can be seen as a preprocessing step to an estimator.
		Removes all but a user-specified highest scoring percentage (default - 10%) of features


	"""
	model: Optional[ object ]
	prediction: Optional[ np.ndarray ]
	score_function: Optional[ object ]
	percentile: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, score_func: object = sf.chi2, pct: int = 10 ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the SelectPercentile wrapper.
		
			Parameters:
			-----------
			score_func (object): Univariate scoring function used to rank
				features.
			pct (int): Percentile of top-ranked features to retain.
		
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.score_function = score_func
		self.percentile = pct
		self.model = sf.SelectPercentile( score_func=self.score_function,
			percentile=self.percentile )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return a list of strings representing class members.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[str]:
				A list of member names.
		
		"""
		return [ 'model',
		         'prediction',
		         'score_function',
		         'percentile',
		         'transformed_data',
		         'split_data',
		         'chi_square',
		         'train',
		         'project',
		         'score',
		         'transform',
		         'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
			--------
			Split the specified feature matrix and target vector into training
			and testing subsets.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Proportion of the dataset to include in the test split.
			random (int): Random seed used by the splitter.
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
				A four-item tuple in the form
				( X_train, X_test, y_train, y_test ).
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = ('split_data( self, X: np.ndarray, y: np.ndarray, '
			                    'size: float=0.2, random: int=42 ) -> '
			                    'Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]')
			raise exception
	
	def chi_square( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ] | None:
		"""
		
			Purpose:
			--------
			Compute chi-square scores and p-values for the specified feature
			matrix and target vector.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray] | None:
				A two-item tuple containing chi-square statistics and p-values.
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return sf.chi2( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'chi_square( self, X: np.ndarray, y: np.ndarray )'
			raise exception
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""
		
			Purpose:
			--------
			Fit the SelectPercentile selector using the specified feature matrix
			and optional target vector.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Target vector of shape ( n_samples, ).
				This is generally required for supervised scoring functions such
				as chi-square.
		
			Returns:
			--------
			object | None:
				The fitted wrapper instance.
		
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Project the specified feature matrix into the retained feature space.
			For this selector, projection is equivalent to transformation.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used during transformation.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only the selected
				features.
		
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Return selector summary metrics describing feature ranking and
			selection results for the fitted SelectPercentile model.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility.
		
			Returns:
			--------
			pd.DataFrame:
				A dataframe containing feature scores, p-values, and selection
				indicators.
		
		"""
		try:
			throw_if( 'X', X )
			if self.transformed_data is None:
				self.transformed_data = self.transform( X )
			_support = self.model.get_support( )
			_scores = self.model.scores_
			_pvalues = self.model.pvalues_ if hasattr( self.model, 'pvalues_' ) else None
			df_scores = pd.DataFrame(
			{
				'Feature': np.arange( 0, X.shape[ 1 ] ),
				'Score': _scores,
				'PValue': _pvalues if _pvalues is not None else np.full(
					X.shape[ 1 ], np.nan ),
				'Selected': _support,
			} )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Apply the fitted SelectPercentile selector to the specified feature
			matrix.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used during transformation.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only the selected
				features.
		
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Fit the SelectPercentile selector and immediately transform the
			specified feature matrix into the reduced feature space.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Target vector of shape ( n_samples, ).
				This is generally required for supervised scoring functions such
				as chi-square.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only the selected
				features.
		
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = ('train_transform( self, X: np.ndarray, '
			                    'y: Optional[ np.ndarray ]=None ) -> np.ndarray')
			raise exception
			

class SBS( Selector ):
	'''
	
			Purpose:
			--------
			Implement Sequential Backward Selection (SBS) using a supplied
			classification estimator and scoring function. The algorithm begins with
			the full feature set and greedily removes one feature at a time until the
			desired number of features remains.
			
	'''
	scoring: Optional[ callable ]
	prediction: Optional[ np.ndarray ]
	classifier: Optional[ Classifier ]
	random_state: Optional[ int ]
	test_size: Optional[ float ]
	k_features: Optional[ int ]
	accuracy: Optional[ float ]
	indices_: Optional[ Tuple[ int, ... ] ]
	subsets_: Optional[ List[ Tuple[ int, ... ] ] ]
	scores_: Optional[ List[ float ] ]
	k_score_: Optional[ float ]
	
	def __init__( self, classifier: Classifier, k_features: int, scoring: callable=accuracy_score,
			test_size: float=0.25, random_state: int=1 ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the Sequential Backward Selection wrapper.
		
			Parameters:
			-----------
			classifier (Classifier): Estimator used to evaluate feature subsets.
			k_features (int): Desired number of features to retain.
			scoring (callable): Scoring function used to evaluate predictions.
			test_size (float): Proportion of the dataset reserved for internal
				validation during subset search.
			random_state (int): Random seed used for the internal train/test
				split.
		
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.scoring = scoring
		self.classifier = clone( classifier )
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.indices_ = None
		self.subsets_ = None
		self.scores_ = None
		self.k_score_ = None
	
	def __dir__( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return a list of strings representing class members.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[str]:
				A list of member names.
		
		"""
		return [ 'scoring',
		         'classifier',
		         'k_features',
		         'test_size',
		         'random_state',
		         'prediction',
		         'transformed_data',
		         'indices_',
		         'subsets_',
		         'scores_',
		         'k_score_',
		         'split_data',
		         'train',
		         'project',
		         'score',
		         'transform',
		         'train_transform',
		         'calc_score' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
			--------
			Split the specified feature matrix and target vector into training
			and testing subsets.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Proportion of the dataset to include in the test split.
			random (int): Random seed used by the splitter.
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
				A four-item tuple in the form
				( X_train, X_test, y_train, y_test ).
		
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
			exception.cause = 'SBS'
			exception.method = ('split_data( self, X: np.ndarray, y: np.ndarray, '
			                    'size: float=0.2, random: int=42 ) -> '
			                    'Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]')
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""
		
			Purpose:
			--------
			Fit the Sequential Backward Selection wrapper by iteratively
			evaluating reduced feature subsets until the desired number of
			features remains.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
		
			Returns:
			--------
			object | None:
				The fitted SBS wrapper.
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y,
				test_size=self.test_size,
				random_state=self.random_state )
			dim = X_train.shape[ 1 ]
			self.indices_ = tuple( range( dim ) )
			self.subsets_ = [ self.indices_ ]
			score = self.calc_score( X_train, y_train, X_test, y_test, self.indices_ )
			self.scores_ = [ score ]
			while dim > self.k_features:
				scores = [ ]
				subsets = [ ]
				
				for p in combinations( self.indices_, r=dim - 1 ):
					score = self.calc_score( X_train, y_train, X_test, y_test, p )
					scores.append( score )
					subsets.append( p )
				
				best = int( np.argmax( scores ) )
				self.indices_ = subsets[ best ]
				self.subsets_.append( self.indices_ )
				dim -= 1
				self.scores_.append( scores[ best ] )
			
			self.k_score_ = self.scores_[ -1 ]
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> object | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Project the specified feature matrix onto the selected feature subset.
			For this wrapper, projection is equivalent to transformation.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by this method.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix restricted to the selected
				feature indices.
		
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Return summary metrics describing the selected feature subset and
			the SBS search history.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not required for summary output.
		
			Returns:
			--------
			pd.DataFrame:
				A one-row dataframe containing the number of original features,
				retained features, and the best SBS score.
		
		"""
		try:
			throw_if( 'X', X )
			_original_count = int( X.shape[ 1 ] )
			_retained_count = len( self.indices_ ) if self.indices_ is not None else 0
			_metrics = \
				{
						'Original Features': [ _original_count ],
						'Retained Features': [ _retained_count ],
						'Removed Features': [ _original_count - _retained_count ],
						'Best Score': [
								float( self.k_score_ ) if self.k_score_ is not None else np.nan ],
						'Iterations': [ len( self.subsets_ ) if self.subsets_ is not None else 0 ],
				}
			return pd.DataFrame( _metrics )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Apply the selected feature subset to the specified feature matrix.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility. It is not used by this method.
		
			Returns:
			--------
			np.ndarray:
				The feature matrix restricted to the selected feature indices.
		
		"""
		try:
			throw_if( 'X', X )
			if self.indices_ is None:
				raise ValueError( 'The SBS wrapper must be trained before calling transform.' )
			self.transformed_data = X[ :, self.indices_ ]
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def train_transform( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Fit the SBS wrapper and immediately transform the specified feature
			matrix using the selected subset.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
		
			Returns:
			--------
			np.ndarray:
				The feature matrix restricted to the selected feature indices.
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.train( X, y )
			self.transformed_data = self.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'train_transform( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def calc_score( self, X_train: np.ndarray, y_train: np.ndarray,
			X_test: np.ndarray, y_test: np.ndarray, indices: Tuple[ int, ... ] ) -> float:
		"""
		
			Purpose:
			--------
			Calculate the validation score for a specified feature subset.
		
			Parameters:
			-----------
			X_train (np.ndarray): Training feature matrix.
			y_train (np.ndarray): Training target vector.
			X_test (np.ndarray): Validation feature matrix.
			y_test (np.ndarray): Validation target vector.
			indices (Tuple[int, ...]): Feature indices defining the subset to
				evaluate.
		
			Returns:
			--------
			float:
				The score produced by the configured scoring function.
		
		"""
		try:
			throw_if( 'X_train', X_train )
			throw_if( 'y_train', y_train )
			throw_if( 'X_test', X_test )
			throw_if( 'y_test', y_test )
			throw_if( 'indices', indices )
			self.classifier.fit( X_train[ :, indices ], y_train )
			y_pred = self.classifier.predict( X_test[ :, indices ] )
			score = self.scoring( y_test, y_pred )
			return score
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = ('calc_score( self, X_train: np.ndarray, y_train: np.ndarray, '
			                    'X_test: np.ndarray, y_test: np.ndarray, '
			                    'indices: Tuple[ int, ... ] ) -> float')
			raise exception
			

class RFE( Selector ):
	"""

		Purpose:
		---------
		Recursive Feature Elimination (RFE) Given an external estimator that assigns weights
		to features (e.g., the coefficients of a linear model), recursive feature elimination (RFE)
		is to select features by recursively considering smaller and smaller sets of features.
		
		First, the estimator is trained on the initial set of features and the importance of each
		feature is obtained either through a coef_ attribute or
		through a feature_importances_ attribute. Then, the least important features are pruned
		from current set of features. That procedure is recursively repeated on the pruned set
		until the desired number of features to select is eventually reached.

	"""
	model: Optional[ sf.RFE ]
	prediction: Optional[ np.ndarray ]
	classifier: Optional[ NearestNeighbor ]
	transformed_data: Optional[ np.ndarray ]
	n_features_to_select: Optional[ int ]
	verbose: Optional[ int ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, k_features: int = None, verbose: int = 0 ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the Recursive Feature Elimination wrapper.
		
			Parameters:
			-----------
			k_features (int): Number of features to select. If None, half of the
				features are selected by the underlying estimator.
			verbose (int): Controls verbosity of the underlying RFE estimator.
		
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.n_features_to_select = k_features
		self.classifier = NearestNeighbor( )
		self.verbose = verbose
		self.model = sf.RFE( estimator=self.classifier, n_features_to_select=self.n_features_to_select,
			verbose=self.verbose )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return a list of strings representing class members.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[str]:
				A list of member names.
		
		"""
		return [ 'classifier',
		         'n_features_to_select',
		         'verbose',
		         'prediction',
		         'transformed_data',
		         'features_in',
		         'ranking',
		         'split_data',
		         'train',
		         'project',
		         'score',
		         'transform',
		         'train_transform' ]
	
	@property
	def features_in( self ) -> int:
		"""
		
			Purpose:
			--------
			Return the number of input features seen during fitting.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			int:
				The number of fitted input features.
		
		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def ranking( self ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Return the fitted feature-ranking array, where rank 1 indicates a
			selected feature.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			np.ndarray:
				An array whose entries represent feature rank positions.
		
		"""
		if not hasattr( self.model, 'ranking_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.ranking_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
			--------
			Split the specified feature matrix and target vector into training
			and testing subsets.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
			size (float): Proportion of the dataset to include in the test split.
			random (int): Random seed used by the splitter.
		
			Returns:
			--------
			Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
				A four-item tuple in the form
				( X_train, X_test, y_train, y_test ).
		
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
			exception.cause = 'RFE'
			exception.method = ('split_data( self, X: np.ndarray, y: np.ndarray, '
			                    'size: float=0.2, random: int=42 ) -> '
			                    'Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]')
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""
		
			Purpose:
			--------
			Fit the RFE selector using the specified feature matrix and target
			vector.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
		
			Returns:
			--------
			object | None:
				The fitted wrapper instance.
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> object | None'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Project the specified feature matrix into the retained feature space.
			For this selector, projection is equivalent to transformation.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only retained features.
		
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame:
		"""
		
			Purpose:
			--------
			Return selector summary metrics describing feature ranking and
			selection results for the fitted RFE model.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility.
		
			Returns:
			--------
			pd.DataFrame:
				A dataframe containing feature ranking and selection indicators.
		
		"""
		try:
			throw_if( 'X', X )
			_support = self.model.get_support( )
			df_scores = pd.DataFrame(
				{
						'Feature': np.arange( 0, X.shape[ 1 ] ),
						'Ranking': self.model.ranking_,
						'Selected': _support,
				}
			)
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Apply the fitted RFE transformation to the specified feature matrix.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Optional target vector preserved for
				interface compatibility.
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only retained features.
		
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None )'
			raise exception
	
	def train_transform( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""
		
			Purpose:
			--------
			Fit the RFE selector and immediately transform the specified feature
			matrix into the reduced feature space.
		
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): Target vector of shape ( n_samples, ).
		
			Returns:
			--------
			np.ndarray:
				The transformed feature matrix containing only retained features.
		
		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'train_transform( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray'
			raise exception
	
			
