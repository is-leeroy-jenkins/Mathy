'''
	******************************************************************************************
	  Assembly:                mathy
	  Filename:                clusters.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022
	
	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="clusters.py" company="Terry D. Eppler">
	
	     mathy Clusters
	
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
		clusters.py
	</summary>
	******************************************************************************************
'''
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from boogr import Error
import sklearn.cluster as skc
from sklearn.metrics import (silhouette_score, completeness_score, homogeneity_score,
                             mutual_info_score, v_measure_score)

def throw_if( name: str, value: object ):
	if not value:
		raise ValueError( f'Argument "{name}" cannot be empty!' )


class Cluster( ):
	"""

		Purpose:
		--------
		Abstract base class for clustering wrappers with a uniform interface:
		train → project → score → analyze.

		Methods:
		--------
		train( X ) -> Cluster | None
		project( X ) -> np.ndarray | None
		score( X, y=None ) -> pd.DataFrame | None
		analyze( X, y=None ) -> Dict[ str, Any ] | None

	"""
	n_clusters: Optional[ int ]
	random_state: Optional[ int ]
	max_iter: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self ) -> None:
		pass
	
	def train( self, X: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
			Fit the clustering model to the input samples.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
				object | None: Trained wrapper instance or None.

		"""
		raise NotImplementedError
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Generate cluster assignments for the supplied samples.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None: Cluster labels for each sample.

		"""
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Compute clustering evaluation metrics for the supplied samples.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ) for external clustering metrics.

			Returns:
			--------
				pd.DataFrame | None: DataFrame containing one or more clustering
					evaluation metrics.

		"""
		raise NotImplementedError
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			---------
			Analyze clustering results using visualizations and summary metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ) for comparison against predicted clusters.

			Returns:
			--------
				Dict[ str, Any ] | None: Analysis results or None.

		"""
		raise NotImplementedError


class KMeans( Cluster ):
	"""

		Purpose:
		---------
		The KMeans algorithm clusters stores by trying to separate samples in n groups of equal
		variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares.
		This algorithm requires the number of clusters to be specified.
		It scales well to large number of samples and has been used across a
		large range of application areas in many different fields.

		The algorithm has three steps. The first step chooses the initial centroids,
		with the most basic method being to choose samples from the dataset. After initialization,
		K-means consists of looping between the two other steps. The first step assigns each sample
		to its nearest centroid. The second step creates new centroids by taking the mean value of
		all of the samples assigned to each previous centroid. The difference between the old and
		the new centroids are computed and the algorithm repeats these last two steps until this
		value is less than a threshold. In other words, it repeats until the centroids do not move
		significantly.

	"""
	model: skc.KMeans
	n_clusters: Optional[ int ]
	init: object
	n_init: object
	tolerance: Optional[ float ]
	random_state: Optional[ int ]
	max_iter: Optional[ int ]
	verbose: Optional[ int ]
	copy_x: Optional[ bool ]
	algorithm: Optional[ str ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, clusters: int = 8, init: object = 'k-means++',
			n_init: object = 'auto', tol: float = 0.0001, rando: int | None = 42,
			max_iter: int = 300, verbose: int = 0, copy_x: bool = True,
			algorithm: str = 'lloyd', n_clusters: int | None = None,
			random_state: int | None = None ) -> None:
		"""

			Purpose:
			---------
			Initialize the KMeans clustering wrapper.

			Parameters:
			-----------
				clusters (int): Legacy alias for the number of clusters.
				init (object): Centroid initialization strategy.
				n_init (object): Number of initializations to perform.
				tol (float): Relative convergence tolerance.
				rando (int | None): Legacy alias for random_state.
				max_iter (int): Maximum iterations for a single run.
				verbose (int): Verbosity mode.
				copy_x (bool): Whether to preserve the original input data.
				algorithm (str): KMeans algorithm to use.
				n_clusters (int | None): Explicit scikit-learn style cluster count.
				random_state (int | None): Explicit scikit-learn style random state.

			Returns:
			--------
				None

		"""
		super( ).__init__( )
		self.n_clusters = n_clusters if n_clusters is not None else clusters
		self.init = init
		self.n_init = n_init
		self.random_state = random_state if random_state is not None else rando
		self.max_iter = max_iter
		self.tolerance = tol
		self.verbose = verbose
		self.copy_x = copy_x
		self.algorithm = algorithm
		self.model = skc.KMeans(
			n_clusters=self.n_clusters,
			init=self.init,
			n_init=self.n_init,
			max_iter=self.max_iter,
			tol=self.tolerance,
			verbose=self.verbose,
			random_state=self.random_state,
			copy_x=self.copy_x,
			algorithm=self.algorithm
		)
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the primary public members exposed by the wrapper.

			Parameters:
			-----------
				None

			Returns:
			--------
				list[ str ]:
					Member names.

		"""
		return [
				'model',
				'train',
				'score',
				'project',
				'transform',
				'analyze',
				'n_clusters',
				'init',
				'n_init',
				'random_state',
				'tolerance',
				'max_iter',
				'verbose',
				'copy_x',
				'algorithm',
				'clusters',
				'centroids_',
				'labels',
				'inertia',
				'silouette',
				'homogeneity',
				'mutual_info',
				'v_measure',
				'iterations',
				'features',
				'prediction'
		]
	
	@property
	def clusters( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster centers.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster centers.

		"""
		if not hasattr( self.model, 'cluster_centers_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.cluster_centers_
	
	@property
	def centroids_( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster centers using the name expected by app.py.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster centers.

		"""
		return self.clusters
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted sample labels.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster labels.

		"""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def inertia( self ) -> float:
		"""

			Purpose:
			---------
			Return the fitted within-cluster sum of squares.

			Parameters:
			-----------
				None

			Returns:
			--------
				float:
					Cluster inertia.

		"""
		if not hasattr( self.model, 'inertia_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.inertia_
	
	@property
	def iterations( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of iterations run during fitting.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Number of iterations.

		"""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during fitting.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Number of fitted features.

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> KMeans | None:
		"""

			Purpose:
			---------
			Fit the KMeans model on the supplied data.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ).

			Returns:
			--------
				KMeans | None:
					Trained wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'train( self, X: np.ndarray ) -> KMeans | None'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Generate cluster assignments for the supplied samples.

			If the estimator has not yet been fitted, this method fits and predicts
			in a single step so it remains compatible with the current app.py
			clustering execution path.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None:
					Cluster labels for each sample.

		"""
		try:
			throw_if( 'X', X )
			
			if hasattr( self.model, 'cluster_centers_' ):
				self.prediction = self.model.predict( X )
			else:
				self.prediction = self.model.fit_predict( X )
			
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Return distances from samples to fitted cluster centers.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None:
					Distance matrix.

		"""
		try:
			throw_if( 'X', X )
			
			if not hasattr( self.model, 'cluster_centers_' ):
				self.model.fit( X )
				self.prediction = self.model.labels_
			
			return self.model.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Evaluate KMeans clustering performance using intrinsic and optional
			external clustering metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				pd.DataFrame | None:
					DataFrame containing clustering metrics.

		"""
		try:
			throw_if( 'X', X )
			
			if hasattr( self.model, 'cluster_centers_' ):
				labels = self.model.predict( X )
			else:
				labels = self.model.fit_predict( X )
			
			self.prediction = labels
			scores = { }
			
			unique_labels = np.unique( labels )
			if len( unique_labels ) >= 2 and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			scores[ 'Inertia' ] = self.inertia
			scores[ 'Iterations' ] = self.iterations
			scores[ 'Clusters' ] = len( unique_labels )
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = \
				'score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			---------
			Produce a summary analysis payload for the fitted clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels.

			Returns:
			--------
				Dict[ str, Any ] | None:
					Analysis details and metrics.

		"""
		try:
			throw_if( 'X', X )
			
			df_score = self.score( X, y )
			return {
					'labels': self.prediction,
					'centroids': self.centroids_,
					'inertia': self.inertia,
					'iterations': self.iterations,
					'features': self.features,
					'score': df_score
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = \
				'analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None'
			raise exception


class DBSCAN( Cluster ):
	"""

		Purpose:
		---------
		The DBSCAN algorithm views clusters as areas of high density separated by areas of low
		density. Due to this rather generic view, clusters found by DBSCAN can be any shape,
		as opposed to k-means which assumes that clusters are convex shaped. The central component
		to the DBSCAN is the concept of core samples, which are samples that are in areas of high
		density.

		A cluster is therefore a set of core samples, each close to each other (measured
		by some distance measure) and a set of non-core samples that are close to a core sample
		(but are not themselves core samples). There are two parameters to the algorithm,
		min_samples and eps, which define formally what we mean when we say dense. Higher
		min_samples or lower eps indicate higher density necessary to form a cluster.

	"""
	model: skc.DBSCAN
	epsilon: Optional[ float ]
	min_samples: Optional[ int ]
	metric: object
	metric_params: Optional[ Dict[ str, Any ] ]
	algorithm: Optional[ str ]
	leaf_size: Optional[ int ]
	p: Optional[ float ]
	n_jobs: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, eps: float = 0.5, samples: int = 5,
			metric: object = 'euclidean',
			metric_params: Dict[ str, Any ] | None = None,
			algorithm: str = 'auto', leaf_size: int = 30,
			p: float | None = None, n_jobs: int | None = None,
			min_samples: int | None = None ) -> None:
		"""

			Purpose:
			---------
			Initialize the DBSCAN clustering wrapper.

			Parameters:
			-----------
				eps (float): Maximum neighborhood distance.
				samples (int): Legacy alias for min_samples.
				metric (object): Distance metric name or callable.
				metric_params (Dict[ str, Any ] | None): Additional metric
					keyword arguments.
				algorithm (str): Neighbor search algorithm.
				leaf_size (int): Leaf size for BallTree or KDTree.
				p (float | None): Power parameter for the Minkowski metric.
				n_jobs (int | None): Number of parallel jobs.
				min_samples (int | None): Explicit scikit-learn style
					min_samples value.

			Returns:
			--------
				None

		"""
		super( ).__init__( )
		self.epsilon = eps
		self.min_samples = min_samples if min_samples is not None else samples
		self.metric = metric
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.n_jobs = n_jobs
		self.model = skc.DBSCAN(
			eps=self.epsilon,
			min_samples=self.min_samples,
			metric=self.metric,
			metric_params=self.metric_params,
			algorithm=self.algorithm,
			leaf_size=self.leaf_size,
			p=self.p,
			n_jobs=self.n_jobs
		)
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the primary public members exposed by the wrapper.

			Parameters:
			-----------
				None

			Returns:
			--------
				list[ str ]:
					Member names.

		"""
		return [
				'model',
				'train',
				'score',
				'project',
				'analyze',
				'epsilon',
				'eps',
				'min_samples',
				'metric',
				'metric_params',
				'algorithm',
				'leaf_size',
				'p',
				'n_jobs',
				'labels',
				'core_samples',
				'components',
				'features',
				'prediction',
				'silouette',
				'homogeneity',
				'mutual_info',
				'v_measure',
				'completeness'
		]
	
	@property
	def eps( self ) -> float:
		"""

			Purpose:
			---------
			Return the configured epsilon value using sklearn naming.

			Parameters:
			-----------
				None

			Returns:
			--------
				float:
					Epsilon value.

		"""
		return self.epsilon
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster labels.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster labels.

		"""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def core_samples( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the indices of core samples.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Core sample indices.

		"""
		if not hasattr( self.model, 'core_sample_indices_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.core_sample_indices_
	
	@property
	def components( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the fitted core sample feature vectors.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Core sample components.

		"""
		if not hasattr( self.model, 'components_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.components_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during fitting.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Number of fitted features.

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> DBSCAN | None:
		"""

			Purpose:
			---------
			Fit the DBSCAN model on the supplied data.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ).

			Returns:
			--------
				DBSCAN | None:
					Trained wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = 'train( self, X: np.ndarray ) -> DBSCAN | None'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Generate cluster assignments for the supplied samples.

			DBSCAN does not expose predict for unseen samples, so this method
			fits and returns labels for the supplied data.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None:
					Cluster labels for each sample.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Evaluate DBSCAN clustering performance using intrinsic and optional
			external clustering metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				pd.DataFrame | None:
					DataFrame containing clustering metrics.

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			
			unique_labels = np.unique( labels )
			cluster_labels = unique_labels[ unique_labels != -1 ]
			scores[ 'Clusters' ] = len( cluster_labels )
			scores[ 'Noise' ] = int( np.sum( labels == -1 ) )
			
			if len( cluster_labels ) >= 2:
				mask = labels != -1
				if np.sum( mask ) >= 2 and len( np.unique( labels[ mask ] ) ) >= 2:
					self.silouette = silhouette_score( X[ mask ], labels[ mask ] )
					scores[ 'Silouette' ] = self.silouette
				else:
					self.silouette = np.nan
					scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = \
				'score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			---------
			Produce a summary analysis payload for the fitted clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels.

			Returns:
			--------
				Dict[ str, Any ] | None:
					Analysis details and metrics.

		"""
		try:
			throw_if( 'X', X )
			df_score = self.score( X, y )
			
			core_samples = self.core_samples if hasattr(
				self.model, 'core_sample_indices_' ) else np.array( [ ] )
			components = self.components if hasattr(
				self.model, 'components_' ) else np.empty( (0, X.shape[ 1 ]) )
			
			return {
					'labels': self.prediction,
					'core_samples': core_samples,
					'components': components,
					'features': self.features,
					'score': df_score
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = \
				'analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None'
			raise exception


class Agglomerative( Cluster ):
	"""

		Purpose:
		---------
		The Agglomerative Cluster object performs a hierarchical clustering using a
		bottom up approach: each observation starts in its own cluster, and clusters are
		successively merged together. The linkage criteria determines the metric used for the merge
		strategy:

		'Minimizes' the sum of squared differences within all clusters. It is a
		variance-minimizing approach and in this sense is similar to the k-means objective
		function but tackled with an agglomerative hierarchical approach.

		'Maximum' or complete linkage minimizes the maximum distance between observations of
		pairs of clusters. Average linkage minimizes the average of the distances between all observations of
		pairs of clusters.

		'Single' linkage minimizes the distance between the closest observations of pairs of
		clusters. Agglomerative Cluster can also scale to large number of samples when it is used jointly
		with a connectivity matrix, but is computationally expensive when no connectivity
		constraints are added between samples: it considers at each step all the possible merges.

	"""
	model: skc.AgglomerativeClustering
	n_clusters: Optional[ int ]
	metric: object
	affinity: object
	memory: object
	connectivity: object
	compute_full_tree: object
	linkage: Optional[ str ]
	distance_threshold: Optional[ float ]
	compute_distances: Optional[ bool ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, clusters: int = 2, affinity: object = 'euclidean',
			linkage: str = 'ward', metric: object | None = None,
			memory: object = None, connectivity: object = None,
			compute_full_tree: object = 'auto',
			distance_threshold: float | None = None,
			compute_distances: bool = False,
			n_clusters: int | None = None ) -> None:
		"""

			Purpose:
			---------
			Initialize the Agglomerative clustering wrapper.

			Parameters:
			-----------
				clusters (int): Legacy alias for the number of clusters.
				affinity (object): Legacy alias for metric.
				linkage (str): Linkage criterion.
				metric (object | None): Distance metric name or callable.
				memory (object): Optional joblib memory or cache path.
				connectivity (object): Optional connectivity constraint.
				compute_full_tree (object): Whether to compute the full tree.
				distance_threshold (float | None): Linkage distance threshold.
				compute_distances (bool): Whether to compute distances_.
				n_clusters (int | None): Explicit scikit-learn style cluster count.

			Returns:
			--------
				None

		"""
		super( ).__init__( )
		self.n_clusters = n_clusters if n_clusters is not None else clusters
		self.metric = metric if metric is not None else affinity
		self.affinity = self.metric
		self.memory = memory
		self.connectivity = connectivity
		self.compute_full_tree = compute_full_tree
		self.linkage = linkage
		self.distance_threshold = distance_threshold
		self.compute_distances = compute_distances
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
		self.model = self.create_model( )
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the primary public members exposed by the wrapper.

			Parameters:
			-----------
				None

			Returns:
			--------
				list[ str ]:
					Member names.

		"""
		return [
				'model',
				'train',
				'score',
				'project',
				'analyze',
				'create_model',
				'n_clusters',
				'metric',
				'affinity',
				'memory',
				'connectivity',
				'compute_full_tree',
				'linkage',
				'distance_threshold',
				'compute_distances',
				'labels',
				'children',
				'distances',
				'leaves',
				'connected_components',
				'features',
				'prediction',
				'silouette',
				'homogeneity',
				'mutual_info',
				'v_measure',
				'completeness'
		]
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster labels.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster labels.

		"""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def children( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the hierarchical merge structure.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Merge tree children.

		"""
		if not hasattr( self.model, 'children_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.children_
	
	@property
	def distances( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the fitted linkage distances when available.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Distances between merged nodes.

		"""
		if not hasattr( self.model, 'distances_' ):
			raise AttributeError( 'The model distances are not available!' )
		return self.model.distances_
	
	@property
	def leaves( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of leaves in the hierarchical tree.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Number of leaves.

		"""
		if not hasattr( self.model, 'n_leaves_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_leaves_
	
	@property
	def connected_components( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of connected components.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Connected components.

		"""
		if not hasattr( self.model, 'n_connected_components_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_connected_components_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during fitting.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Number of fitted features.

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def create_model( self ) -> skc.AgglomerativeClustering:
		"""

			Purpose:
			---------
			Construct a validated AgglomerativeClustering estimator using the
			current wrapper state.

			Parameters:
			-----------
				None

			Returns:
			--------
				skc.AgglomerativeClustering:
					Configured estimator.

		"""
		try:
			n_clusters = self.n_clusters
			compute_full_tree = self.compute_full_tree
			
			if self.linkage == 'ward' and self.metric != 'euclidean':
				raise ValueError(
					'When linkage is "ward", metric must be "euclidean".'
				)
			
			if self.distance_threshold is not None:
				n_clusters = None
				compute_full_tree = True
			
			return skc.AgglomerativeClustering(
				n_clusters=n_clusters,
				metric=self.metric,
				memory=self.memory,
				connectivity=self.connectivity,
				compute_full_tree=compute_full_tree,
				linkage=self.linkage,
				distance_threshold=self.distance_threshold,
				compute_distances=self.compute_distances
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = \
				'create_model( self ) -> skc.AgglomerativeClustering'
			raise exception
	
	def train( self, X: np.ndarray ) -> Agglomerative | None:
		"""

			Purpose:
			---------
			Fit the Agglomerative clustering model on the supplied data.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ) or a distance matrix when using
					metric='precomputed'.

			Returns:
			--------
				Agglomerative | None:
					Trained wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model = self.create_model( )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = \
				'train( self, X: np.ndarray ) -> Agglomerative | None'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Generate cluster assignments for the supplied samples.

			AgglomerativeClustering does not expose predict for unseen samples,
			so this method fits and returns labels for the supplied data.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ) or a distance matrix when using
					metric='precomputed'.

			Returns:
			--------
				np.ndarray | None:
					Cluster labels for each sample.

		"""
		try:
			throw_if( 'X', X )
			self.model = self.create_model( )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = \
				'project( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Evaluate Agglomerative clustering performance using intrinsic and
			optional external clustering metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ) or a distance matrix when using
					metric='precomputed'.
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				pd.DataFrame | None:
					DataFrame containing clustering metrics.

		"""
		try:
			throw_if( 'X', X )
			self.model = self.create_model( )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			
			unique_labels = np.unique( labels )
			scores[ 'Clusters' ] = len( unique_labels )
			scores[ 'Leaves' ] = self.leaves
			scores[ 'Connected-Components' ] = self.connected_components
			
			if self.metric != 'precomputed' and len( unique_labels ) >= 2 \
					and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			if hasattr( self.model, 'distances_' ):
				scores[ 'Distances-Computed' ] = True
			else:
				scores[ 'Distances-Computed' ] = False
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = \
				'score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			---------
			Produce a summary analysis payload for the fitted clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape
					( n_samples, n_features ) or a distance matrix when using
					metric='precomputed'.
				y (Optional[np.ndarray]): Optional reference labels.

			Returns:
			--------
				Dict[ str, Any ] | None:
					Analysis details and metrics.

		"""
		try:
			throw_if( 'X', X )
			df_score = self.score( X, y )
			
			distances = self.distances if hasattr(
				self.model, 'distances_' ) else np.array( [ ] )
			
			return {
					'labels': self.prediction,
					'children': self.children,
					'distances': distances,
					'leaves': self.leaves,
					'connected_components': self.connected_components,
					'features': self.features,
					'score': df_score
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = \
				'analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None'
			raise exception


class Spectral( Cluster ):
	"""

		Purpose:
		---------
		Spectral Cluster does a low-dimension embedding of the affinity matrix between samples,
		followed by a KMeans in the low dimensional space. It is especially efficient if the
		affinity matrix is sparse and the pyamg module is installed. SpectralCluster requires
		the number of clusters to be specified. It works well for a small number of clusters but
		is not advised when using many clusters.

		For two clusters, it solves a convex relaxation of the normalised cuts problem on the
		similarity graph: cutting the graph in two so that the weight of the edges cut is small
		compared to the weights of the edges inside each cluster. This criteria is especially
		interesting when working on images: graph vertices are pixels, and edges of the similarity
		graph are a function of the gradient of the image.

	"""
	model: skc.SpectralClustering
	n_clusters: Optional[ int ]
	eigen_solver: Optional[ str ]
	n_components: Optional[ int ]
	random_state: Optional[ int ]
	n_init: Optional[ int ]
	gamma: Optional[ float ]
	affinity: object
	n_neighbors: Optional[ int ]
	eigen_tolerance: object
	assign_labels: Optional[ str ]
	degree: Optional[ float ]
	coef0: Optional[ float ]
	kernel_params: Optional[ Dict[ str, Any ] ]
	n_jobs: Optional[ int ]
	verbose: Optional[ bool ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, clusters: int = 8, random_state: int | None = 42,
			n_init: int = 10, gama: float = 1.0, distance: object = 'rbf',
			neighbors: int = 10, tolerance: object = 'auto',
			assign: str = 'kmeans', eigen_solver: str | None = None,
			n_components: int | None = None, degree: float = 3,
			coef0: float = 1, kernel_params: Dict[ str, Any ] | None = None,
			n_jobs: int | None = None, verbose: bool = False,
			n_clusters: int | None = None, gamma: float | None = None,
			affinity: object | None = None, n_neighbors: int | None = None,
			eigen_tol: object | None = None,
			assign_labels: str | None = None ) -> None:
		"""

			Purpose:
			---------
			Initialize the Spectral clustering wrapper.

			Parameters:
			-----------
				clusters (int): Legacy alias for the number of clusters.
				random_state (int | None): Random seed for reproducibility.
				n_init (int): Number of initializations for label assignment.
				gama (float): Legacy alias for gamma.
				distance (object): Legacy alias for affinity.
				neighbors (int): Legacy alias for n_neighbors.
				tolerance (object): Legacy alias for eigen_tol.
				assign (str): Legacy alias for assign_labels.
				eigen_solver (str | None): Eigenvalue decomposition strategy.
				n_components (int | None): Number of eigenvectors to use.
				degree (float): Degree for polynomial kernels.
				coef0 (float): Zero coefficient for poly and sigmoid kernels.
				kernel_params (Dict[ str, Any ] | None): Parameters for a
					callable kernel.
				n_jobs (int | None): Number of parallel jobs.
				verbose (bool): Verbose mode.
				n_clusters (int | None): Explicit scikit-learn style cluster count.
				gamma (float | None): Explicit scikit-learn style gamma value.
				affinity (object | None): Explicit scikit-learn style affinity.
				n_neighbors (int | None): Explicit scikit-learn style neighbor count.
				eigen_tol (object | None): Explicit scikit-learn style eigen tolerance.
				assign_labels (str | None): Explicit scikit-learn style label assignment.

			Returns:
			--------
				None

		"""
		super( ).__init__( )
		self.n_clusters = n_clusters if n_clusters is not None else clusters
		self.eigen_solver = eigen_solver
		self.n_components = n_components
		self.random_state = random_state
		self.n_init = n_init
		self.gamma = gamma if gamma is not None else gama
		self.affinity = affinity if affinity is not None else distance
		self.n_neighbors = n_neighbors if n_neighbors is not None else neighbors
		self.eigen_tolerance = eigen_tol if eigen_tol is not None else tolerance
		self.assign_labels = assign_labels if assign_labels is not None else assign
		self.degree = degree
		self.coef0 = coef0
		self.kernel_params = kernel_params
		self.n_jobs = n_jobs
		self.verbose = verbose
		self.model = skc.SpectralClustering(
			n_clusters=self.n_clusters,
			eigen_solver=self.eigen_solver,
			n_components=self.n_components,
			random_state=self.random_state,
			n_init=self.n_init,
			gamma=self.gamma,
			affinity=self.affinity,
			n_neighbors=self.n_neighbors,
			eigen_tol=self.eigen_tolerance,
			assign_labels=self.assign_labels,
			degree=self.degree,
			coef0=self.coef0,
			kernel_params=self.kernel_params,
			n_jobs=self.n_jobs,
			verbose=self.verbose
		)
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the primary public members exposed by the wrapper.

			Parameters:
			-----------
				None

			Returns:
			--------
				list[ str ]:
					Member names.

		"""
		return [
				'model',
				'n_clusters',
				'eigen_solver',
				'n_components',
				'random_state',
				'n_init',
				'gamma',
				'affinity',
				'n_neighbors',
				'eigen_tolerance',
				'assign_labels',
				'degree',
				'coef0',
				'kernel_params',
				'n_jobs',
				'verbose',
				'labels',
				'features',
				'train',
				'score',
				'project',
				'analyze',
				'silouette',
				'homogeneity',
				'mutual_info',
				'v_measure',
				'completeness',
				'prediction'
		]
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster labels.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster labels.

		"""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during fitting.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Number of fitted features.

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> Spectral | None:
		"""

			Purpose:
			---------
			Fit the Spectral clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				Spectral | None:
					Trained wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = 'train( self, X: np.ndarray ) -> Spectral | None'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Generate cluster assignments for the supplied samples.

			SpectralClustering exposes fit_predict for the supplied data, so this
			method fits and returns labels for the supplied samples.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None:
					Cluster labels for each sample.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Evaluate spectral clustering performance using intrinsic and optional
			external clustering metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				pd.DataFrame | None:
					DataFrame containing clustering metrics.

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			
			unique_labels = np.unique( labels )
			scores[ 'Clusters' ] = len( unique_labels )
			
			if len( unique_labels ) >= 2 and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = \
				'score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			---------
			Produce a summary analysis payload for the fitted clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				Dict[ str, Any ] | None:
					Dictionary containing analysis results.

		"""
		try:
			throw_if( 'X', X )
			
			if self.prediction is None or len( self.prediction ) != len( X ):
				labels = self.model.fit_predict( X )
				self.prediction = labels
			else:
				labels = self.prediction
			
			df_score = self.score( X, y )
			
			return {
					'Labels': labels,
					'Clusters': self.n_clusters,
					'Eigen-Solver': self.eigen_solver,
					'N-Components': self.n_components,
					'Random-State': self.random_state,
					'N-Init': self.n_init,
					'Gamma': self.gamma,
					'Affinity': self.affinity,
					'N-Neighbors': self.n_neighbors,
					'Eigen-Tolerance': self.eigen_tolerance,
					'Assign-Labels': self.assign_labels,
					'Degree': self.degree,
					'Coef0': self.coef0,
					'N-Jobs': self.n_jobs,
					'Verbose': self.verbose,
					'Features': self.features,
					'Score': df_score
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = \
				'analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None'
			raise exception


class MeanShift( Cluster ):
	"""

		Purpose:
		---------
		Mean Shift clustering aims to discover blobs in a smooth density of samples.
		It is a centroid based algorithm, which works by updating candidates for centroids to be
		the mean of the points within a given region. These candidates are then filtered in a
		post-processing stage to eliminate near-duplicates to form the final set of centroids.

		The algorithm automatically sets the number of clusters, instead of relying on a parameter
		bandwidth, which dictates the size of the region to search through. This parameter can be
		set manually, but can be estimated using the provided estimate_bandwidth function, which
		is called if the bandwidth is not set.

		The algorithm is not highly scalable, as it requires multiple nearest neighbor searches
		during the execution of the algorithm. The algorithm is guaranteed to converge,
		however the algorithm will stop iterating when the change in centroids is small.

	"""
	model: skc.MeanShift
	bandwidth: Optional[ float ]
	seeds: Optional[ np.ndarray ]
	bin_seeding: Optional[ bool ]
	min_bin_freq: Optional[ int ]
	cluster_all: Optional[ bool ]
	n_jobs: Optional[ int ]
	max_iter: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, min_bin: int = 1, group_all: bool = True,
			bandwidth: float | None = None, seeds: np.ndarray | None = None,
			bin_seeding: bool = False, n_jobs: int | None = None,
			max_iter: int = 300, min_bin_freq: int | None = None,
			cluster_all: bool | None = None ) -> None:
		"""

			Purpose:
			---------
			Initialize the MeanShift clustering wrapper.

			Parameters:
			-----------
				min_bin (int): Legacy alias for min_bin_freq.
				group_all (bool): Legacy alias for cluster_all.
				bandwidth (float | None): Bandwidth used in the flat kernel.
				seeds (np.ndarray | None): Seed points used to initialize kernels.
				bin_seeding (bool): Whether to initialize kernels from binned seeds.
				n_jobs (int | None): Number of parallel jobs.
				max_iter (int): Maximum iterations per seed point.
				min_bin_freq (int | None): Explicit scikit-learn style minimum
					bin frequency.
				cluster_all (bool | None): Explicit scikit-learn style
					cluster-all behavior.

			Returns:
			--------
				None

		"""
		super( ).__init__( )
		self.bandwidth = bandwidth
		self.seeds = seeds
		self.bin_seeding = bin_seeding
		self.min_bin_freq = min_bin_freq if min_bin_freq is not None else min_bin
		self.cluster_all = cluster_all if cluster_all is not None else group_all
		self.n_jobs = n_jobs
		self.max_iter = max_iter
		self.model = skc.MeanShift(
			bandwidth=self.bandwidth,
			seeds=self.seeds,
			bin_seeding=self.bin_seeding,
			min_bin_freq=self.min_bin_freq,
			cluster_all=self.cluster_all,
			n_jobs=self.n_jobs,
			max_iter=self.max_iter
		)
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the primary public members exposed by the wrapper.

			Parameters:
			-----------
				None

			Returns:
			--------
				list[ str ]:
					Member names.

		"""
		return [
				'model',
				'train',
				'score',
				'project',
				'predict',
				'analyze',
				'bandwidth',
				'seeds',
				'bin_seeding',
				'min_bin_freq',
				'cluster_all',
				'n_jobs',
				'max_iter',
				'labels',
				'clusters',
				'centroids_',
				'iterations',
				'features',
				'silouette',
				'homogeneity',
				'mutual_info',
				'v_measure',
				'completeness',
				'prediction'
		]
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster labels.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster labels.

		"""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def clusters( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster centers.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster centers.

		"""
		if not hasattr( self.model, 'cluster_centers_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.cluster_centers_
	
	@property
	def centroids_( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster centers using the name expected by app.py.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster centers.

		"""
		return self.clusters
	
	@property
	def iterations( self ) -> int:
		"""

			Purpose:
			---------
			Return the maximum number of iterations performed on each seed.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Iteration count.

		"""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during fitting.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Number of fitted features.

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> MeanShift | None:
		"""

			Purpose:
			---------
			Fit the MeanShift clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				MeanShift | None:
					Trained wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'train( self, X: np.ndarray ) -> MeanShift | None'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Generate cluster assignments for the supplied samples.

			MeanShift exposes fit_predict for the supplied data, so this
			method fits and returns labels for the supplied samples.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None:
					Cluster labels for each sample.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict the closest cluster each sample in X belongs to.

			Parameters:
			-----------
				X (np.ndarray): New samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None:
					Cluster labels for each sample.

		"""
		try:
			throw_if( 'X', X )
			
			if not hasattr( self.model, 'cluster_centers_' ):
				raise AttributeError( 'The model data has not been trained!' )
			
			return self.model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Evaluate MeanShift clustering performance using intrinsic and optional
			external clustering metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				pd.DataFrame | None:
					DataFrame containing clustering metrics.

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			
			unique_labels = np.unique( labels )
			scores[ 'Clusters' ] = len( unique_labels )
			scores[ 'Iterations' ] = self.iterations
			
			if len( unique_labels ) >= 2 and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = \
				'score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			---------
			Produce a summary analysis payload for the fitted clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				Dict[ str, Any ] | None:
					Dictionary containing analysis results.

		"""
		try:
			throw_if( 'X', X )
			
			if self.prediction is None or len( self.prediction ) != len( X ):
				labels = self.model.fit_predict( X )
				self.prediction = labels
			else:
				labels = self.prediction
			
			df_score = self.score( X, y )
			
			return {
					'Labels': labels,
					'Bandwidth': self.bandwidth,
					'Bin-Seeding': self.bin_seeding,
					'Min-Bin-Freq': self.min_bin_freq,
					'Cluster-All': self.cluster_all,
					'N-Jobs': self.n_jobs,
					'Max-Iter': self.max_iter,
					'Iterations': self.iterations,
					'Centroids': self.centroids_,
					'Features': self.features,
					'Score': df_score
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = \
				'analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None'
			raise exception


class AffinityPropagation( Cluster ):
	"""

		Purpose:
		---------
		Affinity Propagation creates clusters by sending messages between pairs of samples until
		convergence. A dataset is then described using a small number of exemplars, which are
		identified as those most representative of other samples. The messages sent between pairs
		represent the suitability for one sample to be the exemplar of the other, which is updated
		in response to the values from other pairs. This updating happens iteratively until
		convergence, at which point the final exemplars are chosen,
		and hence the final clustering is given.

	"""
	model: skc.AffinityPropagation
	damping: Optional[ float ]
	max_iter: Optional[ int ]
	convergence_iter: Optional[ int ]
	copy: Optional[ bool ]
	preference: object
	affinity: Optional[ str ]
	verbose: Optional[ bool ]
	random_state: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, damping: float = 0.5, max_iter: int = 200,
			convergence_iter: int = 15, preference: object = None,
			affinity: str = 'euclidean', copy: bool = True,
			verbose: bool = False, random_state: int | None = None ) -> None:
		"""

			Purpose:
			---------
			Initialize the AffinityPropagation clustering wrapper.

			Parameters:
			-----------
				damping (float): Damping factor in the range [0.5, 1.0).
				max_iter (int): Maximum number of iterations.
				convergence_iter (int): Convergence window size.
				preference (object): Preferences for each sample or a single
					float preference value.
				affinity (str): Similarity measure to use.
				copy (bool): Whether to copy the input data.
				verbose (bool): Whether to enable verbose output.
				random_state (int | None): Random state for reproducibility.

			Returns:
			--------
				None

		"""
		super( ).__init__( )
		self.damping = damping
		self.max_iter = max_iter
		self.convergence_iter = convergence_iter
		self.copy = copy
		self.preference = preference
		self.affinity = affinity
		self.verbose = verbose
		self.random_state = random_state
		self.model = skc.AffinityPropagation(
			damping=self.damping,
			max_iter=self.max_iter,
			convergence_iter=self.convergence_iter,
			copy=self.copy,
			preference=self.preference,
			affinity=self.affinity,
			verbose=self.verbose,
			random_state=self.random_state
		)
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the primary public members exposed by the wrapper.

			Parameters:
			-----------
				None

			Returns:
			--------
				list[ str ]:
					Member names.

		"""
		return [
				'model',
				'train',
				'score',
				'project',
				'predict',
				'analyze',
				'damping',
				'max_iter',
				'convergence_iter',
				'copy',
				'preference',
				'affinity',
				'verbose',
				'random_state',
				'labels',
				'clusters',
				'centroids_',
				'affinity_matrix',
				'iterations',
				'features',
				'prediction',
				'silouette',
				'homogeneity',
				'mutual_info',
				'v_measure',
				'completeness'
		]
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster labels.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster labels.

		"""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def clusters( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster centers.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster centers.

		"""
		if not hasattr( self.model, 'cluster_centers_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.cluster_centers_
	
	@property
	def centroids_( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster centers using the name expected by app.py.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster centers.

		"""
		return self.clusters
	
	@property
	def affinity_matrix( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the learned affinity matrix.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Affinity matrix.

		"""
		if not hasattr( self.model, 'affinity_matrix_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.affinity_matrix_
	
	@property
	def iterations( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of iterations run by the estimator.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Iteration count.

		"""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during fitting.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Number of fitted features.

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> AffinityPropagation | None:
		"""

			Purpose:
			---------
			Fit the AffinityPropagation clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ) or similarity matrix when
					affinity='precomputed'.

			Returns:
			--------
				AffinityPropagation | None:
					Trained wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = \
				'train( self, X: np.ndarray ) -> AffinityPropagation | None'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Generate cluster assignments for the supplied samples.

			AffinityPropagation exposes fit_predict for the supplied data, so this
			method fits and returns labels for the supplied samples.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ) or similarity matrix when
					affinity='precomputed'.

			Returns:
			--------
				np.ndarray | None:
					Cluster labels for each sample.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = \
				'project( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict the closest cluster each sample in X belongs to.

			Parameters:
			-----------
				X (np.ndarray): New samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None:
					Cluster labels for each sample.

		"""
		try:
			throw_if( 'X', X )
			
			if not hasattr( self.model, 'cluster_centers_' ):
				raise AttributeError( 'The model data has not been trained!' )
			
			return self.model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = \
				'predict( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Evaluate AffinityPropagation clustering performance using intrinsic
			and optional external clustering metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ) or similarity matrix when
					affinity='precomputed'.
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				pd.DataFrame | None:
					DataFrame containing clustering metrics.

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			
			unique_labels = np.unique( labels )
			scores[ 'Clusters' ] = len( unique_labels )
			scores[ 'Iterations' ] = self.iterations
			
			if self.affinity != 'precomputed' and len( unique_labels ) >= 2 \
					and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = \
				'score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			---------
			Produce a summary analysis payload for the fitted clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ) or similarity matrix when
					affinity='precomputed'.
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				Dict[ str, Any ] | None:
					Dictionary containing analysis results.

		"""
		try:
			throw_if( 'X', X )
			
			if self.prediction is None or len( self.prediction ) != len( X ):
				labels = self.model.fit_predict( X )
				self.prediction = labels
			else:
				labels = self.prediction
			
			df_score = self.score( X, y )
			
			return {
					'Labels': labels,
					'Damping': self.damping,
					'Max-Iter': self.max_iter,
					'Convergence-Iter': self.convergence_iter,
					'Copy': self.copy,
					'Preference': self.preference,
					'Affinity': self.affinity,
					'Verbose': self.verbose,
					'Random-State': self.random_state,
					'Iterations': self.iterations,
					'Centroids': self.centroids_,
					'Features': self.features,
					'Score': df_score
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = \
				'analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None'
			raise exception


class Birch( Cluster ):
	"""

		Purpose:
		---------
		The Birch builds a tree called the Clustering Feature Tree (CFT) for the given stores.
		The stores is essentially lossy compressed to a set of Clustering Feature nodes (CF Nodes).
		The CF Nodes have a number of subclusters called Clustering Feature subclusters
		(CF Subclusters) and these CF Subclusters located in the non-terminal
		CF Nodes can have CF Nodes as children.

		The BIRCH algorithm has two parameters, the threshold and the branching factor.
		The branching factor limits the number of subclusters in a node and the threshold limits
		the distance between the entering sample and the existing subclusters.

		This algorithm can be viewed as an instance or stores reduction method, since it reduces
		the input stores to a set of subclusters which are obtained directly from the leaves of the
		CFT. This reduced stores can be further processed by feeding it into a global clusterer.
		This global clusterer can be set by n_clusters. If n_clusters is set to None,
		the subclusters from the leaves are directly read off, otherwise a global clustering step
		target_names these subclusters into global clusters (target_names) and the samples are
		mapped to the global label of the nearest subcluster.

	"""
	model: skc.Birch
	threshold: Optional[ float ]
	branching_factor: Optional[ int ]
	n_clusters: object
	compute_labels: Optional[ bool ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, threshold: float = 0.5, branching_factor: int = 50,
			n_clusters: object = 3, compute_labels: bool = True ) -> None:
		"""

			Purpose:
			---------
			Initialize the Birch clustering wrapper.

			Parameters:
			-----------
				threshold (float): Radius threshold used to merge new samples into
					existing subclusters.
				branching_factor (int): Maximum number of CF subclusters in each node.
				n_clusters (object): Number of global clusters after the final
					clustering step, None, or another cluster model instance.
				compute_labels (bool): Whether to compute labels for each sample.

			Returns:
			--------
				None

		"""
		super( ).__init__( )
		self.threshold = threshold
		self.branching_factor = branching_factor
		self.n_clusters = n_clusters
		self.compute_labels = compute_labels
		self.model = skc.Birch(
			threshold=self.threshold,
			branching_factor=self.branching_factor,
			n_clusters=self.n_clusters,
			compute_labels=self.compute_labels
		)
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the primary public members exposed by the wrapper.

			Parameters:
			-----------
				None

			Returns:
			--------
				list[ str ]:
					Member names.

		"""
		return [
				'model',
				'score',
				'project',
				'predict',
				'train',
				'transform',
				'analyze',
				'threshold',
				'branching_factor',
				'n_clusters',
				'compute_labels',
				'labels',
				'subcluster_centers',
				'centroids_',
				'subcluster_labels',
				'features',
				'silouette',
				'homogeneity',
				'mutual_info',
				'v_measure',
				'completeness',
				'prediction'
		]
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted sample labels.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster labels.

		"""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def subcluster_centers( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return subcluster centers learned by Birch.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Subcluster centers.

		"""
		if not hasattr( self.model, 'subcluster_centers_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.subcluster_centers_
	
	@property
	def centroids_( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return subcluster centers using the centroid name expected by app.py.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Subcluster centers.

		"""
		return self.subcluster_centers
	
	@property
	def subcluster_labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return global labels assigned to each subcluster.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Subcluster labels.

		"""
		if not hasattr( self.model, 'subcluster_labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.subcluster_labels_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during fitting.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Number of fitted features.

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> Birch | None:
		"""

			Purpose:
			---------
			Fit the Birch clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				Birch | None:
					Trained wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			if hasattr( self.model, 'labels_' ):
				self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'train( self, X: np.ndarray ) -> Birch | None'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Generate cluster assignments for the supplied samples.

			Birch exposes fit_predict for the supplied data, so this method
			fits and returns labels for the supplied samples.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None:
					Cluster labels for each sample.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict the closest cluster each sample in X belongs to.

			Parameters:
			-----------
				X (np.ndarray): New samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None:
					Cluster labels for each sample.

		"""
		try:
			throw_if( 'X', X )
			
			if not hasattr( self.model, 'subcluster_centers_' ):
				raise AttributeError( 'The model data has not been trained!' )
			
			return self.model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform X into subcluster centroids distance space.

			Parameters:
			-----------
				X (np.ndarray): Input samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None:
					Distance matrix to subcluster centroids.

		"""
		try:
			throw_if( 'X', X )
			
			if not hasattr( self.model, 'subcluster_centers_' ):
				self.model.fit( X )
				if hasattr( self.model, 'labels_' ):
					self.prediction = self.model.labels_
			
			return self.model.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Evaluate Birch clustering performance using intrinsic and optional
			external clustering metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				pd.DataFrame | None:
					DataFrame containing clustering metrics.

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			
			unique_labels = np.unique( labels )
			scores[ 'Clusters' ] = len( unique_labels )
			scores[ 'Subclusters' ] = len( self.subcluster_centers )
			
			if len( unique_labels ) >= 2 and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = \
				'score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			---------
			Produce a summary analysis payload for the fitted clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				Dict[ str, Any ] | None:
					Dictionary containing analysis results.

		"""
		try:
			throw_if( 'X', X )
			
			if self.prediction is None or len( self.prediction ) != len( X ):
				labels = self.model.fit_predict( X )
				self.prediction = labels
			else:
				labels = self.prediction
			
			df_score = self.score( X, y )
			
			return {
					'Labels': labels,
					'Threshold': self.threshold,
					'Branching-Factor': self.branching_factor,
					'N-Clusters': self.n_clusters,
					'Compute-Labels': self.compute_labels,
					'Subcluster-Centers': self.subcluster_centers,
					'Subcluster-Labels': self.subcluster_labels,
					'Features': self.features,
					'Score': df_score
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = \
				'analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None'
			raise exception


class OPTICS( Cluster ):
	"""

		Purpose:
		---------
		The OPTICS is a generalization of DBSCAN that relaxes the eps requirement from a single
		value to a value range. The key difference between DBSCAN and OPTICS is that the OPTICS
		algorithm builds a reachability graph, which assigns each sample both a reachability_
		distance, and a spot within the cluster ordering_ attribute; these two attributes are
		assigned when the model is fitted, and are used to determine cluster membership.

		If OPTICS is run with the default value of inf set for max_eps, then DBSCAN style
		cluster extraction can be performed repeatedly in linear time for any given eps value
		using the cluster_optics_dbscan method. Setting max_eps to a lower value will result
		in shorter run times, and can be thought of as the maximum neighborhood radius from
		each point to find other potential reachable points.

	"""
	model: skc.OPTICS
	min_samples: Optional[ int ]
	max_eps: Optional[ float ]
	metric: object
	p: Optional[ float ]
	metric_params: Optional[ Dict[ str, Any ] ]
	cluster_method: Optional[ str ]
	eps: Optional[ float ]
	xi: Optional[ float ]
	predecessor_correction: Optional[ bool ]
	min_cluster_size: object
	algorithm: Optional[ str ]
	leaf_size: Optional[ int ]
	memory: object
	n_jobs: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, samples: int = 5, max_eps: float = np.inf,
			metric: object = 'minkowski', algorithm: str = 'auto',
			leaf_size: int = 30, eps: float | None = None,
			predecessor_correction: bool = True,
			min_cluster_size: int | float | None = None,
			min_samples: int | None = None, p: float = 2,
			metric_params: Dict[ str, Any ] | None = None,
			cluster_method: str = 'xi', xi: float = 0.05,
			memory: object = None, n_jobs: int | None = None ) -> None:
		"""

			Purpose:
			---------
			Initialize the OPTICS clustering wrapper.

			Parameters:
			-----------
				samples (int): Legacy alias for min_samples.
				max_eps (float): Maximum reachability distance.
				metric (object): Distance metric name or callable.
				algorithm (str): Neighbor search algorithm.
				leaf_size (int): Leaf size for BallTree or KDTree.
				eps (float | None): Extraction cutoff for DBSCAN-style extraction.
				predecessor_correction (bool): Whether predecessor correction is used.
				min_cluster_size (int | float | None): Minimum cluster size.
				min_samples (int | None): Explicit scikit-learn style min_samples.
				p (float): Power parameter for the Minkowski metric.
				metric_params (Dict[ str, Any ] | None): Additional metric
					keyword arguments.
				cluster_method (str): Cluster extraction method.
				xi (float): Minimum steepness on the reachability plot.
				memory (object): Optional joblib memory or cache path.
				n_jobs (int | None): Number of parallel jobs.

			Returns:
			--------
				None

		"""
		super( ).__init__( )
		self.min_samples = min_samples if min_samples is not None else samples
		self.max_eps = max_eps
		self.metric = metric
		self.p = p
		self.metric_params = metric_params
		self.cluster_method = cluster_method
		self.eps = eps
		self.xi = xi
		self.predecessor_correction = predecessor_correction
		self.min_cluster_size = min_cluster_size
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.memory = memory
		self.n_jobs = n_jobs
		self.model = skc.OPTICS(
			min_samples=self.min_samples,
			max_eps=self.max_eps,
			metric=self.metric,
			p=self.p,
			metric_params=self.metric_params,
			cluster_method=self.cluster_method,
			eps=self.eps,
			xi=self.xi,
			predecessor_correction=self.predecessor_correction,
			min_cluster_size=self.min_cluster_size,
			algorithm=self.algorithm,
			leaf_size=self.leaf_size,
			memory=self.memory,
			n_jobs=self.n_jobs
		)
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""

			Purpose:
			---------
			Return the primary public members exposed by the wrapper.

			Parameters:
			-----------
				None

			Returns:
			--------
				list[ str ]:
					Member names.

		"""
		return [
				'model',
				'train',
				'score',
				'project',
				'analyze',
				'min_samples',
				'max_eps',
				'metric',
				'p',
				'metric_params',
				'cluster_method',
				'eps',
				'xi',
				'predecessor_correction',
				'min_cluster_size',
				'algorithm',
				'leaf_size',
				'memory',
				'n_jobs',
				'labels',
				'ordering',
				'reachability',
				'core_distances',
				'predecessor',
				'features',
				'prediction',
				'silouette',
				'homogeneity',
				'mutual_info',
				'v_measure',
				'completeness'
		]
	
	@property
	def labels( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return fitted cluster labels.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Cluster labels.

		"""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def ordering( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the OPTICS ordering of samples.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Ordering indices.

		"""
		if not hasattr( self.model, 'ordering_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.ordering_
	
	@property
	def reachability( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return reachability distances.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Reachability distances.

		"""
		if not hasattr( self.model, 'reachability_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.reachability_
	
	@property
	def core_distances( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return core distances.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Core distances.

		"""
		if not hasattr( self.model, 'core_distances_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.core_distances_
	
	@property
	def predecessor( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return predecessor indices.

			Parameters:
			-----------
				None

			Returns:
			--------
				np.ndarray:
					Predecessor indices.

		"""
		if not hasattr( self.model, 'predecessor_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.predecessor_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features seen during fitting.

			Parameters:
			-----------
				None

			Returns:
			--------
				int:
					Number of fitted features.

		"""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> OPTICS | None:
		"""

			Purpose:
			---------
			Fit the OPTICS clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				OPTICS | None:
					Trained wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = 'train( self, X: np.ndarray ) -> OPTICS | None'
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Generate cluster assignments for the supplied samples.

			OPTICS exposes fit_predict for the supplied data, so this method
			fits and returns labels for the supplied samples.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).

			Returns:
			--------
				np.ndarray | None:
					Cluster labels for each sample.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Evaluate OPTICS clustering performance using intrinsic and optional
			external clustering metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				pd.DataFrame | None:
					DataFrame containing clustering metrics.

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			
			unique_labels = np.unique( labels )
			cluster_labels = unique_labels[ unique_labels != -1 ]
			scores[ 'Clusters' ] = len( cluster_labels )
			scores[ 'Noise' ] = int( np.sum( labels == -1 ) )
			
			mask = labels != -1
			if np.sum( mask ) >= 2 and len( np.unique( labels[ mask ] ) ) >= 2:
				self.silouette = silhouette_score( X[ mask ], labels[ mask ] )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = \
				'score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			---------
			Produce a summary analysis payload for the fitted clustering model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix/input samples of shape
					( n_samples, n_features ).
				y (Optional[np.ndarray]): Optional reference labels of shape
					( n_samples, ).

			Returns:
			--------
				Dict[ str, Any ] | None:
					Dictionary containing analysis results.

		"""
		try:
			throw_if( 'X', X )
			
			if self.prediction is None or len( self.prediction ) != len( X ):
				labels = self.model.fit_predict( X )
				self.prediction = labels
			else:
				labels = self.prediction
			
			df_score = self.score( X, y )
			
			return {
					'Labels': labels,
					'Min-Samples': self.min_samples,
					'Max-Eps': self.max_eps,
					'Metric': self.metric,
					'P': self.p,
					'Metric-Params': self.metric_params,
					'Cluster-Method': self.cluster_method,
					'Eps': self.eps,
					'Xi': self.xi,
					'Predecessor-Correction': self.predecessor_correction,
					'Min-Cluster-Size': self.min_cluster_size,
					'Algorithm': self.algorithm,
					'Leaf-Size': self.leaf_size,
					'N-Jobs': self.n_jobs,
					'Ordering': self.ordering,
					'Reachability': self.reachability,
					'Core-Distances': self.core_distances,
					'Predecessor': self.predecessor,
					'Features': self.features,
					'Score': df_score
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = \
				'analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None'
			raise exception
			
