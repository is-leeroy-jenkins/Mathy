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
		train(X, y=None) -> self
		project(X) -> np.ndarray
		score(X, y=None) -> float
		analyze(X, y=None) -> Dict | None

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
	
	def __init__( self ):
		pass
	
	def train( self, X: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
			Fit the linerar_model to the training df.

			Parameters:
			-----------
				X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
				y (np.ndarray): Target vector w/shape ( n_samples, ).

			Returns:
			--------
				None

		"""
		raise NotImplementedError
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Generate predictions from  the trained linerar_model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).

			Returns:
			-----------
				np.ndarray: Predicted target_names or class target_names.

		"""
		raise NotImplementedError
	
	def score( self, X: np.ndarray ) -> pd.DataFrame | None:
		"""

			Purpose:
			---------
			Compute the core metric (e.g., R²) of the model on test df.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): True target target_names.

			Returns:
			-----------
				float: Score value (e.g., R² for regressors).

		"""
		raise NotImplementedError
	
	def analyze( self, X: np.ndarray ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			---------
			Evaluate the model using multiple performance metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target_names.

			Returns:
			-----------
				dict: Dictionary containing multiple evaluation metrics.

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
	init: Optional[ str ]
	n_init: Optional[ str ]
	tolerance: Optional[ str ]
	random_state: Optional[ int ]
	max_iter: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, clusters: int=8, init: str='k-means++', n_init: str='auto',
			tol: float=0.0001, rando: int=42, max_iter: int=300 ) -> None:
		"""
			Purpose:
			---------
			Initialize the KMeans model.

			Parameters:
			----------
			num: Number of clusters to form.
			rando: Random seed for reproducibility.
			max: number of iterations.

		"""
		super( ).__init__( )
		self.n_clusters = clusters
		self.init = init
		self.n_init = n_init
		self.random_state = rando
		self.max_iter = max_iter
		self.tolerance = tol
		self.model = skc.KMeans( n_clusters=self.n_clusters, init=self.init, n_init=self.n_init,
			random_state=self.random_state, max_iter=self.max_iter )
		self.prediction = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ):
		'''
			
			Returns
			-------
			A list of strings repreenting members
			
		'''
		return [ 'model',
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
		         'clusters',
		         'labels',
		         'inertia',
		         'silouette',
		         'homogeneity',
		         'mutual_info',
		         'v_measuere',
		         'iterations',
		         'features',
		         'prediction',
		         'accuracy' ]
	
	@property
	def clusters( self ) -> np.ndarray:
		'''
	
			Returns
			-------
			cluster_centers_ : ndarray of shape (n_clusters, n_features )
	
		'''
		if self.model.cluster_centers_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.cluster_centers_
	
	@property
	def labels( self ) -> np.ndarray:
		'''
	
			Returns
			-------
			labels_ : ndarray of shape (n_samples,)
	
		'''
		if self.model.labels_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.labels_
	
	@property
	def inertia( self ) -> float:
		'''
	
			Returns
			-------
			inertia_ (float):
			Sum of squared distances of samples to their closest cluster center,
			weighted by the sample weights if provided.
	
		'''
		if self.model.inertia_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.inertia_
	
	@property
	def iterations( self ) -> int:
		'''
	
			Returns
			-------
			n_iter_ (int) is ndarray of shape ( n_classes, )
			Represents the number of iterations run by the coordinate descent solver
			to reach the specified tolerance.
	
		'''
		if self.model.n_iter_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		'''
	
			Returns
			-------
			n_features_in_
			The number of features seen during training
	
		'''
		if self.model.n_features_in_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> KMeans | None:
		"""
	
			Purpose:
			---------
			Fit the KMeans model on the dataset.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'train( self, X: np.ndarray ) -> None'
			raise exception
			
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
	
			Purpose:
			---------
			Predict the closest cluster each sample in X belongs to.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
			Returns:
			--------
			np.ndarray
	
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def score( self, X: np.ndarray ) -> pd.DataFrame:
		"""
	
			Purpose:
			---------
			Evaluate clustering performance using silhouette score.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
			Returns:
			---------
			float
	
		"""
		try:
			throw_if( 'X', X )
			labels = self.model.predict( X )
			self.siloutte = silhouette_score( X, labels )
			self.homogeneity = homogeneity_score( X, labels )
			self.mutual_info = mutual_info_score( X, labels )
			self.completeness = completeness_score( X, labels )
			self.v_measure = v_measure_score( X, labels )
			scores = \
			{
				'Silouette': self.silouette,
				'Homogeneity': self.homogeneity,
				'Mutual-Info': self.mutual_info,
				'Completeness': self.completeness,
				'V-Measure': self.v_measure,
			}
			cols = list( scores.keys( ) )
			vals = list( scores.values( ) )
			data = pd.DataFrame( data=vals, columns=cols )
			return data
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'score( self, X: np.ndarray ) -> float'
			raise exception
			
	
	def analyze( self, X: np.ndarray ) -> None:
		"""
	
			Purpose:
			---------
			Visualize clustering result using a scatter plot.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
	
		"""
		try:
			throw_if( 'X', X )
			labels = self.model.predict( X )
			plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels, cmap='viridis' )
			plt.title( "K-Means" )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'analyze( self, X: np.ndarray ) -> None'
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
	eps: Optional[ float ]
	min_samples: Optional[ int ]
	algorithm: Optional[ str ]
	metric: Optional[ str ]
	leaf_size: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, max_distance: float=0.5, samples: int=5, measure: str='euclidean',
			algorithm: str='auto', size: int=30 ) -> None:
		"""

			Purpose:
			---------
			Initialize the DBSCAN model.

			Parameters:
			----------
			max_distance: float
			eps - The maximum distance between two samples for one to be considered as in the
			neighborhood of the other. This is not a maximum bound on the distances of points
			within a cluster. This is the most important DBSCAN parameter to choose appropriately
			for your data set and distance function
			
			samples: int
			min_samples - The number of samples (or total weight) in a neighborhood for a point
			to be considered as a core point. This includes the point itself.
			
			measure: str
			metric - The metric to use when calculating distance between instances in a feature array.
			
			strategy: str
			algorith - The algorithm to be used by the NearestNeighbors module to compute
			pointwise distances and find nearest neighbors
			
			size: int
			leaf_size - Leaf size passed to BallTree or cKDTree. This can affect the speed of the
			construction and query, as well as the memory required to store the tree.
			The optimal value depends on the nature of the problem.

		"""
		super( ).__init__( )
		self.eps = max_distance
		self.min_samples = samples
		self.algorithm = algorithm
		self.metric = measure
		self.leaf_size = size
		self.model = skc.DBSCAN( eps=self.eps, min_samples=self.min_samples, metric=self.metric,
			algorithm=self.algorithm, leaf_size=self.leaf_size )
		self.prediction = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings repreenting members

		'''
		return [ 'model',
		         'train',
		         'score',
		         'project',
		         'transform',
		         'analyze',
		         'min_samples',
		         'algorithm',
		         'eps',
		         'leaf_size',
		         'silouette',
		         'homogeneity',
		         'mutual_info',
		         'v_measuere',
		         'clusters',
		         'prediction',]
	
	def train( self, X: np.ndarray ) -> DBSCAN | None:
		"""
	
			Purpose:
			---------
			Fit the DBSCAN model to the stores.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = 'train( self, X: np.ndarray ) -> None'
			raise exception
			
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
	
			Purpose:
			---------
			Predict clusters using DBSCAN fit.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
			Returns:
			---------
			np.ndarray
	
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def score( self, X: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Evaluate clustering performance using silhouette score.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Returns:
			---------
			float

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.predict( X )
			self.siloutte = silhouette_score( X, labels )
			self.homogeneity = homogeneity_score( X, labels )
			self.mutual_info = mutual_info_score( X, labels )
			self.completeness = completeness_score( X, labels )
			self.v_measure = v_measure_score( X, labels )
			scores = \
			{
				'Silouette': self.silouette,
				'Homogeneity': self.homogeneity,
				'Mutual-Info': self.mutual_info,
				'Completeness': self.completeness,
				'V-Measure': self.v_measure,
			}
			cols = list( scores.keys( ) )
			data = pd.DataFrame( data=scores, columns=cols )
			return data
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = 'score( self, X: np.ndarray ) -> float'
			raise exception
			
	
	def analyze( self, X: np.ndarray ) -> None:
		"""
	
			Purpose:
			---------
			Visualize DBSCAN clusters.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels, cmap='plasma' )
			plt.title( 'DBSCAN Cluster' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = 'analyze( self, X: np.ndarray ) -> None'
			raise exception
			

class Agglomerative( Cluster ):
	"""

		Purpose:
		---------
		The AgglomerativeCluster object performs a hierarchical clustering using a
		bottom up approach: each observation starts in its own cluster, and clusters are
		successively merged together. The linkage criteria determines the metric used for the merge
		strategy:

		Minimizes the sum of squared differences within all clusters. It is a
		variance-minimizing approach and in this sense is similar to the k-means objective
		function but tackled with an agglomerative hierarchical approach.

		Maximum or complete linkage minimizes the maximum distance between observations of
		pairs of clusters.

		Average linkage minimizes the average of the distances between all observations of
		pairs of clusters.

		Single linkage minimizes the distance between the closest observations of pairs of
		clusters.
		AgglomerativeCluster can also scale to large number of samples when it is used jointly
		with a connectivity matrix, but is computationally expensive when no connectivity
		constraints are added between samples: it considers at each step all the possible merges.

	"""
	model: skc.AgglomerativeClustering
	n_clusters: Optional[ int ]
	affinity: Optional[ str ]
	compute_full_tree: Optional[ str ]
	linkage: Optional[ str ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, clusters: int=2, distance: str='euclidean',
			full_tree: str='auto', link: str='ward' ) -> None:
		"""

			Purpose:
			---------
			Initialize AgglomerativeCluster.

			Parameters:
			----------
			num: int

		"""
		super( ).__init__( )
		self.n_clusters = clusters
		self.affinity = distance
		self.compute_full_tree = full_tree
		self.linkage = link
		self.model = skc.AgglomerativeClustering( n_clusters=self.n_clusters,
			affinity=self.affinity, compute_full_tree=self.compute_full_tree, linkage=self.linkage )
		self.prediction = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings repreenting members

		'''
		return [ 'model',
		         'n_clusters',
		         'affinity',
		         'compute_ful_tree',
		         'linkage',
		         'train',
		         'score',
		         'project',
		         'transform',
		         'analyze',
		         'silouette',
		         'homogeneity',
		         'mutual_info',
		         'v_measuere',
		         'prediction', ]
	
	def train( self, X: np.ndarray ) -> Agglomerative | None:
		"""
	
			Purpose:
			---------
			Fit Agglomerative model to stores.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
	
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = 'train( self, X: np.ndarray ) -> None'
			raise exception
			
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
	
			Purpose:
			---------
			Predict clusters using agglomerative clustering.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
			Returns:
			---------
			np.ndarray
	
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def score( self, X: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Evaluate clustering performance using silhouette score.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Returns:
			---------
			float

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.predict( X )
			self.siloutte = silhouette_score( X, labels )
			self.homogeneity = homogeneity_score( X, labels )
			self.mutual_info = mutual_info_score( X, labels )
			self.completeness = completeness_score( X, labels )
			self.v_measure = v_measure_score( X, labels )
			scores = \
			{
				'Silouette': self.silouette,
				'Homogeneity': self.homogeneity,
				'Mutual-Info': self.mutual_info,
				'Completeness': self.completeness,
				'V-Measure': self.v_measure,
			}
			cols = list( scores.keys( ) )
			data = pd.DataFrame( data=scores, columns=cols )
			return data
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = 'score( self, X: np.ndarray ) -> float'
			raise exception
			
	
	def analyze( self, X: np.ndarray ) -> None:
		"""
	
			Purpose:
			---------
			Visualize agglomerative clustering results.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			Z = X[ :, :2 ] if X.shape[ 1 ] >= 2 else np.hstack( [ X,
			                                                      X ] )
			labels = self.model.fit_predict( X )
			plt.scatter( Z[ :, 0 ], Z[ :, 1 ], c=labels, cmap='tab10' )
			plt.title( 'Agglomerative Cluster' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = 'analyze( self, X: np.ndarray ) -> None'
			raise exception
			

class Spectral( Cluster ):
	"""

		Purpose:
		---------
		SpectralCluster does a low-dimension embedding of the affinity matrix between samples,
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
	random_state: Optional[ int ]
	n_init: Optional[ int ]
	gamma: Optional[ float ]
	n_neighbors: Optional[ int ]
	eigen_tolerance: Optional[ float ]
	affinity: Optional[ str ]
	assign_labels: Optional[ str ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, clusters=8, random_state: int=42, n_init=10,
			gama=1.0, distance='rbf', neighbors=10,
			tolerance=0.0, assign='kmeans'  ) -> None:
		"""

			Purpose:
			---------
			Initialize the SpectralCluster model.

			Parameters:
			----------
			num: int

		"""
		super( ).__init__( )
		self.n_clusters = clusters
		self.random_state = random_state
		self.n_init = n_init
		self.gamma = gama
		self.affinity = distance
		self.n_neighbors = neighbors
		self.eigen_tolerance = tolerance
		self.assign_labels = assign
		self.model = skc.SpectralClustering( n_clusters=self.n_clusters,
			random_state=self.random_state, n_init=self.n_init, gamma=self.gamma,
			affinity=self.affinity, n_neighbors=self.n_neighbors,
			eigen_tol=self.eigen_tolerance, assign_labels=self.assign_labels )
		self.prediction = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings repreenting members

		'''
		return [ 'model',
		         'n_clusters',
		         'random_state',
		         'n_init',
		         'gamma',
		         'affinity',
		         'n_neighbors',
		         'eigen_tolerance',
		         'assign_labels',
		         'train',
		         'score',
		         'project',
		         'transform',
		         'analyze',
		         'silouette',
		         'homogeneity',
		         'mutual_info',
		         'v_measuere',
		         'prediction', ]
	
	def train( self, X: np.ndarray ) -> Spectral | None:
		"""
	
			Purpose:
			---------
			Fit the SpectralCluster model.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = 'train( self, X: np.ndarray ) -> None'
			raise exception
			
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
	
			Purpose:
			---------
			Predict clusters using SpectralCluster.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
	
			Return:
			--------
			np.ndarray
	
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def score( self, X: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Evaluate clustering performance using silhouette score.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Returns:
			---------
			float

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.predict( X )
			self.siloutte = silhouette_score( X, labels )
			self.homogeneity = homogeneity_score( X, labels )
			self.mutual_info = mutual_info_score( X, labels )
			self.completeness = completeness_score( X, labels )
			self.v_measure = v_measure_score( X, labels )
			scores = \
			{
				'Silouette': self.silouette,
				'Homogeneity': self.homogeneity,
				'Mutual-Info': self.mutual_info,
				'Completeness': self.completeness,
				'V-Measure': self.v_measure,
			}
			cols = list( scores.keys( ) )
			data = pd.DataFrame( data=scores, columns=cols )
			return data
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = 'score( self, X: np.ndarray ) -> float'
			raise exception
			
	
	def analyze( self, X: np.ndarray ) -> None:
		"""
	
			Purpose:
			---------
			Visualize Spectral Cluster results.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels, cmap='Accent' )
			plt.title( 'Spectral Cluster' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = 'analyze( self, X: np.ndarray ) -> None'
			raise exception
			

class MeanShift( Cluster ):
	"""

		Purpose:
		---------
		MeanShift clustering aims to discover blobs in a smooth density of samples.
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
	min_bin_freq: Optional[ int ]
	cluster_all: Optional[ bool ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, min_bin: int=1, group_all: bool=True ) -> None:
		"""

			Purpose:
			---------
			Initialize MeanShift model.

		"""
		super( ).__init__( )
		self.min_bin_freq = min_bin
		self.cluster_all = group_all
		self.model = skc.MeanShift( min_bin_freq=self.min_bin_freq, cluster_all=self.cluster_all )
		self.prediction = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings repreenting members

		'''
		return [ 'model',
		         'train',
		         'score',
		         'project',
		         'transform',
		         'analyze',
		         'silouette',
		         'homogeneity',
		         'mutual_info',
		         'v_measuere',
		         'prediction', ]
	
	def train( self, X: np.ndarray ) -> MeanShift | None:
		"""
	
			Purpose:
			---------
			Fit MeanShift model to the stores.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'train( self, X: np.ndarray ) -> None'
			raise exception
			
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
	
			Purpose:
			---------
			Predict clusters using MeanShift.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
			Return:
			--------
			np.ndarray
	
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def score( self, X: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Evaluate clustering performance using silhouette score.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Returns:
			---------
			float

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.predict( X )
			self.siloutte = silhouette_score( X, labels )
			self.homogeneity = homogeneity_score( X, labels )
			self.mutual_info = mutual_info_score( X, labels )
			self.completeness = completeness_score( X, labels )
			self.v_measure = v_measure_score( X, labels )
			scores = \
			{
				'Silouette': self.silouette,
				'Homogeneity': self.homogeneity,
				'Mutual-Info': self.mutual_info,
				'Completeness': self.completeness,
				'V-Measure': self.v_measure,
			}
			cols = list( scores.keys( ) )
			data = pd.DataFrame( data=scores, columns=cols )
			return data
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'score( self, X: np.ndarray ) -> float'
			raise exception
			
	
	def analyze( self, X: np.ndarray ) -> None:
		"""
	
			Purpose:
			---------
			Visualize MeanShift clustering.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels, cmap='Set1' )
			plt.title( 'MeanShift Cluster' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'analyze( self, X: np.ndarray ) -> None'
			raise exception
			

class AffinityPropagation( Cluster ):
	"""

		Purpose:
		---------
		AffinityPropagation creates clusters by sending messages between pairs of samples until
		convergence. A dataset is then described using a small number of exemplars, which are
		identified as those most representative of other samples. The messages sent between pairs
		represent the suitability for one sample to be the exemplar of the other, which is updated
		in response to the values from other pairs. This updating happens iteratively until
		convergence, at which point the final exemplars are chosen,
		and hence the final clustering is given.

	"""
	model: skc.AffinityPropagation
	damping: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	preference: Optional[ np.ndarray ]
	affinity: Optional[ str ]
	convergence_iter: Optional[ int ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, damp: float=0.5, iters: int=200, convergence: int=15,
			prefer: np.ndarray=None, distance: str='euclidean' ) -> None:
		"""

			Purpose:
			---------
			Initialize AffinityPropagation model.

		"""
		super( ).__init__( )
		self.damping = damp
		self.max_iter = iters
		self.convergence_iter = convergence
		self.preference = prefer
		self.affinity = distance
		self.model = skc.AffinityPropagation( damping=self.damping, max_iter=self.max_iter,
			convergence_iter=self.convergence_iter, preference=self.preference, affinity=self.affinity )
		self.prediction = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings repreenting members

		'''
		return [ 'model',
		         'damping',
		         'max_iter',
		         'convergence_iter',
		         'preference',
		         'affinity',
		         'train',
		         'score',
		         'project',
		         'transform',
		         'analyze',
		         'silouette',
		         'homogeneity',
		         'mutual_info',
		         'v_measuere',
		         'prediction', ]
	
	def train( self, X: np.ndarray ) -> AffinityPropagation | None:
		"""
	
			Purpose:
			---------
			Fit the model to stores.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = 'train( self, X: np.ndarray ) -> None'
			raise exception
			
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
	
			Purpose:
			---------
			Predict clusters.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
			Returns:
			--------
			np.ndarray
	
		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit( X ).labels_
			return np.ndarray( labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def score( self, X: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Evaluate clustering performance using silhouette score.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Returns:
			---------
			float

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.predict( X )
			self.siloutte = silhouette_score( X, labels )
			self.homogeneity = homogeneity_score( X, labels )
			self.mutual_info = mutual_info_score( X, labels )
			self.completeness = completeness_score( X, labels )
			self.v_measure = v_measure_score( X, labels )
			scores = \
			{
				'Silouette': self.silouette,
				'Homogeneity': self.homogeneity,
				'Mutual-Info': self.mutual_info,
				'Completeness': self.completeness,
				'V-Measure': self.v_measure,
			}
			cols = list( scores.keys( ) )
			data = pd.DataFrame( data=scores, columns=cols )
			return data
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = 'score( self, X: np.ndarray ) -> float'
			raise exception
			
	
	def analyze( self, X: np.ndarray ) -> None:
		"""
	
			Purpose:
			---------
			Visualize clustering with AffinityPropagation.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit( X ).labels_
			plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels, cmap='Paired' )
			plt.title( 'AffinityPropagation Cluster' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = 'analyze( self, X: np.ndarray ) -> None'
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
		the
		input stores to a set of subclusters which are obtained directly from the leaves of the
		CFT.
		This reduced stores can be further processed by feeding it into a global clusterer.
		This global clusterer can be set by n_clusters. If n_clusters is set to None,
		the subclusters from the leaves are directly read off, otherwise a global clustering step
		target_names these subclusters into global clusters (target_names) and the samples are
		mapped to the global label of the nearest subcluster.

	"""
	model: skc.Birch
	n_clusters: Optional[ int ]
	threshold: Optional[ float ]
	branching_factor: Optional[ int ]
	copy: Optional[ bool ]
	compute_labels: Optional[ bool ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, threshold=0.5, branching_factor=50, n_clusters=3,
			compute_labels=True, copy=True ) -> None:
		"""

			Purpose:
			---------
			Initialize Birch clustering.

			Parameters:
			----------
			n_clusters: Optional[int]
			threshold: Optional[ float ]
			branching_factor: Optional[ int ]
			copy: Optional[ bool ]
			compute_labels: Optional[ bool ]

		"""
		super( ).__init__( )
		self.threshold = threshold
		self.branching_factor = branching_factor
		self.n_clusters = n_clusters
		self.compute_labels = compute_labels
		self.copy = copy
		self.model = skc.Birch( threshold=self.threshold, branching_factor=self.branching_factor,
			n_clusters=self.n_clusters, copy=self.copy, compute_labels=self.compute_labels )
		self.prediction = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings repreenting members

		'''
		return [ 'model',
		         'score',
		         'project',
		         'train',
		         'transform',
		         'analyze',
		         'subcluster_centers',
		         'subcluster_labels',
		         'silouette',
		         'homogeneity',
		         'mutual_info',
		         'v_measuere',
		         'prediction', ]
	
	@property
	def subcluster_centers( self ) -> np.ndarray:
		'''

			Returns
			-------
			n_iter_ (int) is ndarray of shape ( n_classes, )
			Represents the number of iterations run by the coordinate descent solver
			to reach the specified tolerance.

		'''
		if self.model.subcluster_centers_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.subcluster_centers_
	
	@property
	def subcluster_labels( self ) -> np.ndarray:
		'''

			Returns
			-------
			n_features_in_
			The number of features seen during training

		'''
		if self.model.subcluster_labels_ is None:
			raise AttributeError( 'The model data has not been trained!' )
		else:
			return self.model.subcluster_labels_
	
	def train( self, X: np.ndarray ) -> Birch | None:
		"""
	
			Purpose:
			---------
			Fit Birch clustering model.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'train( self, X: np.ndarray ) -> None'
			raise exception
			
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
	
			Purpose:
			---------
			Predict clusters with Birch.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
			Returns:
			--------
			np.ndarray
	
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def score( self, X: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Evaluate clustering performance using silhouette score.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Returns:
			---------
			float

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.predict( X )
			self.siloutte = silhouette_score( X, labels )
			self.homogeneity = homogeneity_score( X, labels )
			self.mutual_info = mutual_info_score( X, labels )
			self.completeness = completeness_score( X, labels )
			self.v_measure = v_measure_score( X, labels )
			scores = \
			{
				'Silouette': self.silouette,
				'Homogeneity': self.homogeneity,
				'Mutual-Info': self.mutual_info,
				'Completeness': self.completeness,
				'V-Measure': self.v_measure,
			}
			cols = list( scores.keys( ) )
			data = pd.DataFrame( data=scores, columns=cols )
			return data
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'score( self, X: np.ndarray ) -> float'
			raise exception
			
	
	def analyze( self, X: np.ndarray ) -> None:
		"""
	
			Purpose:
			---------
			Visualize Birch clustering.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			labels = self.model.predict( X )
			plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels, cmap='Dark2' )
			plt.title( 'Birch Cluster' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'analyze( self, X: np.ndarray ) -> None'
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
	metric: Optional[ str ]
	eps: Optional[ float ]
	xi: Optional[ float ]
	min_cluster_size: Optional[ int ]
	algorith: Optional[ str ]
	leaf_size: Optional[ int ]
	predecessor_correction: Optional[ bool ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, samples: int=5, max_distance: float=np.inf, measure: str='minkowski',
			distance: float=None, correction: bool=True, min_size: int=None,
			method: str='auto', leaf_size: int=30 ) -> None:
		"""

			Purpose:
			---------
			Initialize OPTICS model.

			Parameters:
			----------
			min: int

		"""
		super( ).__init__( )
		self.min_samples = samples
		self.max_eps = max_distance
		self.metric = measure
		self.eps = distance
		self.predecessor_correction = correction
		self.min_cluster_size = min_size
		self.algorith = method
		self.leaf_size = leaf_size
		self.model = skc.OPTICS( min_samples=self.min_samples, max_eps=self.max_eps,
			algorithm=self.algorith, leaf_size=self.leaf_size, metric=self.metric,
			predecessor_correction=self.predecessor_correction, eps=self.eps,
			min_cluster_size=self.min_cluster_size, )
		self.prediction = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings repreenting members

		'''
		return [ 'model',
		         'max_eps',
		         'min_samples',
		         'metric',
		         'eps',
		         'predecessor_correction',
		         'min_cluster_size',
		         'leaf_size',
		         'score',
		         'project',
		         'train',
		         'transform',
		         'analyze',
		         'silouette',
		         'homogeneity',
		         'mutual_info',
		         'v_measuere',
		         'prediction', ]
	
	def train( self, X: np.ndarray ) -> OPTICS | None:
		"""
	
			Purpose:
			---------
			Fit OPTICS model.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = 'train( self, X: np.ndarray ) -> None'
			raise exception
			
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""
	
			Purpose:
			---------
			Predict clusters with OPTICS.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
			Return:
			--------
			np.ndarray
	
		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			raise exception
			
	
	def score( self, X: np.ndarray ) -> pd.DataFrame:
		"""

			Purpose:
			---------
			Evaluate clustering performance using silhouette score.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Returns:
			---------
			float

		"""
		try:
			throw_if( 'X', X )
			labels = self.model.predict( X )
			self.siloutte = silhouette_score( X, labels )
			self.homogeneity = homogeneity_score( X, labels )
			self.mutual_info = mutual_info_score( X, labels )
			self.completeness = completeness_score( X, labels )
			self.v_measure = v_measure_score( X, labels )
			scores = \
				{
						'Silouette': self.silouette,
						'Homogeneity': self.homogeneity,
						'Mutual-Info': self.mutual_info,
						'Completeness': self.completeness,
						'V-Measure': self.v_measure,
				}
			cols = list( scores.keys( ) )
			data = pd.DataFrame( data=scores, columns=cols )
			return data
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = 'score( self, X: np.ndarray ) -> float'
			raise exception
			
	
	def analyze( self, X: np.ndarray ) -> None:
		"""
	
			Purpose:
			---------
			Visualize OPTICS clustering result.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).
	
	
		"""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels, cmap='rainbow' )
			plt.title( 'OPTICS Cluster' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = 'analyze( self, X: np.ndarray ) -> None'
			raise exception
			
