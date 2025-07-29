'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                clusters.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="clusters.py" company="Terry D. Eppler">

     Mathy Clusters

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
from typing import Optional, Dict
from booger import Error, ErrorDialog
import sklearn.cluster as skc
from sklearn.metrics import silhouette_score


class Cluster( ):
	"""

		Purpose:
		---------
		Abstract base class that defines the interface for all linerar_model wrappers.

	"""

	def __init__( self ):
		pass


	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
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


	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> float | None:
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


	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Dict | None:
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



class KMeansCluster( Cluster ):
	"""

		Purpose:
		---------
		The KMeans algorithm clusters data by trying to separate samples in n groups of equal
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
	kmeans_cluster: skc.KMeans
	n_clusters: Optional[ int ]
	random_state: Optional[ int ]
	max_iter: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]

	def __init__( self, num: int=8, rando: int=42, max: int=300 ) -> None:
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
		self.n_clusters = num
		self.random_state = rando
		self.max_iter = max
		self.kmeans_cluster = skc.KMeans( n_clusters=self.n_clusters,
			random_state=self.random_state, max_iter=self.max_iter )
		self.prediction = None
		self.accuracy = 0.0


	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> KMeansCluster | None:
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
			self.kmeans_cluster.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansCluster'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


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
			self.prediction = self.kmeans_cluster.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansCluster'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> float | None:
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
			labels = self.kmeans_cluster.predict( X )
			return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansCluster'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
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
			if X is None:
				raise Exception( 'The input arguement "X" is required.' )
			else:
				labels = self.kmeans_cluster.predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels, cmap='viridis' )
				plt.title( "KMeans Cluster" )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansCluster'
			exception.method = 'visualize( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class DbscanCluster( Cluster ):
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
	db_scan: skc.DBSCAN
	eps: Optional[ float ]
	min_samples: Optional[ int ]
	algorithm: Optional[ str ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]

	def __init__( self, eps: float=0.5, min: int=5, algo: str='auto' ) -> None:
		"""

			Purpose:
			---------
			Initialize the DBSCAN model.

			Parameters:
			----------
			eps: float
			min: int

		"""
		super( ).__init__( )
		self.eps = eps
		self.min_samples = min
		self.algorithm = algo
		self.model = skc.DBSCAN( eps=self.eps, min_samples=self.min_samples,
			algorithm=self.algorithm )
		self.prediction = None
		self.accuracy = 0.0


	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> DbscanCluster | None:
		"""

			Purpose:
			---------
			Fit the DBSCAN model to the data.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.db_scan.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanCluster'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.prediction = self.db_scan.fit_predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanCluster'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> float | None:
		"""

			Purpose:
			---------
			Evaluate DBSCAN clusters with silhouette score.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Returns:
			---------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.db_scan.fit_predict( X )
				return silhouette_score( X, labels ) if len( set( labels ) ) > 1 else -1.0
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanCluster'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.db_scan.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels, cmap='plasma' )
				plt.title( 'DBSCAN Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanCluster'
			exception.method = 'analyze( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class AgglomerativeCluster( Cluster ):
	"""

		Purpose:
		---------
		The AgglomerativeCluster object performs a hierarchical clustering using a
		bottom up approach: each observation starts in its own cluster, and clusters are
		successively merged together. The linkage criteria determines the metric used for the merge
		strategy:

			Ward minimizes the sum of squared differences within all clusters. It is a
			variance-minimizing approach and in this sense is similar to the k-means objective
			function but tackled with an agglomerative hierarchical approach.

			Maximum or complete linkage minimizes the maximum distance between observations of
			pairs of clusters.

			Average linkage minimizes the average of the distances between all observations of
			pairs of clusters.

		Single linkage minimizes the distance between the closest observations of pairs of clusters.
		AgglomerativeCluster can also scale to large number of samples when it is used jointly
		with a connectivity matrix, but is computationally expensive when no connectivity
		constraints are added between samples: it considers at each step all the possible merges.

	"""
	agg_cluster: skc.AgglomerativeClustering
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]

	def __init__( self, num: int = 2 ) -> None:
		"""

			Purpose:
			---------
			Initialize AgglomerativeCluster.

			Parameters:
			----------
			num: int

		"""
		super( ).__init__( )
		self.agg_cluster = skc.AgglomerativeClustering( n_clusters=num )
		self.prediction = None
		self.accuracy = 0.0


	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> AgglomerativeCluster | None:
		"""

			Purpose:
			---------
			Fit Agglomerative model to data.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).


		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.agg_cluster.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.agg_cluster.fit_predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> float | None:
		"""

			Purpose:
			---------
			Evaluate agglomerative clustering using silhouette score.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Returns:
			-------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.agg_cluster.fit_predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.agg_cluster.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'tab10' )
				plt.title( 'Agglomerative Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class SpectralCluster( Cluster ):
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
	spectral_clustering: skc.SpectralClustering
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]


	def __init__( self, num: int=8 ) -> None:
		"""

			Purpose:
			---------
			Initialize the SpectralCluster model.

			Parameters:
			----------
			num: int

		"""
		super( ).__init__( )
		self.spectral_clustering = skc.SpectralClustering( n_clusters=num )
		self.prediction = None
		self.accuracy = 0.0


	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> SpectralCluster | None:
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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.spectral_clustering.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.prediction = self.spectral_clustering.fit_predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> float | None:
		"""

			Purpose:
			---------
			Evaluate SpectralCluster results.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.spectral_clustering.fit_predict( X )
				self.accuracy = silhouette_score( X, labels )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.spectral_clustering.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'Accent' )
				plt.title( 'Spectral Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class MeanShiftCluster( Cluster ):
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
	mean_shift: skc.MeanShift
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]


	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize MeanShift model.

		"""
		super( ).__init__( )
		self.mean_shift = skc.MeanShift( )
		self.prediction = None
		self.accuracy = 0.0


	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
		"""

			Purpose:
			---------
			Fit MeanShift model to the data.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.mean_shift.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftCluster'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.prediction = self.mean_shift.fit_predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftCluster'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> float | None:
		"""

			Purpose:
			---------
			Evaluate MeanShift clustering.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.mean_shift.fit_predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftCluster'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.mean_shift.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'Set1' )
				plt.title( 'MeanShift Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftCluster'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class AffinityPropagationCluster( Cluster ):
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
	affinity_propagation: skc.AffinityPropagation
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]


	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize AffinityPropagation model.

		"""
		super( ).__init__( )
		self.affinity_propagation = skc.AffinityPropagation( )
		self.prediction = None
		self.accuracy = 0.0


	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> AffinityPropagationCluster | None:
		"""

			Purpose:
			---------
			Fit the model to data.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.affinity_propagation.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationCluster'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.affinity_propagation.fit( X ).labels_
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationCluster'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> float | None:
		"""

			Purpose:
			---------
			Evaluate clustering result.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.affinity_propagation.fit( X ).labels_
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationCluster'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.affinity_propagation.fit( X ).labels_
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels, cmap='Paired' )
				plt.title( 'AffinityPropagation Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationCluster'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BirchCluster( Cluster ):
	"""

		Purpose:
		---------
		The Birch builds a tree called the Clustering Feature Tree (CFT) for the given data.
		The data is essentially lossy compressed to a set of Clustering Feature nodes (CF Nodes).
		The CF Nodes have a number of subclusters called Clustering Feature subclusters
		(CF Subclusters) and these CF Subclusters located in the non-terminal
		CF Nodes can have CF Nodes as children.

		The BIRCH algorithm has two parameters, the threshold and the branching factor.
		The branching factor limits the number of subclusters in a node and the threshold limits
		the distance between the entering sample and the existing subclusters.

		This algorithm can be viewed as an instance or data reduction method, since it reduces the
		input data to a set of subclusters which are obtained directly from the leaves of the CFT.
		This reduced data can be further processed by feeding it into a global clusterer.
		This global clusterer can be set by n_clusters. If n_clusters is set to None,
		the subclusters from the leaves are directly read off, otherwise a global clustering step
		target_names these subclusters into global clusters (target_names) and the samples are
		mapped to the global label of the nearest subcluster.

	"""
	birch_clustering: skc.Birch
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]


	def __init__( self, num: int=3 ) -> None:
		"""

			Purpose:
			---------
			Initialize Birch clustering.

			Parameters:
			----------
			num: Optional[int]

		"""
		super( ).__init__( )
		self.birch_clustering = skc.Birch( n_clusters=num )
		self.prediction = None
		self.accuracy = 0.0


	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> BirchCluster | None:
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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.birch_clustering.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchCluster'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.prediction = self.birch_clustering.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchCluster'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> float | None:
		"""

			Purpose:
			---------
			Evaluate Birch clustering.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.birch_clustering.predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchCluster'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.birch_clustering.predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'Dark2' )
				plt.title( 'Birch Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchCluster'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class OpticsCluster( Cluster ):
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
	optics_clustering: skc.OPTICS
	min_samples: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]


	def __init__( self, min: int=5 ) -> None:
		"""

			Purpose:
			---------
			Initialize OPTICS model.

			Parameters:
			----------
			min: int

		"""
		super( ).__init__( )
		self.min_samples = min
		self.optics_clustering = skc.OPTICS( min_samples=self.min_samples )
		self.prediction = None
		self.accuracy = 0.0


	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> OpticsCluster | None:
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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.optics_clustering.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsCluster'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.prediction = self.optics_clustering.fit_predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsCluster'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: Optional[ np.ndarray]=None ) -> float | None:
		"""

			Purpose:
			---------
			Evaluate OPTICS clustering.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (Optional[np.ndarray]): Optional target array  of shape ( n_samples, ).

			Returns:
			---------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.optics_clustering.fit_predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsCluster'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
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
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.optics_clustering.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'rainbow' )
				plt.title( 'OPTICS Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsCluster'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )