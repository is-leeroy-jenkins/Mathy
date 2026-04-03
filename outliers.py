'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                outliers.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="outliers.py" company="Terry D. Eppler">

	     outliers.py
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
    outliers.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations
from typing import Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.ensemble as en
import sklearn.svm as sv
import sklearn.neighbors as nn
import sklearn.covariance as cv
from sklearn.metrics import classification_report
from boogr import Error


def throw_if( name: str, value: object ) -> None:
	if value is None:
		raise Exception( f'Argument "{name}" cannot be empty!' )


class Outlier( ):
	"""

		Purpose:
		---------
		Abstract base class that defines the interface for all linerar_model wrappers.

	"""
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	anomaly_scores: Optional[ float ]
	learning_rate: Optional[ float ]
	outliers: Optional[ float ]
	inliers: Optional[ float ]
	
	def __init__( self ):
		self.outliers = 0.0
		self.inliers = 0.0
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
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
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Generate predictions from  the trained linerar_model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True target target_names.

			Returns:
			-----------
			np.ndarray: Predicted target_names or class target_names.

		"""
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> float:
		"""

			Purpose:
			---------
			Compute the core metric (e.g., R²) of the model on test df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True target target_names.

			Returns:
			-----------
				float: Score value (e.g., R² for regressors).

		"""
		raise NotImplementedError
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			---------
			Evaluate the model using multiple performance metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
				y (np.ndarray): Ground truth target_names.

			Returns:
			-----------
				dict: Dictionary containing multiple evaluation metrics.

		"""
		raise NotImplementedError


class IsolationForest( Outlier ):
	"""
	
		Purpose:
		--------
		The IsolationForest ‘isolates’ observations by randomly selecting a feature and then
		randomly selecting a split value between the maximum and minimum values of
		the selected feature. Since recursive partitioning can be represented by a tree structure,
		the number of splittings required to isolate a sample is equivalent to the path
		length from the root node to the terminating node. This path length, averaged over a
		forest of such random trees, is a measure of normality and our decision function.

	"""
	model: en.IsolationForest
	contamination: float
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	
	def __init__( self, contamination: float=0.1 ) -> None:
		"""

			Purpose:
			--------
			Initialize the IsolationForest wrapper with the specified contamination rate.

			Parameters:
			-----------
			contamination (float): Expected proportion of outliers in the data.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.contamination = contamination
		self.model = en.IsolationForest( contamination=self.contamination )
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> IsolationForest | None:
		"""

			Purpose:
			--------
			Fit the IsolationForest model on the supplied feature matrix.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored by IsolationForest; present for API consistency.

			Returns:
			--------
			IsolationForest | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.predict( X )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( self.prediction == -1 ) )
			self.inliers = float( np.sum( self.prediction == 1 ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IsolationForest'
			exception.method = 'train'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Predict whether each sample is an inlier or outlier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			np.ndarray | None: Prediction vector where 1 denotes inlier and -1 denotes outlier.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IsolationForest'
			exception.method = 'project'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""

			Purpose:
			--------
			Compute per-sample anomaly outputs and summary-friendly columns for review.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			pd.DataFrame | None: DataFrame containing per-sample prediction, anomaly score,
			inlier flag, and outlier flag.

		"""
		try:
			throw_if( 'X', X )
			y_pred = self.project( X, y )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( y_pred == -1 ) )
			self.inliers = float( np.sum( y_pred == 1 ) )
			df_scores = pd.DataFrame( )
			df_scores[ 'Prediction' ] = y_pred
			df_scores[ 'Anomaly' ] = self.anomaly_scores
			df_scores[ 'Inlier' ] = (y_pred == 1).astype( int )
			df_scores[ 'Outlier' ] = (y_pred == -1).astype( int )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IsolationForest'
			exception.method = 'score'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""

			Purpose:
			--------
			Create a compact anomaly summary and render a bar chart of inlier versus outlier
			counts for the supplied samples.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			pd.DataFrame | None: Summary DataFrame containing aggregate anomaly counts and
			quality statistics.

		"""
		try:
			throw_if( 'X', X )
			df_scores = self.score( X, y )
			df_summary = pd.DataFrame(
			{
				'Metric': [ 'Inliers', 'Outliers', 'Contamination', 'Quality' ],
				'Value': [ float( self.inliers ), float( self.outliers ),
						float( self.model.contamination ),
						float( round( self.inliers / len( df_scores ), 4 ) ) ]
			} )
			
			df_plot = pd.DataFrame(
			{
				'Label': [ 'Inliers', 'Outliers' ],
				'Count': [ float( self.inliers ), float( self.outliers ) ]
			} )
			
			plt.figure( figsize=(8, 6) )
			sns.barplot( data=df_plot, x='Label', y='Count' )
			plt.title( 'IsolationForest Detection Summary' )
			plt.xlabel( 'Classification' )
			plt.ylabel( 'Count' )
			plt.tight_layout( )
			plt.show( )
			
			return df_summary
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IsolationForest'
			exception.method = 'analyze'
			raise exception
			

class OneClass( Outlier ):
	"""
	
		Purpose:
		--------
		Encapsulates scikit-learn's OneClassSVM for novelty detection on high-dimensional data.
		The estimator learns a boundary around normal samples and flags observations
		outside that boundary as anomalies.

	"""
	model: Optional[ sv.OneClassSVM ]
	data: Optional[ np.ndarray ]
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	kernel: Optional[ str ]
	
	def __init__( self, kernel: str = 'rbf', nu: float=0.05, gamma: str='scale' ) -> None:
		"""

			Purpose:
			--------
			Initialize the OneClassSVM wrapper with the specified kernel parameters.

			Parameters:
			-----------
			kernel (str): Kernel type used by the model.
			nu (float): Upper bound on the fraction of training errors and lower bound
				on the fraction of support vectors.
			gamma (str): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.kernel = kernel
		self.model = sv.OneClassSVM( kernel=kernel, nu=nu, gamma=gamma )
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> OneClass | None:
		"""

			Purpose:
			--------
			Fit the OneClassSVM model using feature data that represents the normal class.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			OneClass | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.predict( X )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( self.prediction == -1 ) )
			self.inliers = float( np.sum( self.prediction == 1 ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneClass'
			exception.method = 'train'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Predict whether each supplied sample is an inlier or an outlier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			np.ndarray | None: Prediction vector where 1 denotes inlier and -1 denotes
			outlier.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneClass'
			exception.method = 'project'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""

			Purpose:
			--------
			Compute per-sample OneClassSVM predictions and anomaly scores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			pd.DataFrame | None: DataFrame containing prediction, anomaly score,
			inlier flag, and outlier flag for each sample.

		"""
		try:
			throw_if( 'X', X )
			y_pred = self.project( X, y )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( y_pred == -1 ) )
			self.inliers = float( np.sum( y_pred == 1 ) )
			
			df_scores = pd.DataFrame( )
			df_scores[ 'Prediction' ] = y_pred
			df_scores[ 'Anomaly' ] = self.anomaly_scores
			df_scores[ 'Inlier' ] = (y_pred == 1).astype( int )
			df_scores[ 'Outlier' ] = (y_pred == -1).astype( int )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneClass'
			exception.method = 'score'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""

			Purpose:
			--------
			Create a compact novelty-detection summary and render a bar chart of inlier
			versus outlier counts.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			pd.DataFrame | None: Summary DataFrame containing aggregate anomaly
			detection metrics.

		"""
		try:
			throw_if( 'X', X )
			df_scores = self.score( X, y )
			df_summary = pd.DataFrame(
			{
				'Metric': [ 'Inliers', 'Outliers', 'Quality' ],
				'Value': [ float( self.inliers ), float( self.outliers ),
						float( round( self.inliers / len( df_scores ), 4 ) ) ]
			} )
			
			df_plot = pd.DataFrame(
			{
				'Label': [ 'Inliers', 'Outliers' ],
				'Count': [ float( self.inliers ), float( self.outliers ) ]
			} )
			
			plt.figure( figsize=(8, 6) )
			sns.barplot( data=df_plot, x='Label', y='Count' )
			plt.title( 'OneClassSVM Detection Summary' )
			plt.xlabel( 'Classification' )
			plt.ylabel( 'Count' )
			plt.tight_layout( )
			plt.show( )
			
			return df_summary
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneClass'
			exception.method = 'analyze'
			raise exception
			

class OutlierFactor( Outlier ):
	"""
	
		Purpose:
		--------
		LocalOutlierFactor for unsupervised or novelty-based outlier detection.
		Provides decision function, prediction, and scoring interfaces.

	"""
	model: nn.LocalOutlierFactor
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	neighbors: Optional[ int ]
	contamination: Optional[ float ]
	novelty: Optional[ bool ]
	
	def __init__( self, n_neighbors: int = 20, contamination: float = 0.1,
			novelty: bool = True ) -> None:
		"""

			Purpose:
			--------
			Initialize the LocalOutlierFactor wrapper with neighborhood, contamination,
			and novelty-detection settings.

			Parameters:
			-----------
			n_neighbors (int): Number of neighbors used for local density estimation.
			contamination (float): Estimated fraction of outliers in the data.
			novelty (bool): If True, enables novelty detection on unseen samples.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.neighbors = n_neighbors
		self.contamination = contamination
		self.novelty = novelty
		self.model = nn.LocalOutlierFactor(
			n_neighbors=self.neighbors,
			contamination=self.contamination,
			novelty=self.novelty
		)
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> OutlierFactor | None:
		"""

			Purpose:
			--------
			Fit the LocalOutlierFactor model on the supplied feature matrix.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			OutlierFactor | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			if self.novelty:
				self.model.fit( X )
				self.prediction = None
				self.anomaly_scores = None
			else:
				self.prediction = self.model.fit_predict( X )
				self.anomaly_scores = self.model.negative_outlier_factor_
				self.outliers = float( np.sum( self.prediction == -1 ) )
				self.inliers = float( np.sum( self.prediction == 1 ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'train'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Predict whether each sample is an inlier or outlier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			np.ndarray | None: Prediction vector where 1 denotes inlier and -1 denotes
			outlier.

		"""
		try:
			throw_if( 'X', X )
			if self.novelty:
				self.prediction = self.model.predict( X )
			else:
				self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'project'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""

			Purpose:
			--------
			Compute per-sample LOF predictions and anomaly scores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			pd.DataFrame | None: DataFrame containing per-sample prediction, anomaly
			score, inlier flag, and outlier flag.

		"""
		try:
			throw_if( 'X', X )
			if self.novelty:
				y_pred = self.project( X, y )
				self.anomaly_scores = self.model.decision_function( X )
			else:
				y_pred = self.model.fit_predict( X )
				self.prediction = y_pred
				self.anomaly_scores = self.model.negative_outlier_factor_
			
			self.outliers = float( np.sum( y_pred == -1 ) )
			self.inliers = float( np.sum( y_pred == 1 ) )
			
			df_scores = pd.DataFrame( )
			df_scores[ 'Prediction' ] = y_pred
			df_scores[ 'Anomaly' ] = self.anomaly_scores
			df_scores[ 'Inlier' ] = (y_pred == 1).astype( int )
			df_scores[ 'Outlier' ] = (y_pred == -1).astype( int )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'score'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""

			Purpose:
			--------
			Create a compact LOF anomaly summary and render a bar chart of inlier versus
			outlier counts.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			pd.DataFrame | None: Summary DataFrame containing aggregate anomaly
			detection metrics.

		"""
		try:
			throw_if( 'X', X )
			df_scores = self.score( X, y )		
			df_summary = pd.DataFrame(
			{
				'Metric': [ 'Inliers', 'Outliers', 'Contamination', 'Quality' ],
				'Value': [ float( self.inliers ), float( self.outliers ), 
				           float( self.contamination ), 
				           float( round( self.inliers / len( df_scores ), 4 ) ) ]
			} )
			
			df_plot = pd.DataFrame(
			{
					'Label': [ 'Inliers', 'Outliers' ],
					'Count': [ float( self.inliers ), float( self.outliers ) ]
			} )
			
			plt.figure( figsize=(8, 6) )
			sns.barplot( data=df_plot, x='Label', y='Count' )
			plt.title( 'LocalOutlierFactor Detection Summary' )
			plt.xlabel( 'Classification' )
			plt.ylabel( 'Count' )
			plt.tight_layout( )
			plt.show( )
			
			return df_summary
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'analyze'
			raise exception
			
			
class EllipticSquare( Outlier ):
	"""

		Purpose:
		--------
		Encapsulates  EllipticEnvelope for multivariate Gaussian-based outlier detection.
		This method is based on Mahalanobis distances under an elliptical (normal) distribution.

	"""
	model: cv.EllipticEnvelope
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	contamination: Optional[ float ]
	
	def __init__( self, contamination: float = 0.1 ) -> None:
		"""

			Purpose:
			--------
			Initialize the EllipticEnvelope wrapper with the specified contamination rate.

			Parameters:
			-----------
			contamination (float): Estimated proportion of outliers in the dataset.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.contamination = contamination
		self.model = cv.EllipticEnvelope( contamination=contamination )
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> EllipticSquare | None:
		"""

			Purpose:
			--------
			Fit the EllipticEnvelope model to the supplied feature matrix.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			EllipticSquare | None: The fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.predict( X )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( self.prediction == -1 ) )
			self.inliers = float( np.sum( self.prediction == 1 ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'EllipticSquare'
			exception.method = 'train'
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Predict whether each supplied sample is an inlier or outlier.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			np.ndarray | None: Prediction vector where 1 denotes inlier and -1 denotes
			outlier.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'EllipticSquare'
			exception.method = 'project'
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""

			Purpose:
			--------
			Compute per-sample EllipticEnvelope predictions and anomaly scores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			pd.DataFrame | None: DataFrame containing prediction, anomaly score,
			inlier flag, and outlier flag for each sample.

		"""
		try:
			throw_if( 'X', X )
			y_pred = self.project( X, y )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( y_pred == -1 ) )
			self.inliers = float( np.sum( y_pred == 1 ) )
			
			df_scores = pd.DataFrame( )
			df_scores[ 'Prediction' ] = y_pred
			df_scores[ 'Anomaly' ] = self.anomaly_scores
			df_scores[ 'Inlier' ] = (y_pred == 1).astype( int )
			df_scores[ 'Outlier' ] = (y_pred == -1).astype( int )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'EllipticSquare'
			exception.method = 'score'
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""

			Purpose:
			--------
			Create a compact EllipticEnvelope summary and render a bar chart of inlier
			versus outlier counts.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (Optional[np.ndarray]): Ignored; present for API consistency.

			Returns:
			--------
			pd.DataFrame | None: Summary DataFrame containing aggregate anomaly
			detection metrics.

		"""
		try:
			throw_if( 'X', X )
			df_scores = self.score( X, y )			
			df_summary = pd.DataFrame(
			{
				'Metric': [ 'Inliers', 'Outliers', 'Contamination', 'Quality' ],
				'Value': [ float( self.inliers ), float( self.outliers ),
						float( self.contamination ),
						float( round( self.inliers / len( df_scores ), 4 ) ) ]
			} )
			
			df_plot = pd.DataFrame(
			{
				'Label': [ 'Inliers', 'Outliers' ],
				'Count': [ float( self.inliers ), float( self.outliers ) ]
			} )
			
			plt.figure( figsize=(8, 6) )
			sns.barplot( data=df_plot, x='Label', y='Count' )
			plt.title( 'EllipticEnvelope Detection Summary' )
			plt.xlabel( 'Classification' )
			plt.ylabel( 'Count' )
			plt.tight_layout( )
			plt.show( )
			
			return df_summary
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'EllipticSquare'
			exception.method = 'analyze'
			raise exception
