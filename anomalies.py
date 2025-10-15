'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                anomalies.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="anomalies.py" company="Terry D. Eppler">

	     anomalies.py
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
    anomalies.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations
from typing import Optional, Dict
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import classification_report
from boogr import Error, ErrorDialog

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
	
	def __init__( self ):
		pass
	
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
	model: IsolationForest
	contamination: float
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	
	def __init__( self, contamination: float=0.1 ) -> None:
		"""

			Purpose:
			--------
			Initialize IsolationForestWrapper with a contamination threshold.
	
			Parameters:
			-----------
			contamination (float): Expected proportion of outliers in the data.
			kwargs (dict): Additional keyword arguments passed to IsolationForest.
	
			Returns:
			--------
			None

		"""
		self.contamination = contamination
		self.model = IsolationForest( contamination=self.contamination )
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> IsolationForest | None:
		"""

			Purpose:
			--------
			Train the IsolationForest model on input data.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
	
			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.predict( X )  # -1 = outlier, 1 = inlier
			self.anomaly_scores = self.model.decision_function( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IsolationForest'
			exception.method = 'train'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray | None:
		"""
	
			Purpose:
			--------
			Predict whether new samples are inliers or outliers.
	
			Parameters:
			-----------
			X (np.ndarray): Feature matrix of unseen data.
	
			Returns:
			--------
			np.ndarray: Array of predictions (-1 = outlier, 1 = inlier).

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
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> float | None:
		"""
	
			Purpose:
			--------
			Returns the proportion of inliers detected during training.
	
			Returns:
			--------
			float: Percentage of samples labeled as inliers.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return np.mean( self.prediction == 1 )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IsolationForest'
			exception.method = 'score'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y:np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			--------
			Evaluates outlier detection results, optionally against ground-truth labels.
	
			Parameters:
			-----------
			true_labels (Optional[np.ndarray]): Actual binary labels (1 = inlier, -1 = outlier).
	
			Returns:
			--------
			Dict: Classification report or descriptive summary.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			outliers = int( np.sum( self.prediction == -1 ) )
			inliers = int( np.sum( self.prediction == 1 ) )
			return \
			{
				'Outliers': float( outliers ),
				'Inliers': float( inliers ),
				'Contamination': float( self.model.contamination ),
				'Quality': float( round( inliers / len( self.prediction ), 4 ) )
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IsolationForest'
			exception.method = 'analyze'
			error = ErrorDialog( exception )
			error.show( )

class OneClass( Outlier ):
	"""
	
		Purpose:
		--------
		Encapsulates scikit-learn's OneClassSVM for novelty detection on high-dimensional data.
		The model learns a boundary around "normal" samples and identifies novel deviations.

	"""
	model: Optional[ OneClassSVM ]
	data: Optional[ np.ndarray ]
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	kernel: Optional[ str ]
	
	def __init__( self, kernel: str='rbf', nu: float=0.05, gamma: str='scale' ) -> None:
		"""
	
			Purpose:
			--------
			Initializes the OneClassSVM model with the specified kernel and hyperparameters.
	
			Parameters:
			-----------
			kernel (str): Kernel type to be used in the SVM ('linear', 'poly', 'rbf', 'sigmoid').
			nu (float): An upper bound on the fraction of training errors (0 < nu ≤ 1).
			gamma (str): Kernel coefficient ('scale', 'auto', or float).
			kwargs (dict): Additional arguments passed to OneClassSVM.
	
			Returns:
			--------
			None

		"""
		self.model = OneClassSVM( kernel=kernel, nu=nu, gamma=gamma )
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> OneClass | None:
		"""

			Purpose:
			--------
			Fits the OneClassSVM model using only inlier training data.
	
			Parameters:
			-----------
			X (np.ndarray): Input training data (inlier-only).
	
			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )  # -1 = outlier, 1 = inlier
			self.anomaly_scores = self.model.decision_function( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'train'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Applies the trained model to new samples to detect novel anomalies.
	
			Parameters:
			-----------
			X (np.ndarray): New samples to evaluate.
	
			Returns:
			--------
			np.ndarray: Prediction array (1 = inlier, -1 = novel/outlier).

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'project'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> float | None:
		"""
	
			Purpose:
			--------
			Returns the percentage of samples classified as inliers during training.
	
			Returns:
			--------
			float: Proportion of training samples classified as inliers.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			_score = np.mean( self.prediction == 1 )
			return _score
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'score'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y:np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			--------
			Evaluates outlier detection results, optionally against ground-truth labels.
	
			Parameters:
			-----------
			true_labels (Optional[np.ndarray]): Actual binary labels (1 = inlier, -1 = outlier).
	
			Returns:
			--------
			Dict: Classification report or descriptive summary.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			outliers = int( np.sum( self.prediction == -1 ) )
			inliers = int( np.sum( self.prediction == 1 ) )
			return \
			{
				'Outliers': float( outliers ),
				'Inliers': float( inliers ),
				'Contamination': float( self.model.contamination ),
				'Quality': float( round( inliers / len( self.prediction ), 4 ) )
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'analyze'
			error = ErrorDialog( exception )
			error.show( )

class OutlierFactor( Outlier ):
	"""
	
		Purpose:
		--------
		Wraps scikit-learn's LocalOutlierFactor for unsupervised or novelty-based outlier detection.
		Provides decision function, prediction, and scoring interfaces.

	"""
	model: LocalOutlierFactor
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	neighbors: Optional[ int ]
	containment: Optional[ float]
	
	def __init__( self, n_neighbors: int=20, contamination: float=0.1, novelty: bool=True ) -> None:
		"""
	
			Purpose:
			--------
			Initializes the LOF model with neighborhood and contamination settings.
	
			Parameters:
			-----------
			n_neighbors (int): Number of neighbors to use for local density estimation.
			contamination (float): Estimated fraction of outliers in the data.
			novelty (bool): If True, enables prediction on unseen data (novelty detection).
			kwargs (dict): Additional arguments for LocalOutlierFactor.
	
			Returns:
			--------
			None

		"""
		self.neighbors = n_neighbors
		self.contamination = contamination
		self.novelty = novelty
		self.model = LocalOutlierFactor( n_neighbors=self.neighbors,
			contamination=self.contamination, novelty=self.novelty )
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> LocalOutlierFactor | None:
		"""

			Purpose:
			--------
			Fit the LOF model to inlier-only data (for novelty detection).
	
			Parameters:
			-----------
			X (np.ndarray): Training samples presumed to be normal (non-outliers).
	
			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.predict( X )
			self.anomaly_scores = self.model.decision_function( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'train'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray | None:
		"""
	
			Purpose:
			--------
			Applies the trained LOF model to detect outliers in unseen samples.
	
			Parameters:
			-----------
			X (np.ndarray): Test samples.
	
			Returns:
			--------
			np.ndarray: Predictions (-1 = outlier, 1 = inlier).

		"""
		try:
			throw_if( 'X', X )
			self.predicate = self.model.predict( X )
			return self.predicate
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'project'
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> float | None:
		"""
	
			Purpose:
			--------
			Computes the proportion of training samples classified as inliers.
	
			Returns:
			--------
			float: Fraction of inliers.

		"""
		try:
			throw_if( 'predicate', self.predicate )
			return np.mean( self.predicate == 1 )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'score'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y:np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			--------
			Evaluates outlier detection results, optionally against ground-truth labels.
	
			Parameters:
			-----------
			true_labels (Optional[np.ndarray]): Actual binary labels (1 = inlier, -1 = outlier).
	
			Returns:
			--------
			Dict: Classification report or descriptive summary.

		"""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.model.predict( X )
			outliers = int( np.sum( self.prediction == -1 ) )
			inliers = int( np.sum( self.prediction == 1 ) )
			return \
			{
				'Outliers': float( outliers ),
				'Inliers': float( inliers ),
				'Contamination': float( self.model.contamination ),
				'Quality': float( round( inliers / len( self.prediction ), 4 ) )
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'analyze'
			error = ErrorDialog( exception )
			error.show( )

class EllipticSquare( Outlier ):
	"""

		Purpose:
		--------
		Encapsulates scikit-learn's EllipticEnvelope for multivariate Gaussian-based outlier detection.
		This method is based on Mahalanobis distances under an elliptical (normal) distribution.

	"""
	model: EllipticEnvelope
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	
	def __init__( self, contamination: float=0.1 ) -> None:
		"""
	
			Purpose:
			--------
			Initializes the EllipticEnvelope with a contamination rate.
	
			Parameters:
			-----------
			contamination (float): Estimated proportion of outliers in the dataset.
			kwargs (dict): Additional keyword arguments passed to EllipticEnvelope.
	
			Returns:
			--------
			None

		"""
		self.model = EllipticEnvelope( contamination=contamination )
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]  ) -> EllipticSquare | None:
		"""

			Purpose:
			--------
			Fit the Gaussian envelope model to multivariate data.
	
			Parameters:
			-----------
			X (np.ndarray): Input matrix of inlier training samples.
		
			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.predict( X )
			self.anomaly_scores = self.model.decision_function( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'EllipticSquare'
			exception.method = 'train'
			error = ErrorDialog( exception )
			error.show( )
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Classifies new samples as inliers or outliers using Mahalanobis distance.
	
			Parameters:
			-----------
			X (np.ndarray): New observations to classify.
	
			Returns:
			--------
			np.ndarray: Array of predictions (1 = inlier, -1 = outlier).

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
			error = ErrorDialog( exception )
			error.show( )
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> float | None:
		"""
	
			Purpose:
			--------
			Computes proportion of inliers detected during training.
	
			Returns:
			--------
			float: Fraction of training samples classified as inliers.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return np.mean( self.prediction == 1 )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'EllipticSquare'
			exception.method = 'score'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, X: np.ndarray, y:np.ndarray ) -> Dict[ str, float ] | None:
		"""

			Purpose:
			--------
			Evaluates outlier detection results, optionally against ground-truth labels.
	
			Parameters:
			-----------
			true_labels (Optional[np.ndarray]): Actual binary labels (1 = inlier, -1 = outlier).
	
			Returns:
			--------
			Dict: Classification report or descriptive summary.

		"""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			outliers = int( np.sum( self.prediction == -1 ) )
			inliers = int( np.sum( self.prediction == 1 ) )
			return \
			{
				'Outliers': float( outliers ),
				'Inliers': float( inliers ),
				'Contamination': float( self.model.contamination ),
				'Quality': float( round( inliers / len( self.prediction ), 4 ) )
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'EllipticSquare'
			exception.method = 'analyze'
			error = ErrorDialog( exception )
			error.show( )