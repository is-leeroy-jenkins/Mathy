'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
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
    name.py
  </summary>
  ******************************************************************************************
  '''
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import sklearn.feature_extraction as fe
import sklearn.feature_extraction.text as sk
import sklearn.preprocessing as pp
import sklearn.compose as sc
from boogr import Error


def throw_if( name: str, value: object ):
	if not value:
		raise ValueError( f'Argument "{name}" cannot be empty!' )


class Transformer( ):
	"""

		Purpose:
		---------
		Base interface for all preprocessors. Provides standard `train`, `transform`,
		`train_transform`, and `inverse_transform` methods.

	"""
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the base transformer state.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		"""
		self.transformed_data = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
		"""

			Purpose:
			---------
			Train hook for concrete subclasses.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array of shape ( n_samples, ).

			Returns:
			--------
			object | None: Concrete fitted transformer instance.

		"""
		raise NotImplementedError( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform input features using a fitted transformer.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError( )
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit to X, optionally using y, and return the transformed feature matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array of shape ( n_samples, ).

			Returns:
			--------
			np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError( )
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Inverse a transformation when supported by the concrete subclass.

			Parameters:
			-----------
			X ( np.ndarray ): Transformed feature matrix.

			Returns:
			--------
			np.ndarray: Inverse-transformed data.

		"""
		raise NotImplementedError( )


class Binarizer( Transformer ):
	"""

		Purpose:
		_______
		Binarize data (set feature values to 0 or 1) according to a threshold.
		Values greater than the threshold map to 1, while values less than or equal to the
		threshold map to 0. With the default threshold of 0, only positive values map to 1.
		
		Binarization is a common operation on text count data where the analyst can decide to only
		consider the presence or absence of a feature rather than a quantified number of
		occurrences for instance.

		It can also be used as a pre-processing step for estimators that consider boolean random
		variables (e.g. modelled using the Bernoulli distribution in a Bayesian setting).

	"""
	model: pp.Binarizer
	threshold: Optional[ float ]
	copy: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, threshold: float = 0.0, copy: bool = True ) -> None:
		"""

			Purpose:
			_______
			Initialize the Binarizer wrapper.

			Parameters:
			-----------
			threshold ( float ): Threshold used to binarize values.
			copy ( bool ): Indicates whether to perform the transformation on a copy.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.threshold = threshold
		self.copy = copy
		self.model = pp.Binarizer( threshold=self.threshold, copy=self.copy )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			_______
			Return a list of class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Member names exposed by the wrapper.

		"""
		return [ 'threshold',
		         'copy',
		         'model',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Binarizer | None:
		"""

			Purpose:
			_______
			Fit the binarizer on X. In sklearn, fit validates parameters and establishes the
			feature-count metadata for the estimator API.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored.

			Returns:
			--------
			Binarizer | None: Fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Binarizer'
			exception.method = 'train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Binarizer'
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Binarize X using the configured threshold.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Binarized feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Binarizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			_______
			Fit the binarizer on X and return the binarized output.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored.

			Returns:
			--------
			np.ndarray: Binarized feature matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Binarizer'
			exception.method = 'train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception


class LabelBinarizer( Transformer ):
	"""

		Purpose:
		_______
		Binarize labels in a one-vs-all fashion. This wrapper fits on target labels and
		transforms them to a binary matrix representation. It also supports converting the
		binary representation back to the original labels.

	"""
	model: pp.LabelBinarizer
	pos_label: Optional[ int ]
	neg_label: Optional[ int ]
	sparse_output: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, pos_label: int = 1, neg_label: int = 0,
			sparse_output: bool = False ) -> None:
		"""

			Purpose:
			_______
			Initialize the LabelBinarizer wrapper.

			Parameters:
			-----------
			pos_label ( int ): Value with which positive labels must be encoded.
			neg_label ( int ): Value with which negative labels must be encoded.
			sparse_output ( bool ): Indicates whether the transform should return a sparse
				matrix.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.pos_label = pos_label
		self.neg_label = neg_label
		self.sparse_output = sparse_output
		self.model = pp.LabelBinarizer( neg_label=self.neg_label,
			pos_label=self.pos_label, sparse_output=self.sparse_output )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			_______
			Return a list of class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Member names exposed by the wrapper.

		"""
		return [ 'pos_label',
		         'neg_label',
		         'sparse_output',
		         'model',
		         'transformed_data',
		         'classes',
		         'types',
		         'train',
		         'transform',
		         'train_transform',
		         'inverse_transform' ]
	
	@property
	def classes( self ) -> List[ str ]:
		"""

			Purpose:
			_______
			Return the learned class labels.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Learned class labels.

		"""
		if getattr( self.model, 'classes_', None ) is None:
			raise AttributeError( 'LabelBinarizer has not been initialized.' )
		else:
			return self.model.classes_
	
	@property
	def types( self ) -> str:
		"""

			Purpose:
			_______
			Return the inferred target type for the fitted labels.

			Parameters:
			-----------
			None

			Returns:
			--------
			str: Inferred target type.

		"""
		if getattr( self.model, 'y_type_', None ) is None:
			raise AttributeError( 'LabelBinarizer has not been initialized.' )
		else:
			return self.model.y_type_
	
	def train( self, y: np.ndarray ) -> LabelBinarizer | None:
		"""

			Purpose:
			_______
			Fit the label binarizer on target labels.

			Parameters:
			-----------
			y ( np.ndarray ): Target vector of shape ( n_samples, ).

			Returns:
			--------
			LabelBinarizer | None: Fitted wrapper instance.

		"""
		try:
			throw_if( 'y', y )
			self.model.fit( y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'train( self, y: np.ndarray ) -> LabelBinarizer'
			raise exception
	
	def transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Transform target labels to a binary matrix.

			Parameters:
			-----------
			y ( np.ndarray ): Target vector of shape ( n_samples, ).

			Returns:
			--------
			np.ndarray: Binary-encoded label matrix.

		"""
		try:
			throw_if( 'y', y )
			self.transformed_data = self.model.transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'transform( self, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Fit on y and transform y to a binary matrix.

			Parameters:
			-----------
			y ( np.ndarray ): Target vector of shape ( n_samples, ).

			Returns:
			--------
			np.ndarray: Binary-encoded label matrix.

		"""
		try:
			throw_if( 'y', y )
			self.transformed_data = self.model.fit_transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'train_transform( self, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def inverse_transform( self, Y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Convert a binary matrix back to the original labels.

			Parameters:
			-----------
			Y ( np.ndarray ): Binary-encoded label matrix.

			Returns:
			--------
			np.ndarray: Original labels.

		"""
		try:
			throw_if( 'Y', Y )
			return self.model.inverse_transform( Y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'inverse_transform( self, Y: np.ndarray ) -> np.ndarray'
			raise exception


class MultiLabelBinarizer( Transformer ):
	"""

		Purpose:
		_______
		Transform between an iterable of iterables and the multilabel binary matrix format.
		Each row in the transformed output indicates the presence or absence of each class
		label for a given sample.

	"""
	model: pp.MultiLabelBinarizer
	sparse_output: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, classes: Optional[ np.ndarray ] = None,
			sparse_output: bool = False ) -> None:
		"""

			Purpose:
			_______
			Initialize the MultiLabelBinarizer wrapper.

			Parameters:
			-----------
			classes ( Optional[ np.ndarray ] ): Optional fixed ordering of class labels.
			sparse_output ( bool ): Indicates whether the transformed output should be
				returned as a sparse matrix.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.sparse_output = sparse_output
		self.model = pp.MultiLabelBinarizer( classes=classes,
			sparse_output=self.sparse_output )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			_______
			Return a list of class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Member names exposed by the wrapper.

		"""
		return [ 'sparse_output',
		         'model',
		         'transformed_data',
		         'classes',
		         'train',
		         'transform',
		         'train_transform',
		         'inverse_transform' ]
	
	@property
	def classes( self ) -> List[ str ]:
		"""

			Purpose:
			_______
			Return the learned class labels.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Learned class labels.

		"""
		if getattr( self.model, 'classes_', None ) is None:
			raise AttributeError( 'MultiLabelBinarizer has not been initialized.' )
		else:
			return self.model.classes_
	
	def train( self, y: np.ndarray ) -> MultiLabelBinarizer | None:
		"""

			Purpose:
			_______
			Fit the multi-label binarizer on multilabel targets.

			Parameters:
			-----------
			y ( np.ndarray ): Iterable of iterables containing labels for each sample.

			Returns:
			--------
			MultiLabelBinarizer | None: Fitted wrapper instance.

		"""
		try:
			throw_if( 'y', y )
			self.model.fit( y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'train( self, y: np.ndarray ) -> MultiLabelBinarizer'
			raise exception
	
	def transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Transform multilabel targets to a binary indicator matrix.

			Parameters:
			-----------
			y ( np.ndarray ): Iterable of iterables containing labels for each sample.

			Returns:
			--------
			np.ndarray: Binary-encoded multilabel matrix.

		"""
		try:
			throw_if( 'y', y )
			self.transformed_data = self.model.transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'transform( self, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Fit on multilabel targets and return the binary indicator matrix.

			Parameters:
			-----------
			y ( np.ndarray ): Iterable of iterables containing labels for each sample.

			Returns:
			--------
			np.ndarray: Binary-encoded multilabel matrix.

		"""
		try:
			throw_if( 'y', y )
			self.transformed_data = self.model.fit_transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'train_transform( self, y: np.ndarray ) -> np.ndarray'
			raise exception
	
	def inverse_transform( self, Y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Convert a binary multilabel matrix back to the original label collections.

			Parameters:
			-----------
			Y ( np.ndarray ): Binary-encoded multilabel matrix.

			Returns:
			--------
			np.ndarray: Original multilabel collections.

		"""
		try:
			throw_if( 'Y', Y )
			return self.model.inverse_transform( Y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'inverse_transform( self, Y: np.ndarray ) -> np.ndarray'
			raise exception


class TfidfTransformer( Transformer ):
	"""

		Purpose:
		---------
		Transform a count matrix to a normalized tf or tf-idf representation. Tf means
		term-frequency while tf-idf means term-frequency times inverse document-frequency.
		This is a common term-weighting scheme in information retrieval and document
		classification.

	"""
	model: sk.TfidfTransformer
	norm: Optional[ str ]
	use_idf: Optional[ bool ]
	smooth_idf: Optional[ bool ]
	sublinear_tf: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, norm: str = 'l2', use_idf: bool = True,
			smooth_idf: bool = True, sublinear_tf: bool = False ) -> None:
		"""

			Purpose:
			---------
			Initialize the TfidfTransformer wrapper.

			Parameters:
			-----------
			norm ( str ): Norm used to normalize term vectors.
			use_idf ( bool ): Indicates whether inverse-document-frequency reweighting
				should be enabled.
			smooth_idf ( bool ): Indicates whether document frequencies should be
				smoothed by adding one to document frequencies.
			sublinear_tf ( bool ): Indicates whether sublinear term-frequency scaling
				should be applied.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.norm = norm
		self.use_idf = use_idf
		self.smooth_idf = smooth_idf
		self.sublinear_tf = sublinear_tf
		self.model = sk.TfidfTransformer( norm=self.norm, use_idf=self.use_idf,
			smooth_idf=self.smooth_idf, sublinear_tf=self.sublinear_tf )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Return a list of class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Member names exposed by the wrapper.

		"""
		return [ 'norm',
		         'use_idf',
		         'smooth_idf',
		         'sublinear_tf',
		         'model',
		         'transformed_data',
		         'idf_vector',
		         'features',
		         'train',
		         'transform',
		         'train_transform' ]
	
	@property
	def idf_vector( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the inverse document frequency vector learned during fitting.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray: Inverse document frequency vector.

		"""
		if getattr( self.model, 'idf_', None ) is None:
			raise AttributeError( 'TfidfTransformer must be initialized.' )
		else:
			return self.model.idf_
	
	@property
	def features( self ) -> int:
		"""

			Purpose:
			---------
			Return the number of features observed during fitting.

			Parameters:
			-----------
			None

			Returns:
			--------
			int: Number of observed input features.

		"""
		if getattr( self.model, 'n_features_in_', None ) is None:
			raise AttributeError( 'TfidfTransformer must be initialized.' )
		else:
			return self.model.n_features_in_
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> TfidfTransformer | None:
		"""

			Purpose:
			---------
			Fit the transformer to a count matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Count matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored by the estimator.

			Returns:
			--------
			TfidfTransformer | None: Fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> TfidfTransformer'
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform a count matrix to a TF-IDF representation.

			Parameters:
			-----------
			X ( np.ndarray ): Count matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Dense TF-IDF-weighted document-term matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X, copy=True ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the transformer and return the TF-IDF-transformed matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Count matrix of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored by the estimator.

			Returns:
			--------
			np.ndarray: Dense TF-IDF-weighted document-term matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X, y ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
			

class ColumnTransformer( Transformer ):
	"""

		Purpose:
		---------
		Applies transformers to columns of an array or pandas DataFrame.
		This estimator allows different columns or column subsets of the input to be transformed
		separately and the features generated by each transformer will be concatenated to form
		a single feature space. This is useful for heterogeneous or columnar data,
		to combine several feature extraction mechanisms or transformations
		into a single transformer.

	"""
	model: sc.ColumnTransformer
	transformers: Optional[ List[ Tuple[ str, object, List[ str ] ] ] ]
	remainder: Optional[ str ]
	transformer_weights: Optional[ Dict[ str, float ] ]
	sparse_threshold: Optional[ float ]
	n_jobs: Optional[ int ]
	verbose: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, transformers: List[ Tuple[ str, object, List[ str ] ] ],
			remainder: str = 'drop', sparse_threshold: float = 0.3,
			n_jobs: Optional[ int ] = None,
			transformer_weights: Optional[ Dict[ str, float ] ] = None,
			verbose: bool = False ) -> None:
		"""

			Purpose:
			---------
			Initialize the ColumnTransformer wrapper.

			Parameters:
			-----------
			transformers ( List[ Tuple[ str, object, List[ str ] ] ] ): List of
				(name, transformer, columns) tuples.
			remainder ( str ): Handling for non-specified columns.
			sparse_threshold ( float ): Threshold for sparse stacking behavior.
			n_jobs ( Optional[ int ] ): Number of jobs to run in parallel.
			transformer_weights ( Optional[ Dict[ str, float ] ] ): Optional weights
				applied to transformer outputs.
			verbose ( bool ): Indicates whether execution timing should be printed.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.transformers = transformers
		self.remainder = remainder
		self.sparse_threshold = sparse_threshold
		self.n_jobs = n_jobs
		self.transformer_weights = transformer_weights
		self.verbose = verbose
		self.model = sc.ColumnTransformer(
			transformers=self.transformers,
			remainder=self.remainder,
			sparse_threshold=self.sparse_threshold,
			n_jobs=self.n_jobs,
			transformer_weights=self.transformer_weights,
			verbose=self.verbose )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Return a list of class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Member names exposed by the wrapper.

		"""
		return [ 'model',
		         'transformers',
		         'remainder',
		         'transformer_weights',
		         'sparse_threshold',
		         'n_jobs',
		         'verbose',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform' ]
	
	def train( self, X: np.ndarray,
			y: Optional[ np.ndarray ] = None ) -> ColumnTransformer | None:
		"""

			Purpose:
			---------
			Fit all configured transformers using X and optional y.

			Parameters:
			-----------
			X ( np.ndarray ): Input data of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array passed through to
				underlying transformers that accept it.

			Returns:
			--------
			ColumnTransformer | None: Fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ColumnTransformer'
			exception.method = 'train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> ColumnTransformer'
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform X using the fitted column transformers and return a dense matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Input data of shape ( n_samples, n_features ).

			Returns:
			--------
			np.ndarray: Dense transformed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			result = self.model.transform( X )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ColumnTransformer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, X: np.ndarray,
			y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit all transformers, transform X, and concatenate the results into a dense
			matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Input data of shape ( n_samples, n_features ).
			y ( Optional[ np.ndarray ] ): Optional target array passed through to
				underlying transformers that accept it.

			Returns:
			--------
			np.ndarray: Dense transformed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			result = self.model.fit_transform( X, y )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ColumnTransformer'
			exception.method = 'train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
			

class TfidfVectorizer( Transformer ):
	"""

		Purpose:
		---------

		Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency.
		This is a common term-weighting scheme in information retrieval, that has also found good
		use in document classification. The goal of using tf-idf instead of the raw frequencies of
		occurrence of a token in a given document is to scale down the impact of tokens that occur
		very frequently in a given corpus and that are hence empirically less informative than
		feature_names that occur in a small fraction of the training corpus.

		The formula that is used to compute the tf-idf for a term t of a document d in a
		document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf
		is computed as idf(t) = log [ n / df(t) ] + 1 (if smooth_idf=False), where n is the total
		number of text in the document set and df(t) is the document frequency of t;
		the document frequency is the number of text in the document set that contain
		the term t. The effect of adding “1” to the idf in the equation above is that
		terms with zero idf, i.e., terms that occur in all text in a training set,
		will not be entirely ignored. (Note that the idf formula above differs from the
		standard textbook notation that defines the idf as idf(t) = log [ n / (df(t) + 1) ]).

	"""
	model: sk.TfidfVectorizer
	input: Optional[ str ]
	encoding: Optional[ str ]
	decode_error: Optional[ str ]
	strip_accents: Optional[ Any ]
	lowercase: Optional[ bool ]
	preprocessor: Optional[ Any ]
	tokenizer: Optional[ Any ]
	analyzer: Optional[ str | Any ]
	stop_words: Optional[ Any ]
	token_pattern: Optional[ str ]
	ngram_range: Optional[ Tuple[ int, int ] ]
	max_df: Optional[ float | int ]
	min_df: Optional[ float | int ]
	max_features: Optional[ int | None ]
	vocabulary: Optional[ Dict[ str, int ] | List[ str ] | None ]
	binary: Optional[ bool ]
	dtype: Optional[ Any ]
	norm: Optional[ str | None ]
	use_idf: Optional[ bool ]
	smooth_idf: Optional[ bool ]
	sublinear_tf: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, input: str = 'content', encoding: str = 'utf-8',
			decode_error: str = 'strict', strip_accents: Any = None,
			lowercase: bool = True, preprocessor: Any = None, tokenizer: Any = None,
			analyzer: str | Any = 'word', stop_words: Any = None,
			token_pattern: str = r'(?u)\b\w\w+\b',
			ngram_range: Tuple[ int, int ] = (1, 1),
			max_df: float | int = 1.0, min_df: float | int = 1,
			max_features: int | None = None,
			vocabulary: Dict[ str, int ] | List[ str ] | None = None,
			binary: bool = False, dtype: Any = np.float64, norm: str | None = 'l2',
			use_idf: bool = True, smooth_idf: bool = True,
			sublinear_tf: bool = False ) -> None:
		"""

			Purpose:
			---------
			Initialize the TfidfVectorizer wrapper.

			Parameters:
			-----------
			input ( str ): Input mode for documents.
			encoding ( str ): Encoding used to decode byte sequences.
			decode_error ( str ): Instruction on what to do if a byte sequence cannot be
				decoded.
			strip_accents ( Any ): Accent-stripping strategy.
			lowercase ( bool ): Indicates whether text should be lowercased before
				tokenization.
			preprocessor ( Any ): Optional preprocessing callable.
			tokenizer ( Any ): Optional tokenizer callable.
			analyzer ( str | Any ): Feature extraction mode or callable.
			stop_words ( Any ): Stop words passed to the vectorizer.
			token_pattern ( str ): Regular expression denoting what constitutes a token.
			ngram_range ( Tuple[ int, int ] ): Lower and upper boundary of the n-grams.
			max_df ( float | int ): Ignore terms with document frequency strictly higher
				than this threshold.
			min_df ( float | int ): Ignore terms with document frequency strictly lower
				than this threshold.
			max_features ( int | None ): Maximum size of the vocabulary.
			vocabulary ( Dict[ str, int ] | List[ str ] | None ): Optional fixed
				vocabulary.
			binary ( bool ): Indicates whether all nonzero term counts should be set to 1.
			dtype ( Any ): Type of the matrix returned by fit_transform or transform.
			norm ( str | None ): Norm used to normalize term vectors.
			use_idf ( bool ): Indicates whether inverse-document-frequency reweighting
				should be enabled.
			smooth_idf ( bool ): Indicates whether document frequencies should be
				smoothed.
			sublinear_tf ( bool ): Indicates whether sublinear term-frequency scaling
				should be applied.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.input = input
		self.encoding = encoding
		self.decode_error = decode_error
		self.strip_accents = strip_accents
		self.lowercase = lowercase
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.analyzer = analyzer
		self.stop_words = stop_words
		self.token_pattern = token_pattern
		self.ngram_range = ngram_range
		self.max_df = max_df
		self.min_df = min_df
		self.max_features = max_features
		self.vocabulary = vocabulary
		self.binary = binary
		self.dtype = dtype
		self.norm = norm
		self.use_idf = use_idf
		self.smooth_idf = smooth_idf
		self.sublinear_tf = sublinear_tf
		self.model = sk.TfidfVectorizer(
			input=self.input,
			encoding=self.encoding,
			decode_error=self.decode_error,
			strip_accents=self.strip_accents,
			lowercase=self.lowercase,
			preprocessor=self.preprocessor,
			tokenizer=self.tokenizer,
			analyzer=self.analyzer,
			stop_words=self.stop_words,
			token_pattern=self.token_pattern,
			ngram_range=self.ngram_range,
			max_df=self.max_df,
			min_df=self.min_df,
			max_features=self.max_features,
			vocabulary=self.vocabulary,
			binary=self.binary,
			dtype=self.dtype,
			norm=self.norm,
			use_idf=self.use_idf,
			smooth_idf=self.smooth_idf,
			sublinear_tf=self.sublinear_tf )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Return a list of class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Member names exposed by the wrapper.

		"""
		return [ 'model',
		         'input',
		         'encoding',
		         'decode_error',
		         'strip_accents',
		         'lowercase',
		         'preprocessor',
		         'tokenizer',
		         'analyzer',
		         'stop_words',
		         'token_pattern',
		         'ngram_range',
		         'max_df',
		         'min_df',
		         'max_features',
		         'vocabulary',
		         'binary',
		         'dtype',
		         'norm',
		         'use_idf',
		         'smooth_idf',
		         'sublinear_tf',
		         'transformed_data',
		         'idf_vector',
		         'train',
		         'transform',
		         'train_transform',
		         'inverse_transform' ]
	
	@property
	def idf_vector( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the inverse document frequency vector learned during fitting.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray: Inverse document frequency vector.

		"""
		if getattr( self.model, 'idf_', None ) is None:
			raise AttributeError( 'TfidfVectorizer must be initialized.' )
		else:
			return self.model.idf_
	
	def train( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ] = None ) -> TfidfVectorizer | None:
		"""

			Purpose:
			---------
			Learn the vocabulary and inverse document frequency values from the training
			documents.

			Parameters:
			-----------
			text ( List[ str ] | np.ndarray ): Iterable of raw documents.
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored by the estimator.

			Returns:
			--------
			TfidfVectorizer | None: Fitted wrapper instance.

		"""
		try:
			throw_if( 'text', text )
			self.model.fit( text, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'train( self, text: List[ str ] | np.ndarray, y: Optional[ np.ndarray ] = None ) -> TfidfVectorizer'
			raise exception
	
	def transform( self, text: List[ str ] | np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform raw documents to a dense TF-IDF document-term matrix.

			Parameters:
			-----------
			text ( List[ str ] | np.ndarray ): Iterable of raw documents.

			Returns:
			--------
			np.ndarray: Dense TF-IDF-weighted document-term matrix.

		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, text: List[ str ] | np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the vectorizer and transform the raw documents to a dense TF-IDF
			document-term matrix.

			Parameters:
			-----------
			text ( List[ str ] | np.ndarray ): Iterable of raw documents.
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored by the estimator.

			Returns:
			--------
			np.ndarray: Dense TF-IDF-weighted document-term matrix.

		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.fit_transform( text, y ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'train_transform( self, text: List[ str ] | np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> List[ np.ndarray ] | None:
		"""

			Purpose:
			---------
			Return terms per document corresponding to the nonzero entries in the
			document-term matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Document-term matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			List[ np.ndarray ] | None: List of arrays of terms for each document.

		"""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> List[ np.ndarray ] | None'
			raise exception
			

class CountVectorizer( Transformer ):
	"""

		Purpose:
		---------
		Convert a collection of text to a matrix of token counts. This implementation
		produces a sparse representation of the counts using scipy.sparse.csr_matrix. If you do not
		provide an a-priori dictionary and you do not use an analyzer that does some kind of
		feature selection then the number of feature_names will be equal to the vocabulary
		size found by analyzing the stores.

	"""
	model: sk.CountVectorizer
	input: Optional[ str ]
	encoding: Optional[ str ]
	decode_error: Optional[ str ]
	strip_accents: Optional[ Any ]
	lowercase: Optional[ bool ]
	preprocessor: Optional[ Any ]
	tokenizer: Optional[ Any ]
	analyzer: Optional[ str | Any ]
	stop_words: Optional[ Any ]
	token_pattern: Optional[ str ]
	ngram_range: Optional[ Tuple[ int, int ] ]
	max_df: Optional[ float | int ]
	min_df: Optional[ float | int ]
	max_features: Optional[ int | None ]
	vocabulary: Optional[ Dict[ str, int ] | List[ str ] | None ]
	binary: Optional[ bool ]
	dtype: Optional[ Any ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, input: str = 'content', encoding: str = 'utf-8',
			decode_error: str = 'strict', strip_accents: Any = None,
			lowercase: bool = True, preprocessor: Any = None, tokenizer: Any = None,
			analyzer: str | Any = 'word', stop_words: Any = None,
			token_pattern: str = r'(?u)\b\w\w+\b',
			ngram_range: Tuple[ int, int ] = (1, 1),
			max_df: float | int = 1.0, min_df: float | int = 1,
			max_features: int | None = None,
			vocabulary: Dict[ str, int ] | List[ str ] | None = None,
			binary: bool = False, dtype: Any = np.int64 ) -> None:
		"""

			Purpose:
			---------
			Initialize the CountVectorizer wrapper.

			Parameters:
			-----------
			input ( str ): Input mode for documents.
			encoding ( str ): Encoding used to decode byte sequences.
			decode_error ( str ): Instruction on what to do if a byte sequence cannot be
				decoded.
			strip_accents ( Any ): Accent-stripping strategy.
			lowercase ( bool ): Indicates whether text should be lowercased before
				tokenization.
			preprocessor ( Any ): Optional preprocessing callable.
			tokenizer ( Any ): Optional tokenizer callable.
			analyzer ( str | Any ): Feature extraction mode or callable.
			stop_words ( Any ): Stop words passed to the vectorizer.
			token_pattern ( str ): Regular expression denoting what constitutes a token.
			ngram_range ( Tuple[ int, int ] ): Lower and upper boundary of the n-grams.
			max_df ( float | int ): Ignore terms with document frequency strictly higher
				than this threshold.
			min_df ( float | int ): Ignore terms with document frequency strictly lower
				than this threshold.
			max_features ( int | None ): Maximum size of the vocabulary.
			vocabulary ( Dict[ str, int ] | List[ str ] | None ): Optional fixed
				vocabulary.
			binary ( bool ): Indicates whether all nonzero term counts should be set to 1.
			dtype ( Any ): Type of the matrix returned by fit_transform or transform.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.input = input
		self.encoding = encoding
		self.decode_error = decode_error
		self.strip_accents = strip_accents
		self.lowercase = lowercase
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.analyzer = analyzer
		self.stop_words = stop_words
		self.token_pattern = token_pattern
		self.ngram_range = ngram_range
		self.max_df = max_df
		self.min_df = min_df
		self.max_features = max_features
		self.vocabulary = vocabulary
		self.binary = binary
		self.dtype = dtype
		self.model = sk.CountVectorizer(
			input=self.input,
			encoding=self.encoding,
			decode_error=self.decode_error,
			strip_accents=self.strip_accents,
			lowercase=self.lowercase,
			preprocessor=self.preprocessor,
			tokenizer=self.tokenizer,
			analyzer=self.analyzer,
			stop_words=self.stop_words,
			token_pattern=self.token_pattern,
			ngram_range=self.ngram_range,
			max_df=self.max_df,
			min_df=self.min_df,
			max_features=self.max_features,
			vocabulary=self.vocabulary,
			binary=self.binary,
			dtype=self.dtype )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Return a list of class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Member names exposed by the wrapper.

		"""
		return [ 'model',
		         'input',
		         'encoding',
		         'decode_error',
		         'strip_accents',
		         'lowercase',
		         'preprocessor',
		         'tokenizer',
		         'analyzer',
		         'stop_words',
		         'token_pattern',
		         'ngram_range',
		         'max_df',
		         'min_df',
		         'max_features',
		         'vocabulary',
		         'binary',
		         'dtype',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform',
		         'inverse_transform' ]
	
	def train( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ] = None ) -> CountVectorizer | None:
		"""

			Purpose:
			---------
			Learn the vocabulary dictionary from the training documents.

			Parameters:
			-----------
			text ( List[ str ] | np.ndarray ): Iterable of raw documents.
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored by the estimator.

			Returns:
			--------
			CountVectorizer | None: Fitted wrapper instance.

		"""
		try:
			throw_if( 'text', text )
			self.model.fit( text, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'train( self, text: List[ str ] | np.ndarray, y: Optional[ np.ndarray ] = None ) -> CountVectorizer'
			raise exception
	
	def transform( self, text: List[ str ] | np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform raw documents to a dense count matrix.

			Parameters:
			-----------
			text ( List[ str ] | np.ndarray ): Iterable of raw documents.

			Returns:
			--------
			np.ndarray: Dense count-weighted document-term matrix.

		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, text: List[ str ] | np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the vectorizer and transform the raw documents to a dense count matrix.

			Parameters:
			-----------
			text ( List[ str ] | np.ndarray ): Iterable of raw documents.
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored by the estimator.

			Returns:
			--------
			np.ndarray: Dense count-weighted document-term matrix.

		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.fit_transform( text, y ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'train_transform( self, text: List[ str ] | np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> List[ np.ndarray ] | None:
		"""

			Purpose:
			---------
			Return terms per document corresponding to the nonzero entries in the
			document-term matrix.

			Parameters:
			-----------
			X ( np.ndarray ): Document-term matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			List[ np.ndarray ] | None: List of arrays of terms for each document.

		"""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> List[ np.ndarray ] | None'
			raise exception
			

class HashVectorizer( Transformer ):
	"""

		Purpose:
		---------
		Convert a collection of text to a matrix of token occurrences. It turns a
		collection of text into a scipy.sparse matrix holding token occurrence counts
		(or binary occurrence information), possibly normalized as token frequencies
		if norm=’l1’ or projected on the Euclidean unit sphere if norm=’l2’.

		This text vectorizer implementation uses the hashing trick to find the token
		string name to feature integer index mapping. This strategy has several advantages it is
		very low memory scalable to large datasets as there is no need to store a vocabulary
		dictionary in memory.

		It is fast to pickle and un-pickle as it holds no state besides the constructor parameters.
		it can be used in a streaming (partial fit) or parallel pipeline as there is no state
		computed during fit.

		There are also a couple of cons (vs using a CountVectorizer with an in-memory vocabulary):
		there is no way to compute the inverse transform (from feature indices to string feature
		names) which can be a problem when trying to introspect which features are most
		important to a model.

		There can be collisions: distinct tokens can be mapped to the same feature index.
		However, in practice this is rarely an issue if n_features is large enough (e.g. 2 ** 18
		for text classification problems).

	"""
	vectorizer: sk.HashingVectorizer
	input: Optional[ str ]
	encoding: Optional[ str ]
	decode_error: Optional[ str ]
	strip_accents: Optional[ Any ]
	lowercase: Optional[ bool ]
	preprocessor: Optional[ Any ]
	tokenizer: Optional[ Any ]
	analyzer: Optional[ str | Any ]
	stop_words: Optional[ Any ]
	token_pattern: Optional[ str ]
	ngram_range: Optional[ Tuple[ int, int ] ]
	binary: Optional[ bool ]
	norm: Optional[ str | None ]
	alternate_sign: Optional[ bool ]
	n_features: Optional[ int ]
	dtype: Optional[ Any ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, num: int = 1048576, input: str = 'content',
			encoding: str = 'utf-8', decode_error: str = 'strict',
			strip_accents: Any = None, lowercase: bool = True,
			preprocessor: Any = None, tokenizer: Any = None,
			analyzer: str | Any = 'word', stop_words: Any = None,
			token_pattern: str = r'(?u)\b\w\w+\b',
			ngram_range: Tuple[ int, int ] = (1, 1), binary: bool = False,
			norm: str | None = 'l2', alternate_sign: bool = True,
			dtype: Any = np.float64 ) -> None:
		"""

			Purpose:
			---------
			Initialize the HashVectorizer wrapper.

			Parameters:
			-----------
			num ( int ): Number of hashed features.
			input ( str ): Input mode for documents.
			encoding ( str ): Encoding used to decode byte sequences.
			decode_error ( str ): Instruction on what to do if a byte sequence cannot be
				decoded.
			strip_accents ( Any ): Accent-stripping strategy.
			lowercase ( bool ): Indicates whether text should be lowercased before
				tokenization.
			preprocessor ( Any ): Optional preprocessing callable.
			tokenizer ( Any ): Optional tokenizer callable.
			analyzer ( str | Any ): Feature extraction mode or callable.
			stop_words ( Any ): Stop words passed to the vectorizer.
			token_pattern ( str ): Regular expression denoting what constitutes a token.
			ngram_range ( Tuple[ int, int ] ): Lower and upper boundary of the n-grams.
			binary ( bool ): Indicates whether all nonzero term counts should be set to 1.
			norm ( str | None ): Norm used to normalize term vectors.
			alternate_sign ( bool ): Indicates whether alternating signs should be used
				to approximately conserve inner products.
			dtype ( Any ): Type of the matrix returned by transform.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.input = input
		self.encoding = encoding
		self.decode_error = decode_error
		self.strip_accents = strip_accents
		self.lowercase = lowercase
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.analyzer = analyzer
		self.stop_words = stop_words
		self.token_pattern = token_pattern
		self.ngram_range = ngram_range
		self.binary = binary
		self.norm = norm
		self.alternate_sign = alternate_sign
		self.n_features = num
		self.dtype = dtype
		self.vectorizer = sk.HashingVectorizer(
			n_features=self.n_features,
			input=self.input,
			encoding=self.encoding,
			decode_error=self.decode_error,
			strip_accents=self.strip_accents,
			lowercase=self.lowercase,
			preprocessor=self.preprocessor,
			tokenizer=self.tokenizer,
			analyzer=self.analyzer,
			stop_words=self.stop_words,
			token_pattern=self.token_pattern,
			ngram_range=self.ngram_range,
			binary=self.binary,
			norm=self.norm,
			alternate_sign=self.alternate_sign,
			dtype=self.dtype )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Return a list of class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Member names exposed by the wrapper.

		"""
		return [ 'vectorizer',
		         'input',
		         'encoding',
		         'decode_error',
		         'strip_accents',
		         'lowercase',
		         'preprocessor',
		         'tokenizer',
		         'analyzer',
		         'stop_words',
		         'token_pattern',
		         'ngram_range',
		         'binary',
		         'norm',
		         'alternate_sign',
		         'n_features',
		         'dtype',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform',
		         'inverse_transform' ]
	
	def train( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ] = None ) -> HashVectorizer | None:
		"""

			Purpose:
			---------
			Validate the input documents and return the wrapper. HashingVectorizer is
			stateless and does not learn a vocabulary during fitting.

			Parameters:
			-----------
			text ( List[ str ] | np.ndarray ): Iterable of raw documents.
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored.

			Returns:
			--------
			HashVectorizer | None: Wrapper instance.

		"""
		try:
			throw_if( 'text', text )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HashVectorizer'
			exception.method = 'train( self, text: List[ str ] | np.ndarray, y: Optional[ np.ndarray ] = None ) -> HashVectorizer'
			raise exception
	
	def transform( self, text: List[ str ] | np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform raw documents to a dense hashed document-term matrix.

			Parameters:
			-----------
			text ( List[ str ] | np.ndarray ): Iterable of raw documents.

			Returns:
			--------
			np.ndarray: Dense hashed feature matrix.

		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.vectorizer.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HashVectorizer'
			exception.method = 'transform( self, text: List[ str ] | np.ndarray ) -> np.ndarray'
			raise exception
	
	def train_transform( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Validate the input and transform raw documents to a dense hashed document-term
			matrix.

			Parameters:
			-----------
			text ( List[ str ] | np.ndarray ): Iterable of raw documents.
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored.

			Returns:
			--------
			np.ndarray: Dense hashed feature matrix.

		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.vectorizer.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HashVectorizer'
			exception.method = 'train_transform( self, text: List[ str ] | np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Indicate that inverse transformation is not supported for hashed features.

			Parameters:
			-----------
			X ( np.ndarray ): Hashed document-term matrix.

			Returns:
			--------
			None

		"""
		raise NotImplementedError( 'HashingVectorizer does not support inverse_transform.' )


class DictVectorizer( Transformer ):
	"""

		Purpose:
		---------
		Transform lists of feature-value mappings to vectors. String-valued features are
		expanded using one-of-K style encoding, while numeric values are passed through
		as numeric feature values.

	"""
	model: fe.DictVectorizer
	dtype: Optional[ Any ]
	separator: Optional[ str ]
	sparse: Optional[ bool ]
	sort: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, dtype: Any = np.float64, separator: str = '=',
			sparse: bool = True, sort: bool = True ) -> None:
		"""

			Purpose:
			---------
			Initialize the DictVectorizer wrapper.

			Parameters:
			-----------
			dtype ( Any ): Type used for the output matrix values.
			separator ( str ): Separator used when constructing one-hot encoded feature
				names from string-valued mappings.
			sparse ( bool ): Indicates whether output should be sparse internally.
			sort ( bool ): Indicates whether feature names should be sorted when fitting.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.dtype = dtype
		self.separator = separator
		self.sparse = sparse
		self.sort = sort
		self.model = fe.DictVectorizer( dtype=self.dtype, separator=self.separator,
			sparse=self.sparse, sort=self.sort )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Return a list of class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Member names exposed by the wrapper.

		"""
		return [ 'model',
		         'dtype',
		         'separator',
		         'sparse',
		         'sort',
		         'transformed_data',
		         'feature_names',
		         'vocabulary',
		         'train',
		         'transform',
		         'train_transform',
		         'inverse_transform' ]
	
	@property
	def feature_names( self ) -> np.ndarray:
		"""

			Purpose:
			---------
			Return the learned feature names.

			Parameters:
			-----------
			None

			Returns:
			--------
			np.ndarray: Learned feature names.

		"""
		try:
			return self.model.get_feature_names_out( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'feature_names( self ) -> np.ndarray'
			raise exception
	
	@property
	def vocabulary( self ) -> Dict[ str, int ]:
		"""

			Purpose:
			---------
			Return the learned vocabulary mapping.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[ str, int ]: Mapping from feature name to column index.

		"""
		try:
			return self.model.vocabulary_
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'vocabulary( self ) -> Dict[ str, int ]'
			raise exception
	
	def train( self, X: List[ Dict[ str, Any ] ],
			y: Optional[ np.ndarray ] = None ) -> DictVectorizer | None:
		"""

			Purpose:
			---------
			Fit the vectorizer on a list of feature-value mappings.

			Parameters:
			-----------
			X ( List[ Dict[ str, Any ] ] ): List of mapping objects describing samples.
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored by the estimator.

			Returns:
			--------
			DictVectorizer | None: Fitted wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'train( self, X: List[ Dict[ str, Any ] ], y: Optional[ np.ndarray ] = None ) -> DictVectorizer'
			raise exception
	
	def transform( self, X: List[ Dict[ str, Any ] ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform a list of feature-value mappings to a dense feature matrix.

			Parameters:
			-----------
			X ( List[ Dict[ str, Any ] ] ): List of mapping objects describing samples.

			Returns:
			--------
			np.ndarray: Dense feature matrix.

		"""
		try:
			throw_if( 'X', X )
			result = self.model.transform( X )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'transform( self, X: List[ Dict[ str, Any ] ] ) -> np.ndarray'
			raise exception
	
	def train_transform( self, X: List[ Dict[ str, Any ] ],
			y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the vectorizer and transform the mappings to a dense feature matrix.

			Parameters:
			-----------
			X ( List[ Dict[ str, Any ] ] ): List of mapping objects describing samples.
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored by the estimator.

			Returns:
			--------
			np.ndarray: Dense feature matrix.

		"""
		try:
			throw_if( 'X', X )
			result = self.model.fit_transform( X, y )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'train_transform( self, X: List[ Dict[ str, Any ] ], y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> List[ Dict[ str, Any ] ] | None:
		"""

			Purpose:
			---------
			Transform a feature matrix back to a list of feature-value mappings.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix of shape ( n_samples, n_features ).

			Returns:
			--------
			List[ Dict[ str, Any ] ] | None: Reconstructed feature-value mappings.

		"""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> List[ Dict[ str, Any ] ] | None'
			raise exception


class FeatureHasher( Transformer ):
	"""

		Purpose:
		---------
		Convert symbolic feature names to a matrix using feature hashing. This estimator
		is stateless and is intended for large-scale or memory-constrained workflows.

	"""
	model: fe.FeatureHasher
	n_features: Optional[ int ]
	input_type: Optional[ str ]
	dtype: Optional[ Any ]
	alternate_sign: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, n_features: int = 1048576, input_type: str = 'dict',
			dtype: Any = np.float64, alternate_sign: bool = True ) -> None:
		"""

			Purpose:
			---------
			Initialize the FeatureHasher wrapper.

			Parameters:
			-----------
			n_features ( int ): Number of output features.
			input_type ( str ): Type of the input data. Supported values are controlled by
				sklearn, such as 'dict', 'pair', and 'string'.
			dtype ( Any ): Type used for the output matrix values.
			alternate_sign ( bool ): Indicates whether alternating signs should be used to
				approximately conserve inner products.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.n_features = n_features
		self.input_type = input_type
		self.dtype = dtype
		self.alternate_sign = alternate_sign
		self.model = fe.FeatureHasher( n_features=self.n_features,
			input_type=self.input_type, dtype=self.dtype,
			alternate_sign=self.alternate_sign )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""

			Purpose:
			---------
			Return a list of class members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ]: Member names exposed by the wrapper.

		"""
		return [ 'model',
		         'n_features',
		         'input_type',
		         'dtype',
		         'alternate_sign',
		         'transformed_data',
		         'train',
		         'transform',
		         'train_transform',
		         'inverse_transform' ]
	
	def train( self, X: List[ Dict[ str, Any ] ] | List[ Tuple[ str, Any ] ] | List[ str ],
			y: Optional[ np.ndarray ] = None ) -> FeatureHasher | None:
		"""

			Purpose:
			---------
			Validate the input and return the wrapper. FeatureHasher is stateless and does
			not learn parameters during fitting.

			Parameters:
			-----------
			X ( List[ Dict[ str, Any ] ] | List[ Tuple[ str, Any ] ] | List[ str ] ):
				Input samples compatible with the configured input_type.
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored.

			Returns:
			--------
			FeatureHasher | None: Wrapper instance.

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'FeatureHasher'
			exception.method = 'train( self, X, y: Optional[ np.ndarray ] = None ) -> FeatureHasher'
			raise exception
	
	def transform( self, X: List[ Dict[ str, Any ] ] | List[ Tuple[ str, Any ] ] | List[ str ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform symbolic features to a dense hashed feature matrix.

			Parameters:
			-----------
			X ( List[ Dict[ str, Any ] ] | List[ Tuple[ str, Any ] ] | List[ str ] ):
				Input samples compatible with the configured input_type.

			Returns:
			--------
			np.ndarray: Dense hashed feature matrix.

		"""
		try:
			throw_if( 'X', X )
			result = self.model.transform( X )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'FeatureHasher'
			exception.method = 'transform( self, X ) -> np.ndarray'
			raise exception
	
	def train_transform( self, X: List[ Dict[ str, Any ] ] | List[ Tuple[ str, Any ] ] | List[ str ],
			y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Validate the input and transform symbolic features to a dense hashed matrix.
	
			Parameters:
			-----------
			X ( List[ Dict[ str, Any ] ] | List[ Tuple[ str, Any ] ] | List[ str ] ] ):
				Input samples compatible with the configured input_type.
			y ( Optional[ np.ndarray ] ): Optional target array. Ignored.
	
			Returns:
			--------
			np.ndarray: Dense hashed feature matrix.
			
		"""
		try:
			throw_if( 'X', X )
			result = self.model.fit_transform( X, y )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'FeatureHasher'
			exception.method = 'train_transform( self, X, y: Optional[ np.ndarray ] = None ) -> np.ndarray'
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Indicate that inverse transformation is not supported for hashed features.

			Parameters:
			-----------
			X ( np.ndarray ): Hashed feature matrix.

			Returns:
			--------
			None

		"""
		raise NotImplementedError( 'FeatureHasher does not support inverse_transform.' )
	
