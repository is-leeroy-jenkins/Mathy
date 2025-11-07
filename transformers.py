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
import sklearn.feature_extraction.text as sk
import sklearn.preprocessing as pp
import sklearn.compose as sc
from boogr import Error, ErrorDialog

def throw_if( name: str, value: object ):
	if not value:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Transformer( ):
	"""

		Purpose:
		---------
		Base interface for all preprocessors. Provides standard `fit`, `transform`, and
		`fit_transform` methods.

	"""
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ):
		self.transformed_data = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> object | None:
		"""

			Purpose:
			---------
			Train/flex hook; return self in concrete subclasses.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )
			y ( Optional[ np.ndarray ] ): Optional target array  of shape ( n_samples, ).

			Returns:
			-----------
			object | None

		"""
		raise NotImplementedError
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform X using a fitted preprocessor; return transformed X.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit to X (and y if used) then transform X in one step.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError
	
	def inverse_transform( self, text: list[ str ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Invert the transformation when supported; raise NotImplementedError otherwise.

			:param text: List of text text.
			:type text: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		return NotImplementedError

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
	
	def __init__( self, threshold=0.0, copy=True ) -> None:
		"""

		Purpose:
		_______
		Initializes the Binarizer.

		"""
		super( ).__init__( )
		self.threshold = threshold
		self.copy = copy
		self.model = pp.Binarizer( threshold=self.threshold, copy=self.copy )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'threshold',
		  'copy',
		  'model',
		  'transformer',
		  'transformed_data',
		  'classes',
		  'types',
		  'train',
		  'transform',
		  'train_transform',
		  'inverse_transform', ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> LabelBinarizer | None:
		"""

			Purpose:
			_______
			Fit the label binarizer on target values y.

			Parameters:
			-----------
			y ( np.ndarray ): target array  of shape ( n_features ).

			Returns:
			-----------
			LabelBinarizer | None

		"""
		try:
			throw_if( 'y', y )
			self.model.fit( y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'fit( self, y: Optional[ np.ndarray ] ) -> LabelBinarizer'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Purpose:
			--------
			This method is just there to implement the usual API and hence work in pipelines.
			
			Parameters:
			-----------
			X ( np.ndarray ): Labels to binarize.

			Returns:
			X ( np.ndarray )
			
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X, copy=None )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Binarizer'
			exception.method = 'transform( self, X: np.ndarray, copy=None ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray:
		"""

			Purpose:
			_______
			Fits transformer to X and optional y and returns a transformed version of X.

			Parameters:
			-----------
			X ( np.ndarray ): Training vector of shape [ n_samples, n_features ]
			y ( np.ndarray ): Target vector of shape [ n_samples ]

			Returns:
			--------
			X_new (np.ndarrya): Transformed array.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Binarizer'
			exception.method = 'train_transform( self, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
			
class LabelBinarizer( Transformer ):
	"""

		Purpose:
		_______
		At learning time, this simply consists in learning one regressor or binary classifier
		per class. In doing so, one needs to convert multi-class target_names to binary target_names
		(belong or does not belong to the class). LabelBinarizer does this process with
		the transform method.

		At prediction time, one assigns the class for which the corresponding model gave
		the greatest confidence. LabelBinarizer does this process with
		the inverse_transform method.


	"""
	model: pp.LabelBinarizer
	pos_label: Optional[ int ]
	neg_label: Optional[ int ]
	sparse: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, pos_label: int=1, neg_label: int=0, sparse: bool=False ) -> None:
		"""

		Purpose:
		_______
		Initializes the LabelBinarizerWrapper.

		"""
		super( ).__init__( )
		self.pos_label = pos_label
		self.neg_label = neg_label
		self.sparse = sparse
		self.model = pp.LabelBinarizer( pos_label=self.neg_label,
			neg_label=self.pos_label, sparse=self.sparse )
		self.transformed_data = None
	
	def __dir__( self ):
		'''
			
			Returns
			-------
			A list of strings comprised of class members.
			
		'''
		[ 'pos_label',
		  'neg_label',
		  'sparse',
		  'model',
		  'transformer',
		  'transformed_data',
		  'classes',
		  'types',
		  'train',
		  'transform',
		  'train_transform',
		  'inverse_transform', ]
		
	@property
	def classes( self ) -> List[ str ]:
		if self.model.classes_ is None:
			raise AttributeError( 'LabelBinarizer has not been initialized.' )
		else:
			return self.model.classes_
	
	@property
	def types( self ) -> str :
		if self.model.y_type_ is None:
			raise AttributeError( 'LabelBinarizer has not been initialized.' )
		else:
			return self.model.y_type_
	
	def train( self, y: np.ndarray ) -> LabelBinarizer | None:
		"""

			Purpose:
			_______
			Fit the label binarizer on target values y.

			Parameters:
			-----------
			y ( np.ndarray ): target array  of shape ( n_features ).

			Returns:
			-----------
			LabelBinarizer | None

		"""
		try:
			throw_if( 'y', y )
			self.model.fit( y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'fit( self, y: Optional[ np.ndarray ] ) -> LabelBinarizer'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Transform target y to binary matrix.

			Args:
			-----------
			y ( np.ndarray ): Target vector of shape ( n_features ).

			Returns:
				np.ndarray: Binary-encoded label matrix.
				:param y:
				:type y:
		"""
		try:
			throw_if( 'y', y )
			self.transformed_data = self.model.transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'fit( self, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Fit on y then transform y to binary matrix.

			Parameters:
			-----------
			y ( np.ndarray ): Target vector of shape ( n_features ).

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
			exception.method = 'fit( self, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, Y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Converts binary matrix back to original target_names.

			Parameters:
			----------
			Y (np.ndarray): Binary-encoded label matrix.

			Returns:
			np.ndarray: Original target_names.

		"""
		try:
			throw_if( 'Y', Y )
			return self.model.inverse_transform( Y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'inverse_transform( self, y: np.ndarray, thresh: float=None ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class MultiLabelBinarizer( Transformer ):
	"""

		Purpose:
		_______
		Transform between iterable of iterables and a multilabel format. Although a list of sets
		or tuples is a very intuitive format for multilabel data, it is unwieldy to process.
		This transformer converts between this intuitive format and the supported multilabel
		format: a (samples x classes) binary matrix indicating the presence of a class label.


	"""
	model: pp.MultiLabelBinarizer
	classes: Optional[ np.ndarray ]
	sparse_output: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, classes: np.ndarray, sparse: bool=False ) -> None:
		"""

			Purpose:
			_______
			Initializes the LabelBinarizerWrapper.

		"""
		super( ).__init__( )
		self.classes = classes
		self.sparse_output = sparse
		self.model = pp.MultiLabelBinarizer( classes=self.classes, sparse_output=self.sparse_output )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'classes',
		  'sparse_output',
		  'sparse',
		  'model',
		  'transformer',
		  'transformed_data',
		  'classes',
		  'types',
		  'train',
		  'transform',
		  'train_transform',
		  'inverse_transform', ]
	
	@property
	def classes( self ) -> List[ str ]:
		if self.model.classes_ is None:
			raise AttributeError( 'MultiLabelBinarizer has not been initialized.' )
		else:
			return self.model.classes_
	
	def train( self, y: np.ndarray ) -> MultiLabelBinarizer | None:
		"""

			Purpose:
			_______
			Fit the multi-label binarizer on target values y.

			Parameters:
			-----------
			y ( np.ndarray ): target array  of shape ( n_features ).

			Returns:
			-----------
			MultiLabelBinarizer | None

		"""
		try:
			throw_if( 'y', y )
			self.model.fit( y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'fit( self, y: Optional[ np.ndarray ] ) -> LabelBinarizer'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Transform target y to binary matrix.

			Args:
			-----------
			y ( np.ndarray ): Target vector of shape ( n_features ).

			Returns:
				np.ndarray: Binary-encoded label matrix.
				:param y:
				:type y:
		"""
		try:
			throw_if( 'y', y )
			self.transformed_data = self.model.transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'fit( self, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Fit on y then transform y to binary matrix.

			Parameters:
			-----------
			y ( np.ndarray ): Target vector of shape ( n_features ).

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
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'fit( self, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, Y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_______
			Converts binary matrix back to original target_names.

			Parameters:
			----------
			Y (np.ndarray): Binary-encoded label matrix.

			Returns:
			np.ndarray: Original target_names.

		"""
		try:
			throw_if( 'Y', Y )
			return self.model.inverse_transform( Y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'inverse_transform( self, y: np.ndarray, thresh: float=None ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class TfidfTransformer( Transformer ):
	"""

		Purpose:
		---------
		Transform a count matrix to a normalized tf or tf-idf representation. Tf means
		term-frequency while tf-idf means term-frequency times inverse document-frequency.
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
	model: sk.TfidfTransformer
	norm: Optional[ str ]
	use_idf: Optional[ bool ]
	smooth_idf: Optional[ bool ]
	sublinear_tf: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, norm: str='l2', use_idf: bool=True,
			smooth_idf: bool=True, sublinear_tf: bool=False ) -> None:
		"""

			Purpose:
			---------
			Initialize TfidfTransformer.
		"""
		super( ).__init__( )
		self.norm = norm
		self.use_idf = use_idf
		self.smooth_idf = smooth_idf
		self.sublinear_tf = sublinear_tf
		self.model = sk.TfidfTransformer( norm=self.norm, use_idf=self.use_idf,
			smooth_idf=self.smooth_idf, sublinear_tf=self.sublinear_tf )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'norm',
		  'use_idf',
		  'smooth_idf',
		  'sublinear_tf',
		  'model',
		  'transformed_data',
		  'idf_vector',
		  'features'
		  'train',
		  'transform',
		  'train_transform' ]
	
	@property
	def idf_vector( self ) -> np.ndarray:
		if self.model.idf_ is None:
			raise AttributeError( 'TfidfTransformer must be initialized' )
		else:
			return self.model.idf_
	
	@property
	def features( self ) -> np.ndarray:
		if self.model.n_features_in_ is None:
			raise AttributeError( 'TfidfTransformer must be initialized' )
		else:
			return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> TfidfTransformer | None:
		"""

			Purpose:
			---------
			Fit the transformer to a count matrix.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )

			Returns:
			---------
			TfidfTransformer | None

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'train( self, X: np.ndarray ) -> TfidfTransformer'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform a count matrix to TF-IDF.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )

			Returns:
			--------
			np.ndarray: Dense matrix of tokens of shape ( n_samples, n_features )

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X, copy=True )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit and transform the count matrix.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

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
	
	def __init__( self, transformers, remainder='drop', sparse_threshold=0.3, n_jobs=None,
			transformer_weights=None, verbose=False ) -> None:
		"""

			Purpose:
			---------
			Initialize ColumnTransformer.
			
		"""
		super( ).__init__( )
		self.transformers = transformers
		self.remainder = remainder
		self.sparse_threshold = sparse_threshold
		self.n_jobs = n_jobs
		self.transformer_weights = transformer_weights
		self.verbose = verbose
		self.model = sc.ColumnTransformer( transformers=self.transformers, remainder=self.remainder,
			sparse_threshold=self.sparse_threshold, n_jobs=self.n_jobs, verbose=self.verbose )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'model',
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
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> ColumnTransformer | None:
		"""

			Purpose:
			---------
			Fit all transformers using X.

			Parameters:
			-----------
			X (np.ndarray): array-like or DataFrame of shape [n_samples, n_features]
			Input data, of which specified subsets are used to fit the transformers.

			Returns:
			---------
			ColumnTransformer | None

		"""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ColumnTransformer'
			exception.method = 'train( self, X: np.ndarray, y: Optional[np.ndarray]) -> ColumnTransformer'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit all transformers using X.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )

			Returns:
			--------
			X_t : array-like or sparse matrix, shape (n_samples, sum_n_components )
			hstack of results of transformers. sum_n_components is the sum of n_components
			(output dimension) over transformers. If any result is a sparse matrix,
			everything will be converted to sparse matrices.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ColumnTransformer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]   ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit all transformers, transform the data and concatenate results.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): INGORED
			
			Returns:
			--------
			X_t : array-like or sparse matrix, shape (n_samples, sum_n_components )
			hstack of results of transformers. sum_n_components is the sum of n_components
			(output dimension) over transformers. If any result is a sparse matrix,
			everything will be converted to sparse matrices.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ColumnTransformer'
			exception.method = 'train_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

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
	strip_accents: Optional[ bool ]
	lowercase: Optional[ bool ]
	preprocessor: Optional[ object ]
	tokenizer: Optional[ object ]
	max_features: Optional[ int ]
	vocabulary: Optional[ set ]
	stopwords: Optional[ Any ]
	binary: Optional[ bool ]
	norm: Optional[ str ]
	use_idf: Optional[ bool ]
	smooth_idf: Optional[ bool ]
	sublinear_tf: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, input: str='content', encoding: str='utf-8', decode_error: str='strict',
			strip_accents: Any=None, max_features: Any=None, lowercase: bool = True, stopwords: Any=None,
			preprocessor: Any=None, tokenizer: Any=None, norm: str='l2', use_idf: bool=True,
			smooth_idf: bool=True, sublinear_tf: bool=False ) -> None:
		"""

			Purpose:
			---------
			Initialize TfidfVectorizer.

		"""
		super( ).__init__( )
		self.input = input
		self.encoding = encoding
		self.decode_error = decode_error
		self.strip_accents = strip_accents
		self.max_features = max_features
		self.lowercase = lowercase
		self.stopwords = stopwords
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.norm = norm
		self.use_idf = use_idf
		self.smooth_idf = smooth_idf
		self.sublinear_tf = sublinear_tf
		self.model = sk.TfidfVectorizer( input=self.input, encoding=self.encoding,
			decode_error=self.decode_error, strip_accents=self.strip_accents,
			max_features=self.max_features, lowercase=self.lowercase, preprocessor=self.preprocessor,
			tokenizer=self.tokenizer, norm=self.norm, use_idf=self.use_idf,
			smooth_idf=self.smooth_idf, sublinear_tf=self.sublinear_tf )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'model',
		  'input',
		  'encoding',
		  'decode_error',
		  'strip_accents',
		  'max_features',
		  'lowercase',
		  'preprocessor',
		  'tokenizer',
		  'norm',
		  'use_idf',
		  'smooth_idf',
		  'sublinear_tf',
		  'transformed_data',
		  'classes',
		  'train',
		  'transform',
		  'train_transform',
		  'inverse_transform', ]
	
	@property
	def idf_vector( self ) -> np.ndarray:
		'''

			Returns
			-------
			Inverse document frequency vector, only defined if use_idf=True.

		'''
		if self.model.idf_ is None:
			raise AttributeError( 'TfidfTransformer must be initialized' )
		else:
			return self.model.idf_
	
	def train( self, text: str, y: Optional[ np.ndarry ] ) -> TfidfVectorizer | None:
		"""

			Purpose:
			---------
			Learn vocabulary and idf from training set

			Parameters:
			-----------
			text (str): An iterable which generates either str, unicode or file objects.
			y: np.ndarray - IGNORED

			Returns:
			--------
			self - Fitted vectorizer.

		"""
		try:
			throw_if( 'text', text )
			self.model.fit( text )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'train( self, text: str, y: Optional[ np.ndarray ] ) -> TfidfVectorizer'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, text: str ) -> np.ndarray:
		"""

			Purpose:
			-------
			Transform text into count vectors.

			Parameters:
			-----------
			text (List[ str ]): Feature matrix

			Returns:
			-----------
			np.ndarray | None

		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfdifVectorizer'
			exception.method = 'transform( self, tokens: List[ str ] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, text: str, y: Optional[ np.ndarry ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit and transform the text.

			Parameters:
			----------
			text (str): An iterable which generates either str, unicode or file objects.
			y: np.ndarray - IGNORED


			Returns:
			--------
			X:  sparse matrix of (n_samples, n_features)
			Tf-idf-weighted document-term matrix.

		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.fit_transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'train_transform( self, tokens: list[ str ] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, X: np.ndarray ) -> List[ np.ndarray ] | None:
		"""

			Purpose:
			---------
			Transform text to TF-IDF vectors.

			Parameters:
			----------
			X {array-like, sparse matrix} of shape (n_samples, n_features)
			Document-term matrix.

			Returns:
			--------
			X_original list of arrays of shape (n_samples,)

		"""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> List[ np.ndarray ]'
			error = ErrorDialog( exception )
			error.show( )

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
	preprocessor: Optional[ object ]
	tokenizer: Optional[ object ]
	max_features: Optional[ int ]
	vocabulary: Optional[ set ]
	stopwords: Optional[ set ]
	binary: Optional[ bool ]
	norm: Optional[ str ]
	max_df: Optional[ float ]
	min_df: Optional[ float ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, input: str = 'content', encoding: str = 'utf-8', decode_error: str = 'strict',
			strip_accents: Any=None, max_features: Any=None, lowercase: bool=True, stopwords: Any=None,
			preprocessor: Any=None, tokenizer: Any=None, norm: str = 'l2', max_df: float = 1.0,
			min_df: float = 1 ) -> None:
		"""

			Purpose:
			---------
			Initialize the CountVectorizerWrapper with default parameters.

		"""
		super( ).__init__( )
		self.input = input
		self.encoding = encoding
		self.decode_error = decode_error
		self.strip_accents = strip_accents
		self.max_features = max_features
		self.lowercase = lowercase
		self.stopwords = stopwords
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.norm = norm
		self.max_df = max_df
		self.min_df = min_df
		self.model = sk.CountVectorizer( input=self.input, encoding=self.encoding,
			decode_error=self.decode_error, strip_accents=self.strip_accents, stop_words=self.stopwords,
			max_features=self.max_features, lowercase=self.lowercase, preprocessor=self.preprocessor,
			tokenizer=self.tokenizer, norm=self.norm, binary=self.binary,
			max_df=self.max_df, min_df=self.min_df, )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'model',
		  'input',
		  'encoding',
		  'decode_error',
		  'strip_accents',
		  'max_features',
		  'lowercase',
		  'preprocessor',
		  'tokenizer',
		  'norm',
		  'max_df',
		  'min_df',
		  'binary',
		  'transformed_data',
		  'classes',
		  'train',
		  'transform',
		  'train_transform',
		  'inverse_transform', ]
	
	def train( self, text: str, y: Optional[ np.ndarry ] ) -> CountVectorizer | None:
		"""

			Purpose:
			---------
			Convert a collection of tokens to a matrix of token counts.

			:param tokens:
			:type List[ str ]:

		"""
		try:
			throw_if( 'text', text )
			self.model.fit( text )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'train( self, text: str, y: Optional[ np.ndarray ] ) -> CountVectorizer'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, text: str ) -> np.ndarray:
		"""

			Purpose:
			-------
			Transform text into count vectors.

			Parameters:
			-----------
			text (List[ str ]): Feature matrix

			Returns:
			-----------
			np.ndarray | None

		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, tokens: List[ str ] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, text: str, y: np.ndarray = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit and transform the text.

			Parameters:
			----------
			text (str): An iterable which generates either str, unicode or file objects.
			y: np.ndarray - IGNORED


			Returns:
			--------
			X:  sparse matrix of (n_samples, n_features)
			Count-weighted document-term matrix.

		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.fit_transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'train_transform( self, tokens: List[ str ] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

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
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, num: int = 1048576 ) -> None:
		"""

			Purpose:
			---------
			Initialize the HashingVectorizer with the desired number of feature_names.

		"""
		super( ).__init__( )
		self.vectorizer = sk.HashingVectorizer( n_features=num )
		self.transformed_data = None
	
	def __dir__( self ):
		'''

			Returns
			-------
			A list of strings comprised of class members.

		'''
		[ 'vectorizer',
		  'transformed_data',
		  'classes',
		  'train',
		  'transform',
		  'train_transform',
		  'inverse_transform', ]
	
	def train( self, X: np.ndarray, y: np.ndarray = None ) -> CountVectorizer | None:
		"""

			Purpose:
			---------
			Convert a collection of text text to a matrix of token counts.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.vectorizer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HashingVectorizer'
			exception.method = 'train( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, tokens: List[ str ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform text into hashed token vectors.

			:param tokens: List of input text.
			:type tokens: List[str]

			:return: Matrix of hashed feature_names.
			:rtype: np.ndarray

		"""
		try:
			throw_if( 'tokens', tokens )
			self.transformed_data = self.vectorizer.transform( tokens ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HashingVectorizer'
			exception.method = 'transform( self, tokens: List[ str ] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
