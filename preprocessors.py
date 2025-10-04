'''
	******************************************************************************************
	  Assembly:                mathy
	  Filename:                transformers.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022
	
	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="preprocessors.py" company="Terry D. Eppler">
	
	     mathy Preprocessing
	
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
		preprocessors.py
	</summary>
	******************************************************************************************
'''
from __future__ import annotations

from typing import Optional, List

import numpy as np
import sklearn.feature_extraction.text as sk
import sklearn.impute as ski
import sklearn.preprocessing as skp

from boogr import Error, ErrorDialog

def throw_if( name: str, value: object ):
	if not value:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Preprocessor( ):
	"""

		Purpose:
		---------
		Base interface for all preprocessors. Provides standard `fit`, `transform`, and
		`fit_transform` methods.

	"""
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ):
		self.transformed_data = None
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> object | None:
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
	
	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray:
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

class LabelBinarizer( Preprocessor ):
	"""

		Purpose:
		_______
		Binarize target_names in a one-vs-all fashion. Several regression and binary classification
		algorithms are available in scikit-learn. A simple way to extend these algorithms to the
		multi-class classification case is to use the so-called one-vs-all scheme.

		At learning time, this simply consists in learning one regressor or binary classifier
		per class. In doing so, one needs to convert multi-class target_names to binary target_names
		(belong or does not belong to the class). LabelBinarizer does this process with
		the transform method.

		At prediction time, one assigns the class for which the corresponding model gave
		the greatest confidence. LabelBinarizer does this process with
		the inverse_transform method.


	"""
	label_binarizer: skp.LabelBinarizer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

		Purpose:
		_______
		Initializes the LabelBinarizerWrapper.

		"""
		super( ).__init__( )
		self.label_binarizer = skp.LabelBinarizer( )
		self.transformed_data = None
		
	@property
	def classes( self ) -> List[ str ]:
		if self.label_binarizer.classes_ is None:
			raise AttributeError( 'LabelBinarizer has not been initialized.' )
		else:
			return self.label_binarizer.classes_
	
	def fit( self, y: np.ndarray ) -> LabelBinarizer | None:
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
			self.label_binarizer.fit( y )
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
			self.transformed_data = self.label_binarizer.transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'fit( self, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, y: np.ndarray ) -> np.ndarray:
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
			self.transformed_data = self.label_binarizer.fit_transform( y )
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
			return self.label_binarizer.inverse_transform( Y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'inverse_transform( self, y: np.ndarray, thresh: float=None ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class TfidfTransformer( Preprocessor ):
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
	tfidf_transformer: sk.TfidfTransformer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize TfidfTransformer.
		"""
		super( ).__init__( )
		self.tfidf_transformer = sk.TfidfTransformer( )
		self.transformed_data = None
	
	@property
	def idf_vector( self ) -> np.ndarray:
		if self.tfidf_transformer.idf_ is None:
			raise AttributeError( 'TfidfTransformer must be initialized' )
		else:
			return self.tfidf_transformer.idf_
		
	def fit( self, X: np.ndarray ) -> TfidfTransformer | None:
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
			self.tfidf_transformer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'fit( self, X: np.ndarray, y: np.ndarray ) -> object'
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
			self.transformed_data = self.tfidf_transformer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'transform( self, X: np.ndarray, y: np.ndarray=None  ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, X: np.ndarray ) -> np.ndarray:
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
			self.transformed_data = self.tfidf_transformer.fit_transform( X ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'fit_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class TfidfVectorizer( Preprocessor ):
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
	tfidf_vectorizer: sk.TfidfVectorizer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize TfidfVectorizer.

		"""
		super( ).__init__( )
		self.tfidf_vectorizer = sk.TfidfVectorizer( )
		self.transformed_data = None
	
	def fit( self, text: list[ str ], y: np.ndarray=None ) -> TfidfVectorizer | None:
		"""

			Purpose:
			---------
			Fit the vectorizer to the text.

			Parameters:
			-----------
			text: list[str]
			y: np.ndarray - IGNORED

		"""
		try:
			throw_if( 'text', text )
			self.tfidf_vectorizer.fit( text )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, text: list[ str ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform text to TF-IDF vectors.

			:param y:
			:type y:
			:param text: List of strings.
			:type text: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.tfidf_vectorizer.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, text: list[ str ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit and transform the text.

			:param y:
			:type y:
			:param text: List of text text.
			:type text: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.tfidf_vectorizer.fit_transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, text: list[ str ] ) -> List[ List[ str ] ] | None:
		"""

			Purpose:
			---------
			Transform text to TF-IDF vectors.

			:param text: List of text text.
			:type text: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		try:
			throw_if( 'text', text )
			return self.tfidf_vectorizer.inverse_transform( text )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> List[ List[ str ] ]'
			error = ErrorDialog( exception )
			error.show( )

class CountVectorizer( Preprocessor ):
	"""

		Purpose:
		---------
		Convert a collection of text to a matrix of token counts. This implementation
		produces a sparse representation of the counts using scipy.sparse.csr_matrix. If you do not
		provide an a-priori dictionary and you do not use an analyzer that does some kind of
		feature selection then the number of feature_names will be equal to the vocabulary
		size found by analyzing the stores.

	"""
	count_vectorizer: sk.CountVectorizer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the CountVectorizerWrapper with default parameters.
		"""
		super( ).__init__( )
		self.count_vectorizer = sk.CountVectorizer( )
		self.transformed_data = None
	
	def fit( self, text: List[ str ], y: np.ndarray=None ) -> CountVectorizer | None:
		"""

			Purpose:
			---------
			Convert a collection of tokens to a matrix of token counts.

			:param text:
			:type List[ str ]:

		"""
		try:
			throw_if( 'text', text )
			self.count_vectorizer.fit( text )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, text: List[ str ] ) -> np.ndarray:
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
			self.transformed_data = self.count_vectorizer.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, text: List[ str ] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, text: List[ str ], y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the vectorizer and transform the text.

			:param y:
			:type y:
			:param text: List of input text text.
			:type text: List[str]
			:return: Matrix of token counts.
			:rtype: np.ndarray

		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.count_vectorizer.fit_transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class HashingVectorizer( Preprocessor ):
	"""

		Purpose:
		---------
		Convert a collection of text text to a matrix of token occurrences. It turns a
		collection of text into a scipy.sparse matrix holding token occurrence counts
		(or binary occurrence information), possibly normalized as token frequencies
		if norm=’l1’ or projected on the euclidean unit sphere if norm=’l2’.

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
		However in practice this is rarely an issue if n_features is large enough (e.g. 2 ** 18
		for text classification problems).

	"""
	hash_vectorizer: sk.HashingVectorizer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, num: int=1048576 ) -> None:
		"""

			Purpose:
			---------
			Initialize the HashingVectorizer with the desired number of feature_names.

		"""
		super( ).__init__( )
		self.hash_vectorizer = sk.HashingVectorizer( n_features=num )
		self.transformed_data = None
	
	def fit( self, X: np.ndarray, y: np.ndarray=None ) -> CountVectorizer | None:
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
			self.hash_vectorizer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HashingVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
			
	def transform( self, text: List[ str ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform text into hashed token vectors.

			:param text: List of input text.
			:type text: List[str]
			
			:return: Matrix of hashed feature_names.
			:rtype: np.ndarray
			
		"""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.hash_vectorizer.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HashingVectorizer'
			exception.method = 'transform( self, text: List[ str ], y: np.ndarray=None ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class StandardScaler( Preprocessor ):
	"""

		Purpose:
		--------
		Standardize feature_names by removing the mean and scaling to unit variance. The standard score
		of a sample x is calculated as: z = (x - u) / s where u is the mean of the training
		samples or zero if with_mean=False, and s is the standard deviation of the training
		samples or one if with_std=False.

	"""
	standard_scaler: skp.StandardScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.standard_scaler = skp.StandardScaler( )
		self.transformed_data = None
	
	def fit( self, X: np.ndarray, y: np.ndarray=None ) -> StandardScaler | None:
		"""


			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ). IGNORED

			Returns:
			--------
			StandardScaler | None

		"""
		try:
			throw_if( 'X', X )
			self.standard_scaler.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None  ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms the df using the fitted StandardScaler.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ). IGNORED

			Returns:
			-----------
			np.ndarray: Scaled df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.standard_scaler.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, X: np.ndarray  ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms into a standardized np.ndarray

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )

			Returns:
			-----------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			return self.standard_scaler.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class MinMaxScaler( Preprocessor ):
	"""

		Purpose:
		---------
		This estimator scales and translates each feature individually such that it is in the
		given range on the training set, e.g. between zero and one. This transformation is often
		used as an alternative to zero mean, unit variance scaling.

		MinMaxScaler doesn’t reduce the effect of outliers, but it linearly scales them down
		into a fixed range, where the largest occurring stores point corresponds to the maximum
		value and the smallest one corresponds to the minimum value

	"""
	minmax_scaler: skp.MinMaxScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.minmax_scaler = skp.MinMaxScaler( )
		self.transformed_data = None
	
	def fit( self, X: np.ndarray, y: np.ndarray=None ) -> MinMaxScaler | None:
		"""

			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ). IGNORED

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.minmax_scaler.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = ('fit( self, X: np.ndarray, y: np.ndarray=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms the df using the fitted MinMaxScaler.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ). IGNORED

			Returns:
			-----------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.minmax_scaler.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, X: np.ndarray, y: np.ndarray = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fits & Transforms the df using the fitted MinMaxScaler.

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ). IGNORED

			Returns:
			-----------
			np.ndarray


		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.minmax_scaler.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
			
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms back to original stores

			Parameters:
			-----------
			X ( np.ndarray ): Feature matrix/samples of shape ( n_samples, n_features )

			Returns:
			-----------
			np.ndarray

		"""
		try:
			throw_if( 'X', X )
			return self.minmax_scaler.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class RobustScaler( Preprocessor ):
	"""

		Purpose:
		--------
		This Scaler removes the median and scales the stores according to the quantile range
		(defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile
		(25th quantile) and the 3rd quartile (75th quantile).

		Centering and scaling happen independently on each feature by computing the relevant
		statistics on the samples in the training set. Median and interquartile range are
		then stored to be used on later stores using the transform method.

		Standardization of a dataset is a common preprocessing for many machine learning estimators.
		Typically this is done by removing the mean and scaling to unit variance.
		However, outliers can often influence the sample mean / variance in a negative way.
		In such cases, using the median and the interquartile range often give better results.

	"""
	robust_scaler: skp.RobustScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.robust_scaler = skp.RobustScaler( )
		self.transformed_data = None
	
	def fit( self, X: np.ndarray, y: np.ndarray=None ) -> RobustScaler | None:
		"""


			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.robust_scaler.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = ('fit( self, X: np.ndarray, y: np.ndarray=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms the df using the fitted RobustScaler.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Scaled df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.robust_scaler.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms into robust stores.

			:param X: List of text text.
			:type X: list[str]
			:return: Standardized stores.
			:rtype: np.ndarray
		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.robust_scaler.inverse_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class NormalScaler( Preprocessor ):
	"""

		Purpose:
		---------
		Normalize samples individually to unit norm. Each sample (i.e. each row of the stores matrix)
		with at least one non zero component is rescaled independently of other samples so that
		its norm (l1, l2 or inf) equals one.

		This transformer is able to work both with dense numpy arrays and scipy.sparse matrix
		(use CSR format if you want to avoid the burden of a copy / conversion). Scaling inputs to
		unit norms is a common operation for text classification or clustering for instance.
		For instance the dot product of two l2-normalized TF-IDF vectors is the cosine similarity
		of the vectors and is the base similarity metric for the Vector Space Model.

	"""
	norm: Optional[ str ]
	normal_scaler: skp.Normalizer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, reg: str = 'l2' ) -> None:
		super( ).__init__( )
		self.norm = reg
		self.normal_scaler = skp.Normalizer( norm=self.norm)
		self.transformed_data = None
	
	def fit( self, X: np.ndarray, y: np.ndarray=None ) -> NormalScaler | None:
		"""


			Purpose:
			---------
			Fits the normalizer (no-op for Normalizer).

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.normal_scaler.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = ('fit( self, X: np.ndarray, y: np.ndarray=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""


			Purpose:
			---------
			Applies normalization to each sample.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Normalized df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.normal_scaler.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""


			Purpose:
			---------
			Applies normalization to each sample.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Normalized df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.normal_scaler.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class OneHotEncoder( Preprocessor ):
	"""

		Purpose:
		---------
		Encode categorical feature_names as a one-hot numeric array. The input to this transformer
		should be an array-like of integers or strings, denoting the values taken on by categorical
		(discrete) feature_names. The feature_names are encoded using a one-hot
		(aka ‘one-of-K’ or ‘dummy’) encoding scheme.
		This creates a binary column for each category and returns a sparse
		matrix or dense array (depending on the sparse_output parameter)

		By default, the encoder derives the categories based on the unique values in each feature.
		Alternatively, you can also specify the categories manually. This encoding is needed for
		feeding categorical stores to many scikit-learn estimators, notably linear models and SVMs
		with the standard kernels. Note: a one-hot encoding of y target_names should use a
		LabelBinarizer instead.

	"""
	unknown: Optional[ str ]
	sparse: Optional[ bool ]
	hot_encoder: skp.OneHotEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, sparse: bool=False, unknown: str='ignore' ) -> None:
		super( ).__init__( )
		self.unknown = unknown
		self.sparse = sparse
		self.hot_encoder = skp.OneHotEncoder( sparse_output=self.sparse, handle_unknown=self.unknown )
	
	@property
	def categories( self ):
		if self.hot_encoder.categories_ is None:
			raise AttributeError( 'Hot Encoder data is untrained' )
		else:
			return self.hot_encoder.categories_
	
	def fit( self, X: np.ndarray, y: np.ndarray=None ) -> OneHotEncoder | None:
		"""


			Purpose:
			---------
			Fits the hot_encoder to the categorical df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.hot_encoder.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = ('fit( self, X: np.ndarray, y: np.ndarray=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""


			Purpose:
			---------
			Transforms the categorical matrix into one-hot encoded form

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: One-hot encoded matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.hot_encoder.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit the encoder and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.hot_encoder.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class OrdinalEncoder( Preprocessor ):
	"""


			Purpose:
			---------
			This estimator transforms each categorical feature to one new feature of integers
			(0 to n_categories - 1):

			Such integer representation can, however, not be used directly with all scikit-learn
			estimators, as these expect continuous input, and would interpret the categories as
			being ordered, which is often not desired (i.e. the set of browsers was
			ordered arbitrarily).

			By default, OrdinalEncoder will also passthrough missing values that are indicated
			by np.nan. OrdinalEncoder provides a parameter encoded_missing_value to encode
			the missing values without the need to create a pipeline and using SimpleImputer.

	"""
	ordinal_encoder: skp.OrdinalEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.ordinal_encoder = skp.OrdinalEncoder( )
		self.transformed_data = None
	
	@property
	def categories( self ):
		if self.hot_encoder.categories_ is None:
			raise AttributeError( 'Hot Encoder data is untrained' )
		else:
			return self.hot_encoder.categories_
	
	def fit( self, X: np.ndarray, y: np.ndarray=None ) -> OrdinalEncoder | None:
		"""

			Purpose:
			________
			Fits the ordial encoder to the categorical df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.ordinal_encoder.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'fit( self, X: np.ndarray, y: np.ndarray=None ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms the text df into ordinal-encoded format.


			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Ordinal-encoded matrix.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.ordinal_encoder.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit the encoder and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.ordinal_encoder.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Map ordinal-encoded matrix back to original categories.

			Parameters:
			-----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( '"X" cannot be None' )
			else:
				throw_if( 'X', X )
				return self.ordinal_encoder.inverse_transform( X ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class LabelEncoder( Preprocessor ):
	"""

		Purpose:
		--------
		Encode target target_names with value between 0 and n_classes-1.
		This transformer should be used to encode target values, i.e. y, and not the input X.

	"""
	label_encoder: skp.LabelEncoder
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			--------
			Initialize LabelEncoder.

		"""
		super( ).__init__( )
		self.label_encoder = skp.LabelEncoder( )
	
	@property
	def classes( self ):
		if self.label_encoder.classes_ is None:
			raise AttributeError( 'The label encoder data is untrained.' )
		else:
			return self.label_encoder.classes_
		
	def fit( self, y: np.ndarray ) -> LabelEncoder | None:
		"""

			Purpose:
			--------
			Fit the label encoder to the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'y', y )
			self.label_encoder.fit( y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'fit( self, y: np.ndarray ) -> LabelEncoder'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Transform target_names to encoded form.

			Parameters:
			-----------
			X ( List[ str ] ): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		
		try:
			throw_if( 'y', y )
			self.transformed_data = self.label_encoder.transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'transform( self, y: np.ndarray  ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit and transform the label stores.

			Parameters:
			-----------
			X ( List[ str ] ): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		
		try:
			throw_if( 'y', y )
			self.transformed_data = self.label_encoder.fit_transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'fit_transform( self, y: np.ndarray  ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, y: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Map integer labels back to original classes.

			Parameters:
			-----------
			y: np.ndarray

		"""
		try:
			throw_if( 'y', y )
			return self.label_encoder.inverse_transform( y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class PolynomialFeatures( Preprocessor ):
	"""

		Purpose:
		--------
		Generate a new feature matrix consisting of all polynomial combinations of the features
		with degree less than or equal to the specified degree. For example, if an input sample is
		two dimensional and of the form [a, b], the degree-2 polynomial
		features are [1, a, b, a^2, ab, b^2].




	"""
	degree: Optional[ int ]
	interaction_only: Optional[ bool ]
	polynomial_features: skp.PolynomialFeatures
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, degree: int=2, interaction: bool=True ) -> None:
		"""

			Purpose:
			--------
			Initialize PolynomialFeatures.

			:param degree: Degree of polynomial terms.
			:type degree: int
		"""
		super( ).__init__( )
		self.degree = degree
		self.interaction_only = interaction
		self.polynomial_features = skp.PolynomialFeatures( degree=self.degree,
			interaction_only=self.interaction_only )
	
	@property
	def powers( self ):
		if self.polynomial_features.powers_ is None:
			raise AttributeError( 'The polynomial data is untrained.' )
		else:
			return self.polynomial_features.powers_
		
	def fit( self, X: np.ndarray, y: np.ndarray=None ) -> PolynomialFeatures | None:
		"""

			Purpose:
			--------
			Fit polynomial transformer to stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.polynomial_features.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'fit( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Transform stores into polynomial feature_names.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.polynomial_features.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit and transform stores using polynomial expansion.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.polynomial_features.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class MeanImputer( Preprocessor ):
	"""

		Purpose:
		-----------
		Fills missing target_names using the average.

	"""
	strategy: Optional[ str ]
	mean_imputer: ski.SimpleImputer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, strategy: str='mean' ) -> None:
		super( ).__init__( )
		self.strategy = strategy
		self.mean_imputer = ski.SimpleImputer( strategy=self.strategy )
		self.transformed_data = None
	
	def fit( self, X: np.ndarray, y: np.ndarray=None ) -> MeanImputer | None:
		"""


			Purpose:
			---------
			Fits the simple_imputer to the df.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			--------
			Pipeline

		"""
		try:
			throw_if( 'X', X )
			self.mean_imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'fit( self, X: np.ndarray, y: np.ndarray=None ) -> MeanImputer'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""


			Purpose:
			---------
			Transforms the text df by filling in missing target_names.

			Parameters:
			-----------
			X (np.ndarray): Input df with missing target_names.

			Returns:
			-----------
			np.ndarray: Imputed df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.mean_imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit the iterative imputer and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.mean_imputer.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform text to TF-IDF vectors.

			:param X: np.ndarray
		"""
		try:
			throw_if( 'X', X )
			return self.mean_imputer.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class NearestNeighborImputer( Preprocessor ):
	"""

		Purpose:
		---------
		The NearestNeighborImputer class provides imputation for filling in missing values using
		the k-Nearest Neighbors approach. By default, a euclidean distance metric that supports
		missing values, nan_euclidean_distances, is used to find the nearest neighbors.
		Each missing feature is imputed using values from n_neighbors nearest neighbors that have
		a value for the feature. The feature of the neighbors are averaged uniformly or weighted
		by distance to each neighbor.

		If a sample has more than one feature missing, then the neighbors for that sample can be
		different depending on the particular feature being imputed. When the number of available
		neighbors is less than n_neighbors and there are no defined distances to the training set,
		the training set average for that feature is used during imputation. If there is at least
		one neighbor with a defined distance, the weighted or unweighted average of the
		remaining neighbors will be used during imputation. If a feature is always missing in
		training, it is removed during transform.

	"""
	n_neighbors: Optional[ int ]
	knn_imputer: ski.KNNImputer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, neighbors: int=5 ) -> None:
		super( ).__init__( )
		self.n_neighbors = neighbors
		self.knn_imputer = ski.KNNImputer( n_neighbors=self.n_neighbors )
		self.transformed_data = None
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> NearestNeighborImputer | None:
		"""

			Purpose:
			________
			Fits the simple_imputer to the df.

			Parameters:
			___________
			X (np.ndarray): Feature matrix of shape ( n_samples, n_features ).
			y (np.ndarray): True class target vector of shape ( n_samples, ).

			Returns:
			--------
			self

		"""
		try:
			throw_if( 'X', X )
			self.knn_imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborImputer'
			exception.method = 'fit( self, X: np.ndarray ) -> NearestNeighborImputer'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			_________

			Transforms the text df by imputing missing target_names.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

			Returns:
			-----------
			np.ndarray: Imputed df.

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.knn_imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborImputer'
			exception.method = 'transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the iterative imputer and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.knn_imputer.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighborImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class IterativeImputer( Preprocessor ):
	"""
		Purpose:
		--------
		The IterativeImputer class, which models each feature with missing values as a function of
		other features, and uses that estimate for imputation. It does so in an iterated
		round-robin fashion: at each step, a feature column is designated as output y and the
		other feature columns are treated as inputs X. A regressor is fit on (X, y) for known y.

		Then, the regressor is used to predict the missing values of y. This is done for each
		feature in an iterative fashion, and then is repeated for max_iter imputation rounds.
		The results of the final imputation round are returned.

	"""
	iterative_imputer: ski.IterativeImputer
	max_iter: Optional[ int ]
	random_state: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, max: int=10, rando: int=0 ) -> None:
		"""

			Purpose:
			--------
			Initialize the IterativeImputer.

			:param max: Maximum number of imputation iterations.
			:type max: int
			:param rando: Random seed.
			:type rando: int

		"""
		super( ).__init__( )
		self.max_iter = max
		self.random_state = rando
		self.iterative_imputer = ski.IterativeImputer( max_iter=self.max_iter, random_state=self.random_state )
	
	def fit( self, X: np.ndarray, y: np.ndarray=None ) -> object | None:
		"""

			Purpose:
			--------
			Fit the iterative imputer to the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.iterative_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform stores by iteratively imputing missing values.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.iterative_imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = 'transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit the iterative imputer and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.iterative_imputer.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class SimpleImputer( Preprocessor ):
	"""

		Wrapper for sklearn's SimpleImputer.

	"""
	simple_imputer: ski.SimpleImputer
	transformed_data: Optional[ np.ndarray ]
	strategy: Optional[ str ]
	fill_value: Optional[ float ]
	
	def __init__( self, strategy: str='mean', fill_value: float=0.0 ) -> None:
		"""

			Purpose:
			---------
			Initialize the SimpleImputer.

			:param strategy: The imputation strategy ('mean', 'median', 'most_frequent', or 'constant').
			:type strategy: str
			:param fill_value: Value to use when strategy is 'constant'.
			:type fill_value: float

		"""
		super( ).__init__( )
		self.strategy = strategy
		self.fill_value = fill_value
		self.simple_imputer = ski.SimpleImputer( strategy=self.strategy, fill_value=self.fill_value )
		self.transformed_data = None
	
	def fit( self, X: np.ndarray, y: np.ndarray=None ) -> object | None:
		"""

			Purpose:
			--------
			Fit the imputer to the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.simple_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit( X: np.ndarray ) -> self'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Transform stores by imputing missing values.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			return self.simple_imputer.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit( X: np.ndarray ) -> self'
			error = ErrorDialog( exception )
			error.show( )
	
	def fit_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray:
		"""

			Purpose:
			--------
			Fit the imputer and transform the stores.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix/input samples of shape ( n_samples, n_features )
			y (np.ndarray): Target vector of shape ( n_samples, ).

		"""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.simple_imputer.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
