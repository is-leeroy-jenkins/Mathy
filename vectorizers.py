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
from typing import Optional, List
import numpy as np
import sklearn.feature_extraction.text as sk
from boogr import Error, ErrorDialog

def throw_if( name: str, value: object ):
	if not value:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Vectorizer( ):
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

			:param text: List of text.
			:type text: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		return NotImplementedError

class TfidfVectorizer( Vectorizer ):
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
	vectorizer: sk.TfidfVectorizer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize TfidfVectorizer.

		"""
		super( ).__init__( )
		self.vectorizer = sk.TfidfVectorizer( )
		self.transformed_data = None
	
	def train( self, tokens: list[ str ], y: np.ndarray = None ) -> TfidfVectorizer | None:
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
			throw_if( 'tokens', tokens )
			self.vectorizer.fit( tokens )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, tokens: list[ str ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transform text to TF-IDF vectors.

			:param y:
			:type y:
			:param tokens: List of strings.
			:type tokens: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		try:
			throw_if( 'tokens', tokens )
			self.transformed_data = self.vectorizer.transform( tokens ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, tokens: list[ str ] ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit and transform the text.

			:param y:
			:type y:
			:param tokens: List of text text.
			:type tokens: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		try:
			throw_if( 'tokens', tokens )
			self.transformed_data = self.vectorizer.fit_transform( tokens ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def inverse_transform( self, tokens: list[ str ] ) -> List[ List[ str ] ] | None:
		"""

			Purpose:
			---------
			Transform text to TF-IDF vectors.

			:param tokens: List of text text.
			:type tokens: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		try:
			throw_if( 'tokens', tokens )
			return self.vectorizer.inverse_transform( tokens )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> List[ List[ str ] ]'
			error = ErrorDialog( exception )
			error.show( )

class CountVectorizer( Vectorizer ):
	"""

		Purpose:
		---------
		Convert a collection of text to a matrix of token counts. This implementation
		produces a sparse representation of the counts using scipy.sparse.csr_matrix. If you do not
		provide an a-priori dictionary and you do not use an analyzer that does some kind of
		feature selection then the number of feature_names will be equal to the vocabulary
		size found by analyzing the stores.

	"""
	vectorizer: sk.CountVectorizer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the CountVectorizerWrapper with default parameters.
		"""
		super( ).__init__( )
		self.vectorizer = sk.CountVectorizer( )
		self.transformed_data = None
	
	def train( self, tokens: List[ str ], y: np.ndarray = None ) -> CountVectorizer | None:
		"""

			Purpose:
			---------
			Convert a collection of tokens to a matrix of token counts.

			:param tokens:
			:type List[ str ]:

		"""
		try:
			throw_if( 'tokens', tokens )
			self.vectorizer.fit( tokens )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def transform( self, tokens: List[ str ] ) -> np.ndarray:
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
			throw_if( 'tokens', tokens )
			self.transformed_data = self.vectorizer.transform( tokens ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, text: List[ str ] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	def train_transform( self, tokens: List[ str ], y: np.ndarray = None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fit the vectorizer and transform the text.

			:param y:
			:type y:
			:param tokens: List of input text text.
			:type tokens: List[str]
			:return: Matrix of token counts.
			:rtype: np.ndarray

		"""
		try:
			throw_if( 'tokens', tokens )
			self.transformed_data = self.vectorizer.fit_transform( tokens ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class HashVectorizer( Vectorizer ):
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
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
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
			exception.method = 'transform( self, text: List[ str ], y: np.ndarray=None ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
