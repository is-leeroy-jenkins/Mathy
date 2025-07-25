'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                transformers.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="transformers.py" company="Terry D. Eppler">

     Mathy Encoders

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
	encoders.py
</summary>
******************************************************************************************
'''
from __future__ import annotations

from booger import Error, ErrorDialog
import numpy as np
from typing import Optional, List
import sklearn.feature_extraction.text as sk
from pydantic import BaseModel


class Metric( BaseModel ):
	"""

		Purpose:
		---------
		Base interface for all preprocessors. Provides standard `fit`, `transform`, and
	    `fit_transform` methods.

	"""

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )
		self.pipeline = None
		self.transformed_data = np.ndarray | None
		self.transformed_values = [ ]

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
		"""

			Purpose:
			---------
			Fits the preprocessor to the text df.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (Optional[np.ndarray]): Optional target array.

		"""
		raise NotImplementedError


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms the text df using the fitted preprocessor.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Fits the preprocessor and then transforms the text df.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (Optional[np.ndarray]): Optional target array.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError


	def inverse_transform( self, documents: list[ str ] ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform documents to TF-IDF vectors.

			:param documents: List of text documents.
			:type documents: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		return NotImplementedError


class TfidfTransformer( Metric ):
	"""

		Purpose:
		---------

		Transform a count matrix to a normalized tf or tf-idf representation.

	"""

	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize TfidfTransformer.
		"""
		super( ).__init__( )
		self.transformer = sk.TfidfTransformer( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
		"""

			Purpose:
			---------
			Fit the transformer to a count matrix.

			:param y:
			:type y:
			:param X: Input count matrix.
			:type X: np.ndarray
		"""
		self.transformer.fit ( X)


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform a count matrix to TF-IDF.

			:param X: Input count matrix.
			:type X: np.ndarray
			:return: TF-IDF matrix.
			:rtype: np.ndarray
		"""
		return self.transformer.transform( X ).toarray ()


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Fit and transform the count matrix.

			:param y:
			:type y:
			:param X: Input count matrix.
			:type X: np.ndarray
			:return: TF-IDF matrix.
			:rtype: np.ndarray
		"""
		return self.transformer.fit_transform( X ).toarray ()


	def inverse_transform( self, documents: list[ str ] ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform documents to TF-IDF vectors.

			:param documents: List of text documents.
			:type documents: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		try:
			if documents is None:
				raise Exception( '"documents" cannot be None' )
			else:
				return self.vectorizer.transform( documents ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class TfidfVectorizer( Metric ):
	"""

		Purpose:
		---------

		Convert a collection of raw documents to a matrix of TF-IDF features.

	"""

	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize TfidfVectorizer.

		"""
		super( ).__init__( )
		self.vectorizer = sk.TfidfVectorizer( )


	def fit( self, documents: list[ str ], y: Optional[ np.ndarray ]=None ) -> object | None:
		"""

			Purpose:
			---------
			Fit the vectorizer to the documents.

			:param y:
			:type y:
			:param documents: List of text documents.
			:type documents: list[str]
		"""
		try:
			if documents is None:
				raise Exception( '"documents" cannot be None' )
			else:
				return self.vectorizer.fit( documents )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, documents: list[ str ] ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform documents to TF-IDF vectors.

			:param documents: List of text documents.
			:type documents: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		try:
			if documents is None:
				raise Exception( '"documents" cannot be None' )
			else:
				return self.vectorizer.transform( documents ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, documents: list[ str ], y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Fit and transform the documents.

			:param y:
			:type y:
			:param documents: List of text documents.
			:type documents: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		try:
			if documents is None:
				raise Exception( '"documents" cannot be None' )
			else:
				return self.vectorizer.fit_transform( documents ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def inverse_transform( self, documents: list[ str ] ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform documents to TF-IDF vectors.

			:param documents: List of text documents.
			:type documents: list[str]
			:return: TF-IDF vectorized output.
			:rtype: np.ndarray
		"""
		try:
			if documents is None:
				raise Exception( '"documents" cannot be None' )
			else:
				return self.vectorizer.inverse_transform( documents ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class CountVectorizer( Metric ):
	"""

		Purpose:
		---------
		Wrapper for sklearn's CountVectorizer.

		This class converts a collection of text documents to a matrix of token counts.
	"""

	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the CountVectorizerWrapper with default parameters.
		"""
		super( ).__init__( )
		self.vectorizer = sk.CountVectorizer( )


	def fit( self, documents: List[ str ], y: Optional[ np.ndarray ]=None ) -> object | None:
		"""

			Purpose:
			---------
			Convert a collection of text documents to a matrix of token counts.

			:param y:
			:type y:
			:param documents: List of input text documents.
			:type documents: List[str]

		"""
		try:
			if documents is None:
				raise Exception( '"documents" cannot be None' )
			else:
				return self.vectorizer.fit( documents )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, documents: List[ str ], y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			-------
			Transform documents into count vectors.

			:param y:
			:type y: np.ndarray
			:param documents: List of input text documents.
			:type documents: List[str]
			:return: Matrix of token counts.
			:rtype: np.ndarray

		"""
		try:
			if documents is None:
				raise Exception( '"documents" cannot be None' )
			else:
				return self.vectorizer.transform( documents ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, documents: List[ str ], y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Fit the vectorizer and transform the documents.

			:param y:
			:type y:
			:param documents: List of input text documents.
			:type documents: List[str]
			:return: Matrix of token counts.
			:rtype: np.ndarray

		"""
		try:
			if documents is None:
				raise Exception( '"documents" cannot be None' )
			else:
				return self.vectorizer.fit_transform( documents ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class HashingVectorizer( Metric ):
	"""

		Purpose:
		---------
		Convert a collection of text documents to a matrix of token occurrences.

	"""


	def __init__( self, num: int=1048576 ) -> None:
		"""

			Purpose:
			---------
			Initialize the HashingVectorizer with the desired number of features.

		"""
		super( ).__init__( )
		self.vectorizer = sk.HashingVectorizer( n_features=num )


	def transform( self, documents: List[ str ] ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform documents into hashed token vectors.

			:param documents: List of input text documents.
			:type documents: List[str]
			:return: Matrix of hashed features.
			:rtype: np.ndarray
		"""
		try:
			if documents is None:
				raise Exception( '"documents" cannot be None' )
			else:
				return self.vectorizer.transform( documents ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HasVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )