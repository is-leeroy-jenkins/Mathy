'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                Extractors.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="Extractors.py" company="Terry D. Eppler">

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
	Encoders.py
</summary>
******************************************************************************************
'''
from Data import Metric
from Booger import Error, ErrorDialog
import numpy as np
from typing import Optional, List
import sklearn.feature_extraction.text as sk


class TfidfTransformer( Metric ):
	"""
	Wrapper for TfidfTransformer.
	"""

	def __init__( self ) -> None:
		"""
		Initialize TfidfTransformer.
		"""
		super( ).__init__( )
		self.transformer = sk.TfidfTransformer( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> None:
		"""
		Fit the transformer to a count matrix.

		:param y:
		:type y:
		:param X: Input count matrix.
		:type X: np.ndarray
		"""
		self.transformer.fit ( X)


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""
		Transform a count matrix to TF-IDF.

		:param X: Input count matrix.
		:type X: np.ndarray
		:return: TF-IDF matrix.
		:rtype: np.ndarray
		"""
		return self.transformer.transform ( X ).toarray ()


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""
		Fit and transform the count matrix.

		:param y:
		:type y:
		:param X: Input count matrix.
		:type X: np.ndarray
		:return: TF-IDF matrix.
		:rtype: np.ndarray
		"""
		return self.transformer.fit_transform ( X ).toarray ()


class TfidfVectorizer( Metric ):
	"""
	Wrapper for TfidfVectorizer.
	"""

	def __init__( self ) -> None:
		"""
		Initialize TfidfVectorizer.
		"""
		super( ).__init__( )
		self.vectorizer = sk.TfidfVectorizer( )


	def fit( self, documents: list[ str ], y: Optional[ np.ndarray ]=None ) -> None:
		"""
		Fit the vectorizer to the documents.

		:param y:
		:type y:
		:param documents: List of text documents.
		:type documents: list[str]
		"""
		self.vectorizer.fit( documents )


	def transform( self, documents: list[ str ] ) -> np.ndarray:
		"""
		Transform documents to TF-IDF vectors.

		:param documents: List of text documents.
		:type documents: list[str]
		:return: TF-IDF vectorized output.
		:rtype: np.ndarray
		"""
		return self.vectorizer.transform( documents ).toarray( )

	def fit_transform( self, documents: list[ str ], y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""
		Fit and transform the documents.

		:param y:
		:type y:
		:param documents: List of text documents.
		:type documents: list[str]
		:return: TF-IDF vectorized output.
		:rtype: np.ndarray
		"""
		return self.vectorizer.fit_transform( documents ).toarray( )


class CountVectorizer( Metric ):
	"""
	Wrapper for sklearn's CountVectorizer.

	This class converts a collection of text documents to a matrix of token counts.
	"""

	def __init__( self ) -> None:
		"""
			Initialize the CountVectorizerWrapper with default parameters.
		"""
		super( ).__init__( )
		self.vectorizer = sk.CountVectorizer( )

	def fit( self, documents: List[ str ], y: Optional[ np.ndarray ]=None ) -> None:
		"""

			Purpose:
			---------
			Fit the vectorizer on a list of documents.

			:param y:
			:type y:
			:param documents: List of input text documents.
			:type documents: List[str]

		"""
		self.vectorizer.fit( documents )

	def transform( self, documents: List[ str ], y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""

			Purpose:
			-------
			Transform documents into count vectors.

			:param y:
			:type y:
			:param documents: List of input text documents.
			:type documents: List[str]
			:return: Matrix of token counts.
			:rtype: np.ndarray

		"""
		return self.vectorizer.transform( documents ).toarray( )

	def fit_transform( self, documents: List[ str ], y: Optional[ np.ndarray ]=None ) -> np.ndarray:
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
		return self.vectorizer.fit_transform( documents ).toarray( )


class HashingVectorizer( Metric ):
	"""

		Wrapper for sklearn's HashingVectorizer.
		This class transforms text documents to feature vectors using the hashing trick.

	"""

	def __init__( self, n_features: int=1048576 ) -> None:
		"""
		Initialize the HashingVectorizer with the desired number of features.

		:param n_features: Number of features (columns) in the output vectors.
		:type n_features: int
		"""
		super( ).__init__( )
		self.vectorizer = sk.HashingVectorizer( n_features = n_features )

	def transform( self, documents: List[ str ] ) -> np.ndarray:
		"""
		Transform documents into hashed token vectors.

		:param documents: List of input text documents.
		:type documents: List[str]
		:return: Matrix of hashed features.
		:rtype: np.ndarray
		"""
		return self.vectorizer.transform( documents ).toarray( )