'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                transformers.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="preprocessing.py" company="Terry D. Eppler">

     Mathy Preprocessing

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
	preprocessing.py
</summary>
******************************************************************************************
'''
from __future__ import annotations
from booger import Error, ErrorDialog
import numpy as np
from typing import Optional, List, Union, Any
import sklearn.feature_extraction.text as sk
import sklearn.preprocessing as skp
import sklearn.impute as ski
from pydantic import BaseModel


class Processor(  ):
	"""

		Purpose:
		---------
		Base interface for all preprocessors. Provides standard `fit`, `transform`, and
	    `fit_transform` methods.

	"""
	transformed_data: np.ndarray | None

	def __init__( self ):
		pass


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


class LabelBinarizer(  ):
	"""

		Purpose:
		_______
		Binarize labels in a one-vs-all fashion. Several regression and binary classification
		algorithms are available in scikit-learn. A simple way to extend these algorithms to the
		multi-class classification case is to use the so-called one-vs-all scheme.

		At learning time, this simply consists in learning one regressor or binary classifier
		per class. In doing so, one needs to convert multi-class labels to binary labels
		(belong or does not belong to the class). LabelBinarizer makes this process easy
		with the transform method.

		At prediction time, one assigns the class for which the corresponding model gave
		the greatest confidence. LabelBinarizer makes this easy with the inverse_transform method.


	"""
	label_binarizer: LabelBinarizer
	transformed_data: np.ndarray | None

	def __init__( self ) -> None:
		"""

		Purpose:
		_______
		Initializes the LabelBinarizerWrapper.

		"""
		self.label_binarizer = skp.LabelBinarizer( )
		self.transformed_data = None



	def fit( self, y: np.ndarray ) -> LabelBinarizer | None:
		"""

			Purpose:
			_______
			Fits the label binarizer on the input labels.

			Args:
				y (np.ndarray): Target labels for fitting.

		"""
		try:
			if y is None:
				raise Exception( '"y" cannot be None' )
			else:
				return self.label_binarizer.fit( y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'fit( self, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, y: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			_______
			Transforms labels into a binary format.

			Args:
				y (Union[List, np.ndarray]): Labels to encode.

			Returns:
				np.ndarray: Binary-encoded label matrix.
		"""
		try:
			if y is None:
				raise Exception( '"y" cannot be None' )
			else:
				self.transformed_data = self.label_binarizer.transform( y )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'fit( self, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, y: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			_______
			Fits the encoder and transforms the input labels in one step.

			Args:
				y np.ndarray: Labels to fit and transform.

			Returns:
				np.ndarray: Binary-encoded label matrix.
		"""
		try:
			if y is None:
				raise Exception( '"y" cannot be None' )
			else:
				self.transformed_data = self.label_binarizer.fit_transform( y )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'fit( self, y: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def inverse_transform( self, y: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			_______
			Converts binary matrix back to original labels.

			Args:
				y (np.ndarray): Binary-encoded label matrix.

			Returns:
				np.ndarray: Original labels.
		"""
		try:
			if y is None:
				raise Exception( '"y" cannot be None' )
			else:
				return self.label_binarizer.inverse_transform( y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'inverse_transform( self, y: np.ndarray, thresh: float=None ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class TfidfTransformer( Processor ):
	"""

		Purpose:
		---------
		Transform a count matrix to a normalized tf or tf-idf representation. Tf means
		term-frequency while tf-idf means term-frequency times inverse document-frequency.
		This is a common term weighting scheme in information retrieval, that has also found good
		use in document classification. The goal of using tf-idf instead of the raw frequencies of
		occurrence of a token in a given document is to scale down the impact of tokens that occur
		very frequently in a given corpus and that are hence empirically less informative than
		feature_names that occur in a small fraction of the training corpus.

		The formula that is used to compute the tf-idf for a term t of a document d in a
		document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf
		is computed as idf(t) = log [ n / df(t) ] + 1 (if smooth_idf=False), where n is the total
		number of documents in the document set and df(t) is the document frequency of t;
		the document frequency is the number of documents in the document set that contain
		the term t. The effect of adding “1” to the idf in the equation above is that
		terms with zero idf, i.e., terms that occur in all documents in a training set,
		will not be entirely ignored. (Note that the idf formula above differs from the
		standard textbook notation that defines the idf as idf(t) = log [ n / (df(t) + 1) ]).

	"""

	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize TfidfTransformer.
		"""
		super( ).__init__( )
		self.transformer = sk.TfidfTransformer( )
		self.transformed_data = None

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
		self.transformer.fit( X )

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
		self.transformed_data = self.transformer.transform( X ).toarray( )
		return self.transformed_data

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
		self.transformed_data = self.transformer.fit_transform( X ).toarray( )
		return self.transformed_data

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
				self.transformed_data = self.transformer.transform( documents ).toarray( )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class TfidfVectorizer( Processor ):
	"""

		Purpose:
		---------

		Convert a collection of raw documents to a matrix of TF-IDF feature_names. Equivalent to
		CountVectorizer followed by TfidfTransformer. Tf means term-frequency while tf–idf means
		 term-frequency times inverse document-frequency:

	"""

	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize TfidfVectorizer.

		"""
		super( ).__init__( )
		self.vectorizer = sk.TfidfVectorizer( )
		self.transformed_data = None

	def fit( self, documents: list[ str ], y: Optional[ np.ndarray ]=None ) -> TfidfVectorizer | None:
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
				self.vectorizer.fit( documents )
				return self
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
				self.transformed_data = self.vectorizer.transform( documents ).toarray( )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def fit_transform( self, documents: list[ str ],
	                   y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
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
				self.transformed_data = self.vectorizer.fit_transform( documents ).toarray( )
				return self.transformed_data
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


class CountVectorizer( Processor ):
	"""

		Purpose:
		---------
		Convert a collection of text documents to a matrix of token counts. This implementation
		produces a sparse representation of the counts using scipy.sparse.csr_matrix. If you do not
		provide an a-priori dictionary and you do not use an analyzer that does some kind of
		feature selection then the number of feature_names will be equal to the vocabulary
		size found by analyzing the data.

	"""

	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize the CountVectorizerWrapper with default parameters.
		"""
		super( ).__init__( )
		self.vectorizer = sk.CountVectorizer( )
		self.transformed_data = None


	def fit( self, documents: List[ str ], y: Optional[ np.ndarray ]=None ) -> CountVectorizer | None:
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
				self.vectorizer.fit( documents )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def transform( self, documents: List[ str ],
	               y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
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
				self.transformed_data = self.vectorizer.transform( documents ).toarray( )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def fit_transform( self, documents: List[ str ],
	                   y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
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
				self.transformed_data = self.vectorizer.fit_transform( documents ).toarray( )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class HashingVectorizer( Processor ):
	"""

		Purpose:
		---------
		Convert a collection of text documents to a matrix of token occurrences. It turns a
		collection of text documents into a scipy.sparse matrix holding token occurrence counts
		(or binary occurrence information), possibly normalized as token frequencies
		if norm=’l1’ or projected on the euclidean unit sphere if norm=’l2’. This text vectorizer
		implementation uses the hashing trick to find the token string name to feature integer
		index mapping.

	"""

	def __init__( self, num: int = 1048576 ) -> None:
		"""

			Purpose:
			---------
			Initialize the HashingVectorizer with the desired number of feature_names.

		"""
		super( ).__init__( )
		self.vectorizer = sk.HashingVectorizer( n_features = num )

	def transform( self, documents: List[ str ] ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform documents into hashed token vectors.

			:param documents: List of input text documents.
			:type documents: List[str]
			:return: Matrix of hashed feature_names.
			:rtype: np.ndarray
		"""
		try:
			if documents is None:
				raise Exception( '"documents" cannot be None' )
			else:
				self.transformed_data = self.vectorizer.transform( documents ).toarray( )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HasVectorizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class StandardScaler( Processor ):
	"""

		Purpose:
		--------
		Standardize feature_names by removing the mean and scaling to unit variance. The standard score
		of a sample x is calculated as: z = (x - u) / s where u is the mean of the training
		samples or zero if with_mean=False, and s is the standard deviation of the training
		samples or one if with_std=False.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.standard_scaler = skp.StandardScaler( )
		self.transformed_data = None


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> StandardScaler | None:
		"""


			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Input df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.standard_scaler.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StandardScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms the df using the fitted StandardScaler.

			Parameters:
			-----------
			X (np.ndarray): Input df.

			Returns:
			-----------
			np.ndarray: Scaled df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.standard_scaler.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def inverse_transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms into standardized data.

			:param X: List of text documents.
			:type X: list[str]
			:return: Standardized data.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( '"X" cannot be None' )
			else:
				return self.standard_scaler.inverse_transform( X ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class MinMaxScaler( Processor ):
	"""

		Purpose:
		---------
		Transforms feature_names by scaling each feature to a given range. This estimator scales and
		translates each feature individually such that it is in the given range on the
		training set, e.g. between zero and one.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.minmax_scaler = skp.MinMaxScaler( )
		self.transformed_data = None


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> MinMaxScaler | None:
		"""

			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Input df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.minmax_scaler.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms the df using the fitted MinMaxScaler.

			Parameters:
			-----------
			X (np.ndarray): Input df.

			Returns:
			-----------
			np.ndarray: Scaled df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.minmax_scaler.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def inverse_transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms into min-maxed data.

			:param X: List of text documents.
			:type X: list[str]
			:return: Standardized data.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( '"X" cannot be None' )
			else:
				return self.minmax_scaler.inverse_transform( X ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class RobustScaler( Processor ):
	"""

		Purpose:
		--------
		This Scaler removes the median and scales the data according to the
		quantile range (defaults to IQR: Interquartile Range).
		The IQR is the range between the 1st quartile (25th quantile)
		and the 3rd quartile (75th quantile).

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.robust_scaler = skp.RobustScaler( )
		self.transformed_data = None


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> RobustScaler | None:
		"""


			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Input df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.robust_scaler.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RobustScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms the df using the fitted RobustScaler.

			Parameters:
			-----------
			X (np.ndarray): Input df.

			Returns:
			-----------
			np.ndarray: Scaled df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.robust_scaler.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RobustScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def inverse_transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms into robust data.

			:param X: List of text documents.
			:type X: list[str]
			:return: Standardized data.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( '"X" cannot be None' )
			else:
				self.transformed_data = self.robust_scaler.inverse_transform( X ).toarray( )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class NormalScaler( Processor ):
	"""

		Purpose:
		---------
		Normalize samples individually to unit norm. Each sample (i.e. each row of the data matrix)
		with at least one non zero component is rescaled independently of other samples
		so that its regularlization (l1 or l2) equals one.

	"""

	def __init__( self, reg: str='l2' ) -> None:
		super( ).__init__( )
		self.normal_scaler = skp.Normalizer( norm=reg )
		self.transformed_data = None

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> NormalScaler | None:
		"""


			Purpose:
			---------
			Fits the normalizer (no-op for Normalizer).

			Parameters:
			-----------
			X (np.ndarray): Input df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.normal_scaler.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Normalizer'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			---------
			Applies normalization to each sample.

			Parameters:
			-----------
			X (np.ndarray): Input df.

			Returns:
			-----------
			np.ndarray: Normalized df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.normal_scaler.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Normalizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class OneHotEncoder( Processor ):
	"""

		Purpose:
		---------
		Encode categorical feature_names as a one-hot numeric array. The input to this transformer
		should be an array-like of integers or strings, denoting the values taken on by categorical
		(discrete) feature_names. The feature_names are encoded using a one-hot (aka ‘one-of-K’ or ‘dummy’)
		encoding scheme. This creates a binary column for each category and returns a sparse
		matrix or dense array (depending on the sparse_output parameter)

		By default, the encoder derives the categories based on the unique values in each feature.
		Alternatively, you can also specify the categories manually. This encoding is needed for
		feeding categorical data to many scikit-learn estimators, notably linear models and SVMs
		with the standard kernels. Note: a one-hot encoding of y labels should use a
		LabelBinarizer instead.

	"""

	def __init__( self, unknown: str = 'ignore' ) -> None:
		super( ).__init__( )
		self.hot_encoder = skp.OneHotEncoder( sparse_output=False, handle_unknown=unknown )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> OneHotEncoder | None:
		"""


			Purpose:
			---------
			Fits the hot_encoder to the categorical df.

			Parameters:
			-----------
			X (np.ndarray): Categorical text df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.hot_encoder.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			---------
			Transforms the text
			df into a one-hot encoded format.

			Parameters:
			-----------
			X (np.ndarray): Categorical text df.

			Returns:
			-----------
			np.ndarray: One-hot encoded matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.hot_encoder.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit the encoder and transform the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.hot_encoder.fit_transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class OrdinalEncoder( Processor ):
	"""


			Purpose:
			---------
			Encodes categorical feature_names as ordinal integers.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.ordinal_encoder = skp.OrdinalEncoder( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> OrdinalEncoder | None:
		"""

			Purpose:
			________
			Fits the ordial encoder to the categorical df.

			Parameters:
			_____
			X (np.ndarray): Categorical text df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.ordinal_encoder.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transforms the text df into ordinal-encoded format.


			Parameters:
			-----------
			X (np.ndarray): Categorical text df.

			Returns:
			-----------
			np.ndarray: Ordinal-encoded matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.ordinal_encoder.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit the encoder and transform the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray
		"""

		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.ordinal_encoder.fit_transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def inverse_transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform documents to TF-IDF vectors.

			:param X: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( '"documents" cannot be None' )
			else:
				return self.ordinal_encoder.inverse_transform( X ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class LabelEncoder( Processor ):
	"""

		Purpose:
		--------
		Encode target labels with value between 0 and n_classes-1. This transformer should be
		used to encode target values, i.e. y, and not the input X.

	"""

	def __init__( self ) -> None:
		"""

			Purpose:
			--------
			Initialize LabelEncoder.
		"""
		super( ).__init__( )
		self.label_encoder = skp.LabelEncoder( )


	def fit( self, labels: list[ str ], y: Optional[ np.ndarray ]=None ) -> LabelEncoder | None:
		"""

			Purpose:
			--------
			Fit the label encoder to the data.

			:param y:
			:type y:
			:param labels: List of labels.
			:type labels: list[str]
		"""
		try:
			if labels is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.label_encoder.fit( labels )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LabellEncoder'
			exception.method = 'fit( self, labels: list[str] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, labels: list[ str ] ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Transform labels to encoded form.

			:param labels: List of labels.
			:type labels: list[str]
			:return: Encoded labels.
			:rtype: np.ndarray
		"""

		try:
			if labels is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.label_encoder.transform( labels )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LabellEncoder'
			exception.method = 'fit( self, labels: list[str] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, labels: list[ str ],
	                   y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit and transform the label data.

			:param y:
			:type y:
			:param labels: List of labels.
			:type labels: list[str]
			:return: Encoded labels.
			:rtype: np.ndarray
		"""

		try:
			if labels is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.label_encoder.fit_transform( labels )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LabellEncoder'
			exception.method = 'fit_transform( self, labels: list[str] ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def inverse_transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform documents to TF-IDF vectors.

			:param X: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( '"X" cannot be None' )
			else:
				return self.label_encoder.inverse_transform( X ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class PolynomialFeatures( Processor ):
	"""

		Purpose:
		--------
        Wrapper for PolynomialFeatures.
    """

	def __init__( self, degree: int = 2 ) -> None:
		"""

			Purpose:
			--------
			Initialize PolynomialFeatures.

			:param degree: Degree of polynomial terms.
			:type degree: int
		"""
		super( ).__init__( )
		self.polynomial_features = skp.PolynomialFeatures( degree = degree )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> PolynomialFeatures| None:
		"""

			Purpose:
			--------
			Fit polynomial transformer to data.

			:param y:
			:type y:
			:param X: Feature matrix.
			:type X: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.polynomial_features.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'fit( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Transform data into polynomial feature_names.

			:param X: Feature matrix.
			:type X: np.ndarray
			:return: Transformed feature matrix.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.polynomial_features.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit and transform data using polynomial expansion.

			:param y:
			:type y:
			:param X: Feature matrix.
			:type X: np.ndarray
			:return: Transformed feature matrix.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.polynomial_features.fit_transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'fit_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class MeanImputer( Processor ):
	"""

		Purpose:
		-----------
		Fills missing target_values using the average.

	"""

	def __init__( self, strat: str = 'mean' ) -> None:
		super( ).__init__( )
		self.mean_imputer = ski.SimpleImputer( strategy = strat )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""


			Purpose:
			---------
			Fits the simple_imputer to the df.

			Parameters:
			-----------
			X (np.ndarray): Input df with missing target_values.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.mean_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""


			Purpose:
			---------
			Transforms the text
			df by filling in missing target_values.

			Parameters:
			-----------
			X (np.ndarray): Input df with missing target_values.

			Returns:
			-----------
			np.ndarray: Imputed df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.mean_imputer.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit the iterative imputer and transform the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( '"X" cannot be None' )
			else:
				return self.mean_imputer.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def inverse_transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform documents to TF-IDF vectors.

			:param X: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( '"X" cannot be None' )
			else:
				return self.mean_imputer.inverse_transform( X ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelEncoder'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class NearestNeighborImputer( Processor ):
	"""

		Purpose:
		---------
		Fills missing target_values using k-nearest neighbors.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.knn_imputer = ski.KNNImputer( )

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""

			Purpose:
			________
			Fits the simple_imputer to the df.

			Parameters:
			_____
			X (np.ndarray): Input df with missing target_values.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.knn_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborImputer'
			exception.method = 'fit( self, X: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			_________

			Transforms the text df by imputing missing target_values.

			Parameters:
			-----------
			X (np.ndarray): Input df

			Returns:
			-----------
			np.ndarray: Imputed df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.knn_imputer.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Fit the iterative imputer and transform the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.knn_imputer.fit_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class IterativeImputer( Processor ):
	"""


		Purpose:
		--------
		A strategy for imputing missing values by modeling each feature with
		missing values as a function of other feature_names in a round-robin fashion.
	"""

	def __init__( self, max: int = 10, rando: int = 0 ) -> None:
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
		self.iterative_imputer = ski.IterativeImputer( max_iter = max, random_state = rando )

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""

			Purpose:
			--------
			Fit the iterative imputer to the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.iterative_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Transform data by iteratively imputing missing values.

			:param X: Data to transform.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray

		"""
		return self.iterative_imputer.transform( X )

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit the iterative imputer and transform the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.iterative_imputer.fit_transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class SimpleImputer( Processor ):
	"""

		Wrapper for sklearn's SimpleImputer.

	"""

	def __init__( self, strategy: str = 'mean', fill_value: float = 0.0 ) -> None:
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
		self.simple_imputer = ski.SimpleImputer( strategy = strategy, fill_value = fill_value )

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""

			Purpose:
			--------
			Fit the imputer to the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.simple_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit( X: np.ndarray ) -> self'
			error = ErrorDialog( exception )
			error.show( )

	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Transform data by imputing missing values.

			:param X: Data to transform.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.simple_imputer.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit( X: np.ndarray ) -> self'
			error = ErrorDialog( exception )
			error.show( )

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Fit the imputer and transform the data.

			:param y:
			:type y:
			:param X: Input array with missing values.
			:type X: np.ndarray
			:return: Transformed data with imputed values.
			:rtype: np.ndarray
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.simple_imputer.fit_transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit_transform( X: np.ndarray ) -> np.ndarray '
			error = ErrorDialog( exception )
			error.show( )
