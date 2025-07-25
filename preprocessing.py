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
from typing import Optional, List
import sklearn.feature_extraction.text as sk
import sklearn.preprocessing as skp
import sklearn.impute as ski
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

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> None:
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

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> None:
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
		return self.transformer.transform( X ).toarray( )

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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
		return self.transformer.fit_transform( X ).toarray( )

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

	def fit( self, documents: list[ str ], y: Optional[ np.ndarray ] = None ) -> object | None:
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

	def fit_transform( self, documents: list[ str ],
	                   y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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

	def fit( self, documents: List[ str ], y: Optional[ np.ndarray ] = None ) -> object | None:
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

	def transform( self, documents: List[ str ],
	               y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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

	def fit_transform( self, documents: List[ str ],
	                   y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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

	def __init__( self, num: int = 1048576 ) -> None:
		"""

			Purpose:
			---------
			Initialize the HashingVectorizer with the desired number of features.

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


class StandardScaler( Metric ):
	"""

		Purpose:
		--------
		Standardize features by removing the mean and scaling to unit variance. The standard score
		of a sample x is calculated as: z = (x - u) / s where u is the mean of the training
		samples or zero if with_mean=False, and s is the standard deviation of the training
		samples or one if with_std=False.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.standard_scaler = skp.StandardScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> StandardScaler | None:
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


class MinMaxScaler( Metric ):
	"""

		Purpose:
		---------
		Transforms features by scaling each feature to a given range. This estimator scales and
		translates each feature individually such that it is in the given range on the
		training set, e.g. between zero and one.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.minmax_scaler = skp.MinMaxScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> MinMaxScaler | None:
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
				return self.standard_scaler.inverse_transform( X ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class RobustScaler( Metric ):
	"""

		Purpose:
		--------
		This Scaler wraps the RobustScaler and removes the median and scales
		the data according to the quantile range (defaults to IQR: Interquartile Range).
		The IQR is the range between the 1st quartile (25th quantile)
		and the 3rd quartile (75th quantile).

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.robust_scaler = skp.RobustScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> RobustScaler | None:
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
				return self.robust_scaler.transform( X )
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
				return self.robust_scaler.inverse_transform( X ).toarray( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class NormalScaler( Metric ):
	"""

		Purpose:
		---------
		Normalize samples individually to unit norm. Each sample (i.e. each row of the data matrix)
		with at least one non zero component is rescaled independently of other samples
		so that its norm (l1 or l2) equals one.

	"""

	def __init__( self, norm: str = 'l2' ) -> None:
		super( ).__init__( )
		self.normal_scaler = skp.Normalizer( norm = norm )

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> NormalScaler | None:
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
				return self.normal_scaler.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Normalizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

class OneHotEncoder( Metric ):
	"""

		Purpose:
		---------
		Encodes categorical features as a one-hot numeric array.

	"""

	def __init__( self, unknown: str = 'ignore' ) -> None:
		super( ).__init__( )
		self.hot_encoder = skp.OneHotEncoder( sparse_output = False, handle_unknown = unknown )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> OneHotEncoder | None:
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

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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


class OrdinalEncoder( Metric ):
	"""


			Purpose:
			---------
			Encodes categorical features as ordinal integers.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.ordinal_encoder = skp.OrdinalEncoder( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> OrdinalEncoder | None:
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


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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


class LabelEncoder( Metric ):
	"""

		Purpose:
		--------
		Wrapper for LabelEncoder.
	"""

	def __init__( self ) -> None:
		"""

			Purpose:
			--------
			Initialize LabelEncoder.
		"""
		super( ).__init__( )
		self.label_encoder = skp.LabelEncoder( )


	def fit( self, labels: list[ str ], y: Optional[ np.ndarray ] = None ) -> LabelEncoder | None:
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
	                   y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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


class PolynomialFeatures( Metric ):
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


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> PolynomialFeatures| None:
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
			exception.module = 'Mathy'
			exception.cause = 'PolynomialFeatures'
			exception.method = 'fit( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			--------
			Transform data into polynomial features.

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


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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

class MeanImputer( Metric ):
	"""

		Purpose:
		-----------
		Fills missing target_values using the average.

	"""

	def __init__( self, strat: str = 'mean' ) -> None:
		super( ).__init__( )
		self.mean_imputer = ski.SimpleImputer( strategy = strat )

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
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

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
		"""
		Fit the iterative imputer and transform the data.

		:param y:
		:type y:
		:param X: Input array with missing values.
		:type X: np.ndarray
		:return: Transformed data with imputed values.
		:rtype: np.ndarray
		"""
		return self.mean_imputer.fit_transform( X )

class NearestNeighborImputer( Metric ):
	"""

		Purpose:
		---------
		Fills missing target_values using k-nearest neighbors.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.knn_imputer = ski.KNNImputer( )

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ):
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

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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

class IterativeImputer( Metric ):
	"""


		Purpose:
		--------
		A strategy for imputing missing values by modeling each feature with
		missing values as a function of other features in a round-robin fashion.
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

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
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

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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

class SimpleImputer( Metric ):
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

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
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

	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray | None:
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
