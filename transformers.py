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
import sklearn.impute as im
import sklearn.preprocessing as pp

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

class LabelBinarizer( Transformer ):
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
	transformer: pp.LabelBinarizer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

		Purpose:
		_______
		Initializes the LabelBinarizerWrapper.

		"""
		super( ).__init__( )
		self.transformer = pp.LabelBinarizer( )
		self.transformed_data = None
	
	@property
	def classes( self ) -> List[ str ]:
		if self.transformer.classes_ is None:
			raise AttributeError( 'LabelBinarizer has not been initialized.' )
		else:
			return self.transformer.classes_
	
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
			self.transformer.fit( y )
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
			self.transformed_data = self.transformer.transform( y )
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
			self.transformed_data = self.transformer.fit_transform( y )
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
			return self.transformer.inverse_transform( Y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
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
	transformer: sk.TfidfTransformer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize TfidfTransformer.
		"""
		super( ).__init__( )
		self.transformer = sk.TfidfTransformer( )
		self.transformed_data = None
	
	@property
	def idf_vector( self ) -> np.ndarray:
		if self.transformer.idf_ is None:
			raise AttributeError( 'TfidfTransformer must be initialized' )
		else:
			return self.transformer.idf_
	
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
			self.transformer.fit( X )
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
			self.transformed_data = self.transformer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'transform( self, X: np.ndarray, y: np.ndarray=None  ) -> np.ndarray'
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
			self.transformed_data = self.transformer.fit_transform( X ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'fit_transform( self, X: np.ndarray, y: np.ndarray=None ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
