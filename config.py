'''
  ******************************************************************************************
      Assembly:                mathy
      Filename:                config.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="config.py" company="Terry D. Eppler">

	     config.py
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
    config.py
  </summary>
  ******************************************************************************************
'''
import os
import multiprocessing

# ------------- COMMON CONSTANTS ---------------------
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )
LOGO = r'resources/img/mathy_logo.png'
FAVICON = r'resources/favicon.ico'
BLUE_DIVIDER = "<div style='height:1.5px;align:left;background:#0078FC;margin:20px 0px 20px 0px;'></div>"
DEFAULT_DATA = r'stores/excel/Combined Schedules.xlsx'
DB_PATH = r'stores/sqlite/Data.db'
LLM_PATH = ''
DEFAULT_CTX = 4096
CORES = multiprocessing.cpu_count( )
MODES = [ 'Data Processing', 'Descriptive Statistics',  'Inferential Statistics', 'Anomaly Detection',
          'Feature Engineering', 'Classifications', 'Regressions', 'Clustering', 'Time-Series', 'Database' ]

MODE = { 'Data Profile': '🏗️ Data Profiling',
       'Descriptive Statistics': '🔍 Descriptive Statistics',
       'Inferential Statistics': '🧠 Inferential Statistics',
       'Anomaly Detection': '🛸 Anomaly Detection',
       'Data Plumbing': '🔧 Data Plumbing',
       'Feature Engineering': '🛠️ Feature Engineering',
       'Classifications': '📊 Classification Models',
       'Regressions': '📉 Regression Models',
       'Clustering': '🕸️ Clustering Models',
       'Time-Series': '⏱️ Time-Series Models',
       'Data Management': '💻 SQLite Database'  }

# ------------- DEFINITIONS ---------------------

PCA = r'''PCA (Principal Component Analysis) is a dimensionality reduction technique and helps
to reduce the number of features in a dataset while keeping the most important information. It changes
complex datasets by transforming correlated features into a smaller set of uncorrelated components.
It removes redundancy, improves computational efficiency and makes data easier to visualize and analyze.

PCA uses linear algebra to transform data into new features called principal components. It finds
these by calculating eigenvectors (directions) and eigenvalues (importance) from the covariance matrix.
PCA selects the top components with the highest eigenvalues and projects the data onto them simplify the dataset.
'''

CCA = r'''Canonical Correlation Analysis (CCA) is a multivariate statistical method used to
identify and quantify the relationships between two sets of variables, and measured on the same
subjects. It finds linear combinations of variables—canonical variates—that are maximally correlated.

'''

QQ_PLOT = r'''A Q-Q (Quantile-Quantile) plot is a graphical tool used to assess if a data set
follows a specific theoretical distribution (commonly normal) by plotting sample quantiles against
theoretical quantiles. Points falling along a straight 45-degree line indicate a strong match,
while deviations suggest differences in distribution, skewness, or outliers.
'''

DESCRIPTIVE_STATISTICS = r'''Descriptive statistics summarize and organize data features using
measures of central tendency (mean, median, mode), variability (range, standard deviation, variance),
and shape (skewness). They provide simple, actionable summaries of a sample's characteristics without
making inferences about a larger population. Key types include measures of distribution, central tendency, and dispersion
'''

INFERENTIAL_STATISTICS = r'''Inferential statistics allows researchers to draw conclusions, make predictions,
or generalize findings about a large population based on data analyzed from a smaller sample. It uses
probability theory and hypothesis testing to determine if patterns are significant or due to chance,
helping to make informed decisions despite data limitations.
'''

CORRELATION_ANALYSIS = r'''Correlation analysis is a statistical method used to measure the
strength and direction of the relationship between two variables, yielding a coefficient (\(r\))
between -1 and +1. It identifies patterns (positive/negative) but does not prove causation.
Common types include Pearson (linear), Spearman (monotonic), and Kendall, crucial for finance,
research, and data analysis
'''

CORRELATION_STRUCTURE = r'''Correlation structures define the pattern of dependence between observations
in a dataset, crucial for analyzing repeated measures or clustered data where observations within subjects
are correlated. Common types include compound symmetry (constant correlation), AR(1) (decaying correlation over time),
and unstructured (unique correlations)
'''

NORMALITY_TESTING = r'''A normality test determines if a data set is well-modeled by a normal
distribution, a key assumption for parametric tests like t-tests and ANOVA. It uses graphical
methods (Q-Q plots, histograms) or statistical tests (Shapiro-Wilk, Kolmogorov-Smirnov) to check
if data follows a bell-shaped curve. A non-significant result (\(p > 0.05\)) generally indicates
the data is normally distributed.
'''

SHAPIRO_WILK = r'''The Shapiro-Wilk test is a formal statistical method to determine if a data set
follows a normal distribution (bell-shaped curve), with 'H': Data is normally distributed. A p-value
< 0.5 indicates the data deviates significantly from normality. It is highly effective for small
sample sizes (< 50)
'''

ANOVA = r'''Analysis of Variance (ANOVA) is a statistical method used to compare the means of three
or more groups to determine if at least one group mean is significantly different from the others.
It evaluates the importance of1 or more factors by comparing the variance between groups to the
variance within groups using an F-statistic: (Between-group variance) divided by (Within-group variance).

ANOVA partitions the total variability of a dataset into two components: variance between sample
means and variance within each sample. If the variance between groups is significantly higher than
within-group variance, the means are likely different.
'''

CATEGORICAL_ASSOCIATION_TEST = r'''A categorical association test, primarily the
Chi-Square Test of Independence, determines if a significant relationship exists between two
categorical variables by comparing observed frequencies to expected frequencies in a contingency table.
It tests the null hypothesis that variables are independent (no association). Common methods include
Chi-Square for large samples, Fisher’s exact test for small samples, and McNemar's for paired data.
'''

PEARSON_COEFFICIENT = r'''linear relationship between two continuous variables, ranging from -1 to +1.
A value of +1 indicates a perfect positive linear relationship, -1 a perfect negative relationship,
and 0 no linear correlation.
'''

SPEARMAN_COEFFICIENT = r'''Spearman's rank correlation coefficient 'rho' or 's' is a non-parametric
measure that assesses the strength and direction of the monotonic relationship between two ranked or
continuous variables. Ranging from -1 to +1, it evaluates how well the relationship can be described
by a monotonic function, without requiring normally distributed data.
'''

ECDF = r'''The Empirical Cumulative Distribution Function (ECDF) is a step function that represents
the fraction of data points less than or equal to a specific value, providing an empirical estimate
of the underlying cumulative distribution. It is calculated by sorting  observations and increasing
the function by 1/n at each data point, with values ranging from 0 to 1.
'''

Z_SCORE = r'''Score flags observations whose values are a specified number of standard deviations
away from the mean. This method works best when the variable is roughly symmetric and not
dominated by extreme skew or heavy tails.
'''

MODIFIED_Z = r'''Modified Z-Score uses the median and median absolute deviation (MAD) instead of
the mean and standard deviation. It is more robust than the standard Z-Score when the data contain
skew, heavy tails, or existing outliers.
'''

IQR = r'''IQR Fence flags observations below Q1 - k×IQR or above Q3 + k×IQR, where IQR is the
interquartile range. This is a simple and robust rule for detecting unusually low or high values
without assuming normality.
'''

MAHALANOBIS = r'''Mahalanobis Distance detects multivariate outliers by measuring how far each
observation is from the center of the data while accounting for covariance between variables.
It is  useful when unusual combinations of values matter more than extreme values in a single column.
'''


ISOLATION_FOREST = r'''Isolation Forest is an ensemble method that isolates unusual observations
through random partitioning. Points that are easier to isolate are treated as anomalies. It works
well for  nonlinear and high-dimensional patterns and does not require the data to be normally distributed.
'''

LOF = r'''Local Outlier Factor (LOF) compares the local density of each observation to the density
of its nearest neighbors. Points that lie in much sparser neighborhoods than nearby points are
flagged as anomalies. It is useful for detecting local anomalies that may not look extreme globally.
'''

Z_THRESHOLD = r'''Sets the cutoff used by both Z-Score and Modified Z-Score. Larger values make the
 rule more conservative and reduce the number of observations flagged as anomalies.
'''

IQR_MULTIPLIER = r'''Sets the multiplier applied to the interquartile range when building the lower
and upper IQR fences. Larger multipliers widen the fence and make the rule less sensitive.
'''

LOF_K = r'''Sets the number of nearest neighbors used by Local Outlier Factor. Smaller values
emphasize very local structure, while larger values smooth the density comparison over a broader neighborhood.
'''

MIN_METHODS = r'''Controls the consensus threshold. A row must be flagged by at least this many
methods before it is included in the final anomaly table.
'''

ANALYSIS_SCALE = r'''When enabled, the selected variables are standardized for analysis only.
This puts variables on a comparable scale so that multivariate methods are less dominated by columns
with large numeric ranges. The underlying dataset is not changed.
'''

# ---------- Classifiers

LEAST_SQUARES = r'''Least Squares Regression fits a linear model with coefficients w = (w1, …, wp)
		to minimize the residual sum of squares between the observed targets
		in the dataset, and the targets predicted by the linear approximation.
'''

LOGISTIC_REGRESSION = r''''A machine learning algorithm used for binary classification
		(predicting one of two outcomes, e.g., yes/no) by modeling probabilities using a
		sigmoid function. It calculates the likelihood of an event occurring, making it ideal
		for spam detection, credit scoring, and medical diagnosis.
'''

RIDGE_CLASSIFIER = r''''A classifier that first converts binary targets to {-1, 1} and then treats the problem as a
		regression task, optimizing the same objective as above. The predicted class corresponds
		to the sign of the regressor’s prediction. For multiclass classification, the problem is
		treated as multi-output regression, and the predicted class corresponds to the output
		with the highest value.
'''

LASSO_CLASSIFIER = r''''(Least Absolute Shrinkage and Selection Operator) is a regression analysis
		method that performs both variable selection and regularization to enhance model prediction
		accuracy and interpretability. By applying an  penalty to the regression model, it shrinks
		less important feature coefficients to exactly zero, effectively removing them.
'''

GRADIENT_DESCENT = r''''Linear classifiers (SVM, logistic regression, etc.) with
		Stochastic Gradient Descent (SGD) training.  This estimator implements regularized
		linear models with stochastic gradient descent learning:
		
		The gradient of the loss is estimated each sample at a time and the model is updated along
		the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch
		(online/out-of-core) learning via the partial_fit method. For best results using the
		default learning rate schedule, the stores should have zero mean and unit variance.
'''

NEAREST_NEIGHBOR_CLASSFIER = r''''The principle behind the k-nearest neighbor methods is to find
		a predefined number of training samples closest in distance to the new point,
		and predict the label from these. The number of samples can be a user-defined constant
		(k-nearest neighbor rate), or vary based on the local density of points
		(radius-based neighbor rate).
		
		The distance can, in general, be any metric measure: standard Euclidean distance is the
		most common choice. Neighbors-based methods are known as non-generalizing
		machine rate methods, since they simply “remember” all of its training df
		(possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).
'''

DESICION_TREE_CLASSIFIER = r''''Decision Trees (DTs) are a non-parametric supervised learning method used for
		classification. The goal is to create a model that predicts the value of a
		target variable by learning simple decision rules inferred from the stores feature_names.

		A tree can be seen as a piecewise constant approximation. Decision trees learn from stores
		to approximate a sine curve with a set of if-then-else decision rules.
		The deeper the tree, the more complex the decision rules and the fitter the model.
'''

RANDOM_FOREST_CLASSIFIER = r''''In random forests, each tree in the ensemble is built from a sample
		drawn with replacement (i.e., a bootstrap sample) from the training set. Splitting each node
		during the construction of a tree, the best split is found either from all input
		feature_names or a random subset of size max_features. The injected randomness in forests
		yield decision trees with decoupled prediction errors. By taking an average of those predictions,
		errors can cancel out. Random forests achieve a reduced variance
		by combining diverse trees, sometimes at the cost of a slight increase in bias.
		The variance reduction is often significant hence yielding an overall better model.
'''

GRADIENT_BOOST_CLASSIFIER = r''''A Boost classifier is a meta-estimator that begins by fitting a classifier
		on the original dataset and then fits additional copies of the classifier on the
		same dataset but where the weights of incorrectly classified instances are
		adjusted such that subsequent classifiers focus more on difficult cases.
'''

ADAPTIVE_BOOST_CLASSIFIER = r''''A Boost classifier is a meta-estimator that begins by fitting a classifier
		on the original dataset and then fits additional copies of the classifier on the
		same dataset but where the weights of incorrectly classified instances are
		adjusted such that subsequent classifiers focus more on difficult cases.
'''

BAGGING_CLASSIFIER = r''''Bagging methods form a class of algorithms which build several instances of a black-box
		 estimator on random subsets of the original training set and then aggregate their
		 individual predictions to form a final prediction. These methods are used as a way
		 to reduce the variance of a base estimator (e.g., a decision tree), by introducing
		 randomization into its construction procedure and then making an ensemble out of it.
		 In many cases, bagging methods constitute a very simple way to improve with respect
		 to a single model, without making it necessary to adapt the underlying base algorithm.
		 As they provide a way to reduce overfitting, bagging methods work best with strong and
		 complex models (e.g., fully developed decision trees), in contrast with boosting methods
		 which usually work best with weak models (e.g., shallow decision trees).
'''

VOTING_CLASSFIER = r''''The Voting Model is to combine conceptually different machine rate
		classifiers and use a majority vote or the average predicted probabilities (soft vote)
		to predict the class target_names. Such a classifier can be useful for a set of equally
		well performing model in order to balance out their individual weaknesses.
'''

STACKING_MODEL = r''''Stack of estimators with a final classifier. Stacked generalization consists in stacking
		the output of individual estimator and use a classifier to compute the final prediction.
		Stacking allows to use the strength of each individual estimator by using their output
		as input of a final estimator. Note that estimators_ are fitted on the full X while
		final_estimator_ is trained using cross-validated predictions of the base
		estimators using cross_val_predict.
'''

SUPPORT_VECTOR_CLASSIFIER = r'''' Support Vector Classifier (SVC) is asupervised machine
		learning algorithm used primarily for classification, though it also handles regression.
		It works by finding an optimal "hyperplane"—a decision boundary—that maximizes the margin
		(distance) between different data classes, which improves prediction accuracy and
		generalization to new datais based on libsvm. The fit time scales at least quadratically
		with the number of samples  and may be impractical beyond tens of thousands of samples.
'''

MULTILAYER_PERCEPTRON_CLASSIFIER = r'''Model optimizes the squared error using LBFGS or
		stochastic gradient descent.

		Activation function for the hidden layers:
		- ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
		- ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
		- ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
		- ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
'''

# -------- Scalers

STANDARD_SCALER = r'''Standardize features by removing the mean and scaling to unit variance.
		The standard score of a sample x is calculated as: z = ( x - u ) / s,
		where u is the mean of the training samples or zero if with_mean=False,
		and s is the standard deviation of the training samples or one if
		with_std=False.
'''

MINMAX_SCALER = r'''Transform features by scaling each feature to a given range.
		This estimator scales and translates each feature individually such
		that it is in the given range on the training set, e.g. between zero
		and one. This transformation is often used as an alternative to zero
		mean, unit variance scaling.
'''

ROBUST_SCALER = r'''Remove the median and scale features according to the quantile range.

		By default, the quantile range is the interquartile range ( IQR ), which
		is the range between the 1st quartile ( 25th quantile ) and the 3rd
		quartile ( 75th quantile ).

		Centering and scaling happen independently on each feature by computing
		the relevant statistics on the samples in the training set. The median
		and interquartile range are then stored for use on later data during
		transformation.

		Robust scaling is useful when outliers would otherwise negatively affect
		mean- and variance-based scaling methods.
'''

NORMAL_SCALER = r''' Normalizes samples individually to unit norm. Each sample ( that is,
		each row of the feature matrix ) with at least one non-zero component
		is rescaled independently of the other samples so that its norm
		( l1, l2, or max ) equals one.

		This transformer can work with dense NumPy arrays and sparse matrices.
		Scaling inputs to unit norms is a common preprocessing step for text
		classification and clustering. For example, the dot product of two
		l2-normalized TF-IDF vectors is the cosine similarity between them.
'''

MAXABS_SCALER = r'''Scale each feature by its maximum absolute value.

		This estimator scales and transforms each feature individually such
		that the maximal absolute value of each feature in the training set
		will be 1.0. It does not shift or center the data, and therefore
		does not destroy sparsity.

		This scaler can also be applied to sparse CSR or CSC matrices.
		MaxAbsScaler does not reduce the effect of outliers; it only linearly
		scales them down.
'''

MINMAX_SCALER = r'''Transform features by scaling each feature to a given range.
		This estimator scales and translates each feature individually such
		that it is in the given range on the training set, e.g. between zero
		and one. This transformation is often used as an alternative to zero
		mean, unit variance scaling.

		Min-Max Scaler does not reduce the effect of outliers, but it linearly
		scales them down into a fixed range, where the largest occurring data
		point corresponds to the maximum value and the smallest one
		corresponds to the minimum value.
'''



