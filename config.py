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
LOGO = r'resources/img/mathy_logo.ico'
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
       'Descriptive Statistics': '📊 Descriptive Statistics',
       'Inferential Statistics': '🧠 Inferential Statistics',
       'Anomaly Detection': '🛸 Anomaly Detection',
       'Data Plumbing': '🔧 Data Plumbing',
       'Feature Engineering': '🔩 Feature Engineering',
       'Classifications': '🔠 Classification Models',
       'Regressions': '📉 Regression Models',
       'Clustering': '🕸️ Clustering Models',
       'Time-Series': '⏱️ Time-Series Models',
       'Database': '💻 Database'  }

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
