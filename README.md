###### Mathy-Py

<p align="center">
  <img src="https://github.com/is-leeroy-jenkins/Mathy/blob/main/resources/Mathy.png">
</p>


## 🧠 Overview

**Mathy** is a clean abstraction over clustering algorithms, preprocessing tools, 
and classification/regression models with custom wrappers, error handling, and diagnostic tools for 
machine-learning workflows. Mathy targets professionals in data science and analytics who require reusable, robust, and 
interpretable tools for building intelligent models—especially within government and research domains.
___
## 🧰 Core Modules
- **`data.py`**: Abstract base classes (`Model`, `Metric`) and the `Dataset` class for loading and splitting data.
- **`models.py`**: Wrappers for common classifiers/regressors such as:
- Perceptron, Ridge, SGD, MLP (Multilayer Perceptron)
- Decision Trees, KNN, Logistic Regression, SVMs, Ensemble Methods
- **`clusters.py`**: Unified interface for clustering techniques including:
- `KMeans`, `DBSCAN`, `Agglomerative`, and other sklearn-based clustering algorithms
- **`processors.py`**: Scalers and transformers for preprocessing:
- `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`, `OneHotEncoder`

## 🧠 Machine Learning
- Supports both classification and regression models.
- Model analysis using:
- Accuracy, Precision, Recall, F1 Score, ROC AUC, R², MAE, MSE, RMSE, etc.
- Seamless training, scoring, prediction (`train`, `score`, `project`, `analyze`)

## 📊 Clustering & Visualization
- Integrated support for:
  - KMeans: Centroid-based partitioning
  - DBSCAN: Density-based clustering
  - Agglomerative: Hierarchical linkage models
- 2D cluster visualizations using Matplotlib

## 🛠️ Data Preprocessing
- Scaling and normalization strategies
- Missing value imputation
- Transformation pipelines for numerical and categorical data

## 📁 Project Structure

```plaintext
mathy/
├── data.py           # Dataset handler and model interface
├── models.py         # Encapsulated classifiers and regressors
├── clusters.py       # Clustering models (KMeans, DBSCAN, etc.)
├── processors.py     # Preprocessing tools (scalers, encoders)
├── static/           # Static constants, helpers
├── README.md         # This file
```



### 📊 Classification Models
- Classification models are used to assign data samples into discrete categories or classes. These
models are foundational for tasks such as spam detection, disease diagnosis, and image recognition.
Below is a list of supported classifiers in Mathy:

___
| Class Name                   | Description                                                  |
|------------------------------|--------------------------------------------------------------|
| Model                        | Base wrapper for all classification models.                  |
| PerceptronClassifier         | Linear classifier using the perceptron learning rule.        |
| MultilayerClassification     | Multi-layer perceptron (MLP) for non-linear classification.  |
| RidgeClassification          | Classifier with L2 regularization to prevent overfitting.    |
| StochasticDescentClassification | Learns model using stochastic gradient descent.         |
| NearestNeighborClassification | Instance-based classifier using nearest neighbors.         |
| DecisionTreeClassification   | Splits data into decision paths using feature thresholds.    |
| RandomForestClassification   | Ensemble of decision trees trained with bagging.             |
| GradientBoostingClassification | Sequential ensemble reducing bias with boosting.          |
| AdaBoostClassification       | Boosts weak learners to correct classification errors.       |
| BaggingClassification        | Aggregates predictions from multiple bootstrapped models.    |
| VotingClassification         | Combines multiple models through majority voting.            |
| StackClassification          | Meta-learner trained on outputs of base classifiers.         |
| SupportVectorClassification  | Support Vector Machine (SVC) classifier.                     |

- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/classifications.py)




### 📈 Regression Models
- Regression models predict continuous numerical outcomes and are crucial in applications like
forecasting, pricing, and trend analysis. Mathy provides a range of linear and non-linear regression
models, listed below:

___
| Class Name                  | Description                                                        |
|-----------------------------|--------------------------------------------------------------------|
| Model                       | Base model interface for regression learners.                      |
| MultilayerRegression        | Multi-layer neural network for regression.                         |
| LinearRegressor             | Ordinary least squares regression.                                |
| RidgeRegression             | Linear regression with L2 regularization.                         |
| LassoRegression             | Linear regression with L1 penalty for sparsity.                   |
| ElasticNetRegression        | Combines L1 and L2 penalties for robustness.                      |
| LogisticRegressor           | Logistic regression for binary outcomes.                          |
| BayesianRidgeRegression     | Bayesian linear model with priors on coefficients.               |
| StochasticDescentRegression | Optimizes regression with stochastic gradient descent.            |
| NearestNeighborRegression   | Instance-based regression using k-nearest neighbors.              |
| DecisionTreeRegression      | Tree-based model for continuous targets.                          |
| RandomForestRegression      | Ensemble of trees trained on bootstrapped samples.                |
| GradientBoostingRegression  | Boosting technique for improved predictive accuracy.              |
| AdaBoostRegression          | Adaptive boosting for regression tasks.                           |
| BaggingRegression           | Bagging ensemble to reduce variance.                             |
| VotingRegression            | Aggregates predictions from multiple regressors.                  |
| StackRegression             | Trains meta-regressor on top of base models.                      |
| SupportVectorRegression     | Support Vector Regression (SVR) for high-dimensional data.        |

- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/regressions.py)



### 🔍 Clustering Models
- Clustering is an unsupervised technique used to discover natural groupings in data without labeled
outcomes. Mathy supports a variety of clustering algorithms suitable for both spherical and
irregular cluster shapes:

___
| Class Name                    | Description                                                     |
|-------------------------------|-----------------------------------------------------------------|
| Cluster                       | Abstract base for clustering methods.                           |
| KMeansClustering              | Clusters data into k partitions via centroid minimization.      |
| DbscanClustering              | Density-based clustering that handles noise and outliers.       |
| AgglomerativeClusteringModel  | Hierarchical clustering by iterative merging.                   |
| SpectralClusteringModel       | Uses spectral decomposition for clustering.                     |
| MeanShiftClustering           | Clusters by finding dense regions (modes) in feature space.     |
| AffinityPropagationClustering | Message-passing clustering based on exemplar similarity.        |
| BirchClustering               | Clusters large datasets using hierarchical CF trees.            |
| OpticsClustering              | Orders points to extract density-based clusters.                |

- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/clusters.py)



### 📁 Data
- Encapsulates datasets and implements dimensionality reduction, correlation analysis, and feature
selection.

  | Class Name | Description |
  |----------------------|-------------------------------------------------------------------|
  | Metric | Base class for transformations and evaluation metrics. |
  | VarianceThreshold | Removes low-variance features from the dataset. |
  | CorrelationAnalysis | Analyzes relationships using Canonical Correlation Analysis. |
  | ComponentAnalysis | Performs PCA or similar for dimensionality reduction. |
  | Dataset | Encapsulates data loading, transformation, and partitioning. |

- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/data.py)
___



### ⚡ Encoders
- Provides encoders to transform categorical features into numeric representations for model
compatibility.

| Class Name         | Description                                                       |
|--------------------|-------------------------------------------------------------------|
| Metric             | Abstract base for transformation classes.                         |
| OneHotEncoder      | Transforms categorical variables into binary one-hot vectors.     |
| OrdinalEncoder     | Encodes categories as integer values based on rank or order.      |
| LabelEncoder       | Converts labels into integer representations.                     |
| PolynomialFeatures | Generates polynomial combinations of features.                    |

- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/encoders.py)
___


### 🧠 Extractors
- Converts text data into structured numerical formats through vectorization and transformation.

| Class Name         | Description                                                               |
|--------------------|---------------------------------------------------------------------------|
| Metric             | Interface for feature extraction utilities.                               |
| TfidfTransformer   | Applies TF-IDF transformation to count matrices.                          |
| TfidfVectorizer    | Extracts TF-IDF features directly from raw text.                          |
| CountVectorizer    | Creates a document-term matrix of token counts.                           |
| HashingVectorizer  | Applies hashing trick to vectorize text without building vocabulary.      |

- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/extractors.py)
___


### 💻 Imputers
- Implements techniques for handling missing values using statistical or learned methods.
  | Class Name | Description |
  |--------------------------|----------------------------------------------------------------------|
  | Metric | Base for missing value processors. |
  | MeanImputer | Fills missing values with feature-wise means. |
  | NearestNeighborImputer | Imputes based on values of nearest neighbors. |
  | IterativeImputer | Estimates missing values using other features iteratively. |
  | SimpleImputer | Wrapper for sklearn’s basic imputation strategies (mean, median). |

- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/imputers.py)

___


### 🛠️ Scalers.py
- Provides feature scaling and normalization techniques critical for model convergence and
performance.

| Class Name       | Description                                                               |
|------------------|---------------------------------------------------------------------------|
| Metric           | Abstract base class for preprocessing utilities.                         |
| StandardScaler   | Standardizes features by removing mean and scaling to unit variance.     |
| MinMaxScaler     | Scales features to a defined range (usually [0, 1]).                      |
| RobustScaler     | Uses medians and IQR for outlier-resistant scaling.                      |
| NormalScaler     | Normalizes each sample to have unit norm.                                |

- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/scalers.py)
___



## 📦 Dependencies

###### Mathy requires Python 3.9+ and the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `pydantic`
- `boogr` (custom error handler module)


- To install all dependencies:

```
bash
pip install -r requirements.txt
```

## ⚡ Quickstart

#### 1. **Load and Split Data**  
- Load a dataset and define the target column.

```
python
from data import Dataset
import pandas as pd

df = pd.read_csv("data.csv")
dataset = Dataset(df, target="Label")
```

#### 2. **Preprocessing**

``` 
from processors import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(dataset.training_data)
y_train = dataset.training_values
```

#### 3. **Train and Evaluate Model**

``` 
from models import PerceptronClassifier
model = PerceptronClassifier()
model.train(X_train, y_train)
print(model.score(X_train, y_train))
print(model.analyze(X_train, y_train))
```

## 📝 License

Mathy is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Mathy/blob/main/LICENSE).

