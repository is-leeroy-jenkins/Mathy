# 🧠 Mathy-Py




## 📦 Overview

**Mathy** is a clean abstraction over clustering algorithms, preprocessing tools, 
and classification/regression models with custom wrappers, error handling, and diagnostic tools for 
machine-learning workflows. Mathy targets professionals in data science and analytics who require reusable, robust, and 
interpretable tools for building intelligent models—especially within government and research domains.

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

## 💡 Design Principles
- Abstract base classes for extensibility
- Pydantic validation and type hinting
- Custom error handling via `boogr` for GUI/CLI debugging
- Adherence to clean architecture and SOLID principles

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

## 📊 Classification Models
> Classification models are used to assign data samples into discrete categories or classes. These
> models are foundational for tasks such as spam detection, disease diagnosis, and image recognition.
> Below is a list of supported classifiers in Mathy:

| Name                       | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| PerceptronClassifier       | Linear classifier based on the perceptron algorithm for binary classification. |
| RidgeClassifier            | Linear classifier using L2 regularization to prevent overfitting.           |
| SGDClassifier              | Stochastic Gradient Descent for efficient training of linear classifiers.  |
| LogisticRegression         | Probabilistic linear classifier suitable for binary and multi-class tasks. |
| KNeighborsClassifier       | Instance-based learner using proximity (k-nearest neighbors).               |
| DecisionTreeClassifier     | Tree-based model that splits data based on feature thresholds.             |
| RandomForestClassifier     | Ensemble method combining multiple decision trees via bagging.             |
| GradientBoostingClassifier | Boosted ensemble that reduces bias via stage-wise optimization.         |
| MLPClassifier              | Feedforward neural network (multi-layer perceptron) for non-linear learning. |

___

## 📈 Regression Models
> Regression models predict continuous numerical outcomes and are crucial in applications like
> forecasting, pricing, and trend analysis. Mathy provides a range of linear and non-linear regression
> models, listed below:

| Name                      | Description                                                             |
|---------------------------|-------------------------------------------------------------------------|
| LinearRegression          | Ordinary least squares linear regression.                               |
| RidgeRegression           | Linear regression with L2 regularization to prevent overfitting.        |
| LassoRegression           | Linear regression with L1 regularization for feature selection.         |
| ElasticNet                | Combines L1 and L2 penalties for balanced regularization.               |
| SGDRegressor              | Stochastic Gradient Descent optimizer for large-scale linear regression. |
| DecisionTreeRegressor     | Tree-based model for piecewise constant regression.                     |
| RandomForestRegressor     | Ensemble of decision trees using bootstrap aggregation.                 |
| GradientBoostingRegressor | Boosted ensemble for reducing bias in regression tasks.                 |
| MLPRegressor              | Multi-layer perceptron for capturing complex non-linear relationships.  |

___

## 🔍 Clustering Models
> Clustering is an unsupervised technique used to discover natural groupings in data without labeled
> outcomes. Mathy supports a variety of clustering algorithms suitable for both spherical and
> irregular cluster shapes:

| Name                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| KMeansClustering        | Partitions data into k clusters by minimizing within-cluster variance.     |
| DBSCANClustering        | Density-Based Spatial Clustering for detecting arbitrary-shaped clusters.  |
| AgglomerativeClustering | Hierarchical clustering using a bottom-up merging strategy.                |

___

## ⚙️ Preprocessors
> Preprocessing transforms raw input data into a format suitable for modeling. Mathy includes a
> robust suite of feature scaling, normalization, encoding, and imputation tools, detailed below:

| Processor         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| StandardScaler    | Scales features by removing the mean and scaling to unit variance (z-score normalization). |
| MinMaxScaler      | Transforms features by scaling each feature to a given range (typically 0 to 1).         |
| RobustScaler      | Scales features using statistics robust to outliers by removing the median and scaling by IQR. |
| Normalizer        | Normalizes samples individually to unit norm, often useful for text classification or clustering. |
| OneHotEncoder     | Encodes categorical variables as binary vectors using one-hot encoding.                   |
| OrdinalEncoder    | Encodes categorical features as ordinal integers based on their order of appearance.     |
| MeanImputer       | Fills missing values using the mean of each feature.                                     |
| NearestImputer    | Imputes missing values using the nearest neighbor algorithm based on feature similarity. |

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


