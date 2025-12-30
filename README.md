
###### Mathy-Py

<p align="center">
  <img src="resources/Mathy.png" alt="Mathy logo" width="800">
</p>



## 🧠 Overview


- A modular, object-oriented framework for machine learning and deep learning framework designed to unify data preprocessing, feature engineering, model training, and evaluation under a single, consistent interface. 
- Built entirely on top of open-source scientific libraries such as NumPy, Pandas, scikit-learn, Statsmodels, and PyTorch, to provide functionality across classifiers, regressors, clusterers, forecasters, and outlier detectors. 
- Every component—from encoders, scalers, and imputers to neural networks and ensemble models—follows the same method pattern (`train`, `project`, `score`, and `analyze`), enabling seamless pipeline integration, cross-module interoperability, and rapid experimentation. 
- Mathy offers clarity, consistency, and composability in machine learning workflows.

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/is-leeroy-jenkins/Mathy/blob/main/ipynb/board.ipynb)

![](https://github.com/is-leeroy-jenkins/Mathy/blob/main/resources/mathy-notebook.gif)

## 📦 Layout

```plaintext
mathy/
├── classifications.py     # Classification wrappers
├── regressions.py         # Regression wrappers
├── clusters.py            # Clustering algorithms
├── forecasting.py         # Time-series forecasting
├── outliers.py            # Outlier and novelty detection
├── encoders.py            # Label and categorical encoders
├── scalers.py             # Feature scaling wrappers
├── imputers.py            # Missing-value imputers
├── transformers.py        # Text and column transformers
├── features.py            # Feature selection utilities
├── boogr.py               # Error handling and diagnostics
├── minion.py              # Utility helpers
└── README.md              # This documentation
```

---

## 🔡 Encoders 

| Class Name                                                                      | Description (concise)          |
|---------------------------------------------------------------------------------| ------------------------------ |
| [Encoder](https://github.com/is-leeroy-jenkins/Mathy/blob/main/encoders.py#L54) | Abstract base for encoders.    |
| [OneHotEncoder](https://github.com/is-leeroy-jenkins/Mathy/blob/main/encoders.py#L139)                                                                 | One-hot (dummy) encoding.      |
| [OrdinalEncoder](https://github.com/is-leeroy-jenkins/Mathy/blob/main/encoders.py#L280)                                                                   | Ordinal category mapping.      |
| [LabelEncoder](https://github.com/is-leeroy-jenkins/Mathy/blob/main/encoders.py#L440)                                                                     | Single-column label encoding.  |
| [TargetEncoder](https://github.com/is-leeroy-jenkins/Mathy/blob/main/encoders.py#L583)                                                                    | Mean target encoding.          |


---

## ⚖️ Scalers  
- Classes that put features on a common scale by preventing features with larger values from disproportionately influencing the model.

| Class Name     | Description (concise)       |
|----------------| --------------------------- |
| [Scaler](https://github.com/is-leeroy-jenkins/Mathy/blob/main/scalers.py#L54)       | Abstract base for scalers.  |
| [StandardScaler](https://github.com/is-leeroy-jenkins/Mathy/blob/main/scalers.py#L139)  | Z-score scaling.            |
| [RobustScaler](https://github.com/is-leeroy-jenkins/Mathy/blob/main/scalers.py#L444)    | IQR-based robust scaling.   |
| [NormalScaler](https://github.com/is-leeroy-jenkins/Mathy/blob/main/scalers.py#L568)    | L2 normalization.           |
| [MinMaxScaler](https://github.com/is-leeroy-jenkins/Mathy/blob/main/scalers.py#L291)    | Min–Max feature scaling.    |


---

## 🩹 Imputers 
- Functionality to replace missing data with substituted values to ensure dataset completenes

| Class Name           | Description (concise)                         |
| -------------------- | --------------------------------------------- |
| [Imputer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/imputers.py#L55)           | Abstract base for imputers.                   |
| [SimpleImputer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/imputers.py#L533)     | Mean/median/most-frequent/simple strategies.  |
| [NearestImputer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/imputers.py#L277)    | k-NN based imputation.                        |
| [IterativeImputer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/imputers.py#L407)  | Iterative chained models imputation.          |


---

## 🧩 Transformers 
-  Algorithms that processes the input sequence, converting it into a rich, contextualized representation.
-  NLP classes for the conversion text features

| Class Name                                                                                    | Description (concise)                                 |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------|
| [Transformer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/transformers.py#L56)       | Abstract base for transformers.                       |
| [CountVectorizer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/transformers.py#L1185) | Bag-of-words counts with stopword support.            |
| [TfidfTransformer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/transformers.py#L637) | TF-IDF weighting transformer.                         |
| [HashVectorizer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/transformers.py#L1355)  | Hashing trick vectorization.                          |
| [TfidfVectorizer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/transformers.py#L953)  | TF-IDF vectorizer (combined tokenizer + weighting)    |
 | [LabelBinarizer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/transformers.py#L281)    | Learning one regressor or binary classifier per class |
 | [Binarizer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/transformers.py#L141)         | Set feature values to 0 or 1)                         |
 | [MultiLabelBinarizer](https://github.com/is-leeroy-jenkins/Mathy/blob/main/transformers.py#L468)| Transform between iterables and a multilabel format   |

---

## 🧪 Classifications 
- Supervised learning algorythms that assign data points to predefined categories to learn patterns.
- Training to predict category of new, unseen data

| Class              | Description                                            |
|--------------------|--------------------------------------------------------|
| Perceptron         | Single-layer perceptron for linear classification.     |
| LogisticRegression | Logistic regression classifier.                        |
| Ridge              | Ridge-regularized linear classifier.                   |
| SupportVector      | Support Vector Machine for classification.             |
| NearestNeighbor    | K-Nearest Neighbors classifier.                        |
| DecisionTree       | Non-parametric decision tree classifier.               |
| RandomForest       | Ensemble of randomized decision trees.                 |
| AdaptiveBoost      | Adaptive boosting classifier.                          |
| GradientBoost      | Gradient boosting ensemble classifier.                 |
| BaggingModel       | Bootstrap aggregation ensemble.                        |
| VotingModel        | Hard or soft voting ensemble of base classifiers.      |
| StackingModel      | Stacked meta-classifier combining multiple learners.   |
| LeastSquares       | OLS regression for classification.                     |
| GradientDescent    | Linear model optimized via stochastic gradient descent. |
| Lasso              | A linear model that estimates sparse coefficients    | 

---

## 📈 Regressions 
- Regression in machine learning refers to a set of supervised learning techniques used to predict a continuous output variable based on one or more input variables. 
- Unlike classification, which predicts discrete categories, regression models predict numerical values.

| Class Name               | Description (concise)                                                 |
| ------------------------ | --------------------------------------------------------------------- |
| Regressor            | Abstract base for all regressors.                                     |
| MultilayerPerceptron | MLP regressor.                                                        |
| LeastSquares         | Ordinary least squares (OLS).                                         |
| Ridge                | L2-regularized linear regression. (See `plot_ridge_path` code path.)  |
| Lasso                | L1-regularized linear regression.                                     |
| ElasticNet           | Combined L1/L2 penalty.                                               |
| BayesianRidge        | Bayesian linear regression.                                           |
| GaussianProcess      | Gaussian process regressor (GPR). (Imported/used in module.)          |
| GradientDescent      | SGD regressor.                                                        |
| NearestNeighbor      | k-NN regressor.                                                       |
| DecisionTree         | CART regression tree.                                                 |
| RandomForest         | Random forest regressor.                                              |
| GradientBoost        | Gradient boosting regressor.                                          |
| AdaBoost             | Adaptive boosting regressor.                                          |
| BaggingModel         | Bagging meta-regressor.                                               |
| VotingModel          | Voting regressor.                                                     |
| StackModel           | Stacked meta-regressor.                                               |


---

## 🧭 Clusters 
- Unsupervised learning technique used to group similar data points into clusters without requiring any prior knowledge or labels about the data. 
- The core idea is that data points within the same cluster exhibit more similarity to each other than to those in different clusters

| Class Name        | Description (concise)                                                   |
| ----------------- | ----------------------------------------------------------------------- |
| Cluster       | Abstract base for all clustering models.                                |
| KMeans        | Lloyd-style centroid clustering; includes elbow/visualization helpers.  |
| DBSCAN        | Density-based clustering.                                               |
| Agglomerative | Hierarchical/agglomerative clustering.                                  |
| OPTICS        | Ordering-based density clustering.                                      |
| MeanShift     | Mode-seeking (kernel) clustering.                                       |
| Spectral      | Spectral graph clustering (normalized cuts).                            |
| Birch         | CF-tree incremental clustering.                                         |



---

## 🔍 Outliers 
- Data points that are significantly different from other observations.  
- Can arise from errors, fraud, or natural deviations. 
- Can be detected using methods like the z-score or the interquartile range (IQR), and then either remove them, use a robust model like a tree-based method, or transform the data to reduce their influence

| Class Name          | Description (concise)                                     |
| ------------------- | --------------------------------------------------------- |
| Outlier         | Abstract base for outlier/novelty models.                 |
| IsolationForest | Isolation Forest with contamination & decision function.  |
| OutlierFactor   | LocalOutlierFactor (supports novelty mode).               |
| OneClass        | One-Class SVM novelty detector.                           |
| EllipticSquare  | EllipticEnvelope (Gaussian/Mahalanobis).                  |


---

## ⏱️ Forecasting  
- Algorithms to analyze data points indexed in a time sequence.
- Trains model to find patterns and make predictions about future values

| Class Name          | Description (concise)                                |
| ------------------- | ---------------------------------------------------- |
| TimeSeries      | Base class for TS wrappers.                          |
| LaggingSeries   | OLS with lag features and recursive projection.      |
| ARIMA           | Statsmodels ARIMA(p,d,q) with fit/forecast/metrics.  |
| SARIMA          | Seasonal ARIMA via SARIMAX.                          |
| ExpandingWindow | Expanding-window CV splitter + visualization.        |

---

## 🧮 Feature Selection
- Finding a subset of relevant input features from a dataset.
- Machine learning models to reduce complexity, decrease training time, prevent overfitting, and improve accuracy and interpretability.

| Class Name        | Description (concise)                                              |
| ----------------- | ------------------------------------------------------------------ |
| Selector      | Base for feature selectors.                                        |
| SelectBest    | `SelectKBest` with configurable `score_func` and `k`.              |
| SelectPercent | `SelectPercentile` with configurable `score_func` and percentile.  |

---

## ⚙️ Quickstart Example

```python
import pandas as pd
from encoders import LabelEncoder
from scalers import StandardScaler
from classifications import LogisticModel

df = pd.read_csv("data.csv")
y = df.pop("Label").values
X = df.values

y_enc = LabelEncoder().train_transform(y)
X_scaled = StandardScaler().train_transform(X)

model = LogisticRegression().train(X_scaled, y_enc)
print("Accuracy:", model.score(X_scaled, y_enc))
```


---

## 🧠 Dependencies

| Package / Module                   | Used For                                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **numpy** (`np`)                   | Arrays, math, window ops.                                                                                                 |
| **pandas** (`pd`)                  | DataFrames for reports/metrics.                                                                                           |
| **matplotlib.pyplot** (`plt`)      | Visualizations (clusters, CV splits, anomaly bars).                                                                       |
| **seaborn** (`sns`)                | Bar plots in outlier analysis.                                                                                            |
| **scikit-learn** (`sklearn.*`)     | Core ML estimators: classifiers, regressors, clustering, encoders, scalers, imputers, metrics, model selection utilities. |
| **statsmodels** (`statsmodels.*`)  | Time-series (ARIMA/SARIMAX), OLS for `LaggingSeries`.                                                                     |
| **typing** (`Optional`, etc.)      | Type hints across all modules.                                                                                            |
| **itertools** (`combinations`)     | Feature set enumeration in selectors.                                                                                     |
| **boogr** (`Error`, `ErrorDialog`) | Standardized exception handling/log dialogs. (Local module)                                                               |


---

## 📄 License

MIT License © 2022–2025 **Terry D. Eppler**
Contact: [terryeppler@gmail.com](mailto:terryeppler@gmail.com)

---
