
###### Mathy-Py

<p align="center">
  <img src="resources/Mathy.png" alt="Mathy logo" width="800">
</p>

---

## 🧠 Overview

A unified, modular framework for **machine learning (ML)** and **deep learning (DL)** that consolidates core analytic tasks — preprocessing, feature engineering, classification, regression, clustering, forecasting, and anomaly detection — into a coherent, composable toolkit.



| Category      | Core Methods                                                 | Optional               |
| ------------- | ------------------------------------------------------------ | ---------------------- |
| Models        | `train(X, y)` · `project(X, y=None)` · `score(X, y)`         | `analyze(X, y)`        |
| Preprocessors | `train(X[, y])` · `transform(X)` · `train_transform(X[, y])` | `inverse_transform(X)` |

This uniformity allows every class — from encoders to regressors — to interoperate seamlessly.

---

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

| Class Name         | Description (concise)          |
| ------------------ | ------------------------------ |
| Encoder        | Abstract base for encoders.    |
| OneHotEncoder  | One-hot (dummy) encoding.      |
| OrdinalEncoder | Ordinal category mapping.      |
| LabelEncoder   | Single-column label encoding.  |
| TargetEncoder  | Mean target encoding.          |


---

## ⚖️ Scalers  
| Class Name         | Description (concise)       |
| ------------------ | --------------------------- |
| Scaler         | Abstract base for scalers.  |
| StandardScaler | Z-score scaling.            |
| RobustScaler   | IQR-based robust scaling.   |
| NormalScaler   | L2 normalization.           |
| MinMaxScaler   | Min–Max feature scaling.    |


---

## 🩹 Imputers 

| Class Name           | Description (concise)                         |
| -------------------- | --------------------------------------------- |
| Imputer          | Abstract base for imputers.                   |
| SimpleImputer    | Mean/median/most-frequent/simple strategies.  |
| NearestImputer   | k-NN based imputation.                        |
| IterativeImputer | Iterative chained models imputation.          |


---

## 🧩 Transformers 

| Class Name           | Description (concise)                                                                  |
| -------------------- | -------------------------------------------------------------------------------------- |
| Transformer      | Abstract base for transformers.                                                        |
| CountVectorizer  | Bag-of-words counts with stopword support.                                             |
| TfidfTransformer | TF-IDF weighting transformer.                                                          |
| HashVectorizer   | Hashing trick vectorization.                                                           |
| TfidfVectorizer  | TF-IDF vectorizer (combined tokenizer + weighting). (Present in method paths/causes.)  |

---

## 🧪 Classifications 

| Class                  | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| PerceptronModel    | Single-layer perceptron for linear classification.      |
| LogisticModel      | Logistic regression classifier.                         |
| RidgeModel         | Ridge-regularized linear classifier.                    |
| SVMModel           | Support Vector Machine for classification.              |
| KnnModel           | K-Nearest Neighbors classifier.                         |
| DecisionTreeModel  | Non-parametric decision tree classifier.                |
| RandomForestModel  | Ensemble of randomized decision trees.                  |
| AdaBoostModel      | Adaptive boosting classifier.                           |
| GradientBoostModel | Gradient boosting ensemble classifier.                  |
| BaggingModel       | Bootstrap aggregation ensemble.                         |
| VotingModel        | Hard or soft voting ensemble of base classifiers.       |
| StackModel         | Stacked meta-classifier combining multiple learners.    |
| MlpModel           | Feedforward neural network for classification.          |
| SgdModel           | Linear model optimized via stochastic gradient descent. |

---

## 📈 Regressions 

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

| Class Name          | Description (concise)                                     |
| ------------------- | --------------------------------------------------------- |
| Outlier         | Abstract base for outlier/novelty models.                 |
| IsolationForest | Isolation Forest with contamination & decision function.  |
| OutlierFactor   | LocalOutlierFactor (supports novelty mode).               |
| OneClass        | One-Class SVM novelty detector.                           |
| EllipticSquare  | EllipticEnvelope (Gaussian/Mahalanobis).                  |


---

## ⏱️ Forecasting  

| Class Name          | Description (concise)                                |
| ------------------- | ---------------------------------------------------- |
| TimeSeries      | Base class for TS wrappers.                          |
| LaggingSeries   | OLS with lag features and recursive projection.      |
| ARIMA           | Statsmodels ARIMA(p,d,q) with fit/forecast/metrics.  |
| SARIMA          | Seasonal ARIMA via SARIMAX.                          |
| ExpandingWindow | Expanding-window CV splitter + visualization.        |

---

## 🧮 Features (`features.py`)

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

model = LogisticModel().train(X_scaled, y_enc)
print("Accuracy:", model.score(X_scaled, y_enc))
```

---

## 🧩 Design Principles

* **Unified API:** Every model and transformer implements the same verbs (`train`, `project`, etc.).
* **Composable Pipelines:** All wrappers interoperate natively.
* **Transparent Errors:** Common exception handling via `boogr.Error` and `ErrorDialog`.
* **Extensible Framework:** Add wrappers easily using the shared base interface.
* **Interoperable:** Fully compatible with `scikit-learn`, `statsmodels`, and `PyTorch`.

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
