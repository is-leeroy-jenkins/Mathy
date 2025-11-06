## .
<p align="left">
  <img src="resources/Mathy.png" alt="Mathy logo" width="800">
</p>

## 🧠 Overview

A modular ML/DL framework unifying **supervised learning**, **unsupervised learning**, **forecasting**, and **data preprocessing** through a uniform API design.
Every model, transformer, or preprocessor implements:

| Category      | Core Methods                                                 | Optional               |
| ------------- | ------------------------------------------------------------ | ---------------------- |
| Models        | `train(X, y)` · `project(X, y=None)` · `score(X, y)`         | `analyze(X, y)`        |
| Preprocessors | `train(X[, y])` · `transform(X)` · `train_transform(X[, y])` | `inverse_transform(X)` |

This symmetry allows easy chaining in pipelines and reliable cross-module reuse.

---

## 📦 Project Layout

```plaintext
mathy/
├── classifications.py     # Classifier wrappers
├── regressions.py         # Regression wrappers
├── clusters.py            # Clustering algorithms
├── forecasting.py         # ARIMA, SARIMA, Lagging OLS
├── outliers.py            # Outlier and novelty detection
├── encoders.py            # Label/Ordinal/Target/One-Hot encoders
├── scalers.py             # Standard, MinMax, Robust, Normal
├── imputers.py            # Simple, KNN, Iterative imputers
├── features.py            # Feature selection and ranking
├── transformers.py        # Text and column transformers (TF-IDF, Binarizers, etc.)
├── boogr.py               # Error & diagnostic framework
├── minion.py              # Utility and helper functions
└── README.md              # This documentation
```

---

## 🧰 Core Abstractions

All Mathy components derive from minimal base classes that enforce the method contracts above.
The benefit: uniform handling across supervised models, unsupervised models, and preprocessing utilities.

---

## 🔡 Preprocessing Stack

### 1. Encoders

From `encoders.py`:

* `LabelEncoder` — encode labels to integers
* `OrdinalEncoder` — numeric mapping of categorical features
* `OneHotEncoder` — dummy variable expansion
* `TargetEncoder` — supervised category encoding

---

### 2. Scalers

From `scalers.py`:

* `StandardScaler` — mean=0, variance=1
* `MinMaxScaler` — rescale to [0, 1]
* `RobustScaler` — median and IQR normalization
* `NormalScaler` — L1/L2 normalization per sample

---

### 3. Imputers

From `imputers.py`:

* `SimpleImputer` — mean, median, mode, or constant fill
* `NearestImputer` — K-NN distance imputation
* `IterativeImputer` — regression-based iterative strategy
* `MeanImputer` — shorthand wrapper for mean substitution

---

### 4. Transformers 

From `transformers.py`, a full abstraction for binarization, vectorization, and column composition.

| Class                     | Description                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| **`Transformer`**         | Abstract base class defining `train`, `transform`, `train_transform`, and `inverse_transform`.      |
| **`Binarizer`**           | Converts features above a threshold to 1; otherwise 0. Often used on count or boolean data.         |
| **`LabelBinarizer`**      | One-vs-all binarization for class labels. Supports `inverse_transform` to recover original classes. |
| **`MultiLabelBinarizer`** | Handles sets or lists of labels per instance (multi-label classification).                          |
| **`TfidfTransformer`**    | Converts count matrices to TF-IDF representations.                                                  |
| **`TfidfVectorizer`**     | Directly vectorizes raw text to TF-IDF (learns vocabulary + idf weights).                           |
| **`CountVectorizer`**     | Converts text to token count matrices.                                                              |
| **`HashVectorizer`**      | Stateless hashing-based vectorizer — scalable and memory-efficient.                                 |
| **`ColumnTransformer`**   | Combines multiple transformers on different feature subsets (tabular pipelines).                    |

These provide the full text-processing and column-wise transformation capabilities missing from the older README.

Example (TF-IDF):

```python
from transformers import TfidfVectorizer

docs = ["Mathy is modular", "Mathy unifies preprocessing"]
tfidf = TfidfVectorizer(max_features=10).train_transform(docs, None)
print(tfidf.shape)
```

Example (Column transformer):

```python
from transformers import ColumnTransformer
from scalers import StandardScaler
from encoders import OneHotEncoder

transformers = [
    ("numeric", StandardScaler(), ["age", "income"]),
    ("categorical", OneHotEncoder(), ["gender", "region"])
]
ct = ColumnTransformer(transformers=transformers)
X_new = ct.train_transform(df, None)
```

---

## 🧪 Supervised Learning

From `classifications.py` and `regressions.py`:

* Logistic, Ridge, Lasso, ElasticNet, Perceptron, MLP, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, SVC/SVR, Gaussian Process, Bayesian Ridge, KNN, SGD, Bagging, Voting, Stacking, etc.
* Consistent signatures:

  ```python
  model.train(X, y)
  y_pred = model.project(X)
  score = model.score(X, y)
  model.analyze(X, y)
  ```

---

## 🧭 Clustering

From `clusters.py`:
KMeans · DBSCAN · OPTICS · Birch · Agglomerative · Spectral · Affinity Propagation · Mean-Shift — each supports unsupervised scoring metrics (silhouette, homogeneity, completeness, etc.) and optional visual diagnostics.

---

## 🔍 Outlier & Novelty Detection

From `outliers.py`:

* `IsolationForest`
* `OutlierFactor` (LOF)
* `OneClass`
* `EllipticSquare` (Elliptic Envelope)

Each wrapper implements `train`, `project`, `score`, and optional `analyze` with inlier/outlier visualization.

---

## ⏱️ Forecasting

From `forecasting.py`:

* `ARIMA` / `SARIMA` — classical time-series forecasting (via `statsmodels`)
* `LaggingSeries` — simple lag-based OLS forecaster
* `TimeSeriesSplitter` — rolling window cross-validation for sequential data

Example:

```python
from forecasting import LaggingSeries
ts = LaggingSeries(lag=6).train(X_lag, y_lag)
y_pred = ts.project(X_lag)
```

---

## ⚙️ Pipeline Integration Example

```python
from transformers import ColumnTransformer, TfidfVectorizer
from classifications import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(max_features=500)),
    ("model", LogisticRegression())
])

X = ["Mathy automates ML", "Mathy improves workflows"]
y = [1, 0]
pipeline.fit(X, y)
```

---

## ✅ Installation

```bash
pip install -r requirements.txt
```

Dependencies:
`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `statsmodels`, `seaborn`.

---

## 💡 Design Philosophy

* **Unified interface** — every component obeys the same verbs.
* **Interchangeable transformers** — composable into any preprocessing pipeline.
* **Robust diagnostics** — `analyze()` methods output metrics and optional plots.
* **Error transparency** — all classes use `boogr.Error` and `ErrorDialog` for standardized exception reporting.


---

## 📄 License

MIT License © Terry D. Eppler
Contact: [terryeppler@gmail.com](mailto:terryeppler@gmail.com)
