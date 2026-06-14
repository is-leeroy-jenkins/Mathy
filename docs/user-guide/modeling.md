# Modeling

Mathy modeling workflows use prepared feature matrices and target values to train regression,
classification, clustering, and outlier-detection models through consistent wrapper classes.

This page explains how the modeling modules fit into the broader workflow after data preparation and
feature engineering.

## 🧭 Purpose

The modeling layer provides reusable wrappers around supervised and unsupervised machine learning
estimators. These wrappers expose consistent method names for training, prediction, scoring, and
inspection while preserving access to the underlying estimator behavior.

The main modeling modules are:

| Module               | Purpose                                                            |
| -------------------- | ------------------------------------------------------------------ |
| `regressions.py`     | Trains supervised models that predict continuous numeric outcomes. |
| `classifications.py` | Trains supervised models that predict categorical labels.          |
| `clusters.py`        | Groups observations without a supervised target label.             |
| `outliers.py`        | Identifies anomalous observations or unusual records.              |

## 🧱 Modeling Position

Modeling occurs after data preparation and, when needed, feature engineering.

```text
Raw Data
   |
   v
DataSource
   |
   v
Scaling, Encoding, Imputation, Transformation
   |
   v
Feature Engineering
   |
   v
Modeling
   |
   +--> Regression
   +--> Classification
   +--> Clustering
   +--> Outlier Detection
   |
   v
Predictions, Scores, Labels, Outliers, Diagnostics
```

## 🗃️ Prepare Data for Modeling

Start by creating a `DataSource` and preparing the feature matrix.

```python
import pandas as pd

from data import DataSource
from imputers import SimpleImputer
from scalers import StandardScaler

df = pd.read_csv("data.csv")

source = DataSource(
    df=df,
    target="target",
    size=0.25,
    rando=42
)

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.train_transform(source.numeric_data)

scaler = StandardScaler()
X_prepared = scaler.train_transform(X_imputed)

y = source.targets
```

Use `X_prepared` and `y` with regression and classification wrappers. Use `X_prepared` alone with
clustering and outlier-detection wrappers.

## 📈 Regression Models

Regression models predict continuous numeric values.

Use regression wrappers when the target column represents a measurable quantity, amount, score,
rate, index, or other numeric outcome.

```python
from regressions import LinearRegression

model = LinearRegression()
model.train(X_prepared, y)

predictions = model.predict(X_prepared)
score = model.score(X_prepared, y)
```

Common regression outputs include:

| Output              | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| Predictions         | Continuous numeric model estimates.                              |
| Scores              | Estimator-specific performance scores.                           |
| Coefficients        | Linear-model weights where supported.                            |
| Feature importances | Tree-based or ensemble feature influence values where supported. |
| Diagnostics         | Model-specific fitted metadata or evaluation details.            |

## 📊 Regression Workflow

A typical regression workflow is:

```python
import pandas as pd

from data import DataSource
from imputers import SimpleImputer
from scalers import StandardScaler
from regressions import LinearRegression

df = pd.read_csv("data.csv")

source = DataSource(
    df=df,
    target="target",
    size=0.25,
    rando=42
)

imputer = SimpleImputer(strategy="mean")
X_train = imputer.train_transform(source.X_training[source.numeric_columns])
X_test = imputer.transform(source.X_testing[source.numeric_columns])

scaler = StandardScaler()
X_train = scaler.train_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.train(X_train, source.y_training)

predictions = model.predict(X_test)
score = model.score(X_test, source.y_testing)
```

## 🧮 Regression Model Families

The regression module supports multiple estimator families.

| Family                  | Purpose                                                                       |
| ----------------------- | ----------------------------------------------------------------------------- |
| Linear models           | Estimate continuous outcomes from weighted feature combinations.              |
| Regularized models      | Control model complexity with penalties such as ridge, lasso, or elastic net. |
| Tree-based models       | Learn nonlinear splits and feature interactions.                              |
| Ensemble models         | Combine multiple estimators for stronger predictive performance.              |
| Support-vector models   | Fit margin-based regression estimators.                                       |
| Nearest-neighbor models | Predict from nearby training observations.                                    |
| Bayesian models         | Estimate parameters with probabilistic assumptions.                           |
| Boosting models         | Build additive ensembles from sequential learners.                            |

## 🏷️ Classification Models

Classification models predict categorical labels.

Use classification wrappers when the target column represents classes, categories, groups, outcomes,
statuses, or labels.

```python
from classifications import LogisticRegression

model = LogisticRegression()
model.train(X_prepared, y)

predictions = model.predict(X_prepared)
score = model.score(X_prepared, y)
```

Common classification outputs include:

| Output           | Description                                   |
| ---------------- | --------------------------------------------- |
| Predicted labels | Class predictions for each observation.       |
| Scores           | Estimator-specific performance scores.        |
| Probabilities    | Class probabilities where supported.          |
| Decision values  | Margins or confidence scores where supported. |
| Classes          | Fitted class labels where supported.          |

## 🧪 Classification Workflow

A typical classification workflow is:

```python
import pandas as pd

from data import DataSource
from encoders import LabelEncoder
from imputers import SimpleImputer
from scalers import StandardScaler
from classifications import LogisticRegression

df = pd.read_csv("data.csv")

source = DataSource(
    df=df,
    target="class_label",
    size=0.25,
    rando=42
)

labeler = LabelEncoder()
y_train = labeler.train_transform(source.y_training)
y_test = labeler.transform(source.y_testing)

imputer = SimpleImputer(strategy="mean")
X_train = imputer.train_transform(source.X_training[source.numeric_columns])
X_test = imputer.transform(source.X_testing[source.numeric_columns])

scaler = StandardScaler()
X_train = scaler.train_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.train(X_train, y_train)

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
```

## 🧠 Classification Model Families

The classification module supports multiple estimator families.

| Family                       | Purpose                                                              |
| ---------------------------- | -------------------------------------------------------------------- |
| Linear classifiers           | Separate classes with linear decision boundaries.                    |
| Probabilistic classifiers    | Estimate classes using probability models.                           |
| Tree-based classifiers       | Learn nonlinear rules and feature interactions.                      |
| Ensemble classifiers         | Combine multiple estimators for improved classification performance. |
| Support-vector classifiers   | Fit margin-based classification models.                              |
| Nearest-neighbor classifiers | Classify observations from nearby training examples.                 |
| Discriminant classifiers     | Model class separation using distributional assumptions.             |
| Boosting classifiers         | Build sequential ensembles from weak learners.                       |

## 🧩 Clustering Models

Clustering models group observations without a supervised target label.

Use clustering wrappers when the goal is segmentation, grouping, structure discovery, or
unsupervised exploratory analysis.

```python
from clusters import KMeans

clusterer = KMeans(n_clusters=3)
clusterer.train(X_prepared)

labels = clusterer.predict(X_prepared)
```

Common clustering outputs include:

| Output          | Description                                        |
| --------------- | -------------------------------------------------- |
| Cluster labels  | Assigned group for each observation.               |
| Cluster centers | Centroid coordinates where supported.              |
| Fitted model    | Underlying estimator state.                        |
| Diagnostics     | Algorithm-specific metadata or clustering metrics. |

## 🧭 Clustering Workflow

A typical clustering workflow is:

```python
import pandas as pd

from data import DataSource
from imputers import SimpleImputer
from scalers import StandardScaler
from clusters import KMeans

df = pd.read_csv("data.csv")

source = DataSource(
    df=df,
    target="reference_label",
    size=0.25,
    rando=42
)

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.train_transform(source.numeric_data)

scaler = StandardScaler()
X_prepared = scaler.train_transform(X_imputed)

clusterer = KMeans(n_clusters=3)
clusterer.train(X_prepared)

labels = clusterer.predict(X_prepared)
```

The `target` in a clustering workflow can still be useful for dataframe management or later
comparison, but the clustering estimator itself does not require a supervised target.

## 🧬 Clustering Model Families

The clustering module supports several unsupervised approaches.

| Class                 | Purpose                                                         |
| --------------------- | --------------------------------------------------------------- |
| `KMeans`              | Groups observations around learned centroids.                   |
| `DBSCAN`              | Finds dense regions and marks noise points.                     |
| `Agglomerative`       | Builds hierarchical clusters.                                   |
| `Spectral`            | Uses graph-based structure for clustering.                      |
| `MeanShift`           | Finds modes in the feature space.                               |
| `AffinityPropagation` | Identifies exemplars and assigns observations to them.          |
| `Birch`               | Builds compact clustering structures for large datasets.        |
| `OPTICS`              | Detects density-based structure across variable density levels. |

## ⚠️ Outlier Detection

Outlier-detection models identify unusual, anomalous, or low-density observations.

Use outlier wrappers when the goal is to flag records that differ substantially from the rest of the
dataset.

```python
from outliers import IsolationForest

detector = IsolationForest()
detector.train(X_prepared)

labels = detector.predict(X_prepared)
```

Common outlier outputs include:

| Output          | Description                                                     |
| --------------- | --------------------------------------------------------------- |
| Outlier labels  | Indicator values identifying normal and anomalous observations. |
| Anomaly scores  | Relative abnormality or decision scores where supported.        |
| Fitted detector | Underlying estimator state.                                     |
| Diagnostics     | Estimator-specific fitted metadata.                             |

## 🔎 Outlier Detection Workflow

A typical outlier-detection workflow is:

```python
import pandas as pd

from data import DataSource
from imputers import SimpleImputer
from scalers import StandardScaler
from outliers import IsolationForest

df = pd.read_csv("data.csv")

source = DataSource(
    df=df,
    target="reference_label",
    size=0.25,
    rando=42
)

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.train_transform(source.numeric_data)

scaler = StandardScaler()
X_prepared = scaler.train_transform(X_imputed)

detector = IsolationForest()
detector.train(X_prepared)

outlier_labels = detector.predict(X_prepared)
```

## 🧯 Error Handling

Model wrappers use the project exception pattern when wrapped exception handlers are present.

```python
except Exception as e:
    exception = Error( e )
    exception.module = 'mathy'
    exception.cause = 'ClassName'
    exception.method = 'method_name( self, *args ) -> return_type'
    Logger( ).write( exception )
    raise exception
```

For multi-parameter methods, the logged method string uses `*args` to avoid storing live values.

## ✅ Modeling Selection Guide

Use this table to choose the appropriate modeling module.

| Goal                       | Module               | Example Output                     |
| -------------------------- | -------------------- | ---------------------------------- |
| Predict a continuous value | `regressions.py`     | Numeric predictions and scores.    |
| Predict a class label      | `classifications.py` | Labels, probabilities, and scores. |
| Group similar observations | `clusters.py`        | Cluster labels.                    |
| Detect anomalous records   | `outliers.py`        | Outlier labels and anomaly scores. |

## ✅ Recommended Modeling Sequence

For most modeling workflows:

```text
1. Create a DataSource.
2. Impute missing values.
3. Scale numeric values.
4. Encode categorical values where needed.
5. Select or reduce features where needed.
6. Choose a modeling family.
7. Train the model.
8. Generate predictions, labels, or scores.
9. Review diagnostics.
10. Check the API reference for class-specific methods.
```

## 🔗 Related API Pages

| API Page                                     | Description                                              |
| -------------------------------------------- | -------------------------------------------------------- |
| [Regressions](../api/regressions.md)         | Supervised regression wrappers.                          |
| [Classifications](../api/classifications.md) | Supervised classification wrappers.                      |
| [Clusters](../api/clusters.md)               | Unsupervised clustering wrappers.                        |
| [Outliers](../api/outliers.md)               | Outlier-detection wrappers.                              |
| [Data](../api/data.md)                       | `DataSource` and dataframe preparation helpers.          |
| [Features](../api/features.md)               | Feature selection and dimensionality reduction wrappers. |
