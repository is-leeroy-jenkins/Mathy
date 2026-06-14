# Feature Engineering

Mathy feature engineering workflows prepare model-ready predictor sets by selecting, reducing,
ranking, or eliminating features before regression, classification, clustering, outlier detection,
or forecasting.

This page explains how to use the feature-engineering layer after data preparation and before model
training.

## 🧭 Purpose

Feature engineering improves modeling workflows by reducing unnecessary predictors, identifying
informative columns, managing dimensionality, and preparing cleaner feature matrices.

The main feature-engineering module is:

| Module        | Purpose                                                                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `features.py` | Provides wrappers for feature selection, decomposition, canonical correlation analysis, sequential selection, and recursive elimination. |

## 🧱 Feature Engineering Position

Feature engineering sits between preprocessing and modeling.

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
Regression, Classification, Clustering, Outlier Detection, Forecasting
```

The feature-engineering layer expects prepared numeric or encoded feature matrices.

## 🧬 Core Feature Classes

Mathy includes feature wrappers for common dimensionality reduction and feature selection tasks.

| Class               | Purpose                                        |
| ------------------- | ---------------------------------------------- |
| `Selector`          | Base interface for feature-selection wrappers. |
| `VarianceThreshold` | Removes features with low variance.            |
| `CCA`               | Applies canonical correlation analysis.        |
| `PCA`               | Applies principal component analysis.          |
| `SelectBest`        | Selects the highest-scoring features.          |
| `SelectPercent`     | Selects features by percentile rank.           |
| `SBS`               | Applies sequential backward selection.         |
| `RFE`               | Applies recursive feature elimination.         |

## 📊 Prepare Input Data

Feature selectors generally require numeric model-ready arrays. Start with a `DataSource`, then
apply preprocessing as needed.

```python
import pandas as pd

from data import DataSource
from scalers import StandardScaler
from imputers import SimpleImputer

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
X_scaled = scaler.train_transform(X_imputed)
```

The resulting `X_scaled` array can be used by feature-selection and dimensionality-reduction
wrappers.

## 🔻 Remove Low-Variance Features

Use `VarianceThreshold` to remove predictors that do not vary enough to provide useful modeling
signal.

```python
from features import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)
X_selected = selector.train_transform(X_scaled)
```

Use this early in the workflow when feature columns include constants or near-constants.

## 🧮 Reduce Dimensionality with PCA

Use `PCA` to transform correlated numeric predictors into a smaller set of principal components.

```python
from features import PCA

pca = PCA(n_components=3)
X_components = pca.train_transform(X_scaled)
```

Principal component analysis is useful when:

| Condition                      | Reason                                        |
| ------------------------------ | --------------------------------------------- |
| Features are highly correlated | PCA can compress shared variance.             |
| Feature count is high          | PCA can reduce dimensionality.                |
| Model speed matters            | Fewer columns can reduce training time.       |
| Visualization is needed        | Two or three components can support plotting. |

## 🔗 Apply Canonical Correlation Analysis

Use `CCA` when the workflow requires relationships between two multivariate sets.

```python
from features import CCA

cca = CCA(n_components=2)
cca.train(X_scaled, source.targets)
```

Canonical correlation analysis is most useful when the goal is to understand shared structure
between feature blocks and target-related data.

## ⭐ Select the Best Features

Use `SelectBest` to select the top-scoring predictors.

```python
from features import SelectBest

selector = SelectBest(k=5)
X_best = selector.train_transform(X_scaled, source.targets)
```

Use this when you want a fixed number of high-scoring predictors.

## 📐 Select Features by Percentile

Use `SelectPercent` to retain a percentage of predictors based on feature scores.

```python
from features import SelectPercent

selector = SelectPercent(percentile=50)
X_percent = selector.train_transform(X_scaled, source.targets)
```

Percentile selection is useful when the number of input features changes across datasets but the
desired selection proportion stays stable.

## 🔁 Apply Sequential Backward Selection

Use `SBS` to remove features iteratively while preserving model performance.

```python
from features import SBS
from regressions import LinearRegression

model = LinearRegression()

selector = SBS(
    estimator=model.model,
    k_features=5
)

selector.train(X_scaled, source.targets)
```

Sequential backward selection is useful when you want a smaller feature set but still want the
selection process to account for estimator behavior.

## 🧹 Apply Recursive Feature Elimination

Use `RFE` to recursively train an estimator and remove less important predictors.

```python
from features import RFE
from regressions import LinearRegression

model = LinearRegression()

selector = RFE(
    estimator=model.model,
    n_features_to_select=5
)

X_rfe = selector.train_transform(X_scaled, source.targets)
```

Recursive feature elimination is useful for estimators that expose coefficients or feature
importances.

## 🧪 Example End-to-End Feature Workflow

```python
import pandas as pd

from data import DataSource
from imputers import SimpleImputer
from scalers import StandardScaler
from features import VarianceThreshold, PCA, SelectBest

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
X_scaled = scaler.train_transform(X_imputed)

variance = VarianceThreshold(threshold=0.0)
X_variance = variance.train_transform(X_scaled)

selector = SelectBest(k=5)
X_best = selector.train_transform(X_variance, source.targets)

pca = PCA(n_components=3)
X_components = pca.train_transform(X_best)
```

This workflow:

```text
1. Loads a dataframe.
2. Creates a DataSource.
3. Imputes missing numeric values.
4. Scales numeric values.
5. Removes low-variance features.
6. Selects top-scoring features.
7. Reduces the selected feature set with PCA.
```

## 🧯 Error Handling

Feature-engineering wrappers use the project exception pattern when wrapped handlers are present.

```python
except Exception as e:
    exception = Error( e )
    exception.module = 'mathy'
    exception.cause = 'ClassName'
    exception.method = 'method_name( self, *args ) -> return_type'
    Logger( ).write( exception )
    raise exception
```

The logged method string uses stable signatures and avoids live runtime data.

## ✅ Recommended Feature Engineering Sequence

For most tabular modeling workflows:

```text
1. Impute missing values.
2. Scale numeric features.
3. Encode categorical features.
4. Remove low-variance columns.
5. Select or rank predictors.
6. Reduce dimensionality if needed.
7. Train the model.
```

## 🔗 Related API Pages

| API Page                               | Description                                                            |
| -------------------------------------- | ---------------------------------------------------------------------- |
| [Features](../api/features.md)         | Feature selection, dimensionality reduction, and elimination wrappers. |
| [Data](../api/data.md)                 | `DataSource` and dataframe preparation helpers.                        |
| [Scalers](../api/scalers.md)           | Numeric scaling wrappers.                                              |
| [Encoders](../api/encoders.md)         | Encoding and polynomial feature wrappers.                              |
| [Imputers](../api/imputers.md)         | Missing-value imputation wrappers.                                     |
| [Transformers](../api/transformers.md) | Transformation and vectorization wrappers.                             |
