# Data Preparation

Mathy data preparation workflows begin with a pandas dataframe and move through source management,
column inspection, train/test splitting, scaling, encoding, imputation, and transformation.

This page explains how to prepare tabular data for downstream feature engineering, regression,
classification, clustering, outlier detection, and forecasting workflows.

## 🧭 Purpose

Data preparation establishes the working dataset used by the rest of Mathy. The preparation layer
identifies target values, feature columns, numeric columns, categorical columns, descriptive
statistics, train/test splits, and reusable transformed data.

The main preparation modules are:

| Module            | Purpose                                                                       |
| ----------------- | ----------------------------------------------------------------------------- |
| `data.py`         | Creates the `DataSource` wrapper and provides statistical helper functions.   |
| `scalers.py`      | Scales numeric features.                                                      |
| `encoders.py`     | Encodes labels, categories, targets, and polynomial features.                 |
| `imputers.py`     | Replaces missing values.                                                      |
| `transformers.py` | Applies binary, label, text, dictionary, hashing, and column transformations. |

## 🗃️ Create a DataSource

The `DataSource` class wraps a pandas dataframe and prepares it for modeling.

```python
import pandas as pd
from data import DataSource

df = pd.read_csv("data.csv")

source = DataSource(
    df=df,
    target="target",
    size=0.25,
    rando=42
)
```

The `target` argument identifies the column to predict or analyze. The `size` argument controls the
test-set proportion. The `rando` argument controls the random seed used for reproducible train/test
splitting.

## 🔎 Inspect Derived Metadata

After initialization, `DataSource` exposes metadata derived from the dataframe.

```python
source.feature_names
source.target
source.targets
source.target_names
source.numeric_columns
source.categorical_columns
source.n_samples
source.n_features
```

Common metadata includes:

| Attribute             | Description                               |
| --------------------- | ----------------------------------------- |
| `data`                | Working dataframe copy.                   |
| `target`              | Name of the selected target column.       |
| `targets`             | Target values from the working dataframe. |
| `target_names`        | Unique target values.                     |
| `feature_names`       | Non-target feature columns.               |
| `numeric_columns`     | Numeric dataframe columns.                |
| `categorical_columns` | Object or category dataframe columns.     |
| `n_samples`           | Number of rows in the source dataframe.   |
| `n_features`          | Number of non-target feature columns.     |

## ✂️ Use Train/Test Splits

`DataSource` creates train/test splits during initialization.

```python
X_train = source.X_training
X_test = source.X_testing
y_train = source.y_training
y_test = source.y_testing
```

These attributes are useful when passing prepared data into model wrappers.

| Attribute    | Description              |
| ------------ | ------------------------ |
| `X_training` | Training feature matrix. |
| `X_testing`  | Testing feature matrix.  |
| `y_training` | Training target values.  |
| `y_testing`  | Testing target values.   |

## 📊 Review Numeric Statistics

`DataSource` computes common numeric statistics during initialization.

```python
source.numeric_metrics
source.average
source.variance
source.standard_deviation
source.skew
source.kurtosis
source.covariance
source.mean_standard_error
```

These values support exploratory analysis and feature review before model training.

| Attribute             | Description                                    |
| --------------------- | ---------------------------------------------- |
| `numeric_metrics`     | Percentile and descriptive summary statistics. |
| `average`             | Numeric column means.                          |
| `variance`            | Numeric column variances.                      |
| `standard_deviation`  | Numeric column standard deviations.            |
| `skew`                | Numeric column skewness.                       |
| `kurtosis`            | Numeric column kurtosis.                       |
| `covariance`          | Numeric covariance matrix.                     |
| `mean_standard_error` | Numeric standard error of the mean.            |

## 📈 Create Exploratory Plots

`DataSource` includes basic plotting helpers.

```python
source.create_histogram()
source.create_heatmap()
```

Use `create_histogram()` to inspect numeric distribution behavior. Use `create_heatmap()` to inspect
pairwise correlations among numeric columns.

## 🧮 Create a Pivot Table

Use `create_pivot()` to create a grouped summary table from the working dataframe.

```python
pivot = source.create_pivot(
    cols=["category_column"],
    vals=["value_column"],
    idx=["group_column"]
)
```

The generated pivot table is stored on the instance.

```python
source.pivot_table
```

## 📏 Scale Numeric Features

Scaling prepares numeric features for algorithms that are sensitive to feature magnitude.

Mathy includes these scaler wrappers:

| Class            | Purpose                                                                  |
| ---------------- | ------------------------------------------------------------------------ |
| `StandardScaler` | Standardizes features by removing the mean and scaling to unit variance. |
| `MinMaxScaler`   | Scales features into a bounded minimum/maximum range.                    |
| `RobustScaler`   | Scales features using statistics that are robust to outliers.            |
| `NormalScaler`   | Normalizes individual sample vectors.                                    |
| `MaxAbsScaler`   | Scales each feature by its maximum absolute value.                       |

Example:

```python
from scalers import StandardScaler

scaler = StandardScaler()
scaled = scaler.train_transform(source.numeric_data)
```

The common scaler pattern is:

```python
scaler.train(X)
scaled = scaler.transform(X)
```

or:

```python
scaled = scaler.train_transform(X)
```

## 🧬 Encode Categorical Values

Encoding converts labels and categorical columns into numeric representations.

Mathy includes these encoder wrappers:

| Class                | Purpose                                                           |
| -------------------- | ----------------------------------------------------------------- |
| `LabelEncoder`       | Encodes one-dimensional label values.                             |
| `OrdinalEncoder`     | Encodes categorical feature values as ordinal codes.              |
| `OneHotEncoder`      | Encodes categorical values as one-hot columns.                    |
| `TargetEncoder`      | Encodes categorical features using target-conditioned statistics. |
| `PolynomialFeatures` | Expands numeric features into polynomial and interaction terms.   |

Example:

```python
from encoders import OrdinalEncoder

encoder = OrdinalEncoder()
encoded = encoder.train_transform(source.categorical_data)
```

For target labels:

```python
from encoders import LabelEncoder

labeler = LabelEncoder()
encoded_target = labeler.train_transform(source.targets)
```

## 🧼 Impute Missing Values

Imputation replaces missing values before feature engineering or modeling.

Mathy includes these imputer wrappers:

| Class              | Purpose                                                                   |
| ------------------ | ------------------------------------------------------------------------- |
| `MeanImputer`      | Replaces missing values using column means.                               |
| `NearestImputer`   | Replaces missing values using nearest-neighbor imputation.                |
| `IterativeImputer` | Replaces missing values using iterative feature modeling.                 |
| `SimpleImputer`    | Replaces missing values using configurable sklearn imputation strategies. |

Example:

```python
from imputers import SimpleImputer

imputer = SimpleImputer(strategy="mean")
imputed = imputer.train_transform(source.numeric_data)
```

Nearest-neighbor imputation:

```python
from imputers import NearestImputer

imputer = NearestImputer(neighbors=5)
imputed = imputer.train_transform(source.numeric_data)
```

## 🔄 Transform Feature Matrices

The transformation layer handles binary, label, text, dictionary, hashing, and column
transformations.

Representative transformer wrappers include:

| Class                 | Purpose                                                |
| --------------------- | ------------------------------------------------------ |
| `Binarizer`           | Converts numeric values into binary indicators.        |
| `LabelBinarizer`      | Converts labels into binary indicator matrices.        |
| `MultiLabelBinarizer` | Converts multilabel targets into indicator matrices.   |
| `TfidfTransformer`    | Converts count matrices into TF-IDF matrices.          |
| `TfidfVectorizer`     | Converts raw text into TF-IDF token-feature matrices.  |
| `CountVectorizer`     | Converts raw text into token-count matrices.           |
| `DictVectorizer`      | Converts dictionaries into numeric feature matrices.   |
| `HashVectorizer`      | Converts text into hashed feature matrices.            |
| `FeatureHasher`       | Converts feature mappings into hashed sparse matrices. |
| `ColumnTransformer`   | Applies named transformations to selected columns.     |

Example text vectorization:

```python
from transformers import TfidfVectorizer

documents = [
    "machine learning workflow",
    "statistical modeling library",
    "feature engineering and forecasting"
]

vectorizer = TfidfVectorizer()
matrix = vectorizer.train_transform(documents)
```

## 🧱 Apply Column Transformations

Column transformations allow different preprocessing steps to be applied to different column groups.

```python
from transformers import ColumnTransformer
from scalers import StandardScaler
from encoders import OneHotEncoder

transformer = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler().model, source.numeric_columns),
        ("categorical", OneHotEncoder().model, source.categorical_columns)
    ],
    remainder="drop"
)

prepared = transformer.train_transform(source.data)
```

This pattern is useful when numeric and categorical columns require different preprocessing
operations.

## 🧪 Prepare Data for Modeling

A typical preparation path is:

```python
import pandas as pd

from data import DataSource
from scalers import StandardScaler
from encoders import OrdinalEncoder
from imputers import SimpleImputer

df = pd.read_csv("data.csv")

source = DataSource(
    df=df,
    target="target",
    size=0.25,
    rando=42
)

imputer = SimpleImputer(strategy="mean")
numeric_imputed = imputer.train_transform(source.numeric_data)

scaler = StandardScaler()
numeric_scaled = scaler.train_transform(numeric_imputed)

encoder = OrdinalEncoder()
categorical_encoded = encoder.train_transform(source.categorical_data)
```

The transformed outputs can then be joined, selected, reduced, or passed into model wrappers.

## 🧯 Error Handling

Data preparation wrappers use the project logging pattern when existing exception handlers are
present.

```python
except Exception as e:
    exception = Error( e )
    exception.module = 'mathy'
    exception.cause = 'ClassName'
    exception.method = 'method_name( self, *args ) -> return_type'
    Logger( ).write( exception )
    raise exception
```

The `exception.method` field stores a stable method signature. It does not store live dataframe
values, input arrays, target values, file paths, or user-provided data.

## ✅ Recommended Sequence

Use this order for most tabular workflows:

```text
1. Load dataframe.
2. Create DataSource.
3. Inspect numeric and categorical columns.
4. Impute missing values.
5. Scale numeric features.
6. Encode categorical features.
7. Apply transformations.
8. Move to feature engineering or modeling.
```

## 🔗 Related API Pages

| API Page                               | Description                                |
| -------------------------------------- | ------------------------------------------ |
| [Data](../api/data.md)                 | `DataSource` and statistical helpers.      |
| [Scalers](../api/scalers.md)           | Numeric scaling wrappers.                  |
| [Encoders](../api/encoders.md)         | Encoding and polynomial feature wrappers.  |
| [Imputers](../api/imputers.md)         | Missing-value imputation wrappers.         |
| [Transformers](../api/transformers.md) | Transformation and vectorization wrappers. |
