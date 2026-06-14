![](./img/Mathy.png)
___
Mathy is a Python utility library for statistical analysis, preprocessing, feature engineering,
machine learning, clustering, outlier detection, regression modeling, classification modeling, and
time-series forecasting.

The library wraps common NumPy, pandas, SciPy, statsmodels, and scikit-learn workflows behind
consistent Python classes that can be reused across exploratory analysis, model development, and
documentation-driven projects.

## Purpose

Mathy provides a structured modeling toolkit for tabular and time-series workflows. It includes
helper functions and wrapper classes for preparing data, transforming features, selecting
predictors, identifying clusters and outliers, training supervised models, and building forecasting
models.

The source modules are documented with Google-style docstrings so MkDocs and mkdocstrings can
generate browsable API documentation directly from the Python source code.

## Core capabilities

| Area                | Modules                                                                  | Purpose                                                                                                                            |
|---------------------|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Data preparation    | `data.py`, `scalers.py`, `encoders.py`, `imputers.py`, `transformers.py` | Prepare tabular data, scale numeric features, encode categorical variables, impute missing values, and transform feature matrices. |
| Feature engineering | `features.py`                                                            | Select, reduce, and rank features using statistical and model-driven methods.                                                      |
| Clustering          | `clusters.py`                                                            | Apply unsupervised clustering algorithms through consistent wrapper classes.                                                       |
| Outlier detection   | `outliers.py`                                                            | Detect anomalous observations with isolation, density, local-neighbor, and covariance-based methods.                               |
| Regression          | `regressions.py`                                                         | Train and evaluate supervised regression estimators.                                                                               |
| Classification      | `classifications.py`                                                     | Train and evaluate supervised classification estimators.                                                                           |
| Forecasting         | `forecasting.py`                                                         | Build time-series splits, lagged features, ARIMA models, SARIMA models, and lag-based forecasting estimators.                      |

## Documentation layout

The documentation is organized into three major sections:

1. **Architecture** explains how the modules fit together.
2. **User Guide** provides task-oriented usage examples.
3. **API Reference** is generated from the Google-style documentation comments in the source files.

## Build the documentation

Install the documentation dependencies:

```powershell
pip install -r requirements.txt