# Forecasting

Mathy forecasting workflows support ordered time-series data, temporal train/test splitting, lagged
feature construction, lag-based supervised forecasting, ARIMA models, and SARIMA models.

This page explains how to use the forecasting layer when observations must be modeled in time order.

## 🧭 Purpose

Forecasting differs from ordinary regression because observations are ordered by time. Training,
validation, lag construction, and prediction must preserve that temporal order.

The main forecasting module is:

| Module           | Purpose                                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `forecasting.py` | Provides time-series splitters, expanding-window workflows, lagged feature builders, lag-based estimators, ARIMA wrappers, and SARIMA wrappers. |

## 🧱 Forecasting Position

Forecasting can branch directly from prepared time-series data or from feature-engineered
time-series inputs.

```text
Raw Time-Series Data
   |
   v
DataSource
   |
   +--> time-series split
   +--> expanding-window split
   +--> lagged feature construction
   |
   v
Forecasting Models
   |
   +--> lag-based models
   +--> ARIMA
   +--> SARIMA
   |
   v
Forecasts, fitted values, residuals, diagnostics
```

## 🕒 Forecasting Requirements

Time-series workflows require ordered observations.

Before using forecasting classes, confirm that the source data has:

| Requirement            | Description                                                               |
| ---------------------- | ------------------------------------------------------------------------- |
| Ordered observations   | Rows should be sorted by time.                                            |
| Stable frequency       | Observations should represent a consistent period where possible.         |
| Numeric target series  | The forecast target should be numeric.                                    |
| Minimal missing values | Missing time periods or missing values should be handled before modeling. |
| Temporal validation    | Training and testing should preserve chronological order.                 |

## 🗃️ Prepare Time-Series Data

Start with a pandas dataframe that contains a time column and a numeric target column.

```python
import pandas as pd

df = pd.read_csv("time_series.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
```

Set the date column as the index when the model or analysis requires a datetime index.

```python
df = df.set_index("date")
```

Create a target series.

```python
series = df["target"]
```

## ✂️ Time-Series Splitting

Time-series validation should not randomly shuffle observations. Older records should train the
model, and later records should test it.

```python
from forecasting import TimeSeriesSpliter

splitter = TimeSeriesSpliter(
    n_splits=5
)

splits = splitter.split(series)
```

Use time-series splits when evaluating model performance across multiple ordered train/test windows.

## 📈 Expanding-Window Validation

Expanding-window validation starts with an initial training window and grows the training set over
successive splits.

```python
from forecasting import ExpandingWindow

window = ExpandingWindow(
    n_splits=5
)

splits = window.split(series)
```

This pattern is useful when a forecasting process should learn from all prior observations before
testing on later observations.

## 🔁 Create Lagged Features

Lagged features convert previous observations into predictors for future values.

```python
from forecasting import LaggingSeries

lagger = LaggingSeries(
    lags=3
)

lagged = lagger.transform(series)
```

For a series like this:

```text
t      y
1      100
2      110
3      108
4      115
```

A lagged feature matrix can represent:

```text
target   lag_1   lag_2   lag_3
115      108     110     100
```

Lagged features allow supervised learning estimators to forecast future values from prior values.

## 🧪 Lag-Based Forecasting

Lag-based forecasting uses previous observations as predictors in a supervised model.

```python
from forecasting import LagBoostingSeries

model = LagBoostingSeries(
    lags=5
)

model.train(series)

forecast = model.predict(series)
```

Lag-based models are useful when:

| Condition                                         | Reason                                                             |
| ------------------------------------------------- | ------------------------------------------------------------------ |
| The series has autoregressive behavior            | Prior observations help predict future observations.               |
| Nonlinear patterns exist                          | Tree-based or boosting estimators can learn nonlinear lag effects. |
| Traditional ARIMA assumptions are too restrictive | Supervised models can capture broader predictor behavior.          |
| Additional engineered features are available      | Lagged data can be combined with external predictors.              |

## 📊 Quantile Forecasting

Quantile forecasting estimates conditional forecast intervals or distribution-aware outputs instead
of only point forecasts.

```python
from forecasting import LagQuantileSeries

model = LagQuantileSeries(
    lags=5
)

model.train(series)

forecast = model.predict(series)
```

Quantile forecasting is useful when uncertainty ranges matter.

Common outputs may represent:

| Output          | Purpose                            |
| --------------- | ---------------------------------- |
| Lower quantile  | Conservative lower forecast bound. |
| Median quantile | Central forecast estimate.         |
| Upper quantile  | Conservative upper forecast bound. |

## 🧮 ARIMA Modeling

ARIMA models use autoregressive terms, differencing, and moving-average terms to model time-series
behavior.

```python
from forecasting import ARIMA

model = ARIMA(
    order=(1, 1, 1)
)

model.train(series)

forecast = model.predict(steps=12)
```

The ARIMA order is usually expressed as:

```text
(p, d, q)
```

| Term | Meaning               |
| ---- | --------------------- |
| `p`  | Autoregressive order. |
| `d`  | Differencing order.   |
| `q`  | Moving-average order. |

Use ARIMA when the series has autoregressive or moving-average structure and seasonality is not the
primary modeling concern.

## 🌊 SARIMA Modeling

SARIMA extends ARIMA with seasonal terms.

```python
from forecasting import SARIMA

model = SARIMA(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

model.train(series)

forecast = model.predict(steps=12)
```

The seasonal order is usually expressed as:

```text
(P, D, Q, s)
```

| Term | Meaning                        |
| ---- | ------------------------------ |
| `P`  | Seasonal autoregressive order. |
| `D`  | Seasonal differencing order.   |
| `Q`  | Seasonal moving-average order. |
| `s`  | Seasonal period length.        |

Use SARIMA when the series contains recurring seasonal patterns such as monthly, quarterly, weekly,
or annual behavior.

## 🧪 End-to-End Forecasting Workflow

```python
import pandas as pd

from forecasting import LaggingSeries, LagBoostingSeries, ARIMA

df = pd.read_csv("time_series.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.set_index("date")

series = df["target"]

lagger = LaggingSeries(lags=5)
lagged = lagger.transform(series)

lag_model = LagBoostingSeries(lags=5)
lag_model.train(series)

lag_forecast = lag_model.predict(series)

arima = ARIMA(order=(1, 1, 1))
arima.train(series)

arima_forecast = arima.predict(steps=12)
```

This workflow:

```text
1. Loads time-series data.
2. Converts the date column to datetime.
3. Sorts observations by time.
4. Sets the time column as the index.
5. Selects the numeric target series.
6. Creates lagged features.
7. Trains a lag-based forecasting model.
8. Trains an ARIMA model.
9. Generates forecasts.
```

## ✅ Forecasting Model Selection Guide

| Goal                                              | Recommended Class   |
| ------------------------------------------------- | ------------------- |
| Create chronological train/test splits            | `TimeSeriesSpliter` |
| Use expanding training windows                    | `ExpandingWindow`   |
| Build lagged predictors                           | `LaggingSeries`     |
| Train lag-based boosted forecasts                 | `LagBoostingSeries` |
| Train quantile forecasts                          | `LagQuantileSeries` |
| Model autoregressive and moving-average structure | `ARIMA`             |
| Model seasonal time-series behavior               | `SARIMA`            |

## 🧯 Error Handling

Forecasting wrappers use the project exception pattern when wrapped exception handlers are present.

```python
except Exception as e:
    exception = Error( e )
    exception.module = 'mathy'
    exception.cause = 'ClassName'
    exception.method = 'method_name( self, *args ) -> return_type'
    Logger( ).write( exception )
    raise exception
```

For methods with multiple parameters, the logged method string uses the compact `*args` form to
avoid storing live runtime values.

## ✅ Recommended Forecasting Sequence

Use this sequence for most forecasting workflows:

```text
1. Load time-series data.
2. Convert the time column to datetime.
3. Sort observations by time.
4. Handle missing observations or missing values.
5. Create a target series.
6. Choose a temporal validation strategy.
7. Build lagged features where needed.
8. Train a lag-based, ARIMA, or SARIMA model.
9. Generate forecasts.
10. Review diagnostics and forecast behavior.
```

## 🔗 Related API Pages

| API Page                             | Description                                                                      |
| ------------------------------------ | -------------------------------------------------------------------------------- |
| [Forecasting](../api/forecasting.md) | Time-series splitters, lagged feature wrappers, ARIMA, and SARIMA classes.       |
| [Data](../api/data.md)               | `DataSource` and dataframe preparation helpers.                                  |
| [Regressions](../api/regressions.md) | Supervised regression wrappers that may support lag-based forecasting workflows. |
| [Features](../api/features.md)       | Feature selection and dimensionality reduction wrappers.                         |
