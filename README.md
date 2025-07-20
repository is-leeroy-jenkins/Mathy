# 🧠 Mathy-Py

**Modular Machine Learning Framework for Data Scientists, Analysts, and Researchers**  
Created by [Terry D. Eppler](mailto:terryeppler@gmail.com)


## 📦 Overview

**Mathy** is a powerful and extensible Python library designed to streamline machine learning workflows. 
It provides a clean abstraction over clustering algorithms, preprocessing tools, 
and classification/regression models with custom wrappers, error handling, and diagnostic tools. 
Mathy targets professionals in data science and analytics who require reusable, robust, and 
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


