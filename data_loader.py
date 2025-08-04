import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def _load_and_preprocess_glass_data():
    """Fetches and preprocesses the Glass Identification dataset."""
    glass_identification = fetch_ucirepo(id=42)
    X = glass_identification.data.features
    y = glass_identification.data.targets
    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Glass dataset loaded and preprocessed.")
    return X, y, X_scaled, feature_names

def _load_and_preprocess_iris_data():
    """Loads and preprocesses the Iris dataset."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target, columns=['target'])
    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Iris dataset loaded and preprocessed.")
    return X, y, X_scaled, feature_names

def load_data(dataset_name='glass'):
    """
    Loads and preprocesses a specified dataset.

    Args:
        dataset_name (str): The name of the dataset to load ('glass' or 'iris').

    Returns:
        tuple: A tuple containing X, y, X_scaled, and feature_names.
    """
    if dataset_name.lower() == 'iris':
        return _load_and_preprocess_iris_data()
    elif dataset_name.lower() == 'glass':
        return _load_and_preprocess_glass_data()
    else:
        raise ValueError("Invalid dataset name. Choose 'glass' or 'iris'.")