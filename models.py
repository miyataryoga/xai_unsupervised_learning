import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_logistic_regression_importance(X, cluster_labels, feature_names):
    """
    Trains a Logistic Regression model on cluster labels and returns feature importance.

    Returns:
        pd.Series: Feature importances from the model.
    """
    logreg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
    logreg.fit(X, cluster_labels)
    importances = np.mean(np.abs(logreg.coef_), axis=0)
    print("Logistic Regression importances calculated.")
    return pd.Series(importances, index=feature_names)

def get_random_forest_importance(X, cluster_labels, feature_names):
    """
    Trains a RandomForest Classifier on cluster labels and returns feature importance.

    Returns:
        pd.Series: Feature importances from the model.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, cluster_labels)
    print("Random Forest importances calculated.")
    return pd.Series(rf.feature_importances_, index=feature_names)

def get_lightgbm_importance(X, cluster_labels, feature_names):
    """
    Trains a LightGBM Classifier on cluster labels and returns feature importance.

    Returns:
        pd.Series: Feature importances from the model.
    """
    lgbm = lgb.LGBMClassifier(objective='multiclass', num_class=len(np.unique(cluster_labels)), random_state=42)
    lgbm.fit(X, cluster_labels)
    print("LightGBM importances calculated.")
    return pd.Series(lgbm.feature_importances_, index=feature_names)