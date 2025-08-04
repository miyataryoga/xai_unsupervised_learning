import pandas as pd
import numpy as np
import shap
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def get_shap_importance(X, y_original, feature_names):
    """
    Trains an XGBoost classifier and calculates global feature importances
    using mean absolute SHAP values.

    Args:
        y_original (pd.DataFrame): The original target labels for stratification.

    Returns:
        pd.Series: Mean absolute SHAP values for each feature.
    """
    # Encode target variable for XGBoost
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_original.values.ravel())

    # Split data
    X_train, X_test, y_train_encoded, _ = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train XGBoost model
    model = xgboost.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train_encoded)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Calculate mean absolute SHAP values across all classes and instances
    mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
    
    print("SHAP importances calculated.")
    return pd.Series(mean_abs_shap, index=feature_names)