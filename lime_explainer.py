import pandas as pd
from lime import lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_lime_importance(X, cluster_labels, feature_names):
    """
    Calculates global feature importances by aggregating LIME explanations
    from a RandomForest model trained on cluster labels.

    Returns:
        pd.Series: Aggregated LIME importance scores.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, cluster_labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train.values, y_train)

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=[str(i) for i in range(len(set(cluster_labels)))],
        mode='classification'
    )

    # Aggregate feature importances from local explanations
    global_feature_importances = {feature: 0.0 for feature in feature_names}
    for i in range(len(X_test)):
        instance = X_test.iloc[i].values
        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=len(feature_names)
        )
        for feature, weight in explanation.as_list():
            # Parse the feature name from the explanation string
            for f_name in feature_names:
                if f_name in feature:
                    global_feature_importances[f_name] += abs(weight)
                    break

    # Average the importances
    for feature in global_feature_importances:
        global_feature_importances[feature] /= len(X_test)
    
    print("LIME importances calculated.")
    return pd.Series(global_feature_importances)