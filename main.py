import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Import functions from other modules
from data_loader import load_data # Updated import
from fics import get_fics_scores
from models import (
    get_logistic_regression_importance,
    get_random_forest_importance,
    get_lightgbm_importance
)
from lime_explainer import get_lime_importance
from shap_explainer import get_shap_importance

# --- CHOOSE YOUR DATASET HERE ---
DATASET_NAME = 'iris'  # Options: 'iris' or 'glass'
# ------------------------------------

def main():
    """Main function to run the entire analysis pipeline."""
    # 1. Load and preprocess the chosen data
    X, y, X_scaled, feature_names = load_data(dataset_name=DATASET_NAME)

    # 2. K-Means Clustering (parameters adjusted for the dataset)
    n_clusters = 3 if DATASET_NAME == 'iris' else 2
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    print(f"Data clustered into {n_clusters} groups.")

    # 3. Calculate feature importances using all methods
    importance_df = pd.DataFrame(index=feature_names)
    importance_df['FICS_Method'] = get_fics_scores(centroids, feature_names)
    importance_df['Logistic_Regression'] = get_logistic_regression_importance(X, cluster_labels, feature_names)
    importance_df['Random_Forest'] = get_random_forest_importance(X, cluster_labels, feature_names)
    importance_df['LightGBM'] = get_lightgbm_importance(X, cluster_labels, feature_names)
    importance_df['LIME'] = get_lime_importance(X, cluster_labels, feature_names)
    importance_df['SHAP'] = get_shap_importance(X, y, feature_names)

    # 4. Display and correlate results
    print("\n--- Feature Importance Scores ---")
    print(importance_df.round(4)) # Rounded for cleaner output
    
    correlation_matrix = importance_df.corr()
    print("\n--- Correlation Matrix ---")
    print(correlation_matrix.round(4))

    # 5. Plot heatmap with a dynamic title
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Correlation of Feature Importance Metrics (Dataset: {DATASET_NAME.capitalize()}, k={n_clusters})')
    plt.savefig(f'heatmap_{DATASET_NAME}.png')
    plt.show()
    print(f"\nHeatmap saved as 'heatmap_{DATASET_NAME}.png'")

if __name__ == '__main__':
    main()