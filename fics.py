import pandas as pd

def get_fics_scores(centroids, feature_names):
    """
    Calculates feature importance for clustering based on the average squared
    distance between centroid coordinates for each feature.

    Args:
        centroids (np.ndarray): The cluster centroids from a clustering algorithm.
        feature_names (list): The names of the features.

    Returns:
        pd.Series: A pandas Series with feature names as the index and their
                   importance scores as values.
    """
    feature_avg_diffs = {}

    # Iterate through each feature
    for i in range(centroids.shape[1]):
        feature_name = feature_names[i]
        total_diff = 0
        count = 0
        # Iterate through all unique pairs of centroids
        for j in range(centroids.shape[0]):
            for k in range(j + 1, centroids.shape[0]):
                diff_sq = (centroids[j][i] - centroids[k][i])**2
                total_diff += diff_sq
                count += 1

        avg_diff = total_diff / count if count > 0 else 0
        feature_avg_diffs[feature_name] = avg_diff

    # Convert to a pandas Series for consistency
    fics_series = pd.Series(feature_avg_diffs)
    print("FICS scores calculated.")
    return fics_series