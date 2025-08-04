# MSc Project: Explainable AI for Unsupervised Learning

This project breaks down the analysis of my MSc Project. It compares a custom feature importance method for clustering (FICS) with traditional and model-agnostic methods (LIME, SHAP).

## File Structure

- `main.py`: The main script to run the entire analysis pipeline.
- `data_loader.py`: Handles fetching and preprocessing the dataset.
- `fics.py`: Contains the custom FICS (Feature Importance for Clustering based on Separation) method.
- `models.py`: Contains functions to get feature importances from supervised models (Logistic Regression, RF, LightGBM).
- `lime_explainer.py`: Calculates global feature importance using LIME.
- `shap_explainer.py`: Calculates global feature importance using SHAP.
- `requirements.txt`: A list of required Python packages.

## Setup and Execution

1.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the main analysis script:**
    ```bash
    python main.py
    ```


This will execute the full pipeline, print the resulting DataFrames to the console, and save the final correlation heatmap as `feature_importance_correlation_heatmap.png`.
