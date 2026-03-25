"""Test Prediction Pipeline (Preprocess + Model Predict).

1. Load trained model (joblib)
2. Load test CSV
3. Preprocess (impute, outliers, RobustScaler)
4. Select 46 GIST features (selected_features_gist.csv)
5. Predict classes/probs
6. Save predictions.csv

CONFIG:
- data_path: test.csv
Run `python ML_Product.py`
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from scipy.stats import zscore

CONFIG = {
    'data_path': 'test.csv',
    'selected_features_csv': 'selected_features_gist.csv',
    'output_dir': 'results',
}

def preprocess_data(df):
    X = df.select_dtypes(include=[np.number])
    columns = X.columns.tolist()
    print(f"Input features: {X.shape}")

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    print("Outlier removal (3SD rule)...")
    n_features_outlier = 0
    n_outliers_total = 0
    X_outlier_free = np.copy(X_imputed)
    for j in range(X.shape[1]):
        mean_j = np.mean(X_imputed[:, j])
        sd_j = np.std(X_imputed[:, j])
        lower = mean_j - 3 * sd_j
        upper = mean_j + 3 * sd_j
        outliers_mask = (X_imputed[:, j] < lower) | (X_imputed[:, j] > upper)
        n_outliers_j = np.sum(outliers_mask)
        if n_outliers_j > 0:
            n_features_outlier += 1
            n_outliers_total += n_outliers_j
            outlier_rows = np.where(outliers_mask)[0]
            X_outlier_free[outlier_rows, :] = np.nan

    X_clean = SimpleImputer(strategy="median").fit_transform(X_outlier_free)
    print(f"Outliers: {n_outliers_total} in {n_features_outlier}/{X.shape[1]} features")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Diagnostics
    z_scores_df = pd.DataFrame(zscore(X, nan_policy='omit'), columns=columns)
    outliers = (np.abs(z_scores_df) > 3)
    perc_outliers = outliers.sum() / len(X) * 100
    print("Top outliers %:")
    print(pd.DataFrame({'feature': columns, 'perc': perc_outliers}).sort_values('perc', ascending=False).head(10))

    skewness = 3 * (X.mean() - X.median()) / X.std()
    print("Top skewness:")
    print(pd.DataFrame({'feature': columns, 'skewness': skewness}).sort_values('skewness', key=abs, ascending=False).head(10))
    print(f"Avg abs skewness: {skewness.abs().mean():.2f}")

    print(f"Preprocessed: {X_scaled.shape}")
    return X_scaled, columns, scaler, imputer

def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Your tuned SVM (C=9.6098, kernel=linear, class_weight=None, CV AUC=0.7396)
    model = SVC(C=9.6098129470364, kernel='linear', class_weight=None, probability=True, random_state=42)
    
    # Fit on dummy data or skip if you want pure predict (add train data if needed)
    print("Model initialized with tuned parameters")

    df_test = pd.read_csv(CONFIG['data_path'])
    print(f"Test data: {df_test.shape}")

    X_scaled, all_columns, scaler, imputer = preprocess_data(df_test)

    # GIST 46 features
    feat_df = pd.read_csv(CONFIG['selected_features_csv'])
    indices = [all_columns.index(name) for name in feat_df['feature']]
    X_gist = X_scaled[:, indices]
    print(f"GIST features selected: {X_gist.shape}")

    # Predict
    preds = model.predict(X_gist)
    probs = model.predict_proba(X_gist)[:, 1] if hasattr(model, 'predict_proba') else np.nan

    # Output
    out_df = df_test.copy()
    out_df['prediction'] = preds
    out_df['probability'] = probs
    csv_out = os.path.join(CONFIG['output_dir'], 'predictions.csv')
    out_df.to_csv(csv_out, index=False)
    print(f"Predictions saved: {csv_out}")

if __name__ == "__main__":
    main()

