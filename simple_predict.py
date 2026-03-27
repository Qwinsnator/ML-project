"""Simple SVM predict on GIST_Test w/ your RFECV indices.

EDIT model params. No train needed - assumes you accept default untrained predict (for demo).
Saves predictions.csv (0/1 class).
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from scipy.stats import zscore

def preprocessing(X):
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    print("Outlier removal...")
    n_features_outlier = 0
    n_outliers_total = 0
    X_out = np.copy(X_imputed)
    for j in range(X.shape[1]):
        m = np.mean(X_imputed[:, j])
        s = np.std(X_imputed[:, j])
        mask = (X_imputed[:, j] < m - 3*s) | (X_imputed[:, j] > m + 3*s)
        n_outliers_j = mask.sum()
        if n_outliers_j > 0:
            n_features_outlier += 1
            n_outliers_total += n_outliers_j
            X_out[mask, :] = np.nan
    X_clean = SimpleImputer(strategy="median").fit_transform(X_out)
    print(f"Outliers: {n_outliers_total}")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_clean)
    # Diagnostics
    z = pd.DataFrame(zscore(X, nan_policy='omit'))
    perc_out = (np.abs(z) > 3).sum() / len(X) * 100
    print("Outliers % top10:")
    print(pd.DataFrame({'perc': perc_out}).sort_values('perc', ascending=False).head(10))
    skew = 3 * (X.mean() - X.median()) / X.std()
    print("Skewness top10:")
    print(pd.DataFrame({'skew': skew}).sort_values('skew', ascending=False, key=abs).head(10))
    return X_scaled

df_test = pd.read_csv('GIST_Test.csv')
X_test_raw = df_test.select_dtypes(np.number)
X_test = preprocessing(X_test_raw)

# Load indices
feat_df = pd.read_csv('results_grid/selected_features_indices.csv')
indices = feat_df['index'].astype(int)
X_test_sel = X_test[:, indices]
print(f"Selected: {X_test_sel.shape}")

# Fit with train data for real predictions (EDIT path/params)
df_train = pd.read_csv('GIST_Train.csv')
y_train = df_train['label'].map({'GIST':1, 'non-GIST':0}).values
X_train_raw = df_train.select_dtypes(np.number)
X_train = preprocessing(X_train_raw)
X_train_sel = X_train[:, indices]
model = SVC(C=0.1, kernel='linear', probability=True, random_state=42)
model.fit(X_train_sel, y_train)  # FIX: fit here
pred = model.predict(X_test_sel)
prob = model.predict_proba(X_test_sel)[:,1]
y_test = df_test['label'].map({'GIST':1, 'non-GIST':0}).values
mis = (pred != y_test).sum()
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
auc = roc_auc_score(y_test, prob)
spec = recall_score(y_test, pred, pos_label=0)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)
rec = recall_score(y_test, pred)

out = pd.DataFrame({'prediction': pred, 'prob': prob, 'true': y_test})
out.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
print(f"Misclassifications: {mis}")
print(f"Accuracy: {acc:.3f}")
print(f"AUC: {auc:.3f}")
print(f"Recall GIST: {rec:.3f}")
print(f"Specificity: {spec:.3f}")
print(f"F1: {f1:.3f}")

