"""GIST GridSearchCV + Full Metrics.

Preprocess/RFECV/GridSearchCV + CV metrics (AUC, Accuracy, Recall, F1, Specificity) printed + CSV.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, make_scorer
from scipy.stats import zscore, loguniform

RANDOM_STATE = 42

def load_gist_train_data():
    df = pd.read_csv("GIST_Train.csv")
    print(f"Data: {df.shape}, Classes: {df['label'].value_counts()}")
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number])
    y = df["label"].map({"GIST": 1, "non-GIST": 0}).astype(int).values
    return X, y

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

def rfecv_feature_select(X, y, output_dir):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    svc = SVC(kernel='linear')
    rfecv = RFECV(svc, step=1, cv=StratifiedKFold(5), scoring='roc_auc')
    rfecv.fit(X_s, y)
    X_sel = rfecv.transform(X_s)
    print(f"Features: {X.shape[1]} -> {X_sel.shape[1]}")
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score'])+1), rfecv.cv_results_['mean_test_score'])
    plt.title('RFECV')
    plt.savefig(os.path.join(output_dir, 'rfecv.png'))
    plt.close()
    return X_sel
#%% Gridsearch tuning metrics
def grid_tune_metrics(X_sel, y, cv):
    param_distributions = {
        "LogisticRegression": [
            {
                "penalty": ["l1", "l2"],
                "C": [0.001, 0.01, 0.1],
                "solver": ["liblinear"],
                "class_weight": [None, "balanced"],
            }
        ],
        "SVM": [
            {
                "kernel": ["linear"],
                "C": [0.001, 0.01, 0.1, 1],
                "class_weight": [None],
            },
            {
                "kernel": ["rbf"],
                "C": [0.01, 0.1, 1, 10],
                "gamma": [0.001, 0.01, 0.1],
                "class_weight": [None],
            },
        ],
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        "DecisionTree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [5, 10, 15, 20, 30],
            "min_samples_split": [2, 3, 4, 5, 10],
            "min_samples_leaf": [1, 2, 5, 7],
            "splitter": ["random"],   # FIXED      # random introduces randomness in splits, can help generalization
            "ccp_alpha": [0.0005, 0.001, 0.005, 0.01],
            "class_weight": [None],
        },
        "RandomForest": {
            "n_estimators": [200, 250, 300, 350],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4], # higher means underfitting
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
            "class_weight": [None],
        },
    }
    clfs = {
        'LogisticRegression': LogisticRegression(max_iter=2000, solver='liblinear', random_state=RANDOM_STATE),
        'SVM': SVC(random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
    }
    results = []
    tuned = {}
    print("\n=== GRIDSEARCH + METRICS ===")
    spec_scorer = make_scorer(recall_score, pos_label=0)
    for name, clf_base in clfs.items():
        print(f"{name}...")
        search = GridSearchCV(clf_base, param_distributions[name], scoring='roc_auc', n_jobs=-1, cv=cv)
        search.fit(X_sel, y)
        best_clf = search.best_estimator_
        tuned[name] = best_clf
        # Metrics
        acc = cross_val_score(best_clf, X_sel, y, cv=cv, scoring='accuracy')
        recall = cross_val_score(best_clf, X_sel, y, cv=cv, scoring='recall')
        auc_ = cross_val_score(best_clf, X_sel, y, cv=cv, scoring='roc_auc')
        f1 = cross_val_score(best_clf, X_sel, y, cv=cv, scoring='f1')
        spec = cross_val_score(best_clf, X_sel, y, cv=cv, scoring=spec_scorer)
        row = {
            'classifier': name,
            'tune_auc': search.best_score_,
            'auc_mean': auc_.mean(), 'auc_std': auc_.std(),
            'acc_mean': acc.mean(), 'acc_std': acc.std(),
            'recall_mean': recall.mean(), 'recall_std': recall.std(),
            'f1_mean': f1.mean(), 'f1_std': f1.std(),
            'spec_mean': spec.mean(), 'spec_std': spec.std(),
            'best_params': str(search.best_params_)
        }
        results.append(row)
        print(f"  Best params: {search.best_params_}")
        print(f"  AUC CV: {auc_.mean():.4f} ± {auc_.std():.4f}")
        print(f"  Accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
        print(f"  Recall (Sensitivity): {recall.mean():.4f} ± {recall.std():.4f}")
        print(f"  F1: {f1.mean():.4f} ± {f1.std():.4f}")
        print(f"  Specificity: {spec.mean():.4f} ± {spec.std():.4f}")
    pd.DataFrame(results).to_csv('grid_tuning_metrics.csv', index=False)
    return tuned

def main():
    os.makedirs('results_grid', exist_ok=True)
    X, y = load_gist_train_data()
    X_scaled = preprocessing(X)
    X_sel = rfecv_feature_select(X_scaled, y, 'results_grid')
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    tuned = grid_tune_metrics(X_sel, y, cv)
    print("\nTuning complete. Full metrics printed + grid_tuning_metrics.csv")

if __name__ == "__main__":
    main()


# %%
