#!/usr/bin/env python3
"""Quick comparison PCA vs RFECV"""
import sys
sys.path.insert(0, '/Users/karime/Documents/VSC/ML-project')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from load_data import load_data
import warnings
warnings.filterwarnings('ignore')

# Load data
df = load_data()
y_raw = df['label']
X = df.drop(columns=['label'])
X = X.select_dtypes(include=[np.number])
mapping = {'GIST': 1, 'non-GIST': 0}
y = y_raw.map(mapping).astype(int).values

print("StandardScaler + PCA Pipeline")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA - use 50 components
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X_pca.shape[1]} components, explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Classifiers
classifiers = [
    ('LDA', LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')),
    ('GaussianNB', GaussianNB()),
    ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
    ('SGDClassifier', SGDClassifier(random_state=42)),
    ('KNeighbors', KNeighborsClassifier()),
    ('DecisionTree', DecisionTreeClassifier(random_state=42)),
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42))
]

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, clf in classifiers:
    y_pred = model_selection.cross_val_predict(clf, X_pca, y, cv=cv)
    try:
        if hasattr(clf, 'predict_proba'):
            y_score = model_selection.cross_val_predict(clf, X_pca, y, cv=cv, method='predict_proba')[:, 1]
        else:
            y_score = y_pred
    except:
        y_score = y_pred
    
    accuracy = metrics.accuracy_score(y, y_pred)
    auc = metrics.roc_auc_score(y, y_score)
    f1 = metrics.f1_score(y, y_pred)
    
    results.append({'classifier': name, 'accuracy': accuracy, 'auc': auc, 'f1': f1})
    print(f"{name}: Acc={accuracy:.4f}, AUC={auc:.4f}, F1={f1:.4f}")

# Save
results_df = pd.DataFrame(results)
results_df.to_csv('/Users/karime/Documents/VSC/ML-project/results/GIST_cv_metrics_standard_pca.csv', index=False)
print("\nSaved to results/GIST_cv_metrics_standard_pca.csv")
