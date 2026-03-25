#!/usr/bin/env python3
"""
Quick script to run PCA pipeline and compare with RFECV
"""
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
print("="*50)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA - use 50 components
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X_pca.shape[1]} components, explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Classifiers
classifiers = {
    'LDA': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
    'GaussianNB': GaussianNB(),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SGDClassifier': SGDClassifier(random_state=42),
    'KNeighbors': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, clf in classifiers.items():
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
results_df.to_csv('results/GIST_cv_metrics_standard_pca.csv', index=False)
print("\nSaved to results/GIST_cv_metrics_standard_pca.csv")

# Compare with RFECV
print("\n" + "="*50)
print("COMPARISON: PCA vs RFECV")
print("="*50)

# Previous RFECV results (MinMaxScaler + RFECV)
rfecv_results = [
    {'classifier': 'LDA', 'accuracy': 0.699, 'auc': 0.748, 'f1': 0.694},
    {'classifier': 'LogisticRegression', 'accuracy': 0.671, 'auc': 0.714, 'f1': 0.672},
    {'classifier': 'SGDClassifier', 'accuracy': 0.650, 'auc': 0.686, 'f1': 0.639},
    {'classifier': 'KNeighbors', 'accuracy': 0.610, 'auc': 0.643, 'f1': 0.600},
    {'classifier': 'RandomForest', 'accuracy': 0.614, 'auc': 0.642, 'f1': 0.609},
    {'classifier': 'GaussianNB', 'accuracy': 0.549, 'auc': 0.548, 'f1': 0.431},
    {'classifier': 'DecisionTree', 'accuracy': 0.516, 'auc': 0.515, 'f1': 0.554},
]

print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "Classifier", "RFECV_AUC", "PCA_AUC", "RFECV_Acc", "PCA_Acc", "RFECV_F1", "PCA_F1"))
print("-"*80)

for i, name in enumerate(['LDA', 'LogisticRegression', 'SGDClassifier', 'KNeighbors', 'RandomForest', 'GaussianNB', 'DecisionTree']):
    pca_row = results_df[results_df['classifier'] == name].iloc[0]
    rfecv_row = rfecv_results[i]
    print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        name, rfecv_row['auc'], pca_row['auc'], 
        rfecv_row['accuracy'], pca_row['accuracy'],
        rfecv_row['f1'], pca_row['f1']))

# Summary
pca_best_auc = results_df['auc'].max()
rfecv_best_auc = 0.748
print(f"\nBest AUC - RFECV: {rfecv_best_auc:.4f}, PCA: {pca_best_auc:.4f}")
if rfecv_best_auc > pca_best_auc:
    print("=> RFECV performs better for this dataset!")
else:
    print("=> PCA performs better for this dataset!")
