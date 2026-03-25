import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

print("GIST Classifiers - Minimal Recall Ranking")

# Load
df = pd.read_csv('GIST_Train.csv')
X = df.drop('label', axis=1).select_dtypes(float)
y = df['label'].map({'GIST': 1, 'non-GIST': 0}).astype(int)

# Preprocess + 3SD outlier removal
imp = SimpleImputer(strategy='median')
X = imp.fit_transform(X)
n_feat_out, n_out_tot = 0, 0
for j in range(X.shape[1]):
    m, s = np.mean(X[:, j]), np.std(X[:, j])
    mask = np.abs(X[:, j] - m) > 3*s
    n_out_tot += mask.sum()
    if mask.sum() > 0:
        n_feat_out += 1
        X[mask, j] = np.nan
X = SimpleImputer(strategy='median').fit_transform(X)
print(f"Outliers removed: {n_out_tot} ({n_out_tot/len(y)*100:.1f}%) in {n_feat_out} features")

sca = RobustScaler()
X = sca.fit_transform(X)

# Classifiers
clfs = {
'LR': LogisticRegression(max_iter=2000, solver='liblinear', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LDA': LinearDiscriminantAnalysis(),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)

results = []
for name, clf in clfs.items():
    rec = cross_val_score(clf, X, y, cv=cv, scoring='recall')
    results.append({'clf': name, 'recall_mean': rec.mean(), 'recall_std': rec.std()})
    
df_res = pd.DataFrame(results).sort_values('recall_mean', ascending=False)
print("\nRANKED BY RECALL:")
print(df_res.round(4))

print(f"\nTOP: {df_res.iloc[0]['clf']} ({df_res.iloc[0]['recall_mean']:.3f} ± {df_res.iloc[0]['recall_std']:.3f})")
print("Done.")
