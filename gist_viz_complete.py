#!/usr/bin/env python3
"""
GIST Data Visualization - COMPLETE READY-TO-RUN SCRIPT
Loads GIST_Train.csv → Distributions, outliers, feature selection.

RUN ME DIRECTLY!
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import mannwhitneyu

# Config
CWD = '/Users/karime/Documents/VSC/ML-project'
os.makedirs(os.path.join(CWD, 'Visualization'), exist_ok=True)
os.makedirs(os.path.join(CWD, 'results'), exist_ok=True)

print('🚀 GIST Analysis Starting...')
print('=' * 50)

# 1. LOAD DATA
df = pd.read_csv(os.path.join(CWD, 'GIST_Train.csv'))
print(f'📊 Shape: {df.shape}')
print(f'🏷️ Classes:\n{df["label"].value_counts()}')

le = LabelEncoder()
y = le.fit_transform(df['label'])
X_raw = df.select_dtypes(include=[np.number]).drop('label', axis=1, errors='ignore')

# Preprocess
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X = scaler.fit_transform(imputer.fit_transform(X_raw))
feature_names = X_raw.columns.tolist()

print(f'✨ Processed: {X.shape}')

# 2. FEATURE SELECTION STATS
print('\n🔍 Computing discriminating features...')
stats = {}

for i, feat in enumerate(feature_names):
    if i % 50 == 0: print(f'  {i}/{len(feature_names)}')
    
    x_gist = X[y == 1, i]
    x_non = X[y == 0, i]
    
    if len(x_gist) < 2 or len(x_non) < 2: continue
    
    # Mann-Whitney U
    _, p_val = mannwhitneyu(x_gist, x_non)
    
    # Cohen's d
    mean_diff = np.mean(x_gist) - np.mean(x_non)
    pooled_var = (np.var(x_gist, ddof=1) + np.var(x_non, ddof=1)) / 2
    cohens_d = abs(mean_diff / np.sqrt(pooled_var)) if pooled_var > 1e-10 else 0
    
    # Mutual Info
    mi = mutual_info_classif(X[:, i:i+1], y)[0]
    
    stats[feat] = {'p_val': p_val, 'cohens_d': cohens_d, 'mi': mi}

# 3. RANK FEATURES
stats_df = pd.DataFrame(stats).T
stats_df['rank'] = stats_df.rank(axis=0, ascending=[True, False, False]).mean(axis=1)
top20 = stats_df.sort_values('rank').head(20)

print('\n🏆 TOP 5 GIST DISCRIMINATORS:')
print(top20.round(4)[['p_val', 'cohens_d', 'mi']])

top20.to_csv(os.path.join(CWD, 'results/gist_top_features.csv'))

# 4. VISUALIZATIONS
print('\n📊 Creating plots...')

# Class distribution
plt.figure(figsize=(8, 6))
df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('GIST vs non-GIST Distribution')
plt.ylabel('')
plt.savefig(os.path.join(CWD, 'Visualization/gist_classes.png'), dpi=300, bbox_inches='tight')
plt.close()

# Top features boxplots
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()
for i, feat in enumerate(top20.index):
    ax = axes[i]
    ax.boxplot([
        df[df['label'] == 'non-GIST'][feat].dropna(),
        df[df['label'] == 'GIST'][feat].dropna()
    ], labels=['non-GIST', 'GIST'])
    ax.set_title(f"{feat.split('_')[-1]}\np={top20.loc[feat, 'p_val']:.1e}")
    ax.tick_params(axis='x', rotation=45)

for j in range(i+1, 20): axes[j].set_visible(False)
plt.suptitle('TOP 20 GIST Discriminators (Boxplots)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(CWD, 'Visualization/gist_discriminators.png'), dpi=300, bbox_inches='tight')
plt.close()

print('\n✅ COMPLETE!')
print('📁 Files:')
print('   results/gist_top_features.csv')
print('   Visualization/gist_classes.png')
print('   Visualization/gist_discriminators.png')
print('\n🏆 Ready for ML with top20!')

