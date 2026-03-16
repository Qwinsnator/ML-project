"""
GIST Data Exploration Visualization Script
Adapted from E1.2 Basic Classifiers, E1.3 Generalization, E1.4 Features exercises
Uses only GIST_Train.csv data for EDA and visualization.

Generates plots in Visualization/ folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets as ds
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import  StratifiedKFold
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from scipy.stats import mannwhitneyu

# Classifiers from E1.2
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Create visualization output directory
output_dir = 'Visualization'
os.makedirs(output_dir, exist_ok=True)

# Create results directory
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

def colorplot(clf, ax, x, y, h=100):
    """
    Overlay decision areas as colors (from E1.2/E1.3 exercises).
    """
    xstep = (x.max() - x.min()) / 20.0
    ystep = (y.max() - y.min()) / 20.0
    x_min, x_max = x.min() - xstep, x.max() + xstep
    y_min, y_max = y.min() - ystep, y.max() + ystep
    h = max((x_max - x_min, y_max - y_min)) / h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    features = np.c_[xx.ravel(), yy.ravel()]
    
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(features)
    else:
        Z = clf.predict_proba(features)
    if len(Z.shape) > 1:
        Z = Z[:, 1]

    cm = plt.cm.RdBu_r
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

def load_gist_train_data():
    """
    Load and preprocess GIST_Train.csv (from split.py).
    """
    print("Loading GIST Train dataset...")
    df = pd.read_csv('GIST_Train.csv')
    print(f"Data shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    y_raw = df['label']
    X = df.drop(columns=['label'])
    X = X.select_dtypes(include=[np.number])
    
    # Encode labels
    mapping = {"GIST": 1, "non-GIST": 0}
    y = y_raw.map(mapping).astype(int).values
    
    # Preprocess
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    print(f"Feature matrix shape: {X_scaled.shape}")
    return X_scaled, y, scaler, imputer, df.columns.drop('label').tolist(), df

def plot_data_exploration(df, output_dir):
    """
    Basic EDA plots from E1.4 Features exercise style.
    """
    # Encode label for correlation
    df_corr = df.copy()
    mapping = {"GIST": 1, "non-GIST": 0}
    df_corr['label_num'] = df_corr['label'].map(mapping)
    
    plt.figure(figsize=(15, 10))
    
    # Class distribution
    plt.subplot(2, 3, 1)
    df['label'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.ylabel('Count')
    
    # Missing values heatmap (if any)
    plt.subplot(2, 3, 2)
    missing_pct = (df.isnull().sum() / len(df)) * 100
    plt.bar(range(len(missing_pct)), missing_pct)
    plt.title('Missing Values % per Feature')
    plt.ylabel('% Missing')
    plt.xticks([])
    
    # Feature correlations (top 10 numeric only)
    plt.subplot(2, 3, 3)
    numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
    corr = df_corr[numeric_cols].corrwith(df_corr['label_num']).sort_values(ascending=False)
    if len(corr) > 0:
        corr.head(10).plot(kind='bar')
        plt.title('Top 10 Feature Correlations with Label')
    else:
        plt.text(0.5, 0.5, 'No numeric features for corr', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Correlation Plot')
    
    # Sample feature distributions
    feat1 = 'PREDICT_original_sf_compactness_avg_2.5D'
    feat2 = 'PREDICT_original_sf_rad_dist_avg_2.5D'
    plt.subplot(2, 3, 4)
    sns.boxplot(data=df, x='label', y=feat1)
    plt.title(f'{feat1} by Class')
    mapping = {"GIST": 1, "non-GIST": 0}
    
    plt.subplot(2, 3, 5)
    sns.boxplot(data=df, x='label', y=feat2)
    plt.title(f'{feat2} by Class')
    
    plt.subplot(2, 3, 6)
    plt.scatter(df[feat1], df[feat2], c=df['label'].map(mapping), alpha=0.6)
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.title('Feature Scatter Plot')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gist_eda.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("EDA plots saved to Visualization/gist_eda.png")

def plot_pca_visualization(X_scaled, y, output_dir):
    """
    PCA visualization from E1.4 Features exercise.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Paired', s=50, edgecolor='k')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('GIST Train Data - PCA (2 components)')
    plt.colorbar(scatter, label='Class (0=non-GIST, 1=GIST)')
    
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum)+1), cumsum, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gist_pca.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("PCA plots saved to Visualization/gist_pca.png")
    print(f"Top 2 PCs explain {sum(pca.explained_variance_ratio_[:2]):.2%} variance")

def plot_feature_importance_ranking(X_scaled, y, feature_names, output_dir, results_dir):
    """
    Compute mutual information scores to rank features by relationship to label.
    Save full ranking table as CSV and top20 bar plot.
    Returns top 20 feature names.
    """
    print("Computing feature importance ranking using mutual information...")
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42, n_jobs=-1)
    
    # Univariate tests table (chi2, ANOVA F, Mann-Whitney U)
    X_num = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Chi2 (non-neg)
    X_chi = X_num.clip(lower=0)
    chi_stat, _ = chi2(X_chi, y)
    
    # ANOVA F
    f_stat, _ = f_classif(X_num, y)
    
    # Mann-Whitney U stat (smaller = better separation)
    mw_stats = []
    for col in feature_names:
        _, mw_stat = mannwhitneyu(X_num[col][y==0], X_num[col][y==1])
        mw_stats.append(mw_stat)
    
    ranking_df = pd.DataFrame({
        'feature': feature_names,
        'chi2_stat': chi_stat,
        'anova_f': f_stat,
        'mw_u': mw_stats
    })
    # Avg normalized score (higher better)
    ranking_df['score'] = (ranking_df['chi2_stat'].rank(ascending=False) + ranking_df['anova_f'].rank(ascending=False) + (len(feature_names) - ranking_df['mw_u'].rank()).rank(ascending=False)) / 3
    ranking_df = ranking_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Top 20 only for CSV
    ranking_df_top20 = ranking_df.head(20)[['feature', 'chi2_stat', 'anova_f', 'mw_u', 'score']]
    ranking_df_top20.to_csv(os.path.join(results_dir, 'gist_feature_ranking.csv'), index=False)
    
    # Save full ranking table
    ranking_df.to_csv(os.path.join(results_dir, 'gist_feature_ranking.csv'), index=False)
    print(f"Full feature ranking saved to results/gist_feature_ranking.csv")
    print("\nTop 10 features:")
    print(ranking_df.head(10))
    
    # Plot top 20 bar
    plt.figure(figsize=(12, 8))
    top20 = ranking_df.head(20)
    plt.barh(range(len(top20)), top20['score'])
    plt.yticks(range(len(top20)), top20['feature'])
    plt.xlabel('Univariate Score')
    plt.title('Top 20 Features by Univariate Tests (chi2 + ANOVA F + MW U)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gist_feature_ranking_top20.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Top 20 univariate plot saved.")
    
    top_features = top20['feature'].tolist()
    return top_features

def plot_top_features_distributions(df, top_features, output_dir):
    """
    Box plots for top 20 features' distributions split by label, with outlier circles.
    Uses raw df (handle NaNs by dropping per plot).
    """
    print("Plotting box plots for top 20 features...")
    df_top = df[top_features + ['label']].copy()
    
    # Melt for seaborn
    df_melted = df_top.melt(id_vars='label', var_name='feature', value_name='value')
    df_melted = df_melted.dropna()  # Drop NaNs for clean plots
    
    plt.figure(figsize=(20, 15))
    sns.boxplot(data=df_melted, x='feature', y='value', hue='label')
    plt.title('Top 20 Features: Box Plots by Label (GIST vs non-GIST)\nOutliers shown as circles')
    plt.xlabel('Features')
    plt.ylabel('Feature Value')
    plt.xticks(rotation=90)
    plt.legend(title='Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gist_top20_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Top 20 box plots saved to Visualization/gist_top20_distributions.png")

def plot_classifier_decision_boundaries(X_pca, y, output_dir):
    """
    Classifier decision boundaries from E1.2 Basic Classifiers exercise.
    """
    clsfs = [
        ('LDA', LinearDiscriminantAnalysis()),
        ('QDA', QuadraticDiscriminantAnalysis()),
        ('GaussianNB', GaussianNB()),
        ('LogisticRegression', LogisticRegression(max_iter=1000)),
        ('KNeighbors', KNeighborsClassifier(n_neighbors=5)),
        ('DecisionTree', DecisionTreeClassifier(max_depth=5, random_state=42))
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (name, clf) in enumerate(clsfs):
        clf.fit(X_pca, y)
        y_pred = clf.predict(X_pca)
        
        axes[idx].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Paired', s=50, edgecolor='k')
        try:
            colorplot(clf, axes[idx], X_pca[:, 0], X_pca[:, 1])
        except:
            pass
        
        misclass = (y != y_pred).sum()
        acc = 1 - misclass / len(y)
        axes[idx].set_title(f"{name}\nMisclass: {misclass}/{len(y)} (Acc: {acc:.3f})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gist_classifiers_pca.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Classifier decision boundaries saved to Visualization/gist_classifiers_pca.png")

def plot_learning_curves(X_scaled, y, output_dir):
    """
    Learning curves from E1.3 Generalization exercise.
    """
    from sklearn.model_selection import learning_curve
    
    knn = KNeighborsClassifier(n_neighbors=5)
    train_sizes, train_scores, val_scores = learning_curve(
        knn, X_scaled, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='roc_auc'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training')
    plt.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation')
    plt.xlabel('Training Set Size')
    plt.ylabel('ROC-AUC Score')
    plt.title('GIST Learning Curve (KNN)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'gist_learning_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Learning curve saved to Visualization/gist_learning_curve.png")

if __name__ == "__main__":
    # Load train data only (now returns feature_names and df too)
    X_scaled, y, scaler, imputer, feature_names, df = load_gist_train_data()
    
    # Generate all visualizations
    plot_data_exploration(df, output_dir)
    plot_pca_visualization(X_scaled, y, output_dir)
    plot_classifier_decision_boundaries(PCA(n_components=2).fit_transform(X_scaled), y, output_dir)
    plot_learning_curves(X_scaled, y, output_dir)
    
    # New: Feature ranking and top distributions
    top_features = plot_feature_importance_ranking(X_scaled, y, feature_names, output_dir, results_dir)
    plot_top_features_distributions(df, top_features, output_dir)
    
    print(f"\nAll GIST train data visualizations saved to {output_dir}/")
    print(f"Feature ranking table saved to {results_dir}/gist_feature_ranking.csv")
    print("Run: python Visualization/gist_data_exploration.py")
