"""
GIST vs non-GIST Feature Selection: Find Discriminating Features
Identifies features that statistically differ most between classes.

Methods: Mann-Whitney U, Cohen's d (effect size), Mutual Information
Composite ranking → Top discriminators for ML.

Run: python feature_selection_gist.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Directories
cwd = '/Users/karime/Documents/VSC/ML-project'
output_dir = os.path.join(cwd, 'Visualization')
results_dir = os.path.join(cwd, 'results')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

print("🔍 GIST Feature Selection: Finding Class-Discriminating Features")
print("=" * 60)

def load_data():
    df = pd.read_csv(os.path.join(cwd, 'GIST_Train.csv'))
    print(f"📊 Loaded: {df.shape} (samples, features)")
    print(f"🏷️  Classes:\n{df['label'].value_counts()}")
    
    # Prepare
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    X_raw = df.drop('label', axis=1).select_dtypes(include=[np.number])
    
    # Impute & scale
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X_raw))
    
    feature_names = X_raw.columns.tolist()
    print(f"✨ Preprocessed: {X.shape}")
    return X, y, feature_names, df

def compute_univariate_stats(X, y, feature_names):
    """Mann-Whitney U, Cohen's d, MI for each feature."""
    stats_df = pd.DataFrame(index=feature_names)
    
    gist_mask = y == 1
    n_gist, n_non = gist_mask.sum(), (~gist_mask).sum()
    
    print(f"Computing stats for {len(feature_names)} features...")
    
    for i, feat in enumerate(feature_names):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(feature_names)}")
            
        x_gist = X[gist_mask, i]
        x_non = X[~gist_mask, i]
        
        # Skip if any class empty or all identical
        if len(x_gist) == 0 or len(x_non) == 0:
            stats_df.loc[feat, 'mw_p'] = np.nan
            stats_df.loc[feat, 'cohens_d'] = 0
            stats_df.loc[feat, 'mi'] = 0
            continue
            
        # Mann-Whitney U
        try:
            u_stat, p_val = mannwhitneyu(x_gist, x_non, alternative='two-sided')
            stats_df.loc[feat, 'mw_u'] = u_stat / (len(x_gist) * len(x_non))
            stats_df.loc[feat, 'mw_p'] = p_val
        except:
            stats_df.loc[feat, 'mw_u'] = np.nan
            stats_df.loc[feat, 'mw_p'] = np.nan
        
        # Cohen's d (safe computation)
        try:
            var_gist = np.var(x_gist, ddof=1) if len(x_gist) > 1 else 0
            var_non = np.var(x_non, ddof=1) if len(x_non) > 1 else 0
            pooled_var = (var_gist + var_non) / 2
            if pooled_var > 1e-10:  # Avoid div by zero
                cohens_d = (np.mean(x_gist) - np.mean(x_non)) / np.sqrt(pooled_var)
                stats_df.loc[feat, 'cohens_d'] = abs(cohens_d)
            else:
                stats_df.loc[feat, 'cohens_d'] = 0
        except:
            stats_df.loc[feat, 'cohens_d'] = 0
        
        # Mutual Information
        try:
            mi = mutual_info_classif(X[:, i].reshape(-1, 1), y, random_state=42)[0]
            stats_df.loc[feat, 'mi'] = mi
        except:
            stats_df.loc[feat, 'mi'] = 0
    
    # Handle NaNs in ranking
    stats_df = stats_df.fillna(0)
    
    # Composite score
    stats_df['mw_rank'] = stats_df['mw_u'].rank(ascending=True)
    stats_df['d_rank'] = stats_df['cohens_d'].rank(ascending=False)
    stats_df['mi_rank'] = stats_df['mi'].rank(ascending=False)
    stats_df['composite_score'] = (stats_df['mw_rank'] + stats_df['d_rank'] + stats_df['mi_rank']) / 3
    stats_df['rank'] = stats_df['composite_score'].rank(ascending=True)
    
    return stats_df.sort_values('composite_score')

def plot_top_discriminators(df_raw, top_features, stats_df, output_dir):
    \"\"\"Matplotlib boxplots - bulletproof, no seaborn.\"\"\"
    print(f"📈 Plotting top {len(top_features)} discriminators...")
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16), sharey=True)
    axes = axes.flatten()
    
    top_feats_subset = top_features[:20]  # Top 20 max
    
    for idx, feat in enumerate(top_feats_subset):
        ax = axes[idx]
        
        # Data for this feature
        data_gist = df_raw[df_raw['label'] == 'GIST'][feat].dropna()
        data_non = df_raw[df_raw['label'] == 'non-GIST'][feat].dropna()
        
        if len(data_gist) > 0 and len(data_non) > 0:
            box = ax.boxplot([data_non, data_gist], labels=['non-GIST', 'GIST'], patch_artist=True)
            box['boxes'][0].set_facecolor('lightblue')
            box['boxes'][1].set_facecolor('salmon')
            ax.set_title(f"{feat.split('_')[-1]}\np={stats_df.loc[feat, 'mw_p']:.2e}", fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{feat.split('_')[-1]}", fontsize=9)
        
        ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for idx in range(len(top_feats_subset), 20):
        axes[idx].set_visible(False)
    
    plt.suptitle('🏆 TOP 20 GIST Discriminators: Boxplots by Class (MW p-values)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gist_top_discriminators_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Top discriminators boxplots saved ✓")

def main():
    X, y, feature_names, df_raw = load_data()
    
    # Compute stats
    stats_df = compute_univariate_stats(X, y, feature_names)
    
    # Save full results
    stats_df.to_csv(os.path.join(results_dir, 'gist_discriminating_features.csv'))
    
    # Top results
    top20 = stats_df.head(20)
    print("\n🏆 TOP 10 GIST DISCRIMINATORS:")
    print(top20[['mw_p', 'cohens_d', 'mi', 'composite_score']].round(4).head(10))
    
    print(f"\n📈 Features with p < 0.001: {len(stats_df[stats_df['mw_p'] < 0.001])}")
    print(f"💪 Large effect size (|d| > 0.8): {len(stats_df[stats_df['cohens_d'] > 0.8])}")
    
    # Plots
    plot_top_discriminators(df_raw, top20.index.tolist(), stats_df, output_dir)
    
    # Summary stats plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # p-value histogram
    axes[0,0].hist(stats_df['mw_p'], bins=50, alpha=0.7)
    axes[0,0].set_title('Mann-Whitney p-values Distribution')
    axes[0,0].set_xlabel('p-value')
    axes[0,0].set_yscale('log')
    
    # Cohen's d histogram
    axes[0,1].hist(stats_df['cohens_d'], bins=50, alpha=0.7)
    axes[0,1].set_title("Cohen's d (Effect Size) Distribution")
    axes[0,1].axvline(0.8, color='red', ls='--', label='Large effect')
    axes[0,1].legend()
    
    # Top 20 scores bar
    top20.plot(x='composite_score', y=feature_names[:20], kind='barh', ax=axes[1,0])
    axes[1,0].set_title('Top 20 Composite Score')
    
    # Scatter: effect size vs MI
    axes[1,1].scatter(stats_df['cohens_d'], stats_df['mi'], alpha=0.6)
    axes[1,1].set_xlabel("Cohen's d")
    axes[1,1].set_ylabel('Mutual Information')
    axes[1,1].set_title('Effect Size vs MI (correlation)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gist_feature_selection_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Results saved:")
    print(f"   📄 results/gist_discriminating_features.csv (ALL features ranked)")
    print(f"   🖼️  Visualization/gist_top_discriminators_boxplots.png")
    print(f"   🖼️  Visualization/gist_feature_selection_stats.png")
    print("\n🚀 TOP RECOMMENDED FEATURES for GIST detection:", ', '.join(top20.index[:5].tolist()))

if __name__ == "__main__":
    main()

