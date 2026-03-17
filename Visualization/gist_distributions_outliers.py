"""
GIST Train Data Distributions & Outliers Visualization
Standalone script focused on univariate distributions and outlier detection.
Saves plots to Visualization/ folder.

Dependencies: pandas, numpy, matplotlib, seaborn (in requirements.txt)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ensure output directory
output_dir = '/Users/karime/Documents/VSC/ML-project/Visualization'
os.makedirs(output_dir, exist_ok=True)

def load_and_preprocess():
    """Load GIST_Train.csv and basic preprocessing."""
    print("Loading GIST_Train.csv...")
    df = pd.read_csv('/Users/karime/Documents/VSC/ML-project/GIST_Train.csv')
    print(f"Shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts(normalize=True)}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric features: {len(numeric_cols)}")
    
    # Basic stats
    print("\nDataset info:")
    print(df.info())
    
    return df, numeric_cols

def detect_outliers(df, method='iqr', top_n=20):
    """Detect outliers using IQR or Z-score."""
    outliers_df = pd.DataFrame(index=df.index)
    
    if method == 'iqr':
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers_df[col] = ((df[col] < lower) | (df[col] > upper)).astype(int)
    elif method == 'zscore':
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                z = np.abs(stats.zscore(df[col].dropna()))
                outliers_df[col] = (z > 3).reindex(df.index, fill_value=0).astype(int)
    
    outlier_pct = (outliers_df.sum() / len(df) * 100).sort_values(ascending=False)
    return outliers_df, outlier_pct

def plot_class_distribution(df, output_dir):
    """Class balance pie chart."""
    fig, ax = plt.subplots(figsize=(8, 8))
    label_counts = df['label'].value_counts()
    wedges, texts, autotexts = ax.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('GIST vs non-GIST Class Distribution')
    plt.savefig(os.path.join(output_dir, 'gist_class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Class distribution plot saved")

def plot_missing_values(df, output_dir):
    """Missing values bar chart."""
    missing_pct = df.isnull().sum()[df.isnull().sum() > 0] / len(df) * 100
    if len(missing_pct) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        missing_pct.plot(kind='bar', ax=ax)
        ax.set_title('Missing Values % per Feature')
        ax.set_ylabel('% Missing')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gist_missing_values.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Missing values plot saved")
    else:
        print("No missing values")

def plot_top_features_boxplots(df, top_features, output_dir):
    """Box plots for top features stratified by class."""
    df_plot = df[['label'] + top_features].melt(id_vars='label', var_name='feature', value_name='value')
    df_plot = df_plot.dropna()
    
    plt.figure(figsize=(20, 12))
    sns.boxplot(data=df_plot, x='feature', y='value', hue='label')
    plt.title('Top 20 Features: Distributions by Class (IQR Outliers Shown)', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Feature Value', fontsize=12)
    plt.xticks(rotation=90)
    plt.legend(title='Label', title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gist_top20_boxplots_outliers.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Top 20 box plots saved")

def plot_feature_correlation(df, top_features, output_dir):
    """Correlation heatmap for top features."""
    df_corr = df[top_features].corr()
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(df_corr, annot=False, cmap='coolwarm', center=0, square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Top 20 Features Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gist_top20_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Correlation heatmap saved")

def plot_outlier_summary(outlier_pct, output_dir):
    """Outlier percentage bar chart."""
    top_outliers = outlier_pct.sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    top_outliers.plot(kind='bar')
    plt.title('Top 20 Features with Most Outliers (% of samples)')
    plt.ylabel('% Samples with Outlier')
    plt.xlabel('Features')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gist_outlier_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Outlier summary plot saved")

def main():
    df, numeric_cols = load_and_preprocess()
    
    # Detect outliers (IQR preferred)
    outliers_df, outlier_pct = detect_outliers(df, method='iqr')
    
    print(f"\nOutlier detection complete:")
    print(f"Total outliers: {outliers_df.sum().sum()}")
    print(f"Features with >10% outliers:\n{outlier_pct[outlier_pct > 10].head()}")
    
    # Select top features (variance-based for general viz)
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.head(20).index.tolist()
    print(f"\nTop 20 high-variance features: {top_features[:5]}...")
    
    # Generate plots
    plot_class_distribution(df, output_dir)
    plot_missing_values(df, output_dir)
    plot_top_features_boxplots(df, top_features, output_dir)
    plot_feature_correlation(df, top_features, output_dir)
    plot_outlier_summary(outlier_pct, output_dir)
    
    print(f"\n✅ All visualizations saved to {output_dir}/")
    print("📊 Key files:")
    print("   - gist_class_distribution.png")
    print("   - gist_top20_boxplots_outliers.png (main distributions + outliers)")
    print("   - gist_outlier_summary.png")
    print("   - gist_top20_correlation.png")

if __name__ == "__main__":
    main()

