"""
GIST Classifier Analysis
Applies the E1.2 Basic Classifiers code to the GIST radiomic features dataset.
Generates decision boundary plots and evaluation metrics.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets as ds
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Classifiers (from E1.2)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Import data loading function
from load_data import load_data

# Try to import plotly for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")
    print("For interactive matplotlib plots in Python script, use: plt.show()")


# ============== HELPER FUNCTIONS FROM E1.2 ==============

def colorplot(clf, ax, x, y, h=100, precomputer=None):
    """
    Overlay the decision areas as colors in an axes.
    Input:
        clf: trained classifier
        ax: axis to overlay color mesh on
        x: feature on x-axis
        y: feature on y-axis
        h(optional): steps in the mesh
    """
    # Create a meshgrid the size of the axis
    xstep = (x.max() - x.min()) / 20.0
    ystep = (y.max() - y.min()) / 20.0
    x_min, x_max = x.min() - xstep, x.max() + xstep
    y_min, y_max = y.min() - ystep, y.max() + ystep
    h = max((x_max - x_min, y_max - y_min)) / h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    features = np.c_[xx.ravel(), yy.ravel()]

    # Plot the decision boundary
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(features)
    else:
        Z = clf.predict_proba(features)

    if len(Z.shape) > 1:
        Z = Z[:, 1]

    # Put the result into a color plot
    cm = plt.cm.RdBu_r
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    del xx, yy, x_min, x_max, y_min, y_max, Z, cm


def load_gist_data():
    """
    Load and preprocess the GIST radiomic features dataset.
    Returns X (features) and y (labels) after PCA to 2D for visualization.
    Also returns full feature set before PCA for metric evaluation.
    """
    print("Loading GIST radiomic features data...")
    df = load_data()
    print(f"Data shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    # Extract features and labels
    y_raw = df['label']
    X = df.drop(columns=['label'])
    
    # Ensure numeric features only
    X = X.select_dtypes(include=[np.number])
    
    # Encode labels: GIST=1, non-GIST=0
    mapping = {"GIST": 1, "non-GIST": 0}
    y = y_raw.map(mapping).astype(int).values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Class distribution: GIST={np.sum(y==1)}, non-GIST={np.sum(y==0)}")
    
    # Preprocess: handle missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # For visualization: apply PCA to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA reduced shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
    return X_pca, y, X_scaled, pca, scaler, imputer


def create_synthetic_datasets():
    """
    Create synthetic datasets similar to E1.2 for comparison.
    """
    print("\nCreating synthetic datasets...")
    
    # Dataset 1: Two informative features, one cluster per class
    X1, Y1 = ds.make_classification(n_samples=100, n_features=2, n_redundant=0,
                                    n_informative=2, n_clusters_per_class=1)
    
    # Dataset 2: One informative feature, one cluster per class
    X2, Y2 = ds.make_classification(n_samples=100, n_features=2, n_redundant=0,
                                    n_informative=1, n_clusters_per_class=1)
    
    # Dataset 3: Two blobs, two classes
    X3, Y3 = ds.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=5)
    
    return (X1, Y1), (X2, Y2), (X3, Y3)


def run_classifiers(X, y, output_dir, dataset_name="GIST"):
    """
    Run all classifiers and create decision boundary plots.
    Returns fitted classifiers and their predictions.
    """
    print(f"\n{'='*60}")
    print(f"Running classifiers on {dataset_name} dataset")
    print(f"{'='*60}")
    
    # Define classifiers
    clsfs = [
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
        LogisticRegression(max_iter=1000, random_state=42),
        SGDClassifier(random_state=42),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=42)
    ]
    
    clf_names = [
        "LDA", "QDA", "GaussianNB", "LogisticRegression", 
        "SGDClassifier", "KNeighbors", "DecisionTree"
    ]
    
    # Store results
    results = []
    clfs_fit = []
    
    # Create figure for decision boundary plots
    n_classifiers = len(clsfs)
    fig = plt.figure(figsize=(21, 3 * n_classifiers))
    
    for idx, (clf, clf_name) in enumerate(zip(clsfs, clf_names)):
        print(f"\nTraining {clf_name}...")
        
        # Fit classifier
        clf.fit(X, y)
        y_pred = clf.predict(X)
        
        # Calculate metrics
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X)[:, 1]
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(X)
        else:
            y_score = y_pred
        
        # Compute metrics
        try:
            auc = metrics.roc_auc_score(y, y_score)
        except:
            auc = float('nan')
        
        accuracy = metrics.accuracy_score(y, y_pred)
        f1 = metrics.f1_score(y, y_pred, zero_division=0)
        precision = metrics.precision_score(y, y_pred, zero_division=0)
        recall = metrics.recall_score(y, y_pred, zero_division=0)
        
        print(f"  Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        
        # Store result
        result = {
            'classifier': clf_name,
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'misclassified': int((y != y_pred).sum()),
            'total': len(y)
        }
        results.append(result)
        
        # Create subplot
        ax = fig.add_subplot(n_classifiers, 1, idx + 1)
        ax.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                   s=25, edgecolor='k', cmap=plt.cm.Paired)
        
        # Add decision boundary
        try:
            colorplot(clf, ax, X[:, 0], X[:, 1])
        except Exception as e:
            print(f"  Warning: Could not create colorplot: {e}")
        
        # Add title with misclassification info
        title = f"{clf_name}: Misclass {result['misclassified']}/{result['total']} (Acc: {accuracy:.3f}, AUC: {auc:.3f})"
        ax.set_title(title)
        
        clfs_fit.append(clf)
    
    # Save figure
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"{dataset_name}_decision_boundaries.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved decision boundary plot to: {plot_path}")
    
    # Save metrics to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"{dataset_name}_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved metrics to: {csv_path}")
    
    return clfs_fit, results


def run_comparison_analysis(X_pca, y, X_full, output_dir):
    """
    Run classifiers on multiple datasets and create comparison visualization.
    """
    print(f"\n{'='*60}")
    print("Running comparison analysis")
    print(f"{'='*60}")
    
    # Create synthetic datasets
    (X1, Y1), (X2, Y2), (X3, Y3) = create_synthetic_datasets()
    
    # All datasets to process
    datasets = [
        (X_pca, y, "GIST_PCA"),
        (X1, Y1, "Synthetic_2informative"),
        (X2, Y2, "Synthetic_1informative"),
        (X3, Y3, "Synthetic_Blobs")
    ]
    
    # Define classifiers
    clsfs = [
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
        LogisticRegression(max_iter=1000, random_state=42),
        SGDClassifier(random_state=42),
        KNeighborsClassifier()
    ]
    
    clf_names = [
        "LDA", "QDA", "GaussianNB", "LogisticRegression", 
        "SGDClassifier", "KNeighbors"
    ]
    
    # Create figure for all classifiers on all datasets
    n_datasets = len(datasets)
    n_classifiers = len(clsfs)
    
    fig = plt.figure(figsize=(5 * n_datasets, 3 * n_classifiers))
    
    all_results = []
    
    for ds_idx, (X, Y, ds_name) in enumerate(datasets):
        print(f"\nProcessing dataset: {ds_name}")
        
        for clf_idx, (clf, clf_name) in enumerate(zip(clsfs, clf_names)):
            # Create fresh classifier instance
            if clf_name == "LDA":
                clf_inst = LinearDiscriminantAnalysis()
            elif clf_name == "QDA":
                clf_inst = QuadraticDiscriminantAnalysis()
            elif clf_name == "GaussianNB":
                clf_inst = GaussianNB()
            elif clf_name == "LogisticRegression":
                clf_inst = LogisticRegression(max_iter=1000, random_state=42)
            elif clf_name == "SGDClassifier":
                clf_inst = SGDClassifier(random_state=42)
            elif clf_name == "KNeighbors":
                clf_inst = KNeighborsClassifier()
            
            # Fit and predict
            clf_inst.fit(X, Y)
            y_pred = clf_inst.predict(X)
            
            # Calculate metrics
            if hasattr(clf_inst, 'predict_proba'):
                y_score = clf_inst.predict_proba(X)[:, 1]
            elif hasattr(clf_inst, 'decision_function'):
                y_score = clf_inst.decision_function(X)
            else:
                y_score = y_pred
            
            try:
                auc = metrics.roc_auc_score(Y, y_score)
            except:
                auc = float('nan')
            
            accuracy = metrics.accuracy_score(Y, y_pred)
            f1 = metrics.f1_score(Y, y_pred, zero_division=0)
            precision = metrics.precision_score(Y, y_pred, zero_division=0)
            recall = metrics.recall_score(Y, y_pred, zero_division=0)
            
            all_results.append({
                'dataset': ds_name,
                'classifier': clf_name,
                'accuracy': accuracy,
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'misclassified': int((Y != y_pred).sum()),
                'total': len(Y)
            })
            
            # Create subplot
            ax = fig.add_subplot(n_classifiers, n_datasets, clf_idx * n_datasets + ds_idx + 1)
            ax.scatter(X[:, 0], X[:, 1], marker='o', c=Y,
                       s=25, edgecolor='k', cmap=plt.cm.Paired)
            
            try:
                colorplot(clf_inst, ax, X[:, 0], X[:, 1])
            except:
                pass
            
            title = f"{clf_name}\nMisclass: {(Y != y_pred).sum()}/{len(Y)}"
            ax.set_title(title)
    
    # Save figure
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "comparison_decision_boundaries.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot to: {plot_path}")
    
    # Save comparison metrics
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, "comparison_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved comparison metrics to: {csv_path}")
    
    return results_df


def main():
    """
    Main function to run the complete analysis.
    """
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Load GIST data
    X_pca, y, X_full, pca, scaler, imputer = load_gist_data()
    
    # Run classifiers on GIST dataset
    clfs_fit, gist_results = run_classifiers(X_pca, y, output_dir, "GIST")
    
    # Run comparison analysis with synthetic datasets
    comparison_results = run_comparison_analysis(X_pca, y, X_full, output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - GIST_decision_boundaries.png: Decision boundaries for all classifiers on GIST data")
    print("  - GIST_metrics.csv: Evaluation metrics for all classifiers on GIST data")
    print("  - comparison_decision_boundaries.png: All classifiers on all datasets")
    print("  - comparison_metrics.csv: Metrics for comparison analysis")
    
    # Print best classifier for GIST
    gist_df = pd.DataFrame(gist_results)
    best_idx = gist_df['accuracy'].idxmax()
    best_clf = gist_df.loc[best_idx]
    print(f"\nBest classifier for GIST (by accuracy):")
    print(f"  {best_clf['classifier']} with accuracy={best_clf['accuracy']:.4f}, AUC={best_clf['auc']:.4f}")
    
    best_auc_idx = gist_df['auc'].idxmax()
    best_auc_clf = gist_df.loc[best_auc_idx]
    print(f"\nBest classifier for GIST (by AUC):")
    print(f"  {best_auc_clf['classifier']} with AUC={best_auc_clf['auc']:.4f}, accuracy={best_auc_clf['accuracy']:.4f}")


if __name__ == "__main__":
    main()


# ============== ADDITIONAL INTERACTIVE PLOTTING FUNCTIONS ==============

def create_interactive_scatter_plot(X, y, title, output_path):
    """
    Create an interactive scatter plot using Plotly.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping interactive plot")
        return
    
    df = pd.DataFrame({
        'PC1': X[:, 0],
        'PC2': X[:, 1],
        'Label': ['GIST' if label == 1 else 'non-GIST' for label in y]
    })
    
    fig = px.scatter(
        df, x='PC1', y='PC2', color='Label',
        title=title,
        color_discrete_map={'GIST': 'red', 'non-GIST': 'blue'},
        hover_data={'PC1': ':.3f', 'PC2': ':.3f', 'Label': True}
    )
    
    fig.update_layout(
        width=800, height=600,
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2'
    )
    
    fig.write_html(output_path)
    print(f"Interactive scatter plot saved to: {output_path}")


def create_interactive_decision_boundaries(X, y, clf, clf_name, title, output_path):
    """
    Create an interactive decision boundary plot using Plotly.
    """
    if not PLOTLY_AVAILABLE:
        return
    
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Get predictions for meshgrid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    if hasattr(clf, "predict_proba"):
        Z = clf.predict_proba(mesh_points)[:, 1]
    elif hasattr(clf, "decision_function"):
        Z = clf.decision_function(mesh_points)
    else:
        Z = clf.predict(mesh_points)
    
    Z = Z.reshape(xx.shape)
    
    # Create figure
    fig = go.Figure()
    
    # Add decision boundary as heatmap
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        colorscale='RdBu_r',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        colorbar=dict(title='Probability'),
        name='Decision Boundary'
    ))
    
    # Add scatter points
    gist_mask = y == 1
    non_gist_mask = y == 0
    
    fig.add_trace(go.Scatter(
        X=X[ gist_mask, 0], y=X[ gist_mask, 1],
        mode='markers',
        marker=dict(color='red', size=10, symbol='circle', line=dict(color='black', width=1)),
        name='GIST'
    ))
    
    fig.add_trace(go.Scatter(
        X=X[non_gist_mask, 0], y=X[non_gist_mask, 1],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='circle', line=dict(color='black', width=1)),
        name='non-GIST'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        width=900, height=700,
        legend=dict(x=0.02, y=0.98)
    )
    
    fig.write_html(output_path)
    print(f"Interactive decision boundary saved to: {output_path}")


def create_pca_explained_variance_plot(pca, output_path):
    """
    Create interactive PCA explained variance plot.
    """
    if not PLOTLY_AVAILABLE:
        return
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    fig = go.Figure()
    
    # Individual variance
    fig.add_trace(go.Bar(
        x=np.arange(1, len(pca.explained_variance_ratio_) + 1),
        y=pca.explained_variance_ratio_,
        name='Individual'
    ))
    
    # Cumulative variance
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(cumsum) + 1),
        y=cumsum,
        mode='lines+markers',
        name='Cumulative'
    ))
    
    fig.update_layout(
        title='PCA Explained Variance Ratio',
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance Ratio',
        width=800, height=500
    )
    
    fig.write_html(output_path)
    print(f"PCA variance plot saved to: {output_path}")


def create_confusion_matrix_plot(y_true, y_pred, clf_name, output_path):
    """
    Create interactive confusion matrix plot.
    """
    if not PLOTLY_AVAILABLE:
        return
    
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['non-GIST', 'GIST'],
        y=['non-GIST', 'GIST'],
        colorscale='Blues',
        text=cm,
        texttemplate='%d',
        textfont={"size": 20}
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {clf_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500, height=400
    )
    
    fig.write_html(output_path)
    print(f"Confusion matrix saved to: {output_path}")


def show_interactive_plots(X, y, output_dir):
    """
    Generate all interactive plots.
    """
    print("\n" + "="*60)
    print("Creating Interactive Plots")
    print("="*60)
    
    if not PLOTLY_AVAILABLE:
        print("Plotly is not installed. Install with: pip install plotly")
        print("Showing matplotlib interactive window instead...")
        # Fallback to matplotlib interactive
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired', s=50, edgecolor='k')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('GIST Data - Interactive Scatter Plot')
        plt.colorbar(scatter, ax=ax, label='Class (0=non-GIST, 1=GIST)')
        plt.show()
        return
    
    # Load data for interactive plots
    X_pca, y, X_full, pca, scaler, imputer = load_gist_data()
    
    # 1. Interactive scatter plot
    create_interactive_scatter_plot(
        X_pca, y, 'GIST Radiomic Features (PCA)',
        os.path.join(output_dir, 'interactive_scatter.html')
    )
    
    # 2. PCA explained variance
    create_pca_explained_variance_plot(pca, os.path.join(output_dir, 'pca_variance.html'))
    
    # 3. Interactive decision boundaries for each classifier
    classifiers = {
        'KNeighbors': KNeighborsClassifier(),
        'LDA': LinearDiscriminantAnalysis(),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'GaussianNB': GaussianNB()
    }
    
    for clf_name, clf in classifiers.items():
        clf.fit(X_pca, y)
        create_interactive_decision_boundaries(
            X_pca, y, clf, clf_name,
            f'Decision Boundary - {clf_name}',
            os.path.join(output_dir, f'interactive_boundary_{clf_name}.html')
        )
    
    print("\nAll interactive plots saved to:", output_dir)

