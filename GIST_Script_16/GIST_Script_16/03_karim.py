import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.feature_selection import SelectKBest, f_classif

# Classifiers (from E1.2)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Import data loading function


def load_gist_train_data():
    """
    Load and preprocess the GIST Train dataset - train set (75% of data).
    """
    print("Loading GIST Train dataset (75% of data)...")
    df = pd.read_csv('GIST_Train.csv')
    print(f"Data shape: {df.shape}")
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
    
    return X, y


def preprocess_data(X):
    """
    Preprocess data: handle missing values and scale features.
    """
    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, scaler, imputer


def run_classifiers_with_cv(X, y, output_dir, n_splits=5):
    """
    Run all classifiers with cross-validation using full feature set.
    """
    print(f"\n{'='*60}")
    print(f"Running classifiers with {n_splits}-fold Cross-Validation")
    print(f"Using full feature set: {X.shape[1]} features")
    print(f"{'='*60}")
    
    # Preprocess data
    X_scaled, scaler, imputer = preprocess_data(X)
    
    # Define classifiers - QDA/LDA need regularization or feature reduction for high-dimensional data
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, C=0.1),
        'SGDClassifier': SGDClassifier(random_state=42, max_iter=1000),
        'KNeighbors': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results
    results = []
    
    # Create figure for cross-validation results
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (clf_name, clf) in enumerate(classifiers.items()):
        print(f"\nTraining {clf_name}...")
        
        # Cross-validation scores
        acc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1')
        roc_auc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
        precision_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='precision')
        recall_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='recall')
        
        # Fit on full data for visualization
        clf.fit(X_scaled, y)
        y_pred = clf.predict(X_scaled)
        
        # Calculate training metrics
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_scaled)[:, 1]
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(X_scaled)
        else:
            y_score = y_pred
        
        try:
            train_auc = metrics.roc_auc_score(y, y_score)
        except ValueError:
            train_auc = float('nan')
        
        train_acc = metrics.accuracy_score(y, y_pred)
        
        # Store result
        result = {
            'classifier': clf_name,
            'cv_accuracy_mean': np.mean(acc_scores),
            'cv_accuracy_std': np.std(acc_scores),
            'cv_f1_mean': np.mean(f1_scores),
            'cv_f1_std': np.std(f1_scores),
            'cv_auc_mean': np.mean(roc_auc_scores),
            'cv_auc_std': np.std(roc_auc_scores),
            'cv_precision_mean': np.mean(precision_scores),
            'cv_precision_std': np.std(precision_scores),
            'cv_recall_mean': np.mean(recall_scores),
            'cv_recall_std': np.std(recall_scores),
            'train_accuracy': train_acc,
            'train_auc': train_auc
        }
        results.append(result)
        
        print(f"  CV Accuracy: {result['cv_accuracy_mean']:.4f} (+/- {result['cv_accuracy_std']:.4f})")
        print(f"  CV AUC: {result['cv_auc_mean']:.4f} (+/- {result['cv_auc_std']:.4f})")
        print(f"  CV F1: {result['cv_f1_mean']:.4f} (+/- {result['cv_f1_std']:.4f})")
        
        # Plot CV scores as boxplot
        ax = axes[idx]
        cv_data = [acc_scores, f1_scores, roc_auc_scores, precision_scores, recall_scores]
        ax.boxplot(cv_data, labels=['Acc', 'F1', 'AUC', 'Prec', 'Rec'])
        ax.set_title(f"{clf_name}\nCV Acc: {result['cv_accuracy_mean']:.3f}±{result['cv_accuracy_std']:.3f}")
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
    
    # Hide unused subplots
    for idx in range(len(classifiers), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cv_results_boxplot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved CV boxplot to: {plot_path}")
    
    # Save metrics to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "cv_metrics_full.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved CV metrics to: {csv_path}")
    
    return results, scaler, imputer


def run_with_feature_selection(X, y, output_dir, n_splits=5):
    """
    Run classifiers with feature selection (top k features) and cross-validation.
    """
    print(f"\n{'='*60}")
    print("Running with Feature Selection + Cross-Validation")
    print(f"{'='*60}")
    
    # Preprocess data
    X_scaled, scaler, imputer = preprocess_data(X)
    
    # Test different numbers of features
    k_values = [10, 25, 50, 100, 'all']
    
    # Define classifiers to test
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNeighbors': KNeighborsClassifier(n_neighbors=5),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []
    
    for k in k_values:
        print(f"\nTesting with k={k} features...")
        
        # Apply feature selection
        if k == 'all':
            X_selected = X_scaled
        else:
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X_scaled, y)
        
        for clf_name, clf_template in classifiers.items():
            # Create fresh classifier instance
            if clf_name == 'LogisticRegression':
                clf = LogisticRegression(max_iter=1000, random_state=42)
            elif clf_name == 'KNeighbors':
                clf = KNeighborsClassifier(n_neighbors=5)
            elif clf_name == 'RandomForest':
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Cross-validation
            acc_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='accuracy')
            auc_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='roc_auc')
            f1_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='f1')
            
            result = {
                'n_features': k if k == 'all' else str(k),
                'classifier': clf_name,
                'cv_accuracy_mean': np.mean(acc_scores),
                'cv_accuracy_std': np.std(acc_scores),
                'cv_auc_mean': np.mean(auc_scores),
                'cv_auc_std': np.std(auc_scores),
                'cv_f1_mean': np.mean(f1_scores),
                'cv_f1_std': np.std(f1_scores)
            }
            results.append(result)
            
            print(f"  {clf_name}: Acc={result['cv_accuracy_mean']:.3f}±{result['cv_accuracy_std']:.3f}, "
                  f"AUC={result['cv_auc_mean']:.3f}±{result['cv_auc_std']:.3f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "feature_selection_cv_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved feature selection results to: {csv_path}")
    
    # Create visualization
    plot_feature_selection_results(results_df, output_dir)
    
    return results_df


def plot_feature_selection_results(results_df, output_dir):
    """
    Plot feature selection results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(['cv_accuracy_mean', 'cv_auc_mean', 'cv_f1_mean']):
        ax = axes[idx]
        
        for clf_name in results_df['classifier'].unique():
            clf_data = results_df[results_df['classifier'] == clf_name]
            x_vals = range(len(clf_data))
            ax.errorbar(x_vals, clf_data[metric], yerr=clf_data[metric.replace('_mean', '_std')], 
                       label=clf_name, marker='o', capsize=3)
        
        ax.set_xticks(range(len(results_df['n_features'].unique())))
        ax.set_xticklabels(results_df['n_features'].unique())
        ax.set_xlabel('Number of Features')
        ax.set_ylabel(metric.replace('cv_', '').replace('_mean', '').upper())
        ax.set_title(metric.replace('cv_', '').replace('_mean', ' Score'))
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "feature_selection_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature selection plot to: {plot_path}")


def create_learning_curve(X, y, output_dir):
    """
    Create learning curves for the best classifiers.
    """
    print(f"\n{'='*60}")
    print("Creating Learning Curves")
    print(f"{'='*60}")
    
    X_scaled, scaler, imputer = preprocess_data(X)
    
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNeighbors': KNeighborsClassifier(n_neighbors=5),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GaussianNB': GaussianNB()
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (clf_name, clf) in enumerate(classifiers.items()):
        print(f"\nCreating learning curve for {clf_name}...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            clf, X_scaled, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax = axes[idx]
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training')
        ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation')
        
        ax.set_xlabel('Training Size')
        ax.set_ylabel('ROC-AUC Score')
        ax.set_title(f'Learning Curve - {clf_name}')
        ax.legend()
        ax.set_ylim([0.5, 1.0])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to: {plot_path}")


def main():
    """
    Main function to run the complete analysis.
    """
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Load train data
    X, y = load_gist_train_data()
    
    # Run classifiers with cross-validation on full features
    cv_results, scaler, imputer = run_classifiers_with_cv(X, y, output_dir)
    
    # Run with feature selection + CV
    run_with_feature_selection(X, y, output_dir)
    
    # Create learning curves
    create_learning_curve(X, y, output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    # Find best model
    cv_df = pd.DataFrame(cv_results)
    best_auc_idx = cv_df['cv_auc_mean'].idxmax()
    best_model = cv_df.loc[best_auc_idx]
    
    print("\nBest classifier (by CV AUC):")
    print(f"  {best_model['classifier']}")
    print(f"  CV Accuracy: {best_model['cv_accuracy_mean']:.4f} ± {best_model['cv_accuracy_std']:.4f}")
    print(f"  CV AUC: {best_model['cv_auc_mean']:.4f} ± {best_model['cv_auc_std']:.4f}")
    print(f"  CV F1: {best_model['cv_f1_mean']:.4f} ± {best_model['cv_f1_std']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - cv_results_boxplot.png: Cross-validation results boxplot")
    print("  - cv_metrics_full.csv: All CV metrics")
    print("  - feature_selection_cv_results.csv: Feature selection results")
    print("  - feature_selection_plot.png: Feature selection visualization")
    print("  - learning_curves.png: Learning curves for best models")


if __name__ == "__main__":
    main()

