import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from scipy.stats import zscore

def load_gist_train_data():
    print("Loading GIST Train dataset...")
    df = pd.read_csv('GIST_Train.csv')
    print(f"Data shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    y_raw = df['label']
    X = df.drop(columns=['label']).select_dtypes(include=[np.number])
    
    y = y_raw.map({"GIST": 1, "non-GIST": 0}).astype(int).values
    print(f"Features: {X.shape}, Target: {y.shape}")
    return X, y


def preprocess_data(X):
    # Step 1: Impute
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    
    print("Outlier removal (3SD rule) on imputed data...")
    
    # Step 2: Outlier removal (3SD from mean per feature)
    n_features_outlier = 0
    n_outliers_total = 0
    n_original = X.shape[0]
    
    X_outlier_free = np.copy(X_imputed)
    for j in range(X.shape[1]):
        mean_j = np.mean(X_imputed[:, j])
        sd_j = np.std(X_imputed[:, j])
        lower = mean_j - 3 * sd_j
        upper = mean_j + 3 * sd_j
        
        outliers_mask = (X_imputed[:, j] < lower) | (X_imputed[:, j] > upper)
        n_outliers_j = np.sum(outliers_mask)
        if n_outliers_j > 0:
            n_features_outlier += 1
            n_outliers_total += n_outliers_j
            # Remove entire row if outlier in this feature
            outlier_rows = np.where(outliers_mask)[0]
            X_outlier_free[outlier_rows, :] = np.nan  # Mark for re-impute later
    
    # Step 3: Re-impute rows with any outliers (median per feature)
    X_clean = SimpleImputer(strategy="median").fit_transform(X_outlier_free)
    
    print(f"Outliers: {n_outliers_total} in {n_features_outlier}/{X.shape[1]} features")
    
    # Step 4: Robust scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Z-score for outlier
    z_scores = pd.DataFrame(zscore(X, nan_policy='omit'), columns=X.columns)
    outliers = (np.abs(z_scores) > 3) # outliers: absolute z-score > 3
    n_outliers = outliers.sum()  # aantal outliers per feature

    # percentage outliers per feature
    perc_outliers = n_outliers / len(X) * 100

    # combineer in dataframe
    outlier_df = pd.DataFrame({
        "feature": X.columns,
        "n_outliers": n_outliers,
        "percentage": perc_outliers
    })

    print(f"Total outliers: {n_outliers.sum()}")
    print(outlier_df[['feature', 'percentage']].sort_values("percentage", ascending=False).head(10))

    #skewness per feature
    skewness = 3 * (X.mean() - X.median()) / X.std()

    skew_df = pd.DataFrame({
        "feature": X.columns,
        "skewness": skewness
    }).sort_values(by="skewness", key=abs, ascending=False)

    print(skew_df.head(10))
    print("\nAverage absolute skewness:", skew_df["skewness"].abs().mean())
    
    return X_scaled, scaler, imputer


def perform_feature_selection(X, y, output_dir):
    """
    Perform RFECV feature selection with SVM linear estimator.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Feature selection using RFECV")
    print("=" * 60)
    print("This may take a moment...")

    # Scale data using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svc = SVC(kernel="linear")
    rfecv = RFECV(
        estimator=svc,
        step=1,
        cv=StratifiedKFold(5),
        scoring='roc_auc'
    )

    rfecv.fit(X_scaled, y)
    X_selected = rfecv.transform(X_scaled)

    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Optimal number of features: {rfecv.n_features_}")

    feature_names = X.columns.values
    selected_mask = rfecv.support_
    selected_features = feature_names[selected_mask]
    print(f"\nSelected feature indices: {np.where(selected_mask)[0]}")

    # Save RFECV plot (optional)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (ROC AUC)")
    plt.title("RFECV - Feature Selection Results")
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, "rfecv_feature_selection.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved RFECV plot: {plot_path}")

    # return selected matrix, selected feature names, scaler and selector
    return X_selected, selected_features, scaler, rfecv


def run_classifiers_with_cv(X, y, output_dir, n_splits=5):
    X_scaled, scaler, imputer = preprocess_data(X)
    
    # Perform feature selection on scaled data
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_selected, selected_features, scaler_fs, rfecv = perform_feature_selection(X_scaled_df, y, output_dir)

    print(f"\nRunning {n_splits}-fold CV on classifiers with {X_selected.shape[1]} selected features")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=2000, solver='liblinear', random_state=42, C=0.1),
        'SVM': SVC(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }
    
    results = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (clf_name, clf) in enumerate(classifiers.items()):
        print(f"\n{clf_name}...")
        
        acc_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='accuracy')
        recall_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='recall')
        auc_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='roc_auc')
        
        clf.fit(X_selected, y)
        train_acc = metrics.accuracy_score(y, clf.predict(X_selected))
        
        result = {
            'classifier': clf_name,
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'accuracy_mean': np.mean(acc_scores),
            'accuracy_std': np.std(acc_scores),
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'train_accuracy': train_acc
        }
        results.append(result)
        
        print(f"  Recall: {result['recall_mean']:.4f} ± {result['recall_std']:.4f}")
        print(f"  Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
        
        ax = axes[idx]
        ax.boxplot([acc_scores, recall_scores, auc_scores], tick_labels=['Acc', 'Recall', 'AUC'])
        ax.set_title(f"{clf_name} Recall={result['recall_mean']:.3f}")
        ax.set_ylim([0, 1])
    
    for idx in range(len(classifiers), 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "classifiers_recall_boxplot_rfecv.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "classifiers_recall_metrics_rfecv.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"\nSaved: {plot_path}, {csv_path}")
    return results_df


def main():
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    
    X, y = load_gist_train_data()
    
    results_df = run_classifiers_with_cv(X, y, output_dir)
    
    # RANK BY RECALL
    ranked_df = results_df.sort_values('recall_mean', ascending=False)
    print("\n" + "="*80)
    print("CLASSIFIERS RANKED BY RECALL (RFECV Feature Selection, highest first)")
    print("="*80)
    print(ranked_df[['classifier', 'recall_mean', 'recall_std', 'auc_mean']].round(4).to_string(index=False))
    
    best = ranked_df.iloc[0]
    print(f"\nTOP RANKED BY RECALL: {best['classifier']} | Recall = {best['recall_mean']:.4f} ± {best['recall_std']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()

