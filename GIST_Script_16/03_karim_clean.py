import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


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
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, scaler, imputer


def run_classifiers_with_cv(X, y, output_dir, n_splits=5):
    print(f"\nRunning {n_splits}-fold CV on classifiers")
    
    X_scaled, scaler, imputer = preprocess_data(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42, C=0.1),
        'SGDClassifier': SGDClassifier(random_state=42, max_iter=1000),
        'KNeighbors': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    }
    
    results = []
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (clf_name, clf) in enumerate(classifiers.items()):
        print(f"\n{clf_name}...")
        
        acc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        recall_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='recall')
        auc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
        
        clf.fit(X_scaled, y)
        train_acc = metrics.accuracy_score(y, clf.predict(X_scaled))
        
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
        ax.boxplot([acc_scores, recall_scores, auc_scores], labels=['Acc', 'Recall', 'AUC'])
        ax.set_title(f"{clf_name} Recall={result['recall_mean']:.3f}")
        ax.set_ylim([0, 1])
    
    for idx in range(len(classifiers), 8):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "classifiers_recall_boxplot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "classifiers_recall_metrics.csv")
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
    print("CLASSIFIERS RANKED BY RECALL (highest first)")
    print("="*80)
    print(ranked_df[['classifier', 'recall_mean', 'recall_std', 'auc_mean']].round(4).to_string(index=False))
    
    best = ranked_df.iloc[0]
    print(f"\nTOP RANKED: {best['classifier']} | Recall = {best['recall_mean']:.4f} ± {best['recall_std']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
