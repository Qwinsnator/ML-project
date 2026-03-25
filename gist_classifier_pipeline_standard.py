"""
GIST vs Non-GIST Classification Pipeline (StandardScaler)
=========================================================
This version uses StandardScaler instead of MinMaxScaler to compare results.

Based on code from Excercises E1.2, E1.4, and E3.1
"""

# Import required libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Preprocessing and feature selection
from sklearn.preprocessing import StandardScaler  # Changed from MinMaxScaler
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import svm

# Metrics
from sklearn import metrics

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Import data loading function
from load_data import load_data


def preprocess_data():
    """
    Load and preprocess the GIST radiomic features dataset.
    Uses StandardScaler instead of MinMaxScaler.
    """
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    
    # Load data
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values total: {df.isna().sum().sum()}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    
    # Extract features and labels
    y_raw = df['label']
    X = df.drop(columns=['label'])
    X = X.select_dtypes(include=[np.number])
    
    # Encode labels: GIST=1, non-GIST=0
    mapping = {'GIST': 1, 'non-GIST': 0}
    y = y_raw.map(mapping).astype(int).values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # STEP 2: Scale using StandardScaler (z-score normalization)
    print("\n" + "=" * 60)
    print("STEP 2: Scaling using StandardScaler (z-score normalization)")
    print("=" * 60)
    
    scaler = StandardScaler()  # Changed from MinMaxScaler
    X_scaled = scaler.fit_transform(X)
    
    print(f"Mean after scaling: {X_scaled.mean():.4f}")
    print(f"Std after scaling: {X_scaled.std():.4f}")
    
    # STEP 3: Feature selection using RFECV
    print("\n" + "=" * 60)
    print("STEP 3: Feature selection using RFECV")
    print("=" * 60)
    
    svc = svm.SVC(kernel='linear')
    rfecv = feature_selection.RFECV(
        estimator=svc,
        step=1,
        cv=model_selection.StratifiedKFold(4),
        scoring='roc_auc'
    )
    rfecv.fit(X_scaled, y)
    X_selected = rfecv.transform(X_scaled)
    
    print(f"Optimal number of features: {rfecv.n_features_}")
    
    return X_selected, y, scaler, rfecv


def evaluate_classifiers(X, y):
    """
    Train and evaluate classifiers using 5-fold cross-validation.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Training and evaluating classifiers")
    print("=" * 60)
    print("Using 5-fold Stratified Cross-Validation\n")
    
    # Define classifiers
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
        print(f"Training {name}...")
        y_pred = model_selection.cross_val_predict(clf, X, y, cv=cv)
        
        try:
            if hasattr(clf, 'predict_proba'):
                y_score = model_selection.cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
            elif hasattr(clf, 'decision_function'):
                y_score = model_selection.cross_val_predict(clf, X, y, cv=cv, method='decision_function')
            else:
                y_score = y_pred
        except:
            y_score = y_pred
        
        accuracy = metrics.accuracy_score(y, y_pred)
        auc = metrics.roc_auc_score(y, y_score)
        f1 = metrics.f1_score(y, y_pred)
        precision = metrics.precision_score(y, y_pred)
        recall = metrics.recall_score(y, y_pred)
        misclassified = (y != y_pred).sum()
        
        results.append({
            'classifier': name,
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'misclassified': misclassified,
            'total': len(y)
        })
        
        print(f"   Acc={accuracy:.4f}, AUC={auc:.4f}, F1={f1:.4f}, Mis={misclassified}/{len(y)}")
    
    results_df = pd.DataFrame(results)
    return results_df


def main():
    """Main function to run the pipeline with StandardScaler."""
    X_selected, y, scaler, rfecv = preprocess_data()
    results_df = evaluate_classifiers(X_selected, y)
    
    # Save results
    output_path = 'results/GIST_cv_metrics_standard.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n=== SUMMARY (StandardScaler) ===")
    results_df_sorted = results_df.sort_values('auc', ascending=False)
    print(results_df_sorted.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    results = main()
