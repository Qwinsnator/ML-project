
<<<<<<< HEAD
=======
This version combines:
- StandardScaler for feature scaling (z-score normalization)
- RFECV for feature selection (Recursive Feature Elimination with CV)
- 10-fold Stratified Cross-Validation for evaluation
"""
>>>>>>> 047fc3d707f9a4f768f5e5733f85e6728a95b5f1
# Import required libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Preprocessing and feature selection
from sklearn.preprocessing import StandardScaler
from sklearn import feature_selection
from sklearn import model_selection

# Metrics
from sklearn import metrics

# Random Forest (from E3.1)
from sklearn.ensemble import RandomForestClassifier

# Classifiers (from E1.2)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Import data loading function
from load_data import load_data

def preprocess_data():
    """
    Load and preprocess the GIST radiomic features dataset.
    Uses StandardScaler + RFECV for preprocessing.
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
    
    # STEP 2: Scale using StandardScaler
    print("\n" + "=" * 60)
    print("STEP 2: Scaling using StandardScaler (z-score normalization)")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Mean after scaling: {X_scaled.mean():.4f}")
    print(f"Std after scaling: {X_scaled.std():.4f}")
    
    # STEP 3: Apply RFECV for feature selection
    print("\n" + "=" * 60)
    print("STEP 3: Feature selection using RFECV")
    print("=" * 60)
    print("This may take a moment...")
    
    # Use SVM with linear kernel for RFECV (as shown in E1.4)
    from sklearn import svm
    svc = svm.SVC(kernel="linear")
    
    rfecv = feature_selection.RFECV(
        estimator=svc, 
        step=1,
        cv=model_selection.StratifiedKFold(5),  # 5-fold for RFECV
        scoring='roc_auc'
    )
    
    rfecv.fit(X_scaled, y)
    
    X_selected = rfecv.transform(X_scaled)
    
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Optimal number of features: {rfecv.n_features_}")
    
    # Show which features were selected
    feature_names = X.columns.values
    selected_mask = rfecv.support_
    selected_features = feature_names[selected_mask]
    print(f"\nSelected feature indices: {np.where(selected_mask)[0]}")
    
    return X_selected, y, scaler, rfecv


def evaluate_classifiers(X, y):
    """
    Train and evaluate classifiers using 10-fold cross-validation.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Training and evaluating classifiers (10-fold CV)")
    print("=" * 60)
    print("Using 10-fold Stratified Cross-Validation\n")
    
    # Define classifiers
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SGDClassifier': SGDClassifier(random_state=42),
        'KNeighbors': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    # 10-fold Stratified CV
    cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        
        # Get predictions using cross-validation
        y_pred = model_selection.cross_val_predict(clf, X, y, cv=cv)
        
        # Get probability scores for AUC
        try:
            if hasattr(clf, 'predict_proba'):
                y_score = model_selection.cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
            elif hasattr(clf, 'decision_function'):
                y_score = model_selection.cross_val_predict(clf, X, y, cv=cv, method='decision_function')
            else:
                y_score = y_pred
        except Exception as e:
            print(f"  Warning: Could not get probability scores: {e}")
            y_score = y_pred
        
        # Compute metrics
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
        
        print(f"   Acc={accuracy:.4f}, AUC={auc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        print(f"   Misclassified: {misclassified}/{len(y)}")
    
    results_df = pd.DataFrame(results)
    return results_df


def main():
    """Main function to run the complete pipeline."""
    # Preprocess data with StandardScaler + RFECV
    X_selected, y, scaler, rfecv = preprocess_data()
    
    # Evaluate classifiers with 10-fold CV
    results_df = evaluate_classifiers(X_selected, y)
    
    # Save results
    output_path = 'results/GIST_final_10-03.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (StandardScaler + RFECV + 10-fold CV)")
    print("=" * 60)
    
    # Sort by AUC
    results_df_sorted = results_df.sort_values('auc', ascending=False)
    print(results_df_sorted.to_string(index=False))
    
    # Best classifier
    best_idx = results_df['auc'].idxmax()
    best_clf = results_df.loc[best_idx]
    print(f"\n*** Best classifier (by AUC): {best_clf['classifier']} ***")
    print(f"    AUC: {best_clf['auc']:.4f}")
    print(f"    Accuracy: {best_clf['accuracy']:.4f}")
    print(f"    F1: {best_clf['f1']:.4f}")
    
    return results_df


if __name__ == "__main__":
    results = main()

