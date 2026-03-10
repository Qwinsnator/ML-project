"""
GIST vs Non-GIST Classification Pipeline
=========================================
This script classifies GIST (Gastrointestinal Stromal Tumor) from radiomic features
using various machine learning classifiers.

Pipeline Steps:
1. Load data and check for missing values
2. Scale features to [0,1] using MinMaxScaler
3. Feature selection using RFECV (Recursive Feature Elimination with Cross-Validation)
4. Train and evaluate classifiers using 5-fold cross-validation

Based on code from Excercises E1.2, E1.4, and E3.1
"""

# Import required libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Preprocessing and feature selection
from sklearn.preprocessing import MinMaxScaler
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import svm

# Metrics
from sklearn import metrics

# Classifiers (from E1.2 Basic Classifiers)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Random Forest (from E3.1)
from sklearn.ensemble import RandomForestClassifier

# Import data loading function
from load_data import load_data


def preprocess_data():
    """
    Load and preprocess the GIST radiomic features dataset.
    
    Steps:
    1. Load data
    2. Check for missing values
    3. Encode labels (GIST=1, non-GIST=0)
    4. Scale features to [0,1] using MinMaxScaler
    5. Feature selection using RFECV
    
    Returns:
        X_selected: Selected features after RFECV
        y: Target labels
        scaler: Fitted MinMaxScaler
        rfecv: Fitted RFECV object
    """
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    
    # Load data using the provided load_data function
    df = load_data()
    
    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Total cells: {df.size}")
    
    # Check for missing values
    missing_total = df.isna().sum().sum()
    print(f"Missing values total: {missing_total}")
    
    # Check class distribution
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    
    # Extract features and labels
    y_raw = df['label']
    X = df.drop(columns=['label'])
    
    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    
    # Encode labels: GIST=1, non-GIST=0
    mapping = {'GIST': 1, 'non-GIST': 0}
    y = y_raw.map(mapping).astype(int).values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Class distribution: GIST={np.sum(y==1)}, non-GIST={np.sum(y==0)}")
    
    # =========================================================================
    # STEP 2: Scale features to [0,1] using MinMaxScaler
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Scaling features to [0,1] using MinMaxScaler")
    print("=" * 60)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Data scaled to [0,1] range")
    print(f"Min value after scaling: {X_scaled.min():.4f}")
    print(f"Max value after scaling: {X_scaled.max():.4f}")
    
    # =========================================================================
    # STEP 3: Feature selection using RFECV
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Feature selection using RFECV")
    print("=" * 60)
    print("Running RFECV (this may take a while with 493 features)...")
    
    # Create the RFE object with SVC (as shown in E1.4)
    # RFECV finds the optimal number of features by cross-validation
    svc = svm.SVC(kernel='linear')
    
    # RFECV with 4-fold stratified cross-validation
    # Scoring uses ROC AUC
    rfecv = feature_selection.RFECV(
        estimator=svc,
        step=1,
        cv=model_selection.StratifiedKFold(4),
        scoring='roc_auc'
    )
    
    # Fit RFECV on the scaled data
    rfecv.fit(X_scaled, y)
    
    # Transform data to keep only selected features
    X_selected = rfecv.transform(X_scaled)
    
    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Selected features shape: {X_selected.shape}")
    
    return X_selected, y, scaler, rfecv


def evaluate_classifiers(X, y):
    """
    Train and evaluate classifiers using 5-fold cross-validation.
    
    Args:
        X: Feature matrix (selected features)
        y: Target labels
    
    Returns:
        results_df: DataFrame with evaluation metrics for each classifier
    """
    print("\n" + "=" * 60)
    print("STEP 4: Training and evaluating classifiers")
    print("=" * 60)
    print("Using 5-fold Stratified Cross-Validation\n")
    
    # Define classifiers (from E1.2 and E3.1)
    classifiers = {
        # LDA with shrinkage to handle high-dimensional data
        'LDA': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
        
        # Gaussian Naive Bayes
        'GaussianNB': GaussianNB(),
        
        # Logistic Regression
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        
        # Stochastic Gradient Descent Classifier
        'SGDClassifier': SGDClassifier(random_state=42),
        
        # K-Nearest Neighbors
        'KNeighbors': KNeighborsClassifier(),
        
        # Decision Tree
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        
        # Random Forest (from E3.1)
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Cross-validation with 5 folds
    cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    results = []
    
    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        
        # Get CV predictions (each sample is predicted when it's in test fold)
        y_pred = model_selection.cross_val_predict(clf, X, y, cv=cv)
        
        # Get probability scores for AUC calculation
        # Use predict_proba if available, otherwise use predictions
        try:
            if hasattr(clf, 'predict_proba'):
                y_score = model_selection.cross_val_predict(
                    clf, X, y, cv=cv, method='predict_proba'
                )[:, 1]
            elif hasattr(clf, 'decision_function'):
                y_score = model_selection.cross_val_predict(
                    clf, X, y, cv=cv, method='decision_function'
                )
            else:
                y_score = y_pred
        except:
            y_score = y_pred
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(y, y_pred)
        auc = metrics.roc_auc_score(y, y_score)
        f1 = metrics.f1_score(y, y_pred)
        precision = metrics.precision_score(y, y_pred)
        recall = metrics.recall_score(y, y_pred)
        misclassified = (y != y_pred).sum()
        
        # Store result
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
        
        # Print results
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   AUC:       {auc:.4f}")
        print(f"   F1:        {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   Misclassified: {misclassified}/{len(y)}")
        print()
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    return results_df


def main():
    """
    Main function to run the complete classification pipeline.
    """
    # Step 1 & 2: Load data and preprocess
    X_selected, y, scaler, rfecv = preprocess_data()
    
    # Step 3: Evaluate classifiers
    results_df = evaluate_classifiers(X_selected, y)
    
    # Save results to CSV
    output_path = 'results/GIST_cv_metrics.csv'
    results_df.to_csv(output_path, index=False)
    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print("=" * 60)
    
    # Print summary sorted by AUC
    print("\n=== SUMMARY (sorted by AUC) ===")
    results_df_sorted = results_df.sort_values('auc', ascending=False)
    print(results_df_sorted.to_string(index=False))
    
    # Identify best classifier
    best_idx = results_df['auc'].idxmax()
    best_clf = results_df.loc[best_idx]
    print(f"\n=== BEST CLASSIFIER (by AUC) ===")
    print(f"  {best_clf['classifier']} with AUC={best_clf['auc']:.4f}")
    
    return results_df


if __name__ == "__main__":
    # Run the pipeline
    results = main()

