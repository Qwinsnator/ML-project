import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from load_data import load_data


RANDOM_STATE = 42
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gist_best_model.joblib")
CV_RESULTS_PATH = os.path.join(MODEL_DIR, "cv_results_top.json")


def prepare_data(df: pd.DataFrame):
    # Expect 'label' column with strings 'GIST' and 'non-GIST'
    if 'label' not in df.columns:
        raise ValueError("The dataset must contain a 'label' column.")

    y_raw = df['label']
    X = df.drop(columns=['label'])

    # Ensure numeric features only (drop any non-numeric if present)
    X = X.select_dtypes(include=[np.number])

    # Encode labels
    mapping = {"GIST": 1, "non-GIST": 0}
    if set(y_raw.unique()) - set(mapping.keys()):
        raise ValueError(f"Unexpected labels found: {set(y_raw.unique())}. Expected {set(mapping.keys())}.")
    y = y_raw.map(mapping).astype(int).values

    return X, y, mapping


def build_pipeline():
    # Base preprocessing and feature selection + classifier placeholder
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("var", VarianceThreshold(threshold=0.0)),
        ("select", SelectKBest(score_func=f_classif, k=50)),
        ("clf", LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    # Define model search space using multiple estimators
    param_grid = [
        {
            "select__k": [10, 25, 50, 100, "all"],
            "clf": [LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE)],
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"],
        },
        {
            "select__k": [10, 25, 50, 100, "all"],
            "clf": [LinearSVC(dual=False, class_weight="balanced", random_state=RANDOM_STATE, max_iter=10000)],
            "clf__C": [0.01, 0.1, 1.0, 10.0],
        },
        {
            "select__k": [25, 50, 100],
            "clf": [RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")],
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5, 10],
        },
    ]

    return pipe, param_grid


def fit_and_evaluate(X: pd.DataFrame, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe, param_grid = build_pipeline()

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=skf,
        refit=True,
        verbose=1,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    # Scores for ROC-AUC
    if hasattr(best_model.named_steps['clf'], "predict_proba"):
        y_score = best_model.predict_proba(X_test)[:, 1]
    elif hasattr(best_model.named_steps['clf'], "decision_function"):
        y_score = best_model.decision_function(X_test)
    else:
        # Fallback: use predictions (not ideal for ROC-AUC)
        y_score = y_pred

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
    }

    # Feature names after selection
    try:
        support_mask = best_model.named_steps["select"].get_support()
        selected_features = list(X.columns[support_mask])
    except Exception:
        selected_features = list(X.columns)  # If selection step not present

    metrics["selected_features"] = selected_features

    # Save model and top CV results
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    # Save top N CV results sorted by mean_test_score
    results = pd.DataFrame(grid.cv_results_).sort_values(by="mean_test_score", ascending=False)
    top = results.head(10)[[
        "mean_test_score", "std_test_score", "mean_train_score", "std_train_score", "param_select__k", "params"
    ]]
    with open(CV_RESULTS_PATH, "w") as f:
        json.dump(json.loads(top.to_json(orient="records")), f, indent=2)

    return best_model, metrics


def main():
    print("Loading data...")
    df = load_data()
    print(f"Data shape: {df.shape}")
    print(f"Columns: {len(df.columns)} | Missing values: {df.isna().sum().sum()}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    X, y, mapping = prepare_data(df)
    print(f"Feature matrix shape: {X.shape}; Target shape: {y.shape}")

    best_model, metrics = fit_and_evaluate(X, y)

    print("\n===== Evaluation on held-out test set =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(np.array(metrics["confusion_matrix"]))

    print("\nClassification report:")
    # Pretty print the main stats
    report_df = pd.DataFrame(metrics["classification_report"]).T
    print(report_df)

    print("\nBest CV params:")
    print(metrics["best_params"])
    print(f"Best CV ROC-AUC: {metrics['best_cv_score']:.4f}")

    print(f"\nSelected features ({len(metrics['selected_features'])}):")
    print(metrics["selected_features"][:50])
    if len(metrics["selected_features"]) > 50:
        print("... (truncated)")

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved top CV results to: {CV_RESULTS_PATH}")


if __name__ == "__main__":
    main()
