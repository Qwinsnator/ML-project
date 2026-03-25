"""GIST classifier pipeline.

Dit script doet stap voor stap:
1) data inladen
2) opschonen en schalen
3) feature selectie (RFECV)
4) hyperparameter tuning (RandomizedSearchCV)
5) evaluatie + visualisaties

De code is opgeschoond en voorzien van uitleg zodat beginners makkelijker kunnen volgen.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, zscore
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")

RANDOM_STATE = 42

def load_gist_train_data():
    """Laad de trainset en zet labels om naar 0/1."""
    print("Loading GIST Train dataset...")
    df = pd.read_csv("GIST_Train.csv")
    print(f"Data shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    y_raw = df["label"]
    # Alleen numerieke features houden.
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number])

    y = y_raw.map({"GIST": 1, "non-GIST": 0}).astype(int).values
    print(f"Features: {X.shape}, Target: {y.shape}")
    return X, y


def preprocessing(X):
    """Impute missende waarden, behandel outliers en schaal features."""
    # 1) Vul missende waarden met mediaan.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    print("Outlier removal (3SD rule) on imputed data...")

    # 2) Markeer waarden buiten mean +/- 3*std als outlier.
    n_features_outlier = 0
    n_outliers_total = 0

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
            # Markeer volledige rij voor re-imputatie.
            outlier_rows = np.where(outliers_mask)[0]
            X_outlier_free[outlier_rows, :] = np.nan

    # 3) Vul opnieuw met mediaan nadat outliers op NaN zijn gezet.
    X_clean = SimpleImputer(strategy="median").fit_transform(X_outlier_free)

    print(f"Outliers: {n_outliers_total} in {n_features_outlier}/{X.shape[1]} features")

    # 4) Robuust schalen zodat features vergelijkbaar zijn.
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Extra diagnostiek voor inzicht: outliers en skewness per feature.
    z_scores = pd.DataFrame(zscore(X, nan_policy="omit"), columns=X.columns)
    outliers = np.abs(z_scores) > 3
    n_outliers = outliers.sum()

    perc_outliers = n_outliers / len(X) * 100

    outlier_df = pd.DataFrame({
        "feature": X.columns,
        "n_outliers": n_outliers,
        "percentage": perc_outliers
    })

    print(f"Total outliers: {n_outliers.sum()}")
    print(outlier_df[["feature", "percentage"]].sort_values("percentage", ascending=False).head(10))

    # Skewness laat zien hoe scheef de verdeling van een feature is.
    skewness = 3 * (X.mean() - X.median()) / X.std()

    skew_df = pd.DataFrame({
        "feature": X.columns,
        "skewness": skewness
    }).sort_values(by="skewness", key=abs, ascending=False)

    print(skew_df.head(10))
    print("\nAverage absolute skewness:", skew_df["skewness"].abs().mean())

    return X_scaled


def feature_selection(X, y, output_dir):
    """Selecteer automatisch de meest informatieve features met RFECV."""
    print("\n" + "=" * 60)
    print("STEP 3: Feature selection using RFECV")
    print("=" * 60)
    print("This may take a moment...")

    # Scale data using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Linear SVM werkt goed als basis-estimator voor RFECV.
    # RFECV verwijdert iteratief features en houdt de beste set over.
    svc = SVC(kernel="linear")
    rfecv = RFECV(
        estimator=svc,
        step=1,
        cv=StratifiedKFold(5),
        scoring="roc_auc",
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

    # Plot: score als functie van aantal geselecteerde features.
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

    return X_selected, selected_features


def roc_curve(classifiers, X_selected, y, cv, output_dir):
    _, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    auc_summary = {}

    for idx, (clf_name, clf) in enumerate(classifiers.items()):
        # Gebruik out-of-fold scores zodat evaluatie eerlijk blijft.
        if hasattr(clf, "predict_proba"):
            y_score = cross_val_predict(clf, X_selected, y, cv=cv, method="predict_proba")[:, 1]
        elif hasattr(clf, "decision_function"):
            y_score = cross_val_predict(clf, X_selected, y, cv=cv, method="decision_function")
        else:
            y_score = cross_val_predict(clf, X_selected, y, cv=cv, method="predict")

        fpr, tpr, _ = metrics.roc_curve(y, y_score)
        auc_score = metrics.roc_auc_score(y, y_score)
        auc_summary[clf_name] = auc_score

        ax = axes[idx]
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.3f}")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{clf_name} ROC")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    for idx in range(len(classifiers), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "classifiers_roc_auc_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved ROC curves: {plot_path}")
    return auc_summary


def hyperparameter(classifiers, X_selected, y, cv, output_dir, n_iter=100):
    """Zoek per classifier een goede hyperparameter-set met RandomizedSearchCV."""
    param_distributions = {
        "LogisticRegression": [
            {
                "C": loguniform(1e-3, 1e2),
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
                "class_weight": [None, "balanced"],
            }
        ],
        "SVM": [
            {
                "kernel": ["linear"],
                "C": loguniform(1e-3, 1e2),
                "class_weight": [None, "balanced"],
            },
            {
                "kernel": ["rbf"],
                "C": loguniform(1e-3, 1e2),
                "gamma": loguniform(1e-4, 1e0),
                "class_weight": [None, "balanced"],
            },
        ],
        "RandomForest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
            "class_weight": [None, "balanced"],
        },
        "KNN": {
            "n_neighbors": list(range(3, 21, 2)),
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        "DecisionTree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 3, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "splitter": ["best", "random"],
            "ccp_alpha": [0.0, 0.001, 0.01],
            "class_weight": [None, "balanced"],
        },
    }

    tuned_classifiers = {}
    tuning_results = []

    print("\n" + "=" * 60)
    print("STEP 4: Hyperparameter tuning with RandomizedSearchCV")
    print("=" * 60)

    # We tunen elk model apart, omdat elk model andere hyperparameters heeft.
    for clf_name, clf in classifiers.items():
        print(f"\nTuning {clf_name}...")

        search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=param_distributions[clf_name],
            n_iter=n_iter,
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv,
            random_state=RANDOM_STATE,
            refit=True,
        )
        search.fit(X_selected, y)

        tuned_classifiers[clf_name] = search.best_estimator_
        tuning_results.append({
            "classifier": clf_name,
            "best_score_cv_auc": search.best_score_,
            "best_params": str(search.best_params_),
        })

        print(f"  Best CV AUC: {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")

    tuning_df = pd.DataFrame(tuning_results)
    tuning_csv_path = os.path.join(output_dir, "classifiers_best_params_randomizedsearchcv.csv")
    tuning_df.to_csv(tuning_csv_path, index=False)
    print(f"\nSaved tuning results: {tuning_csv_path}")

    return tuned_classifiers, tuning_df


def classifier(X, y, output_dir, n_splits=5):
    """Draai de volledige ML-pipeline en sla resultaten/plots op.

    Pipeline-overzicht:
    1) preprocess_data
    2) perform_feature_selection
    3) tune_classifiers_randomized_search
    4) CV-evaluatie op getunede modellen
    5) ROC-plot + metrics export

    Args:
        X (pd.DataFrame): Feature-data.
        y (np.ndarray): Binaire labels.
        output_dir (str): Map voor outputbestanden.
        n_splits (int): Aantal folds voor CV.

    Returns:
        pd.DataFrame: Tabel met recall/accuracy/auc statistieken per classifier.
    """
    X_scaled = preprocessing(X)

    # Feature selectie op de opgeschoonde data.
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_selected, _ = feature_selection(X_scaled_df, y, output_dir)

    print(f"\nRunning {n_splits}-fold CV on classifiers with {X_selected.shape[1]} selected features")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # Basismodellen (voor tuning).
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
        "SVM": SVC(),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }

    tuned_classifiers, _ = hyperparameter(
        classifiers=classifiers,
        X_selected=X_selected,
        y=y,
        cv=cv,
        output_dir=output_dir,
        n_iter=100
    )
    
    results = []
    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (clf_name, clf) in enumerate(tuned_classifiers.items()):
        print(f"\n{clf_name}...")
        
        acc_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring="accuracy")
        recall_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring="recall")
        auc_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring="roc_auc")
        
        clf.fit(X_selected, y)
        train_acc = metrics.accuracy_score(y, clf.predict(X_selected))
        
        result = {
            "classifier": clf_name,
            "recall_mean": np.mean(recall_scores),
            "recall_std": np.std(recall_scores),
            "accuracy_mean": np.mean(acc_scores),
            "accuracy_std": np.std(acc_scores),
            "auc_mean": np.mean(auc_scores),
            "auc_std": np.std(auc_scores),
            "train_accuracy": train_acc,
        }
        results.append(result)
        
        print(f"  Recall: {result['recall_mean']:.4f} ± {result['recall_std']:.4f}")
        print(f"  Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
        
        ax = axes[idx]
        ax.boxplot([acc_scores, recall_scores, auc_scores], tick_labels=["Acc", "Recall", "AUC"])
        ax.set_title(f"{clf_name} Recall={result['recall_mean']:.3f}")
        ax.set_ylim([0, 1])
    
    for idx in range(len(tuned_classifiers), 6):
        axes[idx].axis('off')

    auc_summary = roc_curve(tuned_classifiers, X_selected, y, cv, output_dir)
    print("Out-of-fold ROC AUC per classifier:")
    for clf_name, auc_score in auc_summary.items():
        print(f"  {clf_name}: {auc_score:.4f}")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "classifiers_recall_boxplot_rfecv.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "classifiers_recall_metrics_rfecv.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"\nSaved: {plot_path}, {csv_path}")
    return results_df


def main():
    """Startpunt van het script."""
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    
    X, y = load_gist_train_data()
    
    results_df = classifier(X, y, output_dir)
    
    ranked_df = results_df.sort_values("recall_mean", ascending=False)
    print("\n" + "="*80)
    print("CLASSIFIERS RANKED BY RECALL (RFECV Feature Selection, highest first)")
    print("="*80)
    print(ranked_df[["classifier", "recall_mean", "recall_std", "auc_mean"]].round(4).to_string(index=False))

    best = ranked_df.iloc[0]
    print(f"\nTOP RANKED BY RECALL: {best['classifier']} | Recall = {best['recall_mean']:.4f} ± {best['recall_std']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()