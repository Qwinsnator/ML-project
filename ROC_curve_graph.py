#import libraries 
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def load_data(): #function to load training data from CSV file, select only numeric features, and encode labels as binary values (1 for GIST, 0 for non-GIST)
    df = pd.read_csv("GIST_Train.csv")
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number])
    y = df["label"].map({"GIST": 1, "non-GIST": 0}).astype(int).values
    return X, y


def plot_roc_auc_curves(X, y, cv, output_path):  #function to plot ROC AUC curves for multiple classifiers with hyperparameter tuning using GridSearchCV, and save the figure to the specified output path
    classifiers = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
            {
                "model__penalty": ["l1", "l2"],
                "model__C": [0.001, 0.01, 0.1],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "SVM": (
            SVC(random_state=RANDOM_STATE),
            [
                {
                    "model__kernel": ["linear"],
                    "model__C": [0.001, 0.01, 0.1, 1],
                    "model__class_weight": [None],
                },
                {
                    "model__kernel": ["rbf"],
                    "model__C": [0.01, 0.1, 1, 10],
                    "model__gamma": [0.001, 0.01, 0.1],
                    "model__class_weight": [None],
                },
            ],
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [5, 10, 15, 20, 30],
                "model__min_samples_split": [2, 3, 4, 5, 10],
                "model__min_samples_leaf": [1, 2, 5, 7],
                "model__splitter": ["random"],
                "model__ccp_alpha": [0.0005, 0.001, 0.005, 0.01],
                "model__class_weight": [None],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 250, 300, 350],
                "model__max_depth": [None, 5, 10, 15, 20],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
                "model__bootstrap": [True, False],
                "model__class_weight": [None],
            },
        ),
    }

    _, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classifiers)))

    for idx, (name, (model, params)) in enumerate(classifiers.items()):
        print(f"Tuning {name} for ROC AUC...")

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", model),
            ]
        )

        search = GridSearchCV( 
            estimator=pipeline,
            param_grid=params,
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv,
        )
        search.fit(X, y)
        best_model = search.best_estimator_

        if hasattr(best_model.named_steps["model"], "predict_proba"):
            y_score = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba")[:, 1]
        elif hasattr(best_model.named_steps["model"], "decision_function"):
            y_score = cross_val_predict(best_model, X, y, cv=cv, method="decision_function")
        else:
            y_score = cross_val_predict(best_model, X, y, cv=cv, method="predict")

        fpr, tpr, _ = roc_curve(y, y_score)
        auc_score = roc_auc_score(y, y_score)
        ax.plot(fpr, tpr, lw=2, color=colors[idx], label=f"{name} (AUC={auc_score:.3f})")
        print(f"  Best ROC AUC (CV): {search.best_score_:.4f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Tuned Classifiers")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve figure: {output_path}")


def main(): #main function to load data, define cross-validation strategy, and call the function to plot ROC AUC curves for multiple classifiers with hyperparameter tuning using GridSearchCV, and save the figure to the specified output path
    os.makedirs("results_grid", exist_ok=True)
    X, y = load_data()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    plot_roc_auc_curves(X, y, cv, "results_grid/combined_roc_curves.png")


if __name__ == "__main__":
    main()

