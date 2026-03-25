import json
import os
import builtins
import unittest
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd

import train_gist_classifier as mod


class TestPrepareData(unittest.TestCase):
    def test_raises_when_label_missing(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with self.assertRaisesRegex(ValueError, "must contain a 'label'"):
            mod.prepare_data(df)

    def test_encodes_labels_and_drops_non_numeric(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [10, 20, 30],
            "txt": ["x", "y", "z"],
            "label": ["GIST", "non-GIST", "GIST"],
        })
        X, y, mapping = mod.prepare_data(df)
        # Non-numeric column should be dropped
        self.assertListEqual(list(X.columns), ["f1", "f2"])
        # Labels encoded
        np.testing.assert_array_equal(y, np.array([1, 0, 1]))
        self.assertEqual(mapping, {"GIST": 1, "non-GIST": 0})

    def test_unexpected_labels_raise(self):
        df = pd.DataFrame({
            "f": [1, 2, 3],
            "label": ["GIST", "OTHER", "non-GIST"],
        })
        with self.assertRaisesRegex(ValueError, "Unexpected labels"):
            mod.prepare_data(df)


class _SelectFake:
    def __init__(self, n_features, support_idx=None):
        self.n_features = n_features
        self.support_idx = support_idx or list(range(min(2, n_features)))

    def get_support(self):
        mask = [False] * self.n_features
        for i in self.support_idx:
            if 0 <= i < self.n_features:
                mask[i] = True
        return np.array(mask, dtype=bool)


class _ClfWithProba:
    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        # Return two-class probabilities, predict class 1 with prob 0.7
        return np.vstack([0.3 * np.ones(len(X)), 0.7 * np.ones(len(X))]).T


class _ClfWithDecFunc:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def decision_function(self, X):
        # Positive scores near zero
        return 0.1 * np.ones(len(X))


class _BestEstimatorFake:
    def __init__(self, n_features, clf):
        self.named_steps = {
            "clf": clf,
            "select": _SelectFake(n_features=n_features, support_idx=[0, 1] if n_features >= 2 else [0]),
        }

    def predict(self, X):
        return self.named_steps["clf"].predict(X)

    def predict_proba(self, X):
        # Delegate only if available
        return self.named_steps["clf"].predict_proba(X)

    def decision_function(self, X):
        return self.named_steps["clf"].decision_function(X)


class TestBuildPipeline(unittest.TestCase):
    def test_build_pipeline_structure(self):
        pipe, grid = mod.build_pipeline()
        # Ensure expected steps exist in pipeline
        step_names = [name for name, _ in pipe.steps]
        self.assertEqual(step_names[:5], ["imputer", "scaler", "var", "select", "clf"])
        # Ensure param grid contains expected keys
        self.assertTrue(any("select__k" in d for d in grid))
        self.assertTrue(any("clf" in d for d in grid))


class TestFitAndEvaluate(unittest.TestCase):
    def setUp(self):
        # Small synthetic dataset
        self.X = pd.DataFrame({
            "a": np.random.randn(40),
            "b": np.random.randn(40),
            "c": np.random.randn(40),
        })
        # Balanced binary labels
        self.y = np.array([0, 1] * 20)

        # Minimal cv_results_ content used for JSON export
        self.cv_results = {
            "mean_test_score": [0.7, 0.68],
            "std_test_score": [0.02, 0.03],
            "mean_train_score": [0.9, 0.88],
            "std_train_score": [0.01, 0.02],
            "param_select__k": [10, 25],
            "params": [{"select__k": 10}, {"select__k": 25}],
            # sklearn adds many keys; we only need the ones accessed and serializable to DataFrame
        }

    def _setup_grid_mock(self, best_estimator):
        grid_mock = MagicMock()
        grid_mock.best_estimator_ = best_estimator
        grid_mock.best_params_ = {"select__k": 10}
        grid_mock.best_score_ = 0.71
        # Pandas DataFrame creation expects dict-like; we'll create matching sklearn-like structure
        # by expanding keys over candidates. Using a dict-of-lists is fine.
        grid_mock.cv_results_ = {
            "mean_test_score": np.array(self.cv_results["mean_test_score"]),
            "std_test_score": np.array(self.cv_results["std_test_score"]),
            "mean_train_score": np.array(self.cv_results["mean_train_score"]),
            "std_train_score": np.array(self.cv_results["std_train_score"]),
            "param_select__k": np.array(self.cv_results["param_select__k"], dtype=object),
            "params": np.array(self.cv_results["params"], dtype=object),
        }
        return grid_mock

    @patch.object(mod, "joblib")
    @patch.object(mod, "GridSearchCV")
    @patch.object(mod, "os")
    def test_fit_and_evaluate_with_predict_proba(self, os_mock, grid_cls_mock, joblib_mock):
        n_features = self.X.shape[1]
        best_estimator = _BestEstimatorFake(n_features=n_features, clf=_ClfWithProba())
        grid_mock = self._setup_grid_mock(best_estimator)
        grid_cls_mock.return_value = grid_mock

        # Avoid real filesystem writes
        os_mock.path.join = os.path.join
        os_mock.path.dirname = os.path.dirname
        os_mock.path.abspath = os.path.abspath
        os_mock.makedirs = MagicMock()

        with patch.object(builtins, "open", mock_open()) as m:
            model, metrics = mod.fit_and_evaluate(self.X, self.y)

        # GridSearchCV should be constructed and fit called
        grid_cls_mock.assert_called()
        self.assertTrue(grid_mock.fit.called)

        # Metrics should include expected keys
        for key in [
            "accuracy",
            "roc_auc",
            "confusion_matrix",
            "classification_report",
            "best_params",
            "best_cv_score",
            "selected_features",
        ]:
            self.assertIn(key, metrics)

        # Selected features length equals number of features with True in support
        self.assertTrue(len(metrics["selected_features"]) >= 1)

        # Files saved
        self.assertTrue(joblib_mock.dump.called)
        # JSON file written
        self.assertTrue(m().write.called)

        # ROC-AUC should be finite and between 0 and 1
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertLessEqual(metrics["roc_auc"], 1.0)

    @patch.object(mod, "joblib")
    @patch.object(mod, "GridSearchCV")
    @patch.object(mod, "os")
    def test_fit_and_evaluate_with_decision_function(self, os_mock, grid_cls_mock, joblib_mock):
        n_features = self.X.shape[1]
        best_estimator = _BestEstimatorFake(n_features=n_features, clf=_ClfWithDecFunc())
        grid_mock = self._setup_grid_mock(best_estimator)
        grid_cls_mock.return_value = grid_mock

        # Avoid real filesystem writes
        os_mock.path.join = os.path.join
        os_mock.path.dirname = os.path.dirname
        os_mock.path.abspath = os.path.abspath
        os_mock.makedirs = MagicMock()

        with patch.object(builtins, "open", mock_open()) as m:
            model, metrics = mod.fit_and_evaluate(self.X, self.y)

        # Ensure decision_function path was used: still returns proper metrics
        self.assertIn("roc_auc", metrics)
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertLessEqual(metrics["roc_auc"], 1.0)

        # Model dump and JSON write attempted
        self.assertTrue(joblib_mock.dump.called)
        self.assertTrue(m().write.called)


if __name__ == "__main__":
    unittest.main()
