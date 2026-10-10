"""
Supervised classifier policy.
Trained on oracle labels from Phase 1. Predicts the best strategy per example.
"""

from __future__ import annotations
import numpy as np
import pickle
import os
from .base import Policy, extract_features, STRATEGY_NAMES
from ..strategies.base import QueryContext, StrategyResult


class ClassifierPolicy:
    """
    Multiclass classifier: features -> strategy name.
    Defaults to LogisticRegression (interpretable, fast to train on small datasets).
    Swap for GBM or MLP if you want more capacity.
    """
    name = "classifier"

    def __init__(self, clf=None):
        if clf is None:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, C=1.0)
        self.clf = clf
        self._fitted = False

    def fit(self, examples: list[QueryContext], labels: list[str]):
        """
        Train on (example, best_strategy) pairs from oracle Phase 1.

        labels: list of strategy names (from oracle.best_strategy())
        """
        X = np.stack([extract_features(ex) for ex in examples])
        y = [STRATEGY_NAMES.index(l) for l in labels]
        self.clf.fit(X, y)
        self._fitted = True
        print(f"Classifier trained on {len(examples)} examples, "
              f"classes: {self.clf.classes_}")

    def select(self, qc: QueryContext) -> str:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .select()")
        feat = extract_features(qc).reshape(1, -1)
        idx  = self.clf.predict(feat)[0]
        return STRATEGY_NAMES[idx]

    def predict_proba(self, qc: QueryContext) -> dict[str, float]:
        """Return per-strategy probability estimates."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict_proba()")
        feat  = extract_features(qc).reshape(1, -1)
        probs = self.clf.predict_proba(feat)[0]
        return {STRATEGY_NAMES[i]: float(p)
                for i, p in zip(self.clf.classes_, probs)}

    def feature_importance(self) -> dict[str, np.ndarray]:
        """
        Return per-feature importances per class.
        Only meaningful for linear models (coefficients).
        """
        from sklearn.linear_model import LogisticRegression
        if not isinstance(self.clf, LogisticRegression):
            raise NotImplementedError("feature_importance only for LogisticRegression")
        feature_names = [
            "log_context_len", "log_query_len", "context_scale",
            "norm_query_len", "is_question", "is_list_query",
            "is_specific_lookup", "context_entropy", "query_density",
            "answer_locality", "is_count_query", "is_sum_query",
            "is_majority_query", "n_records_estimate", "requires_full_scan",
        ]
        return {
            STRATEGY_NAMES[i]: dict(zip(feature_names, self.clf.coef_[i]))
            for i in range(len(self.clf.classes_))
        }

    def update(self, qc: QueryContext, chosen: str, result: StrategyResult):
        pass  # static classifier doesn't update online; use BanditPolicy for that

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "fitted": self._fitted}, f)

    @classmethod
    def load(cls, path: str) -> "ClassifierPolicy":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(clf=state["clf"])
        obj._fitted = state["fitted"]
        return obj
