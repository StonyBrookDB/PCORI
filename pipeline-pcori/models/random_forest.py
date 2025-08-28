#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from .registry import register

@register("rf")
class SkRandomForest:
    """
    RandomForest
    input (N,T,D) flat to(N, T*D)。
    without normalizate data (not required for tree models).
    """
    def __init__(self, n_estimators: int = 300, max_depth: int = None,
                 class_weight: str = "balanced_subsample", random_state: int = 42, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.pipe = Pipeline([("clf", self.clf)])

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (N,T,D), got {X.shape}")
        N, T, D = X.shape
        return X.reshape(N, T * D)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xf = self._flatten(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
        self.pipe.fit(Xf, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = self._flatten(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
        return self.pipe.predict_proba(Xf)[:, 1]
