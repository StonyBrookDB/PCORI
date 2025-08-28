#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from .registry import register

@register("dt")
class SkDecisionTree:
    """
    Decision tree baseline. Input (N,T,D) is automatically flattened to (N, T*D).
    """
    def __init__(self, max_depth: int = None, class_weight: str = "balanced", random_state: int = 42):
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.clf = DecisionTreeClassifier(
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state
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
        # DecisionTreeClassifier support predict_proba
        return self.pipe.predict_proba(Xf)[:, 1]
