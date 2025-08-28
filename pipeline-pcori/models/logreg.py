#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .registry import register

@register("logreg")
class SkLogReg:
    """Flatten (N,T,D) -> (N, T*D), Standardize, then LogisticRegression."""
    def __init__(self, random_state: int = 42, class_weight: str = "balanced", C: float = 1.0, solver: str = "liblinear"):
        self.random_state = random_state
        self.class_weight = class_weight
        self.C = C
        self.solver = solver
        self.pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),  # for NaN constant padding
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                random_state=self.random_state,
                class_weight=self.class_weight,
                C=self.C,
                solver=self.solver,
                max_iter=200
            ))
        ])

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        # X: (N,T,D) -> (N, T*D)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (N,T,D), got shape {X.shape}")
        N, T, D = X.shape
        return X.reshape(N, T*D)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xf = self._flatten(X)
        self.pipe.fit(Xf, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = self._flatten(X)
        proba = self.pipe.predict_proba(Xf)[:, 1]  # positive class
        return proba
