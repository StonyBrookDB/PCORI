#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from .registry import register

@register("mlp")
class SkMLP:
    """Flatten (N,T,D) -> (N,T*D), Impute(0) -> Standardize -> small MLP."""
    def __init__(self, hidden_layer_sizes=(128, 64), alpha: float = 1e-4, random_state: int = 42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.random_state = random_state
        self.pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),  # for NaN constant padding
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=self.alpha,
                batch_size="auto",
                learning_rate_init=1e-3,
                max_iter=50,
                random_state=self.random_state,
                early_stopping=True,
                n_iter_no_change=5,
                verbose=False
            ))
        ])

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (N,T,D), got shape {X.shape}")
        N, T, D = X.shape
        return X.reshape(N, T * D)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xf = self._flatten(X)
        self.pipe.fit(Xf, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = self._flatten(X)
        proba = self.pipe.predict_proba(Xf)[:, 1]
        return proba