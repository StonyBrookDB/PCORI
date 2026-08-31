"""
Central registry for all models.

Each entry is (display_name, factory, data_type) where:
  - factory(feature_name=None) returns a fresh sklearn-compatible estimator.
    Some sequence models need to know vocab_size, which depends on the
    feature set, so the factory takes the feature set name.
  - data_type ∈ {"matrix", "sequence", "graph"} tells 03_train_eval.py
    what input array to load from data/.

Factories keep heavy imports (torch, PyG) lazy: a model's heavy import only
happens when that model is actually requested.

Add new models by appending to MODEL_REGISTRY.
"""

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_DIR = Path(__file__).parent.parent / "data"


def _vocab_size_for(feature_name: str) -> int:
    """Vocab = number of codes in top{N} feature list + 2 (PAD, UNK)."""
    n_codes = len((DATA_DIR / f"{feature_name}_features.txt").read_text().strip().splitlines())
    return n_codes + 2


# ── tabular (matrix) models ───────────────────────────────────────────────
def _logistic_regression(feature_name=None):
    # See note/model_logistic_regression.txt
    return LogisticRegression(
        solver="lbfgs", max_iter=1000, class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def _decision_tree(feature_name=None):
    # See note/model_decision_tree.txt
    return DecisionTreeClassifier(
        max_depth=20, min_samples_leaf=50, class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def _random_forest(feature_name=None):
    # See note/model_random_forest.txt
    return RandomForestClassifier(
        n_estimators=50, max_depth=20, min_samples_leaf=50,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )


def _dnn(feature_name=None):
    # See note/model_dnn.txt
    from dnn_model import DNNClassifier
    return DNNClassifier(random_state=RANDOM_STATE, verbose=True)


# ── sequence models (Phase 1) ─────────────────────────────────────────────
def _lstm(feature_name):
    from seq_models import LSTMClassifier
    return LSTMClassifier(vocab_size=_vocab_size_for(feature_name),
                          random_state=RANDOM_STATE, verbose=True)


def _bilstm(feature_name):
    from seq_models import BiLSTMClassifier
    return BiLSTMClassifier(vocab_size=_vocab_size_for(feature_name),
                            random_state=RANDOM_STATE, verbose=True)


def _attention(feature_name):
    from seq_models import AttentionClassifier
    return AttentionClassifier(vocab_size=_vocab_size_for(feature_name),
                               random_state=RANDOM_STATE, verbose=True)


def _transformer(feature_name):
    from seq_models import TransformerClassifier
    return TransformerClassifier(vocab_size=_vocab_size_for(feature_name),
                                 random_state=RANDOM_STATE, verbose=True)


# ── registry ──────────────────────────────────────────────────────────────
MODEL_REGISTRY: list[tuple[str, callable, str]] = [
    # tabular
    ("Logistic Regression", _logistic_regression, "matrix"),
    ("Decision Tree",       _decision_tree,       "matrix"),
    ("Random Forest",       _random_forest,       "matrix"),
    ("DNN",                 _dnn,                 "matrix"),
    # sequence (Phase 1)
    ("LSTM",                _lstm,                "sequence"),
    ("Bi-LSTM",             _bilstm,              "sequence"),
    ("Attention",           _attention,           "sequence"),
    ("Transformer",         _transformer,         "sequence"),
    # graph (Phase 2)   — to be added
    # hybrid (Phase 3)  — to be added
]


def get_model_names() -> list[str]:
    return [name for name, _, _ in MODEL_REGISTRY]


def get_model(name: str, feature_name: str | None = None):
    for n, factory, _ in MODEL_REGISTRY:
        if n == name:
            return factory(feature_name)
    raise ValueError(f"Model '{name}' not found in registry. "
                     f"Available: {get_model_names()}")


def get_data_type(name: str) -> str:
    for n, _, data_type in MODEL_REGISTRY:
        if n == name:
            return data_type
    raise ValueError(f"Model '{name}' not found in registry. "
                     f"Available: {get_model_names()}")
