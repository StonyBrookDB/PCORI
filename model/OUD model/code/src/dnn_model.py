"""
PyTorch MLP with a scikit-learn-compatible interface (fit / predict / predict_proba).

Design notes — see note/model_dnn.txt for the rationale and history.

Implementation summary
----------------------
- Pre-densify X to a uint8 torch tensor on the GPU once per fit()/predict(),
  then iterate by torch indexing (no DataLoader, no per-row sparse slicing).
- Mixed precision training with bfloat16 autocast (RTX 5090 native).
- Cosine learning-rate annealing across epochs for stable convergence at a
  fixed epoch count.
- torch.compile is attempted with graceful fallback if unavailable.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def _to_gpu_uint8(X, device: str) -> torch.Tensor:
    """Convert scipy.sparse (or numpy) matrix to a dense uint8 torch tensor on device.

    Binary ICD-10 indicators fit in uint8; we cast to float per minibatch
    inside fit()/predict() to avoid holding the full dataset in float32.
    """
    if sp.issparse(X):
        X_dense = X.toarray().astype(np.uint8, copy=False)
    else:
        X_dense = np.asarray(X, dtype=np.uint8)
    return torch.from_numpy(X_dense).to(device)


class DNNClassifier:
    """Sklearn-style wrapper. See note/model_dnn.txt for hyperparameter rationale."""

    def __init__(
        self,
        hidden=(512, 256, 128),
        dropout=0.3,
        lr=1e-3,
        epochs=15,
        batch_size=32768,
        weight_decay=1e-5,
        class_weight="balanced",
        device=None,
        random_state=42,
        use_amp=True,
        amp_dtype=torch.bfloat16,
        use_compile=True,
        verbose=True,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.use_amp = use_amp and self.device == "cuda"
        self.amp_dtype = amp_dtype
        self.use_compile = use_compile
        self.verbose = verbose
        self.model_ = None

    def get_params(self, deep=True):
        return {
            "hidden": self.hidden, "dropout": self.dropout, "lr": self.lr,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "weight_decay": self.weight_decay, "class_weight": self.class_weight,
            "device": self.device, "random_state": self.random_state,
            "use_amp": self.use_amp, "amp_dtype": self.amp_dtype,
            "use_compile": self.use_compile, "verbose": self.verbose,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _maybe_compile(self, model):
        if not self.use_compile:
            return model
        try:
            compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)
            if self.verbose:
                print("      [DNN] using torch.compile (reduce-overhead)", flush=True)
            return compiled
        except Exception as e:
            if self.verbose:
                print(f"      [DNN] torch.compile failed, eager mode: {e}", flush=True)
            return model

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        y_np = np.asarray(y).astype(np.float32)
        n_samples = y_np.shape[0]
        in_dim = X.shape[1]

        # One-shot densify to GPU as uint8
        X_gpu = _to_gpu_uint8(X, self.device)
        y_gpu = torch.from_numpy(y_np).to(self.device)

        self.model_ = _MLP(in_dim, self.hidden, self.dropout).to(self.device)
        self.model_ = self._maybe_compile(self.model_)

        # pos_weight up-weights the rare OUD class in the BCE loss
        if self.class_weight == "balanced":
            n_pos = max(int(y_np.sum()), 1)
            n_neg = n_samples - n_pos
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=self.device)
        else:
            pos_weight = None

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        self.model_.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples, device=self.device)
            total_loss = 0.0
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = X_gpu[idx].float()
                yb = y_gpu[idx]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype,
                                    enabled=self.use_amp):
                    logits = self.model_(xb)
                    loss = criterion(logits, yb)
                # bfloat16 has wider dynamic range than fp16, no GradScaler needed
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            scheduler.step()
            if self.verbose:
                lr_now = scheduler.get_last_lr()[0]
                print(f"      [DNN] epoch {epoch+1:2d}/{self.epochs}  "
                      f"loss={total_loss/n_samples:.5f}  lr={lr_now:.2e}",
                      flush=True)

        # Free training tensors so predict() can re-allocate for X_test
        del X_gpu, y_gpu
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def _scores(self, X) -> np.ndarray:
        X_gpu = _to_gpu_uint8(X, self.device)
        n = X_gpu.shape[0]
        self.model_.eval()
        out_chunks = []
        for start in range(0, n, self.batch_size):
            xb = X_gpu[start:start + self.batch_size].float()
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype,
                                enabled=self.use_amp):
                logits = self.model_(xb)
            out_chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
        del X_gpu
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return np.concatenate(out_chunks)

    def predict_proba(self, X) -> np.ndarray:
        p = self._scores(X)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self._scores(X) >= 0.5).astype(np.int8)
