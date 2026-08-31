"""
Sequence-based classifiers with a scikit-learn-compatible interface
(fit / predict / predict_proba).

Models defined here:
  - LSTMClassifier
  - BiLSTMClassifier
  - AttentionClassifier      (multi-head self-attention pool over embeddings)
  - TransformerClassifier    (TransformerEncoder + CLS-token-style pool)

All four share a base class that handles:
  - Training loop with bf16 AMP, cosine LR schedule, optional torch.compile
  - pos_weight in BCEWithLogitsLoss for class imbalance
  - One-shot copy of int16 sequence arrays to the GPU, then torch indexing
  - PAD-aware mean pooling / attention masking

Input convention (from data/sequences_top{N}.npy):
  X      : (N, max_len) int16
  tokens : 0=PAD, 1=UNK, 2..vocab_size-1 = top-N feature codes
  Padding pads at the END of each row.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# Base class
# ────────────────────────────────────────────────────────────────────────────
class _SequenceClassifierBase:
    """Shared sklearn-style training loop. Subclasses define _build_model()."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_len: int = 256,
        lr: float = 1e-3,
        epochs: int = 15,
        batch_size: int = 1024,
        weight_decay: float = 1e-5,
        class_weight: str | None = "balanced",
        device: str | None = None,
        random_state: int = 42,
        use_amp: bool = True,
        amp_dtype=torch.bfloat16,
        use_compile: bool = True,
        verbose: bool = True,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len = max_len
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
            "vocab_size": self.vocab_size, "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim, "num_layers": self.num_layers,
            "dropout": self.dropout, "max_len": self.max_len, "lr": self.lr,
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

    def _build_model(self) -> nn.Module:
        raise NotImplementedError("Subclass must implement _build_model()")

    def _maybe_compile(self, model):
        if not self.use_compile:
            return model
        try:
            return torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception as e:
            if self.verbose:
                print(f"      [SEQ] torch.compile failed, eager: {e}", flush=True)
            return model

    def _to_gpu(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(X, dtype=np.int64)).to(self.device)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        y_np = np.asarray(y).astype(np.float32)
        n = y_np.shape[0]

        X_gpu = self._to_gpu(X)             # (n, max_len) int64 on GPU
        y_gpu = torch.from_numpy(y_np).to(self.device)

        self.model_ = self._build_model().to(self.device)
        self.model_ = self._maybe_compile(self.model_)

        if self.class_weight == "balanced":
            n_pos = max(int(y_np.sum()), 1)
            n_neg = n - n_pos
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=self.device)
        else:
            pos_weight = None

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        self.model_.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = X_gpu[idx]
                yb = y_gpu[idx]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype,
                                    enabled=self.use_amp):
                    logits = self.model_(xb)
                    loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * xb.size(0)
            scheduler.step()
            if self.verbose:
                lr_now = scheduler.get_last_lr()[0]
                print(f"      [{self.__class__.__name__}] epoch {epoch+1:2d}/"
                      f"{self.epochs}  loss={total/n:.5f}  lr={lr_now:.2e}",
                      flush=True)

        del X_gpu, y_gpu
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def _scores(self, X) -> np.ndarray:
        X_gpu = self._to_gpu(X)
        n = X_gpu.shape[0]
        self.model_.eval()
        out = []
        for start in range(0, n, self.batch_size):
            xb = X_gpu[start:start + self.batch_size]
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype,
                                enabled=self.use_amp):
                logits = self.model_(xb)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
        del X_gpu
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return np.concatenate(out)

    def predict_proba(self, X) -> np.ndarray:
        p = self._scores(X)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self._scores(X) >= 0.5).astype(np.int8)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over the time axis ignoring PAD positions.
    h:    (B, T, D)
    mask: (B, T) — 1 where valid, 0 where PAD
    """
    m = mask.unsqueeze(-1).to(h.dtype)
    s = (h * m).sum(dim=1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return s / denom


# ────────────────────────────────────────────────────────────────────────────
# LSTM
# ────────────────────────────────────────────────────────────────────────────
class _LSTMNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, bidir):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidir,
        )
        self.drop = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidir else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x):
        mask = (x != 0)
        h = self.embed(x)
        h, _ = self.lstm(h)
        h = _masked_mean(h, mask)
        h = self.drop(h)
        return self.head(h).squeeze(1)


class LSTMClassifier(_SequenceClassifierBase):
    def _build_model(self):
        return _LSTMNet(self.vocab_size, self.embed_dim, self.hidden_dim,
                        self.num_layers, self.dropout, bidir=False)


class BiLSTMClassifier(_SequenceClassifierBase):
    def _build_model(self):
        return _LSTMNet(self.vocab_size, self.embed_dim, self.hidden_dim,
                        self.num_layers, self.dropout, bidir=True)


# ────────────────────────────────────────────────────────────────────────────
# Attention pooling
# ────────────────────────────────────────────────────────────────────────────
class _AttentionNet(nn.Module):
    """Embedding → single multi-head self-attention layer → masked mean pool."""

    def __init__(self, vocab_size, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        mask = (x != 0)
        h = self.embed(x)
        key_padding_mask = ~mask
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.ln(h + attn_out)
        h = _masked_mean(h, mask)
        h = self.drop(h)
        return self.head(h).squeeze(1)


class AttentionClassifier(_SequenceClassifierBase):
    def __init__(self, *args, num_heads: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def _build_model(self):
        return _AttentionNet(self.vocab_size, self.embed_dim, self.num_heads,
                             self.dropout)

    def get_params(self, deep=True):
        p = super().get_params(deep)
        p["num_heads"] = self.num_heads
        return p


# ────────────────────────────────────────────────────────────────────────────
# Transformer
# ────────────────────────────────────────────────────────────────────────────
class _TransformerNet(nn.Module):
    """Embedding + sinusoidal position + N x TransformerEncoderLayer + masked mean pool."""

    def __init__(self, vocab_size, d_model, num_layers, num_heads, dropout, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.register_buffer("pos_enc", self._sinusoidal_pos(max_len, d_model), persistent=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    @staticmethod
    def _sinusoidal_pos(max_len, d_model):
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
        angle = pos / torch.pow(10000.0, (2 * (i // 2)) / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        return pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, x):
        mask = (x != 0)
        h = self.embed(x) + self.pos_enc[:, :x.size(1)]
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = _masked_mean(h, mask)
        h = self.drop(h)
        return self.head(h).squeeze(1)


class TransformerClassifier(_SequenceClassifierBase):
    def __init__(self, *args, num_heads: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def _build_model(self):
        return _TransformerNet(
            self.vocab_size, self.embed_dim, self.num_layers,
            self.num_heads, self.dropout, self.max_len,
        )

    def get_params(self, deep=True):
        p = super().get_params(deep)
        p["num_heads"] = self.num_heads
        return p
