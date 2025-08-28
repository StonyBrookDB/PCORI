#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .registry import register

class AttnPool(nn.Module):
    # score = Linear(h_t)->(B,T,1)
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1, bias=False)

    def forward(self, H):  # H: (B,T,Hdim)
        scores = self.proj(H)              # (B,T,1)
        w = torch.softmax(scores, dim=1)   # (B,T,1)
        ctx = (w * H).sum(dim=1)           # (B,Hdim)
        return ctx, w

@register("lstm_attn")
class LSTMAttnClassifier(nn.Module):
    """
    LSTM + attention readout binary sequence classifier。input (B,T,D)。
    fit(X,y)、predict_proba(X)；same as the base LSTM class.
    """
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.1, bidirectional: bool = False,
                 lr: float = 1e-3, batch_size: int = 128, epochs: int = 10,
                 device: str = None, use_amp: bool = False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.use_amp = bool(use_amp)

        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lstm = nn.LSTM(
            input_size=self.input_dim, hidden_size=self.hidden_size,
            num_layers=self.num_layers, batch_first=True,
            dropout=(self.dropout if self.num_layers > 1 else 0.0),
            bidirectional=self.bidirectional
        )
        out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.attn = AttnPool(out_dim)
        self.fc = nn.Linear(out_dim, 1)

        self.is_torch = True
        self._optimizer = None
        self._pos_weight = None

    def forward(self, x):  # x: (B,T,D)
        H, _ = self.lstm(x)              # (B,T,Hdim)
        ctx, w = self.attn(H)            # (B,Hdim)
        logit = self.fc(ctx).squeeze(-1) # (B,)
        return logit

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        n_pos = float((y > 0.5).sum()); n_neg = float((y <= 0.5).sum())
        pos_weight = (n_neg / max(n_pos, 1.0))
        self._pos_weight = torch.tensor([pos_weight], dtype=torch.float32, device=self.device)

        self.to(self.device); self.train()
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.device.type == "cuda"))
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)

        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                self._optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    with torch.autocast("cuda"):
                        logits = self.forward(xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                    scaler.step(self._optimizer); scaler.update()
                else:
                    logits = self.forward(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                    self._optimizer.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        dev = next(self.parameters()).device
        x_tensor = torch.from_numpy(X).to(dev)
        probs = []
        for i in range(0, x_tensor.size(0), self.batch_size):
            xb = x_tensor[i:i+self.batch_size]
            p = torch.sigmoid(self.forward(xb))
            probs.append(p.detach().cpu())
        return torch.cat(probs, dim=0).numpy()

    def save_checkpoint(self, path: str):
        ckpt = {
            "init_args": {
                "input_dim": self.input_dim, "hidden_size": self.hidden_size,
                "num_layers": self.num_layers, "dropout": self.dropout,
                "bidirectional": self.bidirectional, "lr": self.lr,
                "batch_size": self.batch_size, "epochs": self.epochs, "use_amp": self.use_amp
            },
            "state_dict": self.state_dict(),
        }
        torch.save(ckpt, path)

    @staticmethod
    def load_checkpoint(path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        model = LSTMAttnClassifier(**ckpt["init_args"])
        model.load_state_dict(ckpt["state_dict"])
        # 同步 device
        model.device = next(model.parameters()).device
        return model
