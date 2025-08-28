#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .registry import register


def _read_table(base: Path) -> pd.DataFrame:
    pqt = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    if pqt.exists():
        return pd.read_parquet(pqt)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {pqt.name} nor {csv.name} exists in {base.parent}")


def _load_spec(spec_path: Path) -> dict:
    return json.loads(Path(spec_path).read_text(encoding="utf-8"))


@register("lstm_gcn")
class LSTMWithGCN(nn.Module):
    """
    fusion model：Encounter<-Feature graph embedding， t_index align to (T,H) seq → LSTM → classification
    features: midware + feature embedding。
    """

    def __init__(
        self,
        dataset_dir: str,
        spec_path: str,
        ids_train: List[str],
        ids_val: List[str],
        ids_test: List[str],
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 8,
        device: str = None,
        use_amp: bool = False,
        normalize_rows: bool = True,
    ):
        super().__init__()
        self.dataset_dir = str(dataset_dir)
        self.spec_path = str(spec_path)
        self.ids_train = [str(x) for x in ids_train]
        self.ids_val   = [str(x) for x in ids_val]
        self.ids_test  = [str(x) for x in ids_test]
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.use_amp = bool(use_amp)
        self.normalize_rows = bool(normalize_rows)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # -------- build graph（same gcn）--------
        ddir = Path(self.dataset_dir)
        spec = _load_spec(Path(self.spec_path))
        self.T = int(spec["sequence"]["seq_len"])
        self.padding_policy = spec["sequence"].get("padding_policy", "repeat_last")

        encounters = _read_table(ddir / "encounters")
        ef = _read_table(ddir / "edges_encounter_feature")
        features = pd.read_csv(ddir / "features.csv")

        ef = ef.copy()
        ef["value"] = pd.to_numeric(ef["value"], errors="coerce").fillna(0.0)
        ef["feature_id"] = pd.to_numeric(ef["feature_id"], downcast="integer", errors="coerce").fillna(-1).astype(int)

        feat_sorted = features.sort_values(["feature_id"]).drop_duplicates("feature_id")
        self.fid2col: Dict[int, int] = {int(fid): i for i, fid in enumerate(feat_sorted["feature_id"].astype(int).tolist())}
        F = len(self.fid2col)

        enc_sorted = encounters.drop_duplicates("encounter_id")[["encounter_id","patient_id","t_index"]]
        enc_sorted["encounter_id"] = enc_sorted["encounter_id"].astype(str)
        enc_sorted["patient_id"]   = enc_sorted["patient_id"].astype(str)
        enc_sorted["t_index"]      = pd.to_numeric(enc_sorted["t_index"], errors="coerce").fillna(0).astype(int)
        self.eid2row: Dict[str, int] = {eid: i for i, eid in enumerate(enc_sorted["encounter_id"].tolist())}
        E = len(self.eid2row)

        ef = ef[ef["feature_id"].isin(self.fid2col.keys())]
        ef = ef[ef["encounter_id"].astype(str).isin(self.eid2row.keys())]
        g = ef.groupby(["encounter_id","feature_id"], as_index=False)["value"].sum()

        rows = g["encounter_id"].astype(str).map(self.eid2row).astype(int).to_numpy()
        cols = g["feature_id"].astype(int).map(self.fid2col).astype(int).to_numpy()
        vals = g["value"].astype(float).to_numpy()

        if self.normalize_rows:
            rs = np.zeros(E, dtype=np.float32)
            np.add.at(rs, rows, vals.astype(np.float32))
            rs[rs == 0.0] = 1.0
            vals = (vals / rs[rows]).astype(np.float32)
        else:
            vals = vals.astype(np.float32)

        idx = torch.tensor([rows, cols], dtype=torch.long)
        val = torch.tensor(vals, dtype=torch.float32)
        self.Inc = torch.sparse_coo_tensor(idx, val, size=(E, F)).coalesce().to(self.device)

        enc_sorted["row"] = enc_sorted["encounter_id"].map(self.eid2row)
        pid2rows: Dict[str, List[int]] = {}
        for pid, sub in enc_sorted.groupby("patient_id"):
            sub = sub.sort_values("t_index")
            rows_list = sub["row"].astype(int).tolist()
            if len(rows_list) >= self.T:
                pid2rows[pid] = rows_list[:self.T]
            else:
                if self.padding_policy == "repeat_last" and len(rows_list) > 0:
                    pad = [rows_list[-1]] * (self.T - len(rows_list))
                else:
                    pad = [-1] * (self.T - len(rows_list))
                pid2rows[pid] = rows_list + pad

        def build_rowmat(ids: List[str]) -> torch.Tensor:
            mat = np.full((len(ids), self.T), fill_value=-1, dtype=np.int64)
            for i, pid in enumerate(ids):
                mat[i, :] = pid2rows.get(pid, [-1] * self.T)
            return torch.from_numpy(mat)

        self.rowmat = {
            "train": build_rowmat(self.ids_train),
            "val":   build_rowmat(self.ids_val),
            "test":  build_rowmat(self.ids_test),
        }

        # -------- feat_emb + LSTM + FC --------
        self.feat_emb = nn.Embedding(num_embeddings=F, embedding_dim=self.hidden_size)
        nn.init.xavier_uniform_(self.feat_emb.weight)

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=(self.dropout if self.num_layers > 1 else 0.0),
            bidirectional=self.bidirectional,
        )
        out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

        self.is_torch = True
        self._optimizer = None
        self._pos_weight = None

    # ---- 构建 (B,T,H) 序列 ----
    def _encounter_embeddings(self) -> torch.Tensor:
        return torch.sparse.mm(self.Inc, self.feat_emb.weight)  # (E,H)

    def _seq_patient(self, enc_emb: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """
        enc_emb: (E,H) ; rows: (B,T) with -1 pad
        返回: X_seq: (B,T,H)
        """
        B, T = rows.shape
        H = enc_emb.size(1)
        device = enc_emb.device
        mask = (rows >= 0).to(enc_emb.dtype).unsqueeze(-1)  # (B,T,1)
        rows_clamped = rows.clone()
        rows_clamped[rows_clamped < 0] = 0
        enc_sel = enc_emb.index_select(0, rows_clamped.view(-1)).view(B, T, H) * mask
        return enc_sel

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        enc_emb = self._encounter_embeddings().to(self.device)  # (E,H)
        X_seq = self._seq_patient(enc_emb, rows.to(self.device))  # (B,T,H)
        out, _ = self.lstm(X_seq)  # (B,T,H*dir)
        last = out[:, -1, :]       # 取最后一步
        logit = self.fc(last).squeeze(-1)  # (B,)
        return logit

    # ---- 公共接口 ----
    def fit(self, X: np.ndarray, y: np.ndarray):
        y = y.astype(np.float32, copy=False)
        if len(y) == len(self.ids_train):
            split = "train"
        else:
            raise ValueError("LSTMWithGCN.fit 期望接收 train split 的 y。")

        rows = self.rowmat[split]
        ds = TensorDataset(torch.arange(len(self.ids_train)), torch.from_numpy(y))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        n_pos = float((y > 0.5).sum()); n_neg = float((y <= 0.5).sum())
        pos_weight = (n_neg / max(n_pos, 1.0))
        self._pos_weight = torch.tensor([pos_weight], dtype=torch.float32, device=self.device)

        self.to(self.device); self.train()
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.device.type == "cuda"))
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)

        for _ in range(self.epochs):
            for idx_batch, yb in dl:
                rb = rows[idx_batch.numpy(), :]
                yb = yb.to(self.device, non_blocking=True)
                self._optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    with torch.autocast("cuda"):
                        logits = self.forward(rb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                    scaler.step(self._optimizer); scaler.update()
                else:
                    logits = self.forward(rb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                    self._optimizer.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        if N == len(self.ids_val): split = "val"
        elif N == len(self.ids_test): split = "test"
        elif N == len(self.ids_train): split = "train"
        else: split = "test"
        rows = self.rowmat[split]
        self.eval()
        logits = self.forward(rows)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        return prob

    def save_checkpoint(self, path: str):
        ckpt = {
            "init_args": {
                "dataset_dir": self.dataset_dir,
                "spec_path": self.spec_path,
                "ids_train": self.ids_train,
                "ids_val": self.ids_val,
                "ids_test": self.ids_test,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "use_amp": self.use_amp,
                "normalize_rows": self.normalize_rows,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(ckpt, path)

    @staticmethod
    def load_checkpoint(path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        model = LSTMWithGCN(**ckpt["init_args"])
        model.load_state_dict(ckpt["state_dict"])
        model.device = next(model.parameters()).device
        return model
