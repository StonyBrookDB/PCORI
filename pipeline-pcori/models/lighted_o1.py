#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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


@register("lighted")
class LIGHTED(nn.Module):
    """
    LIGHTED (Version old-1):
    - Use a heterogeneous graph (P→E, F→E) to learn the encounter embedding H_g;
    - For each patient, take the T most recent H_g at t_index, obtaining (T, H_g);
    - Concatenate this with the original feature sequence X∈(T, D) in the last dimension, forming (T, D+H_g);
    - Feed this into an LSTM (or BiLSTM), taking the final output → linear classification.

    -- input X(N,T,D) (from the loader) ，uses the middleware table to construct the graph.
    """

    def __init__(
        self,
        dataset_dir: str,
        spec_path: str,
        ids_train: List[str],
        ids_val: List[str],
        ids_test: List[str],
        input_dim: int,            # D
        hidden_size: int = 128,    # H_g 与 LSTM 隐状态
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
        self.input_dim = int(input_dim)     # D
        self.hidden_size = int(hidden_size) # H_g & LSTM hidden
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

        # ---- Read/compose (same as hetero_rgcn) ----
        ddir = Path(self.dataset_dir)
        spec = _load_spec(Path(self.spec_path))
        self.T = int(spec["sequence"]["seq_len"])
        self.padding_policy = spec["sequence"].get("padding_policy", "repeat_last")

        encounters = _read_table(ddir / "encounters")
        ef = _read_table(ddir / "edges_encounter_feature")
        pe = _read_table(ddir / "edges_patient_encounter")
        features = pd.read_csv(ddir / "features.csv")

        ef = ef.copy()
        ef["value"] = pd.to_numeric(ef["value"], errors="coerce").fillna(0.0)
        ef["feature_id"] = pd.to_numeric(ef["feature_id"], downcast="integer", errors="coerce").fillna(-1).astype(int)
        pe = pe.copy()
        pe["patient_id"] = pe["patient_id"].astype(str)
        pe["encounter_id"] = pe["encounter_id"].astype(str)

        feat_sorted = features.sort_values(["feature_id"]).drop_duplicates("feature_id")
        self.fid2col: Dict[int, int] = {int(fid): i for i, fid in enumerate(feat_sorted["feature_id"].astype(int).tolist())}
        F = len(self.fid2col)

        enc_sorted = encounters.drop_duplicates("encounter_id")[["encounter_id","patient_id","t_index"]]
        enc_sorted["encounter_id"] = enc_sorted["encounter_id"].astype(str)
        enc_sorted["patient_id"]   = enc_sorted["patient_id"].astype(str)
        enc_sorted["t_index"]      = pd.to_numeric(enc_sorted["t_index"], errors="coerce").fillna(0).astype(int)
        self.eid2row: Dict[str, int] = {eid: i for i, eid in enumerate(enc_sorted["encounter_id"].tolist())}
        E = len(self.eid2row)

        pats = enc_sorted["patient_id"].drop_duplicates().tolist()
        self.pid2idx: Dict[str, int] = {pid: i for i, pid in enumerate(pats)}
        P = len(self.pid2idx)

        # F→E
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
        idx = torch.tensor([rows, cols], dtype=torch.long); val = torch.tensor(vals, dtype=torch.float32)
        self.A_EF = torch.sparse_coo_tensor(idx, val, size=(E, F)).coalesce().to(self.device)

        # P→E
        pe = pe[pe["encounter_id"].isin(self.eid2row.keys())]
        erows = pe["encounter_id"].astype(str).map(self.eid2row).astype(int).to_numpy()
        pcols = pe["patient_id"].astype(str).map(self.pid2idx).astype(int).to_numpy()
        evals = np.ones_like(erows, dtype=np.float32)
        idx2 = torch.tensor([erows, pcols], dtype=torch.long)
        val2 = torch.tensor(evals, dtype=torch.float32)
        self.A_EP = torch.sparse_coo_tensor(idx2, val2, size=(E, P)).coalesce().to(self.device)

        enc_sorted["row"] = enc_sorted["encounter_id"].map(self.eid2row)
        pid2rows: Dict[str, List[int]] = {}
        for pid, sub in enc_sorted.groupby("patient_id"):
            sub = sub.sort_values("t_index")
            lst = sub["row"].astype(int).tolist()
            if len(lst) >= self.T:
                pid2rows[pid] = lst[:self.T]
            else:
                if self.padding_policy == "repeat_last" and len(lst) > 0:
                    pad = [lst[-1]] * (self.T - len(lst))
                else:
                    pad = [-1] * (self.T - len(lst))
                pid2rows[pid] = lst + pad

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

        # ---- Heterogeneous message to E, then concatenated with original X and fed into LSTM ----
        self.emb_F = nn.Embedding(num_embeddings=F, embedding_dim=self.hidden_size)
        self.emb_P = nn.Embedding(num_embeddings=P, embedding_dim=self.hidden_size)
        nn.init.xavier_uniform_(self.emb_F.weight); nn.init.xavier_uniform_(self.emb_P.weight)

        self.lin_F2E = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.lin_P2E = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)

        fused_in = self.input_dim + self.hidden_size  # D + H_g
        self.lstm = nn.LSTM(
            input_size=fused_in,
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

    # ---- Graph side: Construct Encounter embedding, taking (B,T,H_g) sequence by rows ----
    def _enc_embeddings(self) -> torch.Tensor:
        H_F = self.emb_F.weight
        H_P = self.emb_P.weight
        msg_F = self.lin_F2E(torch.sparse.mm(self.A_EF, H_F))
        msg_P = self.lin_P2E(torch.sparse.mm(self.A_EP, H_P))
        H_E = self.drop(self.act(msg_F + msg_P))  # (E,H)
        return H_E

    def _enc_seq(self, enc_emb: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        B, T = rows.shape; H = enc_emb.size(1)
        mask = (rows >= 0).to(enc_emb.dtype).unsqueeze(-1)
        rows_c = rows.clone(); rows_c[rows_c < 0] = 0
        sel = enc_emb.index_select(0, rows_c.view(-1)).view(B, T, H) * mask
        return sel  # (B,T,H)

    # ---- forward: rows + corresponding batch X_seq ----
    def _forward_with_X(self, rows: torch.Tensor, X_seq: torch.Tensor) -> torch.Tensor:
        enc_emb = self._enc_embeddings().to(self.device)
        G_seq = self._enc_seq(enc_emb, rows.to(self.device))  # (B,T,Hg)
        fused = torch.cat([X_seq.to(self.device), G_seq], dim=-1)  # (B,T,D+Hg)
        out, _ = self.lstm(fused)
        last = out[:, -1, :]
        logit = self.fc(last).squeeze(-1)
        return logit

    # ---- Unified Interface ----
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N,T,D) original sequence features; y: (N,)
        """
        # 清理
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        if len(y) != len(self.ids_train):
            raise ValueError("LIGHTED.fit expect train splitted X/y。")

        rows = self.rowmat["train"]  # (N_train,T)
        ds = TensorDataset(torch.arange(len(self.ids_train)))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        n_pos = float((y > 0.5).sum()); n_neg = float((y <= 0.5).sum())
        pos_weight = (n_neg / max(n_pos, 1.0))
        self._pos_weight = torch.tensor([pos_weight], dtype=torch.float32, device=self.device)

        self.to(self.device); self.train()
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.device.type == "cuda"))
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)

        for _ in range(self.epochs):
            for idxb, in dl:
                rb = rows[idxb.numpy(), :]                               # (B,T)
                Xb = torch.from_numpy(X[idxb.numpy(), :, :]).to(self.device, non_blocking=True)  # (B,T,D)
                yb = torch.from_numpy(y[idxb.numpy()]).to(self.device, non_blocking=True)        # (B,)
                self._optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    with torch.autocast("cuda"):
                        logits = self._forward_with_X(rb, Xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                    scaler.step(self._optimizer); scaler.update()
                else:
                    logits = self._forward_with_X(rb, Xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                    self._optimizer.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        N = X.shape[0]
        if N == len(self.ids_val): split = "val"
        elif N == len(self.ids_test): split = "test"
        elif N == len(self.ids_train): split = "train"
        else: split = "test"
        rows = self.rowmat[split]

        self.eval()
        probs = []
        for i in range(0, N, self.batch_size):
            rb = rows[i:i+self.batch_size, :]
            Xb = torch.from_numpy(X[i:i+self.batch_size]).to(self.device)
            logits = self._forward_with_X(rb, Xb)
            probs.append(torch.sigmoid(logits).detach().cpu())
        return torch.cat(probs, dim=0).numpy()

    # ---- save/load----
    def save_checkpoint(self, path: str):
        ckpt = {
            "init_args": {
                "dataset_dir": self.dataset_dir, "spec_path": self.spec_path,
                "ids_train": self.ids_train, "ids_val": self.ids_val, "ids_test": self.ids_test,
                "input_dim": self.input_dim, "hidden_size": self.hidden_size,
                "num_layers": self.num_layers, "dropout": self.dropout, "bidirectional": self.bidirectional,
                "lr": self.lr, "batch_size": self.batch_size, "epochs": self.epochs,
                "use_amp": self.use_amp, "normalize_rows": self.normalize_rows
            },
            "state_dict": self.state_dict(),
        }
        torch.save(ckpt, path)

    @staticmethod
    def load_checkpoint(path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        model = LIGHTED(**ckpt["init_args"])
        model.load_state_dict(ckpt["state_dict"])
        model.device = next(model.parameters()).device
        return model
