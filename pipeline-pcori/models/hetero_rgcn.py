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


@register("hetero_rgcn")
class HeteroRGCN(nn.Module):
    """
    Heterogeneous RGCN:
      nodes：Patient(P), Encounter(E), Feature(F)
      edge：P→E（edges_patient_encounter），F→E（edges_encounter_feature）
      Aggregation：E receives messages from P and F (relationship-specific weights) and is activated to obtain the E representation;
                     Patient representations are learned (embedding), and ultimately pooled based on each patient's E representations 
                     from their most recent T encounters to obtain the patient representation → linear classification.

    Only graph views are used
    """

    def __init__(
        self,
        dataset_dir: str,
        spec_path: str,
        ids_train: List[str],
        ids_val: List[str],
        ids_test: List[str],
        hidden_size: int = 128,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 8,
        device: str = None,
        use_amp: bool = False,
        normalize_rows: bool = True,
        pooling: str = "mean",  # "mean" | "last" | "max"
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dataset_dir = str(dataset_dir)
        self.spec_path = str(spec_path)
        self.ids_train = [str(x) for x in ids_train]
        self.ids_val   = [str(x) for x in ids_val]
        self.ids_test  = [str(x) for x in ids_test]
        self.hidden_size = int(hidden_size)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.use_amp = bool(use_amp)
        self.normalize_rows = bool(normalize_rows)
        self.pooling = pooling
        self.dropout = float(dropout)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ---- reading data & Spec ----
        ddir = Path(self.dataset_dir)
        spec = _load_spec(Path(self.spec_path))
        self.T = int(spec["sequence"]["seq_len"])
        self.padding_policy = spec["sequence"].get("padding_policy", "repeat_last")

        encounters = _read_table(ddir / "encounters")
        ef = _read_table(ddir / "edges_encounter_feature")
        pe = _read_table(ddir / "edges_patient_encounter")
        features = pd.read_csv(ddir / "features.csv")

        # clean
        ef = ef.copy()
        ef["value"] = pd.to_numeric(ef["value"], errors="coerce").fillna(0.0)
        ef["feature_id"] = pd.to_numeric(ef["feature_id"], downcast="integer", errors="coerce").fillna(-1).astype(int)
        pe = pe.copy()
        pe["patient_id"] = pe["patient_id"].astype(str)
        pe["encounter_id"] = pe["encounter_id"].astype(str)

        # Dense Index
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

        # F→E Sparse Matrix（E,F）
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

        # P→E sparse matrix (E,P) - one visit belongs to one patient
        pe = pe[pe["encounter_id"].isin(self.eid2row.keys())]
        erows = pe["encounter_id"].astype(str).map(self.eid2row).astype(int).to_numpy()
        pcols = pe["patient_id"].astype(str).map(self.pid2idx).astype(int).to_numpy()
        evals = np.ones_like(erows, dtype=np.float32)
        idx2 = torch.tensor([erows, pcols], dtype=torch.long)
        val2 = torch.tensor(evals, dtype=torch.float32)
        self.A_EP = torch.sparse_coo_tensor(idx2, val2, size=(E, P)).coalesce().to(self.device)

        # Row index matrix (last T visits for each patient)
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

        # ---- Parameters: F/P node embedding + relation-specific linear layer + classification head ----
        self.emb_F = nn.Embedding(num_embeddings=F, embedding_dim=self.hidden_size)
        self.emb_P = nn.Embedding(num_embeddings=P, embedding_dim=self.hidden_size)
        nn.init.xavier_uniform_(self.emb_F.weight); nn.init.xavier_uniform_(self.emb_P.weight)

        self.lin_F2E = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.lin_P2E = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)

        self.fc = nn.Linear(self.hidden_size, 1)

        self.is_torch = True
        self._optimizer = None
        self._pos_weight = None

    # ---- Graph forward: Get Encounter representation → Pooling into Patient representation ----
    def _enc_embeddings(self) -> torch.Tensor:
        # F→E
        H_F = self.emb_F.weight  # (F,H)
        msg_F = torch.sparse.mm(self.A_EF, H_F)     # (E,H)
        msg_F = self.lin_F2E(msg_F)

        # P→E
        H_P = self.emb_P.weight  # (P,H)
        msg_P = torch.sparse.mm(self.A_EP, H_P)     # (E,H)
        msg_P = self.lin_P2E(msg_P)

        H_E = self.act(msg_F + msg_P)
        H_E = self.drop(H_E)
        return H_E  # (E,H)

    def _pool_patient(self, enc_emb: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        B, T = rows.shape; H = enc_emb.size(1)
        mask = (rows >= 0).to(enc_emb.dtype).unsqueeze(-1)
        rows_c = rows.clone(); rows_c[rows_c < 0] = 0
        enc_sel = enc_emb.index_select(0, rows_c.view(-1)).view(B, T, H) * mask
        if self.pooling == "last":
            counts = mask.squeeze(-1).sum(dim=1)
            last_idx = torch.clamp(counts.long() - 1, min=0)
            out = enc_sel[torch.arange(B, device=enc_emb.device), last_idx, :]
            return out
        elif self.pooling == "max":
            out, _ = (enc_sel + (mask - 1) * 1e9).max(dim=1)
            return out
        else:
            denom = torch.clamp(mask.sum(dim=1), min=1.0)
            return enc_sel.sum(dim=1) / denom

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        enc_emb = self._enc_embeddings().to(self.device)         # (E,H)
        pat_emb = self._pool_patient(enc_emb, rows.to(self.device))  # (B,H)
        logit = self.fc(pat_emb).squeeze(-1)                     # (B,)
        return logit

    # ---- Upstream and downstream interfaces ----
    def fit(self, X: np.ndarray, y: np.ndarray):
        y = y.astype(np.float32, copy=False)
        if len(y) != len(self.ids_train):
            raise ValueError("HeteroRGCN.fit expects train split 的 y（same length as ids_train）。")
        rows = self.rowmat["train"]
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
            for idxb, yb in dl:
                rb = rows[idxb.numpy(), :]
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
        return torch.sigmoid(logits).detach().cpu().numpy()

    def save_checkpoint(self, path: str):
        ckpt = {
            "init_args": {
                "dataset_dir": self.dataset_dir, "spec_path": self.spec_path,
                "ids_train": self.ids_train, "ids_val": self.ids_val, "ids_test": self.ids_test,
                "hidden_size": self.hidden_size, "lr": self.lr, "batch_size": self.batch_size,
                "epochs": self.epochs, "use_amp": self.use_amp, "normalize_rows": self.normalize_rows,
                "pooling": self.pooling, "dropout": self.dropout
            },
            "state_dict": self.state_dict(),
        }
        torch.save(ckpt, path)

    @staticmethod
    def load_checkpoint(path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        model = HeteroRGCN(**ckpt["init_args"])
        model.load_state_dict(ckpt["state_dict"])
        model.device = next(model.parameters()).device
        return model
