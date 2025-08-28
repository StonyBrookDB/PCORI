#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
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
    LIGHTED (version old-2):
      - Graph side: Heterogeneous one-hop (F→E, P→E), linear + ReLU synthesis E embedding
        - Sequence alignment: Construct row indices of the T most recent encounters for each patient based on t_index
        - Time: Prioritize days_since_index; secondly, timestamp→days; otherwise, use t_index. Δt = adjacent step difference, added to the fusion after log1p projection
        - Fusion: proj_x(X_t) and proj_g(G_t) are fused via gated g_t: g⊙x' + (1-g)⊙g'
        - Sequence encoding: LSTM + pack (disabled), readout using h_n (supports bidirectionality)
    """

    def __init__(
        self,
        dataset_dir: str,
        spec_path: str,
        ids_train: List[str],
        ids_val: List[str],
        ids_test: List[str],
        input_dim: int,             # D
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

        # new
        fusion: str = "gate",       # "gate" | "concat"
        proj_dim: Optional[int] = None,
        use_pack: bool = True,
        time_feature: str = "auto", # "auto"|"gap"|"pos"|"none"（当前实现用 gap/auto）
    ):
        super().__init__()
        self.dataset_dir = str(dataset_dir)
        self.spec_path = str(spec_path)
        self.ids_train = [str(x) for x in ids_train]
        self.ids_val   = [str(x) for x in ids_val]
        self.ids_test  = [str(x) for x in ids_test]

        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.use_amp = bool(use_amp)
        self.normalize_rows = bool(normalize_rows)

        self.fusion = fusion
        self.proj_dim = int(proj_dim) if proj_dim is not None else int(hidden_size)
        self.use_pack = bool(use_pack)
        self.time_feature = time_feature

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ---- Read/build graph ----
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
        for col in ["encounter_id", "patient_id"]:
            if col in encounters.columns:
                encounters[col] = encounters[col].astype(str)
            if col in ef.columns:
                ef[col] = ef[col].astype(str)
            if col in pe.columns:
                pe[col] = pe[col].astype(str)

        feat_sorted = features.sort_values(["feature_id"]).drop_duplicates("feature_id")
        self.fid2col: Dict[int, int] = {int(fid): i for i, fid in enumerate(feat_sorted["feature_id"].astype(int).tolist())}
        F = len(self.fid2col)

        # enc_sorted
        enc_cols = ["encounter_id", "patient_id", "t_index"]
        enc_sorted = encounters.drop_duplicates("encounter_id")[enc_cols].copy()
        enc_sorted["t_index"] = pd.to_numeric(enc_sorted["t_index"], errors="coerce").fillna(0).astype(int)

        if self.time_feature in ("auto", "gap"):
            if "days_since_index" in encounters.columns:
                enc_days = encounters[["encounter_id", "days_since_index"]].copy()
                enc_days["encounter_id"] = enc_days["encounter_id"].astype(str)
                enc_days["days_since_index"] = pd.to_numeric(enc_days["days_since_index"], errors="coerce").astype(float)
                enc_sorted = enc_sorted.merge(enc_days, on="encounter_id", how="left")
                enc_sorted["time_val"] = enc_sorted["days_since_index"]
                enc_sorted.drop(columns=["days_since_index"], inplace=True)
            elif "timestamp" in encounters.columns:
                tmp = encounters[["encounter_id", "timestamp"]].copy()
                tmp["encounter_id"] = tmp["encounter_id"].astype(str)
                tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
                base = tmp["timestamp"].min()
                tmp["time_val"] = (tmp["timestamp"] - base).dt.total_seconds() / 86400.0
                enc_sorted = enc_sorted.merge(tmp[["encounter_id", "time_val"]], on="encounter_id", how="left")
            else:
                enc_sorted["time_val"] = enc_sorted["t_index"].astype(float)
        else:
            enc_sorted["time_val"] = enc_sorted["t_index"].astype(float)

        # handle missing
        enc_sorted["time_val"] = pd.to_numeric(enc_sorted["time_val"], errors="coerce").fillna(enc_sorted["t_index"]).astype(float)

        # Dense index
        self.eid2row: Dict[str, int] = {eid: i for i, eid in enumerate(enc_sorted["encounter_id"].tolist())}
        E = len(self.eid2row)

        # F→E  Matrix
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
        self.A_EF = torch.sparse_coo_tensor(idx, val, size=(E, F)).coalesce().to(self.device)

        # P→E Matrix
        pats = enc_sorted["patient_id"].drop_duplicates().tolist()
        self.pid2idx: Dict[str, int] = {pid: i for i, pid in enumerate(pats)}
        P = len(self.pid2idx)

        pe = pe[pe["encounter_id"].astype(str).isin(self.eid2row.keys())]
        erows = pe["encounter_id"].astype(str).map(self.eid2row).astype(int).to_numpy()
        pcols = pe["patient_id"].astype(str).map(self.pid2idx).astype(int).to_numpy()
        evals = np.ones_like(erows, dtype=np.float32)
        idx2 = torch.tensor([erows, pcols], dtype=torch.long)
        val2 = torch.tensor(evals, dtype=torch.float32)
        self.A_EP = torch.sparse_coo_tensor(idx2, val2, size=(E, P)).coalesce().to(self.device)

        # Row index + Δt (each patient in ascending order of t_index)
        enc_sorted["row"] = enc_sorted["encounter_id"].map(self.eid2row)
        enc_sorted["time_val"] = pd.to_numeric(enc_sorted["time_val"], errors="coerce").fillna(0.0)

        row_map: Dict[str, List[int]] = {}
        dtime_map: Dict[str, List[float]] = {}
        for pid, sub in enc_sorted.groupby("patient_id"):
            sub = sub.sort_values("t_index")
            r = sub["row"].astype(int).tolist()
            t = sub["time_val"].astype(float).tolist()
            if len(t) >= 2:
                dt = [0.0] + [max(0.0, t[i]-t[i-1]) for i in range(1, len(t))]
            else:
                dt = [0.0]*len(t)

            def pad_list(lst, pad_val):
                if len(lst) >= self.T: return lst[:self.T]
                if self.padding_policy == "repeat_last" and len(lst) > 0:
                    return lst + [lst[-1]]*(self.T-len(lst))
                return lst + [pad_val]*(self.T-len(lst))

            row_map[pid] = pad_list(r, -1)
            dtime_map[pid] = pad_list(dt, 0.0)

        def build_mats(ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            N = len(ids)
            rows_mat = np.full((N, self.T), fill_value=-1, dtype=np.int64)
            dtime_mat = np.zeros((N, self.T, 1), dtype=np.float32)
            lens = np.zeros((N,), dtype=np.int64)
            for i, pid in enumerate(ids):
                rm = row_map.get(pid, [-1]*self.T)
                tm = dtime_map.get(pid, [0.0]*self.T)
                rows_mat[i, :] = rm
                dtime_mat[i, :, 0] = tm
                lens[i] = sum(1 for x in rm if x >= 0)
            return (torch.from_numpy(rows_mat),
                    torch.from_numpy(dtime_mat),
                    torch.from_numpy(lens))

        self.rows = {}
        self.dtime = {}
        self.lengths = {}
        for split, ids in [("train", self.ids_train), ("val", self.ids_val), ("test", self.ids_test)]:
            r, dt, ln = build_mats(ids)
            self.rows[split] = r
            self.dtime[split] = dt
            self.lengths[split] = ln

        # ---- Parameters ----
        self.emb_F = nn.Embedding(num_embeddings=F, embedding_dim=self.hidden_size)
        self.emb_P = nn.Embedding(num_embeddings=P, embedding_dim=self.hidden_size)
        nn.init.xavier_uniform_(self.emb_F.weight); nn.init.xavier_uniform_(self.emb_P.weight)

        self.lin_F2E = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.lin_P2E = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)

        # Projection + Gating + Time Projection
        self.proj_x = nn.Linear(self.input_dim, self.proj_dim)
        self.proj_g = nn.Linear(self.hidden_size, self.proj_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.proj_dim * 2, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.Sigmoid(),
        )
        self.time_proj = nn.Linear(1, self.proj_dim)

        # Sequence Encoding
        lstm_in = (self.proj_dim if self.fusion == "gate" else self.input_dim + self.hidden_size)
        self.lstm = nn.LSTM(
            input_size=lstm_in,
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

    # ---------- Figure side ----------
    def _enc_embeddings(self) -> torch.Tensor:
        H_F = self.emb_F.weight  # (F,H)
        msg_F = torch.sparse.mm(self.A_EF, H_F)
        msg_F = self.lin_F2E(msg_F)

        H_P = self.emb_P.weight  # (P,H)
        msg_P = torch.sparse.mm(self.A_EP, H_P)
        msg_P = self.lin_P2E(msg_P)

        H_E = self.drop(self.act(msg_F + msg_P))  # (E,H)
        return H_E

    def _enc_seq(self, enc_emb: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        B, T = rows.shape; H = enc_emb.size(1)
        mask = (rows >= 0).to(enc_emb.dtype).unsqueeze(-1)
        rows_c = rows.clone(); rows_c[rows_c < 0] = 0
        sel = enc_emb.index_select(0, rows_c.view(-1)).view(B, T, H) * mask
        return sel  # (B,T,H)

    # ---------- Fusion ----------
    def _fuse(self, X_seq: torch.Tensor, G_seq: torch.Tensor, dtime_seq: torch.Tensor) -> torch.Tensor:
        if self.fusion == "concat":
            return torch.cat([X_seq, G_seq], dim=-1)
        x_ = self.proj_x(X_seq)
        # Δt: log1p
        dt = torch.log1p(torch.clamp(dtime_seq, min=0.0))
        x_ = x_ + self.time_proj(dt)
        g_ = self.proj_g(G_seq)
        h = torch.cat([x_, g_], dim=-1)
        gate = self.gate(h)
        fused = gate * x_ + (1.0 - gate) * g_
        return fused

    def _lstm_readout(self, Z: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.use_pack:
            packed = pack_padded_sequence(Z, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(Z)
        num_dirs = (2 if self.bidirectional else 1)
        L = self.num_layers
        h_n = h_n.view(L, num_dirs, Z.size(0), self.hidden_size)  # (L,dir,B,H)
        last = h_n[-1]
        if self.bidirectional:
            last = torch.cat([last[0], last[1]], dim=-1)  # (B,2H)
        else:
            last = last[0]  # (B,H)
        return last

    def _forward_with_X(self, rows: torch.Tensor, X_seq: torch.Tensor, dtime_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        enc_emb = self._enc_embeddings().to(self.device)
        G_seq = self._enc_seq(enc_emb, rows.to(self.device))
        Z = self._fuse(X_seq.to(self.device), G_seq, dtime_seq.to(self.device))
        last = self._lstm_readout(Z, lengths.to(self.device))
        logit = self.fc(last).squeeze(-1)
        return logit

    # ---------- Training/Inference ----------
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        if len(y) != len(self.ids_train):
            raise ValueError("LIGHTED.fit 期望接收 train split 的 X/y。")

        rows = self.rows["train"]
        dtime = self.dtime["train"]
        lengths = self.lengths["train"]

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
                idx = idxb.numpy()
                rb = rows[idx, :]
                dtb = dtime[idx, :, :]
                ln = lengths[idx]
                Xb = torch.from_numpy(X[idx, :, :]).to(self.device, non_blocking=True)
                yb = torch.from_numpy(y[idx]).to(self.device, non_blocking=True)

                self._optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    with torch.autocast("cuda"):
                        logits = self._forward_with_X(rb, Xb, dtb, ln)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                    scaler.step(self._optimizer); scaler.update()
                else:
                    logits = self._forward_with_X(rb, Xb, dtb, ln)
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

        rows = self.rows[split]; dtime = self.dtime[split]; lengths = self.lengths[split]
        self.eval()
        probs = []
        for i in range(0, N, self.batch_size):
            rb = rows[i:i+self.batch_size, :]
            dtb = dtime[i:i+self.batch_size, :, :]
            ln = lengths[i:i+self.batch_size]
            Xb = torch.from_numpy(X[i:i+self.batch_size]).to(self.device)
            logits = self._forward_with_X(rb, Xb, dtb, ln)
            probs.append(torch.sigmoid(logits).detach().cpu())
        return torch.cat(probs, dim=0).numpy()

    def save_checkpoint(self, path: str):
        ckpt = {
            "init_args": {
                "dataset_dir": self.dataset_dir, "spec_path": self.spec_path,
                "ids_train": self.ids_train, "ids_val": self.ids_val, "ids_test": self.ids_test,
                "input_dim": self.input_dim, "hidden_size": self.hidden_size,
                "num_layers": self.num_layers, "dropout": self.dropout, "bidirectional": self.bidirectional,
                "lr": self.lr, "batch_size": self.batch_size, "epochs": self.epochs,
                "use_amp": self.use_amp, "normalize_rows": self.normalize_rows,
                "fusion": self.fusion, "proj_dim": self.proj_dim,
                "use_pack": self.use_pack, "time_feature": self.time_feature
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
