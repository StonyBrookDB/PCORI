#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


@register("gcn")
class GraphOnlyGCN(nn.Module):
    """
    最小可行 Graph-only：Encounter<-Feature 一跳消息聚合
    - 构建二部稀疏矩阵 Inc(E,F)（行=encounter，列=feature，值=value，经行归一）
    - 可学习的 feature embedding: nn.Embedding(F,H)
    - 就诊嵌入 enc_emb = Inc @ feat_emb
    - 病人读出：取最近 T 次就诊的 enc_emb 做池化（mean/last等），得到 patient_emb -> 线性分类

    接口：
      fit(X,y) / predict_proba(X)     # X 仅用于匹配 batch 大小；真实特征来自中间件
      save_checkpoint / load_checkpoint
    """

    def __init__(
        self,
        dataset_dir: str,
        spec_path: str,
        ids_train: List[str],
        ids_val: List[str],
        ids_test: List[str],
        hidden_size: int = 128,
        pooling: str = "mean",     # "mean" | "last" | "max"
        lr: float = 1e-3,
        batch_size: int = 256,
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
        self.pooling = pooling
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.use_amp = bool(use_amp)
        self.normalize_rows = bool(normalize_rows)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # -------- 读取中间件与 Spec --------
        ddir = Path(self.dataset_dir)
        spec = _load_spec(Path(self.spec_path))
        self.T = int(spec["sequence"]["seq_len"])
        self.padding_policy = spec["sequence"].get("padding_policy", "repeat_last")

        encounters = _read_table(ddir / "encounters")
        ef = _read_table(ddir / "edges_encounter_feature")
        features = pd.read_csv(ddir / "features.csv")

        # 清洗 / 类型
        ef = ef.copy()
        ef["value"] = pd.to_numeric(ef["value"], errors="coerce").fillna(0.0)
        ef["feature_id"] = pd.to_numeric(ef["feature_id"], downcast="integer", errors="coerce").fillna(-1).astype(int)

        # -------- 稠密索引：feature_id -> [0..F-1]，encounter_id -> [0..E-1] --------
        feat_sorted = features.sort_values(["feature_id"]).drop_duplicates("feature_id")
        self.fid2col: Dict[int, int] = {int(fid): i for i, fid in enumerate(feat_sorted["feature_id"].astype(int).tolist())}
        F = len(self.fid2col)

        enc_sorted = encounters.drop_duplicates("encounter_id")[["encounter_id","patient_id","t_index"]]
        enc_sorted["encounter_id"] = enc_sorted["encounter_id"].astype(str)
        enc_sorted["patient_id"]   = enc_sorted["patient_id"].astype(str)
        enc_sorted["t_index"]      = pd.to_numeric(enc_sorted["t_index"], errors="coerce").fillna(0).astype(int)
        self.eid2row: Dict[str, int] = {eid: i for i, eid in enumerate(enc_sorted["encounter_id"].tolist())}
        E = len(self.eid2row)

        # -------- 构造稀疏 Inc(E,F) --------
        # 先聚合 (encounter_id, feature_id) -> sum(value)
        ef = ef[ef["feature_id"].isin(self.fid2col.keys())]
        ef = ef[ef["encounter_id"].astype(str).isin(self.eid2row.keys())]
        g = ef.groupby(["encounter_id","feature_id"], as_index=False)["value"].sum()

        rows = g["encounter_id"].astype(str).map(self.eid2row).astype(int).to_numpy()
        cols = g["feature_id"].astype(int).map(self.fid2col).astype(int).to_numpy()
        vals = g["value"].astype(float).to_numpy()

        # 行归一（避免数值爆炸）
        if self.normalize_rows:
            # 计算每行和，再除以行和
            rs = np.zeros(E, dtype=np.float32)
            np.add.at(rs, rows, vals.astype(np.float32))
            rs[rs == 0.0] = 1.0
            vals = (vals / rs[rows]).astype(np.float32)
        else:
            vals = vals.astype(np.float32)

        idx = torch.tensor([rows, cols], dtype=torch.long)
        val = torch.tensor(vals, dtype=torch.float32)
        self.Inc = torch.sparse_coo_tensor(idx, val, size=(E, F)).coalesce().to(self.device)

        # -------- 病人 -> 遇诊行索引序列（按 t_index 升序，截断/补齐到 T）--------
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
                    pad = [-1] * (self.T - len(rows_list))  # -1 代表全零
                pid2rows[pid] = rows_list + pad

        # 预构建三组 split 的 row 索引矩阵（N_split, T），-1 表示 pad
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

        # -------- 可学习参数 --------
        self.feat_emb = nn.Embedding(num_embeddings=F, embedding_dim=self.hidden_size)
        nn.init.xavier_uniform_(self.feat_emb.weight)

        self.fc = nn.Linear(self.hidden_size, 1)

        # 训练时使用
        self.is_torch = True
        self._optimizer = None
        self._pos_weight = None

    # --------- 前向 / 读出 ---------
    def _encounter_embeddings(self) -> torch.Tensor:
        # (E,F) @ (F,H) -> (E,H)
        return torch.sparse.mm(self.Inc, self.feat_emb.weight)

    def _pool_patient(self, enc_emb: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """
        enc_emb: (E,H), rows: (B,T) (含 -1)
        返回: (B,H)
        """
        B, T = rows.shape
        H = enc_emb.size(1)
        device = enc_emb.device
        # gather，-1 位置用 0
        mask = (rows >= 0).to(enc_emb.dtype).unsqueeze(-1)  # (B,T,1)
        rows_clamped = rows.clone()
        rows_clamped[rows_clamped < 0] = 0
        enc_sel = enc_emb.index_select(0, rows_clamped.view(-1)).view(B, T, H)
        enc_sel = enc_sel * mask  # 对 pad 位清零

        if self.pooling == "last":
            # 找到每行最后一个有效位置
            valid_counts = mask.squeeze(-1).sum(dim=1)  # (B,)
            last_idx = torch.clamp(valid_counts.long() - 1, min=0)  # 无效时取 0
            out = enc_sel[torch.arange(B, device=device), last_idx, :]  # (B,H)
            return out
        elif self.pooling == "max":
            out, _ = (enc_sel + (mask - 1) * 1e9).max(dim=1)  # 使 pad 位为 -inf
            return out
        else:  # mean
            denom = torch.clamp(mask.sum(dim=1), min=1.0)  # (B,1)
            out = enc_sel.sum(dim=1) / denom
            return out

    def forward(self, split_rows: torch.Tensor) -> torch.Tensor:
        enc_emb = self._encounter_embeddings()  # (E,H)
        enc_emb = enc_emb.to(self.device)
        pat_emb = self._pool_patient(enc_emb, split_rows.to(self.device))  # (B,H)
        logit = self.fc(pat_emb).squeeze(-1)  # (B,)
        return logit

    # --------- 公共接口 ----------
    def fit(self, X: np.ndarray, y: np.ndarray):
        y = y.astype(np.float32, copy=False)
        # 用 y 的长度来选择 split（与 train.py 的 ids_train 顺序一致）
        if len(y) == len(self.ids_train):
            split = "train"
        else:
            raise ValueError("GraphOnlyGCN.fit 期望接收 train split 的 y。")

        rows = self.rowmat[split]  # (N_train, T)
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
                # 取行索引
                rb = rows[idx_batch.numpy(), :]  # (B,T)
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
        # 用传入 X 的 N 判断 split
        N = X.shape[0]
        if N == len(self.ids_val):
            split = "val"
        elif N == len(self.ids_test):
            split = "test"
        elif N == len(self.ids_train):
            split = "train"
        else:
            # 回退：优先 test
            split = "test"
        rows = self.rowmat[split]
        self.eval()
        logits = self.forward(rows)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        return prob

    # --------- 保存 / 加载 ----------
    def save_checkpoint(self, path: str):
        ckpt = {
            "init_args": {
                "dataset_dir": self.dataset_dir,
                "spec_path": self.spec_path,
                "ids_train": self.ids_train,
                "ids_val": self.ids_val,
                "ids_test": self.ids_test,
                "hidden_size": self.hidden_size,
                "pooling": self.pooling,
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
        model = GraphOnlyGCN(**ckpt["init_args"])
        model.load_state_dict(ckpt["state_dict"])
        model.device = next(model.parameters()).device
        return model
