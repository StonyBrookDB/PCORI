"""
Health Facts Sequential Data Endpoints (report-v2 - standalone version)

Provides API endpoints for training and predicting with sequential models (LSTM/GRU)
on Health Facts pre-cached data.

This version removes yl.* dependencies and uses local config/data_loader modules.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, validator
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

# Local imports (replaces yl.data.pcori.* imports)
from .data_loader import (
    get_health_facts_sequences,
    get_health_facts_sequences_mmap,
)
from .config import (
    get_feature_sets,
    list_feature_set_names,
    SUPPORTED_T as HF_SUPPORTED_T,
    SUPPORTED_NORMALIZATIONS as HF_SUPPORTED_NORMALIZATIONS,
    SUPPORTED_SAMPLES as HF_SUPPORTED_SAMPLES,
    DEFAULT_LABEL as HF_DEFAULT_LABEL,
    CODE_VERSION as HF_CODE_VERSION,
)

# Mark as available since we're using local implementations
HF_SEQ_AVAILABLE = True
HF_SEQ_IMPORT_ERROR = None


# Create router for sequential endpoints
router = APIRouter(prefix="/api/dataset/health_facts_seq", tags=["sequential"])

# Shared storage (imported from main in production)
# These will be injected by main.py
training_jobs: Dict[str, Dict] = {}
trained_models: Dict[str, Dict] = {}


def set_shared_storage(jobs: Dict, models: Dict):
    """Inject shared storage from main.py."""
    global training_jobs, trained_models
    training_jobs = jobs
    trained_models = models


# ============== Pydantic Models ==============


class HFSeqTrainRequest(BaseModel):
    """Request for training a sequential model."""
    feature_set: str
    T: int = 10
    normalization: Literal["standard", "minmax", "none"] = "standard"
    sample_fraction: float = 1.0
    label: Optional[str] = None
    code_version: Optional[str] = None

    # Model params
    model_type: Literal["lstm", "gru"] = "lstm"
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.2
    use_attention: bool = True

    # Training params
    batch_size: int = 64
    max_epochs: int = 30
    learning_rate: float = 1e-3
    early_stopping_patience: Optional[int] = 5
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: Optional[int] = None

    # Memory/perf
    use_mmap: bool = True
    num_workers: int = 2

    name: Optional[str] = None

    @validator("T")
    def validate_T(cls, v):
        if v not in HF_SUPPORTED_T:
            raise ValueError(f"T={v} not supported. Use one of: {HF_SUPPORTED_T}")
        return v

    @validator("normalization")
    def validate_normalization(cls, v):
        if v not in HF_SUPPORTED_NORMALIZATIONS:
            raise ValueError(f"normalization='{v}' not supported. Use: {HF_SUPPORTED_NORMALIZATIONS}")
        return v

    @validator("sample_fraction")
    def validate_sample_fraction(cls, v):
        if v not in HF_SUPPORTED_SAMPLES:
            raise ValueError(f"sample_fraction={v} not supported. Use: {HF_SUPPORTED_SAMPLES}")
        return v


class HFSeqPredictRequest(BaseModel):
    """Request for predictions from a sequential model."""
    model_id: str
    split: Literal["train", "val", "test"] = "val"
    index: Optional[int] = None
    indices: Optional[List[int]] = None
    explain: bool = False

    @validator("indices")
    def validate_indices(cls, v, values):
        idx = values.get("index")
        if idx is None and not v:
            raise ValueError("Provide either 'index' or 'indices'")
        return v


# ============== Dataset and Model ==============


class SequenceNPYDataset(Dataset):
    """PyTorch Dataset for numpy sequence arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        x = self.X[idx]  # (T, D)
        y = self.y[idx]  # scalar
        x_t = torch.from_numpy(np.array(x, dtype=np.float32))
        y_t = torch.tensor(float(y), dtype=torch.float32)
        return x_t, y_t


class SeqClassifier(pl.LightningModule):
    """LSTM/GRU classifier with optional attention for sequential data."""

    def __init__(
        self,
        input_dim: int,
        model_type: str = "lstm",
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.use_attention = use_attention

        rnn_cls = nn.LSTM if model_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)

        if use_attention:
            self.attn_w = nn.Parameter(torch.randn(out_dim))
        else:
            self.attn_w = None

        self.fc = nn.Linear(out_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, return_attn: bool = False):
        h_seq, _ = self.rnn(x)  # (B, T, H)

        if self.use_attention:
            scale = max(1.0, self.attn_w.shape[0] ** 0.5)
            scores = (h_seq @ self.attn_w) / scale
            attn = torch.softmax(scores, dim=1)  # (B, T)
            context = (h_seq * attn.unsqueeze(-1)).sum(dim=1)  # (B, H)
        else:
            context = h_seq[:, -1, :]
            attn = None

        logits = self.fc(context).squeeze(-1)

        if return_attn:
            return logits, attn
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self(x)
        return torch.sigmoid(logits)

    def explain_temporal(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Compute temporal explanations for a single sequence.

        Args:
            x: (1, T, D) tensor

        Returns:
            Dict with attention, time_saliency, feature_saliency
        """
        self.eval()
        x = x.clone().detach().requires_grad_(True)
        logits, attn = self(x, return_attn=True)
        prob = torch.sigmoid(logits)

        # Gradient x input saliency
        self.zero_grad(set_to_none=True)
        logits.backward(torch.ones_like(logits))
        grad = x.grad  # (1, T, D)

        gi = (grad.abs() * x.abs()).detach()
        time_saliency = gi.sum(dim=-1).squeeze(0)  # (T,)
        feat_saliency = gi.sum(dim=1).squeeze(0)   # (D,)

        out = {
            "prob": float(prob.item()),
            "time_saliency": time_saliency.cpu().numpy(),
            "feature_saliency": feat_saliency.cpu().numpy(),
        }
        if attn is not None:
            out["attention"] = attn.squeeze(0).cpu().numpy()
        else:
            out["attention"] = None

        return out


# ============== Training Task ==============


async def train_hf_seq(job_id: str, req: HFSeqTrainRequest):
    """Background task for training sequential model."""
    job = training_jobs[job_id]

    try:
        label = req.label or HF_DEFAULT_LABEL
        code_version = req.code_version or HF_CODE_VERSION

        # Load arrays using local data_loader
        if req.use_mmap:
            data = get_health_facts_sequences_mmap(
                T=req.T,
                feature_set=req.feature_set,
                normalization=req.normalization,
                sample_fraction=req.sample_fraction,
                label=label,
                code_version=code_version,
            )
        else:
            data = get_health_facts_sequences(
                T=req.T,
                feature_set=req.feature_set,
                normalization=req.normalization,
                sample_fraction=req.sample_fraction,
                label=label,
                code_version=code_version,
                load_arrays=True,
            )

        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]

        D = int(X_train.shape[-1])

        model = SeqClassifier(
            input_dim=D,
            model_type=req.model_type,
            hidden_size=req.hidden_size,
            num_layers=req.num_layers,
            bidirectional=req.bidirectional,
            dropout=req.dropout,
            learning_rate=req.learning_rate,
            use_attention=req.use_attention,
        )

        train_ds = SequenceNPYDataset(X_train, y_train)
        val_ds = SequenceNPYDataset(X_val, y_val)
        test_ds = SequenceNPYDataset(X_test, y_test)

        train_loader = DataLoader(
            train_ds, batch_size=req.batch_size, shuffle=True, num_workers=req.num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=req.batch_size, shuffle=False, num_workers=req.num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=req.batch_size, shuffle=False, num_workers=req.num_workers
        )

        class ProgressCallback(pl.Callback):
            def __init__(self, job_dict):
                self.job = job_dict
                self.train_losses = []
                self.val_losses = []

            def on_train_epoch_end(self, trainer, pl_module):
                tl = trainer.callback_metrics.get("train_loss")
                self.train_losses.append({
                    "epoch": trainer.current_epoch,
                    "value": float(tl) if tl is not None else None
                })

            def on_validation_epoch_end(self, trainer, pl_module):
                vl = trainer.callback_metrics.get("val_loss")
                self.val_losses.append({
                    "epoch": trainer.current_epoch,
                    "value": float(vl) if vl is not None else None
                })
                self.job["progress"] = {
                    "current_epoch": trainer.current_epoch + 1,
                    "total_epochs": trainer.max_epochs,
                    "train_metrics": self.train_losses.copy(),
                    "val_metrics": self.val_losses.copy(),
                }

        callbacks = [ProgressCallback(job)]
        if req.early_stopping_patience is not None and req.early_stopping_patience > 0:
            callbacks.append(
                pl.callbacks.EarlyStopping(
                    monitor="val_loss", mode="min", patience=req.early_stopping_patience
                )
            )

        accelerator = req.accelerator
        devices = req.devices if req.devices is not None else (1 if torch.cuda.is_available() else None)

        trainer = pl.Trainer(
            max_epochs=req.max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
        )

        trainer.fit(model, train_loader, val_loader)

        # Final predictions and metrics
        def _collect(loader):
            model.eval()
            preds = []
            targets = []
            with torch.no_grad():
                for xb, yb in loader:
                    pb = model.predict_proba(xb)
                    preds.append(pb.cpu().numpy())
                    targets.append(yb.cpu().numpy())
            return np.concatenate(preds), np.concatenate(targets)

        train_pred, train_y = _collect(train_loader)
        val_pred, val_y = _collect(val_loader)
        test_pred, test_y = _collect(test_loader)

        def _cls_metrics(y_true, y_proba, thr=0.5):
            return {
                "auroc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.5,
                "auprc": float(average_precision_score(y_true, y_proba)),
                "accuracy": float(np.mean((y_proba >= thr).astype(int) == y_true.astype(int))),
            }

        train_metrics = _cls_metrics(train_y, train_pred)
        val_metrics = _cls_metrics(val_y, val_pred)
        test_metrics = _cls_metrics(test_y, test_pred)

        # Store model
        model_id = f"seq_{req.model_type}_{job_id}"
        trained_models[model_id] = {
            "model": model,
            "model_type": f"seq_{req.model_type}",
            "sequence_params": {
                "T": req.T,
                "D": D,
                "feature_set": req.feature_set,
                "normalization": req.normalization,
                "sample_fraction": req.sample_fraction,
                "label": label,
                "code_version": code_version,
            },
            "artifact_paths": {
                "X_train_path": data["X_train_path"],
                "y_train_path": data["y_train_path"],
                "X_val_path": data["X_val_path"],
                "y_val_path": data["y_val_path"],
                "X_test_path": data["X_test_path"],
                "y_test_path": data["y_test_path"],
                "artifact_root": data["artifact_root"],
            },
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "feature_importance": None,
            "training_history": job.get("progress", {}),
            "created_at": datetime.now().isoformat(),
            "name": job.get("name", f"Seq {req.model_type.upper()} {job_id[:8]}"),
        }

        job["status"] = "completed"
        job["model_id"] = model_id
        job["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        import traceback
        job["traceback"] = traceback.format_exc()


# ============== API Endpoints ==============


@router.get("/config")
async def hf_seq_config():
    """Get configuration for Health Facts sequential data."""
    sets = get_feature_sets()
    return {
        "feature_sets": [
            {"name": name, "num_features": len(fs.feature_ids), "description": fs.description}
            for name, fs in sets.items()
        ],
        "supported_T": HF_SUPPORTED_T,
        "normalizations": HF_SUPPORTED_NORMALIZATIONS,
        "sample_fractions": HF_SUPPORTED_SAMPLES,
        "default_label": HF_DEFAULT_LABEL,
        "code_version": HF_CODE_VERSION,
    }


@router.post("/train")
async def hf_seq_train(request: HFSeqTrainRequest, background_tasks: BackgroundTasks):
    """Start training a sequential model."""
    # Validate feature set
    available = set(list_feature_set_names())
    # Allow known feature set names even if features.csv is missing
    known_sets = {"core", "full", "demographics", "temporal", "diagnoses_top100", "diagnoses_all", "procedures"}
    if request.feature_set not in available and request.feature_set not in known_sets:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature_set='{request.feature_set}'. Available: {sorted(list(available | known_sets))}"
        )

    job_id = str(uuid.uuid4())[:8]
    training_jobs[job_id] = {
        "id": job_id,
        "type": "sequential",
        "model_type": f"seq_{request.model_type}",
        "params": request.dict(),
        "name": request.name or f"Seq {request.model_type.upper()} {job_id}",
        "status": "running",
        "progress": {},
        "started_at": datetime.now().isoformat(),
    }

    background_tasks.add_task(train_hf_seq, job_id, request)
    return {"job_id": job_id, "status": "started"}


@router.get("/training/{job_id}")
async def hf_seq_training_status(job_id: str):
    """Get training job status."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]


@router.post("/predict")
async def hf_seq_predict(request: HFSeqPredictRequest):
    """Get predictions from a sequential model."""
    if request.model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    info = trained_models[request.model_id]
    if not info["model_type"].startswith("seq_"):
        raise HTTPException(status_code=400, detail="Model is not a sequential model")

    # Load arrays
    X_path = info["artifact_paths"][f"X_{request.split}_path"]
    y_path = info["artifact_paths"][f"y_{request.split}_path"]
    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    indices = request.indices if request.indices is not None else [request.index]
    n = X.shape[0]

    for idx in indices:
        if idx is None or idx < 0 or idx >= n:
            raise HTTPException(
                status_code=400, detail=f"Index {idx} out of range 0..{n-1}"
            )

    model: SeqClassifier = info["model"]
    model.eval()

    with torch.no_grad():
        xs = torch.from_numpy(np.array([X[i] for i in indices], dtype=np.float32))
        probs = model.predict_proba(xs).cpu().numpy().tolist()
        preds = [float(p) for p in probs]
        preds_bin = [int(p >= 0.5) for p in preds]

    # Explanation for single sample
    explanations = None
    if request.explain and len(indices) == 1:
        x_single = torch.from_numpy(np.array(X[indices[0]][None, ...], dtype=np.float32))
        expl = model.explain_temporal(x_single)
        explanations = {
            "prob": expl["prob"],
            "time_importance": expl["time_saliency"].tolist(),
            "feature_importance": expl["feature_saliency"].tolist(),
            "attention": expl["attention"].tolist() if expl["attention"] is not None else None,
        }

    return {
        "model_id": request.model_id,
        "split": request.split,
        "indices": indices,
        "probabilities": preds,
        "predictions": preds_bin,
        "explanations": explanations,
    }
