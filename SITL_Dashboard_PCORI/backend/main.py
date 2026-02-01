"""
FastAPI backend for SITL prototype.
Handles model training, predictions, and GPT integration.

Phase A Updates:
- Added SQLite database support
- Added cohort filtering API
- Added /api/train/quick with SHAP
- Added per-patient explanation API
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_paths = [
        Path(__file__).parent / ".env",  # backend/.env
        Path(__file__).parent.parent / ".env",  # sitl_dashboard/.env
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key not in os.environ:  # Don't override existing env vars
                            os.environ[key] = value
            print(f"Loaded environment from {env_path}")
            break


# Load .env file before other imports that might need API keys
load_env_file()

# ML imports
import lightgbm as lgb
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
from starlette.middleware.sessions import SessionMiddleware
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Health Facts Sequential imports (local - report-v2)
try:
    from .config import get_feature_sets, HEALTH_FACTS_MIDDLEWARE_PATH
    from .data_loader import get_health_facts_sequences
    from .config import DEFAULT_LABEL
    HAS_HEALTH_FACTS_SEQ = True
except ImportError:
    HAS_HEALTH_FACTS_SEQ = False
    print("Note: Health Facts Sequential API not available")

# SHAP for explanations
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

# OpenAI
from openai import OpenAI

# Anthropic (Claude) - for sandbox analysis
try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print(
        "Note: anthropic SDK not installed. Analysis sandbox will use mock responses."
    )

from . import database as db

# Local imports
from .config import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    MIN_COHORT_SIZE,
    MIN_EVENT_COUNT,
    SHAP_MAX_SAMPLES,
    VERSION,
)

# Dataset registry
try:
    # from yl.data.datasets import get_available_datasets, get_dataset_spec  # Removed for report-v2

    DATASET_REGISTRY_AVAILABLE = False  # Disabled for report-v2
except ImportError:
    DATASET_REGISTRY_AVAILABLE = False
    print("Note: yl.data.datasets not available. Using default synthetic dataset.")

# Data adapter for multi-dataset support
from .data_adapter import (
    AVAILABLE_DATASETS,
    get_dataset_config,
    get_samples_for_display,
    get_training_data,
    load_dataset,
)

# Job integration for scheduler-based training
from .job_integration import (
    DashboardJob,
    get_job,
    get_job_log,
    get_training_curves,
    list_jobs,
    submit_training_job_async,
    update_job_status,
)

# Sandbox executor for Claude analysis
from .sandbox_executor import (
    ExecutionResult,
    cleanup_pickle_file,
    execute_in_sandbox,
    extract_code_from_response,
    prepare_data_for_sandbox,
    validate_code_safety,
)

# Analytics for user behavior tracking
from .analytics import EventTypes, log_event

# User credentials (prototype only - not for production)
USERS = {
    "demo": "demo",
}

# Paths that don't require authentication
PUBLIC_PATHS = {"/", "/login", "/api/login", "/health", "/api/dataset/health_facts_seq", "/sequential", "/static"}


# Auth middleware class (needs to be defined before app.add_middleware)
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    """Check authentication for all requests except public paths."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow public paths
        if any(path.startswith(p) for p in PUBLIC_PATHS):
            return await call_next(request)

        # Allow static files (CSS, JS, etc.)
        if path.startswith("/static"):
            return await call_next(request)

        # Check for valid session
        username = request.session.get("username")
        if not username:
            # For API requests, return 401 JSON
            if path.startswith("/api/"):
                return JSONResponse({"error": "Not authenticated"}, status_code=401)
            # For page requests, redirect to login with next URL
            next_url = str(request.url.path)
            if request.url.query:
                next_url += f"?{request.url.query}"
            return RedirectResponse(f"/login?next={next_url}", status_code=302)

        # Store username in request state for handlers
        request.state.username = username
        return await call_next(request)

# Initialize app
app = FastAPI(title="PCORI SITL Prototype", version=VERSION)

# Middleware order: last added = outermost (processes first)
# We need: CORS -> Session -> Auth -> route handlers
# So add in reverse: Auth first, Session second, CORS last

app.add_middleware(AuthMiddleware)

app.add_middleware(
    SessionMiddleware,
    secret_key="sitl-dashboard-secret-key-prototype",
    session_cookie="sitl_session",
    max_age=86400,  # 24 hours
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "patients.json"
FRONTEND_PATH = BASE_DIR / "frontend"

# In-memory storage for training jobs and models
training_jobs: Dict[str, Dict] = {}
trained_models: Dict[str, Dict] = {}  # Legacy in-memory cache, now backed by DB


def migrate_jobs_to_models_db():
    """Migrate completed jobs to the models database (one-time migration)."""
    from .job_integration import (
        TrainedModel,
        get_model,
        get_training_curves,
        list_jobs,
        save_model,
    )

    jobs = list_jobs(limit=100)
    migrated = 0

    for job in jobs:
        if job.status != "completed":
            continue

        # Check if already in models DB
        existing = get_model(job.job_id)
        if existing:
            continue

        # Get training curves
        curves = get_training_curves(job.job_id)

        # Try to load final metrics (new format)
        final_metrics_path = Path(job.work_dir) / "final_metrics.json"
        metrics = {}
        if final_metrics_path.exists():
            try:
                with open(final_metrics_path) as f:
                    metrics = json.load(f)
            except Exception:
                pass

        # Fallback: try to extract from metrics.json (old format)
        if not metrics:
            metrics_path = Path(job.work_dir) / "metrics.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        data = json.load(f)
                        if data.get("train"):
                            last_train = data["train"][-1]
                            metrics.update(
                                {
                                    f"train_{k}": v
                                    for k, v in last_train.items()
                                    if k != "epoch"
                                }
                            )
                        if data.get("val"):
                            last_val = data["val"][-1]
                            metrics.update(
                                {
                                    f"val_{k}": v
                                    for k, v in last_val.items()
                                    if k != "epoch"
                                }
                            )
                except Exception:
                    pass

        # Save to models database
        model = TrainedModel(
            model_id=job.job_id,
            job_id=job.job_id,
            dataset_id=job.dataset_id,
            model_type=job.model_type,
            model_name=job.model_name,
            task_type="classification",  # Default, will be updated when we have proper task type
            features=job.features,
            train_metrics={
                k.replace("train_", ""): v
                for k, v in metrics.items()
                if k.startswith("train_")
            },
            val_metrics={
                k.replace("val_", ""): v
                for k, v in metrics.items()
                if k.startswith("val_")
            },
            feature_importance=[],
            training_history={
                "train": curves.get("train", []),
                "val": curves.get("val", []),
            },
            model_path=str(Path(job.work_dir) / f"{job.job_id}.pt"),
            work_dir=job.work_dir,
            created_at=job.created_at,
        )
        save_model(model)
        migrated += 1

    if migrated > 0:
        print(f"Migrated {migrated} completed jobs to models database")


def get_models_count():
    """Get count of models in database."""
    from .job_integration import list_models

    return len(list_models(limit=1000))


# Initialize database and migrate on startup
migrate_jobs_to_models_db()
print(f"Models database initialized with {get_models_count()} models")

# Load patient data from JSON (legacy support)
# New code uses SQLite database, but we keep JSON for backward compatibility
try:
    with open(DATA_PATH) as f:
        patient_data = json.load(f)
        PATIENTS = {p["id"]: p for p in patient_data["patients"]}
except FileNotFoundError:
    PATIENTS = {}
    print("Note: patients.json not found. Using SQLite database.")

# OpenAI client and model configuration
# Using GPT-5 for better performance at lower cost (vs GPT-4o)
# Updated: Phase A - December 2025
# Fallback chain: gpt-5 -> gpt-4o -> gpt-4o-mini
OPENAI_CHAT_MODELS = ["gpt-5", "gpt-4o", "gpt-4o-mini"]  # Primary + fallbacks
OPENAI_CHAT_MODEL = OPENAI_CHAT_MODELS[0]  # Default primary model
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Anthropic client for Claude sandbox analysis
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # Fast and capable
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
anthropic_client = (
    Anthropic(api_key=anthropic_api_key)
    if ANTHROPIC_AVAILABLE and anthropic_api_key
    else None
)

# ============== Pydantic Models ==============


class TrainRequest(BaseModel):
    model_type: str  # "lightgbm" or "mlp"
    features: List[str]
    name: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str


# Claude sandbox analysis request/response
class AnalysisChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []


class AnalysisChatResponse(BaseModel):
    response_text: str
    code_generated: Optional[str] = None
    execution_result: Optional[str] = None
    plot_base64: Optional[str] = None
    error: Optional[str] = None
    execution_time_seconds: Optional[float] = None


# Job scheduler training request/response
class ScheduledTrainRequest(BaseModel):
    dataset_id: str
    model_type: str  # "lightgbm", "logreg", "pytorch"
    features: List[str]
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = {}


class JobStatusResponse(BaseModel):
    job_id: str
    session_id: str
    dataset_id: str
    model_type: str
    model_name: str
    status: str
    created_at: str
    error: Optional[str] = None


class JobLogResponse(BaseModel):
    log: str
    log_path: Optional[str] = None
    lines_returned: int = 0
    error: Optional[str] = None


class TrainingCurvesResponse(BaseModel):
    source: Optional[str] = None
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    error: Optional[str] = None


# ============== Phase A: New Pydantic Models ==============


class CohortFilterRequest(BaseModel):
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    cancer_status: Optional[bool] = None
    opioid_threshold: Optional[int] = None
    mental_health_min: Optional[int] = None

    @validator("age_max")
    def validate_age_range(cls, v, values):
        if v is not None and values.get("age_min") is not None:
            if v < values["age_min"]:
                raise ValueError("age_max must be >= age_min")
        return v


class QuickTrainRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = {}
    model_type: str = "lightgbm"
    features: List[str]
    name: Optional[str] = None

    @validator("model_type")
    def validate_model_type(cls, v):
        if v not in ["lightgbm", "logreg"]:
            raise ValueError('model_type must be "lightgbm" or "logreg"')
        return v

    @validator("features")
    def validate_features(cls, v):
        if not v:
            raise ValueError("At least one feature required")
        invalid = [f for f in v if f not in FEATURE_NAMES]
        if invalid:
            raise ValueError(f"Unknown features: {invalid}. Valid: {FEATURE_NAMES}")
        return v


# ============== MLP Model (PyTorch Lightning) ==============


class MLPModel(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [32, 16]):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.BCELoss()

        # Store metrics for plotting
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


# ============== Training Functions ==============


def prepare_data(features: List[str]):
    """Prepare train/val data arrays."""
    train_X, train_y = [], []
    val_X, val_y = [], []

    for patient in PATIENTS.values():
        x = [patient["features"][f] for f in features]
        y = patient["outcome"]

        if patient["split"] == "train":
            train_X.append(x)
            train_y.append(y)
        else:
            val_X.append(x)
            val_y.append(y)

    return (
        np.array(train_X, dtype=np.float32),
        np.array(train_y, dtype=np.float32),
        np.array(val_X, dtype=np.float32),
        np.array(val_y, dtype=np.float32),
    )


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "auc": (
            float(roc_auc_score(y_true, y_pred_proba))
            if len(np.unique(y_true)) > 1
            else 0.5
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Confusion matrix
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    metrics["confusion_matrix"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    return metrics


async def train_lightgbm(job_id: str, features: List[str]):
    """Train LightGBM model."""
    job = training_jobs[job_id]

    try:
        train_X, train_y, val_X, val_y = prepare_data(features)

        # Create datasets
        train_data = lgb.Dataset(train_X, label=train_y, feature_name=features)
        val_data = lgb.Dataset(
            val_X, label=val_y, feature_name=features, reference=train_data
        )

        # Training parameters
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 15,
            "learning_rate": 0.1,
            "feature_fraction": 0.9,
            "verbose": -1,
        }

        # Track metrics during training
        train_aucs = []
        val_aucs = []

        def callback(env):
            if env.iteration % 5 == 0:
                # Get current predictions
                train_pred = env.model.predict(train_X)
                val_pred = env.model.predict(val_X)

                train_auc = (
                    roc_auc_score(train_y, train_pred)
                    if len(np.unique(train_y)) > 1
                    else 0.5
                )
                val_auc = (
                    roc_auc_score(val_y, val_pred) if len(np.unique(val_y)) > 1 else 0.5
                )

                train_aucs.append({"iteration": env.iteration, "value": train_auc})
                val_aucs.append({"iteration": env.iteration, "value": val_auc})

                job["progress"] = {
                    "current_iteration": env.iteration,
                    "total_iterations": 100,
                    "train_metrics": train_aucs.copy(),
                    "val_metrics": val_aucs.copy(),
                }

        from sklearn.metrics import roc_auc_score

        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[callback],
        )

        # Final predictions
        train_pred = model.predict(train_X)
        val_pred = model.predict(val_X)

        # Calculate final metrics
        train_metrics = calculate_metrics(train_y, train_pred)
        val_metrics = calculate_metrics(val_y, val_pred)

        # Feature importance
        importance = dict(
            zip(features, model.feature_importance(importance_type="gain").tolist())
        )

        # Store model info
        model_id = f"lgb_{job_id}"
        trained_models[model_id] = {
            "model": model,
            "model_type": "lightgbm",
            "features": features,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_importance": importance,
            "training_history": {"train_auc": train_aucs, "val_auc": val_aucs},
            "created_at": datetime.now().isoformat(),
            "name": job.get("name", f"LightGBM {job_id[:8]}"),
        }

        job["status"] = "completed"
        job["model_id"] = model_id
        job["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        import traceback

        job["traceback"] = traceback.format_exc()


async def train_mlp(job_id: str, features: List[str]):
    """Train MLP model using PyTorch Lightning."""
    job = training_jobs[job_id]

    try:
        train_X, train_y, val_X, val_y = prepare_data(features)

        # Convert to tensors
        train_X_t = torch.tensor(train_X)
        train_y_t = torch.tensor(train_y)
        val_X_t = torch.tensor(val_X)
        val_y_t = torch.tensor(val_y)

        # Create datasets
        train_dataset = TensorDataset(train_X_t, train_y_t)
        val_dataset = TensorDataset(val_X_t, val_y_t)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Create model
        model = MLPModel(input_dim=len(features))

        # Custom callback to track progress
        class ProgressCallback(pl.Callback):
            def __init__(self, job_dict):
                self.job = job_dict
                self.train_losses = []
                self.val_losses = []

            def on_train_epoch_end(self, trainer, pl_module):
                train_loss = trainer.callback_metrics.get("train_loss", 0)
                self.train_losses.append(
                    {
                        "epoch": trainer.current_epoch,
                        "value": float(train_loss) if train_loss else 0,
                    }
                )

            def on_validation_epoch_end(self, trainer, pl_module):
                val_loss = trainer.callback_metrics.get("val_loss", 0)
                self.val_losses.append(
                    {
                        "epoch": trainer.current_epoch,
                        "value": float(val_loss) if val_loss else 0,
                    }
                )

                self.job["progress"] = {
                    "current_epoch": trainer.current_epoch + 1,
                    "total_epochs": trainer.max_epochs,
                    "train_metrics": self.train_losses.copy(),
                    "val_metrics": self.val_losses.copy(),
                }

        progress_callback = ProgressCallback(job)

        # Train
        trainer = pl.Trainer(
            max_epochs=50,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[progress_callback],
            accelerator="cpu",  # Use CPU for small dataset
        )

        trainer.fit(model, train_loader, val_loader)

        # Final predictions
        model.eval()
        with torch.no_grad():
            train_pred = model(train_X_t).numpy()
            val_pred = model(val_X_t).numpy()

        # Calculate final metrics
        train_metrics = calculate_metrics(train_y, train_pred)
        val_metrics = calculate_metrics(val_y, val_pred)

        # Store model info (MLP doesn't have feature importance like LightGBM)
        model_id = f"mlp_{job_id}"
        trained_models[model_id] = {
            "model": model,
            "model_type": "mlp",
            "features": features,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_importance": None,  # MLP doesn't have built-in feature importance
            "training_history": {
                "train_loss": progress_callback.train_losses,
                "val_loss": progress_callback.val_losses,
            },
            "created_at": datetime.now().isoformat(),
            "name": job.get("name", f"MLP {job_id[:8]}"),
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        patient_count = db.get_patient_count()
    except Exception:
        patient_count = 0

    return {
        "status": "ok",
        "version": VERSION,
        "patient_count": patient_count,
        "models_loaded": len(trained_models),
    }


# ============== Authentication ==============

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SITL Dashboard - Login</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .login-container {
            text-align: center;
        }
        .login-box {
            background: rgba(22, 33, 62, 0.9);
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            width: 340px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        h1 {
            margin: 0 0 0.5rem 0;
            font-size: 1.75rem;
            background: linear-gradient(90deg, #4a90d9, #67b26f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #888;
            margin-bottom: 1.5rem;
            font-size: 0.9rem;
        }
        input {
            width: 100%;
            padding: 0.875rem 1rem;
            margin: 0.5rem 0;
            border: 1px solid #333;
            border-radius: 6px;
            background: #0f0f1a;
            color: #eee;
            font-size: 1rem;
            transition: border-color 0.2s;
        }
        input:focus {
            outline: none;
            border-color: #4a90d9;
        }
        button {
            width: 100%;
            padding: 0.875rem;
            margin-top: 1.25rem;
            background: linear-gradient(90deg, #4a90d9, #357abd);
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            transition: transform 0.1s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(74, 144, 217, 0.4);
        }
        button:active { transform: translateY(0); }
        .error {
            color: #ff6b6b;
            background: rgba(255, 107, 107, 0.1);
            padding: 0.75rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
        .logo {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-box">
            <div class="logo">🔬</div>
            <h1>SITL Dashboard</h1>
            <p class="subtitle">Scientist-in-the-Loop ML Platform</p>
            {error_html}
            <form action="/api/login" method="POST">
                <input type="hidden" name="next" value="{next_url}">
                <input type="text" name="username" placeholder="Username" required autofocus>
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Sign In</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

ALREADY_LOGGED_IN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SITL Dashboard - Already Logged In</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .login-container { text-align: center; }
        .login-box {
            background: rgba(22, 33, 62, 0.9);
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            width: 380px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        h1 {
            margin: 0 0 0.5rem 0;
            font-size: 1.75rem;
            background: linear-gradient(90deg, #4a90d9, #67b26f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #888; margin-bottom: 1.5rem; font-size: 0.9rem; }
        .username {
            color: #67b26f;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 1rem 0;
        }
        .message { color: #aaa; margin-bottom: 1.5rem; }
        .btn {
            display: inline-block;
            padding: 0.875rem 1.5rem;
            margin: 0.5rem;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 500;
            text-decoration: none;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.2s;
        }
        .btn:hover { transform: translateY(-1px); }
        .btn-primary {
            background: linear-gradient(90deg, #4a90d9, #357abd);
            color: white;
        }
        .btn-primary:hover { box-shadow: 0 4px 12px rgba(74, 144, 217, 0.4); }
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: #ccc;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .btn-secondary:hover { background: rgba(255,255,255,0.15); }
        .logo { font-size: 3rem; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-box">
            <div class="logo">🔬</div>
            <h1>SITL Dashboard</h1>
            <p class="subtitle">Scientist-in-the-Loop ML Platform</p>
            <p class="message">You are already logged in as:</p>
            <div class="username">{username}</div>
            <div style="margin-top: 1.5rem;">
                <a href="{next_url}" class="btn btn-primary">Continue to Dashboard</a>
                <a href="/logout" class="btn btn-secondary">Logout / Switch User</a>
            </div>
        </div>
    </div>
</body>
</html>
"""


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/", error: str = None):
    """Serve the login page."""
    # If already logged in, show options instead of auto-redirecting
    username = request.session.get("username")
    if username:
        html = ALREADY_LOGGED_IN_HTML.replace("{username}", username).replace("{next_url}", next)
        return HTMLResponse(content=html)

    error_html = ""
    if error:
        error_html = '<div class="error">Invalid username or password</div>'

    html = LOGIN_HTML.replace("{next_url}", next).replace("{error_html}", error_html)
    return HTMLResponse(content=html)


@app.post("/api/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    """Process login form submission."""
    if USERS.get(username) == password:
        request.session["username"] = username
        # Log successful login
        log_event(
            event_type=EventTypes.LOGIN,
            user_id=username,
            source="backend",
            payload={"success": True, "next": next},
            request=request,
        )
        return RedirectResponse(next, status_code=302)
    # Log failed login attempt
    log_event(
        event_type=EventTypes.LOGIN,
        user_id=username,
        source="backend",
        payload={"success": False, "reason": "invalid_credentials"},
        request=request,
    )
    # Invalid credentials - redirect back to login with error
    return RedirectResponse(f"/login?error=1&next={next}", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    """Clear session and redirect to login."""
    username = request.session.get("username")
    # Log logout
    if username:
        log_event(
            event_type=EventTypes.LOGOUT,
            user_id=username,
            source="backend",
            payload={},
            request=request,
        )
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/api/me")
async def get_current_user(request: Request):
    """Get current logged-in user info."""
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"username": username}


# ============== Analytics / Event Logging ==============


class FrontendEvent(BaseModel):
    """Event sent from frontend tracker."""

    event_type: str
    session_id: Optional[str] = None
    payload: Dict[str, Any] = {}
    timestamp: Optional[str] = None


@app.post("/api/analytics/log")
async def log_frontend_event(event: FrontendEvent, request: Request):
    """
    Receive and store events from frontend tracker.

    Events are stored with the authenticated user's username.
    """
    username = request.session.get("username")

    event_id = log_event(
        event_type=event.event_type,
        user_id=username,
        session_id=event.session_id,
        source="frontend",
        payload=event.payload,
        request=request,
    )

    return {"status": "ok", "event_id": event_id}


# ============== Dataset Selection ==============


@app.get("/api/datasets")
async def list_datasets():
    """
    List all available datasets for the homepage.
    Returns dataset metadata from data_adapter.
    """
    # Get datasets from data_adapter which supports synthetic, breast_cancer, diabetes
    datasets = []
    for dataset_id in AVAILABLE_DATASETS:
        try:
            config = get_dataset_config(dataset_id)
            datasets.append(
                {
                    "dataset_id": config.dataset_id,
                    "display_name": config.display_name,
                    "description": config.description,
                    "task_type": config.task_type,
                    "sample_count": config.sample_count,
                    "feature_count": len(config.features),
                    "tags": (
                        ["sklearn"]
                        if dataset_id != "synthetic"
                        else ["healthcare", "synthetic", "pcori"]
                    ),
                    "source": "sklearn" if dataset_id != "synthetic" else "synthetic",
                }
            )
        except Exception as e:
            print(f"Warning: Could not load dataset {dataset_id}: {e}")

    return {"datasets": datasets}


@app.get("/")
async def homepage():
    """Serve the homepage with dataset selection."""
    homepage_path = FRONTEND_PATH / "homepage.html"
    if homepage_path.exists():
        return FileResponse(homepage_path)
    # Fallback to dashboard if homepage doesn't exist yet
    return FileResponse(FRONTEND_PATH / "index.html")


@app.get("/sequential")
async def sequential_page():
    """Serve the sequential model training page for Health Facts."""
    sequential_path = FRONTEND_PATH / "sequential.html"
    if sequential_path.exists():
        return FileResponse(sequential_path)
    return {"error": "Sequential training page not found"}


@app.get("/dataset/{dataset_id}")
async def dataset_dashboard(dataset_id: str):
    """Serve the dashboard for a specific dataset."""
    # For now, serve the same dashboard - frontend will handle dataset_id
    return FileResponse(FRONTEND_PATH / "index.html")


# ============== Dynamic Dataset API ==============


# Health Facts Sequential config must be before the generic route
@app.get("/api/dataset/health_facts_seq/config")
async def get_health_facts_seq_config():
    """Get configuration for Health Facts Sequential training."""
    if not HAS_HEALTH_FACTS_SEQ:
        raise HTTPException(status_code=503, detail="Health Facts Sequential API not available")

    try:
        feature_sets = get_feature_sets()
        feature_sets_info = [
            {
                "name": name,
                "num_features": len(fs.feature_ids),
                "description": fs.description,
            }
            for name, fs in feature_sets.items()
        ]

        return {
            "feature_sets": feature_sets_info,
            "default_label": DEFAULT_LABEL,
            "supported_T": [5, 10, 15, 20],
            "supported_sample_fractions": [0.01, 0.1, 1.0],
            "supported_normalizations": ["standard", "minmax", "none"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dataset/health_facts_seq/train")
async def start_health_facts_seq_training(request: Request, background_tasks: BackgroundTasks):
    """Start Health Facts Sequential model training."""
    if not HAS_HEALTH_FACTS_SEQ:
        raise HTTPException(status_code=503, detail="Health Facts Sequential API not available")

    params = await request.json()
    job_id = str(uuid.uuid4())[:8]

    # Store job info
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "params": params,
        "created_at": datetime.now().isoformat(),
        "progress": 0,
        "epoch": 0,
        "train_loss": None,
        "val_loss": None,
        "train_losses": [],
        "val_losses": [],
        "metrics": None,
    }

    # Run training in a separate thread to avoid blocking the event loop
    import concurrent.futures
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, run_health_facts_seq_training_sync, job_id, params)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/dataset/health_facts_seq/training/{job_id}")
async def get_health_facts_seq_training_status(job_id: str):
    """Get status of a Health Facts Sequential training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return training_jobs[job_id]


@app.get("/api/dataset/health_facts_seq/jobs")
async def list_health_facts_seq_jobs():
    """List all Health Facts Sequential training jobs."""
    # Filter to only health_facts_seq jobs (those with our specific params structure)
    seq_jobs = [
        job for job in training_jobs.values()
        if job.get("params", {}).get("feature_set") is not None
    ]
    # Sort by created_at descending
    seq_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"jobs": seq_jobs}


@app.get("/api/dataset/{dataset_id}/config")
async def get_dataset_config_endpoint(dataset_id: str):
    """
    Get configuration for a specific dataset.
    Returns features, descriptions, task type, etc.
    """
    if dataset_id not in AVAILABLE_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found. Available: {AVAILABLE_DATASETS}",
        )

    config = get_dataset_config(dataset_id)
    return {
        "dataset_id": config.dataset_id,
        "display_name": config.display_name,
        "description": config.description,
        "task_type": config.task_type,
        "features": config.features,
        "feature_descriptions": config.feature_descriptions,
        "target_name": config.target_name,
        "target_description": config.target_description,
        "sample_count": config.sample_count,
    }


@app.get("/api/dataset/{dataset_id}/samples")
async def get_dataset_samples(dataset_id: str, limit: Optional[int] = None):
    """
    Get samples from a dataset for display.
    Returns list of samples with features, target, and split.
    """
    if dataset_id not in AVAILABLE_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found. Available: {AVAILABLE_DATASETS}",
        )

    samples = get_samples_for_display(dataset_id, limit=limit)
    config = get_dataset_config(dataset_id)

    return {
        "dataset_id": dataset_id,
        "task_type": config.task_type,
        "target_name": config.target_name,
        "samples": samples,
        "total_count": config.sample_count,  # Total in dataset, not limited count
        "returned_count": len(samples),
    }


class DatasetTrainRequest(BaseModel):
    """Request for training on a specific dataset."""

    features: List[str]
    model_type: str = "lightgbm"
    name: Optional[str] = None

    @validator("model_type")
    def validate_model_type(cls, v):
        if v not in ["lightgbm", "logreg"]:
            raise ValueError('model_type must be "lightgbm" or "logreg"')
        return v

    @validator("features")
    def validate_features(cls, v):
        if not v:
            raise ValueError("At least one feature required")
        return v


@app.post("/api/dataset/{dataset_id}/train")
async def train_on_dataset(
    dataset_id: str,
    request: DatasetTrainRequest,
    http_request: Request,
):
    """
    Train a model on a specific dataset.
    Works with classification (lightgbm, logreg) and regression datasets.
    """
    # Get username from session
    username = getattr(http_request.state, "username", "unknown")
    if dataset_id not in AVAILABLE_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found. Available: {AVAILABLE_DATASETS}",
        )

    config = get_dataset_config(dataset_id)
    start_time = time.time()

    # Validate features
    invalid_features = [f for f in request.features if f not in config.features]
    if invalid_features:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid features: {invalid_features}. Valid: {config.features}",
        )

    # Get training data
    X_train, y_train, X_val, y_val = get_training_data(dataset_id, request.features)

    # Check we have enough data
    if len(y_train) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough training data: {len(y_train)} samples",
        )

    # Train model based on task type
    if config.task_type == "classification":
        model, metrics, feature_importance = _train_classifier(
            X_train, y_train, X_val, y_val, request.features, request.model_type
        )
    else:  # regression
        model, metrics, feature_importance = _train_regressor(
            X_train, y_train, X_val, y_val, request.features, request.model_type
        )

    # Generate model ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"{dataset_id}_{request.model_type[:4]}_{timestamp}"

    # Store model in memory
    model_name = request.name or f"{config.display_name} - {request.model_type.upper()} {timestamp}"
    created_at = datetime.now().isoformat()

    trained_models[model_id] = {
        "model": model,
        "model_type": request.model_type,
        "dataset_id": dataset_id,
        "task_type": config.task_type,
        "features": request.features,
        "train_metrics": metrics["train"],
        "val_metrics": metrics["val"],
        "feature_importance": feature_importance,
        "training_history": {},
        "created_at": created_at,
        "name": model_name,
    }

    # Also save to database for persistence (both job and model)
    try:
        from .job_integration import DashboardJob, TrainedModel, save_job, save_model

        # Create job entry for Training Status tab
        job = DashboardJob(
            job_id=model_id,
            session_id="direct",  # Mark as direct training (not via scheduler)
            dataset_id=dataset_id,
            model_type=request.model_type,
            model_name=model_name,
            features=request.features,
            config={},
            status="completed",
            created_at=created_at,
            work_dir="",
            log_path=None,
            metrics_path=None,
            error=None,
            username=username,
        )
        save_job(job)

        # Create model entry for Model Results tab
        db_model = TrainedModel(
            model_id=model_id,
            job_id=model_id,
            dataset_id=dataset_id,
            model_type=request.model_type,
            model_name=model_name,
            task_type=config.task_type,
            features=request.features,
            train_metrics=metrics["train"],
            val_metrics=metrics["val"],
            feature_importance=feature_importance,
            training_history={},
            model_path=None,  # Inline training doesn't save model file
            work_dir="",
            created_at=created_at,
            username=username,
        )
        save_model(db_model)
        print(f"[train_on_dataset] Saved job and model {model_id} to database")
    except Exception as e:
        print(f"[train_on_dataset] Warning: Failed to save to database: {e}")

    training_time = time.time() - start_time

    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "model_type": request.model_type,
        "task_type": config.task_type,
        "train_size": len(y_train),
        "val_size": len(y_val),
        "metrics": metrics["val"],
        "feature_importance": feature_importance,
        "training_time_seconds": round(training_time, 2),
    }


def _train_classifier(X_train, y_train, X_val, y_val, features, model_type):
    """Train a classification model."""
    if model_type == "lightgbm":
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=features)
        val_data = lgb.Dataset(
            X_val, label=y_val, feature_name=features, reference=train_data
        )

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 15,
            "learning_rate": 0.1,
            "feature_fraction": 0.9,
            "verbose": -1,
            "seed": 42,
        }

        model = lgb.train(
            params, train_data, num_boost_round=100, valid_sets=[val_data]
        )

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Feature importance
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)
                sample_size = min(100, len(X_train))
                shap_values = explainer.shap_values(X_train[:sample_size])
                importance = np.abs(shap_values).mean(axis=0)
                feature_importance = [
                    {"feature": f, "importance": round(float(imp), 4)}
                    for f, imp in sorted(zip(features, importance), key=lambda x: -x[1])
                ]
            except Exception:
                gain = model.feature_importance(importance_type="gain")
                total = sum(gain) or 1
                feature_importance = [
                    {"feature": f, "importance": round(float(g / total), 4)}
                    for f, g in sorted(zip(features, gain), key=lambda x: -x[1])
                ]
        else:
            gain = model.feature_importance(importance_type="gain")
            total = sum(gain) or 1
            feature_importance = [
                {"feature": f, "importance": round(float(g / total), 4)}
                for f, g in sorted(zip(features, gain), key=lambda x: -x[1])
            ]

    else:  # logreg
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        model.scaler_ = scaler

        train_pred = model.predict_proba(X_train_scaled)[:, 1]
        val_pred = model.predict_proba(X_val_scaled)[:, 1]

        coeffs = np.abs(model.coef_[0])
        total = sum(coeffs) or 1
        feature_importance = [
            {"feature": f, "importance": round(float(c / total), 4)}
            for f, c in sorted(zip(features, coeffs), key=lambda x: -x[1])
        ]

    metrics = {
        "train": {
            "auroc": float(roc_auc_score(y_train, train_pred)),
            "auprc": float(average_precision_score(y_train, train_pred)),
            "accuracy": float(np.mean((train_pred >= 0.5) == y_train)),
        },
        "val": {
            "auroc": float(roc_auc_score(y_val, val_pred)),
            "auprc": float(average_precision_score(y_val, val_pred)),
            "accuracy": float(np.mean((val_pred >= 0.5) == y_val)),
        },
    }

    return model, metrics, feature_importance


def _train_regressor(X_train, y_train, X_val, y_val, features, model_type):
    """Train a regression model."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if model_type == "lightgbm":
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=features)
        val_data = lgb.Dataset(
            X_val, label=y_val, feature_name=features, reference=train_data
        )

        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 15,
            "learning_rate": 0.1,
            "feature_fraction": 0.9,
            "verbose": -1,
            "seed": 42,
        }

        model = lgb.train(
            params, train_data, num_boost_round=100, valid_sets=[val_data]
        )

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Feature importance
        gain = model.feature_importance(importance_type="gain")
        total = sum(gain) or 1
        feature_importance = [
            {"feature": f, "importance": round(float(g / total), 4)}
            for f, g in sorted(zip(features, gain), key=lambda x: -x[1])
        ]

    else:  # logreg -> use Ridge for regression
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = Ridge(random_state=42)
        model.fit(X_train_scaled, y_train)
        model.scaler_ = scaler

        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)

        coeffs = np.abs(model.coef_)
        total = sum(coeffs) or 1
        feature_importance = [
            {"feature": f, "importance": round(float(c / total), 4)}
            for f, c in sorted(zip(features, coeffs), key=lambda x: -x[1])
        ]

    metrics = {
        "train": {
            "rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
            "mae": float(mean_absolute_error(y_train, train_pred)),
            "r2": float(r2_score(y_train, train_pred)),
        },
        "val": {
            "rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
            "mae": float(mean_absolute_error(y_val, val_pred)),
            "r2": float(r2_score(y_val, val_pred)),
        },
    }

    return model, metrics, feature_importance


# ============== Analysis Tab Endpoints ==============


@app.get("/api/dataset/{dataset_id}/features")
async def get_dataset_features(dataset_id: str):
    """
    Get all features with descriptions and basic statistics.
    Used by the Analysis tab to populate the feature list.
    """
    if dataset_id not in AVAILABLE_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found. Available: {AVAILABLE_DATASETS}",
        )

    config = get_dataset_config(dataset_id)
    X, y, _ = load_dataset(dataset_id)

    features_info = []
    for feat in config.features:
        col = X[feat]
        features_info.append(
            {
                "name": feat,
                "description": config.feature_descriptions.get(feat, ""),
                "dtype": str(col.dtype),
                "stats": {
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "median": float(col.median()),
                    "missing": int(col.isna().sum()),
                },
            }
        )

    return {
        "dataset_id": dataset_id,
        "task_type": config.task_type,
        "target_name": config.target_name,
        "sample_count": len(X),
        "features": features_info,
    }


@app.get("/api/dataset/{dataset_id}/feature/{feature_name}/histogram")
async def get_feature_histogram(
    dataset_id: str,
    feature_name: str,
    clip_pct: float = 0.0,
    n_bins: int = 30,
    sample_size: int = 100,
):
    """
    Generate a histogram for a feature.

    Args:
        dataset_id: Dataset to use
        feature_name: Feature to plot
        clip_pct: Percentile to clip at both ends (0-10). E.g., 1 means clip at 1% and 99%
        n_bins: Number of histogram bins
        sample_size: Number of samples to use (0 = all)

    Returns:
        Base64 encoded PNG image
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    if dataset_id not in AVAILABLE_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    config = get_dataset_config(dataset_id)
    if feature_name not in config.features:
        raise HTTPException(
            status_code=404, detail=f"Feature '{feature_name}' not found"
        )

    X, y, _ = load_dataset(dataset_id)
    values = X[feature_name].values

    # Sample if requested
    if sample_size > 0 and sample_size < len(values):
        np.random.seed(42)
        indices = np.random.choice(len(values), sample_size, replace=False)
        values = values[indices]
        y_sample = y.iloc[indices].values if hasattr(y, "iloc") else y[indices]
    else:
        y_sample = y.values if hasattr(y, "values") else y
        sample_size = len(values)

    # Clip outliers
    clip_pct = max(0, min(10, clip_pct))  # Clamp to 0-10%
    if clip_pct > 0:
        lower = np.percentile(values, clip_pct)
        upper = np.percentile(values, 100 - clip_pct)
        values = np.clip(values, lower, upper)

    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    # Color by target for classification
    if config.task_type == "classification":
        mask_0 = y_sample == 0
        mask_1 = y_sample == 1
        ax.hist(
            values[mask_0],
            bins=n_bins,
            alpha=0.6,
            label="Class 0",
            color="#58a6ff",
            edgecolor="#58a6ff",
        )
        ax.hist(
            values[mask_1],
            bins=n_bins,
            alpha=0.6,
            label="Class 1",
            color="#f85149",
            edgecolor="#f85149",
        )
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    else:
        ax.hist(values, bins=n_bins, alpha=0.7, color="#58a6ff", edgecolor="#58a6ff")

    ax.set_xlabel(feature_name, color="#c9d1d9")
    ax.set_ylabel("Count", color="#c9d1d9")
    ax.set_title(f"{feature_name} Distribution (n={sample_size})", color="#c9d1d9")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.2, color="#30363d")

    # Add stats annotation
    stats_text = f"Mean: {np.mean(values):.2f}\nStd: {np.std(values):.2f}\nMin: {np.min(values):.2f}\nMax: {np.max(values):.2f}"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round", facecolor="#21262d", edgecolor="#30363d", alpha=0.9
        ),
        color="#c9d1d9",
    )

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, facecolor="#0d1117", edgecolor="none")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    return {
        "feature": feature_name,
        "clip_pct": clip_pct,
        "sample_size": sample_size,
        "image": f"data:image/png;base64,{img_base64}",
        "stats": {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        },
    }


@app.get("/api/patients")
async def get_patients():
    """Get all patients with basic info."""
    return {
        "patients": [
            {
                "id": p["id"],
                "name": p["name"],
                "split": p["split"],
                "outcome": p["outcome"],
            }
            for p in PATIENTS.values()
        ],
        "features": FEATURE_NAMES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
    }


@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str):
    """Get detailed patient info with predictions from all models."""
    if patient_id not in PATIENTS:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient = PATIENTS[patient_id]

    # Get predictions from all trained models
    predictions = {}
    for model_id, model_info in trained_models.items():
        features = model_info["features"]
        x = np.array([[patient["features"][f] for f in features]], dtype=np.float32)

        if model_info["model_type"] == "lightgbm":
            pred = float(model_info["model"].predict(x)[0])
        elif model_info["model_type"] == "logreg":
            # LogReg requires scaled features and uses predict_proba
            model = model_info["model"]
            # Guard for missing scaler (safety check)
            scaler = getattr(model, "scaler_", None)
            x_scaled = scaler.transform(x) if scaler is not None else x
            pred = float(model.predict_proba(x_scaled)[0, 1])
        elif model_info["model_type"] == "mlp":
            model_info["model"].eval()
            with torch.no_grad():
                output = model_info["model"](torch.tensor(x)).numpy()
                # Handle both 0-d and 1-d arrays (squeeze removes batch dim when batch=1)
                pred = float(output.item() if output.ndim == 0 else output[0])
        else:
            # Skip unknown model types with warning
            import logging

            logging.warning(
                f"Unknown model type: {model_info['model_type']} - skipping predictions"
            )
            continue

        predictions[model_id] = {
            "probability": round(pred, 4),
            "prediction": int(pred >= 0.5),
            "model_name": model_info["name"],
            "model_type": model_info["model_type"],
        }

    return {
        "patient": patient,
        "predictions": predictions,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
    }


@app.post("/api/train")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start model training."""
    # Validate features
    invalid_features = [f for f in request.features if f not in FEATURE_NAMES]
    if invalid_features:
        raise HTTPException(
            status_code=400, detail=f"Invalid features: {invalid_features}"
        )

    if not request.features:
        raise HTTPException(status_code=400, detail="At least one feature required")

    # Create job
    job_id = str(uuid.uuid4())[:8]
    training_jobs[job_id] = {
        "id": job_id,
        "model_type": request.model_type,
        "features": request.features,
        "name": request.name or f"{request.model_type.upper()} {job_id}",
        "status": "running",
        "progress": {},
        "started_at": datetime.now().isoformat(),
    }

    # Start training in background
    if request.model_type == "lightgbm":
        background_tasks.add_task(train_lightgbm, job_id, request.features)
    elif request.model_type == "mlp":
        background_tasks.add_task(train_mlp, job_id, request.features)
    else:
        raise HTTPException(
            status_code=400, detail=f"Unknown model type: {request.model_type}"
        )

    return {"job_id": job_id, "status": "started"}


@app.get("/api/training/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status and progress."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]


@app.get("/api/models")
async def get_models(
    dataset_id: Optional[str] = None,
    model_type: Optional[str] = None,
    search: Optional[str] = None,
):
    """Get all trained models from database with optional filtering."""
    from .job_integration import list_models, model_to_dict

    models = list_models(
        dataset_id=dataset_id,
        model_type=model_type,
        search_query=search,
        limit=100,
    )

    return {model.model_id: model_to_dict(model) for model in models}


@app.get("/api/models/{model_id}")
async def get_model_by_id(model_id: str):
    """Get specific model details from database."""
    from .job_integration import get_model, model_to_dict

    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return model_to_dict(model)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with GPT about models/patients."""
    # Build context message
    context_str = ""
    if request.context:
        context_str = f"\n\nContext provided:\n{json.dumps(request.context, indent=2)}"

    system_prompt = f"""You are a clinical ML assistant helping users understand OUD (opioid use disorder) risk prediction models.

Available features and their descriptions:
{json.dumps(FEATURE_DESCRIPTIONS, indent=2)}

You have access to the following trained models:
{json.dumps([{"id": k, "type": v["model_type"], "name": v["name"], "features": v["features"]} for k, v in trained_models.items()], indent=2)}

When discussing feature importance:
- LightGBM provides feature importance based on gain (how much each feature improves predictions)
- MLP (neural networks) don't have built-in feature importance; we'd need techniques like SHAP or permutation importance

Be concise but helpful. If asked about specific patients or models, use the context provided.
{context_str}"""

    # Try each model in the fallback chain
    last_error = None
    for model_name in OPENAI_CHAT_MODELS:
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.message},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return ChatResponse(response=response.choices[0].message.content)
        except Exception as e:
            last_error = e
            continue  # Try next model in fallback chain

    # All models failed
    return ChatResponse(
        response=f"Error: All models unavailable. Last error: {str(last_error)}"
    )


# ============== Phase A: New Endpoints ==============


@app.post("/api/cohort/filter")
async def filter_cohort(request: CohortFilterRequest):
    """
    Filter patients based on criteria.
    Returns patient IDs, counts, and statistics.
    """
    result = db.filter_patients(
        age_min=request.age_min,
        age_max=request.age_max,
        cancer_status=request.cancer_status,
        opioid_threshold=request.opioid_threshold,
        mental_health_min=request.mental_health_min,
    )
    return result


@app.post("/api/train/quick")
async def train_quick(request: QuickTrainRequest):
    """
    Train a model quickly on filtered cohort.
    Supports lightgbm and logreg.
    Returns model with SHAP feature importance.
    """
    start_time = time.time()

    # Get filtered patients
    filters = request.filters or {}
    cohort = db.filter_patients(
        age_min=filters.get("age_min"),
        age_max=filters.get("age_max"),
        cancer_status=filters.get("cancer_status"),
        opioid_threshold=filters.get("opioid_threshold"),
        mental_health_min=filters.get("mental_health_min"),
    )

    # Check cohort size
    if cohort["total_count"] < MIN_COHORT_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Cohort too small for training. Need >= {MIN_COHORT_SIZE} patients, got {cohort['total_count']}.",
        )

    if cohort["event_count"] < MIN_EVENT_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Too few events for training. Need >= {MIN_EVENT_COUNT} events, got {cohort['event_count']}.",
        )

    non_event_count = cohort["total_count"] - cohort["event_count"]
    if non_event_count < MIN_EVENT_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Too few non-events for training. Need >= {MIN_EVENT_COUNT}, got {non_event_count}.",
        )

    # Get training data
    X_train, y_train, X_val, y_val, feature_names = db.get_training_data(
        cohort["patient_ids"], request.features
    )

    # Check we have both classes in training data
    if len(np.unique(y_train)) < 2:
        raise HTTPException(
            status_code=400,
            detail="Training data must contain both positive and negative examples.",
        )

    # Train model
    if request.model_type == "lightgbm":
        model, metrics, feature_importance = train_lightgbm_quick(
            X_train, y_train, X_val, y_val, request.features
        )
    else:  # logreg
        model, metrics, feature_importance = train_logreg_quick(
            X_train, y_train, X_val, y_val, request.features
        )

    # Generate model ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"{request.model_type[:4]}_{timestamp}"

    # Store model
    trained_models[model_id] = {
        "model": model,
        "model_type": request.model_type,
        "features": request.features,
        "train_metrics": metrics["train"],
        "val_metrics": metrics["val"],
        "feature_importance": feature_importance,
        "cohort_filters": filters,
        "cohort_size": cohort["total_count"],
        "training_history": {},
        "created_at": datetime.now().isoformat(),
        "name": request.name or f"{request.model_type.upper()} {timestamp}",
    }

    training_time = time.time() - start_time

    return {
        "model_id": model_id,
        "model_type": request.model_type,
        "cohort_size": cohort["total_count"],
        "train_size": len(y_train),
        "test_size": len(y_val),
        "event_count": cohort["event_count"],
        "metrics": metrics["val"],
        "feature_importance": feature_importance,
        "training_time_seconds": round(training_time, 2),
        "warnings": cohort["warnings"],
    }


def train_lightgbm_quick(X_train, y_train, X_val, y_val, features):
    """Train LightGBM model synchronously with SHAP."""
    from sklearn.linear_model import LogisticRegression

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=features)
    val_data = lgb.Dataset(
        X_val, label=y_val, feature_name=features, reference=train_data
    )

    # Parameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "feature_fraction": 0.9,
        "verbose": -1,
        "seed": 42,
    }

    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
    )

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Metrics
    metrics = {
        "train": {
            "auroc": float(roc_auc_score(y_train, train_pred)),
            "auprc": float(average_precision_score(y_train, train_pred)),
            "accuracy": float(np.mean((train_pred >= 0.5) == y_train)),
        },
        "val": {
            "auroc": float(roc_auc_score(y_val, val_pred)),
            "auprc": float(average_precision_score(y_val, val_pred)),
            "accuracy": float(np.mean((val_pred >= 0.5) == y_val)),
        },
    }

    # Feature importance using SHAP or gain
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            # Use subset for faster computation
            sample_size = min(SHAP_MAX_SAMPLES, len(X_train))
            shap_values = explainer.shap_values(X_train[:sample_size])
            importance = np.abs(shap_values).mean(axis=0)
            feature_importance = [
                {"feature": f, "importance": round(float(imp), 4)}
                for f, imp in sorted(zip(features, importance), key=lambda x: -x[1])
            ]
        except Exception as e:
            print(f"SHAP failed: {e}, using gain importance")
            gain_importance = model.feature_importance(importance_type="gain")
            total = sum(gain_importance)
            feature_importance = [
                {"feature": f, "importance": round(float(imp / total), 4)}
                for f, imp in sorted(
                    zip(features, gain_importance), key=lambda x: -x[1]
                )
            ]
    else:
        gain_importance = model.feature_importance(importance_type="gain")
        total = sum(gain_importance)
        feature_importance = [
            {"feature": f, "importance": round(float(imp / total), 4)}
            for f, imp in sorted(zip(features, gain_importance), key=lambda x: -x[1])
        ]

    return model, metrics, feature_importance


def train_logreg_quick(X_train, y_train, X_val, y_val, features):
    """Train Logistic Regression model synchronously."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Store scaler with model
    model.scaler_ = scaler

    # Predictions
    train_pred = model.predict_proba(X_train_scaled)[:, 1]
    val_pred = model.predict_proba(X_val_scaled)[:, 1]

    # Metrics
    metrics = {
        "train": {
            "auroc": float(roc_auc_score(y_train, train_pred)),
            "auprc": float(average_precision_score(y_train, train_pred)),
            "accuracy": float(np.mean((train_pred >= 0.5) == y_train)),
        },
        "val": {
            "auroc": float(roc_auc_score(y_val, val_pred)),
            "auprc": float(average_precision_score(y_val, val_pred)),
            "accuracy": float(np.mean((val_pred >= 0.5) == y_val)),
        },
    }

    # Feature importance from coefficients
    coefficients = np.abs(model.coef_[0])
    total = sum(coefficients)
    feature_importance = [
        {"feature": f, "importance": round(float(imp / total), 4)}
        for f, imp in sorted(zip(features, coefficients), key=lambda x: -x[1])
    ]

    return model, metrics, feature_importance


@app.get("/api/model/{model_id}/explain/patient/{patient_id}")
async def explain_patient(model_id: str, patient_id: str):
    """
    Get SHAP explanation for a specific patient.
    """
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    model_info = trained_models[model_id]

    # Get patient from database
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Prepare features
    features = model_info["features"]
    X = np.array([[patient[f] for f in features]], dtype=np.float32)

    # Get prediction
    if model_info["model_type"] == "lightgbm":
        prediction = float(model_info["model"].predict(X)[0])

        # SHAP explanation
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model_info["model"])
                shap_values = explainer.shap_values(X)
                base_value = float(explainer.expected_value)
                contributions = [
                    {
                        "feature": f,
                        "value": float(X[0, i]),
                        "contribution": round(float(shap_values[0, i]), 4),
                    }
                    for i, f in enumerate(features)
                ]
            except Exception as e:
                # Fallback to feature importance
                contributions = [
                    {"feature": f, "value": float(X[0, i]), "contribution": 0}
                    for i, f in enumerate(features)
                ]
                base_value = 0
        else:
            contributions = [
                {"feature": f, "value": float(X[0, i]), "contribution": 0}
                for i, f in enumerate(features)
            ]
            base_value = 0

    elif model_info["model_type"] == "logreg":
        model = model_info["model"]
        X_scaled = model.scaler_.transform(X)
        prediction = float(model.predict_proba(X_scaled)[0, 1])

        # Linear SHAP: coefficient * (feature_value - mean)
        coefficients = model.coef_[0]
        base_value = float(model.intercept_[0])
        contributions = [
            {
                "feature": f,
                "value": float(X[0, i]),
                "contribution": round(float(coefficients[i] * X_scaled[0, i]), 4),
            }
            for i, f in enumerate(features)
        ]

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Explanation not supported for model type: {model_info['model_type']}",
        )

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    return {
        "patient_id": patient_id,
        "model_id": model_id,
        "prediction": round(prediction, 4),
        "base_value": round(base_value, 4),
        "contributions": contributions,
    }


# ============== Claude Sandbox Analysis ==============

ANALYSIS_SYSTEM_PROMPT = """You are a data analysis assistant for a medical research dashboard.
You write Python code to analyze datasets. Your output will be embedded in a web UI.

DATASET CONTEXT:
- Dataset: {dataset_name}
- Columns: {columns}
- Column types: {column_types}
- Sample size: {sample_count}
- Task type: {task_type} (target: {target_name})
- Sample statistics: {sample_stats}

DATA ACCESS:
- `df` = pandas DataFrame with all features AND target column
- `y` = target Series (same as df['{target_name}'])
- Available imports: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns), sklearn

CRITICAL OUTPUT RULES - ALL output must be embeddable HTML or images:
1. PLOTS: Use plt.figure() - plots are auto-captured as PNG images
2. TABLES: Use print(dataframe.to_html()) - renders as HTML table
3. TEXT: Wrap in HTML tags: print("<p>Your text here</p>") or print("<pre>text</pre>")
4. METRICS: Use HTML: print(f"<div><b>AUROC:</b> {{auroc:.3f}}</div>")
5. NEVER use plain print() for text - always wrap in HTML tags

SECURITY:
- DO NOT import os, subprocess, sys, shutil, or network libraries
- DO NOT read/write files - work only with df and y
- Keep code under 30 lines

CODE FORMAT:
```python
# Your analysis code here
```

After the code block, briefly explain what the results show."""


@app.post(
    "/api/dataset/{dataset_id}/analysis/chat", response_model=AnalysisChatResponse
)
async def analysis_chat(dataset_id: str, request: AnalysisChatRequest):
    """
    Claude-powered interactive data analysis.

    1. Send user request + dataset context to Claude
    2. Claude generates pandas/matplotlib code
    3. Execute code in sandbox subprocess
    4. Return results (text, plots, tables)
    """
    # Validate dataset
    if dataset_id not in AVAILABLE_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found. Available: {AVAILABLE_DATASETS}",
        )

    # Load dataset for context
    try:
        X, y, feature_names = load_dataset(dataset_id)
        config = get_dataset_config(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    # Build context for Claude - include sample statistics so Claude knows the data
    columns = list(X.columns)
    column_types = {col: str(X[col].dtype) for col in columns}

    # Compute sample statistics for context (Claude sees this, not the raw data)
    sample_stats = {}
    for col in columns:
        sample_stats[col] = {
            "mean": round(float(X[col].mean()), 2),
            "std": round(float(X[col].std()), 2),
            "min": round(float(X[col].min()), 2),
            "max": round(float(X[col].max()), 2),
        }

    system_prompt = ANALYSIS_SYSTEM_PROMPT.format(
        dataset_name=config.display_name,
        columns=columns,
        column_types=column_types,
        sample_count=len(X),
        task_type=config.task_type,
        target_name=config.target_name,
        sample_stats=json.dumps(sample_stats, indent=2),
    )

    # Check if Claude is available
    if not anthropic_client:
        # Return helpful message if no API key
        return AnalysisChatResponse(
            response_text="Claude API not configured. To enable interactive analysis:\n\n"
            "1. Get an API key from console.anthropic.com\n"
            "2. Set ANTHROPIC_API_KEY environment variable\n"
            "3. Restart the server\n\n"
            f"Your request: {request.message}",
            code_generated=None,
            execution_result=None,
            error="ANTHROPIC_API_KEY not set",
        )

    try:
        # Build messages with conversation history
        messages = []
        for msg in request.conversation_history or []:
            messages.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )
        messages.append({"role": "user", "content": request.message})

        # Call Claude
        response = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        )

        response_text = response.content[0].text

        # Extract code from response
        code = extract_code_from_response(response_text)

        if not code:
            # No code generated - just return the text response
            return AnalysisChatResponse(
                response_text=response_text,
                code_generated=None,
                execution_result=None,
            )

        # Validate code safety
        is_safe, safety_error = validate_code_safety(code)
        if not is_safe:
            return AnalysisChatResponse(
                response_text=response_text,
                code_generated=code,
                execution_result=None,
                error=f"Code safety check failed: {safety_error}",
            )

        # Prepare data for sandbox
        import pandas as pd

        df = pd.concat([X, y.rename(config.target_name)], axis=1)
        pickle_path = prepare_data_for_sandbox(df, y)

        try:
            # Execute in sandbox
            result = execute_in_sandbox(code, pickle_path, timeout=30)

            # Build HTML result
            execution_html = ""
            if result.stdout:
                # Check if stdout contains HTML
                if "<table" in result.stdout or "<div" in result.stdout:
                    execution_html = result.stdout
                else:
                    execution_html = f"<pre>{result.stdout}</pre>"

            return AnalysisChatResponse(
                response_text=response_text,
                code_generated=code,
                execution_result=execution_html if execution_html else None,
                plot_base64=result.plot_base64,
                error=result.error,
                execution_time_seconds=result.execution_time_seconds,
            )

        finally:
            cleanup_pickle_file(pickle_path)

    except Exception as e:
        return AnalysisChatResponse(
            response_text=f"Error calling Claude API: {str(e)}",
            error=str(e),
        )


# ============== Job Scheduler Training ==============


@app.post("/api/train/submit")
async def submit_scheduled_training(request: ScheduledTrainRequest, http_request: Request):
    """
    Submit a training job to the scheduler.

    Returns immediately with job_id. Use /api/jobs/{job_id}/status to check progress.
    """
    # Get username from session
    username = getattr(http_request.state, "username", "unknown")

    # Validate dataset
    if request.dataset_id not in AVAILABLE_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{request.dataset_id}' not found. Available: {AVAILABLE_DATASETS}",
        )

    # Validate model type
    if request.model_type not in ["lightgbm", "logreg", "pytorch"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {request.model_type}. Valid: lightgbm, logreg, pytorch",
        )

    # Load data
    try:
        X_train, y_train, X_val, y_val = get_training_data(
            request.dataset_id,
            request.features,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load training data: {e}"
        )

    # Generate model name
    model_name = request.name or f"{request.model_type}_{request.dataset_id}"

    # Get dataset config for task type
    dataset_config = get_dataset_config(request.dataset_id)

    # Callback to register model when training completes
    def on_training_complete(job_id, result):
        """Register trained model in the database for Model Results tab."""
        import traceback

        try:
            from .job_integration import TrainedModel, get_job, save_model

            print(
                f"[on_training_complete] Called for job {job_id}, success={result.success}"
            )

            if not result.success:
                print(f"[on_training_complete] Training failed: {result.error}")
                return

            if not result.metrics:
                print(f"[on_training_complete] No metrics in result")
                return
            # Get job info for work_dir
            job_info = get_job(job_id)
            work_dir = job_info.work_dir if job_info else ""

            # Create and save model to database
            model = TrainedModel(
                model_id=job_id,
                job_id=job_id,
                dataset_id=request.dataset_id,
                model_type=request.model_type,
                model_name=model_name,
                task_type=dataset_config.task_type,
                features=request.features,
                train_metrics={
                    k.replace("train_", ""): v
                    for k, v in result.metrics.items()
                    if k.startswith("train_")
                },
                val_metrics={
                    k.replace("val_", ""): v
                    for k, v in result.metrics.items()
                    if k.startswith("val_")
                },
                feature_importance=[],  # Not available for async jobs
                training_history=result.training_history,
                model_path=result.model_path,
                work_dir=work_dir,
                created_at=datetime.now().isoformat(),
                username=username,  # Captured from outer scope
            )
            save_model(model)
            print(f"[on_training_complete] Saved model {job_id} to database")

        except Exception as e:
            print(f"[on_training_complete] ERROR: {e}")
            traceback.print_exc()

    # Submit job asynchronously
    job = submit_training_job_async(
        dataset_id=request.dataset_id,
        model_type=request.model_type,
        model_name=model_name,
        features=request.features,
        config=request.config or {},
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        on_complete=on_training_complete,
        username=username,
    )

    # Log training submission event
    log_event(
        event_type=EventTypes.TRAIN_SUBMITTED,
        user_id=username,
        source="backend",
        payload={
            "job_id": job.job_id,
            "dataset_id": request.dataset_id,
            "model_type": request.model_type,
            "model_name": model_name,
            "feature_count": len(request.features),
            "features": request.features,
        },
        request=http_request,
    )

    return {
        "job_id": job.job_id,
        "session_id": job.session_id,
        "status": job.status,
        "message": f"Training job submitted. Monitor at /api/jobs/{job.job_id}/status",
    }


@app.get("/api/jobs")
async def list_training_jobs(dataset_id: Optional[str] = None, limit: int = 50):
    """List training jobs, optionally filtered by dataset."""
    jobs = list_jobs(dataset_id=dataset_id, limit=limit)

    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "session_id": j.session_id,
                "dataset_id": j.dataset_id,
                "model_type": j.model_type,
                "model_name": j.model_name,
                "status": j.status,
                "created_at": j.created_at,
                "error": j.error,
            }
            for j in jobs
        ],
        "count": len(jobs),
    }


@app.get("/api/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status_endpoint(job_id: str):
    """Get status of a training job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return JobStatusResponse(
        job_id=job.job_id,
        session_id=job.session_id,
        dataset_id=job.dataset_id,
        model_type=job.model_type,
        model_name=job.model_name,
        status=job.status,
        created_at=job.created_at,
        error=job.error,
    )


@app.get("/api/jobs/{job_id}/log", response_model=JobLogResponse)
async def get_job_log_endpoint(job_id: str, lines: int = 100):
    """Get last N lines of a job's log file."""
    result = get_job_log(job_id, lines=lines)

    return JobLogResponse(
        log=result.get("log", ""),
        log_path=result.get("log_path"),
        lines_returned=result.get("lines_returned", 0),
        error=result.get("error"),
    )


@app.get("/api/jobs/{job_id}/training_curves", response_model=TrainingCurvesResponse)
async def get_training_curves_endpoint(job_id: str):
    """Get training curves data for visualization."""
    result = get_training_curves(job_id)

    return TrainingCurvesResponse(
        source=result.get("source"),
        train=result.get("train", []),
        val=result.get("val", []),
        error=result.get("error"),
    )


# ============== Health Facts Sequential Training Background Task ==============

def run_health_facts_seq_training_sync(job_id: str, params: dict):
    """Synchronous training function to run in a thread pool."""
    job = training_jobs[job_id]
    job["status"] = "running"

    try:
        # Extract parameters
        feature_set = params.get("feature_set", "core")
        T = params.get("T", 10)
        sample_fraction = params.get("sample_fraction", 0.01)
        model_type = params.get("model_type", "lstm")
        hidden_size = params.get("hidden_size", 128)
        max_epochs = params.get("max_epochs", 10)
        batch_size = params.get("batch_size", 256)
        use_attention = params.get("use_attention", False)
        use_mmap = params.get("use_mmap", True)

        # Load data
        job["status"] = "loading_data"
        data = get_health_facts_sequences(
            T=T,
            feature_set=feature_set,
            sample_fraction=sample_fraction,
            load_arrays=not use_mmap,
        )

        # Load arrays (mmap or full)
        if use_mmap:
            X_train = np.load(data["X_train_path"], mmap_mode="r")
            y_train = np.load(data["y_train_path"], mmap_mode="r")
            X_val = np.load(data["X_val_path"], mmap_mode="r")
            y_val = np.load(data["y_val_path"], mmap_mode="r")
        else:
            X_train, y_train = data["X_train"], data["y_train"]
            X_val, y_val = data["X_val"], data["y_val"]

        D = X_train.shape[2]
        job["data_info"] = {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "T": T,
            "D": D,
            "feature_set": feature_set,
        }

        # Build model
        job["status"] = "building_model"

        if model_type == "lstm":
            rnn = nn.LSTM(D, hidden_size, batch_first=True, num_layers=2, dropout=0.3)
        else:
            rnn = nn.GRU(D, hidden_size, batch_first=True, num_layers=2, dropout=0.3)

        class SeqModel(nn.Module):
            def __init__(self, rnn, hidden_size, use_attention):
                super().__init__()
                self.rnn = rnn
                self.use_attention = use_attention
                if use_attention:
                    self.attn = nn.Linear(hidden_size, 1)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                if self.use_attention:
                    attn_weights = torch.softmax(self.attn(out), dim=1)
                    out = (out * attn_weights).sum(dim=1)
                else:
                    out = out[:, -1, :]
                return self.fc(out).squeeze(-1)

        model = SeqModel(rnn, hidden_size, use_attention)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        job["status"] = "training"
        n_train = len(X_train)

        # For efficiency with mmap, use larger effective batch and sequential access
        # Shuffle indices once, then iterate sequentially
        effective_batch = max(batch_size, 2048)  # Use at least 2048 for efficiency

        for epoch in range(max_epochs):
            model.train()
            train_losses = []

            # Shuffle indices once per epoch
            indices = np.random.permutation(n_train)

            # Process in larger chunks for efficiency, then sub-batch for GPU
            chunk_size = min(50000, n_train)  # Load 50k samples at a time
            n_chunks = (n_train + chunk_size - 1) // chunk_size

            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, n_train)
                chunk_indices = indices[chunk_start:chunk_end]

                # Load entire chunk into memory (sequential mmap access is faster)
                X_chunk = np.array(X_train[sorted(chunk_indices)])
                y_chunk = np.array(y_train[sorted(chunk_indices)])

                # Shuffle within chunk
                perm = np.random.permutation(len(X_chunk))
                X_chunk = X_chunk[perm]
                y_chunk = y_chunk[perm]

                # Mini-batch within chunk
                for start in range(0, len(X_chunk), effective_batch):
                    end = min(start + effective_batch, len(X_chunk))
                    X_batch = torch.tensor(X_chunk[start:end], dtype=torch.float32).to(device)
                    y_batch = torch.tensor(y_chunk[start:end], dtype=torch.float32).to(device)

                    optimizer.zero_grad()
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                # Update progress within epoch
                job["progress"] = int((epoch + (chunk_idx + 1) / n_chunks) / max_epochs * 100)

            avg_train_loss = np.mean(train_losses)

            # Validation
            model.eval()
            val_losses = []
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for start in range(0, len(X_val), batch_size):
                    end = min(start + batch_size, len(X_val))
                    X_batch = torch.tensor(X_val[start:end], dtype=torch.float32).to(device)
                    y_batch = torch.tensor(y_val[start:end], dtype=torch.float32).to(device)

                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_losses.append(loss.item())
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())

            avg_val_loss = np.mean(val_losses)

            # Compute metrics
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            try:
                auc = roc_auc_score(val_labels, val_preds)
                auprc = average_precision_score(val_labels, val_preds)
            except:
                auc = 0.0
                auprc = 0.0

            # Update job status
            job["epoch"] = epoch + 1
            job["progress"] = int((epoch + 1) / max_epochs * 100)
            job["train_loss"] = avg_train_loss
            job["val_loss"] = avg_val_loss
            job["train_losses"].append(avg_train_loss)
            job["val_losses"].append(avg_val_loss)
            job["metrics"] = {"auc": auc, "auprc": auprc}

        job["status"] = "completed"

    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()


# Serve static files
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

if __name__ == "__main__":
    import uvicorn

    from .config import API_HOST, API_PORT

    uvicorn.run(app, host=API_HOST, port=API_PORT)
