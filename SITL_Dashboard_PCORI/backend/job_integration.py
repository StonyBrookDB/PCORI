"""
Job Scheduler Integration for SITL Dashboard.

Provides helpers for:
- Submitting training jobs to the scheduler
- Tracking job status and metadata
- Reading job logs
- Getting training curves from job artifacts
"""

import json
import os
import pickle
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Dashboard job tracking (separate from job_system's tracking)
DASHBOARD_DB_PATH = Path.home() / ".cache" / "sitl_dashboard" / "jobs.db"


@dataclass
class DashboardJob:
    """Metadata for a training job submitted from the dashboard."""

    job_id: str
    session_id: str
    dataset_id: str
    model_type: str
    model_name: str
    features: List[str]
    config: Dict[str, Any]
    status: str  # queued, running, completed, failed
    created_at: str
    work_dir: str
    log_path: Optional[str]
    metrics_path: Optional[str]
    error: Optional[str]
    username: str = "unknown"  # User who created the job


@dataclass
class TrainedModel:
    """A trained model stored in the database."""

    model_id: str
    job_id: str
    dataset_id: str
    model_type: str
    model_name: str
    task_type: str
    features: List[str]
    train_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    feature_importance: List[Dict[str, Any]]
    training_history: Dict[str, Any]
    model_path: Optional[str]
    work_dir: str
    created_at: str
    username: str = "unknown"  # User who created the model


def init_dashboard_db():
    """Initialize the dashboard jobs database."""
    DASHBOARD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DASHBOARD_DB_PATH)
    cursor = conn.cursor()

    # Jobs table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            session_id TEXT,
            dataset_id TEXT,
            model_type TEXT,
            model_name TEXT,
            features TEXT,
            config TEXT,
            status TEXT,
            created_at TEXT,
            work_dir TEXT,
            log_path TEXT,
            metrics_path TEXT,
            error TEXT,
            username TEXT DEFAULT 'unknown'
        )
    """
    )

    # Models table - persistent storage for trained models
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            job_id TEXT,
            dataset_id TEXT,
            model_type TEXT,
            model_name TEXT,
            task_type TEXT,
            features TEXT,
            train_metrics TEXT,
            val_metrics TEXT,
            feature_importance TEXT,
            training_history TEXT,
            model_path TEXT,
            work_dir TEXT,
            created_at TEXT,
            username TEXT DEFAULT 'unknown',
            FOREIGN KEY (job_id) REFERENCES jobs(job_id)
        )
    """
    )

    # Add username column to existing tables (migration for existing DBs)
    try:
        cursor.execute("ALTER TABLE jobs ADD COLUMN username TEXT DEFAULT 'unknown'")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE models ADD COLUMN username TEXT DEFAULT 'unknown'")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Create index for faster dataset filtering
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_models_dataset ON models(dataset_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_dataset ON jobs(dataset_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_models_username ON models(username)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_username ON jobs(username)"
    )

    conn.commit()
    conn.close()


def save_job(job: DashboardJob):
    """Save or update a job in the database."""
    init_dashboard_db()

    conn = sqlite3.connect(DASHBOARD_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO jobs
        (job_id, session_id, dataset_id, model_type, model_name, features, config,
         status, created_at, work_dir, log_path, metrics_path, error, username)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            job.job_id,
            job.session_id,
            job.dataset_id,
            job.model_type,
            job.model_name,
            json.dumps(job.features),
            json.dumps(job.config),
            job.status,
            job.created_at,
            job.work_dir,
            job.log_path,
            job.metrics_path,
            job.error,
            job.username,
        ),
    )

    conn.commit()
    conn.close()


def get_job(job_id: str) -> Optional[DashboardJob]:
    """Get a job by ID."""
    init_dashboard_db()

    conn = sqlite3.connect(DASHBOARD_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return DashboardJob(
        job_id=row[0],
        session_id=row[1],
        dataset_id=row[2],
        model_type=row[3],
        model_name=row[4],
        features=json.loads(row[5]),
        config=json.loads(row[6]),
        status=row[7],
        created_at=row[8],
        work_dir=row[9],
        log_path=row[10],
        metrics_path=row[11],
        error=row[12],
    )


def list_jobs(
    dataset_id: Optional[str] = None,
    username: Optional[str] = None,
    limit: int = 50,
) -> List[DashboardJob]:
    """List jobs, optionally filtered by dataset and/or username."""
    init_dashboard_db()

    conn = sqlite3.connect(DASHBOARD_DB_PATH)
    cursor = conn.cursor()

    # Build query with filters
    query = "SELECT * FROM jobs WHERE 1=1"
    params = []

    if dataset_id:
        query += " AND dataset_id = ?"
        params.append(dataset_id)

    if username:
        query += " AND username = ?"
        params.append(username)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    jobs = []
    for row in rows:
        jobs.append(
            DashboardJob(
                job_id=row[0],
                session_id=row[1],
                dataset_id=row[2],
                model_type=row[3],
                model_name=row[4],
                features=json.loads(row[5]),
                config=json.loads(row[6]),
                status=row[7],
                created_at=row[8],
                work_dir=row[9],
                log_path=row[10],
                metrics_path=row[11],
                error=row[12],
                username=row[13] if len(row) > 13 else "unknown",
            )
        )

    return jobs


def update_job_status(job_id: str, status: str, error: Optional[str] = None):
    """Update a job's status."""
    init_dashboard_db()

    conn = sqlite3.connect(DASHBOARD_DB_PATH)
    cursor = conn.cursor()

    if error:
        cursor.execute(
            "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
            (status, error, job_id),
        )
    else:
        cursor.execute(
            "UPDATE jobs SET status = ? WHERE job_id = ?",
            (status, job_id),
        )

    conn.commit()
    conn.close()


def get_job_log(job_id: str, lines: int = 100) -> Dict[str, Any]:
    """Get the last N lines of a job's log file."""
    job = get_job(job_id)
    if not job:
        return {"error": f"Job {job_id} not found", "log": "", "log_path": None}

    log_path = Path(job.work_dir) / f"{job_id}.log"
    if not log_path.exists():
        return {"error": "Log file not found", "log": "", "log_path": str(log_path)}

    try:
        # Efficiently read last N lines
        with open(log_path, "rb") as f:
            # Seek to end
            f.seek(0, 2)
            file_size = f.tell()

            # Read in chunks from end
            chunk_size = 8192
            collected_lines = deque(maxlen=lines)
            remaining = b""

            while file_size > 0 and len(collected_lines) < lines:
                read_size = min(chunk_size, file_size)
                file_size -= read_size
                f.seek(file_size)
                chunk = f.read(read_size) + remaining

                chunk_lines = chunk.split(b"\n")
                remaining = chunk_lines[0]

                for line in reversed(chunk_lines[1:]):
                    collected_lines.appendleft(line.decode("utf-8", errors="replace"))
                    if len(collected_lines) >= lines:
                        break

            if remaining:
                collected_lines.appendleft(remaining.decode("utf-8", errors="replace"))

        return {
            "log": "\n".join(collected_lines),
            "log_path": str(log_path),
            "lines_returned": len(collected_lines),
        }

    except Exception as e:
        return {"error": str(e), "log": "", "log_path": str(log_path)}


def get_training_curves(job_id: str) -> Dict[str, Any]:
    """Get training curves data from job artifacts."""
    job = get_job(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}

    work_dir = Path(job.work_dir)

    # Try dashboard metrics.json first
    metrics_path = work_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                data = json.load(f)
            return {
                "source": "metrics.json",
                "train": data.get("train", []),
                "val": data.get("val", []),
            }
        except Exception as e:
            pass

    # Try Lightning CSV logs
    lightning_logs = work_dir / "lightning_logs"
    if lightning_logs.exists():
        # Find latest version
        versions = sorted(lightning_logs.glob("version_*"))
        if versions:
            metrics_csv = versions[-1] / "metrics.csv"
            if metrics_csv.exists():
                try:
                    import pandas as pd

                    df = pd.read_csv(metrics_csv)

                    train_data = []
                    val_data = []

                    for _, row in df.iterrows():
                        epoch = int(row.get("epoch", 0))

                        # Collect train metrics
                        train_entry = {"epoch": epoch}
                        for col in df.columns:
                            if col.startswith("train_") and pd.notna(row[col]):
                                train_entry[col.replace("train_", "")] = float(row[col])
                        if len(train_entry) > 1:
                            train_data.append(train_entry)

                        # Collect val metrics
                        val_entry = {"epoch": epoch}
                        for col in df.columns:
                            if col.startswith("val_") and pd.notna(row[col]):
                                val_entry[col.replace("val_", "")] = float(row[col])
                        if len(val_entry) > 1:
                            val_data.append(val_entry)

                    return {
                        "source": "lightning_csv",
                        "train": train_data,
                        "val": val_data,
                    }

                except Exception as e:
                    return {"error": f"Failed to parse Lightning CSV: {e}"}

    return {"error": "No training curves found", "train": [], "val": []}


def create_training_job(
    dataset_id: str,
    model_type: str,
    model_name: str,
    features: List[str],
    config: Dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
) -> DashboardJob:
    """
    Create and submit a training job using threading (report-v2 version).

    This version uses direct execution instead of the job scheduler.
    Returns the DashboardJob with job_id and session_id.
    """
    import threading
    import uuid

    from .training_runner import TrainingRunner

    # Generate job ID
    job_id = f"train_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Work directory for this job
    work_dir = (
        Path.home() / ".cache" / "sitl_dashboard" / "training" / f"{job_id}_{timestamp}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    # Determine task type based on dataset
    from .data_adapter import get_dataset_config

    dataset_config = get_dataset_config(dataset_id)
    task_type = dataset_config.task_type

    # Create job record
    job = DashboardJob(
        job_id=job_id,
        session_id="direct",  # Using direct execution
        dataset_id=dataset_id,
        model_type=model_type,
        model_name=model_name,
        features=features,
        config=config,
        status="queued",
        created_at=datetime.now().isoformat(),
        work_dir=str(work_dir),
        log_path=str(work_dir / f"{job_id}.log"),
        metrics_path=str(work_dir / "metrics.json"),
        error=None,
    )

    # Save initial job record
    save_job(job)

    def run_training():
        """Background thread to run training."""
        try:
            runner = TrainingRunner(
                model_type=model_type,
                model_id=job_id,
                work_dir=work_dir,
                task_type=task_type,
            )
            update_job_status(job_id, "running")

            result = runner.run(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                config=config,
                feature_names=features,
            )

            if result.success:
                update_job_status(job_id, "completed")
            else:
                update_job_status(job_id, "failed", result.error)
        except Exception as e:
            import traceback
            error_msg = f"{e}\n{traceback.format_exc()}"
            update_job_status(job_id, "failed", error_msg)

    # Start background thread
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return job


def submit_training_job_async(
    dataset_id: str,
    model_type: str,
    model_name: str,
    features: List[str],
    config: Dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    on_complete=None,  # Callback(job_id, result) when training completes
    username: str = "unknown",
) -> DashboardJob:
    """
    Submit a training job asynchronously (non-blocking).

    The job will be queued and executed by the job scheduler.
    Use get_job() to check status.
    """
    import threading
    import uuid

    from .training_runner import TrainingRunner

    # Generate job ID
    job_id = f"train_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Work directory for this job
    work_dir = (
        Path.home() / ".cache" / "sitl_dashboard" / "training" / f"{job_id}_{timestamp}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    # Determine task type based on dataset
    from .data_adapter import get_dataset_config

    dataset_config = get_dataset_config(dataset_id)
    task_type = dataset_config.task_type

    # Create job record
    job = DashboardJob(
        job_id=job_id,
        session_id="pending",
        dataset_id=dataset_id,
        model_type=model_type,
        model_name=model_name,
        features=features,
        config=config,
        status="queued",
        created_at=datetime.now().isoformat(),
        work_dir=str(work_dir),
        log_path=str(work_dir / f"{job_id}.log"),
        metrics_path=str(work_dir / "metrics.json"),
        error=None,
        username=username,
    )

    # Save initial job record
    save_job(job)

    def submit_and_run():
        """Background thread to run training.

        Note: We use direct execution instead of job scheduler because
        cloudpickle can't serialize closures that capture numpy arrays.
        The background thread still provides async behavior for the dashboard.
        """
        try:
            runner = TrainingRunner(
                model_type=model_type,
                model_id=job_id,
                work_dir=work_dir,
                task_type=task_type,
            )
            update_job_status(job_id, "running")

            # Update session ID to indicate direct execution
            job_record = get_job(job_id)
            if job_record:
                job_record.session_id = "direct"
                save_job(job_record)

            result = runner.run(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                config=config,
                feature_names=features,
            )

            if result.success:
                update_job_status(job_id, "completed")
                # Call completion callback to register model
                if on_complete:
                    try:
                        on_complete(job_id, result)
                    except Exception as cb_err:
                        print(f"Warning: on_complete callback failed: {cb_err}")
            else:
                update_job_status(job_id, "failed", result.error)
        except Exception as e:
            import traceback

            error_msg = f"{e}\n{traceback.format_exc()}"
            update_job_status(job_id, "failed", error_msg)

    # Start background thread
    thread = threading.Thread(target=submit_and_run, daemon=True)
    thread.start()

    return job


# ============== Model Database Functions ==============


def _json_serialize(obj):
    """Convert object to JSON-serializable format, handling numpy types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_serialize(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_model(model: TrainedModel):
    """Save or update a trained model in the database."""
    init_dashboard_db()

    conn = sqlite3.connect(DASHBOARD_DB_PATH)
    cursor = conn.cursor()

    # Serialize with numpy handling
    features_json = json.dumps(_json_serialize(model.features))
    train_metrics_json = json.dumps(_json_serialize(model.train_metrics))
    val_metrics_json = json.dumps(_json_serialize(model.val_metrics))
    feature_importance_json = json.dumps(_json_serialize(model.feature_importance))

    # Debug: print training_history type
    print(f"[save_model] training_history type: {type(model.training_history)}")
    if hasattr(model.training_history, "__dict__"):
        print(f"[save_model] training_history is object with __dict__")
    try:
        training_history_json = json.dumps(_json_serialize(model.training_history))
    except Exception as e:
        print(f"[save_model] Failed to serialize training_history: {e}")
        print(f"[save_model] training_history value: {model.training_history}")
        # Fallback to empty dict
        training_history_json = json.dumps({})

    # Convert Path objects to strings
    model_path_str = str(model.model_path) if model.model_path else None
    work_dir_str = str(model.work_dir) if model.work_dir else ""

    cursor.execute(
        """
        INSERT OR REPLACE INTO models
        (model_id, job_id, dataset_id, model_type, model_name, task_type,
         features, train_metrics, val_metrics, feature_importance,
         training_history, model_path, work_dir, created_at, username)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            model.model_id,
            model.job_id,
            model.dataset_id,
            model.model_type,
            model.model_name,
            model.task_type,
            features_json,
            train_metrics_json,
            val_metrics_json,
            feature_importance_json,
            training_history_json,
            model_path_str,
            work_dir_str,
            model.created_at,
            model.username,
        ),
    )

    conn.commit()
    conn.close()


def get_model(model_id: str) -> Optional[TrainedModel]:
    """Get a model by ID from the database."""
    init_dashboard_db()

    conn = sqlite3.connect(DASHBOARD_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return TrainedModel(
        model_id=row[0],
        job_id=row[1],
        dataset_id=row[2],
        model_type=row[3],
        model_name=row[4],
        task_type=row[5],
        features=json.loads(row[6]),
        train_metrics=json.loads(row[7]),
        val_metrics=json.loads(row[8]),
        feature_importance=json.loads(row[9]),
        training_history=json.loads(row[10]),
        model_path=row[11],
        work_dir=row[12],
        created_at=row[13],
        username=row[14] if len(row) > 14 else "unknown",
    )


def list_models(
    dataset_id: Optional[str] = None,
    model_type: Optional[str] = None,
    search_query: Optional[str] = None,
    username: Optional[str] = None,
    limit: int = 100,
) -> List[TrainedModel]:
    """List models with optional filtering."""
    init_dashboard_db()

    conn = sqlite3.connect(DASHBOARD_DB_PATH)
    cursor = conn.cursor()

    # Build query with filters
    query = "SELECT * FROM models WHERE 1=1"
    params = []

    if dataset_id:
        query += " AND dataset_id = ?"
        params.append(dataset_id)

    if model_type:
        query += " AND model_type = ?"
        params.append(model_type)

    if username:
        query += " AND username = ?"
        params.append(username)

    if search_query:
        query += " AND (model_name LIKE ? OR model_id LIKE ? OR model_type LIKE ?)"
        search_pattern = f"%{search_query}%"
        params.extend([search_pattern, search_pattern, search_pattern])

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    models = []
    for row in rows:
        models.append(
            TrainedModel(
                model_id=row[0],
                job_id=row[1],
                dataset_id=row[2],
                model_type=row[3],
                model_name=row[4],
                task_type=row[5],
                features=json.loads(row[6]),
                train_metrics=json.loads(row[7]),
                val_metrics=json.loads(row[8]),
                feature_importance=json.loads(row[9]),
                training_history=json.loads(row[10]),
                model_path=row[11],
                work_dir=row[12],
                created_at=row[13],
                username=row[14] if len(row) > 14 else "unknown",
            )
        )

    return models


def delete_model(model_id: str) -> bool:
    """Delete a model from the database."""
    init_dashboard_db()

    conn = sqlite3.connect(DASHBOARD_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted


def model_to_dict(model: TrainedModel) -> Dict[str, Any]:
    """Convert a TrainedModel to dictionary format for API responses."""
    return {
        "model_id": model.model_id,
        "job_id": model.job_id,
        "dataset_id": model.dataset_id,
        "model_type": model.model_type,
        "name": model.model_name,
        "task_type": model.task_type,
        "features": model.features,
        "train_metrics": model.train_metrics,
        "val_metrics": model.val_metrics,
        "feature_importance": model.feature_importance,
        "training_history": model.training_history,
        "model_path": model.model_path,
        "work_dir": model.work_dir,
        "created_at": model.created_at,
        "username": model.username,
    }


def migrate_inmemory_models_to_db(trained_models: Dict[str, Dict]):
    """Migrate existing in-memory models to the database (one-time migration)."""
    init_dashboard_db()
    migrated = 0

    for model_id, m in trained_models.items():
        # Check if already in DB
        existing = get_model(model_id)
        if existing:
            continue

        model = TrainedModel(
            model_id=model_id,
            job_id=model_id,  # job_id same as model_id for legacy models
            dataset_id=m.get("dataset_id", "synthetic"),
            model_type=m.get("model_type", "unknown"),
            model_name=m.get("name", model_id),
            task_type=m.get("task_type", "classification"),
            features=m.get("features", []),
            train_metrics=m.get("train_metrics", {}),
            val_metrics=m.get("val_metrics", {}),
            feature_importance=m.get("feature_importance", []),
            training_history=m.get("training_history", {}),
            model_path=m.get("model"),
            work_dir=m.get("work_dir", ""),
            created_at=m.get("created_at", datetime.now().isoformat()),
        )
        save_model(model)
        migrated += 1

    if migrated > 0:
        print(f"Migrated {migrated} models to database")
