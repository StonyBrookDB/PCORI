"""
Uniform Training Interface for SITL Dashboard.

Provides consistent logging and training interface for:
- LightGBM
- Logistic Regression
- PyTorch Lightning (MLP)

All trainers write to a log file with structured messages.
"""

import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def setup_logger(log_path: Path, name: str = "training") -> logging.Logger:
    """Set up a logger that writes to both file and stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter with timestamp
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


@dataclass
class TrainingResult:
    """Result of a training run."""

    model_id: str
    model_type: str
    success: bool
    metrics: Dict[str, float]
    training_history: Dict[str, List[Dict]]
    model_path: Optional[Path]
    error: Optional[str]
    training_time_seconds: float


class TrainingRunner:
    """
    Uniform training interface for all model types.

    Handles:
    - Logging setup
    - Data validation
    - Model-specific training dispatch
    - Metrics collection
    - Result serialization
    """

    def __init__(
        self,
        model_type: str,
        model_id: str,
        work_dir: Path,
        task_type: str = "classification",
    ):
        """
        Initialize training runner.

        Args:
            model_type: One of "lightgbm", "logreg", "pytorch"
            model_id: Unique identifier for this training run
            work_dir: Directory for logs, checkpoints, metrics
            task_type: "classification" or "regression"
        """
        self.model_type = model_type
        self.model_id = model_id
        self.work_dir = Path(work_dir)
        self.task_type = task_type

        # Create work directory
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.log_path = self.work_dir / f"{model_id}.log"
        self.logger = setup_logger(self.log_path, name=f"train_{model_id}")

        # Metrics file for training curves
        self.metrics_path = self.work_dir / "metrics.json"
        self.training_history: Dict[str, List[Dict]] = {
            "train": [],
            "val": [],
        }

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
    ) -> TrainingResult:
        """
        Run training with the configured model type.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            config: Model-specific configuration
            feature_names: Optional feature names for interpretability

        Returns:
            TrainingResult with metrics and model path
        """
        start_time = time.time()

        self.logger.info("=" * 60)
        self.logger.info(f"Starting training: {self.model_type}")
        self.logger.info(f"Model ID: {self.model_id}")
        self.logger.info(f"Task type: {self.task_type}")
        self.logger.info("=" * 60)

        # Log data shapes
        self.logger.info(f"X_train shape: {X_train.shape}")
        self.logger.info(f"y_train shape: {y_train.shape}")
        self.logger.info(f"X_val shape: {X_val.shape}")
        self.logger.info(f"y_val shape: {y_val.shape}")

        # Log data statistics
        self.logger.info(
            f"y_train distribution: {np.unique(y_train, return_counts=True)}"
        )
        self.logger.info(f"y_val distribution: {np.unique(y_val, return_counts=True)}")

        # Check for NaN values
        train_nan = np.isnan(X_train).sum()
        val_nan = np.isnan(X_val).sum()
        if train_nan > 0 or val_nan > 0:
            self.logger.warning(
                f"NaN values detected: train={train_nan}, val={val_nan}"
            )
            self.logger.info("Filling NaN values with 0")
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0)

        # Log configuration
        self.logger.info(f"Config: {json.dumps(config, indent=2, default=str)}")

        try:
            # Dispatch to model-specific trainer
            if self.model_type == "lightgbm":
                result = self._train_lightgbm(
                    X_train, y_train, X_val, y_val, config, feature_names
                )
            elif self.model_type == "logreg":
                result = self._train_logreg(X_train, y_train, X_val, y_val, config)
            elif self.model_type == "pytorch":
                result = self._train_pytorch(X_train, y_train, X_val, y_val, config)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            training_time = time.time() - start_time

            self.logger.info("=" * 60)
            self.logger.info(f"Training complete in {training_time:.2f}s")
            self.logger.info(
                f"Final metrics: {json.dumps(result['metrics'], indent=2)}"
            )
            self.logger.info("=" * 60)

            # Save training history
            self._save_metrics()

            # Save final metrics separately for loading later
            final_metrics_path = self.work_dir / "final_metrics.json"
            with open(final_metrics_path, "w") as f:
                json.dump(result["metrics"], f, indent=2)

            return TrainingResult(
                model_id=self.model_id,
                model_type=self.model_type,
                success=True,
                metrics=result["metrics"],
                training_history=self.training_history,
                model_path=result.get("model_path"),
                error=None,
                training_time_seconds=training_time,
            )

        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"Training failed: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())

            return TrainingResult(
                model_id=self.model_id,
                model_type=self.model_type,
                success=False,
                metrics={},
                training_history=self.training_history,
                model_path=None,
                error=str(e),
                training_time_seconds=training_time,
            )

    def _log_metric(self, split: str, epoch: int, metrics: Dict[str, float]):
        """Log a metric update for training curves."""
        entry = {"epoch": epoch, **metrics}
        self.training_history[split].append(entry)
        self._save_metrics()

    def _save_metrics(self):
        """Save training history to JSON file."""
        with open(self.metrics_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """Train LightGBM model."""
        import lightgbm as lgb
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )

        self.logger.info("Initializing LightGBM training...")

        # Default parameters
        params = {
            "objective": (
                "binary" if self.task_type == "classification" else "regression"
            ),
            "metric": "auc" if self.task_type == "classification" else "rmse",
            "verbosity": -1,
            "num_leaves": config.get("num_leaves", 31),
            "learning_rate": config.get("learning_rate", 0.05),
            "n_estimators": config.get("n_estimators", 100),
            "max_depth": config.get("max_depth", -1),
        }

        self.logger.info(f"LightGBM params: {params}")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Callback for logging
        eval_results = {}

        def log_callback(env):
            iteration = env.iteration
            if iteration % 10 == 0 or iteration == params["n_estimators"] - 1:
                # Get metrics from evaluation results
                train_metric = (
                    env.evaluation_result_list[0][2]
                    if env.evaluation_result_list
                    else 0
                )
                val_metric = (
                    env.evaluation_result_list[1][2]
                    if len(env.evaluation_result_list) > 1
                    else 0
                )

                self.logger.info(
                    f"Iteration {iteration}: train={train_metric:.4f}, val={val_metric:.4f}"
                )

                # Log for training curves
                metric_name = "auc" if self.task_type == "classification" else "rmse"
                self._log_metric("train", iteration, {metric_name: float(train_metric)})
                self._log_metric("val", iteration, {metric_name: float(val_metric)})

        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=params["n_estimators"],
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[
                log_callback,
                lgb.early_stopping(stopping_rounds=10, verbose=False),
            ],
        )

        # Calculate final metrics
        if self.task_type == "classification":
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            metrics = {
                "train_auc": float(roc_auc_score(y_train, train_pred)),
                "val_auc": float(roc_auc_score(y_val, val_pred)),
                "train_auprc": float(average_precision_score(y_train, train_pred)),
                "val_auprc": float(average_precision_score(y_val, val_pred)),
                "train_acc": float(
                    accuracy_score(y_train, (train_pred > 0.5).astype(int))
                ),
                "val_acc": float(accuracy_score(y_val, (val_pred > 0.5).astype(int))),
            }
        else:
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            metrics = {
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
                "train_r2": float(r2_score(y_train, train_pred)),
                "val_r2": float(r2_score(y_val, val_pred)),
            }

        # Save model
        model_path = self.work_dir / f"{self.model_id}.lgb"
        model.save_model(str(model_path))
        self.logger.info(f"Model saved to {model_path}")

        return {"metrics": metrics, "model_path": model_path, "model": model}

    def _train_logreg(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> Dict:
        """Train Logistic Regression model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            roc_auc_score,
        )
        from sklearn.preprocessing import StandardScaler

        self.logger.info("Initializing Logistic Regression training...")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        self.logger.info("Features scaled with StandardScaler")

        # Train model
        C = config.get("C", 1.0)
        max_iter = config.get("max_iter", 1000)

        self.logger.info(f"LogReg params: C={C}, max_iter={max_iter}")

        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, y_train)

        self.logger.info("Model fitted successfully")

        # Calculate metrics
        train_pred = model.predict_proba(X_train_scaled)[:, 1]
        val_pred = model.predict_proba(X_val_scaled)[:, 1]

        metrics = {
            "train_auc": float(roc_auc_score(y_train, train_pred)),
            "val_auc": float(roc_auc_score(y_val, val_pred)),
            "train_auprc": float(average_precision_score(y_train, train_pred)),
            "val_auprc": float(average_precision_score(y_val, val_pred)),
            "train_acc": float(accuracy_score(y_train, (train_pred > 0.5).astype(int))),
            "val_acc": float(accuracy_score(y_val, (val_pred > 0.5).astype(int))),
        }

        # Log single entry for training curves (logreg has no iterations)
        self._log_metric("train", 0, {"auc": metrics["train_auc"]})
        self._log_metric("val", 0, {"auc": metrics["val_auc"]})

        # Save model with scaler
        model_path = self.work_dir / f"{self.model_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
        self.logger.info(f"Model saved to {model_path}")

        return {
            "metrics": metrics,
            "model_path": model_path,
            "model": model,
            "scaler": scaler,
        }

    def _train_pytorch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> Dict:
        """Train PyTorch Lightning MLP model."""
        import pytorch_lightning as pl
        import torch
        import torch.nn as nn
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            roc_auc_score,
        )
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset

        self.logger.info("Initializing PyTorch Lightning training...")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)

        self.logger.info("Features scaled with StandardScaler")

        # Config
        hidden_sizes = config.get("hidden_sizes", [64, 32])
        lr = config.get("learning_rate", 0.001)
        max_epochs = config.get("max_epochs", 50)
        batch_size = config.get("batch_size", 32)
        patience = config.get("patience", 5)

        self.logger.info(
            f"PyTorch params: hidden_sizes={hidden_sizes}, lr={lr}, epochs={max_epochs}"
        )

        # Create data loaders (ensure float32 for GPU compatibility)
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_scaled).float(),
            torch.from_numpy(y_train.astype(np.float32)),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_scaled).float(),
            torch.from_numpy(y_val.astype(np.float32)),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Define MLP model for classification
        class MLPClassifier(pl.LightningModule):
            def __init__(self, input_dim, hidden_sizes, lr):
                super().__init__()
                self.save_hyperparameters()
                self.lr = lr

                layers = []
                prev_dim = input_dim
                for dim in hidden_sizes:
                    layers.extend(
                        [nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.2)]
                    )
                    prev_dim = dim
                layers.append(nn.Linear(prev_dim, 1))
                layers.append(nn.Sigmoid())

                self.model = nn.Sequential(*layers)
                self.loss_fn = nn.BCELoss()

            def forward(self, x):
                return self.model(x).squeeze()

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = self.loss_fn(y_hat, y)
                self.log("train_loss", loss, on_epoch=True, prog_bar=True)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = self.loss_fn(y_hat, y)
                self.log("val_loss", loss, on_epoch=True, prog_bar=True)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=self.lr)

        # Define MLP model for regression
        class MLPRegressor(pl.LightningModule):
            def __init__(self, input_dim, hidden_sizes, lr):
                super().__init__()
                self.save_hyperparameters()
                self.lr = lr

                layers = []
                prev_dim = input_dim
                for dim in hidden_sizes:
                    layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.2)])
                    prev_dim = dim
                layers.append(nn.Linear(prev_dim, 1))
                # No Sigmoid for regression - output raw values

                self.model = nn.Sequential(*layers)
                self.loss_fn = nn.MSELoss()

            def forward(self, x):
                return self.model(x).squeeze()

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = self.loss_fn(y_hat, y)
                self.log("train_loss", loss, on_epoch=True, prog_bar=True)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = self.loss_fn(y_hat, y)
                self.log("val_loss", loss, on_epoch=True, prog_bar=True)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=self.lr)

        # Create model based on task type
        input_dim = X_train_scaled.shape[1]
        if self.task_type == "regression":
            self.logger.info("Using MLPRegressor for regression task")
            model = MLPRegressor(input_dim, hidden_sizes, lr)
        else:
            self.logger.info("Using MLPClassifier for classification task")
            model = MLPClassifier(input_dim, hidden_sizes, lr)

        # Callbacks
        checkpoint_dir = self.work_dir / "checkpoints"
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            verbose=True,
        )

        # Custom callback for logging
        class LoggingCallback(pl.Callback):
            def __init__(self, runner):
                self.runner = runner

            def on_train_epoch_end(self, trainer, pl_module):
                epoch = trainer.current_epoch
                train_loss = trainer.callback_metrics.get("train_loss", 0)
                self.runner.logger.info(
                    f"Epoch {epoch}: train_loss={float(train_loss):.4f}"
                )
                self.runner._log_metric("train", epoch, {"loss": float(train_loss)})

            def on_validation_epoch_end(self, trainer, pl_module):
                epoch = trainer.current_epoch
                val_loss = trainer.callback_metrics.get("val_loss", 0)
                self.runner.logger.info(
                    f"Epoch {epoch}: val_loss={float(val_loss):.4f}"
                )
                self.runner._log_metric("val", epoch, {"loss": float(val_loss)})

        # Logger
        csv_logger = CSVLogger(save_dir=self.work_dir, name="lightning_logs")

        # Trainer - use single device to avoid distributed training conflicts
        # when running inside uvicorn server
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, early_stopping, LoggingCallback(self)],
            logger=csv_logger,
            enable_progress_bar=False,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,  # Single device to avoid DDP conflicts
            strategy="auto",
        )

        self.logger.info("Starting PyTorch Lightning training...")
        trainer.fit(model, train_loader, val_loader)

        self.logger.info(f"Training stopped at epoch {trainer.current_epoch}")

        # Load best checkpoint for evaluation
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            self.logger.info(f"Loading best model from {best_model_path}")
            if self.task_type == "regression":
                model = MLPRegressor.load_from_checkpoint(best_model_path)
            else:
                model = MLPClassifier.load_from_checkpoint(best_model_path)

        # Calculate final metrics (move to CPU for inference)
        model = model.cpu()
        model.eval()
        with torch.no_grad():
            train_pred = model(torch.from_numpy(X_train_scaled).float()).numpy()
            val_pred = model(torch.from_numpy(X_val_scaled).float()).numpy()

        if self.task_type == "regression":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
                "train_mae": float(mean_absolute_error(y_train, train_pred)),
                "val_mae": float(mean_absolute_error(y_val, val_pred)),
                "train_r2": float(r2_score(y_train, train_pred)),
                "val_r2": float(r2_score(y_val, val_pred)),
            }
        else:
            metrics = {
                "train_auc": float(roc_auc_score(y_train, train_pred)),
                "val_auc": float(roc_auc_score(y_val, val_pred)),
                "train_auprc": float(average_precision_score(y_train, train_pred)),
                "val_auprc": float(average_precision_score(y_val, val_pred)),
                "train_acc": float(accuracy_score(y_train, (train_pred > 0.5).astype(int))),
                "val_acc": float(accuracy_score(y_val, (val_pred > 0.5).astype(int))),
            }

        # Save model and scaler
        model_path = self.work_dir / f"{self.model_id}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "config": config,
            },
            model_path,
        )
        self.logger.info(f"Model saved to {model_path}")

        return {
            "metrics": metrics,
            "model_path": model_path,
            "model": model,
            "scaler": scaler,
        }
