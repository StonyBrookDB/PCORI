#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix
)
from datasets.loader import load_dataset

import models.logreg        # noqa: F401
import models.mlp           # noqa: F401
import models.lstm          # noqa: F401
import models.bilstm        # noqa: F401
import models.lstm_attn     # noqa: F401
import models.random_forest # noqa: F401
import models.decision_tree # noqa: F401
import models.gcn           # noqa: F401
import models.lstm_gcn      # noqa: F401
import models.hetero_rgcn   # noqa: F401
import models.lighted       # noqa: F401

def _read_config(path: Path) -> dict:
    txt = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(txt)
    except Exception:
        return json.loads(txt)

def _load_model(run_dir: Path):
    pt = run_dir / "checkpoint.pt"
    jb = run_dir / "checkpoint.joblib"
    if pt.exists():
        from models.lstm import LSTMSeqClassifier
        model = LSTMSeqClassifier.load_checkpoint(str(pt), map_location="cpu")
        return model, "torch"
    if jb.exists():
        from joblib import load
        return load(jb), "sk"
    raise FileNotFoundError("No checkpoint found (checkpoint.pt|checkpoint.joblib).")

def _pick_split(data: dict, split: str):
    if split == "train": return data["X_train"], data["y_train"], data["ids_train"]
    if split == "val":   return data["X_val"],   data["y_val"],   data["ids_val"]
    if split == "test":  return data["X_test"],  data["y_test"],  data["ids_test"]
    raise ValueError(f"Unknown split: {split}")

def _evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = y_true.astype(np.float32); y_prob = y_prob.astype(np.float32)
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    y_pred_05 = (y_prob >= 0.5).astype(int)
    f1_05 = float(f1_score(y_true, y_pred_05))
    cm_05 = confusion_matrix(y_true, y_pred_05, labels=[0,1]).tolist()
    ps, rs, th = precision_recall_curve(y_true, y_prob)
    f1s = [0.0 if (ps[i]+rs[i])==0 else 2*ps[i]*rs[i]/(ps[i]+rs[i]) for i in range(len(th))]
    best_idx = int(np.argmax(f1s)) if len(f1s) else 0
    best_thresh = float(th[best_idx]) if len(th) else 0.5
    y_pred_best = (y_prob >= best_thresh).astype(int)
    f1_best = float(f1_score(y_true, y_pred_best))
    cm_best = confusion_matrix(y_true, y_pred_best, labels=[0,1]).tolist()
    return {
        "auroc": auroc, "auprc": auprc,
        "f1_at_0_5": f1_05, "confusion_at_0_5": {"labels":["TN","FP","FN","TP"], "matrix": cm_05},
        "f1_best": f1_best, "best_threshold": best_thresh,
        "confusion_at_best": {"labels":["TN","FP","FN","TP"], "matrix": cm_best},
        "n_samples": int(len(y_true)), "pos_rate": float(y_true.mean()),
    }

def _try_plot_curves(run_dir: Path, split: str, y_true: np.ndarray, y_prob: np.ndarray):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not installed; skip plotting."); return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(); plt.plot(fpr, tpr, label="ROC"); plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC ({split})"); plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / f"roc_{split}.png", dpi=160); plt.close()
    ps, rs, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(); plt.plot(rs, ps, label="PR"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR ({split})"); plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / f"pr_{split}.png", dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--device", type=str, default=None, help="torch 模型可选 'cpu'/'cuda'")
    args = ap.parse_args()

    run_dir = Path(args.run); cfg = _read_config(run_dir / "config_resolved.yaml")
    dataset_dir = Path(cfg["dataset"])
    spec_snapshot = run_dir / "FeatureSpec.json"
    if not spec_snapshot.exists():
        spec_snapshot = Path(cfg["spec"])

    data = load_dataset(str(dataset_dir), str(spec_snapshot))
    X, y, ids = _pick_split(data, args.split)

    model, kind = _load_model(run_dir)
    if kind == "torch" and args.device is not None:
        import torch
        dev = torch.device(args.device)
        model.to(dev)
        if hasattr(model, "device"):
            model.device = dev  # 与内部自定义 device 保持一致

    y_prob = model.predict_proba(X)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)

    metrics = _evaluate(y, y_prob)
    out_json = run_dir / f"eval_metrics_{args.split}.json"
    with open(out_json, "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2)
    print(f"[OK] Wrote metrics -> {out_json}")

    import pandas as pd
    pd.DataFrame({"patient_id": ids, "y_true": y.astype(float), "y_prob": y_prob.astype(float)}).to_csv(
        run_dir / f"preds_{args.split}.csv", index=False
    )
    print(f"[OK] Wrote predictions -> {run_dir / f'preds_{args.split}.csv'}")

    _try_plot_curves(run_dir, args.split, y, y_prob)
    print(f"[OK] Plots saved (if matplotlib installed).")

if __name__ == "__main__":
    main()
