"""
Driver: run every model in the registry across a set of feature sizes.

Each (model, feature_set) is run via 03_train_eval.py as a subprocess so a
failure in one combination does not abort the rest. Results land in
experiments/{model_slug}_{feature_set}/ and a summary table is printed and
saved per feature set to results/table2_{feature_set}.csv.

Usage:
  python src/run_all.py                                  # all models, top50..top200
  python src/run_all.py --features top50 top100          # subset of feature sets
  python src/run_all.py --models "Random Forest" "DNN"   # subset of models
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from model_registry import get_model_names

DEFAULT_FEATURE_SETS = ["top50", "top100", "top150", "top200", "top300"]
TRAIN_SCRIPT = SRC / "03_train_eval.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURE_SETS,
                        choices=["top50", "top100", "top150", "top200", "top300"])
    parser.add_argument("--models", nargs="+", default=get_model_names())
    args = parser.parse_args()

    combos = [(m, fs) for fs in args.features for m in args.models]
    print(f"Running {len(combos)} (model, feature_set) combinations:")
    for m, fs in combos:
        print(f"  - {m}  @  {fs}")

    t_start = time.time()
    failures = []

    for model_name, fs in combos:
        print(f"\n{'#'*70}\n# {model_name}  @  {fs}\n{'#'*70}", flush=True)
        cmd = [sys.executable, str(TRAIN_SCRIPT),
               "--model", model_name, "--features", fs]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"!! FAILED: {model_name} @ {fs} (exit {ret.returncode})", flush=True)
            failures.append((model_name, fs))

    # Per-feature-set summary tables
    for fs in args.features:
        subprocess.run([sys.executable, str(TRAIN_SCRIPT),
                        "--summarize", "--features", fs])

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Done in {elapsed/60:.1f} min. "
          f"{len(combos)-len(failures)}/{len(combos)} succeeded.")
    if failures:
        print("Failures:")
        for m, fs in failures:
            print(f"  - {m} @ {fs}")


if __name__ == "__main__":
    main()
