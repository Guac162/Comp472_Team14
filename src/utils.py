"""Shared helpers: seeding, accuracy, timing and the one results log.

Every experiment ends with log_result(...), which appends one row to
results/results.csv and (optionally) saves predictions to preds/val/ and
preds/test/ as <model>_<dataset>.npy for error analysis and ensembling.
"""
import csv
import json
import random
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "results.csv"
PREDS_DIR = ROOT / "preds"
COLUMNS = ["date", "who", "model", "dataset", "params",
           "train_acc", "val_acc", "train_time_s", "predict_time_s"]
SEED = 42


def set_seed(seed=SEED):
    """Seed random, numpy and (if installed) torch. Call at the top of every script."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def accuracy(y_true, y_pred):
    """Share of predictions that match the labels."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, f"shape mismatch {y_true.shape} vs {y_pred.shape}"
    return float((y_true == y_pred).mean())


@contextmanager
def timer():
    """Usage:  with timer() as t: model.fit(X, y)   then   t["s"] is the seconds taken."""
    t = {}
    start = time.perf_counter()
    yield t
    t["s"] = time.perf_counter() - start


def log_result(model, dataset, who, params, train_acc, val_acc,
               train_time_s, predict_time_s, val_pred=None, test_pred=None):
    """Append one row to results/results.csv; optionally save val/test predictions."""
    if dataset not in ("image", "sentiment"):
        raise ValueError(f"dataset must be 'image' or 'sentiment', got {dataset!r}")
    row = {
        "date": date.today().isoformat(),
        "who": who,
        "model": model,
        "dataset": dataset,
        "params": json.dumps(params, sort_keys=True),
        "train_acc": "" if train_acc is None else f"{train_acc:.4f}",
        "val_acc": f"{val_acc:.4f}",
        "train_time_s": f"{train_time_s:.2f}",
        "predict_time_s": f"{predict_time_s:.2f}",
    }
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    is_new = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    for split, pred in (("val", val_pred), ("test", test_pred)):
        if pred is not None:
            out = PREDS_DIR / split / f"{model}_{dataset}.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, np.asarray(pred, dtype=np.int64))
    return row
