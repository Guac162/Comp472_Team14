"""Shared data loading and the fixed train/validation split.

Run once to create the split files:   python src/data.py
Everyone else only calls load() / load_all(); nobody makes their own split.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent   # repo root, wherever it's cloned
DATA_DIR = ROOT / "data"
SPLIT_DIR = ROOT / "splits"
DATASETS = ("image", "sentiment")
SEED = 42
VAL_FRACTION = 0.2


def _read(dataset):
    """Read the train and test CSVs for one dataset and return NumPy arrays."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    train = pd.read_csv(DATA_DIR / f"{dataset}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{dataset}_test.csv")
    feature_cols = [c for c in train.columns if c.startswith("feature_")]
    X = train[feature_cols].to_numpy(dtype=np.float32)
    y = train["label"].to_numpy(dtype=np.int64)
    X_test = test[feature_cols].to_numpy(dtype=np.float32)
    test_ids = test["id"].to_numpy()
    return X, y, X_test, test_ids


def make_split(dataset):
    """Create and save the stratified 80/20 split. Refuses to overwrite."""
    path = SPLIT_DIR / f"{dataset}_split.npz"
    if path.exists():
        raise FileExistsError(f"{path} already exists - delete it only if the whole team agrees")
    _, y, _, _ = _read(dataset)
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=VAL_FRACTION, stratify=y, random_state=SEED
    )
    train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)
    SPLIT_DIR.mkdir(exist_ok=True)
    np.savez(path, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx, y


def load(dataset):
    """For experiments: X_train, y_train, X_val, y_val, X_test, test_ids."""
    X, y, X_test, test_ids = _read(dataset)
    split = np.load(SPLIT_DIR / f"{dataset}_split.npz")
    tr, va = split["train_idx"], split["val_idx"]
    return X[tr], y[tr], X[va], y[va], X_test, test_ids


def load_all(dataset):
    """For the final retrain: all training rows, plus the test set."""
    return _read(dataset)


if __name__ == "__main__":
    for name in DATASETS:
        tr, va, y = make_split(name)
        val_counts = np.bincount(y[va])
        print(f"{name}: train {len(tr)}, val {len(va)}, "
              f"classes {len(val_counts)}, val rows per class {val_counts.min()}-{val_counts.max()}")
        assert len(np.intersect1d(tr, va)) == 0, "train and val overlap"
        assert len(tr) + len(va) == len(y), "rows lost"
