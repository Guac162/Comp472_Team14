"""Feature scaling shared by all models.

Rule: fit on the training rows only, then apply the same transform to
validation and test (no leakage). For the final model, the training rows
are all of load_all().
"""
import numpy as np

METHODS = ("raw", "standard", "l2")


class Standardizer:
    """Z-score per feature: (x - mean) / std, learned from the training rows."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0, dtype=np.float64)
        std = X.std(axis=0, dtype=np.float64)
        std[std == 0] = 1.0                  # constant feature: leave it centred, avoid /0
        self.std_ = std
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def l2_normalize(X, eps=1e-12):
    """Scale each row to length 1. Needs no fitting: each row uses only itself."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(norms, eps)).astype(np.float32)


def preprocess(X_train, *others, method="standard"):
    """Apply one method to X_train and any other sets (val, test).

    Usage: X_tr, X_va, X_te = preprocess(X_tr, X_va, X_te, method="standard")
    """
    if method == "raw":
        return (X_train, *others)
    if method == "l2":
        return tuple(l2_normalize(X) for X in (X_train, *others))
    if method == "standard":
        scaler = Standardizer().fit(X_train)            # fitted on train only
        return tuple(scaler.transform(X) for X in (X_train, *others))
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")
