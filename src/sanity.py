"""Model-agnostic sanity checks: does a result make sense, or is it a bug?

Any teammate can use these with any model.
"""
import numpy as np

from src.utils import accuracy


def prediction_balance(pred, n_classes):
    """Classes are balanced, so predictions should be roughly balanced too.

    Returns (fewest predictions for a class, most predictions for a class,
    share of the most predicted class). A share far above 1/n_classes means
    the model is collapsing onto a few classes.
    """
    counts = np.bincount(pred, minlength=n_classes)
    return int(counts.min()), int(counts.max()), float(counts.max() / counts.sum())


def shuffled_label_check(fit_predict, X_train, y_train, X_val, y_val, seed=42):
    """Train on randomly shuffled labels; validation accuracy must fall to chance.

    fit_predict(X_train, y_train, X_val) -> predictions for X_val.
    If accuracy stays well above chance, information is leaking from validation.
    Returns (accuracy with shuffled labels, chance level).
    """
    rng = np.random.default_rng(seed)
    pred = fit_predict(X_train, rng.permutation(y_train), X_val)
    return accuracy(y_val, pred), 1.0 / len(np.unique(y_train))
