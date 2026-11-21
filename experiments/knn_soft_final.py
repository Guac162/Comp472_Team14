"""Final image k-NN (Nov 20): distance-weighted vote over 40 neighbours, then class balancing.

Why: plain 1-NN (k-NN sweep winner, 57.2% val) lets a single neighbour decide.
Here the 40 nearest neighbours all vote, but each vote shrinks quickly with
distance:  weight = exp(-(d^2 - d_nearest^2) / SIGMA).  The nearest neighbour
still counts most, but when it is a near-tie the next neighbours can outvote it.
k=40 and SIGMA=0.05 were picked on validation (scratch sweep over k in
{5,10,20,40} x SIGMA in {0.01,0.02,0.05,0.1} with sqrt features).

Balancing: the competition data page states the image test set has exactly
100 rows per class (and training has 400 per class). The plain predictions
range from 64 to 149 per class, so per-class weights are adjusted until the
predictions are split evenly. This uses the published class counts only,
never test labels.

Steps:
  1. Validation on the 80/20 split (logged to results.csv).
  2. Refit on all 20,000 labelled rows, predict the test set.
Saves preds/test/knn_soft_full_image.npy (plain) and
      preds/test/knn_soft_full_bal_image.npy (balanced).

Run from the repo root:  python -m experiments.knn_soft_final
"""
import numpy as np

from src.data import load, load_all
from src.knn import KNN
from src.preprocess import preprocess
from src.utils import accuracy, log_result, save_preds, set_seed, timer

K, SIGMA, N_CLASSES = 40, 0.05, 50


def soft_scores(model, X):
    """Class scores (rows sum to 1) from an exponentially distance-weighted vote."""
    dist, idx = model.kneighbors(X, K)
    d2 = dist ** 2
    w = np.exp(-(d2 - d2[:, :1]) / SIGMA)            # nearest neighbour gets weight 1
    scores = np.zeros((len(X), N_CLASSES))
    rows = np.repeat(np.arange(len(X)), K)
    np.add.at(scores, (rows, model.y_[idx].ravel()), w.ravel())
    return scores / scores.sum(axis=1, keepdims=True)


def balance(S, iters=300):
    """Scale per-class weights until the argmax predictions are split evenly across classes."""
    w = np.ones(S.shape[1])
    for _ in range(iters):
        share = np.bincount((S * w).argmax(1), minlength=S.shape[1]) / len(S)
        w *= (1 / S.shape[1] / np.maximum(share, 1e-6)) ** 0.1
    return (S * w).argmax(1)


if __name__ == "__main__":
    set_seed()
    # ---- 1. validation ----
    X_tr, y_tr, X_va, y_va, _, _ = load("image")
    X_tr, X_va = preprocess(X_tr, X_va, method="sqrt")
    model = KNN(k=K).fit(X_tr, y_tr)
    with timer() as t:
        S_va = soft_scores(model, X_va)
    plain, bal = accuracy(y_va, S_va.argmax(1)), accuracy(y_va, balance(S_va))
    print(f"validation: 1-NN baseline 57.2% | soft k={K} {plain:.4f} | soft + balanced {bal:.4f}")
    log_result(model="knn_soft", dataset="image", who="Mohammed",
               params={"k": K, "sigma": SIGMA, "vote": "exp(-d^2/sigma)", "scaling": "sqrt",
                       "balanced": True},
               train_acc=None, val_acc=bal, train_time_s=0.0, predict_time_s=t["s"],
               val_pred=balance(S_va))

    # ---- 2. final model on all labelled rows ----
    X, y, X_te, _ = load_all("image")
    X, X_te = preprocess(X, X_te, method="sqrt")
    S_te = soft_scores(KNN(k=K).fit(X, y), X_te)
    test_plain, test_bal = S_te.argmax(1), balance(S_te)
    save_preds("knn_soft_full", "image", test_pred=test_plain)
    save_preds("knn_soft_full_bal", "image", test_pred=test_bal)
    old = np.load("preds/test/knn_full_image.npy")
    c = np.bincount(test_plain, minlength=N_CLASSES)
    print(f"test: plain per-class {c.min()}-{c.max()} | balanced per-class "
          f"{np.bincount(test_bal, minlength=N_CLASSES).min()}-{np.bincount(test_bal, minlength=N_CLASSES).max()}")
    print(f"agreement with current 1-NN: plain {np.mean(old == test_plain):.1%}, balanced {np.mean(old == test_bal):.1%}")
