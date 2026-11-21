"""Final sentiment NN (Nov 20): fix the test distribution shift, average seeds, train on all rows.

Why: the sentiment test features are shifted (624 of 768 features, see eda.py).
The NN trained on raw features predicts class 0 for ~46% of test rows although
the test set is balanced (the 0.176 baseline proved it), so it loses accuracy
on Kaggle that it does not lose on validation.

What this script does (uses Yousra's NeuralNetwork class and her best settings):
  1. Standardise each set with ITS OWN mean and std (train with train stats,
     validation with validation stats, test with test stats). This lines the
     shifted test features back up with the training features. It uses the
     test FEATURES only, never test labels (test-time normalisation).
  2. Validation run on the 80/20 split, one network per seed, softmax
     probabilities averaged over SEEDS. Logs the validation score and records
     each seed's best epoch (early stopping).
  3. Final run on ALL 12,000 labelled rows for a fixed number of epochs (median
     best epoch from step 2, since there is no validation set left to stop on),
     same seeds, probabilities averaged.
  4. Saves two test prediction files:
        preds/test/nn_full_sentiment.npy      plain argmax
        preds/test/nn_full_bal_sentiment.npy  class weights adjusted so the test
                                              predictions are split 1/3 each
     (the test set is known to be balanced).

Run from the repo root:  python -m experiments.nn_final_sentiment
"""
import contextlib
import io

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.NeuralNetwork import NeuralNetwork, make_loaders, train
from src.data import load, load_all
from src.utils import accuracy, log_result, save_preds, set_seed, timer

SEEDS = (42, 1, 2, 3, 4)
CONFIG = dict(hidden_sizes=[512, 128], dropout=0.2, lr=1e-3, weight_decay=1e-3, batch_size=128)
N_CLASSES = 3


def own_standardize(X):
    """Standardise a set with its own statistics (std 0 -> 1, like Standardizer)."""
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return ((X - X.mean(axis=0)) / std).astype("float32")


def probs(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()


def new_model(n_features, seed):
    set_seed(seed)
    return NeuralNetwork(n_features, CONFIG["hidden_sizes"], N_CLASSES, dropout=CONFIG["dropout"])


def train_fixed_epochs(model, X, y, epochs):
    """Same loss/optimiser as NeuralNetwork.train, but no validation set: fixed epoch count."""
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y.copy())),
                        batch_size=CONFIG["batch_size"], shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


def balance(P, iters=200):
    """Scale per-class weights until the argmax predictions are split evenly across classes."""
    w = np.ones(P.shape[1])
    for _ in range(iters):
        share = np.bincount((P * w).argmax(1), minlength=P.shape[1]) / len(P)
        w *= (1 / P.shape[1] / np.maximum(share, 1e-6)) ** 0.1
    return (P * w).argmax(1), w


def shares(pred):
    return np.round(np.bincount(pred, minlength=N_CLASSES) / len(pred), 3).tolist()


if __name__ == "__main__":
    # ---- 2. validation run (80/20 split) ----
    X_tr, y_tr, X_va, y_va, _, _ = load("sentiment")
    X_tr, X_va = own_standardize(X_tr), own_standardize(X_va)
    tr_loader, va_loader = make_loaders(X_tr, y_tr, X_va, y_va, batch_size=CONFIG["batch_size"])
    P_va, P_tr, best_epochs = [], [], []
    with timer() as t_val:
        for seed in SEEDS:
            model = new_model(X_tr.shape[1], seed)
            with contextlib.redirect_stdout(io.StringIO()):   # silence per-epoch prints
                model, history, best = train(model, tr_loader, va_loader,
                                             lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
            best_epochs.append(history["best_epoch"])
            P_va.append(probs(model, X_va))
            P_tr.append(probs(model, X_tr))
            print(f"seed {seed}: val {best:.4f} (best epoch {history['best_epoch']})")
    val_pred = np.mean(P_va, axis=0).argmax(1)
    val_acc = accuracy(y_va, val_pred)
    train_acc = accuracy(y_tr, np.mean(P_tr, axis=0).argmax(1))
    epochs = int(np.median(best_epochs))
    print(f"{len(SEEDS)}-seed average: val {val_acc:.4f} (train {train_acc:.4f}); final epochs = {epochs}")
    log_result(model="nn_ownstd", dataset="sentiment", who="Mohammed",
               params=dict(CONFIG, scaling="own-set standardise", seeds=list(SEEDS),
                           best_epochs=best_epochs),
               train_acc=train_acc, val_acc=val_acc, train_time_s=t_val["s"], predict_time_s=0.0,
               val_pred=val_pred)

    # ---- 3. final run on all labelled rows ----
    X, y, X_te, _ = load_all("sentiment")
    X, X_te = own_standardize(X), own_standardize(X_te)
    with timer() as t_full:
        P_te = np.mean([probs(train_fixed_epochs(new_model(X.shape[1], s), X, y, epochs), X_te)
                        for s in SEEDS], axis=0)

    # ---- 4. save both variants ----
    test_pred = P_te.argmax(1)
    bal_pred, w = balance(P_te)
    save_preds("nn_full", "sentiment", test_pred=test_pred)
    save_preds("nn_full_bal", "sentiment", test_pred=bal_pred)
    old = np.load("preds/test/nn_sentiment.npy")
    print(f"all {len(y)} rows, {epochs} epochs x {len(SEEDS)} seeds ({t_full['s']:.0f}s)")
    print(f"test class shares  plain {shares(test_pred)} | balanced {shares(bal_pred)} "
          f"(weights {np.round(w, 2).tolist()}) | old nn {shares(old)}")
    print(f"balanced changes {np.mean(bal_pred != test_pred):.1%} of test rows; "
          f"plain agrees with old nn on {np.mean(old == test_pred):.1%}")
