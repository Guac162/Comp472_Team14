"""Trivial baseline: always predict the most common training class.

Its only job is to prove the pipeline works end to end
(load -> predict -> log_result -> make_submission -> Kaggle).
With perfectly balanced classes, expect about 1/50 = 2% on images and 1/3 = 33% on sentiment.

Run from the repo root:  python -m experiments.baseline
"""
import numpy as np

from src.data import load
from src.utils import accuracy, log_result, set_seed, timer

set_seed()

for dataset in ("image", "sentiment"):
    X_tr, y_tr, X_va, y_va, X_te, _ = load(dataset)
    with timer() as t_fit:
        majority = int(np.bincount(y_tr).argmax())    # all classes tie -> lowest label wins
    with timer() as t_pred:
        val_pred = np.full(len(y_va), majority)
        test_pred = np.full(len(X_te), majority)
    row = log_result(
        model="baseline", dataset=dataset, who="Mohammed",
        params={"rule": "majority_class", "class": majority},
        train_acc=accuracy(y_tr, np.full(len(y_tr), majority)),
        val_acc=accuracy(y_va, val_pred),
        train_time_s=t_fit["s"], predict_time_s=t_pred["s"],
        val_pred=val_pred, test_pred=test_pred,
    )
    print(f"{dataset}: predicts class {majority}, val acc {row['val_acc']}")
