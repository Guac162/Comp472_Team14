"""Final k-NN: retrain the chosen configuration on ALL labelled rows (plan item, Nov 18).

The configuration is read from results/knn_best.json (chosen on validation by
experiments/knn_sweep.py). The model is refitted on load_all() - all 20,000
image rows / 12,000 sentiment rows - and predicts the test set.

There is no validation score for this model: the validation rows are now part
of its training data. Evidence that it helps comes from the learning curve
(more training rows -> higher 1-NN accuracy on images).

Saves preds/test/knn_full_<dataset>.npy. Use with make_submission.py.
Run from the repo root:  python -m experiments.knn_final
"""
import json
from pathlib import Path

import numpy as np

from src.data import load_all
from src.knn import KNN
from src.preprocess import preprocess
from src.sanity import prediction_balance
from src.utils import save_preds, set_seed, timer

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = {"image": 50, "sentiment": 3}

if __name__ == "__main__":
    set_seed()
    best = json.loads((ROOT / "results" / "knn_best.json").read_text())
    for dataset in ("image", "sentiment"):
        cfg = best[dataset]
        X, y, X_te, _ = load_all(dataset)
        X, X_te = preprocess(X, X_te, method=cfg["scaling"])   # scaler fitted on all training rows
        model = KNN(k=cfg["k"], metric=cfg["metric"], weights=cfg["weights"])
        with timer() as t:
            test_pred = model.fit(X, y).predict(X_te)
        save_preds("knn_full", dataset, test_pred=test_pred)

        old = np.load(ROOT / "preds" / "test" / f"knn_{dataset}.npy")   # same config, 80% split
        lo, hi, top = prediction_balance(test_pred, N_CLASSES[dataset])
        print(f"{dataset}: {len(y)} training rows, k={cfg['k']} {cfg['metric']} {cfg['scaling']} "
              f"({t['s']:.1f}s) | agrees with 80%-split model on {np.mean(old == test_pred):.1%} of test rows | "
              f"most common class {top:.1%}")
