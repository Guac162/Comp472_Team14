"""First k-NN run (Nov 9 milestone): k=5, Euclidean, majority vote.

Tries raw vs standardized features, logs both, keeps the better one on
validation, runs the sanity checks on it, and saves its validation and test
predictions for the first real Kaggle submission.

The test predictions come from the model fitted on the 80% train split (the
same model that was validated), so validation accuracy and the public score
can be compared directly. The final submission later retrains on all data.

Run from the repo root:  python -m experiments.knn_first
"""
import numpy as np

from src.data import load
from src.knn import KNN
from src.preprocess import preprocess
from src.sanity import prediction_balance, shuffled_label_check
from src.utils import accuracy, log_result, set_seed, timer

set_seed()
K, METRIC, WEIGHTS = 5, "euclidean", "uniform"
N_CLASSES = {"image": 50, "sentiment": 3}

for dataset in ("image", "sentiment"):
    X_tr_raw, y_tr, X_va_raw, y_va, X_te_raw, _ = load(dataset)
    print(f"\n=== {dataset} (chance = {1 / N_CLASSES[dataset]:.1%})")

    runs = []
    for scaling in ("raw", "standard"):
        X_tr, X_va, X_te = preprocess(X_tr_raw, X_va_raw, X_te_raw, method=scaling)
        model = KNN(k=K, metric=METRIC, weights=WEIGHTS)
        with timer() as t_fit:
            model.fit(X_tr, y_tr)
        with timer() as t_pred:
            val_pred = model.predict(X_va)
        train_acc = accuracy(y_tr, model.predict(X_tr))   # each row is its own nearest neighbour
        val_acc = accuracy(y_va, val_pred)
        print(f"  {scaling:8s} train {train_acc:.4f}  val {val_acc:.4f}  ({t_pred['s']:.1f}s)")
        runs.append(dict(scaling=scaling, model=model, X=(X_tr, X_va, X_te), val_pred=val_pred,
                         train_acc=train_acc, val_acc=val_acc, fit_s=t_fit["s"], pred_s=t_pred["s"]))

    best = max(runs, key=lambda r: r["val_acc"])
    test_pred = best["model"].predict(best["X"][2])
    for r in runs:
        is_best = r is best
        log_result(
            model="knn", dataset=dataset, who="Mohammed",
            params={"k": K, "metric": METRIC, "weights": WEIGHTS, "scaling": r["scaling"]},
            train_acc=r["train_acc"], val_acc=r["val_acc"],
            train_time_s=r["fit_s"], predict_time_s=r["pred_s"],
            val_pred=r["val_pred"] if is_best else None,
            test_pred=test_pred if is_best else None,
        )
    print(f"  kept: {best['scaling']} (val {best['val_acc']:.4f}); predictions saved")

    # --- sanity checks on the kept configuration ---
    X_tr, X_va, X_te = best["X"]
    k1_train = accuracy(y_tr, KNN(k=1, metric=METRIC).fit(X_tr, y_tr).predict(X_tr))
    print(f"  [check] k=1 accuracy on its own training rows: {k1_train:.4f} (expect 1.0)")

    lo, hi, top_share = prediction_balance(best["val_pred"], N_CLASSES[dataset])
    print(f"  [check] val predictions per class: {lo}-{hi}, most common class {top_share:.1%} "
          f"(true share {1 / N_CLASSES[dataset]:.1%})")
    lo, hi, top_share = prediction_balance(test_pred, N_CLASSES[dataset])
    print(f"  [check] test predictions per class: {lo}-{hi}, most common class {top_share:.1%}")

    shuffled_acc, chance = shuffled_label_check(
        lambda a, b, c: KNN(k=K, metric=METRIC).fit(a, b).predict(c), X_tr, y_tr, X_va, y_va)
    print(f"  [check] shuffled-label val accuracy: {shuffled_acc:.4f} (chance {chance:.4f})")
