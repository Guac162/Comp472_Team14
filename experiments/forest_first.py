"""First Random Forest run: 100 fully grown trees, sqrt(d) features per split.

Logs validation accuracy after 10, 50 and 100 trees (one forest, read at
three sizes), and saves the 100-tree validation and test predictions for a
first Kaggle check. Like knn_first, the test predictions come from the model
fitted on the 80% split, so validation and public score can be compared.

Run from the repo root:  python -m experiments.forest_first
"""
from src.data import load
from src.forest import RandomForest
from src.sanity import prediction_balance
from src.utils import accuracy, log_result, set_seed, timer

set_seed()
PARAMS = dict(max_depth=None, min_samples_leaf=1, criterion="gini", max_features="sqrt", n_bins=32)
N_TREES = [10, 50, 100]
N_CLASSES = {"image": 50, "sentiment": 3}

for dataset in ("image", "sentiment"):
    X_tr, y_tr, X_va, y_va, X_te, _ = load(dataset)
    print(f"\n=== {dataset} (chance = {1 / N_CLASSES[dataset]:.1%})")

    forest = RandomForest(n_trees=max(N_TREES), seed=42, **PARAMS)
    with timer() as t_fit:
        forest.fit(X_tr, y_tr, n_classes=N_CLASSES[dataset], verbose=True)
    with timer() as t_pred:
        val_preds = forest.predict_for_n_trees(X_va, N_TREES)
    train_acc = accuracy(y_tr, forest.predict(X_tr))
    test_pred = forest.predict(X_te)
    print(f"  fit {t_fit['s']:.0f}s, oob {forest.oob_score_:.4f}, train {train_acc:.4f}")

    for n in N_TREES:
        val_acc = accuracy(y_va, val_preds[n])
        print(f"  {n:4d} trees  val {val_acc:.4f}")
        is_full = n == max(N_TREES)
        log_result(model="forest", dataset=dataset, who="Elena", params={**PARAMS, "n_trees": n},
                   train_acc=train_acc if is_full else None, val_acc=val_acc,
                   train_time_s=t_fit["s"] * n / max(N_TREES), predict_time_s=t_pred["s"],
                   val_pred=val_preds[n] if is_full else None,
                   test_pred=test_pred if is_full else None)

    lo, hi, top_share = prediction_balance(val_preds[max(N_TREES)], N_CLASSES[dataset])
    print(f"  [check] val predictions per class: {lo}-{hi}, most common class {top_share:.1%}")
    lo, hi, top_share = prediction_balance(test_pred, N_CLASSES[dataset])
    print(f"  [check] test predictions per class: {lo}-{hi}, most common class {top_share:.1%}")
