"""First Decision Tree run: max_depth=5, Gini, raw features.

A deliberately small tree, to check the pipeline end to end and get a first
validation number before the full sweep. Trees ignore feature scale, so no
preprocessing.

Run from the repo root:  python -m experiments.tree_first
"""
from src.data import load
from src.sanity import prediction_balance, shuffled_label_check
from src.tree import DecisionTree
from src.utils import accuracy, log_result, set_seed, timer

set_seed()
PARAMS = dict(max_depth=5, min_samples_split=2, min_samples_leaf=1, criterion="gini", n_bins=32)
N_CLASSES = {"image": 50, "sentiment": 3}

for dataset in ("image", "sentiment"):
    X_tr, y_tr, X_va, y_va, X_te, _ = load(dataset)
    print(f"\n=== {dataset} (chance = {1 / N_CLASSES[dataset]:.1%})")

    tree = DecisionTree(**PARAMS)
    with timer() as t_fit:
        tree.fit(X_tr, y_tr, n_classes=N_CLASSES[dataset])
    with timer() as t_pred:
        val_pred = tree.predict(X_va)
    train_acc = accuracy(y_tr, tree.predict(X_tr))
    val_acc = accuracy(y_va, val_pred)
    print(f"  train {train_acc:.4f}  val {val_acc:.4f}  "
          f"depth {tree.depth_}, {tree.n_leaves} leaves  (fit {t_fit['s']:.1f}s)")

    log_result(model="tree", dataset=dataset, who="Elena", params=PARAMS,
               train_acc=train_acc, val_acc=val_acc,
               train_time_s=t_fit["s"], predict_time_s=t_pred["s"])

    lo, hi, top_share = prediction_balance(val_pred, N_CLASSES[dataset])
    print(f"  [check] val predictions per class: {lo}-{hi}, most common class {top_share:.1%}")
    shuffled_acc, chance = shuffled_label_check(
        lambda a, b, c: DecisionTree(**PARAMS).fit(a, b, N_CLASSES[dataset]).predict(c),
        X_tr, y_tr, X_va, y_va)
    print(f"  [check] shuffled-label val accuracy: {shuffled_acc:.4f} (chance {chance:.4f})")
