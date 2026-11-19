"""Does PCA help the Random Forest on images?

50-tree forests (sqrt(d) features per split, fully grown) on the image set,
with and without PCA, on raw and on sqrt-normalized features (Mohammed's
Hellinger mapping, which helped k-NN). 50 trees keeps each run to a few
minutes; the best input then gets the full forest sweep.

Run from the repo root:  python -m experiments.forest_pca
"""
from src.data import load
from src.forest import RandomForest
from src.pca import PCA
from src.preprocess import preprocess
from src.utils import accuracy, log_result, set_seed, timer

set_seed()
N_TREES = 50
COMPONENTS = [None, 50, 100, 200]

X_tr0, y_tr, X_va0, y_va, _, _ = load("image")
for scaling in ("raw", "sqrt"):
    X_tr1, X_va1 = preprocess(X_tr0, X_va0, method=scaling)
    for n_comp in COMPONENTS:
        if n_comp is None:
            X_tr, X_va, kept = X_tr1, X_va1, 1.0
        else:
            pca = PCA(n_comp).fit(X_tr1)
            X_tr, X_va, kept = pca.transform(X_tr1), pca.transform(X_va1), pca.explained_ratio_.sum()
        forest = RandomForest(n_trees=N_TREES, seed=42)
        with timer() as t_fit:
            forest.fit(X_tr, y_tr, n_classes=50)
        with timer() as t_pred:
            val_acc = accuracy(y_va, forest.predict(X_va))
        print(f"  {scaling:4s} pca {str(n_comp):4s} ({kept:.0%} variance)  val {val_acc:.4f}  "
              f"({t_fit['s']:.0f}s, depth {forest.trees_[0].depth_})", flush=True)
        log_result(model="forest", dataset="image", who="Elena",
                   params={"n_trees": N_TREES, "max_features": "sqrt", "max_depth": None,
                           "scaling": scaling, "pca": n_comp},
                   train_acc=None, val_acc=val_acc,
                   train_time_s=t_fit["s"], predict_time_s=t_pred["s"])
