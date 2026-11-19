"""Clustering experiments: k-Means on sentiment (k=3) and images (k sweep, k=50 analysis), plus PCA plots.

Run from the repo root:   python -m src.clustering
Labels are used only to evaluate the clusters, never to fit them.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

from src.data import load_all, ROOT
from src.Kmeans import KMeans
from src.preprocess import preprocess          # <-- adjust to the real filename
from src.utils import set_seed, timer, log_result, SEED

WHO = "your_name"   # <-- set this
METHOD = {"sentiment": "raw", "image": "raw"}   # "standard", "l2" (and "sqrt" for images) also allowed
N_INIT = 10
IMG_KS = (10, 25, 50, 100)

OUT_DIR = ROOT / "results"
FIG_DIR = OUT_DIR / "figures"
CLUSTER_DIR = OUT_DIR / "clusters"
for d in (FIG_DIR, CLUSTER_DIR):
    d.mkdir(parents=True, exist_ok=True)


def purity(y, clusters):
    ct = pd.crosstab(clusters, y).to_numpy()
    return float(ct.max(axis=1).sum() / ct.sum())


def run_one(name, k, X, y, X_te, method):
    """Fit k-Means, evaluate against the labels, log one row. Returns (model, metrics)."""
    with timer() as t_fit:
        km = KMeans(k=k, n_init=N_INIT, seed=SEED).fit(X)
    with timer() as t_pred:
        km.predict(X_te)
    clusters = km.labels_
    m = dict(k=k, inertia=km.inertia_,
             nmi=float(normalized_mutual_info_score(y, clusters)),
             purity=purity(y, clusters))
    params = dict(k=k, method=method, n_init=N_INIT, n_iter=km.n_iter_,
                  inertia=round(m["inertia"], 2), nmi=round(m["nmi"], 4),
                  purity=round(m["purity"], 4), metric_note="val_acc column = purity on the fitted rows")
    # No val_pred/test_pred here: cluster ids are not class labels, so they must not land in preds/
    # (make_submission and ensembling expect class labels). Cluster ids go to results/clusters/ instead.
    log_result(model="kmeans", dataset=name, who=WHO, params=params,
               train_acc=None, val_acc=m["purity"],
               train_time_s=t_fit["s"], predict_time_s=t_pred["s"])
    np.save(CLUSTER_DIR / f"kmeans_{name}_k{k}.npy", clusters.astype(np.int64))
    print(f"{name} k={k}: inertia {m['inertia']:.1f}  NMI {m['nmi']:.4f}  purity {m['purity']:.4f}  "
          f"({t_fit['s']:.1f}s)")
    return km, m


def pca_plot(X, y, clusters, title, path, max_points=5000):
    """2-D PCA of the features, coloured by true label (left) and by cluster (right)."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(max_points, len(X)), replace=False)
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    Z = pca.transform(X[idx])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, colours, label in zip(axes, (y[idx], clusters[idx]), ("true label", "k-Means cluster")):
        ax.scatter(Z[:, 0], Z[:, 1], c=colours, cmap="tab20", s=5, alpha=0.7)
        ax.set_title(f"{title}: coloured by {label}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path.relative_to(ROOT))


def mixing_report(ct, name):
    """Which classes share clusters (rows of ct = clusters, columns = classes)."""
    rows = []
    for c in ct.index:
        order = ct.loc[c].sort_values(ascending=False)
        size = int(order.sum())
        rows.append(dict(cluster=c, size=size,
                         top_class=order.index[0], top_share=round(order.iloc[0] / size, 3),
                         second_class=order.index[1], second_share=round(order.iloc[1] / size, 3)))
    mix = pd.DataFrame(rows).sort_values("second_share", ascending=False)
    mix.to_csv(OUT_DIR / f"{name}_cluster_mixing.csv", index=False)
    print("\nMost mixed clusters (class pairs that share a cluster):")
    print(mix.head(10).to_string(index=False))

    spread = (ct.max(axis=0) / ct.sum(axis=0)).sort_values()      # share of a class in its best cluster
    print("\nClasses most split across clusters (share of the class inside its main cluster):")
    print(spread.head(10).round(3).to_string())


def sentiment():
    X, y, X_te, _ = load_all("sentiment")
    X, X_te = preprocess(X, X_te, method=METHOD["sentiment"])      # fitted on all training rows
    km, m = run_one("sentiment", 3, X, y, X_te, METHOD["sentiment"])
    ct = pd.crosstab(pd.Series(km.labels_, name="cluster"), pd.Series(y, name="label"))
    ct.to_csv(OUT_DIR / "sentiment_contingency_k3.csv")
    print("\nContingency table (rows = clusters, columns = true labels):")
    print(ct)
    pca_plot(X, y, km.labels_, "Sentiment, k=3", FIG_DIR / "pca_sentiment_k3.png")


def image():
    X, y, X_te, _ = load_all("image")
    X, X_te = preprocess(X, X_te, method=METHOD["image"])
    results, models = [], {}
    for k in IMG_KS:
        km, m = run_one("image", k, X, y, X_te, METHOD["image"])
        results.append(m)
        models[k] = km
    sweep = pd.DataFrame(results)
    sweep.to_csv(OUT_DIR / "image_kmeans_sweep.csv", index=False)
    print("\n", sweep.round(4).to_string(index=False))

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(sweep["k"], sweep["inertia"], "o-", color="tab:blue")
    ax1.set_xlabel("k"); ax1.set_ylabel("inertia", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(sweep["k"], sweep["nmi"], "s--", color="tab:red")
    ax2.set_ylabel("NMI", color="tab:red")
    ax1.set_title("Images: k-Means sweep")
    fig.tight_layout(); fig.savefig(FIG_DIR / "image_kmeans_sweep.png", dpi=150); plt.close(fig)

    km50 = models[50]
    ct = pd.crosstab(pd.Series(km50.labels_, name="cluster"), pd.Series(y, name="label"))
    ct.to_csv(OUT_DIR / "image_contingency_k50.csv")
    mixing_report(ct, "image_k50")
    pca_plot(X, y, km50.labels_, "Images, k=50", FIG_DIR / "pca_image_k50.png")


if __name__ == "__main__":
    set_seed()
    sentiment()
    image()