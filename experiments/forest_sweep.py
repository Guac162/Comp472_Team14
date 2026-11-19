"""Random Forest hyperparameter sweep (plan item: sweep n_trees, max_features, depth).

Grid:
    n_trees           50, 100, 200, 300   (one 300-tree forest, read at each size)
    max_features      log2, sqrt, 2*sqrt, 4*sqrt   (images: no 4*sqrt, 20 s per tree)
    min_samples_leaf  1, 5                         (images: 1 only, to keep the run under an hour)
Trees are fully grown (max_depth None): the vote handles overfitting, and the
tree sweep showed deeper is better for single trees too.

Inputs: sentiment uses raw features; images use the input that won
experiments/forest_pca.py (see INPUTS).

Every configuration is logged to results/results.csv. The best one per dataset
gets its validation and test predictions saved as preds/*/forest_<dataset>.npy,
and its settings and size go to results/forest_best_<dataset>.json. Plots of
validation accuracy vs number of trees go to figures/forest_sweep_<dataset>.png.

Run from the repo root:  python -m experiments.forest_sweep [sentiment] [image]
(no arguments = both datasets, one after the other)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load
from src.forest import RandomForest
from src.pca import PCA
from src.preprocess import preprocess
from src.utils import accuracy, log_result, save_preds, set_seed, timer

ROOT = Path(__file__).resolve().parent.parent
N_TREES = [50, 100, 200, 300]
LEAVES = {"image": [1], "sentiment": [1, 5]}
FEATURES = {"image": ["log2", "sqrt", "2*sqrt"], "sentiment": ["log2", "sqrt", "2*sqrt", "4*sqrt"]}
N_CLASSES = {"image": 50, "sentiment": 3}
# forest_pca.py: PCA hurt (50 trees: raw 23.2%, PCA-50 20.1%), sqrt helped a little (24.4%)
INPUTS = {"image": dict(scaling="sqrt", pca=None), "sentiment": dict(scaling="raw", pca=None)}

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def make_inputs(dataset):
    X_tr, y_tr, X_va, y_va, X_te, _ = load(dataset)
    cfg = INPUTS[dataset]
    X_tr, X_va, X_te = preprocess(X_tr, X_va, X_te, method=cfg["scaling"])
    if cfg["pca"]:
        pca = PCA(cfg["pca"]).fit(X_tr)                   # fitted on train only
        X_tr, X_va, X_te = pca.transform(X_tr), pca.transform(X_va), pca.transform(X_te)
    return X_tr, y_tr, X_va, y_va, X_te


def feature_options(d):
    s = round(np.sqrt(d))
    return {"log2": "log2", "sqrt": "sqrt", "2*sqrt": 2 * s, "4*sqrt": min(4 * s, d)}


def forest_size(forest):
    nodes = sum(len(t.feature_) for t in forest.trees_)
    mb = sum(a.nbytes for t in forest.trees_
             for a in (t.feature_, t.threshold_, t.left_, t.right_, t.label_, t.edges_)) / 1e6
    return nodes, mb


def run_dataset(dataset):
    X_tr, y_tr, X_va, y_va, X_te = make_inputs(dataset)
    rows, kept = [], None
    options = feature_options(X_tr.shape[1])
    for leaf in LEAVES[dataset]:
        for name in FEATURES[dataset]:
            mf = options[name]
            forest = RandomForest(n_trees=max(N_TREES), max_features=mf, min_samples_leaf=leaf,
                                  seed=42)
            with timer() as t_fit:
                forest.fit(X_tr, y_tr, n_classes=N_CLASSES[dataset])
            with timer() as t_pred:
                preds = forest.predict_for_n_trees(X_va, N_TREES)
            for n in N_TREES:
                acc = accuracy(y_va, preds[n])
                rows.append(dict(max_features=name, min_samples_leaf=leaf, n_trees=n, val_acc=acc,
                                 oob=forest.oob_score_, fit_s=t_fit["s"]))
                log_result(model="forest", dataset=dataset, who="Elena",
                           params={"n_trees": n, "max_features": name, "min_samples_leaf": leaf,
                                   "max_depth": None, **INPUTS[dataset]},
                           train_acc=None, val_acc=acc,
                           train_time_s=t_fit["s"] * n / max(N_TREES), predict_time_s=t_pred["s"])
            best_here = max(rows[-len(N_TREES):], key=lambda r: r["val_acc"])
            print(f"  leaf {leaf} {name:6s} ({forest.max_features_} features)  "
                  + "  ".join(f"{n}:{accuracy(y_va, preds[n]):.4f}" for n in N_TREES)
                  + f"  oob {forest.oob_score_:.4f}  ({t_fit['s']:.0f}s)", flush=True)
            if kept is None or best_here["val_acc"] > kept[0]["val_acc"]:
                kept = (best_here, forest)

    # the best forest, read at its best size: save predictions, size and train accuracy
    best, forest = kept
    forest.trees_ = forest.trees_[:best["n_trees"]]
    val_pred, test_pred = forest.predict(X_va), forest.predict(X_te)
    assert abs(accuracy(y_va, val_pred) - best["val_acc"]) < 1e-9
    save_preds("forest", dataset, val_pred=val_pred, test_pred=test_pred)
    best["train_acc"] = accuracy(y_tr, forest.predict(X_tr))
    best["nodes"], best["mb"] = forest_size(forest)
    best.update(INPUTS[dataset])
    return rows, best


def plot(dataset, rows, best):
    """Validation accuracy vs number of trees, one line per max_features, at the best leaf size."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ends = {}
    for color, name in zip(COLORS, FEATURES[dataset]):
        pts = [r for r in rows if r["max_features"] == name and r["min_samples_leaf"] == best["min_samples_leaf"]]
        ax.plot([r["n_trees"] for r in pts], [r["val_acc"] * 100 for r in pts], color=color, linewidth=2,
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ends[name] = (color, pts[-1]["val_acc"] * 100)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.08)

    # line labels on the right, pushed apart so they never overlap
    gap, placed = (hi - lo) * 0.06, []
    for name, (color, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.annotate(name, (N_TREES[-1], y), xytext=(N_TREES[-1] + 12, y_text), textcoords="data",
                    va="center", fontsize=9, color=color)

    ax.scatter([best["n_trees"]], [best["val_acc"] * 100], s=120, facecolors="none",
               edgecolors=INK, linewidths=1.5, zorder=5)
    ax.annotate(f"best: {best['val_acc'] * 100:.1f}%\n({best['max_features']}, {best['n_trees']} trees)",
                (best["n_trees"], best["val_acc"] * 100), xytext=(0.80, 0.10), textcoords="axes fraction",
                ha="right", fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.8))
    ax.set_xticks(N_TREES)
    ax.set_xlim(30, 370)
    ax.set_xlabel("number of trees", color=INK_2, fontsize=9)
    ax.set_ylabel("validation accuracy (%)", color=INK_2, fontsize=9)
    ax.set_title(f"Random Forest on {dataset}: validation accuracy vs trees "
                 f"(min_samples_leaf {best['min_samples_leaf']})", loc="left", fontsize=10.5, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    out = ROOT / "figures" / f"forest_sweep_{dataset}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def write_table(dataset, rows, best):
    lines = [f"# Random Forest sweep on {dataset}: validation accuracy by number of trees", "",
             f"Chance {1 / N_CLASSES[dataset]:.1%}. Input: {INPUTS[dataset]}. Best in bold.", "",
             "| max_features | min_samples_leaf | " + " | ".join(f"{n} trees" for n in N_TREES) + " | OOB |",
             "|---|---|" + "---|" * (len(N_TREES) + 1)]
    for leaf in LEAVES[dataset]:
        for name in FEATURES[dataset]:
            pts = [r for r in rows if r["max_features"] == name and r["min_samples_leaf"] == leaf]
            cells = [("**{:.1%}**" if r is best else "{:.1%}").format(r["val_acc"]) for r in pts]
            lines.append(f"| {name} | {leaf} | " + " | ".join(cells) + f" | {pts[-1]['oob']:.1%} |")
    out = ROOT / "results" / f"forest_sweep_{dataset}.md"
    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    set_seed()
    for dataset in sys.argv[1:] or ["sentiment", "image"]:
        print(f"=== {dataset}  input {INPUTS[dataset]}")
        rows, best = run_dataset(dataset)
        print(f"  WINNER: {best}")
        print(f"  plot: {plot(dataset, rows, best).relative_to(ROOT)}")
        print(f"  table: {write_table(dataset, rows, best).relative_to(ROOT)}")
        out = ROOT / "results" / f"forest_best_{dataset}.json"
        out.write_text(json.dumps(best, indent=2) + "\n")
        print(f"  best: {out.relative_to(ROOT)}")
