"""k-NN hyperparameter sweep (plan items: sweep Nov 12, plot Nov 13).

Grid, on both datasets:
    k        1, 3, 5, 10, 20, 50
    distance euclidean, cosine, manhattan
    vote     uniform, distance-weighted
    scaling  image: raw, standard, sqrt   (raw == l2 for images, so l2 is skipped)
             sentiment: raw, standard, l2 (sqrt needs non-negative values)

One neighbour search per (scaling, distance) serves every k and vote type.
Every configuration is logged to results/results.csv. The best one per dataset
(highest validation accuracy) gets its validation and test predictions saved
as preds/*/knn_<dataset>.npy, and is written to results/knn_best.json for the
final retrain. Plots go to figures/knn_sweep_<dataset>.png.

Run from the repo root:  python -m experiments.knn_sweep
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from src.data import load
from src.knn import KNN
from src.preprocess import preprocess
from src.utils import accuracy, log_result, save_preds, set_seed, timer

ROOT = Path(__file__).resolve().parent.parent
KS = [1, 3, 5, 10, 20, 50]
METRICS = ("euclidean", "cosine", "manhattan")
WEIGHTS = ("uniform", "distance")
SCALINGS = {"image": ("raw", "standard", "sqrt"), "sentiment": ("raw", "standard", "l2")}
N_CLASSES = {"image": 50, "sentiment": 3}

# chart styling: validated categorical slots 1-3, recessive grid, text in ink tokens
COLORS = {"euclidean": "#2a78d6", "cosine": "#eb6834", "manhattan": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def run_dataset(dataset):
    X_tr0, y_tr, X_va0, y_va, X_te0, _ = load(dataset)
    rows = []
    for scaling in SCALINGS[dataset]:
        X_tr, X_va = preprocess(X_tr0, X_va0, method=scaling)
        for metric in METRICS:
            model = KNN(metric=metric)
            with timer() as t_fit:
                model.fit(X_tr, y_tr)
            with timer() as t_pred:
                grid = model.predict_grid(X_va, KS, WEIGHTS)
            for (w, k), pred in grid.items():
                acc = accuracy(y_va, pred)
                rows.append(dict(scaling=scaling, metric=metric, weights=w, k=k, val_acc=acc))
                log_result(model="knn", dataset=dataset, who="Mohammed",
                           params={"k": k, "metric": metric, "weights": w, "scaling": scaling},
                           train_acc=None, val_acc=acc,
                           train_time_s=t_fit["s"], predict_time_s=t_pred["s"])
            best_here = max((r for r in rows if r["scaling"] == scaling and r["metric"] == metric),
                            key=lambda r: r["val_acc"])
            print(f"  {scaling:8s} {metric:9s} best: k={best_here['k']:2d} {best_here['weights']:8s} "
                  f"val {best_here['val_acc']:.4f}   (search {t_pred['s']:.1f}s)")

    best = max(rows, key=lambda r: r["val_acc"])
    # refit the winner on the 80% split and save its validation and test predictions
    X_tr, X_va, X_te = preprocess(X_tr0, X_va0, X_te0, method=best["scaling"])
    model = KNN(k=best["k"], metric=best["metric"], weights=best["weights"]).fit(X_tr, y_tr)
    val_pred, test_pred = model.predict(X_va), model.predict(X_te)
    assert abs(accuracy(y_va, val_pred) - best["val_acc"]) < 1e-9
    save_preds("knn", dataset, val_pred=val_pred, test_pred=test_pred)
    return rows, best


def plot(dataset, rows, best):
    """Validation accuracy vs k, one line per distance, at the best scaling and vote type."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    curves = {}
    for metric in METRICS:
        pts = sorted((r["k"], r["val_acc"]) for r in rows
                     if r["metric"] == metric and r["scaling"] == best["scaling"]
                     and r["weights"] == best["weights"])
        curves[metric] = tuple(zip(*pts))
    labelled = set()
    for metric, (ks, accs) in curves.items():
        twins = [m for m in METRICS if m != metric and
                 max(abs(a - b) for a, b in zip(curves[m][1], curves[metric][1])) < 0.0025]
        dashed = any(METRICS.index(t) < METRICS.index(metric) for t in twins)
        ax.plot(ks, [a * 100 for a in accs], color=COLORS[metric], linewidth=2,
                linestyle=(0, (4, 3)) if dashed else "-",
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
        if metric in labelled:
            continue
        group = [metric] + twins           # identical curves share one label
        labelled.update(group)
        ax.annotate(" ≈ ".join(group), (ks[-1], accs[-1] * 100), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.scatter([best["k"]], [best["val_acc"] * 100], s=120, facecolors="none",
               edgecolors=INK, linewidths=1.5, zorder=5)
    at_right = best["k"] == max(KS)
    ax.annotate(f"best: {best['val_acc'] * 100:.1f}% (k={best['k']}, {best['metric']})",
                (best["k"], best["val_acc"] * 100),
                xytext=(-14, 10) if at_right else (14, 8), textcoords="offset points",
                ha="right" if at_right else "left", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.8, 140)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.10)
    ax.set_xlabel("k (number of neighbours, log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("validation accuracy (%)", color=INK_2, fontsize=9)
    ax.set_title(f"k-NN on {dataset}: validation accuracy vs k "
                 f"({best['scaling']} features, {best['weights']} vote)",
                 loc="left", fontsize=10.5, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    out = ROOT / "figures" / f"knn_sweep_{dataset}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def write_table(all_rows, all_best):
    """Best (k, vote) for every scaling x distance, as a markdown table for the report."""
    lines = ["# k-NN sweep: best k and vote for each scaling and distance (validation accuracy)", ""]
    for dataset, rows in all_rows.items():
        lines += [f"## {dataset} (chance {1 / N_CLASSES[dataset]:.1%})", "",
                  "| scaling | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for scaling in SCALINGS[dataset]:
            cells = []
            for metric in METRICS:
                b = max((r for r in rows if r["scaling"] == scaling and r["metric"] == metric),
                        key=lambda r: r["val_acc"])
                mark = "**" if b is all_best[dataset] else ""
                cells.append(f"{mark}{b['val_acc']:.1%} (k={b['k']}, {b['weights']}){mark}")
            lines.append(f"| {scaling} | " + " | ".join(cells) + " |")
        lines.append("")
    (ROOT / "results" / "knn_sweep.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    set_seed()
    all_rows, all_best = {}, {}
    for dataset in ("image", "sentiment"):
        print(f"=== {dataset}")
        rows, best = run_dataset(dataset)
        all_rows[dataset], all_best[dataset] = rows, best
        print(f"  WINNER: {best}")
        print(f"  plot: {plot(dataset, rows, best).relative_to(ROOT)}")
    write_table(all_rows, all_best)
    (ROOT / "results" / "knn_best.json").write_text(json.dumps(all_best, indent=2) + "\n")
    print("wrote results/knn_sweep.md and results/knn_best.json")
