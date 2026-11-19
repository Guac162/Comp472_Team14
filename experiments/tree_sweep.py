"""Decision Tree hyperparameter sweep (plan items: sweep and depth plot).

Grid, on both datasets (raw features; trees ignore scale):
    max_depth         5, 10, 20, None   (sentiment also 3, 4, 6, 7, 8: its best depth is below 10)
    min_samples_leaf  1, 5, 20
    criterion         gini, entropy

Every configuration is logged to results/results.csv with train and
validation accuracy. The best one per dataset gets its validation and test
predictions saved as preds/*/tree_<dataset>.npy, and its settings and size
(nodes, MB) go to results/tree_best.json. Plots of train vs validation
accuracy against depth go to figures/tree_depth_<dataset>.png.

Run from the repo root:  python -m experiments.tree_sweep [image] [sentiment]
(no arguments = both datasets; results for a dataset not run are kept)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load
from src.tree import DecisionTree
from src.utils import accuracy, log_result, save_preds, set_seed, timer

ROOT = Path(__file__).resolve().parent.parent
DEPTHS = {"image": [5, 10, 20, None], "sentiment": [3, 4, 5, 6, 7, 8, 10, 20, None]}
LEAVES = [1, 5, 20]
CRITERIA = ("gini", "entropy")
N_CLASSES = {"image": 50, "sentiment": 3}

# same chart styling as knn_sweep
COLORS = {1: "#2a78d6", 5: "#eb6834", 20: "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def tree_size(tree):
    """Number of nodes and memory of the stored tree (node arrays + thresholds) in MB."""
    arrays = (tree.feature_, tree.threshold_, tree.left_, tree.right_, tree.label_, tree.edges_)
    return len(tree.feature_), sum(a.nbytes for a in arrays) / 1e6


def run_dataset(dataset):
    X_tr, y_tr, X_va, y_va, X_te, _ = load(dataset)
    rows = []
    for criterion in CRITERIA:
        for leaf in LEAVES:
            for depth in DEPTHS[dataset]:
                params = dict(max_depth=depth, min_samples_leaf=leaf, criterion=criterion, n_bins=32)
                tree = DecisionTree(**params)
                with timer() as t_fit:
                    tree.fit(X_tr, y_tr, n_classes=N_CLASSES[dataset])
                with timer() as t_pred:
                    val_pred = tree.predict(X_va)
                train_acc, val_acc = accuracy(y_tr, tree.predict(X_tr)), accuracy(y_va, val_pred)
                nodes, mb = tree_size(tree)
                rows.append(dict(params, train_acc=train_acc, val_acc=val_acc, depth_reached=tree.depth_,
                                 leaves=tree.n_leaves, nodes=nodes, mb=mb, fit_s=t_fit["s"]))
                log_result(model="tree", dataset=dataset, who="Elena", params=params,
                           train_acc=train_acc, val_acc=val_acc,
                           train_time_s=t_fit["s"], predict_time_s=t_pred["s"])
                print(f"  {criterion:7s} leaf {leaf:2d} depth {str(depth):4s}  train {train_acc:.4f}  "
                      f"val {val_acc:.4f}  ({tree.depth_} levels, {nodes} nodes, {t_fit['s']:.1f}s)")

    best = max(rows, key=lambda r: r["val_acc"])
    params = {k: best[k] for k in ("max_depth", "min_samples_leaf", "criterion", "n_bins")}
    tree = DecisionTree(**params).fit(X_tr, y_tr, n_classes=N_CLASSES[dataset])
    val_pred, test_pred = tree.predict(X_va), tree.predict(X_te)
    assert abs(accuracy(y_va, val_pred) - best["val_acc"]) < 1e-9
    save_preds("tree", dataset, val_pred=val_pred, test_pred=test_pred)
    return rows, best


def plot(dataset, rows, best):
    """Train (dashed) and validation (solid) accuracy vs max_depth, one colour per min_samples_leaf."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    depths = DEPTHS[dataset]
    xs = list(range(len(depths)))
    ends = {}
    for leaf in LEAVES:
        pts = [next(r for r in rows if r["max_depth"] == d and r["min_samples_leaf"] == leaf
                    and r["criterion"] == best["criterion"]) for d in depths]
        ax.plot(xs, [p["train_acc"] * 100 for p in pts], color=COLORS[leaf], linewidth=1.5,
                linestyle=(0, (4, 3)), marker="o", markersize=4, markeredgecolor=SURFACE)
        ax.plot(xs, [p["val_acc"] * 100 for p in pts], color=COLORS[leaf], linewidth=2,
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ends[leaf] = pts[-1]["val_acc"] * 100

    # line labels on the right, pushed apart so they never overlap
    lo, hi = ax.get_ylim()
    gap, placed = (hi - lo) * 0.045, []
    for leaf, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.annotate(f"leaf {leaf} (val)", (xs[-1], y), xytext=(xs[-1] + 0.12, y_text),
                    textcoords="data", va="center", fontsize=9, color=COLORS[leaf])

    bx = depths.index(best["max_depth"])
    ax.scatter([bx], [best["val_acc"] * 100], s=120, facecolors="none",
               edgecolors=INK, linewidths=1.5, zorder=5)
    ax.annotate(f"best val: {best['val_acc'] * 100:.1f}%\n(leaf {best['min_samples_leaf']}, depth {best['max_depth']})",
                (bx, best["val_acc"] * 100), xytext=(-14 if bx == xs[-1] else 14, 26),
                textcoords="offset points", ha="right" if bx == xs[-1] else "left", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.8))
    ax.plot([], [], color=INK_2, linestyle=(0, (4, 3)), label="train")
    ax.plot([], [], color=INK_2, label="validation")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK_2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) if d is not None else "None\n(no limit)" for d in depths])
    ax.set_xlim(-0.3, len(depths) - 1 + 0.9 * len(depths) / 4)
    ax.set_xlabel("max_depth", color=INK_2, fontsize=9)
    ax.set_ylabel("accuracy (%)", color=INK_2, fontsize=9)
    ax.set_title(f"Decision Tree on {dataset}: train vs validation accuracy by depth ({best['criterion']})",
                 loc="left", fontsize=10.5, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    out = ROOT / "figures" / f"tree_depth_{dataset}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def write_table(dataset, rows):
    """Validation accuracy for every setting, as a markdown table for the report."""
    best = max(rows, key=lambda r: r["val_acc"])
    depths = DEPTHS[dataset]
    lines = [f"# Decision Tree sweep on {dataset}: validation accuracy (train accuracy in brackets)", "",
             f"Chance {1 / N_CLASSES[dataset]:.1%}. Best in bold.", "",
             "| criterion | min_samples_leaf | " + " | ".join(f"depth {d}" for d in depths) + " |",
             "|---|---|" + "---|" * len(depths)]
    for criterion in CRITERIA:
        for leaf in LEAVES:
            cells = []
            for d in depths:
                r = next(r for r in rows if r["criterion"] == criterion
                         and r["min_samples_leaf"] == leaf and r["max_depth"] == d)
                mark = "**" if r is best else ""
                cells.append(f"{mark}{r['val_acc']:.1%}{mark} ({r['train_acc']:.0%})")
            lines.append(f"| {criterion} | {leaf} | " + " | ".join(cells) + " |")
    out = ROOT / "results" / f"tree_sweep_{dataset}.md"
    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    set_seed()
    datasets = sys.argv[1:] or ["image", "sentiment"]
    best_path = ROOT / "results" / "tree_best.json"
    all_best = json.loads(best_path.read_text()) if best_path.exists() else {}
    for dataset in datasets:
        print(f"=== {dataset}")
        rows, best = run_dataset(dataset)
        all_best[dataset] = best
        print(f"  WINNER: {best}")
        print(f"  plot: {plot(dataset, rows, best).relative_to(ROOT)}")
        print(f"  table: {write_table(dataset, rows).relative_to(ROOT)}")
    best_path.write_text(json.dumps(all_best, indent=2) + "\n")
    print("wrote results/tree_best.json")
