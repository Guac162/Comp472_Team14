"""Final sentiment forest: the best sweep settings, retrained on all training rows.

Settings come from results/forest_best_sentiment.json (300 trees, 2*sqrt features,
min_samples_leaf 1, raw features). There is no validation set left, so the
out-of-bag score is the only accuracy estimate. Test predictions go to
preds/test/forest_full_sentiment.npy; the 80% model's predictions in
preds/*/forest_sentiment.npy stay as they are for the error analysis.

Also plots the 20 most used features (split counts) to figures/forest_importance_sentiment.png.

Run from the repo root:  python -m experiments.forest_final
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_all
from src.forest import RandomForest
from src.utils import save_preds, set_seed, timer

ROOT = Path(__file__).resolve().parent.parent
INK, INK_2, GRID, SURFACE, BAR = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"


def plot_importance(shares, top=20):
    order = np.argsort(shares)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ys = np.arange(top)[::-1]
    ax.barh(ys, shares[order] * 100, color=BAR, height=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"feature_{i}" for i in order], fontsize=8)
    ax.axvline(100 / len(shares), color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("equal share", (100 / len(shares), top - 0.5), xytext=(4, 0), textcoords="offset points",
                fontsize=8.5, color=INK_2, va="center")
    ax.set_xlabel("share of all splits (%)", color=INK_2, fontsize=9)
    ax.set_title(f"Random Forest on sentiment: the {top} most used features", loc="left",
                 fontsize=10.5, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    out = ROOT / "figures" / "forest_importance_sentiment.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    set_seed()
    best = json.loads((ROOT / "results" / "forest_best_sentiment.json").read_text())
    names = {"log2": "log2", "sqrt": "sqrt", "2*sqrt": 56, "4*sqrt": 112}     # 768 features
    X, y, X_te, _ = load_all("sentiment")
    forest = RandomForest(n_trees=best["n_trees"], max_features=names[best["max_features"]],
                          min_samples_leaf=best["min_samples_leaf"], seed=42)
    with timer() as t_fit:
        forest.fit(X, y, n_classes=3, verbose=True)
    test_pred = forest.predict(X_te)
    save_preds("forest_full", "sentiment", test_pred=test_pred)

    old = np.load(ROOT / "preds" / "test" / "forest_sentiment.npy")
    print(f"  trained on {len(y)} rows in {t_fit['s']:.0f}s, oob {forest.oob_score_:.4f}")
    print(f"  test predictions per class: {np.bincount(test_pred, minlength=3).tolist()}")
    print(f"  agrees with the 80% model on {(old == test_pred).mean():.1%} of test rows")

    shares = forest.feature_importance()
    print(f"  top feature share {shares.max():.2%} vs equal share {1 / len(shares):.2%}")
    print(f"  plot: {plot_importance(shares).relative_to(ROOT)}")
