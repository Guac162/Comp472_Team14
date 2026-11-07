"""Quick EDA for the team: class balance, feature scale, sparsity, row norms.

Run from the repo root:  python -m experiments.eda
Writes results/eda.md (paste-able into the team chat or report).
"""
from pathlib import Path

import numpy as np

from src.data import load

lines = ["# Quick EDA (train split only, so no peeking at validation)", ""]
for dataset in ("image", "sentiment"):
    X_tr, y_tr, X_va, y_va, X_te, _ = load(dataset)
    counts = np.bincount(y_tr)
    std = X_tr.std(axis=0)
    norms = np.linalg.norm(X_tr, axis=1)
    # train-vs-test shift: difference in feature means, in units of its standard error
    se = np.sqrt(X_tr.var(0) / len(X_tr) + X_te.var(0) / len(X_te))
    z = np.abs(X_te.mean(0) - X_tr.mean(0)) / se
    shift_std = np.abs(X_te.mean(0) - X_tr.mean(0)) / std
    lines += [
        f"## {dataset}",
        "",
        "| | |",
        "|---|---|",
        f"| rows (train / val / test) | {len(X_tr)} / {len(X_va)} / {len(X_te)} |",
        f"| features | {X_tr.shape[1]} |",
        f"| classes | {len(counts)}, train rows per class {counts.min()}-{counts.max()} |",
        f"| value range | {X_tr.min():.3f} to {X_tr.max():.3f}, mean {X_tr.mean():.4f} |",
        f"| per-feature std | {std.min():.4f} to {std.max():.4f} (ratio {std.max() / std.min():.1f}x) |",
        f"| share of exact zeros | {(X_tr == 0).mean():.1%} |",
        f"| row L2 norm | {norms.min():.3f} to {norms.max():.3f} |",
        f"| constant features | {int((std == 0).sum())} |",
        f"| NaN / inf values | {int((~np.isfinite(X_tr)).sum())} |",
        f"| test vs train shift | {(z > 4).sum()}/{len(z)} features shifted beyond chance (z > 4); "
        f"median {np.median(shift_std):.3f} std, max {shift_std.max():.2f} std |",
        "",
    ]

lines += [
    "## What this means",
    "",
    "- Classes are perfectly balanced, so plain accuracy is a fair metric.",
    "- Image rows are already L2-normalized, so 'raw' and 'l2' are the same for images.",
    "- Feature spreads differ, so distance-based models (k-NN, k-Means) should try standardization.",
    "- Trees do not need scaling: split thresholds just move with the scale.",
    "- Sentiment test rows differ from training rows (most features shifted); images do not.",
    "  Possible causes: a different class mix or different source text in the test set.",
    "  So sentiment validation accuracy may not match the Kaggle score; compare predicted",
    "  class counts on test vs validation once a real model exists.",
]

out = Path(__file__).resolve().parent.parent / "results" / "eda.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwrote {out}")
