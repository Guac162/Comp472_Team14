"""Error analysis and efficiency (plan items, Nov 14-15).

Works on whatever models have saved validation predictions in
preds/val/<model>_<dataset>.npy (the baseline is skipped). Re-run it after
teammates save their predictions and their models appear automatically.

Produces results/error_analysis.md and figures/confusion_<model>_<dataset>.png:
    - accuracy per model, and predicted class balance on validation and test
    - confusion matrix per model and dataset
    - image: the 10 most confused class pairs, best and worst classes
    - sentiment: precision, recall and F1 per class
    - mistake overlap between models (needs 2+ models)
    - efficiency table from results/results.csv

scikit-learn is used only for metrics, which the course rules allow.
Run from the repo root:  python -m experiments.error_analysis
"""
import csv
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.data import load

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = {"image": 50, "sentiment": 3}
SKIP = {"baseline"}
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def available_models(dataset):
    found = sorted(p.stem[: -len(dataset) - 1] for p in (ROOT / "preds" / "val").glob(f"*_{dataset}.npy"))
    return [m for m in found if m not in SKIP]


def plot_confusion(cm, model, dataset):
    """Row-normalised (each row = one true class, shows where its rows went). One-hue sequential ramp."""
    rates = cm / cm.sum(axis=1, keepdims=True)
    n = len(cm)
    size = 7.5 if n > 10 else 4.2
    fig, ax = plt.subplots(figsize=(size, size * 0.92), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("predicted class", color=INK_2, fontsize=9)
    ax.set_ylabel("true class", color=INK_2, fontsize=9)
    ax.set_title(f"{model} on {dataset}: confusion matrix\n(share of each true class's rows)",
                 loc="left", fontsize=10, color=INK)
    ticks = range(0, n, 5) if n > 10 else range(n)
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.tick_params(colors=INK_2, labelsize=7.5 if n > 10 else 9, length=0)
    if n <= 10:  # small matrix: print counts and shares in each cell
        for i, j in itertools.product(range(n), range(n)):
            ax.text(j, i, f"{cm[i, j]}\n{rates[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if rates[i, j] > 0.5 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(colors=INK_2, labelsize=8, length=0)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    out = ROOT / "figures" / f"confusion_{model}_{dataset}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out.relative_to(ROOT)


def efficiency_rows():
    """Best validation row per (model, dataset) from the shared log, with its timings."""
    best = {}
    with open(ROOT / "results" / "results.csv") as f:
        for r in csv.DictReader(f):
            if r["model"] in SKIP or not r["val_acc"]:
                continue
            key = (r["model"], r["dataset"])
            if key not in best or float(r["val_acc"]) > float(best[key]["val_acc"]):
                best[key] = r
    return best


def main():
    md = ["# Error analysis", "",
          "Validation set only (test labels are unknown). Re-run after new predictions are saved.", ""]
    preds = {}
    for dataset in ("image", "sentiment"):
        _, _, _, y_va, _, _ = load(dataset)
        C = N_CLASSES[dataset]
        models = available_models(dataset)
        md += [f"## {dataset} ({len(y_va)} validation rows, chance {1 / C:.1%})", ""]
        if not models:
            md += ["No saved predictions yet.", ""]
            continue
        md += ["| model | val accuracy | val: most predicted class share | test: most predicted class share |",
               "|---|---|---|---|"]
        for m in models:
            v = np.load(ROOT / "preds" / "val" / f"{m}_{dataset}.npy")
            preds[(m, dataset)] = v
            t_path = ROOT / "preds" / "test" / f"{m}_{dataset}.npy"
            t_share = (f"{np.bincount(np.load(t_path), minlength=C).max() / 5000:.1%}"
                       if t_path.exists() else "not saved")
            md.append(f"| {m} | {(v == y_va).mean():.2%} | "
                      f"{np.bincount(v, minlength=C).max() / len(v):.1%} | {t_share} |")
        md += ["", f"Even split would be {1 / C:.1%} per class.", ""]

        for m in models:
            v = preds[(m, dataset)]
            cm = confusion_matrix(y_va, v, labels=range(C))
            md += [f"### {m}", "", f"Confusion matrix: `{plot_confusion(cm, m, dataset)}`", ""]
            if dataset == "image":
                pairs = []
                for i, j in itertools.combinations(range(C), 2):
                    if cm[i, j] + cm[j, i]:
                        pairs.append((cm[i, j] + cm[j, i], i, j, cm[i, j], cm[j, i]))
                pairs.sort(reverse=True)
                md += ["10 most confused class pairs (both directions added):", "",
                       "| classes | mistakes | a→b | b→a |", "|---|---|---|---|"]
                md += [f"| {i} ↔ {j} | {tot} | {ab} | {ba} |" for tot, i, j, ab, ba in pairs[:10]]
                recall = np.diag(cm) / cm.sum(axis=1)
                order = np.argsort(recall)
                md += ["", "Worst classes (share of their validation rows correct): "
                       + ", ".join(f"{c} ({recall[c]:.0%})" for c in order[:5]),
                       "Best classes: " + ", ".join(f"{c} ({recall[c]:.0%})" for c in order[::-1][:5]), ""]
            else:
                p, r, f1, _ = precision_recall_fscore_support(y_va, v, labels=range(C), zero_division=0)
                md += ["| class | precision | recall | F1 |", "|---|---|---|---|"]
                md += [f"| {c} | {p[c]:.1%} | {r[c]:.1%} | {f1[c]:.1%} |" for c in range(C)]
                md += ["", f"Hardest class by F1: {int(np.argmin(f1))}. Check the label mapping on Kaggle "
                       "to see whether it is Neutral.", ""]

        if len(models) >= 2:
            wrong = {m: preds[(m, dataset)] != y_va for m in models}
            all_wrong = np.logical_and.reduce(list(wrong.values()))
            any_wrong = np.logical_or.reduce(list(wrong.values()))
            md += ["### Mistake overlap", "",
                   f"- Rows every model gets wrong: {all_wrong.sum()} "
                   f"({all_wrong.sum() / any_wrong.sum():.0%} of rows at least one model gets wrong)",
                   "", "| A's mistakes … | fixed by B | share |", "|---|---|---|"]
            for a, b in itertools.permutations(models, 2):
                fixed = (wrong[a] & ~wrong[b]).sum()
                md.append(f"| {a} → {b} | {fixed} of {wrong[a].sum()} | {fixed / wrong[a].sum():.0%} |")
            md.append("")
        else:
            md += ["Mistake overlap needs predictions from at least two models.", ""]

    md += ["## Efficiency (best validation configuration per model)", "",
           "| model | dataset | val accuracy | train time (s) | predict time (s) | size |",
           "|---|---|---|---|---|---|"]
    for (m, dataset), r in sorted(efficiency_rows().items()):
        if m == "knn":   # the model is the stored training matrix
            n_train, d = {"image": (16000, 500), "sentiment": (9600, 768)}[dataset]
            size = f"{n_train * d * 4 / 1e6:.1f} MB (stored training data)"
        else:
            size = "ask owner"
        md.append(f"| {m} | {dataset} | {float(r['val_acc']):.2%} | {r['train_time_s']} | "
                  f"{r['predict_time_s']} | {size} |")
    md += ["", "k-NN predict time is one neighbour search over the validation set, shared by all k values."]

    out = ROOT / "results" / "error_analysis.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
