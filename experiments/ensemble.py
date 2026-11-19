"""Simple ensemble (plan item, Nov 14): majority vote of saved model predictions.

For each dataset, every combination of 3 or more models (from knn, forest, nn,
tree) votes on each row. Ties are broken by a fixed rule: the tied class
predicted by the model with the highest single validation accuracy wins.
(With only 2 models every disagreement is a tie, so a pair always equals its
better member; that is why combinations start at 3.)

The ensemble is kept only if it beats the best single model on validation.
Note: picking the best of a handful of combinations on validation is slightly
optimistic, so a small gain should be read with caution.

Uses the predictions already in preds/val and preds/test (all fitted on the
80% split). Writes results/ensemble.md; if an ensemble wins, logs it and saves
preds/*/ensemble_<dataset>.npy.

Run from the repo root:  python -m experiments.ensemble
"""
import itertools
import json
from pathlib import Path

import numpy as np

from src.data import load
from src.utils import accuracy, log_result, save_preds

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("knn", "forest", "nn", "tree")
N_CLASSES = {"image": 50, "sentiment": 3}


def vote(pred_list, n_classes):
    """Majority vote; pred_list is ordered by priority (best model first) for tie-breaks."""
    P = np.stack(pred_list)                       # (n_models, n_rows)
    rows = np.arange(P.shape[1])
    counts = np.zeros((P.shape[1], n_classes), dtype=int)
    for p in P:
        np.add.at(counts, (rows, p), 1)
    tied = counts == counts.max(axis=1, keepdims=True)
    # first model (in priority order) whose prediction is among the tied top classes
    first = np.argmax(tied[rows[None, :], P], axis=0)
    return P[first, rows]


def main():
    md = ["# Ensemble: majority vote of saved predictions", ""]
    summary = {}
    for dataset in ("image", "sentiment"):
        _, _, _, y_va, _, _ = load(dataset)
        C = N_CLASSES[dataset]
        val = {m: np.load(ROOT / "preds" / "val" / f"{m}_{dataset}.npy") for m in MODELS}
        test = {m: np.load(ROOT / "preds" / "test" / f"{m}_{dataset}.npy") for m in MODELS}
        single = {m: accuracy(y_va, val[m]) for m in MODELS}
        ranked = sorted(MODELS, key=single.get, reverse=True)
        best_single = ranked[0]

        results = []
        for r in range(3, len(MODELS) + 1):
            for combo in itertools.combinations(ranked, r):   # keeps priority order
                acc = accuracy(y_va, vote([val[m] for m in combo], C))
                results.append((acc, combo))
        results.sort(reverse=True)
        best_acc, best_combo = results[0]

        md += [f"## {dataset}", "", "Single models: "
               + ", ".join(f"{m} {single[m]:.2%}" for m in ranked), "",
               "| models voting | val accuracy | vs best single |", "|---|---|---|"]
        md += [f"| {' + '.join(c)} | {a:.2%} | {(a - single[best_single]) * 100:+.2f} pts |"
               for a, c in results]
        kept = best_acc > single[best_single]
        md += ["", f"**Decision:** {'keep the ensemble ' + ' + '.join(best_combo) if kept else 'keep the single model ' + best_single}"
               f" ({max(best_acc, single[best_single]):.2%} on validation).", ""]
        print(f"{dataset}: best single {best_single} {single[best_single]:.4f}; "
              f"best ensemble {' + '.join(best_combo)} {best_acc:.4f} -> {'ENSEMBLE' if kept else 'single'}")

        if kept:
            val_pred = vote([val[m] for m in best_combo], C)
            test_pred = vote([test[m] for m in best_combo], C)
            log_result(model="ensemble", dataset=dataset, who="Mohammed",
                       params={"vote": "majority", "tie_break": "best single model",
                               "models": list(best_combo)},
                       train_acc=None, val_acc=best_acc, train_time_s=0.0, predict_time_s=0.0)
            save_preds("ensemble", dataset, val_pred=val_pred, test_pred=test_pred)
            share = np.bincount(test_pred, minlength=C).max() / len(test_pred)
            md += [f"Ensemble test predictions: most common class {share:.1%} of rows.", ""]
        summary[dataset] = {"choice": "ensemble" if kept else best_single,
                            "models": list(best_combo) if kept else [best_single],
                            "val_acc": max(best_acc, single[best_single])}

    (ROOT / "results" / "ensemble.md").write_text("\n".join(md) + "\n")
    (ROOT / "results" / "ensemble_best.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
