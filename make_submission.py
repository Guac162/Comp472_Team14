"""Combine image and sentiment test predictions into one Kaggle submission file.

Usage (from the repo root):
    python make_submission.py --image preds/test/knn_image.npy \
        --sentiment preds/test/nn_sentiment.npy --note "k-NN k=5 + NN baseline"

Writes submissions/<timestamp>.csv in the same row order as
data/sample_submission.csv, and appends a row to results/submissions_log.csv
(fill in public_score by hand after submitting on Kaggle).

TODO: merge in every assert from the course's example submission script.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DATA_DIR, load_all

ROOT = Path(__file__).resolve().parent
N_CLASSES = {"image": 50, "sentiment": 3}
LOG = ROOT / "results" / "submissions_log.csv"


def build(image_pred, sentiment_pred):
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert list(sample.columns) == ["id", "label"], sample.columns
    parts = []
    for name, pred in (("image", image_pred), ("sentiment", sentiment_pred)):
        _, _, X_test, ids = load_all(name)
        pred = np.asarray(pred)
        assert pred.ndim == 1, f"{name}: predictions must be 1-D, got shape {pred.shape}"
        assert len(pred) == len(ids) == len(X_test), f"{name}: {len(pred)} preds for {len(ids)} test rows"
        assert np.issubdtype(pred.dtype, np.integer), f"{name}: labels must be integers, got {pred.dtype}"
        assert pred.min() >= 0 and pred.max() < N_CLASSES[name], \
            f"{name}: labels must be in 0..{N_CLASSES[name] - 1}, got {pred.min()}..{pred.max()}"
        parts.append(pd.DataFrame({"id": ids, "label": pred}))
    preds = pd.concat(parts, ignore_index=True)
    assert preds["id"].is_unique, "duplicate ids"
    assert set(preds["id"]) == set(sample["id"]), "ids do not match sample_submission.csv"

    sub = sample[["id"]].merge(preds, on="id", how="left", validate="one_to_one")
    assert len(sub) == len(sample), f"expected {len(sample)} rows, got {len(sub)}"
    assert (sub["id"].to_numpy() == sample["id"].to_numpy()).all(), "row order changed"
    assert sub["label"].notna().all(), "missing labels"
    sub["label"] = sub["label"].astype(int)
    return sub


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image", required=True, help=".npy with 5000 image test predictions")
    parser.add_argument("--sentiment", required=True, help=".npy with 5000 sentiment test predictions")
    parser.add_argument("--note", required=True, help="what models/settings this is")
    args = parser.parse_args()

    sub = build(np.load(args.image), np.load(args.sentiment))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "submissions" / f"{stamp}.csv"
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)

    is_new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["date", "file", "image_preds", "sentiment_preds", "note", "public_score"])
        w.writerow([datetime.now().isoformat(timespec="minutes"), out.name,
                    args.image, args.sentiment, args.note, ""])
    print(f"wrote {out.relative_to(ROOT)} ({len(sub)} rows) - all checks passed")
    print(f"logged in {LOG.relative_to(ROOT)}; add the public score after submitting")


if __name__ == "__main__":
    main()
