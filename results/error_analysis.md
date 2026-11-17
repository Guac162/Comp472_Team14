# Error analysis

Validation set only (test labels are unknown). Re-run after new predictions are saved.

## image (4000 validation rows, chance 2.0%)

| model | val accuracy | val: most predicted class share | test: most predicted class share |
|---|---|---|---|
| knn | 57.20% | 2.8% | 2.8% |

Even split would be 2.0% per class.

### knn

Confusion matrix: `figures/confusion_knn_image.png`

10 most confused class pairs (both directions added):

| classes | mistakes | a→b | b→a |
|---|---|---|---|
| 37 ↔ 38 | 18 | 9 | 9 |
| 38 ↔ 39 | 11 | 2 | 9 |
| 6 ↔ 7 | 10 | 6 | 4 |
| 2 ↔ 4 | 10 | 4 | 6 |
| 37 ↔ 39 | 8 | 5 | 3 |
| 12 ↔ 13 | 8 | 2 | 6 |
| 6 ↔ 8 | 8 | 5 | 3 |
| 3 ↔ 18 | 8 | 5 | 3 |
| 46 ↔ 47 | 7 | 1 | 6 |
| 25 ↔ 27 | 7 | 0 | 7 |

Worst classes (share of their validation rows correct): 29 (40%), 22 (42%), 9 (44%), 41 (45%), 23 (46%)
Best classes: 46 (88%), 12 (78%), 48 (78%), 5 (72%), 49 (72%)

Mistake overlap needs predictions from at least two models.

## sentiment (2400 validation rows, chance 33.3%)

| model | val accuracy | val: most predicted class share | test: most predicted class share |
|---|---|---|---|
| knn | 58.54% | 44.0% | 60.0% |

Even split would be 33.3% per class.

### knn

Confusion matrix: `figures/confusion_knn_sentiment.png`

| class | precision | recall | F1 |
|---|---|---|---|
| 0 | 57.8% | 76.2% | 65.7% |
| 1 | 59.0% | 30.4% | 40.1% |
| 2 | 59.2% | 69.0% | 63.7% |

Hardest class by F1: 1. Check the label mapping on Kaggle to see whether it is Neutral.

Mistake overlap needs predictions from at least two models.

## Efficiency (best validation configuration per model)

| model | dataset | val accuracy | train time (s) | predict time (s) | size |
|---|---|---|---|---|---|
| knn | image | 57.20% | 0.00 | 0.49 | 32.0 MB (stored training data) |
| knn | sentiment | 58.54% | 0.00 | 6.15 | 29.5 MB (stored training data) |
| tree | image | 5.95% | 1.32 | 0.00 | ask owner |
| tree | sentiment | 48.46% | 1.17 | 0.00 | ask owner |

k-NN predict time is one neighbour search over the validation set, shared by all k values.
