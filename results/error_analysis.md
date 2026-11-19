# Error analysis

Validation set only (test labels are unknown). Re-run after new predictions are saved.

## image (4000 validation rows, chance 2.0%)

| model | val accuracy | val: most predicted class share | test: most predicted class share |
|---|---|---|---|
| forest | 33.95% | 4.3% | 3.6% |
| knn | 57.20% | 2.8% | 2.8% |
| nn | 34.90% | 2.8% | 2.9% |
| tree | 11.65% | 3.4% | 3.5% |

Even split would be 2.0% per class.

### forest

Confusion matrix: `figures/confusion_forest_image.png`

10 most confused class pairs (both directions added):

| classes | mistakes | a→b | b→a |
|---|---|---|---|
| 2 ↔ 4 | 27 | 4 | 23 |
| 2 ↔ 3 | 19 | 1 | 18 |
| 38 ↔ 39 | 18 | 12 | 6 |
| 6 ↔ 7 | 15 | 11 | 4 |
| 37 ↔ 39 | 14 | 10 | 4 |
| 25 ↔ 27 | 13 | 2 | 11 |
| 11 ↔ 14 | 12 | 2 | 10 |
| 6 ↔ 8 | 12 | 6 | 6 |
| 28 ↔ 43 | 10 | 2 | 8 |
| 15 ↔ 41 | 10 | 2 | 8 |

Worst classes (share of their validation rows correct): 9 (9%), 29 (9%), 27 (12%), 21 (12%), 40 (14%)
Best classes: 46 (80%), 2 (65%), 5 (62%), 49 (57%), 11 (57%)

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

### nn

Confusion matrix: `figures/confusion_nn_image.png`

10 most confused class pairs (both directions added):

| classes | mistakes | a→b | b→a |
|---|---|---|---|
| 35 ↔ 36 | 23 | 12 | 11 |
| 6 ↔ 7 | 18 | 15 | 3 |
| 1 ↔ 3 | 18 | 8 | 10 |
| 38 ↔ 39 | 17 | 10 | 7 |
| 2 ↔ 4 | 17 | 1 | 16 |
| 37 ↔ 39 | 14 | 9 | 5 |
| 20 ↔ 21 | 14 | 9 | 5 |
| 2 ↔ 3 | 13 | 5 | 8 |
| 34 ↔ 44 | 12 | 7 | 5 |
| 10 ↔ 12 | 12 | 6 | 6 |

Worst classes (share of their validation rows correct): 29 (11%), 21 (14%), 41 (16%), 22 (18%), 6 (20%)
Best classes: 46 (80%), 2 (66%), 32 (59%), 49 (56%), 5 (56%)

### tree

Confusion matrix: `figures/confusion_tree_image.png`

10 most confused class pairs (both directions added):

| classes | mistakes | a→b | b→a |
|---|---|---|---|
| 2 ↔ 3 | 24 | 17 | 7 |
| 2 ↔ 4 | 21 | 11 | 10 |
| 3 ↔ 4 | 19 | 9 | 10 |
| 15 ↔ 18 | 14 | 11 | 3 |
| 42 ↔ 49 | 12 | 7 | 5 |
| 5 ↔ 48 | 12 | 7 | 5 |
| 37 ↔ 39 | 11 | 10 | 1 |
| 21 ↔ 22 | 11 | 9 | 2 |
| 18 ↔ 33 | 11 | 5 | 6 |
| 12 ↔ 25 | 11 | 2 | 9 |

Worst classes (share of their validation rows correct): 31 (0%), 22 (0%), 21 (1%), 29 (1%), 40 (1%)
Best classes: 46 (54%), 5 (46%), 2 (32%), 45 (31%), 47 (30%)

### Mistake overlap

- Rows every model gets wrong: 1194 (32% of rows at least one model gets wrong)

| A's mistakes … | fixed by B | share |
|---|---|---|
| forest → knn | 1178 of 2642 | 45% |
| forest → nn | 551 of 2642 | 21% |
| forest → tree | 126 of 2642 | 5% |
| knn → forest | 248 of 1712 | 14% |
| knn → nn | 330 of 1712 | 19% |
| knn → tree | 109 of 1712 | 6% |
| nn → forest | 513 of 2604 | 20% |
| nn → knn | 1222 of 2604 | 47% |
| nn → tree | 146 of 2604 | 6% |
| tree → forest | 1018 of 3534 | 29% |
| tree → knn | 1931 of 3534 | 55% |
| tree → nn | 1076 of 3534 | 30% |

## sentiment (2400 validation rows, chance 33.3%)

| model | val accuracy | val: most predicted class share | test: most predicted class share |
|---|---|---|---|
| forest | 60.75% | 37.6% | 48.2% |
| knn | 58.54% | 44.0% | 60.0% |
| nn | 67.17% | 37.2% | 48.0% |
| tree | 50.25% | 37.5% | 45.7% |

Even split would be 33.3% per class.

### forest

Confusion matrix: `figures/confusion_forest_sentiment.png`

| class | precision | recall | F1 |
|---|---|---|---|
| 0 | 63.0% | 71.0% | 66.7% |
| 1 | 54.8% | 50.5% | 52.6% |
| 2 | 63.9% | 60.8% | 62.3% |

Hardest class by F1: 1. Check the label mapping on Kaggle to see whether it is Neutral.

### knn

Confusion matrix: `figures/confusion_knn_sentiment.png`

| class | precision | recall | F1 |
|---|---|---|---|
| 0 | 57.8% | 76.2% | 65.7% |
| 1 | 59.0% | 30.4% | 40.1% |
| 2 | 59.2% | 69.0% | 63.7% |

Hardest class by F1: 1. Check the label mapping on Kaggle to see whether it is Neutral.

### nn

Confusion matrix: `figures/confusion_nn_sentiment.png`

| class | precision | recall | F1 |
|---|---|---|---|
| 0 | 69.0% | 77.1% | 72.8% |
| 1 | 62.6% | 51.2% | 56.4% |
| 2 | 68.7% | 73.1% | 70.9% |

Hardest class by F1: 1. Check the label mapping on Kaggle to see whether it is Neutral.

### tree

Confusion matrix: `figures/confusion_tree_sentiment.png`

| class | precision | recall | F1 |
|---|---|---|---|
| 0 | 52.6% | 59.1% | 55.7% |
| 1 | 44.2% | 33.2% | 37.9% |
| 2 | 51.9% | 58.4% | 55.0% |

Hardest class by F1: 1. Check the label mapping on Kaggle to see whether it is Neutral.

### Mistake overlap

- Rows every model gets wrong: 370 (22% of rows at least one model gets wrong)

| A's mistakes … | fixed by B | share |
|---|---|---|
| forest → knn | 219 of 942 | 23% |
| forest → nn | 369 of 942 | 39% |
| forest → tree | 258 of 942 | 27% |
| knn → forest | 272 of 995 | 27% |
| knn → nn | 419 of 995 | 42% |
| knn → tree | 316 of 995 | 32% |
| nn → forest | 215 of 788 | 27% |
| nn → knn | 212 of 788 | 27% |
| nn → tree | 267 of 788 | 34% |
| tree → forest | 510 of 1194 | 43% |
| tree → knn | 515 of 1194 | 43% |
| tree → nn | 673 of 1194 | 56% |

## Efficiency (best validation configuration per model)

| model | dataset | val accuracy | train time (s) | predict time (s) | size |
|---|---|---|---|---|---|
| forest | image | 33.95% | 1451.23 | 4.09 | 183.0 MB (4,567,078 nodes) |
| forest | sentiment | 60.75% | 293.86 | 0.33 | 57.7 MB (807,818 nodes) |
| knn | image | 57.20% | 0.00 | 0.49 | 32.0 MB (stored training data) |
| knn | sentiment | 58.54% | 0.00 | 6.15 | 29.5 MB (stored training data) |
| nn | image | 34.90% | 90.30 | 0.40 | 1.31 MB (328,626 parameters) |
| nn | sentiment | 67.17% | 18.76 | 0.05 | 1.84 MB (459,779 parameters) |
| tree | image | 11.65% | 39.72 | 0.00 | 0.1 MB (1,203 nodes) |
| tree | sentiment | 50.25% | 1.00 | 0.00 | 0.1 MB (31 nodes) |

k-NN predict time is one neighbour search over the validation set, shared by all k values.
