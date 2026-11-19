# Ensemble: majority vote of saved predictions

## image

Single models: knn 57.20%, nn 34.90%, forest 33.95%, tree 11.65%

| models voting | val accuracy | vs best single |
|---|---|---|
| knn + nn + tree | 56.65% | -0.55 pts |
| knn + nn + forest | 56.47% | -0.72 pts |
| knn + nn + forest + tree | 54.65% | -2.55 pts |
| knn + forest + tree | 54.12% | -3.07 pts |
| nn + forest + tree | 34.30% | -22.90 pts |

**Decision:** keep the single model knn (57.20% on validation).

## sentiment

Single models: nn 67.17%, forest 60.75%, knn 58.54%, tree 50.25%

| models voting | val accuracy | vs best single |
|---|---|---|
| nn + forest + tree | 65.29% | -1.87 pts |
| nn + forest + knn + tree | 64.75% | -2.42 pts |
| nn + knn + tree | 63.58% | -3.58 pts |
| nn + forest + knn | 63.54% | -3.63 pts |
| forest + knn + tree | 60.25% | -6.92 pts |

**Decision:** keep the single model nn (67.17% on validation).

