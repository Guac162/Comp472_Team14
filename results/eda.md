# Quick EDA (train split only, so no peeking at validation)

## image

| | |
|---|---|
| rows (train / val / test) | 16000 / 4000 / 5000 |
| features | 500 |
| classes | 50, train rows per class 320-320 |
| value range | 0.000 to 0.923, mean 0.0152 |
| per-feature std | 0.0073 to 0.1000 (ratio 13.7x) |
| share of exact zeros | 85.2% |
| row L2 norm | 1.000 to 1.000 |
| constant features | 0 |
| NaN / inf values | 0 |
| test vs train shift | 0/500 features shifted beyond chance (z > 4); median 0.011 std, max 0.05 std |

## sentiment

| | |
|---|---|
| rows (train / val / test) | 9600 / 2400 / 5000 |
| features | 768 |
| classes | 3, train rows per class 3200-3200 |
| value range | -5.067 to 1.754, mean -0.0127 |
| per-feature std | 0.1460 to 0.6296 (ratio 4.3x) |
| share of exact zeros | 0.0% |
| row L2 norm | 7.587 to 13.598 |
| constant features | 0 |
| NaN / inf values | 0 |
| test vs train shift | 624/768 features shifted beyond chance (z > 4); median 0.185 std, max 0.77 std |

## What this means

- Classes are perfectly balanced, so plain accuracy is a fair metric.
- Image rows are already L2-normalized, so 'raw' and 'l2' are the same for images.
- Feature spreads differ, so distance-based models (k-NN, k-Means) should try standardization.
- Trees do not need scaling: split thresholds just move with the scale.
- Sentiment test rows differ from training rows (most features shifted); images do not.
  Possible causes: a different class mix or different source text in the test set.
  So sentiment validation accuracy may not match the Kaggle score; compare predicted
  class counts on test vs validation once a real model exists.
