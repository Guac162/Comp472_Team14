# Random Forest sweep on image: validation accuracy by number of trees

Chance 2.0%. Input: {'scaling': 'sqrt', 'pca': None}. Best in bold.

| max_features | min_samples_leaf | 50 trees | 100 trees | 200 trees | 300 trees | OOB |
|---|---|---|---|---|---|---|
| log2 | 1 | 21.1% | 26.7% | 31.6% | 33.7% | 28.2% |
| sqrt | 1 | 24.4% | 27.9% | 31.8% | **34.0%** | 29.4% |
| 2*sqrt | 1 | 24.0% | 28.4% | 31.8% | 33.0% | 29.8% |
