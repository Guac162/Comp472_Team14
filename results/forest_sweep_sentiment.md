# Random Forest sweep on sentiment: validation accuracy by number of trees

Chance 33.3%. Input: {'scaling': 'raw', 'pca': None}. Best in bold.

| max_features | min_samples_leaf | 50 trees | 100 trees | 200 trees | 300 trees | OOB |
|---|---|---|---|---|---|---|
| log2 | 1 | 55.1% | 57.4% | 58.9% | 59.3% | 59.0% |
| sqrt | 1 | 56.2% | 58.1% | 59.3% | 60.7% | 60.1% |
| 2*sqrt | 1 | 57.1% | 59.0% | 60.2% | **60.8%** | 60.2% |
| 4*sqrt | 1 | 56.8% | 59.4% | 60.6% | 60.3% | 60.3% |
| log2 | 5 | 57.2% | 58.0% | 59.4% | 59.7% | 60.3% |
| sqrt | 5 | 56.4% | 57.6% | 58.6% | 59.6% | 60.4% |
| 2*sqrt | 5 | 58.4% | 59.0% | 60.0% | 59.8% | 60.3% |
| 4*sqrt | 5 | 57.8% | 58.6% | 59.6% | 60.0% | 60.2% |
