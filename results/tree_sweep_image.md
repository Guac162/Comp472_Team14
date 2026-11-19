# Decision Tree sweep on image: validation accuracy (train accuracy in brackets)

Chance 2.0%. Best in bold.

| criterion | min_samples_leaf | depth 5 | depth 10 | depth 20 | depth None |
|---|---|---|---|---|---|
| gini | 1 | 5.9% (6%) | 8.0% (13%) | 10.2% (33%) | 10.3% (100%) |
| gini | 5 | 6.0% (6%) | 7.9% (11%) | 9.8% (25%) | 10.1% (49%) |
| gini | 20 | 6.0% (6%) | 8.1% (10%) | 10.6% (19%) | **11.7%** (25%) |
| entropy | 1 | 7.5% (8%) | 8.4% (17%) | 9.3% (69%) | 9.8% (100%) |
| entropy | 5 | 7.4% (8%) | 8.3% (15%) | 8.4% (36%) | 8.7% (44%) |
| entropy | 20 | 7.5% (8%) | 8.8% (12%) | 9.2% (19%) | 9.4% (21%) |
