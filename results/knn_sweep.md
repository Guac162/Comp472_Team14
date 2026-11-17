# k-NN sweep: best k and vote for each scaling and distance (validation accuracy)

## image (chance 2.0%)

| scaling | euclidean | cosine | manhattan |
|---|---|---|---|
| raw | 53.1% (k=1, uniform) | 53.1% (k=1, uniform) | 39.3% (k=1, uniform) |
| standard | 43.4% (k=1, uniform) | 53.4% (k=1, uniform) | 30.9% (k=1, uniform) |
| sqrt | **57.2% (k=1, uniform)** | 57.2% (k=1, uniform) | 44.2% (k=1, uniform) |

## sentiment (chance 33.3%)

| scaling | euclidean | cosine | manhattan |
|---|---|---|---|
| raw | 58.1% (k=50, distance) | 58.0% (k=50, uniform) | 58.5% (k=50, uniform) |
| standard | 58.4% (k=50, distance) | 57.3% (k=20, uniform) | **58.5% (k=50, uniform)** |
| l2 | 58.0% (k=50, uniform) | 58.0% (k=50, uniform) | 58.0% (k=50, distance) |

