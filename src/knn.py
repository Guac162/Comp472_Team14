"""k-Nearest Neighbours from scratch in NumPy (no scikit-learn).

fit() only stores the training rows. predict() takes a batch of query rows,
computes the distance to every training row, keeps the k closest and votes.

Distances (query row q, training row x):
    euclidean  sqrt(sum (q - x)^2), computed as |q|^2 + |x|^2 - 2 q.x (one matrix product)
    cosine     1 - q.x / (|q| |x|)
    manhattan  sum |q - x|  (no matrix trick exists, so it is chunked to bound memory)

Voting:
    uniform    each of the k neighbours gets one vote
    distance   each neighbour's vote is weighted by 1 / distance
    ties       among the tied classes, the one owning the closest neighbour wins
"""
import numpy as np

METRICS = ("euclidean", "cosine", "manhattan")
WEIGHTS = ("uniform", "distance")
MANHATTAN_MAX_BYTES = 256e6     # memory cap for one manhattan chunk


class KNN:
    def __init__(self, k=5, metric="euclidean", weights="uniform", batch_size=500):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if weights not in WEIGHTS:
            raise ValueError(f"weights must be one of {WEIGHTS}, got {weights!r}")
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k, self.metric, self.weights, self.batch_size = k, metric, weights, batch_size

    def fit(self, X, y):
        """k-NN has no training step: store the data and precompute per-row terms."""
        self.X_ = np.ascontiguousarray(X, dtype=np.float32)
        self.y_ = np.asarray(y, dtype=np.int64)
        self.n_classes_ = int(self.y_.max()) + 1
        if self.metric == "euclidean":
            self.sq_norms_ = (self.X_ ** 2).sum(axis=1)
        elif self.metric == "cosine":
            self.unit_ = self.X_ / np.maximum(np.linalg.norm(self.X_, axis=1, keepdims=True), 1e-12)
        return self

    def _distances(self, Q):
        """Distances from each query row in Q to every training row: shape (len(Q), n_train)."""
        if self.metric == "euclidean":
            d2 = (Q ** 2).sum(axis=1)[:, None] + self.sq_norms_[None, :] - 2.0 * (Q @ self.X_.T)
            return np.sqrt(np.maximum(d2, 0.0))          # rounding can give tiny negatives
        if self.metric == "cosine":
            Q_unit = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
            return 1.0 - Q_unit @ self.unit_.T
        # manhattan: a (rows, n_train, n_features) block per chunk, sized to stay under the cap
        out = np.empty((len(Q), len(self.X_)), dtype=np.float32)
        step = max(1, int(MANHATTAN_MAX_BYTES // self.X_.nbytes))
        for i in range(0, len(Q), step):
            out[i:i + step] = np.abs(Q[i:i + step, None, :] - self.X_[None, :, :]).sum(axis=2)
        return out

    def kneighbors(self, X, k=None):
        """The k nearest training rows for each query row, closest first.

        Returns (dist, idx), each of shape (n_queries, k). Equal distances are
        ordered by training-row index, so results are repeatable.
        """
        k = k or self.k
        if k > len(self.X_):
            raise ValueError(f"k={k} is larger than the {len(self.X_)} training rows")
        X = np.asarray(X, dtype=np.float32)
        dist = np.empty((len(X), k), dtype=np.float32)
        idx = np.empty((len(X), k), dtype=np.int64)
        for s in range(0, len(X), self.batch_size):
            D = self._distances(X[s:s + self.batch_size])
            part = np.argpartition(D, k - 1, axis=1)[:, :k]     # the k smallest, unordered
            part_d = np.take_along_axis(D, part, axis=1)
            order = np.lexsort((part, part_d))                  # sort by distance, then index
            idx[s:s + len(D)] = np.take_along_axis(part, order, axis=1)
            dist[s:s + len(D)] = np.take_along_axis(part_d, order, axis=1)
        return dist, idx

    def _vote(self, dist, idx):
        labels = self.y_[idx]                                   # (n, k), closest first
        w = np.ones_like(dist) if self.weights == "uniform" else 1.0 / (dist + 1e-8)
        rows = np.arange(len(labels))[:, None]
        votes = np.zeros((len(labels), self.n_classes_), dtype=np.float64)
        np.add.at(votes, (rows, labels), w)
        tied = votes == votes.max(axis=1, keepdims=True)        # classes with the top score
        first_tied = np.argmax(tied[rows, labels], axis=1)      # closest neighbour in a tied class
        return labels[np.arange(len(labels)), first_tied]

    def predict(self, X):
        return self._vote(*self.kneighbors(X))

    def predict_for_ks(self, X, ks):
        """Predictions for several k values from one neighbour search (for sweeps)."""
        dist, idx = self.kneighbors(X, max(ks))
        return {k: self._vote(dist[:, :k], idx[:, :k]) for k in ks}
