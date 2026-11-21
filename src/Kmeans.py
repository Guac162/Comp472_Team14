"""k-Means from scratch in NumPy: k-means++ initialisation, several restarts, Lloyd iterations.

Distances use |x|^2 + |c|^2 - 2 x.c (one matrix product) and are processed in
batches to bound memory. An empty cluster is re-seeded at the point that is
currently farthest from its centre. The restart with the lowest inertia wins.
"""
import numpy as np


class KMeans:
    def __init__(self, k=3, n_init=10, max_iter=300, tol=1e-4, seed=42, batch_size=2000):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k, self.n_init, self.max_iter = k, n_init, max_iter
        self.tol, self.seed, self.batch_size = tol, seed, batch_size

    @staticmethod
    def _sqdist(X, C):     #expanding (x - c)^2 - use one matrix product instead of a loop
        d2 = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2.0 * (X @ C.T)
        return np.maximum(d2, 0.0)                                 
    def _assign(self, X, C):
        """Nearest centre for every row: (labels, squared distance to it)."""
        labels = np.empty(len(X), dtype=np.int64)
        mind = np.empty(len(X), dtype=np.float64)
        #n x k matrix can get big - do in chunks
        for s in range(0, len(X), self.batch_size):
            D = self._sqdist(X[s:s + self.batch_size], C)
            lab = D.argmin(axis=1)
            labels[s:s + len(lab)] = lab
            mind[s:s + len(lab)] = D[np.arange(len(lab)), lab]
        return labels, mind

    def _init_pp(self, X, rng):
        """k-means++: each new centre is drawn with probability proportional to D(x)^2."""
        n = len(X)
        C = np.empty((self.k, X.shape[1]), dtype=np.float64)
        C[0] = X[rng.integers(n)] #first centre is just a random row
        mind = self._sqdist(X, C[:1])[:, 0]
        for j in range(1, self.k):
            total = mind.sum()
            #if every point sits on a centre already, fall back to a random pick
            idx = rng.integers(n) if total <= 0 else rng.choice(n, p=mind / total)
            C[j] = X[idx]
            #keep the distance to the closest centre so far
            mind = np.minimum(mind, self._sqdist(X, C[j:j + 1])[:, 0])
        return C

    def _lloyd(self, X, C, tol_abs):
        for it in range(1, self.max_iter + 1):
            labels, mind = self._assign(X, C)
            counts = np.bincount(labels, minlength=self.k)
            newC = np.empty_like(C)
            for j in range(self.k):
                if counts[j] > 0:
                    newC[j] = X[labels == j].mean(axis=0)
                else:                                   # empty cluster: take the worst-fitted point
                    far = int(mind.argmax())
                    newC[j] = X[far]
                    mind[far] = 0.0 #two empty clusters dont take the same point
            shift = ((newC - C) ** 2).sum()
            C = newC
            if shift <= tol_abs: #centers barely moved-stop
                break
        labels, mind = self._assign(X, C)               #labels/inertia that match the final centres
        return C, labels, float(mind.sum()), it

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.k > len(X):
            raise ValueError(f"k={self.k} is larger than the {len(X)} rows")
        tol_abs = self.tol * float(X.var(axis=0).mean())   #scale tol to data
        rng = np.random.default_rng(self.seed)
        best = None #run several restarts, keep one with lowest inertia
        for _ in range(self.n_init):
            C, labels, inertia, n_iter = self._lloyd(X, self._init_pp(X, rng), tol_abs)
            if best is None or inertia < best[2]:
                best = (C, labels, inertia, n_iter)
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X):
        return self._assign(np.asarray(X, dtype=np.float64), self.cluster_centers_)[0]
