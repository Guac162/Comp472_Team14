"""Decision Tree classifier from scratch in NumPy (no scikit-learn).

fit() first cuts every feature into n_bins buckets at its quantiles, so a
split only has to be tested at n_bins - 1 thresholds instead of at every
unique value. The tree is then grown recursively: at each node, try every
(feature, threshold) pair, keep the one with the lowest weighted impurity
of the two children, and stop when a stopping rule says so.

Split rule:  a row goes left if x[feature] < threshold.

Criteria (p_c = share of class c in a node):
    gini       1 - sum p_c^2
    entropy    -sum p_c log2 p_c

Leaves predict their majority class (ties go to the smaller class id).
"""
import numpy as np

CRITERIA = ("gini", "entropy")


def impurity(counts, criterion="gini"):
    """Impurity from class counts on the last axis. Works on any shape."""
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    if criterion == "gini":
        return 1.0 - (p ** 2).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)      # 0 * log 0 counts as 0
    return -(p * logp).sum(axis=-1)


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini", max_features=None, n_bins=32, seed=None):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
        if n_bins < 2 or n_bins > 256:
            raise ValueError("n_bins must be between 2 and 256")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = max(1, min_samples_leaf)     # a child can never be empty
        self.criterion = criterion
        self.max_features = max_features     # None = all features; the forest passes a smaller number
        self.n_bins = n_bins
        self.seed = seed

    def fit(self, X, y, n_classes=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes_ = n_classes or int(y.max()) + 1
        self.rng_ = np.random.default_rng(self.seed)

        # thresholds: the inner quantiles of each feature, shape (n_features, n_bins - 1)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = np.quantile(X, qs, axis=0).T.astype(np.float32)
        bins = self._to_bins(X)

        self.feature_, self.threshold_, self.left_, self.right_, self.label_ = [], [], [], [], []
        self.depth_ = 0
        self._grow(bins, y, np.arange(len(y)), depth=0)

        self.feature_ = np.array(self.feature_)
        self.threshold_ = np.array(self.threshold_, dtype=np.float32)
        self.left_ = np.array(self.left_)
        self.right_ = np.array(self.right_)
        self.label_ = np.array(self.label_)
        return self

    def _to_bins(self, X):
        """Bucket index of every value: bin b means edges[b-1] <= x < edges[b]."""
        bins = np.empty(X.shape, dtype=np.uint8)
        for f in range(X.shape[1]):
            bins[:, f] = np.searchsorted(self.edges_[f], X[:, f], side="right")
        return bins

    def _new_node(self, label):
        self.feature_.append(-1)            # -1 marks a leaf until a split is set
        self.threshold_.append(0.0)
        self.left_.append(-1)
        self.right_.append(-1)
        self.label_.append(label)
        return len(self.label_) - 1

    def _grow(self, bins, y, idx, depth):
        counts = np.bincount(y[idx], minlength=self.n_classes_)
        node = self._new_node(int(counts.argmax()))
        self.depth_ = max(self.depth_, depth)

        if (self.max_depth is not None and depth >= self.max_depth) \
                or len(idx) < self.min_samples_split or counts.max() == len(idx):
            return node
        split = self._best_split(bins, y, idx, counts)
        if split is None:
            return node

        f, b = split
        go_left = bins[idx, f] <= b
        self.feature_[node] = f
        self.threshold_[node] = self.edges_[f, b]
        self.left_[node] = self._grow(bins, y, idx[go_left], depth + 1)
        self.right_[node] = self._grow(bins, y, idx[~go_left], depth + 1)
        return node

    def _best_split(self, bins, y, idx, counts):
        """Best (feature, bin) for this node, or None if every split makes impurity worse.

        Zero-gain splits are allowed (like scikit-learn): on XOR-like data no single
        split helps, but the next level can.
        """
        n, C, K = len(idx), self.n_classes_, self.n_bins
        features = np.arange(bins.shape[1])
        if self.max_features is not None and self.max_features < len(features):
            features = self.rng_.choice(features, self.max_features, replace=False)

        # class counts per (feature, bin) in one bincount call
        B = bins[np.ix_(idx, features)].astype(np.int64)
        keys = (np.arange(len(features)) * K + B) * C + y[idx, None]
        hist = np.bincount(keys.ravel(), minlength=len(features) * K * C)
        hist = hist.reshape(len(features), K, C)

        left = hist.cumsum(axis=1)[:, :-1]          # rows with bin <= b, for b = 0 .. K-2
        right = counts - left
        n_left = left.sum(axis=2)
        n_right = n - n_left
        score = (n_left * impurity(left, self.criterion)
                 + n_right * impurity(right, self.criterion)) / n
        too_small = (n_left < self.min_samples_leaf) | (n_right < self.min_samples_leaf)
        score[too_small] = np.inf

        i, b = np.unravel_index(score.argmin(), score.shape)
        if not score[i, b] <= impurity(counts, self.criterion) + 1e-12:
            return None
        return int(features[i]), int(b)

    def predict(self, X):
        """Send all rows down the tree together, one level per loop."""
        X = np.asarray(X, dtype=np.float32)
        node = np.zeros(len(X), dtype=np.int64)
        rows = np.arange(len(X))
        while True:
            inner = self.feature_[node] >= 0
            if not inner.any():
                return self.label_[node]
            r, nd = rows[inner], node[inner]
            go_left = X[r, self.feature_[nd]] < self.threshold_[nd]
            node[inner] = np.where(go_left, self.left_[nd], self.right_[nd])

    @property
    def n_leaves(self):
        return int((self.feature_ < 0).sum())
