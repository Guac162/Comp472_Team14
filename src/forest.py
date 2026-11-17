"""Random Forest from scratch, built on src/tree.py (no scikit-learn).

Each tree is trained on a bootstrap sample (n rows drawn with replacement)
and only looks at a random subset of max_features features at every split.
Both kinds of randomness make the trees different from each other, so their
mistakes partly cancel out when they vote.

max_features:
    "sqrt"     round(sqrt(n_features)): 22 for images, 28 for sentiment
    "log2"     round(log2(n_features))
    int        that many features
    None       all features (bagging only)

Prediction is a majority vote over the trees (ties go to the smaller class id).
Rows a tree never saw (out-of-bag) give a free validation estimate: oob_score_.
"""
import numpy as np

from src.tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini", max_features="sqrt", n_bins=32, bootstrap=True, seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.n_bins = n_bins
        self.bootstrap = bootstrap
        self.seed = seed

    def _n_features_per_split(self, d):
        if self.max_features == "sqrt":
            return max(1, round(np.sqrt(d)))
        if self.max_features == "log2":
            return max(1, round(np.log2(d)))
        if self.max_features is None:
            return d
        return int(self.max_features)

    def fit(self, X, y, n_classes=None, verbose=False):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n = len(y)
        self.n_classes_ = n_classes or int(y.max()) + 1
        self.max_features_ = self._n_features_per_split(X.shape[1])
        rng = np.random.default_rng(self.seed)

        self.trees_ = []
        oob_votes = np.zeros((n, self.n_classes_), dtype=np.int32)
        for t in range(self.n_trees):
            rows = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf, criterion=self.criterion,
                                max_features=self.max_features_, n_bins=self.n_bins,
                                seed=int(rng.integers(2**31)))
            tree.fit(X[rows], y[rows], n_classes=self.n_classes_)
            self.trees_.append(tree)

            if self.bootstrap:
                oob = np.setdiff1d(np.arange(n), rows)       # about 37% of rows per tree
                oob_votes[oob, tree.predict(X[oob])] += 1
            if verbose and (t + 1) % 10 == 0:
                print(f"    {t + 1}/{self.n_trees} trees")

        if self.bootstrap:
            seen = oob_votes.sum(axis=1) > 0                 # rows that were out-of-bag at least once
            self.oob_score_ = float((oob_votes[seen].argmax(axis=1) == y[seen]).mean())
        return self

    def votes(self, X):
        """Vote counts per class, shape (n_rows, n_classes)."""
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros((len(X), self.n_classes_), dtype=np.int32)
        rows = np.arange(len(X))
        for tree in self.trees_:
            out[rows, tree.predict(X)] += 1
        return out

    def predict(self, X):
        return self.votes(X).argmax(axis=1)

    def predict_for_n_trees(self, X, ns):
        """Predictions using only the first n trees, for each n in ns (one pass, for sweeps)."""
        X = np.asarray(X, dtype=np.float32)
        out, votes = {}, np.zeros((len(X), self.n_classes_), dtype=np.int32)
        rows = np.arange(len(X))
        for i, tree in enumerate(self.trees_, start=1):
            votes[rows, tree.predict(X)] += 1
            if i in ns:
                out[i] = votes.argmax(axis=1)
        return out

    def feature_importance(self):
        """How often each feature is used for a split, summed over all trees, as shares."""
        d = len(self.trees_[0].edges_)
        counts = np.zeros(d)
        for tree in self.trees_:
            used = tree.feature_[tree.feature_ >= 0]
            counts += np.bincount(used, minlength=d)
        return counts / counts.sum()
