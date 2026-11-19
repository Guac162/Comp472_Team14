"""PCA from scratch in NumPy, used before the forest.

Trees split on one feature at a time. The image features are 85% zeros, so a
single feature separates few rows. PCA rotates the data onto the directions
of largest variance, giving dense features where each one mixes many of the
original ones, which suits axis-aligned splits better.

Fitted on the training rows only, then applied to validation and test.
"""
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        # rows of Vt are the principal directions, sorted by variance
        _, s, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        var = s ** 2
        self.explained_ratio_ = var[:self.n_components] / var.sum()
        return self

    def transform(self, X):
        return ((np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
