"""Correctness tests for src/forest.py.

Run from the repo root:  python -m tests.test_forest
"""
import numpy as np

from src.forest import RandomForest
from src.tree import DecisionTree


def test_obvious_classes():
    rng = np.random.default_rng(0)
    X = rng.random((300, 5))
    y = (X[:, 2] > 0.5).astype(int)
    forest = RandomForest(n_trees=10, max_features=None, seed=0).fit(X, y)
    assert (forest.predict(X) == y).mean() > 0.95


def test_same_seed_same_forest():
    rng = np.random.default_rng(1)
    X, y = rng.random((200, 8)), rng.integers(0, 3, 200)
    a = RandomForest(n_trees=5, seed=3).fit(X, y).predict(X)
    b = RandomForest(n_trees=5, seed=3).fit(X, y).predict(X)
    assert (a == b).all()


def test_max_features_options():
    forest = RandomForest()
    assert forest._n_features_per_split(500) == 22
    assert forest._n_features_per_split(768) == 28
    assert RandomForest(max_features="log2")._n_features_per_split(512) == 9
    assert RandomForest(max_features=None)._n_features_per_split(10) == 10
    assert RandomForest(max_features=7)._n_features_per_split(10) == 7


def test_votes_add_up():
    rng = np.random.default_rng(2)
    X, y = rng.random((150, 4)), rng.integers(0, 4, 150)
    forest = RandomForest(n_trees=7, seed=0).fit(X, y)
    votes = forest.votes(X)
    assert (votes.sum(axis=1) == 7).all()
    assert (forest.predict(X) == votes.argmax(axis=1)).all()


def test_predict_for_n_trees_matches_smaller_forest():
    rng = np.random.default_rng(3)
    X, y, Q = rng.random((200, 6)), rng.integers(0, 3, 200), rng.random((50, 6))
    forest = RandomForest(n_trees=6, seed=5).fit(X, y)
    small = RandomForest(n_trees=6, seed=5).fit(X, y)
    small.trees_ = small.trees_[:3]
    assert (forest.predict_for_n_trees(Q, [3])[3] == small.predict(Q)).all()
    assert (forest.predict_for_n_trees(Q, [6])[6] == forest.predict(Q)).all()


def test_forest_beats_single_tree_on_noisy_data():
    # labels depend on a sum of features plus noise: one deep tree overfits, a forest averages it out
    rng = np.random.default_rng(4)
    X = rng.random((1200, 10))
    y = ((X[:, :5].sum(axis=1) + rng.normal(0, 0.5, 1200)) > 2.5).astype(int)
    X_tr, y_tr, X_va, y_va = X[:800], y[:800], X[800:], y[800:]
    tree_acc = (DecisionTree().fit(X_tr, y_tr).predict(X_va) == y_va).mean()
    forest_acc = (RandomForest(n_trees=30, seed=0).fit(X_tr, y_tr).predict(X_va) == y_va).mean()
    assert forest_acc > tree_acc


def test_oob_score_is_sensible():
    rng = np.random.default_rng(5)
    X = rng.random((400, 5))
    y = (X[:, 0] > 0.5).astype(int)
    forest = RandomForest(n_trees=20, seed=0).fit(X, y)
    assert 0.9 < forest.oob_score_ <= 1.0
    assert 0.3 < RandomForest(n_trees=20, seed=0).fit(X, rng.permutation(y)).oob_score_ < 0.7


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"passed  {t.__name__}")
    print(f"all {len(tests)} tests passed")
