"""Correctness tests for src/tree.py on tiny datasets where the answer is obvious.

Run from the repo root:  python -m tests.test_tree
"""
import numpy as np

from src.tree import DecisionTree, impurity


def test_impurity_values():
    assert np.isclose(impurity(np.array([5, 5]), "gini"), 0.5)
    assert np.isclose(impurity(np.array([5, 5]), "entropy"), 1.0)
    assert np.isclose(impurity(np.array([10, 0]), "gini"), 0.0)
    assert np.isclose(impurity(np.array([10, 0]), "entropy"), 0.0)
    assert np.isclose(impurity(np.array([1, 1, 1, 1]), "entropy"), 2.0)


def test_obvious_split():
    # only feature 1 matters: class 1 iff x1 > 0.5; feature 0 is noise
    rng = np.random.default_rng(0)
    X = rng.random((200, 2))
    y = (X[:, 1] > 0.5).astype(int)
    for criterion in ("gini", "entropy"):
        tree = DecisionTree(max_depth=1, criterion=criterion).fit(X, y)
        assert tree.feature_[0] == 1, criterion
        assert abs(tree.threshold_[0] - 0.5) < 0.05, criterion
        assert (tree.predict(X) == y).mean() > 0.95, criterion


def test_xor_needs_depth_two():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10, dtype=float)
    y = np.array([0, 1, 1, 0] * 10)
    assert (DecisionTree(max_depth=2, n_bins=2).fit(X, y).predict(X) == y).all()


def test_pure_node_is_a_leaf():
    X = np.random.default_rng(1).random((50, 3))
    tree = DecisionTree().fit(X, np.zeros(50, dtype=int))
    assert tree.n_leaves == 1 and tree.depth_ == 0


def test_no_depth_limit_fits_training_data():
    rng = np.random.default_rng(2)
    X, y = rng.random((300, 4)), rng.integers(0, 3, 300)
    tree = DecisionTree(n_bins=256).fit(X, y)
    assert (tree.predict(X) == y).mean() > 0.9      # bins cap how far it can memorize


def test_stopping_rules():
    rng = np.random.default_rng(3)
    X, y = rng.random((500, 5)), rng.integers(0, 4, 500)
    assert DecisionTree(max_depth=3).fit(X, y).depth_ <= 3
    tree = DecisionTree(min_samples_leaf=40).fit(X, y)
    leaf_sizes = np.bincount(_leaf_of(tree, X))
    assert leaf_sizes[leaf_sizes > 0].min() >= 40


def test_max_features_is_repeatable():
    rng = np.random.default_rng(4)
    X, y = rng.random((300, 10)), rng.integers(0, 3, 300)
    a = DecisionTree(max_features=3, seed=7).fit(X, y).predict(X)
    b = DecisionTree(max_features=3, seed=7).fit(X, y).predict(X)
    assert (a == b).all()


def _leaf_of(tree, X):
    """Slow one-row-at-a-time walk, to check predict() and leaf sizes."""
    out = []
    for x in X:
        node = 0
        while tree.feature_[node] >= 0:
            node = tree.left_[node] if x[tree.feature_[node]] < tree.threshold_[node] else tree.right_[node]
        out.append(node)
    return np.array(out)


def test_predict_matches_slow_walk():
    rng = np.random.default_rng(5)
    X, y = rng.random((400, 6)), rng.integers(0, 5, 400)
    tree = DecisionTree(max_depth=6).fit(X, y)
    Q = rng.random((100, 6))
    assert (tree.predict(Q) == tree.label_[_leaf_of(tree, Q)]).all()


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"passed  {t.__name__}")
    print(f"all {len(tests)} tests passed")
