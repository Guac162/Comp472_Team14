"""Correctness tests for src/knn.py against slow, obviously-correct reference code.

Run from the repo root:  python -m tests.test_knn
"""
import numpy as np

from src.knn import KNN


def reference_distances(Q, X, metric):
    """Plain double loop: slow but easy to verify by eye."""
    D = np.zeros((len(Q), len(X)))
    for i, q in enumerate(Q.astype(np.float64)):
        for j, x in enumerate(X.astype(np.float64)):
            if metric == "euclidean":
                D[i, j] = np.sqrt(((q - x) ** 2).sum())
            elif metric == "cosine":
                D[i, j] = 1 - q @ x / (np.linalg.norm(q) * np.linalg.norm(x))
            else:
                D[i, j] = np.abs(q - x).sum()
    return D


def test_distances_match_reference():
    rng = np.random.default_rng(0)
    X, Q = rng.normal(size=(60, 7)), rng.normal(size=(9, 7))
    for metric in ("euclidean", "cosine", "manhattan"):
        model = KNN(metric=metric).fit(X, np.zeros(60, dtype=int))
        assert np.allclose(model._distances(Q.astype(np.float32)), reference_distances(Q, X, metric),
                           atol=1e-4), metric


def test_neighbours_match_reference():
    rng = np.random.default_rng(1)
    X, Q = rng.normal(size=(200, 5)), rng.normal(size=(30, 5))
    for metric in ("euclidean", "cosine", "manhattan"):
        model = KNN(k=7, metric=metric, batch_size=8).fit(X, np.zeros(200, dtype=int))
        _, idx = model.kneighbors(Q)
        expected = np.argsort(reference_distances(Q, X, metric), axis=1, kind="stable")[:, :7]
        assert (idx == expected).all(), metric


def test_obvious_classes():
    X = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    for weights in ("uniform", "distance"):
        pred = KNN(k=3, weights=weights).fit(X, y).predict([[0.5, 0.5], [10.5, 10.5]])
        assert pred.tolist() == [0, 1], weights


def test_tie_goes_to_class_of_closest_neighbour():
    X = np.array([[0.0], [1.0], [-2.0], [3.0]])
    y = np.array([1, 0, 0, 1])           # k=2 at query 0.1: neighbours 0.0 (class 1) and 1.0 (class 0)
    assert KNN(k=2).fit(X, y).predict([[0.1]])[0] == 1
    assert KNN(k=4).fit(X, y).predict([[0.9]])[0] == 0   # 2-2 tie, closest is 1.0 (class 0)


def test_distance_weighting_changes_vote():
    X = np.array([[0.0], [5.0], [5.1]])
    y = np.array([0, 1, 1])
    assert KNN(k=3, weights="uniform").fit(X, y).predict([[0.2]])[0] == 1    # 2 votes beat 1
    assert KNN(k=3, weights="distance").fit(X, y).predict([[0.2]])[0] == 0   # the close one wins


def test_predict_for_ks_matches_predict():
    rng = np.random.default_rng(2)
    X, y, Q = rng.normal(size=(300, 4)), rng.integers(0, 3, 300), rng.normal(size=(50, 4))
    many = KNN(metric="cosine").fit(X, y).predict_for_ks(Q, [1, 3, 5, 10])
    for k, pred in many.items():
        assert (pred == KNN(k=k, metric="cosine").fit(X, y).predict(Q)).all(), k


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"passed  {t.__name__}")
    print(f"all {len(tests)} tests passed")
