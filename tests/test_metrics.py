import numpy as np

from gfo.metrics import mae, spearman, top_k_recall


def test_mae():
    assert mae(np.array([0.0, 1.0]), np.array([0.5, 0.5])) == 0.5


def test_spearman_perfect_order():
    y = np.array([0.1, 0.5, 0.9])
    assert spearman(y, y * 10) == 1.0


def test_top_k_recall():
    y_true = np.array([0.9, 0.8, 0.1, 0.2])
    good = np.array([0.7, 0.6, 0.0, 0.1])   # finds both top gates
    bad = np.array([0.0, 0.1, 0.9, 0.8])    # finds none
    assert top_k_recall(y_true, good, 2) == 1.0
    assert top_k_recall(y_true, bad, 2) == 0.0


def test_top_k_recall_with_ties():
    # Three gates share the top value; any two of them are a correct top 2.
    y_true = np.array([1.0, 1.0, 1.0, 0.2])
    y_pred = np.array([0.1, 0.9, 0.8, 0.0])
    assert top_k_recall(y_true, y_pred, 2) == 1.0