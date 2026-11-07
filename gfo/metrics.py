import numpy as np
from scipy.stats import spearmanr


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman(y_true, y_pred):
    return float(spearmanr(y_true, y_pred).statistic)


def top_k_recall(y_true, y_pred, k):
    # The model picks its k most sensitive gates. A pick is a hit if its true C
    # is at least as high as the k-th highest true C (this handles ties, e.g. many C = 1).
    threshold = np.sort(y_true)[::-1][k - 1]
    picked = np.argsort(-y_pred, kind="stable")[:k]
    hits = np.sum(y_true[picked] >= threshold)
    return float(hits / k)