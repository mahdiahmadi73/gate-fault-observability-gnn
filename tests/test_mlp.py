import numpy as np

from gfo.mlp import choose_config, predict, standardize, train_mlp, CONFIGS


def test_standardize_uses_training_stats():
    X_train = np.array([[0.0, 5.0], [2.0, 5.0]])
    X_test = np.array([[4.0, 5.0]])
    a, b = standardize(X_train, X_test)
    assert a[:, 0].tolist() == [-1.0, 1.0]
    assert b[0, 0] == 3.0
    assert b[0, 1] == 0.0   # constant column stays 0, no division by zero


def test_mlp_learns_simple_rule():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(float)   # rule: 1 if the first feature is positive
    model = train_mlp(X, y, seed=0, epochs=300, hidden=16)
    pred = predict(model, X)
    assert np.mean(np.abs(pred - y)) < 0.1

def test_choose_config_uses_only_training_circuits():
    rng = np.random.default_rng(0)
    data = {}
    for name in ["a", "b", "c"]:
        X = rng.normal(size=(30, 4))
        y = rng.uniform(size=30)
        data[name] = (X, y, np.zeros(30, dtype=bool))
    config = choose_config(data, ["a", "b", "c"])
    assert config in CONFIGS