import csv

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from gfo.bench_parser import GATE_TYPES
from gfo.features import FEATURE_NAMES
from gfo.metrics import mae, spearman, top_k_recall

CIRCUITS = ["c432", "c499", "c880", "c1355"]
TWINS = {"c499": "c1355", "c1355": "c499"}   # same function, different gates


def load_circuit(name):
    labels = {}
    with open(f"labels/{name}.csv") as f:
        for row in csv.DictReader(f):
            labels[row["gate"]] = float(row["C"])

    X = []
    y = []
    is_output = []
    with open(f"features/{name}.csv") as f:
        for row in csv.DictReader(f):
            numbers = [float(row[k]) for k in FEATURE_NAMES]
            one_hot = [1.0 if row["type"] == t else 0.0 for t in GATE_TYPES]
            X.append(numbers + one_hot)
            y.append(labels[row["gate"]])
            is_output.append(row["is_output"] == "1")

    return np.array(X), np.array(y), np.array(is_output)


def heuristic_score(X):
    # No training: more outputs reached and closer to an output = more sensitive.
    outputs_reached = X[:, FEATURE_NAMES.index("outputs_reached")]
    dist = X[:, FEATURE_NAMES.index("dist_to_output")]
    return outputs_reached / (1 + dist)


def train_gradboost(data, train_names):
    X_train = np.concatenate([data[name][0] for name in train_names])
    y_train = np.concatenate([data[name][1] for name in train_names])
    model = GradientBoostingRegressor(random_state=0)
    model.fit(X_train, y_train)
    return model, y_train.mean()


def print_row(circuit, model, y_true, y_pred, rank_only=False):
    k = max(1, len(y_true) // 10)   # top 10% of the gates
    mae_text = "   -  " if rank_only else f"{mae(y_true, y_pred):.3f}"
    if model == "mean":
        rank_text = "     -        -"
    else:
        rank_text = f"{spearman(y_true, y_pred):6.3f}   {top_k_recall(y_true, y_pred, k):.2f}"
    print(f"{circuit:6} {model:10} {mae_text}  {rank_text}")


if __name__ == "__main__":
    data = {name: load_circuit(name) for name in CIRCUITS}

    print("Leave-one-circuit-out, interior gates only (output gates removed)")
    print("test   model        MAE   Spearman top10%")

    for test in CIRCUITS:
        X_test, y_test, is_output = data[test]
        inside = ~is_output   # interior gates

        train = [name for name in CIRCUITS if name != test]
        model, train_mean = train_gradboost(data, train)
        gb_pred = model.predict(X_test)
        mean_pred = np.full(len(y_test), train_mean)
        heur = heuristic_score(X_test)

        print_row(test, "mean", y_test[inside], mean_pred[inside])
        print_row(test, "heuristic", y_test[inside], heur[inside], rank_only=True)
        print_row(test, "gradboost", y_test[inside], gb_pred[inside])

        if test in TWINS:
            # Same test, but without its twin in training: a harder, more honest test.
            train = [name for name in train if name != TWINS[test]]
            model, _ = train_gradboost(data, train)
            print_row(test, "gb-no-twin", y_test[inside], model.predict(X_test)[inside])
        print()