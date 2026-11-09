import sys
import time

import numpy as np
import torch

from gfo.baselines import CIRCUITS, TWINS, load_circuit
from gfo.bench_parser import parse_bench
from gfo.gnn import DirGraphSAGE
from gfo.graph import build_edges, gate_order
from gfo.metrics import mae, spearman, top_k_recall

SEEDS = [0, 1, 2]
CONFIGS = [(16, 2, 100), (16, 3, 100), (32, 2, 200), (32, 3, 200)]   # (hidden, layers, epochs), fixed in advance


def load_graph(name):
    X, y, is_output = load_circuit(name)
    circuit = parse_bench(f"data/bench/{name}.bench")
    edges = build_edges(circuit, gate_order(name))
    return X, y, is_output, edges


def training_scale(data, train_names):
    # Mean and std of every feature, from the TRAINING circuits only.
    X_train = np.concatenate([data[name][0] for name in train_names])
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def to_tensors(data, name, mean, std):
    X, y, is_output, edges = data[name]
    x = torch.tensor((X - mean) / std, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y, edges


def train_gnn(data, train_names, hidden, n_layers, epochs, seed, lr=0.01):
    torch.manual_seed(seed)
    mean, std = training_scale(data, train_names)
    graphs = [to_tensors(data, name, mean, std) for name in train_names]
    n_gates = sum(len(y) for x, y, edges in graphs)

    model = DirGraphSAGE(graphs[0][0].shape[1], hidden, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        squared_error = 0
        for x, y, edges in graphs:   # one whole circuit at a time
            squared_error = squared_error + ((model(x, edges) - y) ** 2).sum()
        loss = squared_error / n_gates   # every gate counts the same, like the MLP
        loss.backward()
        optimizer.step()

    return model, mean, std


def fit_and_predict(data, train_names, eval_name, hidden, n_layers, epochs, seed):
    model, mean, std = train_gnn(data, train_names, hidden, n_layers, epochs, seed)
    x, y, edges = to_tensors(data, eval_name, mean, std)

    start = time.perf_counter()
    with torch.no_grad():
        pred = model(x, edges).numpy()
    seconds = time.perf_counter() - start

    inside = ~data[eval_name][2]
    return y.numpy()[inside], pred[inside], seconds


def choose_config(data, train_names):
    # Only training circuits are used here: each one takes a turn as validation.
    best_score = None
    best_config = None
    for hidden, n_layers, epochs in CONFIGS:
        errors = []
        for val in train_names:
            inner_train = [name for name in train_names if name != val]
            y, pred, _ = fit_and_predict(data, inner_train, val, hidden, n_layers, epochs, seed=0)
            errors.append(mae(y, pred))
        score = np.mean(errors)
        if best_score is None or score < best_score:
            best_score = score
            best_config = (hidden, n_layers, epochs)
    return best_config


def run(data, test, train_names):
    hidden, n_layers, epochs = choose_config(data, train_names)

    results = []
    times = []
    for seed in SEEDS:
        y, pred, seconds = fit_and_predict(data, train_names, test, hidden, n_layers, epochs, seed)
        k = max(1, len(y) // 10)
        results.append([mae(y, pred), spearman(y, pred), top_k_recall(y, pred, k)])
        times.append(seconds)

    results = np.array(results)
    return (hidden, n_layers, epochs), results.mean(axis=0), results.std(axis=0), np.mean(times)


def print_row(test, model_name, config, mean, std, seconds):
    print(f"{test:6} {model_name:11} {str(config):13} "
          f"{mean[0]:.3f}±{std[0]:.3f}  {mean[1]:.3f}±{std[1]:.3f}  {mean[2]:.2f}±{std[2]:.2f}  "
          f"{seconds * 1000:.1f} ms")


if __name__ == "__main__":
    # Optional: name test circuits on the command line, e.g. python -m gfo.train_gnn c432
    tests = sys.argv[1:] if len(sys.argv) > 1 else CIRCUITS
    data = {name: load_graph(name) for name in CIRCUITS}

    print(f"Direction-aware GraphSAGE, leave-one-circuit-out, interior gates only, seeds {SEEDS}")
    print("config (hidden, layers, epochs) chosen by validation MAE on the training circuits only")
    print("test   model       config        MAE          Spearman      top10%      predict time")

    for test in tests:
        start = time.perf_counter()
        train = [name for name in CIRCUITS if name != test]
        config, mean, std, seconds = run(data, test, train)
        print_row(test, "graphsage", config, mean, std, seconds)

        if test in TWINS:
            train = [name for name in train if name != TWINS[test]]
            config, mean, std, seconds = run(data, test, train)
            print_row(test, "gs-no-twin", config, mean, std, seconds)

        print(f"       (this circuit took {time.perf_counter() - start:.0f} s in total)")