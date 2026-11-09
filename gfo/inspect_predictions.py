import sys
from pathlib import Path

import numpy as np

from gfo.baselines import CIRCUITS
from gfo.features import FEATURE_NAMES
from gfo.graph import gate_order
from gfo.train_gnn import fit_and_predict, load_graph

if __name__ == "__main__":
    test = sys.argv[1] if len(sys.argv) > 1 else "c6288"
    hidden, n_layers, epochs = 32, 3, 200   # the config validation chose for c6288 in the full run
    data = {name: load_graph(name) for name in CIRCUITS}
    train = [name for name in CIRCUITS if name != test]
    y, pred, _ = fit_and_predict(data, train, test, hidden, n_layers, epochs, seed=0)

    print(f"{test}: {len(y)} interior gates, GraphSAGE {(hidden, n_layers, epochs)}, seed 0")
    print(f"mean true C {y.mean():.3f}, mean predicted {pred.mean():.3f}")

    p = np.clip(pred, 1e-7, 1 - 1e-7)   # avoid log(0)
    raw = np.log(p / (1 - p))           # the value before the sigmoid for the predicted C. (i needed to inspect which values go into the sigmoid to get the predicted C, so I added this line to the code.)
    print("value before sigmoid, 5% / median / 95%:", np.percentile(raw, [5, 50, 95]).round(1))

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    true_counts, _ = np.histogram(y, bins=bins)
    pred_counts, _ = np.histogram(pred, bins=bins)
    print("range      true  predicted")
    for i in range(len(bins) - 1):
        print(f"{bins[i]:.1f}-{bins[i + 1]:.1f}  {true_counts[i]:6} {pred_counts[i]:9}")

    high = y >= 0.9
    low = y <= 0.1
    print(f"true C >= 0.9: {high.sum()} gates; predicted < 0.1: {np.sum(pred[high] < 0.1)}, < 0.5: {np.sum(pred[high] < 0.5)}")
    print(f"true C <= 0.1: {low.sum()} gates; predicted > 0.9: {np.sum(pred[low] > 0.9)}, > 0.5: {np.sum(pred[low] > 0.5)}")

    # Save every interior gate for a closer look later.
    X, _, is_output, _ = data[test]
    names = np.array(gate_order(test))[~is_output]
    level = X[~is_output, FEATURE_NAMES.index("level")]
    dist = X[~is_output, FEATURE_NAMES.index("dist_to_output")]
    Path("results").mkdir(exist_ok=True)
    with open(f"results/{test}_predictions.csv", "w") as out:
        out.write("gate,true_C,predicted_C,level,dist_to_output\n")
        for i in range(len(y)):
            out.write(f"{names[i]},{y[i]},{pred[i]:.4f},{int(level[i])},{int(dist[i])}\n")
    print(f"saved results/{test}_predictions.csv")