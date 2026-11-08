import numpy as np
import torch
from torch import nn

from gfo.baselines import CIRCUITS, TWINS, load_circuit
from gfo.metrics import mae, spearman, top_k_recall

SEEDS = [0, 1, 2]


class MLP(nn.Module):
    def __init__(self, n_features, hidden=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),   # output between 0 and 1, like C
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def standardize(X_train, X_test):
    # Scale every feature with the TRAINING mean and std (never look at the test circuit).
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0   # a column that never changes, e.g. a gate type not in training
    return (X_train - mean) / std, (X_test - mean) / std


def train_mlp(X_train, y_train, seed, epochs=50, lr=0.01):
    torch.manual_seed(seed)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    model = MLP(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

    return model


def predict(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def run(data, test, train_names):
    X_train = np.concatenate([data[name][0] for name in train_names])
    y_train = np.concatenate([data[name][1] for name in train_names])
    X_test, y_test, is_output = data[test]
    X_train, X_test = standardize(X_train, X_test)
    inside = ~is_output
    k = max(1, inside.sum() // 10)

    results = []
    for seed in SEEDS:
        model = train_mlp(X_train, y_train, seed)

        train_pred = predict(model, X_train)
        test_pred = predict(model, X_test)

        train_inside = np.concatenate(
            [~data[name][2] for name in train_names]
        )

        print("Training MAE:",
            mae(y_train[train_inside], train_pred[train_inside]))

        print("Test MAE:",
            mae(y_test[inside], test_pred[inside]))
        pred = test_pred[inside]
        y = y_test[inside]
        results.append([mae(y, pred), spearman(y, pred), top_k_recall(y, pred, k)])

    results = np.array(results)
    

    
    return results.mean(axis=0), results.std(axis=0)


def print_row(test, model_name, mean, std):
    print(f"{test:6} {model_name:11} "
          f"{mean[0]:.3f}±{std[0]:.3f}  {mean[1]:.3f}±{std[1]:.3f}  {mean[2]:.2f}±{std[2]:.2f}")


if __name__ == "__main__":
    data = {name: load_circuit(name) for name in CIRCUITS}

    print(f"MLP, leave-one-circuit-out, interior gates only, seeds {SEEDS}")
    print("test   model       MAE          Spearman      top10%")

    for test in CIRCUITS:
        train = [name for name in CIRCUITS if name != test]
        mean, std = run(data, test, train)
        print_row(test, "mlp", mean, std)

        if test in TWINS:
            train = [name for name in train if name != TWINS[test]]
            mean, std = run(data, test, train)
            print_row(test, "mlp-no-twin", mean, std)