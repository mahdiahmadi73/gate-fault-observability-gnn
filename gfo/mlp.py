import numpy as np
import torch
from torch import nn

from gfo.baselines import CIRCUITS, TWINS, load_circuit
from gfo.metrics import mae, spearman, top_k_recall

SEEDS = [0, 1, 2]
CONFIGS = [(16, 50), (16, 100), (64, 100), (64, 500)]   # (hidden, epochs), fixed in advance


class MLP(nn.Module):
    def __init__(self, n_features, hidden):
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


def train_mlp(X_train, y_train, seed, epochs, hidden, lr=0.01):
    torch.manual_seed(seed)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    model = MLP(X.shape[1], hidden)
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


def fit_and_predict(data, train_names, eval_name, hidden, epochs, seed):
    # Train on train_names, predict the interior gates of eval_name.
    X_train = np.concatenate([data[name][0] for name in train_names])
    y_train = np.concatenate([data[name][1] for name in train_names])
    X_eval, y_eval, is_output = data[eval_name]
    X_train, X_eval = standardize(X_train, X_eval)

    model = train_mlp(X_train, y_train, seed, epochs, hidden)
    pred = predict(model, X_eval)
    inside = ~is_output
    return y_eval[inside], pred[inside]


def choose_config(data, train_names):
    # Only training circuits are used here: each one takes a turn as validation.
    best_score = None
    best_config = None
    for hidden, epochs in CONFIGS:
        errors = []
        for val in train_names:
            inner_train = [name for name in train_names if name != val]
            y, pred = fit_and_predict(data, inner_train, val, hidden, epochs, seed=0)
            errors.append(mae(y, pred))
        score = np.mean(errors)
        if best_score is None or score < best_score:
            best_score = score
            best_config = (hidden, epochs)
    return best_config


def run(data, test, train_names):
    hidden, epochs = choose_config(data, train_names)

    results = []
    for seed in SEEDS:
        y, pred = fit_and_predict(data, train_names, test, hidden, epochs, seed)
        k = max(1, len(y) // 10)
        results.append([mae(y, pred), spearman(y, pred), top_k_recall(y, pred, k)])

    results = np.array(results)
    return (hidden, epochs), results.mean(axis=0), results.std(axis=0)


def print_row(test, model_name, config, mean, std):
    print(f"{test:6} {model_name:11} {str(config):10} "
          f"{mean[0]:.3f}±{std[0]:.3f}  {mean[1]:.3f}±{std[1]:.3f}  {mean[2]:.2f}±{std[2]:.2f}")


if __name__ == "__main__":
    data = {name: load_circuit(name) for name in CIRCUITS}

    print(f"MLP, leave-one-circuit-out, interior gates only, seeds {SEEDS}")
    print("config chosen by validation MAE on the training circuits only")
    print("test   model       config     MAE          Spearman      top10%")

    for test in CIRCUITS:
        train = [name for name in CIRCUITS if name != test]
        config, mean, std = run(data, test, train)
        print_row(test, "mlp", config, mean, std)

        if test in TWINS:
            train = [name for name in train if name != TWINS[test]]
            config, mean, std = run(data, test, train)
            print_row(test, "mlp-no-twin", config, mean, std)