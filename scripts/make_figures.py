# Make the figures for the README and the slides.
# Run from the repo root:  python scripts/make_figures.py
# Needs features/, labels/ and results/<circuit>_predictions.csv
# (made by: python -m gfo.inspect_predictions c2670 / c6288).

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE = "#2F62D8"     # GraphSAGE
ORANGE = "#D9822B"   # feature-only models / true labels
INK = "#16213A"
MUTED = "#6B7280"
GRID = "#DAD7CE"
BACKGROUND = "#F6F5F1"
TEAL = "#1E9C86"     # MLP

CIRCUITS = ["c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]

# Spearman on interior gates, leave-one-circuit-out (gnn_results.txt and the
# outputs of gfo.baselines and gfo.mlp). Best feature-only = max(GB, MLP).
SPEARMAN_GB = {"c432": 0.506, "c499": 0.712, "c880": 0.248, "c1355": 0.194, "c1908": 0.294,
               "c2670": 0.391, "c3540": 0.489, "c5315": 0.391, "c6288": 0.408, "c7552": 0.328}
SPEARMAN_MLP = {"c432": 0.393, "c499": 0.377, "c880": 0.313, "c1355": 0.407, "c1908": 0.210,
                "c2670": 0.437, "c3540": 0.592, "c5315": 0.497, "c6288": 0.048, "c7552": 0.296}
SPEARMAN_GNN = {"c432": 0.032, "c499": 0.444, "c880": 0.517, "c1355": 0.608, "c1908": 0.432,
                "c2670": 0.637, "c3540": 0.604, "c5315": 0.531, "c6288": 0.083, "c7552": 0.075}

SPEARMAN_HEURISTIC = {"c432": 0.279, "c499": -0.116, "c880": 0.111, "c1355": -0.456, "c1908": -0.083,
                      "c2670": 0.497, "c3540": 0.159, "c5315": 0.191, "c6288": 0.062, "c7552": 0.246}
MAE_MEAN = {"c432": 0.294, "c499": 0.352, "c880": 0.235, "c1355": 0.310, "c1908": 0.282,
            "c2670": 0.272, "c3540": 0.320, "c5315": 0.318, "c6288": 0.509, "c7552": 0.273}
MAE_GB = {"c432": 0.229, "c499": 0.200, "c880": 0.260, "c1355": 0.371, "c1908": 0.278,
          "c2670": 0.211, "c3540": 0.235, "c5315": 0.267, "c6288": 0.678, "c7552": 0.224}
MAE_MLP = {"c432": 0.240, "c499": 0.234, "c880": 0.287, "c1355": 0.307, "c1908": 0.291,
           "c2670": 0.207, "c3540": 0.221, "c5315": 0.238, "c6288": 0.571, "c7552": 0.234}
MAE_GNN = {"c432": 0.251, "c499": 0.371, "c880": 0.188, "c1355": 0.229, "c1908": 0.261,
           "c2670": 0.156, "c3540": 0.182, "c5315": 0.234, "c6288": 0.877, "c7552": 0.291}
TOP10_GB = {"c432": 0.40, "c499": 1.00, "c880": 0.49, "c1355": 0.00, "c1908": 0.47,
            "c2670": 0.54, "c3540": 0.20, "c5315": 0.28, "c6288": 0.61, "c7552": 0.30}
TOP10_MLP = {"c432": 0.49, "c499": 0.00, "c880": 0.50, "c1355": 0.17, "c1908": 0.38,
             "c2670": 0.57, "c3540": 0.24, "c5315": 0.34, "c6288": 0.50, "c7552": 0.36}
TOP10_GNN = {"c432": 0.47, "c499": 0.69, "c880": 0.98, "c1355": 0.63, "c1908": 0.55,
             "c2670": 0.68, "c3540": 0.42, "c5315": 0.39, "c6288": 0.50, "c7552": 0.33}

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 15,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "savefig.facecolor": BACKGROUND,
})


def read_column(path, column):
    with open(path) as f:
        return [float(row[column]) for row in csv.DictReader(f)]


def clean_axes(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_results(out):
    # One row per test circuit: best feature-only model vs GraphSAGE.
    best = {c: max(SPEARMAN_GB[c], SPEARMAN_MLP[c]) for c in CIRCUITS}
    order = sorted(CIRCUITS, key=lambda c: SPEARMAN_GNN[c] - best[c])

    fig, ax = plt.subplots(figsize=(11, 6.2))
    for y, c in enumerate(order):
        ax.plot([best[c], SPEARMAN_GNN[c]], [y, y], color="#9AA3B2", linewidth=2.5, zorder=1)
        ax.scatter(best[c], y, s=130, color=ORANGE, edgecolor=BACKGROUND, linewidth=2, zorder=2)
        ax.scatter(SPEARMAN_GNN[c], y, s=130, color=BLUE, edgecolor=BACKGROUND, linewidth=2, zorder=3)
        diff = SPEARMAN_GNN[c] - best[c]
        ax.text(0.84, y, f"{diff:+.2f}", va="center", ha="right", fontsize=15,
                color=BLUE if diff > 0 else "#A85A16", fontweight="bold")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, color=INK, fontsize=15)
    ax.set_xlim(-0.02, 0.86)
    ax.set_xlabel("Spearman correlation on the unseen test circuit (higher = better ranking)")
    clean_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=130, color=ORANGE, label="best feature-only model (gradient boosting or MLP)")
    ax.scatter([], [], s=130, color=BLUE, label="direction-aware GraphSAGE")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_true_vs_predicted(out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    for ax, name, title in [(axes[0], "c2670", "c2670: works"), (axes[1], "c6288", "c6288 (multiplier): fails")]:
        path = f"results/{name}_predictions.csv"
        true = read_column(path, "true_C")
        pred = read_column(path, "predicted_C")
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
        ax.scatter(true, pred, s=14, color=BLUE, alpha=0.35, linewidth=0, zorder=2)
        ax.set_title(title, loc="left", fontsize=17, fontweight="bold")
        ax.set_xlabel("true C (fault simulation)")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        mean_true = sum(true) / len(true)
        mean_pred = sum(pred) / len(pred)
        ax.text(0.03, 0.97, f"{len(true)} gates\nmean true {mean_true:.2f}, predicted {mean_pred:.2f}",
                transform=ax.transAxes, va="top", fontsize=13, color=MUTED)
    axes[0].set_ylabel("predicted C (GraphSAGE)")
    axes[1].text(0.03, 0.82, "dashed line = perfect prediction", transform=axes[1].transAxes,
                 fontsize=12, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_out_of_distribution(out):
    # Left: depth (level) of every gate. Right: true C of every interior gate.
    levels = {c: read_column(f"features/{c}.csv", "level") for c in CIRCUITS}
    labels = {}
    for c in CIRCUITS:
        is_output = read_column(f"features/{c}.csv", "is_output")
        with open(f"features/{c}.csv") as f:
            gates = [row["gate"] for row in csv.DictReader(f)]
        with open(f"labels/{c}.csv") as f:
            C = {row["gate"]: float(row["C"]) for row in csv.DictReader(f)}
        labels[c] = [C[g] for g, out_flag in zip(gates, is_output) if out_flag == 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, data, xlabel in [(axes[0], levels, "gate depth (levels from the inputs)"),
                             (axes[1], labels, "true C of interior gates")]:
        parts = ax.boxplot([data[c] for c in CIRCUITS], vert=False, widths=0.6, patch_artist=True,
                           showfliers=True, medianprops={"color": INK, "linewidth": 2},
                           flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED,
                                       "markeredgewidth": 0, "alpha": 0.4})
        for i, box in enumerate(parts["boxes"]):
            color = ORANGE if CIRCUITS[i] == "c6288" else "#C9CED8"
            box.set_facecolor(color)
            box.set_edgecolor(color)
        for whisker in parts["whiskers"] + parts["caps"]:
            whisker.set_color(MUTED)
        ax.set_xlabel(xlabel)
        clean_axes(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(1, len(CIRCUITS) + 1))
    axes[0].set_yticklabels(CIRCUITS, color=INK)
    axes[0].axvline(49, color=MUTED, linestyle="--", linewidth=1.2)
    axes[0].text(52, 2.0, "deepest gate in\nany other circuit: 49", fontsize=12, color=MUTED, va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def interior_labels(circuit):
    with open(f"features/{circuit}.csv") as f:
        rows = list(csv.DictReader(f))
    with open(f"labels/{circuit}.csv") as f:
        C = {row["gate"]: float(row["C"]) for row in csv.DictReader(f)}
    return [C[row["gate"]] for row in rows if row["is_output"] == "0"]


def chance_top10(circuit):
    # What a random top-10% list would score with the tie-aware rule:
    # the share of interior gates whose C is at least the k-th highest C.
    y = sorted(interior_labels(circuit), reverse=True)
    k = max(1, len(y) // 10)
    threshold = y[k - 1]
    return sum(v >= threshold for v in y) / len(y)


def plot_benchmark(out):
    # Top: Spearman per model and circuit. Bottom: how much each model beats the mean predictor in MAE.
    cmap = LinearSegmentedColormap.from_list("div", ["#C46A1C", "#EDEBE6", "#2F62D8"])
    columns = CIRCUITS + ["avg"]

    def with_avg(d):
        values = [d[c] for c in CIRCUITS]
        return values + [sum(values) / len(values)]

    spearman = [("heuristic", with_avg(SPEARMAN_HEURISTIC)), ("gradient boosting", with_avg(SPEARMAN_GB)),
                ("MLP", with_avg(SPEARMAN_MLP)), ("GraphSAGE", with_avg(SPEARMAN_GNN))]
    gain = []
    for name, d in [("gradient boosting", MAE_GB), ("MLP", MAE_MLP), ("GraphSAGE", MAE_GNN)]:
        gain.append((name, with_avg({c: MAE_MEAN[c] - d[c] for c in CIRCUITS})))

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.4), gridspec_kw={"height_ratios": [4, 3]})
    panels = [(axes[0], spearman, TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.75),
               "Spearman (ranking): blue = good, orange = backwards"),
              (axes[1], gain, TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.15),
               "MAE gain over the mean predictor: blue = better than the mean, orange = worse")]
    for ax, rows, norm, title in panels:
        data = [values for name, values in rows]
        ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        for i, values in enumerate(data):
            for j, v in enumerate(values):
                weight = "bold" if (rows[i][0] == "GraphSAGE" or j == len(columns) - 1) else "normal"
                ax.text(j, i, f"{v:+.2f}" if ax is axes[1] else f"{v:.2f}", ha="center", va="center",
                        fontsize=13, color=INK, fontweight=weight)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([name for name, values in rows], color=INK)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, color=INK)
        ax.tick_params(length=0)
        ax.axvline(len(CIRCUITS) - 0.5, color=BACKGROUND, linewidth=6)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, loc="left", fontsize=15, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_topk_vs_chance(out):
    # Top-10% recall per circuit, with the score a random list would get.
    fig, ax = plt.subplots(figsize=(13, 5.6))
    width = 0.26
    for i, c in enumerate(CIRCUITS):
        ax.bar(i - width, TOP10_GB[c], width, color=ORANGE, edgecolor=BACKGROUND, linewidth=1.5)
        ax.bar(i, TOP10_MLP[c], width, color=TEAL, edgecolor=BACKGROUND, linewidth=1.5)
        ax.bar(i + width, TOP10_GNN[c], width, color=BLUE, edgecolor=BACKGROUND, linewidth=1.5)
        chance = chance_top10(c)
        ax.plot([i - 1.6 * width, i + 1.6 * width], [chance, chance], color=INK, linewidth=2.5)
    ax.annotate("random list: 0.50\n(half the gates have C = 1)", xy=(8 + 1.6 * width, 0.504),
                xytext=(8.55, 0.80), fontsize=12, color=INK, ha="center",
                arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xticks(range(len(CIRCUITS)))
    ax.set_xticklabels(CIRCUITS, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("top-10% recall")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    handles = [Patch(color=ORANGE, label="gradient boosting"), Patch(color=TEAL, label="MLP"),
               Patch(color=BLUE, label="GraphSAGE"),
               Line2D([], [], color=INK, linewidth=2.5, label="score of a random list (chance)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    plot_results("figures/results_spearman.png")
    plot_true_vs_predicted("figures/true_vs_predicted.png")
    plot_out_of_distribution("figures/c6288_out_of_distribution.png")
    plot_benchmark("figures/benchmark_heatmap.png")
    plot_topk_vs_chance("figures/topk_vs_chance.png")
    print("saved 5 figures in figures/")
