import csv
import sys

import torch

from gfo.bench_parser import parse_bench


def gate_order(name):
    # The node order must match the rows of the features file.
    with open(f"features/{name}.csv") as f:
        return [row["gate"] for row in csv.DictReader(f)]


def build_edges(circuit, gate_names):
    index = {name: i for i, name in enumerate(gate_names)}
    sources = []
    targets = []
    for gate in circuit.gates.values():
        for wire in gate.inputs:
            if wire in index:   # primary inputs are not nodes, so skip them
                sources.append(index[wire])
                targets.append(index[gate.name])
    return torch.tensor([sources, targets], dtype=torch.long)


if __name__ == "__main__":
    c = parse_bench(sys.argv[1])
    names = gate_order(c.name)
    edges = build_edges(c, names)
    print(f"{c.name}: {len(names)} nodes, {edges.shape[1]} edges")
    if len(names) <= 10:
        print(edges)