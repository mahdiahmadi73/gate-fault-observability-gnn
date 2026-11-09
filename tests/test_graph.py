import csv
from pathlib import Path

from gfo.bench_parser import parse_bench
from gfo.graph import build_edges, gate_order


def test_c17_edges():
    c = parse_bench("data/bench/c17.bench")
    names = gate_order("c17")
    assert names == ["10", "11", "16", "19", "22", "23"]

    edges = build_edges(c, names)
    assert edges.tolist() == [
        [1, 1, 0, 2, 2, 3],   # from: 11, 11, 10, 16, 16, 19
        [2, 3, 4, 4, 5, 5],   # to:   16, 19, 22, 22, 23, 23
    ]


def test_graph_matches_features_for_every_circuit():
    for path in sorted(Path("data/bench").glob("*.bench")):
        c = parse_bench(path)
        names = gate_order(c.name)
        assert sorted(names) == sorted(c.gates)   # every gate exactly once

        with open(f"features/{c.name}.csv") as f:
            rows = list(csv.DictReader(f))
        with open(f"labels/{c.name}.csv") as f:
            label_gates = [row["gate"] for row in csv.DictReader(f)]
        assert sorted(label_gates) == sorted(names)

        src, dst = build_edges(c, names).tolist()
        out_degree = [0] * len(names)
        for s, t in zip(src, dst):
            assert names[s] in c.gates[names[t]].inputs   # every edge is a real wire
            out_degree[s] += 1

        for i, row in enumerate(rows):
            assert row["gate"] == names[i]                # node i = row i of the features file
            assert out_degree[i] == int(row["fan_out"])   # graph and features agree