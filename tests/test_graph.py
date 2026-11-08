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