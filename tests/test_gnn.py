import torch

from gfo.bench_parser import parse_bench
from gfo.gnn import DirGraphSAGE
from gfo.graph import build_edges, gate_order


def test_output_shape_and_range():
    c = parse_bench("data/bench/c17.bench")
    edges = build_edges(c, gate_order("c17"))
    x = torch.rand(6, 4)
    model = DirGraphSAGE(n_features=4, hidden=8, n_layers=2)
    with torch.no_grad():
        out = model(x, edges)
    assert out.shape == (6,)
    assert ((out > 0) & (out < 1)).all()


def test_side_input_reaches_gate_10_only_with_two_layers():
    # c17 node order: 10, 11, 16, 19, 22, 23 -> gate 10 = node 0, gate 16 = node 2
    c = parse_bench("data/bench/c17.bench")
    edges = build_edges(c, gate_order("c17"))
    torch.manual_seed(0)
    x = torch.rand(6, 4)
    x_changed = x.clone()
    x_changed[2] = x_changed[2] + 5.0   # change only gate 16's features

    for n_layers, should_change in [(1, False), (2, True)]:
        torch.manual_seed(0)
        model = DirGraphSAGE(n_features=4, hidden=8, n_layers=n_layers)
        with torch.no_grad():
            before = model(x, edges)[0]
            after = model(x_changed, edges)[0]
        changed = not torch.allclose(before, after)
        assert changed == should_change, f"{n_layers} layer(s)"