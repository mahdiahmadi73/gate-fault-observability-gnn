from gfo.bench_parser import parse_bench
from gfo.features import gate_features

C17 = "data/bench/c17.bench"


def test_c17_features():
    f = gate_features(parse_bench(C17))

    assert f["10"]["fan_out"] == 1
    assert f["11"]["fan_out"] == 2
    assert f["22"]["fan_out"] == 0

    assert f["10"]["level"] == 1
    assert f["16"]["level"] == 2
    assert f["23"]["level"] == 3

    assert f["22"]["dist_to_output"] == 0
    assert f["16"]["dist_to_output"] == 1
    assert f["11"]["dist_to_output"] == 2

    assert f["11"]["outputs_reached"] == 2
    assert f["16"]["outputs_reached"] == 2
    assert f["19"]["outputs_reached"] == 1

    assert f["22"]["is_output"] == 1
    assert f["10"]["is_output"] == 0