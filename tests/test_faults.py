from gfo.bench_parser import parse_bench
from gfo.fault_calculator import observability
from gfo.simulator import all_input_patterns, get_bit

C17 = "data/bench/c17.bench"


def nand(a, b):
    return 1 - (a & b)


def c17_by_hand(x1, x2, x3, x6, x7, flip=None):
    # Same as in test_simulator, but can flip one gate's output.
    def maybe_flip(name, value):
        if name == flip:
            return 1 - value
        return value

    g10 = maybe_flip("10", nand(x1, x3))
    g11 = maybe_flip("11", nand(x3, x6))
    g16 = maybe_flip("16", nand(x2, g11))
    g19 = maybe_flip("19", nand(g11, x7))
    g22 = maybe_flip("22", nand(g10, g16))
    g23 = maybe_flip("23", nand(g16, g19))
    return g22, g23


def test_c17_labels_match_hand_model():
    c = parse_bench(C17)
    inputs, n = all_input_patterns(c)
    labels = observability(c, inputs, n)

    for g in c.gates:
        changed = 0
        for p in range(n):
            x = [get_bit(inputs[w], p) for w in c.inputs]
            if c17_by_hand(*x) != c17_by_hand(*x, flip=g):
                changed += 1
        assert labels[g] == changed / n, f"gate {g}"


def test_output_gates_always_observed():
    c = parse_bench(C17)
    inputs, n = all_input_patterns(c)
    labels = observability(c, inputs, n)
    assert labels["22"] == 1.0
    assert labels["23"] == 1.0