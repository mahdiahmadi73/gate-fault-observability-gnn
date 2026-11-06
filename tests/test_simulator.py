from gfo.bench_parser import parse_bench
from gfo.simulator import (
    all_input_patterns,
    eval_gate,
    get_bit,
    simulate,
    topological_order,
)

C17 = "data/bench/c17.bench"


def test_gate_truth_tables():
    # 4 patterns at once, pattern 0 is the rightmost bit
    a = 0b1100
    b = 0b1010
    mask = 0b1111
    assert eval_gate("AND", [a, b], mask) == 0b1000
    assert eval_gate("NAND", [a, b], mask) == 0b0111
    assert eval_gate("OR", [a, b], mask) == 0b1110
    assert eval_gate("NOR", [a, b], mask) == 0b0001
    assert eval_gate("XOR", [a, b], mask) == 0b0110
    assert eval_gate("XNOR", [a, b], mask) == 0b1001
    assert eval_gate("NOT", [a], mask) == 0b0011
    assert eval_gate("BUFF", [a], mask) == 0b1100


def test_topological_order_c17():
    c = parse_bench(C17)
    order = topological_order(c)
    assert sorted(order) == sorted(c.gates)
    for name in order:
        for wire in c.gates[name].inputs:
            if wire in c.gates:
                assert order.index(wire) < order.index(name)


def nand(a, b):
    return 1 - (a & b)


def c17_by_hand(x1, x2, x3, x6, x7):
    # Written straight from the .bench file, one pattern at a time.
    g10 = nand(x1, x3)
    g11 = nand(x3, x6)
    g16 = nand(x2, g11)
    g19 = nand(g11, x7)
    g22 = nand(g10, g16)
    g23 = nand(g16, g19)
    return g22, g23


def test_c17_all_32_patterns():
    c = parse_bench(C17)
    inputs, n = all_input_patterns(c)
    assert n == 32
    values = simulate(c, inputs, n)

    for p in range(n):
        x = [get_bit(inputs[w], p) for w in c.inputs]
        expected = c17_by_hand(*x)
        got = (get_bit(values["22"], p), get_bit(values["23"], p))
        assert got == expected, f"pattern {p}, inputs {x}"