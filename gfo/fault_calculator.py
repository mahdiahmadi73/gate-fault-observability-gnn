import sys

from gfo.bench_parser import parse_bench
from gfo.simulator import all_input_patterns, simulate


def count_ones(value):
    return bin(value).count("1")


def observability(circuit, input_values, n_patterns):
    good = simulate(circuit, input_values, n_patterns)
    labels = {}

    for g in circuit.gates:
        bad = simulate(circuit, input_values, n_patterns, fault_gate=g)

        # 1 in a pattern's bit if at least one output is different
        changed = 0
        for out in circuit.outputs:
            changed = changed | (good[out] ^ bad[out])

        labels[g] = count_ones(changed) / n_patterns

    return labels


if __name__ == "__main__":
    c = parse_bench(sys.argv[1])
    inputs, n = all_input_patterns(c)
    labels = observability(c, inputs, n)
    for g, value in labels.items():
        print(f"gate {g:>4}: C = {value:.4f}")  