import sys
import time
from pathlib import Path

from gfo.bench_parser import parse_bench
from gfo.simulator import all_input_patterns, random_input_patterns, simulate


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

    if len(c.inputs) <= 16:
        inputs, n = all_input_patterns(c)
        print(f"{c.name}: all {n} input patterns (exact labels)")
    else:
        n = 10000 # number of random input patterns to use for labelling. in future, we can make this a command line argument. also tune
        inputs = random_input_patterns(c, n, seed=0)
        print(f"{c.name}: {n} random input patterns, seed 0")

    start = time.perf_counter()
    labels = observability(c, inputs, n)
    seconds = time.perf_counter() - start

    never_seen = [g for g, value in labels.items() if value == 0]
    mean = sum(labels.values()) / len(labels)
    print(f"gates: {len(labels)}, labelling time: {seconds:.2f} s")
    print(f"mean C: {mean:.4f}")
    print(f"gates never observed: {len(never_seen)}")
    if never_seen and len(c.inputs) > 16:
        print(f"  their true C is below about 3/N = {3 / n:.4f} (95% confidence)")

    Path("labels").mkdir(exist_ok=True)
    with open(f"labels/{c.name}.csv", "w") as out:
        out.write("gate,C\n")
        for g, value in labels.items():
            out.write(f"{g},{value}\n")
    print(f"saved labels/{c.name}.csv")