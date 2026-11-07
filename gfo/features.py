import sys
from pathlib import Path

from gfo.bench_parser import parse_bench
from gfo.simulator import topological_order

FEATURE_NAMES = ["fan_in", "fan_out", "level", "dist_to_output", "outputs_reached", "is_output"]


def find_readers(circuit):
    # For each gate: the gates that use its output wire as an input.
    readers = {name: [] for name in circuit.gates}
    for gate in circuit.gates.values():
        for wire in gate.inputs:
            if wire in readers:
                readers[wire].append(gate.name)
    return readers


def gate_features(circuit):
    order = topological_order(circuit)
    readers = find_readers(circuit)
    outputs = set(circuit.outputs)

    # Forward pass: level = longest path from a primary input.
    level = {wire: 0 for wire in circuit.inputs}
    for name in order:
        gate = circuit.gates[name]
        level[name] = 1 + max(level[wire] for wire in gate.inputs)

    # Backward pass: distance to the nearest output, and which outputs can be reached.
    dist = {}
    reached = {}
    for name in reversed(order):
        distances = []
        outs = set()
        if name in outputs:
            distances.append(0)
            outs.add(name)
        for r in readers[name]:
            distances.append(1 + dist[r])
            outs = outs | reached[r]
        if not distances:
            raise ValueError(f"Gate {name} does not lead to any output")
        dist[name] = min(distances)
        reached[name] = outs

    features = {}
    for name in order:
        features[name] = {
            "type": circuit.gates[name].type,
            "fan_in": len(circuit.gates[name].inputs),
            "fan_out": len(readers[name]),
            "level": level[name],
            "dist_to_output": dist[name],
            "outputs_reached": len(reached[name]),
            "is_output": 1 if name in outputs else 0,
        }
    return features


if __name__ == "__main__":
    c = parse_bench(sys.argv[1])
    features = gate_features(c)

    Path("features").mkdir(exist_ok=True)
    columns = ["gate", "type"] + FEATURE_NAMES
    with open(f"features/{c.name}.csv", "w") as out:
        out.write(",".join(columns) + "\n")
        for name, f in features.items():
            row = [name, f["type"]] + [str(f[k]) for k in FEATURE_NAMES]
            out.write(",".join(row) + "\n")

    print(f"{c.name}: {len(features)} gates, max level {max(f['level'] for f in features.values())}")
    print(f"saved features/{c.name}.csv")