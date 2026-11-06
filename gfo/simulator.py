import sys

from gfo.bench_parser import parse_bench


def topological_order(circuit):
    # A gate can be computed once all its input wires are known.
    known = set(circuit.inputs)
    order = []
    remaining = list(circuit.gates)

    while remaining:
        still_waiting = []
        for name in remaining:
            gate = circuit.gates[name]
            if all(wire in known for wire in gate.inputs):
                order.append(name)
                known.add(name)
            else:
                still_waiting.append(name)

        if len(still_waiting) == len(remaining):
            raise ValueError(f"Circuit has a loop, stuck at: {still_waiting[:5]}")
        remaining = still_waiting

    return order


def eval_gate(gate_type, in_values, mask):
    # Each value is an int; bit p is the wire's value for input pattern p.
    if gate_type in ("AND", "NAND"):
        out = mask
        for v in in_values:
            out = out & v
    elif gate_type in ("OR", "NOR"):
        out = 0
        for v in in_values:
            out = out | v
    elif gate_type in ("XOR", "XNOR"):
        out = 0
        for v in in_values:
            out = out ^ v
    elif gate_type in ("NOT", "BUFF"):
        out = in_values[0]
    else:
        raise ValueError(f"Unknown gate type {gate_type}")

    # The inverting gates: flip all bits.
    if gate_type in ("NAND", "NOR", "XNOR", "NOT"):
        out = out ^ mask
    return out


def simulate(circuit, input_values, n_patterns, fault_gate=None):
    mask = (1 << n_patterns) - 1   # n_patterns ones, e.g. 0b11111
    values = dict(input_values)

    for name in topological_order(circuit):
        gate = circuit.gates[name]
        in_values = [values[wire] for wire in gate.inputs]
        values[name] = eval_gate(gate.type, in_values, mask)

        if name == fault_gate:
            values[name] = values[name] ^ mask   # the fault: flip this gate's output
    return values

# def simulate(circuit, input_values, n_patterns):
#     mask = (1 << n_patterns) - 1   # n_patterns ones, e.g. 0b11111
#     values = dict(input_values)

#     for name in topological_order(circuit):
#         gate = circuit.gates[name]
#         in_values = [values[wire] for wire in gate.inputs]
#         values[name] = eval_gate(gate.type, in_values, mask)

#     return values


def all_input_patterns(circuit):
    # Pattern p sets input i to bit i of p, so we get every combination once.
    n_inputs = len(circuit.inputs)
    n_patterns = 2 ** n_inputs
    input_values = {}

    for i, wire in enumerate(circuit.inputs):
        value = 0
        for p in range(n_patterns):
            if (p >> i) & 1:
                value = value | (1 << p)
        input_values[wire] = value

    return input_values, n_patterns


def get_bit(value, p):
    return (value >> p) & 1


if __name__ == "__main__":
    c = parse_bench(sys.argv[1])
    inputs, n = all_input_patterns(c)
    values = simulate(c, inputs, n)
    print("pattern  " + " ".join(c.inputs) + "  ->  " + " ".join(c.outputs))
    for p in range(n):
        ins = " ".join(str(get_bit(values[w], p)) for w in c.inputs)
        outs = " ".join(str(get_bit(values[w], p)) for w in c.outputs)
        print(f"{p:7}  {ins}  ->  {outs}")