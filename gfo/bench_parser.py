import sys
from dataclasses import dataclass, field
from pathlib import Path

GATE_TYPES = ["AND", "NAND", "OR", "NOR", "XOR", "XNOR", "NOT", "BUFF"]


@dataclass
class Gate:
    name: str      # same as the wire this gate drives
    type: str      # e.g. "NAND"
    inputs: list   # names of the wires going into the gate


@dataclass
class Circuit:
    name: str
    inputs: list = field(default_factory=list)   # primary inputs
    outputs: list = field(default_factory=list)  # primary outputs
    gates: dict = field(default_factory=dict)    # gate name -> Gate


def parse_bench(path):
    circuit = Circuit(Path(path).stem)

    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()   # remove comments
            if line == "":
                continue

            if line.startswith("INPUT("):
                wire = line[len("INPUT("):-1].strip()
                circuit.inputs.append(wire)

            elif line.startswith("OUTPUT("):
                wire = line[len("OUTPUT("):-1].strip()
                circuit.outputs.append(wire)

            elif "=" in line:
                # example: 10 = NAND(1, 3)
                left, right = line.split("=")
                gate_name = left.strip()
                gate_type = right.split("(")[0].strip().upper()
                inside = right.split("(")[1].replace(")", "")
                gate_inputs = [w.strip() for w in inside.split(",")]

                if gate_type not in GATE_TYPES:
                    raise ValueError(f"Unknown gate type {gate_type} in line: {line}")

                circuit.gates[gate_name] = Gate(gate_name, gate_type, gate_inputs)

            else:
                raise ValueError(f"Cannot read line: {line}")

    check_circuit(circuit)
    return circuit


def check_circuit(circuit):
    known_wires = set(circuit.inputs) | set(circuit.gates)

    for gate in circuit.gates.values():
        for wire in gate.inputs:
            if wire not in known_wires:
                raise ValueError(f"Gate {gate.name} uses unknown wire {wire}")

    for wire in circuit.outputs:
        if wire not in known_wires:
            raise ValueError(f"Output {wire} is not driven by any gate")


if __name__ == "__main__":
    c = parse_bench(sys.argv[1])
    print(c.name)
    print("inputs: ", c.inputs)
    print("outputs:", c.outputs)
    for gate in c.gates.values():
        print(gate.name, "=", gate.type, gate.inputs)
