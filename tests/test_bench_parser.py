from pathlib import Path

import pytest

from gfo.bench_parser import parse_bench
from gfo.simulator import topological_order

C17 = "data/bench/c17.bench"


def test_c17_counts():
    c = parse_bench(C17)
    assert c.inputs == ["1", "2", "3", "6", "7"]
    assert c.outputs == ["22", "23"]
    assert len(c.gates) == 6
    assert all(g.type == "NAND" for g in c.gates.values())


def test_c17_wiring():
    c = parse_bench(C17)
    assert c.gates["10"].inputs == ["1", "3"]
    assert c.gates["22"].inputs == ["10", "16"]
    assert c.gates["23"].inputs == ["16", "19"]


def write(tmp_path, text):
    p = tmp_path / "t.bench"
    p.write_text(text)
    return p


def test_undefined_signal(tmp_path):
    p = write(tmp_path, "INPUT(a)\nOUTPUT(y)\ny = AND(a, b)\n")
    with pytest.raises(ValueError, match="unknown wire"):
        parse_bench(p)


def test_unsupported_gate(tmp_path):
    p = write(tmp_path, "INPUT(a)\nOUTPUT(q)\nq = DFF(a)\n")
    with pytest.raises(ValueError, match="Unknown gate type"):
        parse_bench(p)
def test_all_bench_files_parse():
    for path in sorted(Path("data/bench").glob("*.bench")):
        circuit = parse_bench(path)
        topological_order(circuit)   # also fails if a circuit has a loop