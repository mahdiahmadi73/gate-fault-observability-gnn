# gate-fault-observability-gnn

How often does a fault at one gate reach the outputs of a circuit?
This repo computes that for ISCAS'85 circuits and tries to predict it with graph neural networks.

For a gate g, the label is C(g) = P(F(x) != F_g(x)), where F_g is the circuit with
the output of g inverted. Labels come from my own bit-parallel simulator. Models are
tested on circuits they did not see in training.

Fault model: logical masking only. No stuck-at, no SEU, no timing.

Work in progress.

## Tests

    pip install -r requirements.txt
    pytest
