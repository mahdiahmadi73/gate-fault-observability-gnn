# gate-fault-observability-gnn

How often does a fault at one gate reach the outputs of a circuit?
This repo computes that for ISCAS'85 circuits and tries to predict it from the circuit structure,
with feature-based models and a graph neural network.

For a gate g, the label is C(g) = P(F(x) != F_g(x)): the share of input patterns for which
inverting the output of g changes at least one primary output.

Fault model: logical masking only. No stuck-at, no SEU, no timing or electrical effects.

## Data

10 ISCAS'85 circuits (c432, c499, c880, c1355, c1908, c2670, c3540, c5315, c6288, c7552),
13,268 gates. c17 is only used to check the code.

- Parser for `.bench` files, bit-parallel simulator (one Python int per wire).
- The simulator is checked against a separate hand-written model on all 32 inputs of c17,
  and against a one-pattern-at-a-time simulator on random c6288 gates.
- Labels use 10,000 random input patterns (seed 0); c17 uses all 32 patterns.
- 0 out of N observed means C < about 3/N (95%), not C = 0. 257 gates are in this case.

Features per gate: fan-in, fan-out, level, distance to output, number of outputs reached,
drives an output, gate type.

## Models

- mean of the training labels
- heuristic: outputs reached / (1 + distance to output)
- gradient boosting (scikit-learn defaults)
- MLP (PyTorch)
- direction-aware GraphSAGE (PyTorch Geometric): every layer has one `SAGEConv` for messages
  from the driving gates and one for messages from the reading gates, with separate weights

Evaluation: leave-one-circuit-out. Train on 9 circuits, test on the 10th.
Only interior gates are scored (gates that drive an output always have C = 1).
MLP and GraphSAGE settings are chosen by validation on the training circuits, never on the test circuit.
Neural models: mean over 3 seeds.
c499 and c1355 compute the same function, so they are also tested without each other in training.

## Results

MAE (lower is better) and Spearman rank correlation (higher is better), interior gates.

| Test circuit | Mean MAE | GB MAE | MLP MAE | GraphSAGE MAE | GB Spearman | MLP Spearman | GraphSAGE Spearman |
|---|---|---|---|---|---|---|---|
| c432  | 0.294 | 0.229 | 0.240 | 0.251 | 0.506 | 0.393 | 0.032 |
| c499  | 0.352 | 0.200 | 0.234 | 0.371 | 0.712 | 0.377 | 0.444 |
| c880  | 0.235 | 0.260 | 0.287 | 0.188 | 0.248 | 0.313 | 0.517 |
| c1355 | 0.310 | 0.371 | 0.307 | 0.229 | 0.194 | 0.407 | 0.608 |
| c1908 | 0.282 | 0.278 | 0.291 | 0.261 | 0.294 | 0.210 | 0.432 |
| c2670 | 0.272 | 0.211 | 0.207 | 0.156 | 0.391 | 0.437 | 0.637 |
| c3540 | 0.320 | 0.235 | 0.221 | 0.182 | 0.489 | 0.592 | 0.604 |
| c5315 | 0.318 | 0.267 | 0.238 | 0.234 | 0.391 | 0.497 | 0.531 |
| c6288 | 0.509 | 0.678 | 0.571 | 0.877 | 0.408 | 0.048 | 0.083 |
| c7552 | 0.273 | 0.224 | 0.234 | 0.291 | 0.328 | 0.296 | 0.075 |
| average | 0.317 | 0.295 | 0.283 | 0.304 | 0.396 | 0.357 | 0.396 |
| average without c6288 | 0.295 | 0.253 | 0.251 | 0.240 | 0.395 | 0.391 | 0.431 |

GB = gradient boosting. Full output with standard deviations and top-10% recall: `gnn_results.txt`
and the output of `gfo.baselines` and `gfo.mlp`.

What I see:

- GraphSAGE is the best model on 6 of 10 unseen circuits (c880, c1355, c1908, c2670, c3540, c5315).
  On c880 it is the only model that beats the mean predictor.
- It fails on c432 and c7552, and badly on c6288. On average it is not better than gradient boosting.
- c6288 (multiplier) is far outside the training data: about 3x deeper than any training circuit
  and 88% NOR gates. GraphSAGE predicts almost all its gates as safe (mean 0.008) while the true
  mean is 0.894. Labels and graph were checked and are correct; the model output is saturated.
  More on this in `gfo/inspect_predictions.py`.

## Time

Labelling all gates by fault simulation (10,000 patterns) vs one GraphSAGE prediction.

| Circuit | Gates | Labelling | GraphSAGE prediction |
|---|---|---|---|
| c432  | 160  | 0.03 s  | 0.7 ms |
| c880  | 383  | 0.15 s  | 1.5 ms |
| c1908 | 880  | 0.83 s  | 1.8 ms |
| c3540 | 1669 | 2.48 s  | 2.3 ms |
| c6288 | 2416 | 5.74 s  | 2.9 ms |
| c7552 | 3512 | 12.12 s | 3.2 ms |

Prediction is much faster, but training plus validation takes about 2 minutes per test circuit,
so on circuits this small there is no real speed-up. The argument for the model is generalization, not speed.

## Limits

- Logical masking only; no stuck-at, SEU or timing.
- Labels from one pattern seed.
- Only 10 circuits; results depend a lot on which circuit is tested.
- Validation picked the largest GraphSAGE setting offered (32 hidden, 3 layers) for 9 of 10 circuits.
- The model does not generalize to a new kind of circuit (c6288).

## Run

    pip install -r requirements.txt
    ./scripts/make_data.sh          # labels and features for all circuits
    pytest
    python -m gfo.baselines         # mean, heuristic, gradient boosting
    python -m gfo.mlp
    python -m gfo.train_gnn         # about 30 minutes on a laptop

## Files

    gfo/bench_parser.py        .bench parser
    gfo/simulator.py           topological order, bit-parallel simulation
    gfo/fault_calculator.py    fault injection, labels C(g)
    gfo/features.py            structural features
    gfo/graph.py               edge_index from the netlist
    gfo/baselines.py           mean, heuristic, gradient boosting, leave-one-circuit-out
    gfo/metrics.py             MAE, Spearman, top-k recall
    gfo/mlp.py                 MLP with validation
    gfo/gnn.py                 direction-aware GraphSAGE
    gfo/train_gnn.py           GraphSAGE training and evaluation
    gfo/inspect_predictions.py true vs predicted labels for one circuit
