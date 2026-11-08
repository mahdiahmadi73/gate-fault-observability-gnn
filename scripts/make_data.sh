#!/bin/bash
# Build labels and features for every circuit in data/bench. Run from the repo root.
set -e   # stop at the first error

for f in data/bench/*.bench; do
  python -m gfo.fault_calculator "$f"
  python -m gfo.features "$f"
  echo
done