#!/bin/bash
# Parse every .bench file and say OK or FAIL. Run from the repo root.
for f in data/bench/*.bench; do
  if python -m gfo.bench_parser "$f" > /dev/null; then
    echo "OK    $f"
  else
    echo "FAIL  $f"
  fi
done