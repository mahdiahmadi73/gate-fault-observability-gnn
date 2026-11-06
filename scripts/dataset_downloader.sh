#!/bin/bash
# a simple script to download the benchmark files from the ISCAS85 benchmark suite
cd data/bench
for c in c432 c499 c880 c1355; do
  curl -fsSO "https://pld.ttu.ee/~maksim/benchmarks/iscas85/bench/$c.bench"
done
ls -l
cd ../..