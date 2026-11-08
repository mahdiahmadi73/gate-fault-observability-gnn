#!/bin/bash
# a simple script to download the benchmark files from the ISCAS85 benchmark suite
cd data/bench
for c in c1908 c2670 c3540 c5315 c6288 c7552; do
  curl -fsSO "https://pld.ttu.ee/~maksim/benchmarks/iscas85/bench/$c.bench"
done
ls -l
cd ../..