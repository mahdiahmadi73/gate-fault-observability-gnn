#!/bin/bash
for c in c432 c499 c880 c1355; do python -m gfo.bench_parser data/bench/$c.bench | head -3; done