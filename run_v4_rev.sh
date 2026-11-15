#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
for f in $(ls -r ../data/smd/train/*.txt); do n=$(basename $f .txt); [ -f ../out/smd_v4/$n.json ] || [ -f /tmp/claude-0/lock_$n ] || { touch /tmp/claude-0/lock_$n; python3 run_v4.py $n 15 2>/dev/null; }; done
for f in $(ls -r ../data/psm/train/*.txt); do n=$(basename $f .txt); [ -f ../out/psm_v4/$n.json ] || [ -f /tmp/claude-0/lock_$n ] || { touch /tmp/claude-0/lock_$n; RSCR_ROOT=$(cd "$(dirname "$0")/.." && pwd)/data/psm RSCR_OUT=$(cd "$(dirname "$0")/.." && pwd)/out/psm_v4 python3 run_v4.py $n 15 2>/dev/null; }; done
echo REVDONE
