#!/bin/bash
# Chay BetaRCE ma goc cho moi thuc the chua co ket qua (chay lai duoc neu bi dung giua chung)
cd $(cd "$(dirname "$0")/.." && pwd)/code
O=$(cd "$(dirname "$0")/.." && pwd)/out
{ for e in $(cat entities.txt); do [ -s $O/smd_betarce/$e.json ] || echo "smd $e"; done
  for e in $(cat psm_entities.txt); do [ -s $O/psm_betarce/$e.json ] || echo "psm $e"; done; } |
xargs -P 2 -L 1 bash -c 'if [ "$0" = smd ]; then python3 run_betarce_v2.py "$1" 15; else RSCR_ROOT=$(cd "$(dirname "$0")/.." && pwd)/data/psm RSCR_OUT=$(cd "$(dirname "$0")/.." && pwd)/out/psm python3 run_betarce_v2.py "$1" 15; fi' >> $O/v3_off_log.txt 2>&1
echo OFF_DONE >> $O/v3_off_log.txt
