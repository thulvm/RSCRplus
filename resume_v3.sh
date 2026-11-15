#!/bin/bash
# Chay tiep phan con thieu sau khi container khoi dong lai: PSM chinh roi BetaRCE ma goc
cd $(cd "$(dirname "$0")/.." && pwd)/code; O=$(cd "$(dirname "$0")/.." && pwd)/out
for e in $(cat psm_entities.txt); do [ -s $O/psm/$e.json ] || echo $e; done |
  RSCR_ROOT=$(cd "$(dirname "$0")/.." && pwd)/data/psm RSCR_OUT=$O/psm xargs -P 2 -I{} python3 run_v3.py {} 15 >> $O/v3_psm_log.txt 2>&1
echo MAIN_DONE >> $O/v3_psm_log.txt
bash run_official_all.sh
