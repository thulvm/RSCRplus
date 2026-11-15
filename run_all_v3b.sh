#!/bin/bash
# Ban 3b: chay lai toan bo voi bo phieu cat ngan va hai bien the tach tien nghiem; bo qua thuc the da co ket qua
cd $(cd "$(dirname "$0")/.." && pwd)/code; O=$(cd "$(dirname "$0")/.." && pwd)/out
for e in $(cat entities.txt); do [ -s $O/smd/$e.json ] || echo $e; done | xargs -P 2 -I{} python3 run_v3.py {} 15 >> $O/v3b_log.txt 2>&1
for e in $(cat psm_entities.txt); do [ -s $O/psm/$e.json ] || echo $e; done |
  RSCR_ROOT=$(cd "$(dirname "$0")/.." && pwd)/data/psm RSCR_OUT=$O/psm xargs -P 2 -I{} python3 run_v3.py {} 15 >> $O/v3b_log.txt 2>&1
echo MAIN_DONE >> $O/v3b_log.txt
