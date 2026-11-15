#!/bin/bash
# Hang doi thuc nghiem sua theo phan bien: SMD, ASD, PSM, hai tien trinh song song, bo qua thuc the da xong
cd $(cd "$(dirname "$0")/.." && pwd)/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
D=$(cd "$(dirname "$0")/.." && pwd)/data; O=$(cd "$(dirname "$0")/.." && pwd)/out
for ds in smd asd psm; do
  case $ds in smd) L=entities.txt;; asd) L=../asd_entities.txt;; psm) L=psm_entities.txt;; esac
  for e in $(cat $L); do [ -s $O/${ds}_v6/$e.json ] || echo $e; done | \
    xargs -P 2 -I{} sh -c "RSCR_ROOT=$D/$ds RSCR_OUT=$O/${ds}_v6 python3 run_v6.py {} 15 >> $O/v6_$ds.log 2>&1"
done
echo V6_DONE >> $O/v6_smd.log
