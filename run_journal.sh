#!/bin/bash
# Ban tap chi: hai hang doi chay song song tren hai loi, bo qua thuc the da co ket qua.
H="${RSCR_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$H/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
O=$H/out; D=$H/data
qA() {
  for e in $(cat entities.txt); do [ -s $O/smd_v5/$e.json ] || RSCR_ROOT=$D/smd RSCR_OUT=$O/smd_v5 python3 run_v5.py $e 15; done
  for e in $(cat ../asd_entities.txt); do [ -s $O/asd_v5/$e.json ] || RSCR_ROOT=$D/asd RSCR_OUT=$O/asd_v5 python3 run_v5.py $e 15; done
  for e in $(cat psm_entities.txt); do [ -s $O/psm_v5/$e.json ] || RSCR_ROOT=$D/psm RSCR_OUT=$O/psm_v5 python3 run_v5.py $e 15; done
  echo QA_DONE
}
qB() {
  for e in $(cat ../asd_entities.txt); do [ -s $O/asd/$e.json ] || RSCR_ROOT=$D/asd RSCR_OUT=$O/asd python3 run_v3.py $e 15; done
  for e in $(cat ../asd_entities.txt); do [ -s $O/asd_betarce/$e.json ] || RSCR_ROOT=$D/asd RSCR_OUT=$O/asd python3 run_betarce_v2.py $e 15; done
  for e in $(cat ../asd_entities.txt); do [ -s $O/asd_raw/$e.json ] || RSCR_ROOT=$D/asd RSCR_OUT=$O/asd_raw python3 run_raw.py $e 10; done
  for e in $(cat entities.txt); do [ -s $O/smd_raw/$e.json ] || RSCR_ROOT=$D/smd RSCR_OUT=$O/smd_raw python3 run_raw.py $e 10; done
  echo QB_DONE
}
qA > $O/journal_qA.log 2>&1 &
qB > $O/journal_qB.log 2>&1 &
wait
