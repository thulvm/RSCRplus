"""Tao 16 khoi PSM tu ba tep CSV goc (train.csv, test.csv, test_label.csv) theo dung dinh dang SMD.
Cach dung: python3 prep_psm.py <thu_muc_psm_goc> <thu_muc_ra>
- Bo cot timestamp, dien gia tri thieu cua train bang ffill roi bfill (test khong co gia tri thieu).
- Moi khoi dung chung toan bo chuoi huan luyen; chuoi kiem tra chia thanh 16 doan lien nhau, doan cuoi nhan phan du.
- PSM khong co nhan kenh gay loi nen interpretation_label de trong.
"""
import os, sys
import numpy as np, pandas as pd

src, dst = sys.argv[1], sys.argv[2]
NB = 16
tr = pd.read_csv(f"{src}/train.csv").iloc[:, 1:].ffill().bfill().values
te = pd.read_csv(f"{src}/test.csv").iloc[:, 1:].values
lab = pd.read_csv(f"{src}/test_label.csv")["label"].values.astype(int)
for d in ("train", "test", "test_label", "interpretation_label"):
    os.makedirs(f"{dst}/{d}", exist_ok=True)
size = len(te) // NB
for b in range(NB):
    lo, hi = b * size, (len(te) if b == NB - 1 else (b + 1) * size)
    n = f"psm-{b + 1:02d}"
    np.savetxt(f"{dst}/train/{n}.txt", tr, delimiter=",", fmt="%g")
    np.savetxt(f"{dst}/test/{n}.txt", te[lo:hi], delimiter=",", fmt="%g")
    np.savetxt(f"{dst}/test_label/{n}.txt", lab[lo:hi], fmt="%d")
    open(f"{dst}/interpretation_label/{n}.txt", "w").close()
print("da tao", NB, "khoi PSM trong", dst)
