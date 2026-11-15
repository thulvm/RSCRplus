"""Chuyen ASD (Li et al., KDD 2021, kho InterFusion) tu tep pkl sang dinh dang SMD (txt).
Cach dung: python3 prep_asd.py <thu_muc_asd_src> <thu_muc_ra>
ASD da duoc tac gia chuan hoa ve [0, 1]; nhan kenh trong interpretation_label danh so tu 1 nhu SMD."""
import os, sys, pickle, shutil
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
for d in ("train", "test", "test_label", "interpretation_label"):
    os.makedirs(f"{dst}/{d}", exist_ok=True)
names = []
for i in range(1, 13):
    n = f"omi-{i}"
    for part, fmt in (("train", "%.6g"), ("test", "%.6g"), ("test_label", "%d")):
        a = np.asarray(pickle.load(open(f"{src}/processed/{n}_{part}.pkl", "rb")))
        np.savetxt(f"{dst}/{part}/{n}.txt", a.astype(int) if part == "test_label" else a, delimiter=",", fmt=fmt)
    shutil.copy(f"{src}/interpretation_label/{n}.txt", f"{dst}/interpretation_label/{n}.txt")
    names.append(n)
open(f"{dst}/../asd_entities.txt", "w").write("\n".join(names) + "\n")
print("ASD:", len(names), "thuc the")
