"""Xuat du lieu sau xu ly cho tung thuc the (may SMD hoac khoi PSM), dung dung quy trinh common.setup.
Cach dung: python3 export_processed.py <thu_muc_du_lieu_goc> <danh_sach_thuc_the.txt> <thu_muc_ra>
Moi thuc the ghi mot tep <ten>.npz gom:
  F_train      dac trung cua so huan luyen da chuan hoa (trung binh va do lech chuan tung kenh, 2K cot)
  train_start  hang bat dau cua moi cua so huan luyen
  is_fit, is_cal       cua so dung de khop / de hieu chuan (cua so bi bo vi sat ranh gioi: ca hai deu False)
  is_dep, is_later     cua so thuoc giai doan trien khai (70% dau) / giai doan sau (tu moc 30%)
  F_test, test_start, y_test   dac trung, hang bat dau va nhan cua so kiem tra (nhan = it nhat nua so diem bat thuong)
  flag_pca, flag_iforest       cua so kiem tra bi bo phat hien dang chay gan co
  pick_pca, pick_iforest       chi so cac cua so duoc chon de giai thich (toi da 15, rai deu theo thoi gian)
  reference    vecto tham chieu r (trung vi dac trung cac cua so khop giai doan trien khai)
  feat_mean, feat_std  tham so chuan hoa dac trung
Kem theo summary.csv: so hang, so cua so khop, hieu chuan, kiem tra, so cua so bat thuong va so cua so duoc chon.
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, common

root, lst, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
names = [l.strip() for l in open(lst) if l.strip()]
rows = []
for name in names:
    tr, te, lab, segs = lib.load_entity(root, name)
    N = len(tr); cut_row, later_row = int(0.7 * N), int(0.3 * N)
    mu, sd = tr[:cut_row].mean(0), tr[:cut_row].std(0); sd[sd < 1e-3] = 1e-3
    Xtr = (tr - mu) / sd
    starts, Wtr = lib.windows(Xtr)
    n = len(starts); chunk = np.arange(n) // common.CHUNK; nch = chunk.max() + 1
    is_cal_chunk = np.zeros(nch, bool)
    is_cal_chunk[np.random.RandomState(0).permutation(nch)[: max(1, nch // 5)]] = True
    is_cal = is_cal_chunk[chunk]
    keep = np.ones(n, bool)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        if np.any(is_cal[lo:hi] != is_cal[i]):
            keep[i] = False
    c = common.setup(root, name)  # cung quy trinh voi thuc nghiem, dung de lay dac trung va bo phat hien
    S = lambda F: (F - c.fm) / c.fs
    F_train = S(lib.feats(Wtr))
    rec = dict(F_train=F_train.astype(np.float32), train_start=starts.astype(np.int32),
               is_fit=keep & ~is_cal, is_cal=keep & is_cal,
               is_dep=(starts + lib.W) <= cut_row, is_later=starts >= later_row,
               F_test=c.Fte.astype(np.float32), test_start=c.tidx.astype(np.int32), y_test=c.ylab,
               reference=c.base.astype(np.float32), feat_mean=c.fm.astype(np.float32), feat_std=c.fs.astype(np.float32))
    for gk in ("pca", "iforest"):
        dep = lib.Det(gk, 0).fit(c.Fdep, c.Fcal_dep)
        rec[f"flag_{gk}"] = dep.flags(c.Fte)
        rec[f"pick_{gk}"] = np.array(common.picks(c, dep), dtype=np.int32)
    np.savez_compressed(f"{out}/{name}.npz", **rec)
    rows.append(dict(entity=name, rows_train=N, rows_test=len(te), channels=c.K, features=2 * c.K,
                     win_fit=int(rec["is_fit"].sum()), win_cal=int(rec["is_cal"].sum()), win_dropped=int((~keep).sum()),
                     win_test=len(c.Fte), win_test_anomalous=int(c.ylab.sum()),
                     picked_pca=len(rec["pick_pca"]), picked_iforest=len(rec["pick_iforest"])))
    print(name, rows[-1])
with open(f"{out}/summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
