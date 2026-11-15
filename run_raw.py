"""Thuc nghiem ban 5 tren bo phat hien doc cua so tho (PCA va autoencoder tren cua so 30 buoc x K kenh trai phang).
Cach dung: python3 run_raw.py <thuc_the> [so_ca]
Phep sua tren chuoi: thay ca doan 30 buoc cua mot kenh bang kenh tuong ung cua cua so binh thuong gan nhat
(trong cac cua so khop giai doan trien khai); thu tu kenh theo do lech binh phuong giua cua so va cua so dan duong.
Chia du lieu, ban sao theo khoi thoi gian, bo cap nhat va chuoi huan luyen lai giong ban 4."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, common, lib5
from sklearn.neighbors import NearestNeighbors

ROOT = os.environ.get("RSCR_ROOT", RSCR_HOME + "/data/smd")
OUT = os.environ.get("RSCR_OUT", RSCR_HOME + "/out/smd_raw")
A, G, D, B = 0.10, 0.05, 0.25, 40
FAMS = ("rawpca", "rawae")


def masks(n, starts, cut_row, later_row):
    chunk = np.arange(n) // common.CHUNK; nch = chunk.max() + 1
    is_cal_chunk = np.zeros(nch, bool)
    is_cal_chunk[np.random.RandomState(0).permutation(nch)[: max(1, nch // 5)]] = True
    is_cal = is_cal_chunk[chunk]
    keep = np.ones(n, bool)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        if np.any(is_cal[lo:hi] != is_cal[i]):
            keep[i] = False
    dep = (starts + lib.W) <= cut_row
    return keep & ~is_cal, keep & is_cal, dep, starts >= later_row


def split_cal(X, seed):
    r = np.random.RandomState(seed); idx = r.permutation(len(X)); k = max(20, len(X) // 5)
    return X[idx[k:]], X[idx[:k]]


def main(name, maxcase=10):
    t0 = time.time()
    tr, te, lab, segs = lib.load_entity(ROOT, name)
    N = len(tr); cut_row, later_row = int(0.7 * N), int(0.3 * N)
    mu, sd = tr[:cut_row].mean(0), tr[:cut_row].std(0); sd[sd < 1e-3] = 1e-3
    Xtr, Xte = (tr - mu) / sd, (te - mu) / sd
    K = Xtr.shape[1]
    st, Wtr = lib.windows(Xtr); tst, Wte = lib.windows(Xte)
    fit, cal, dep_w, later_w = masks(len(st), st, cut_row, later_row)
    ylab = np.array([lab[i:i + lib.W].mean() >= 0.5 for i in tst])
    Wdep, Wcal_dep = Wtr[fit & dep_w], Wtr[cal & dep_w]
    NT = len(Xte)
    chain = {"chain1": Wtr[st >= int(0.3 * N)],
             "chain2": np.concatenate([Wtr[st >= int(0.6 * N)], Wte[~ylab & (tst < int(0.3 * NT))]]),
             "chain3": np.concatenate([Wtr[st >= int(0.8 * N)], Wte[~ylab & (tst < int(0.6 * NT))]])}
    nn = NearestNeighbors(n_neighbors=1).fit(Wdep.reshape(len(Wdep), -1))
    records = []
    for gk in FAMS:
        dep = lib5.RawDet(gk, 0).fit(Wdep, Wcal_dep)
        L = int(0.6 * len(Wdep)); copies = []
        for b in range(1, B + 1):
            s0 = np.random.RandomState(1000 + b).randint(0, len(Wdep) - L + 1)
            copies.append(lib5.RawDet(gk, 100 + b).fit(Wdep[s0:s0 + L], Wcal_dep))
        ev = {"upd_s0": lib5.RawDet(gk, 7).fit(Wtr[fit], Wtr[cal]), "later": lib5.RawDet(gk, 11).fit(Wtr[fit & later_w], Wtr[cal & later_w])}
        if gk == "rawae":
            for s in range(3):
                ev[f"same_s{s}"] = lib5.RawDet(gk, 21 + s).fit(Wdep, Wcal_dep)
        for k, Wk in chain.items():
            a, b_ = split_cal(Wk, 5); ev[k] = lib5.RawDet(gk, 31).fit(a, b_)
        other = "rawae" if gk == "rawpca" else "rawpca"
        ev["x_" + other] = lib5.RawDet(other, 7).fit(Wtr[fit], Wtr[cal])
        fl = np.where(ylab & dep.flags(Wte) & (tst >= int(0.6 * NT)))[0]
        picks = list(fl[np.linspace(0, len(fl) - 1, min(maxcase, len(fl))).astype(int)]) if len(fl) else []
        for wi in picks:
            x = Wte[wi]
            guide = Wdep[nn.kneighbors(x.reshape(1, -1))[1][0, 0]]
            order = [int(c) for c in np.argsort(-((x - guide) ** 2).sum(0))]
            rep = lambda k: lib5.raw_repair(x, guide, order[:k])
            out = {}
            g, k, q = lib5.generic_scr(dep, rep, K); out["SCR"] = (g, k, q); k_scr = k
            g, k, q = lib5.generic_scr(dep, rep, K, D); out["m-SCR"] = (g, k, q)
            g, k, q2 = lib5.generic_fullvote(dep, copies, rep, K, start_k=k_scr or 1)
            out["BetaRCE-T"] = (g, k, max(0, (k_scr or 1) - 1) + q2)
            g, k, q = lib5.generic_rscr(dep, copies, rep, K, D, A, G, np.random.RandomState(int(wi))); out["R-SCR+"] = (g, k, q)
            flags = {e: bool(d.flags(x[None])[0]) for e, d in ev.items()}
            res = {}
            for nm, (g, k, q) in out.items():
                rr = dict(found=int(g is not None), k=k, frac=(k / K if k else None), calls=int(q))
                for e, d in ev.items():
                    rr[e] = None if not flags[e] else (int(d.score(g[None])[0] <= d.tau) if g is not None else 0)
                res[nm] = rr
            mid = int(tst[wi]) + lib.W // 2
            ch = next((s[2] for s in segs if s[0] <= mid <= s[1]), None)
            records.append(dict(entity=name, gen=gk, win=int(tst[wi]), res=res,
                                prec=None if ch is None else len(set(order[:len(ch)]) & set(ch)) / len(ch)))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(entity=name, n=len(records), secs=time.time() - t0, records=records), open(f"{OUT}/{name}.json", "w"))
    print(name, len(records), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
