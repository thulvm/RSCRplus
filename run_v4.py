"""Thuc nghiem ban 4 (theo phan bien). Cach dung: python3 run_v4.py <thuc_the> [so_ca]
Moi so voi ban 3:
  - them bo phat hien autoencoder (ae) lam bo dang chay, ben canh pca va iforest
  - moi baseline xep kenh theo thu tu do lech (cong bang voi R-SCR+), them Wachter co bien
  - danh gia do ben tren nhieu bo cap nhat: 5 seed cap nhat toan lich su, 5 seed doi chung chi doi seed
    (cung du lieu voi bo dang chay), va chuoi ba lan huan luyen lai dung du lieu ngay cang moi,
    trong do buoc 2 va 3 dung them cua so binh thuong cua giai doan kiem tra (du lieu that su moi)."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, common

ROOT = os.environ.get("RSCR_ROOT", RSCR_HOME + "/data/smd")
OUT = os.environ.get("RSCR_OUT", RSCR_HOME + "/out/smd_v4")
A, G, D = 0.10, 0.05, 0.25
FAMS = os.environ.get("RSCR_FAMS", "pca,iforest,ae").split(",")


def wachter_margin(det, f, base, delta, iters=20):
    """Tim theo duong thang tu f ve muc chuan nhu Wachter, nhung dung o muc co bien thay vi nguong."""
    lvl, q = lib.stop_level(det, delta), 1
    if det.score(base)[0] > lvl:
        return None, q
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2; q += 1
        if det.score(base + (1 - mid) * (f - base))[0] <= lvl:
            hi = mid
        else:
            lo = mid
    return base + (1 - hi) * (f - base), q


def split_cal(F, seed):
    r = np.random.RandomState(seed); idx = r.permutation(len(F)); k = max(20, len(F) // 5)
    return F[idx[k:]], F[idx[:k]]


def main(name, maxcase=15):
    t0 = time.time()
    c = common.setup(ROOT, name)
    K = c.K
    # dac trung cua moi cua so huan luyen kem vi tri, de dung cac lat cat theo thoi gian cho chuoi huan luyen lai
    starts, Wtr = lib.windows(c.Xtr)
    Fall = (lib.feats(Wtr) - c.fm) / c.fs
    N = len(c.Xtr); NT = len(c.Xte)
    norm_te = ~c.ylab
    chain_sets = {
        "chain1": Fall[starts >= int(0.3 * N)],
        "chain2": np.vstack([Fall[starts >= int(0.6 * N)], c.Fte[norm_te & (c.tidx < int(0.3 * NT))]]),
        "chain3": np.vstack([Fall[starts >= int(0.8 * N)], c.Fte[norm_te & (c.tidx < int(0.6 * NT))]]),
    }
    test_cut = int(0.6 * NT)   # chi lay ca o phan sau cua tap kiem tra de khong trung du lieu chuoi
    records = []
    for gk in FAMS:
        dep, upd, cop_t, _ = common.detectors(c, gk)
        evals = {}
        for s in range(5):
            evals[f"upd_s{s}"] = lib.Det(gk, 7 + s).fit(c.Ffit, c.Fcal)
            evals[f"same_s{s}"] = lib.Det(gk, 21 + s).fit(c.Fdep, c.Fcal_dep)
        for k, Fk in chain_sets.items():
            fit, cal = split_cal(Fk, 5)
            evals[k] = lib.Det(gk, 31).fit(fit, cal)
        cand = [wi for wi in common.picks(c, dep, 4 * maxcase) if c.tidx[wi] >= test_cut][:maxcase]
        for wi in cand:
            f = c.Fte[wi]; base = c.base
            dev_all = np.abs(f[:K] - base[:K]) / c.fsd[:K] + np.abs(f[K:] - base[K:]) / c.fsd[K:]
            order = [int(x) for x in np.argsort(-dev_all)]
            rsd = np.random.RandomState(int(wi))
            out = {}
            g, k, q = lib.scr(dep, f, base, order, K); out["SCR dev"] = (g, k, q); k_scr = k if g is not None else None
            g, k, q = lib.scr(dep, f, base, order, K, delta=D); out["m-SCR dev"] = (g, k, q)
            g, _, q = lib.wachter_line(dep, f, base); out["Wachter"] = (g, None, q)
            g, q = wachter_margin(dep, f, base, D); out["Wachter margin"] = (g, None, q)
            r = lib.full_vote(dep, cop_t, f, base, order, K, delta=0.0, alpha=A, gamma=G, start_k=k_scr or 1)
            out["BetaRCE-T dev"] = (r["g"], r["k"], max(0, (k_scr or 1) - 1) + r["q"] + r["qc"])
            r = lib.rscr_v2(dep, cop_t, f, base, order, K, D, A, G, rsd)
            out["R-SCR+"] = (r["g"], r["k"], r["q"] + r["qc"])
            flags = {e: bool(d.flags(f)[0]) for e, d in evals.items()}
            res = {}
            for nm, (g, k, q) in out.items():
                if g is not None and k is None:
                    k = int(((np.abs(g[:K] - f[:K]) > 1e-9) | (np.abs(g[K:] - f[K:]) > 1e-9)).sum())
                rr = dict(found=int(g is not None), k=k, frac=(k / K if k else None), calls=int(q),
                          valid=(int(dep.score(g)[0] <= dep.tau) if g is not None else 0))
                for e, d in evals.items():
                    rr[e] = None if not flags[e] else (int(d.score(g)[0] <= d.tau) if g is not None else 0)
                res[nm] = rr
            records.append(dict(entity=name, gen=gk, win=int(c.tidx[wi]), res=res))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(entity=name, n=len(records), secs=time.time() - t0, records=records), open(f"{OUT}/{name}.json", "w"))
    print(name, len(records), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
