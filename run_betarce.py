"""Doi chieu R-SCR+ voi BetaRCE dung dung ham tim kiem goc (growing spheres, vendor_betarce).

Thiet lap bo phat hien, ban sao va cua so giong het run_smd.py (cung hat giong), nen ket qua ghep cap
duoc voi tep out/<kho>/<thuc the>.json theo (entity, gen, win).
Logic BetaRob duoc chuyen nguyen tu src/explainers/posthoc/betarob.py cua tac gia:
  vi tu = hop le tren mo hinh dang chay VA can trai Beta(0.5 + so dat, 0.5 + so khong dat) tai muc (1 - beta) > delta;
  neu ban sua goc da dat thi tra ve ngay, neu khong thi tim theo vo cau no dan quanh ban sua goc.
Cach dung: python3 run_betarce.py <ten_thuc_the> [so_ca_toi_da]
"""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json, time, os
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from vendor_betarce.growingspheres import growing_spheres_search

ROOT = os.environ.get("RSCR_ROOT", RSCR_HOME + "/data/smd")
OUT = os.environ.get("RSCR_OUT", RSCR_HOME + "/out/smd")
B, NCAND = 40, 10
# (ten cau hinh, so ban sao k, beta tin cay, delta muc ben)
CONFIGS = [("BetaRCE-official default", 32, 0.90, 0.90),   # mac dinh trong config_exp.yaml cua tac gia
           ("BetaRCE-official matched", 40, 0.95, 0.90)]   # cung cam ket voi R-SCR+: can duoi 5% >= 0.9, cung 40 ban sao
GS = dict(n_search_samples=100, p_norm=2, step=0.1, max_iter=100)   # mac dinh cua tac gia


def main(name, maxcase=15):
    t0 = time.time()
    tr, te, lab, segs = lib.load_entity(ROOT, name)
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd < 1e-3] = 1e-3
    Xtr, Xte = (tr - mu) / sd, (te - mu) / sd
    K = Xtr.shape[1]
    _, Wtr = lib.windows(Xtr)
    Ftr = lib.feats(Wtr)
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(Ftr))
    ncal = len(Ftr) // 5
    cal_idx, fit_idx = np.sort(perm[:ncal]), np.sort(perm[ncal:])
    fm, fs = Ftr[fit_idx].mean(0), Ftr[fit_idx].std(0)
    fs[fs < 1e-6] = 1.0
    S = lambda F: (F - fm) / fs
    Ffit, Fcal = S(Ftr[fit_idx]), S(Ftr[cal_idx])
    Fdep = Ffit[: int(0.7 * len(Ffit))]
    base = np.median(Ffit, 0)
    fsd = Ffit.std(0) + 1e-6
    held = {k: lib.Det(k, 0).fit(Ffit, Fcal) for k in ("ocsvm", "lof", "ecod")}
    tidx, Wte = lib.windows(Xte)
    Fte = S(lib.feats(Wte))
    ylab = np.array([lab[i:i + lib.W].mean() >= 0.5 for i in tidx])
    nn_cal = lib.NearestNeighbors(n_neighbors=2).fit(Fcal)
    med_nn = np.median(nn_cal.kneighbors(Fcal)[0][:, 1])
    cols = [f"c{i}" for i in range(2 * K)]

    records = []
    for gk in ("pca", "iforest"):
        dep = lib.Det(gk, 0).fit(Fdep, Fcal)
        upd = lib.Det(gk, 7).fit(Ffit, Fcal)
        copies = []
        L = int(0.6 * len(Fdep))
        for b in range(1, B + 1):
            rb = np.random.RandomState(1000 + b)
            rb.choice(len(Fdep), len(Fdep), replace=True)        # giu dung dong ngau nhien nhu run_smd.py
            s0 = rb.randint(0, len(Fdep) - L + 1)
            copies.append(lib.Det(gk, 100 + b).fit(Fdep[s0:s0 + L], Fcal))
        fl = np.where(ylab & dep.flags(Fte))[0]
        if len(fl) == 0:
            continue
        pick = fl[np.linspace(0, len(fl) - 1, min(maxcase, len(fl))).astype(int)]
        for wi in pick:
            f = Fte[wi]
            start = int(tidx[wi])
            dev_all = np.abs(f[:K] - base[:K]) / fsd[:K] + np.abs(f[K:] - base[K:]) / fsd[K:]
            dev_order = list(np.argsort(-dev_all))
            cand = dev_order[:NCAND]
            occ, _ = lib.rank_occlusion(dep, f, base, cand, K)
            occ = list(occ) + [c for c in dev_order if c not in occ]
            g0, k0, _ = lib.scr(dep, f, base, occ, K)          # ban sua goc, giong BetaRCE-SCR trong run_smd.py
            o = {h: bool(d.flags(f)[0]) for h, d in held.items()}
            upd_flag = bool(upd.flags(f)[0])
            res = {}
            for cname, k, beta_conf, delta in CONFIGS:
                est = copies[:k]
                calls = [0]

                def predicate(X):
                    X = np.atleast_2d(np.asarray(X, dtype=float))
                    valid = dep.score(X) <= dep.tau
                    if not valid.any():
                        return 0 if X.shape[0] == 1 else np.zeros(X.shape[0], dtype=int)
                    ones = np.zeros(X.shape[0])
                    for c in est:
                        ones += c.score(X) <= c.tau
                    calls[0] += X.shape[0] * len(est)
                    lb = stats.beta.ppf(1 - beta_conf, 0.5 + ones, 0.5 + (len(est) - ones))
                    out = (valid & (lb > delta)).astype(int)
                    return int(out[0]) if X.shape[0] == 1 else out

                g = None
                if g0 is not None:
                    if predicate(g0) == 1:
                        g = g0
                    else:
                        np.random.seed(int(wi) + 17)
                        cand_cf = growing_spheres_search(
                            instance=pd.DataFrame(g0.reshape(1, -1), columns=cols), keys_mutable=cols,
                            keys_immutable=[], continuous_cols=cols, binary_cols=[], feature_order=cols,
                            pred_fn_crisp=predicate, **GS)
                        if cand_cf is not None and not np.any(np.isnan(cand_cf)) and predicate(cand_cf) == 1:
                            g = np.asarray(cand_cf, dtype=float)
                if g is None:
                    res[cname] = dict(found=0, valid=0, var=0 if upd_flag else None,
                                      cross={h: (0 if o[h] else None) for h in held},
                                      k=None, frac=None, plaus=None, qc=calls[0])
                    continue
                ch = np.where((np.abs(g[:K] - f[:K]) > 1e-9) | (np.abs(g[K:] - f[K:]) > 1e-9))[0]
                res[cname] = dict(found=1, valid=int(dep.score(g)[0] <= dep.tau),
                                  var=(int(upd.score(g)[0] <= upd.tau) if upd_flag else None),
                                  cross={h: (int(d.score(g)[0] <= d.tau) if o[h] else None) for h, d in held.items()},
                                  k=int(len(ch)), frac=len(ch) / K,
                                  plaus=float(nn_cal.kneighbors(g[None], n_neighbors=1)[0][0, 0] / med_nn),
                                  qc=calls[0], moved=bool(g is not g0), l2_from_base=float(np.linalg.norm(g - g0)))
            records.append(dict(entity=name, gen=gk, win=start, upd_flag=upd_flag, held_flag=o, gen_res=res))
    os.makedirs(OUT + "_betarce", exist_ok=True)
    json.dump(dict(entity=name, n=len(records), secs=time.time() - t0, records=records),
              open(f"{OUT}_betarce/{name}.json", "w"))
    print(name, len(records), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
