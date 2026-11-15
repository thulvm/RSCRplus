"""BetaRCE voi ham tim kiem goc cua tac gia (vendor_betarce), thiet lap dung chung common.py (ban 2).
Logic BetaRob chuyen nguyen tu src/explainers/posthoc/betarob.py: tien nghiem Jeffreys Beta(0.5, 0.5),
dat khi hop le tren mo hinh dang chay va can trai tai muc (1 - beta) > delta; tim theo vo cau no dan quanh ban sua goc.
Sieu tham so tim kiem lay tu configs/config_exp.yaml cua tac gia (max_iter 100, n_search_samples 100, step 0.1)."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, common
from vendor_betarce.growingspheres import growing_spheres_search

ROOT = os.environ.get("RSCR_ROOT", RSCR_HOME + "/data/smd")
OUT = os.environ.get("RSCR_OUT", RSCR_HOME + "/out/smd")
CONFIGS = [("BetaRCE-official default", 32, 0.90, 0.90), ("BetaRCE-official matched", 40, 0.95, 0.90)]
GS = dict(n_search_samples=100, p_norm=2, step=0.1, max_iter=100)


def main(name, maxcase=15):
    t0 = time.time()
    c = common.setup(ROOT, name)
    K = c.K
    cols = [f"c{i}" for i in range(2 * K)]
    records = []
    for gk in ("pca", "iforest"):
        dep, upd, cop_t, _ = common.detectors(c, gk)
        for wi in common.picks(c, dep, maxcase):
            f = c.Fte[wi]; start = int(c.tidx[wi]); base = c.base
            dev_all = np.abs(f[:K] - base[:K]) / c.fsd[:K] + np.abs(f[K:] - base[K:]) / c.fsd[K:]
            dev_order = [int(x) for x in np.argsort(-dev_all)]
            occ, nq = lib.rank_occlusion(dep, f, base, dev_order[:common.NCAND], K)
            occ = [int(x) for x in occ] + [x for x in dev_order if x not in occ]
            g0, k0, q0 = lib.scr(dep, f, base, occ, K)
            res = {}
            for cname, k, beta_conf, delta in CONFIGS:
                est = cop_t[:k]; calls = [0]

                def predicate(X):
                    X = np.atleast_2d(np.asarray(X, dtype=float))
                    valid = dep.score(X) <= dep.tau; calls[0] += X.shape[0]
                    if not valid.any():
                        return 0 if X.shape[0] == 1 else np.zeros(X.shape[0], dtype=int)
                    ones = np.zeros(X.shape[0])
                    for e in est:
                        ones += e.score(X) <= e.tau
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
                        cf = growing_spheres_search(
                            instance=pd.DataFrame(g0.reshape(1, -1), columns=cols), keys_mutable=cols, keys_immutable=[],
                            continuous_cols=cols, binary_cols=[], feature_order=cols, pred_fn_crisp=predicate, **GS)
                        if cf is not None and not np.any(np.isnan(cf)) and predicate(cf) == 1:
                            g = np.asarray(cf, dtype=float)
                r = common.judge(c, dep, upd, f, g, None, nq + q0 + calls[0], 0, dense=True)
                r["moved"] = None if g is None else bool(g is not g0)
                res[cname] = r
            records.append(dict(entity=name, gen=gk, win=start, upd_flag={u: bool(d.flags(f)[0]) for u, d in upd.items()}, gen_res=res))
    os.makedirs(OUT + "_betarce", exist_ok=True)
    json.dump(dict(entity=name, n=len(records), secs=time.time() - t0, records=records),
              open(f"{OUT}_betarce/{name}.json", "w"))
    print(name, len(records), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
