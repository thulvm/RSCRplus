"""Do nhay cua ba baseline tu van lieu bo phan loai (RobX, T-Rex, ROAR) voi sieu tham so, tren 10 may SMD.
Cach dung: python3 run_sens.py <thuc_the>. Cung cua so va cung bo danh gia voi giao thuc mo rong (6 cua so moi ho)."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, common, lib5
from run_v5 import split_cal
ROOT = RSCR_HOME + "/data/smd"; OUT = RSCR_HOME + "/out/smd_sens"
CONF = [("RobX", dict(sigma=0.3, tau_r=0.75), 4.0), ("RobX", dict(sigma=0.1, tau_r=0.75), 4.0), ("RobX", dict(sigma=0.5, tau_r=0.75), 4.0),
        ("RobX", dict(sigma=0.3, tau_r=0.9), 4.0), ("RobX", dict(sigma=0.3, tau_r=0.75), 2.0), ("RobX", dict(sigma=0.3, tau_r=0.75), 8.0),
        ("T-Rex", dict(sigma=0.3, tau_r=0.75), 4.0), ("T-Rex", dict(sigma=0.1, tau_r=0.75), 4.0), ("T-Rex", dict(sigma=0.5, tau_r=0.75), 4.0),
        ("ROAR", dict(delta=0.1), 4.0), ("ROAR", dict(delta=0.05), 4.0), ("ROAR", dict(delta=0.2), 4.0)]


def main(name, maxcase=6):
    if os.path.exists(f"{OUT}/{name}.json"):
        return
    t0 = time.time(); c = common.setup(ROOT, name); K = c.K
    starts, Wtr = lib.windows(c.Xtr); Fall = (lib.feats(Wtr) - c.fm) / c.fs
    N = len(c.Xtr); NT = len(c.Xte); nt = ~c.ylab
    chains = {"chain1": Fall[starts >= int(0.3 * N)],
              "chain2": np.vstack([Fall[starts >= int(0.6 * N)], c.Fte[nt & (c.tidx < int(0.3 * NT))]]),
              "chain3": np.vstack([Fall[starts >= int(0.8 * N)], c.Fte[nt & (c.tidx < int(0.6 * NT))]])}
    recs = []
    for gk in ("pca", "iforest", "ae"):
        dep, upd, cop_t, _ = common.detectors(c, gk)
        ev = {f"upd_s{s}": lib.Det(gk, 7 + s).fit(c.Ffit, c.Fcal) for s in range(3)}
        for k, Fk in chains.items():
            fit, cal = split_cal(Fk, 5); ev[k] = lib.Det(gk, 31).fit(fit, cal)
        ev.update({"x_" + h: d for h, d in c.held.items()})
        cand = [wi for wi in common.picks(c, dep, 60) if c.tidx[wi] >= int(0.6 * NT)]
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, min(maxcase, len(cand))).astype(int)] if cand else []
        for wi in cand:
            f = c.Fte[wi]; base = c.base
            dev = np.abs(f[:K] - base[:K]) / c.fsd[:K] + np.abs(f[K:] - base[K:]) / c.fsd[K:]
            order = [int(x) for x in np.argsort(-dev)]
            g0, _, q0 = lib.scr(dep, f, base, order, K)
            flags = {e: bool(d.flags(f)[0]) for e, d in ev.items()}
            res = {}
            for i, (m, kw, slope) in enumerate(CONF):
                lib5.SLOPE = slope; rs = np.random.RandomState(int(wi))
                if m == "RobX":
                    g, q = lib5.robx(dep, f, g0, c.Fcal_dep, rs=rs, **kw); q += q0
                elif m == "T-Rex":
                    g, q = lib5.trex(dep, f, g0, rs=rs, **kw); q += q0
                else:
                    g, q = lib5.roar(dep, f, rs=rs, **kw)
                lib5.SLOPE = 4.0
                k = int(((np.abs(g[:K] - f[:K]) > 1e-9) | (np.abs(g[K:] - f[K:]) > 1e-9)).sum()) if g is not None else None
                rr = dict(found=int(g is not None), frac=(k / K if k else None), calls=int(q))
                for e, d in ev.items():
                    rr[e] = None if not flags[e] else (int(d.score(g)[0] <= d.tau) if g is not None else 0)
                res[str(i)] = rr
            recs.append(dict(entity=name, gen=gk, win=int(c.tidx[wi]), res=res))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(entity=name, conf=[[m, kw, s] for m, kw, s in CONF], secs=time.time() - t0, records=recs), open(f"{OUT}/{name}.json", "w"))
    print(name, len(recs), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
