"""Thuc nghiem ban 6 (sua theo phan bien tap chi). Cach dung: python3 run_v6.py <thuc_the> [so_ca]
Cung cua so voi ban 5 (giao thuc mo rong, 40% cuoi tap kiem tra), cong them:
  - m-SCR quet bien delta trong {0.25, 0.35, 0.5, 0.75} va R-SCR+ o delta 0.5, de ve duong Pareto do ben, kich thuoc, so lan goi
  - bao dong gia: cua so bi gan co nhung nhan binh thuong, toi da 10 moi thuc the va bo phat hien
  - phieu cua du 40 ban sao cho moi ban sua duoc chap nhan (do tuong quan giua cac ban sao)
  - bien chuan hoa z cua moi ban sua duoi bo dang chay va cac bo cap nhat, va do hop ly (khoang cach toi cua so binh thuong gan nhat)
  - cap (z truoc, z sau) cua cua so binh thuong trong tap kiem tra, de so sanh do dich o cung muc bien ban dau."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, common, lib5
from run_v5 import split_cal

ROOT = os.environ.get("RSCR_ROOT", RSCR_HOME + "/data/smd")
OUT = os.environ.get("RSCR_OUT", RSCR_HOME + "/out/smd_v6")
A, G = 0.10, 0.05
FAMS = os.environ.get("RSCR_FAMS", "pca,iforest,ae").split(",")
MARGINS = (0.25, 0.35, 0.5, 0.75)


def zval(d, X):
    return (d.tau - d.score(np.atleast_2d(X))) / max(d.tau - d.med, 1e-9)


def main(name, maxcase=15, maxfa=10):
    t0 = time.time()
    c = common.setup(ROOT, name)
    K = c.K
    starts, Wtr = lib.windows(c.Xtr)
    Fall = (lib.feats(Wtr) - c.fm) / c.fs
    N = len(c.Xtr); NT = len(c.Xte)
    norm_te = ~c.ylab
    chain_sets = {
        "chain1": Fall[starts >= int(0.3 * N)],
        "chain2": np.vstack([Fall[starts >= int(0.6 * N)], c.Fte[norm_te & (c.tidx < int(0.3 * NT))]]),
        "chain3": np.vstack([Fall[starts >= int(0.8 * N)], c.Fte[norm_te & (c.tidx < int(0.6 * NT))]]),
    }
    test_cut = int(0.6 * NT)
    records, normals = [], []
    for gk in FAMS:
        dep, upd, cop_t, _ = common.detectors(c, gk)
        evals = {}
        for s in range(5):
            evals[f"upd_s{s}"] = lib.Det(gk, 7 + s).fit(c.Ffit, c.Fcal)
            evals[f"same_s{s}"] = lib.Det(gk, 21 + s).fit(c.Fdep, c.Fcal_dep)
        for k, Fk in chain_sets.items():
            fit, cal = split_cal(Fk, 5)
            evals[k] = lib.Det(gk, 31).fit(fit, cal)
        allev = dict(evals); allev.update({"x_" + h: d for h, d in c.held.items()})
        # cua so binh thuong cua phan sau tap kiem tra: z truoc va sau mot lan cap nhat toan lich su
        nidx = np.where(norm_te & (c.tidx >= test_cut))[0]
        if len(nidx):
            sub = nidx[np.linspace(0, len(nidx) - 1, min(400, len(nidx))).astype(int)]
            normals.append(dict(gen=gk, z_dep=zval(dep, c.Fte[sub]).tolist(), z_upd=zval(evals["upd_s0"], c.Fte[sub]).tolist(),
                                z_chain=zval(evals["chain3"], c.Fte[sub]).tolist()))
        fl = dep.flags(c.Fte)
        tp = [wi for wi in common.picks(c, dep, 4 * maxcase) if c.tidx[wi] >= test_cut][:maxcase]
        fa_all = np.where(fl & ~c.ylab & (c.tidx >= test_cut))[0]
        fa = list(fa_all[np.linspace(0, len(fa_all) - 1, min(maxfa, len(fa_all))).astype(int)]) if len(fa_all) else []
        for kind, cand in (("tp", tp), ("fa", fa)):
            for wi in cand:
                f = c.Fte[wi]; base = c.base
                dev_all = np.abs(f[:K] - base[:K]) / c.fsd[:K] + np.abs(f[K:] - base[K:]) / c.fsd[K:]
                order = [int(x) for x in np.argsort(-dev_all)]
                out = {}
                g, k, q = lib.scr(dep, f, base, order, K); out["SCR dev"] = (g, k, q); k_scr = k if g is not None else None
                for d in MARGINS:
                    g, k, q = lib.scr(dep, f, base, order, K, delta=d); out[f"m-SCR {d}"] = (g, k, q)
                r = lib.full_vote(dep, cop_t, f, base, order, K, delta=0.0, alpha=A, gamma=G, start_k=k_scr or 1)
                out["BetaRCE-T dev"] = (r["g"], r["k"], max(0, (k_scr or 1) - 1) + r["q"] + r["qc"])
                for d in (0.25, 0.5):
                    r = lib.rscr_v2(dep, cop_t, f, base, order, K, d, A, G, np.random.RandomState(int(wi)))
                    out[f"R-SCR+ {d}"] = (r["g"], r["k"], r["q"] + r["qc"])
                if kind == "fa":
                    g0 = out["SCR dev"][0]; q0 = out["SCR dev"][2]
                    g, q = lib5.robx(dep, f, g0, c.Fcal_dep, tau_r=0.75, rs=np.random.RandomState(int(wi)))
                    out["RobX strict"] = (g, None, q0 + q)
                flags = {e: bool(d.flags(f)[0]) for e, d in allev.items()}
                res = {}
                for nm, (g, k, q) in out.items():
                    if g is not None and k is None:
                        k = int(((np.abs(g[:K] - f[:K]) > 1e-9) | (np.abs(g[K:] - f[K:]) > 1e-9)).sum())
                    rr = dict(found=int(g is not None), k=k, frac=(k / K if k else None), calls=int(q),
                              plaus=(c.plaus(g) if g is not None else None),
                              z_dep=(float(zval(dep, g)[0]) if g is not None else None),
                              z_upd=(float(zval(evals["upd_s0"], g)[0]) if g is not None else None))
                    for e, d in allev.items():
                        rr[e] = None if not flags[e] else (int(d.score(g)[0] <= d.tau) if g is not None else 0)
                    if g is not None and nm in ("R-SCR+ 0.25", "m-SCR 0.25", "SCR dev"):
                        rr["votes"] = [int(cp.score(g)[0] <= cp.tau) for cp in cop_t]
                    res[nm] = rr
                records.append(dict(entity=name, gen=gk, kind=kind, win=int(c.tidx[wi]), res=res))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(entity=name, n=len(records), secs=time.time() - t0, records=records, normals=normals),
              open(f"{OUT}/{name}.json", "w"))
    print(name, len(records), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
