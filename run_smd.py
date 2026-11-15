"""Chay thuc nghiem R-SCR tren mot thuc the SMD, ghi ket qua tung ca ra JSON.

Cach dung: python3 run_smd.py <ten_thuc_the> [so_ca_toi_da]
"""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

ROOT = os.environ.get("RSCR_ROOT", RSCR_HOME + "/data/smd")
OUT = os.environ.get("RSCR_OUT", RSCR_HOME + "/out/smd")
B, ALPHA, GAMMA, NCAND = 40, 0.10, 0.05, 10


def main(name, maxcase=15):
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
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
    Fdep = Ffit[: int(0.7 * len(Ffit))]          # phien ban dang chay: 70% dau theo thoi gian
    base = np.median(Ffit, 0)
    fsd = Ffit.std(0) + 1e-6

    held = {k: lib.Det(k, 0).fit(Ffit, Fcal) for k in ("ocsvm", "lof", "ecod")}
    gr = lib.Granger(Xtr)

    tidx, Wte = lib.windows(Xte)
    Fte = S(lib.feats(Wte))
    ylab = np.array([lab[i:i + lib.W].mean() >= 0.5 for i in tidx])

    nn_cal = lib.NearestNeighbors(n_neighbors=2).fit(Fcal)
    med_nn = np.median(nn_cal.kneighbors(Fcal)[0][:, 1])

    def plaus(g):
        return float(nn_cal.kneighbors(g[None], n_neighbors=1)[0][0, 0] / med_nn)

    def seg_of(start):
        mid = start + lib.W // 2
        for a, b, ch in segs:
            if a <= mid <= b:
                return ch
        return None

    records = []
    for gk in ("pca", "iforest"):
        dep = lib.Det(gk, 0).fit(Fdep, Fcal)
        upd = lib.Det(gk, 7).fit(Ffit, Fcal)          # phien ban cap nhat theo thoi gian
        copies_iid, copies = [], []
        L = int(0.6 * len(Fdep))
        for b in range(1, B + 1):
            rb = np.random.RandomState(1000 + b)
            boot = Fdep[rb.choice(len(Fdep), len(Fdep), replace=True)]
            copies_iid.append(lib.Det(gk, b).fit(boot, Fcal))
            s0 = rb.randint(0, len(Fdep) - L + 1)      # khoi thoi gian lien nhau, chi trong du lieu luc trien khai
            copies.append(lib.Det(gk, 100 + b).fit(Fdep[s0:s0 + L], Fcal))
        # cua so binh thuong duoc chinh bo phat hien chung nhan (diem duoi trung vi)
        good = Fcal[dep.score(Fcal) <= dep.med]
        nn_good = lib.NearestNeighbors(n_neighbors=1).fit(good)

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
            ext = lambda r: list(r) + [c for c in dev_order if c not in r]
            guide = good[nn_good.kneighbors(f[None])[1][0, 0]]

            R, Rq = {}, {}
            R["deviation"] = lib.rank_deviation(f, base, cand, K, fsd); Rq["deviation"] = 0
            R["occlusion"], Rq["occlusion"] = lib.rank_occlusion(dep, f, base, cand, K)
            R["shapley"], Rq["shapley"] = lib.rank_shapley(dep, f, base, cand, K)
            R["intgrad"], Rq["intgrad"] = lib.rank_ig(dep, f, base, cand, K)
            R["lime"], Rq["lime"] = lib.rank_lime(dep, f, base, cand, K, seed=int(wi))
            R["granger"] = gr.rank(Xte, start, cand); Rq["granger"] = 0
            R["random"] = list(np.random.RandomState(int(wi)).permutation(cand)); Rq["random"] = 0
            R["consensus"] = lib.borda([R["occlusion"], R["intgrad"], R["shapley"], R["granger"]])
            Rq["consensus"] = Rq["occlusion"] + Rq["intgrad"] + Rq["shapley"]

            o = {k: bool(d.flags(f)[0]) for k, d in held.items()}
            upd_flag = bool(upd.flags(f)[0])

            def judge(g, k, q, qc=0, dense=False):
                if g is None:
                    return dict(found=0, valid=0, var=0 if upd_flag else None,
                                cross={h: (0 if o[h] else None) for h in held},
                                k=None, frac=None, plaus=None, q=q, qc=qc)
                if dense:
                    ch = np.where((np.abs(g[:K] - f[:K]) > 1e-9) | (np.abs(g[K:] - f[K:]) > 1e-9))[0]
                    k = len(ch)
                return dict(found=1, valid=int(dep.score(g)[0] <= dep.tau),
                            var=(int(upd.score(g)[0] <= upd.tau) if upd_flag else None),
                            cross={h: (int(d.score(g)[0] <= d.tau) if o[h] else None) for h, d in held.items()},
                            k=int(k), frac=k / K, plaus=plaus(g), q=q, qc=qc)

            res = {}
            # cau hoi 5: SCR voi tung thu tu XAI
            xai = {}
            for rk, order in R.items():
                g, k, q = lib.scr(dep, f, base, ext(order), K)
                xai[rk] = judge(g, k, q + Rq[rk])
            occ = ext(R["occlusion"])
            g, k, q = lib.scr(dep, f, base, occ, K); res["SCR"] = judge(g, k, q + Rq["occlusion"])
            g, k, q = lib.scr(dep, f, base, occ, K, delta=0.25); res["m-SCR"] = judge(g, k, q + Rq["occlusion"])
            g, k, q = lib.p_scr(dep, f, guide, occ, K); res["p-SCR"] = judge(g, k, q + Rq["occlusion"])
            g, k, q = lib.native_guide(dep, f, guide, K); res["Native-Guide"] = judge(g, k, q)
            g, k, q = lib.comte(dep, f, guide, K); res["CoMTE"] = judge(g, k, q)
            g, _, q = lib.wachter_line(dep, f, base); res["Wachter"] = judge(g, None, q, dense=True)
            g, k, q, qc = lib.betarce_scr(dep, copies_iid, f, base, occ, K, ALPHA, GAMMA)
            res["BetaRCE-SCR iid"] = judge(g, k, q + Rq["occlusion"], qc)
            g, k, q, qc = lib.betarce_scr(dep, copies, f, base, occ, K, ALPHA, GAMMA)
            res["BetaRCE-SCR"] = judge(g, k, q + Rq["occlusion"], qc)
            ri = lib.rscr(dep, copies_iid, f, base, ext(R["consensus"]), K, alpha=ALPHA, gamma=GAMMA,
                          rs=np.random.RandomState(int(wi)))
            res["R-SCR iid"] = judge(ri["g"], ri["k"], ri["q"] + Rq["consensus"], ri["qc"])
            r = lib.rscr(dep, copies, f, base, ext(R["consensus"]), K, alpha=ALPHA, gamma=GAMMA,
                         rs=np.random.RandomState(int(wi)), track_full=True)
            res["R-SCR"] = judge(r["g"], r["k"], r["q"] + Rq["consensus"], r["qc"])
            res["R-SCR"].update(delta=r["delta"], disagree=r["disagree"], tried=r["tried"])
            # cat bo tung thanh phan
            r2 = lib.rscr(dep, copies, f, base, occ, K, alpha=ALPHA, gamma=GAMMA,
                          rs=np.random.RandomState(int(wi)))
            res["R-SCR no-consensus"] = judge(r2["g"], r2["k"], r2["q"] + Rq["occlusion"], r2["qc"])
            r3 = lib.rscr(dep, copies, f, base, ext(R["consensus"]), K, deltas=(0.0,), alpha=ALPHA,
                          gamma=GAMMA, rs=np.random.RandomState(int(wi)))
            res["R-SCR no-margin"] = judge(r3["g"], r3["k"], r3["q"] + Rq["consensus"], r3["qc"])

            r4 = lib.rscr(dep, copies, f, base, ext(R["consensus"]), K, deltas=(0.25, 0.5, 1.0), alpha=ALPHA,
                          gamma=GAMMA, rs=np.random.RandomState(int(wi)))
            res["R-SCR+"] = judge(r4["g"], r4["k"], r4["q"] + Rq["consensus"], r4["qc"])
            res["R-SCR+"].update(delta=r4["delta"], tried=r4["tried"])

            ch = seg_of(start)
            pm = None
            if ch is not None:
                m = len(ch)
                pm = {rk: len(set(order[:m]) & set(ch)) / m for rk, order in R.items()}
            records.append(dict(entity=name, gen=gk, win=int(start), upd_flag=upd_flag,
                                held_flag=o, gen_res=res, xai=xai, pm=pm))
    json.dump(dict(entity=name, n=len(records), secs=time.time() - t0, records=records),
              open(f"{OUT}/{name}.json", "w"))
    print(name, len(records), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
