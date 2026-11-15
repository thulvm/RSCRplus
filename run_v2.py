"""Thuc nghiem ban 2 (sau ra soat doc lap). Cach dung: python3 run_v2.py <thuc_the> [so_ca]
Thay doi so voi ban 1: hieu chuan khong ro ri, moi ung vien tham do mot lan, dem du moi lan goi bo phat hien,
them doi chung bo phieu du co bien, R-SCR+ dung thu tu do lech (khong ton lan goi de xep hang)."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, common

ROOT = os.environ.get("RSCR_ROOT", RSCR_HOME + "/data/smd")
OUT = os.environ.get("RSCR_OUT", RSCR_HOME + "/out/smd")
A, G, D = 0.10, 0.05, 0.25


def main(name, maxcase=15):
    t0 = time.time()
    c = common.setup(ROOT, name)
    K = c.K
    gr = lib.Granger(c.Xtr)
    records = []
    for gk in ("pca", "iforest"):
        dep, upd, cop_t, cop_b = common.detectors(c, gk)
        good = c.Fcal_dep[dep.score(c.Fcal_dep) <= dep.med]
        nn_good = lib.NearestNeighbors(n_neighbors=1).fit(good)
        for wi in common.picks(c, dep, maxcase):
            f = c.Fte[wi]; start = int(c.tidx[wi]); base = c.base
            dev_all = np.abs(f[:K] - base[:K]) / c.fsd[:K] + np.abs(f[K:] - base[K:]) / c.fsd[K:]
            dev_order = [int(x) for x in np.argsort(-dev_all)]
            cand = dev_order[:common.NCAND]
            ext = lambda r: [int(x) for x in r] + [x for x in dev_order if x not in r]
            guide = good[nn_good.kneighbors(f[None])[1][0, 0]]
            R, Rq = {}, {}
            R["deviation"], Rq["deviation"] = dev_order, 0
            R["occlusion"], Rq["occlusion"] = lib.rank_occlusion(dep, f, base, cand, K)
            R["shapley"], Rq["shapley"] = lib.rank_shapley(dep, f, base, cand, K)
            R["intgrad"], Rq["intgrad"] = lib.rank_ig(dep, f, base, cand, K)
            R["lime"], Rq["lime"] = lib.rank_lime(dep, f, base, cand, K, seed=int(wi))
            R["granger"], Rq["granger"] = gr.rank(c.Xte, start, cand), 0
            R["random"], Rq["random"] = list(np.random.RandomState(int(wi)).permutation(cand)), 0
            R["consensus"] = lib.borda([R["occlusion"], R["intgrad"], R["shapley"], R["granger"]])
            Rq["consensus"] = Rq["occlusion"] + Rq["intgrad"] + Rq["shapley"]
            O = {k: ext(v) for k, v in R.items()}
            J = lambda g, k, calls, qc=0, dense=False: common.judge(c, dep, upd, f, g, k, calls, qc, dense)
            rsd = lambda: np.random.RandomState(int(wi))
            res, xai = {}, {}
            # bo sinh doi chung
            g, k, q = lib.scr(dep, f, base, O["occlusion"], K); res["SCR"] = J(g, k, q + Rq["occlusion"]); k_scr = k
            g, k, q = lib.scr(dep, f, base, O["occlusion"], K, delta=D); res["m-SCR"] = J(g, k, q + Rq["occlusion"])
            g, k, q = lib.p_scr(dep, f, guide, O["occlusion"], K); res["p-SCR"] = J(g, k, q + Rq["occlusion"])
            g, k, q = lib.native_guide(dep, f, guide, K); res["NG-variant"] = J(g, k, q)
            g, k, q = lib.comte(dep, f, guide, K); res["CoMTE"] = J(g, k, q)
            g, _, q = lib.wachter_line(dep, f, base); res["Wachter-line"] = J(g, None, q, dense=True)
            for nm, cp in (("BetaRCE-B", cop_b), ("BetaRCE-T", cop_t)):
                r = lib.full_vote(dep, cp, f, base, O["occlusion"], K, delta=0.0, alpha=A, gamma=G,
                                  start_k=res["SCR"]["k"] or 1)
                res[nm] = J(r["g"], r["k"], Rq["occlusion"] + (k_scr or 0) + r["q"] + r["qc"], r["qc"])
            r = lib.full_vote(dep, cop_t, f, base, O["deviation"], K, delta=D, alpha=A, gamma=G)
            res["BetaRCE-T+margin"] = J(r["g"], r["k"], r["q"] + r["qc"], r["qc"])
            # phuong phap de xuat va cat bo
            r = lib.rscr_v2(dep, cop_t, f, base, O["deviation"], K, D, A, G, rsd(), track_full=True)
            res["R-SCR+"] = J(r["g"], r["k"], r["q"] + r["qc"], r["qc"])
            res["R-SCR+"].update(tried=r["tried"], disagree=r["disagree"])
            r = lib.rscr_v2(dep, cop_t, f, base, O["deviation"], K, 0.0, A, G, rsd())
            res["R-SCR+ no margin"] = J(r["g"], r["k"], r["q"] + r["qc"], r["qc"])
            r = lib.rscr_v2(dep, cop_b, f, base, O["deviation"], K, D, A, G, rsd())
            res["R-SCR+ bootstrap"] = J(r["g"], r["k"], r["q"] + r["qc"], r["qc"])
            # cau hoi ve thu tu kenh: SCR va R-SCR+ tren moi thu tu
            for rk, od in O.items():
                g, k, q = lib.scr(dep, f, base, od, K)
                r = lib.rscr_v2(dep, cop_t, f, base, od, K, D, A, G, rsd())
                xai[rk] = dict(scr=J(g, k, q + Rq[rk]), rscr=J(r["g"], r["k"], Rq[rk] + r["q"] + r["qc"], r["qc"]))
            mid = start + lib.W // 2
            ch = next((s[2] for s in c.segs if s[0] <= mid <= s[1]), None)
            pm = None if ch is None else {rk: len(set(od[:len(ch)]) & set(ch)) / len(ch) for rk, od in O.items()}
            records.append(dict(entity=name, gen=gk, win=start, upd_flag=bool(upd.flags(f)[0]),
                                gen_res=res, xai=xai, pm=pm))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(entity=name, n=len(records), secs=time.time() - t0, records=records), open(f"{OUT}/{name}.json", "w"))
    print(name, len(records), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
