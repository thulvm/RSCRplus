"""Tong hop ket qua R-SCR: trung binh theo thuc the, KTC BCa, Wilcoxon ghep cap co hieu chinh Holm."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, glob, sys, os
import numpy as np
from scipy.stats import bootstrap, wilcoxon

OUT = sys.argv[1] if len(sys.argv) > 1 else RSCR_HOME + "/out/smd"
GENS = ["SCR", "m-SCR", "p-SCR", "Native-Guide", "CoMTE", "Wachter", "BetaRCE-SCR iid", "BetaRCE-SCR",
        "R-SCR iid", "R-SCR", "R-SCR no-consensus", "R-SCR no-margin", "R-SCR+"]
XAI = ["deviation", "occlusion", "shapley", "intgrad", "lime", "granger", "random", "consensus"]
HELD = ["ocsvm", "lof", "ecod"]
MAIN = os.environ.get("MAIN", "R-SCR+")

recs = []
for f in sorted(glob.glob(f"{OUT}/*.json")):
    recs += json.load(open(f))["records"]
ents = sorted({r["entity"] for r in recs})


def ci(v):
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)])
    if len(v) < 3 or np.ptp(v) == 0:
        return float(np.mean(v)) if len(v) else None, None, None
    r = bootstrap((v,), np.mean, method="BCa", n_resamples=10000, random_state=0)
    return float(v.mean()), float(r.confidence_interval.low), float(r.confidence_interval.high)


def per_entity(sel, key, field):
    """Trung binh theo thuc the cho phuong phap key trong nhom field (gen_res hoac xai)."""
    out = {}
    for e in ents:
        rr = [r for r in recs if r["entity"] == e and sel(r)]
        if not rr:
            continue
        v = [r[field][key] for r in rr]
        val = np.mean([x["valid"] for x in v])
        var_ = [x["var"] for x in v if x["var"] is not None]
        cr = [x["cross"][h] for x in v for h in HELD if x["cross"][h] is not None]
        fr = [x["frac"] for x in v if x["frac"] is not None]
        pl = [x["plaus"] for x in v if x["plaus"] is not None]
        qc = [x["qc"] for x in v]
        out[e] = dict(valid=val, var=np.mean(var_) if var_ else np.nan, cross=np.mean(cr) if cr else np.nan,
                      frac=np.mean(fr) if fr else np.nan, plaus=np.median(pl) if pl else np.nan,
                      qc=np.mean(qc))
    return out


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1, run)
    return adj


def table(sel, label):
    res = {}
    print(f"\n=== {label}: {sum(sel(r) for r in recs)} ca, {len(ents)} thuc the ===")
    print(f"{'phuong phap':22s} {'hop le':>7s} {'VaR':>22s} {'cheo kien truc':>22s} {'ti le kenh':>10s} {'hop ly':>7s} {'goi ban sao':>11s}")
    for g in GENS:
        pe = per_entity(sel, g, "gen_res")
        row = {m: ci([pe[e][m] for e in pe]) for m in ("valid", "var", "cross", "frac", "plaus", "qc")}
        row["_pe"] = pe
        res[g] = row
        f = lambda t: "  -  " if t[0] is None else (f"{t[0]:.3f} [{t[1]:.3f};{t[2]:.3f}]" if t[1] is not None else f"{t[0]:.3f}")
        print(f"{g:22s} {row['valid'][0]:7.3f} {f(row['var']):>22s} {f(row['cross']):>22s} {row['frac'][0]:10.3f} {row['plaus'][0]:7.2f} {row['qc'][0]:11.1f}")
    # kiem dinh ghep cap R-SCR so voi tung bo sinh
    print(f"\n{MAIN} tru phuong phap khac (ghep cap theo thuc the), Holm tren moi kiem dinh cua bang:")
    tests = []
    for metric in ("var", "cross"):
        for g in GENS:
            if g == MAIN:
                continue
            a, b = res[MAIN]["_pe"], res[g]["_pe"]
            common = [e for e in a if e in b and not np.isnan(a[e][metric]) and not np.isnan(b[e][metric])]
            d = np.array([a[e][metric] - b[e][metric] for e in common])
            p = wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
            tests.append((metric, g, d, p))
    adj = holm(np.array([t[3] for t in tests]))
    out = []
    for (metric, g, d, p), pa in zip(tests, adj):
        m, lo, hi = ci(d)
        win = int((d > 0).sum()); lose = int((d < 0).sum())
        print(f"  {metric:5s} vs {g:20s} {m:+.3f} [{lo if lo is None else round(lo,3)};{hi if hi is None else round(hi,3)}] p={p:.2g} pHolm={pa:.2g} thang/thua {win}/{lose} n={len(d)}")
        out.append(dict(metric=metric, vs=g, mean=m, lo=lo, hi=hi, p=float(p), p_holm=float(pa), win=win, lose=lose, n=len(d)))
    return res, out


summary = {}
for lab, sel in [("ca hai bo dang chay", lambda r: True), ("PCA", lambda r: r["gen"] == "pca"),
                 ("Isolation Forest", lambda r: r["gen"] == "iforest")]:
    res, tests = table(sel, lab)
    summary[lab] = dict(table={g: {k: v for k, v in res[g].items() if k != "_pe"} for g in GENS}, tests=tests)

# cau hoi 4: tham do va dung
rr = [r["gen_res"]["R-SCR"] for r in recs]
tried = sum(x["tried"] for x in rr); qc = sum(x["qc"] for x in rr); dis = sum(x["disagree"] for x in rr)
full = 40 * tried
print(f"\nTham do va dung: {qc} lan goi ban sao so voi {full} neu hoi du, tiet kiem {1 - qc / full:.3f}; "
      f"quyet dinh khac hoi du {dis}/{tried} = {dis / max(tried, 1):.3f}")
pe_r = per_entity(lambda r: True, "R-SCR", "gen_res"); pe_b = per_entity(lambda r: True, "BetaRCE-SCR", "gen_res")
summary["aps"] = dict(qc=qc, full=full, saving=1 - qc / full, disagree=dis, tried=tried,
                      rscr_qc=ci([pe_r[e]["qc"] for e in pe_r]), beta_qc=ci([pe_b[e]["qc"] for e in pe_b]))
print("goi ban sao trung binh moi ca: R-SCR", summary["aps"]["rscr_qc"], " BetaRCE", summary["aps"]["beta_qc"])

# cau hoi 3: cam ket tren phien ban cap nhat
summary["coverage"] = {}
for mm in ("R-SCR", "R-SCR+", "BetaRCE-SCR"):
    acc = [r for r in recs if r["gen_res"][mm]["found"] and r["upd_flag"]]
    cov = np.mean([r["gen_res"][mm]["var"] for r in acc])
    print(f"Cam ket {mm}: {len(acc)} ban sua chap nhan, ti le van hop le tren phien ban cap nhat = {cov:.3f}")
    summary["coverage"][mm] = dict(n=len(acc), rate=float(cov))
dl = [r["gen_res"]["R-SCR+"]["delta"] for r in recs if r["gen_res"]["R-SCR+"]["found"]]
summary["rscr_plus_delta"] = {str(d): dl.count(d) for d in sorted(set(dl))}
print("Bien R-SCR+ duoc chon:", summary["rscr_plus_delta"])

# cau hoi 5: tung thu tu XAI voi SCR
print("\nThu tu XAI dung cho SCR:")
xs = {}
for x in XAI:
    pe = per_entity(lambda r: True, x, "xai")
    pm = {}
    for e in ents:
        v = [r["pm"][x] for r in recs if r["entity"] == e and r["pm"] is not None]
        if v:
            pm[e] = np.mean(v)
    xs[x] = dict(var=ci([pe[e]["var"] for e in pe]), cross=ci([pe[e]["cross"] for e in pe]),
                 frac=ci([pe[e]["frac"] for e in pe]), valid=ci([pe[e]["valid"] for e in pe]),
                 pm=ci(list(pm.values())), n_pm=len(pm))
    print(f"  {x:10s} hop le {xs[x]['valid'][0]:.3f} VaR {xs[x]['var'][0]:.3f} cheo {xs[x]['cross'][0]:.3f} "
          f"ti le kenh {xs[x]['frac'][0]:.3f} P@m {xs[x]['pm'][0] if xs[x]['pm'][0] is None else round(xs[x]['pm'][0],3)} (n={len(pm)})")
summary["xai"] = xs
summary["n_cases"] = len(recs); summary["n_entities"] = len(ents)
json.dump(summary, open(f"{OUT}/../summary_{os.path.basename(OUT.rstrip("/"))}.json", "w"), indent=1, default=float)
