"""Kiem dinh ghep cap gop SMD va PSM cho cac phep so then chot cua R-SCR+."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, glob, sys
import numpy as np
from scipy.stats import bootstrap, wilcoxon
sys.argv = [sys.argv[0]]
HELD = ["ocsvm", "lof", "ecod"]

recs = []
for d in ("smd", "psm"):
    for f in sorted(glob.glob(fRSCR_HOME + "/out/{d}/*.json")):
        recs += json.load(open(f))["records"]
ents = sorted({r["entity"] for r in recs})


def pe(method, metric):
    out = {}
    for e in ents:
        v = [r["gen_res"][method] for r in recs if r["entity"] == e]
        if metric == "var":
            x = [a["var"] for a in v if a["var"] is not None]
        elif metric == "cross":
            x = [a["cross"][h] for a in v for h in HELD if a["cross"][h] is not None]
        elif metric == "qc":
            x = [a["qc"] for a in v]
        elif metric == "frac":
            x = [a["frac"] for a in v if a["frac"] is not None]
        if x:
            out[e] = float(np.mean(x))
    return out


def ci(v):
    r = bootstrap((np.asarray(v),), np.mean, method="BCa", n_resamples=10000, random_state=0)
    return float(np.mean(v)), float(r.confidence_interval.low), float(r.confidence_interval.high)


rows, ps = [], []
for metric in ("var", "cross", "qc", "frac"):
    for other in ("BetaRCE-SCR", "BetaRCE-SCR iid", "m-SCR", "SCR", "Native-Guide", "CoMTE", "Wachter", "p-SCR"):
        if metric in ("qc",) and not other.startswith("BetaRCE"):
            continue
        a, b = pe("R-SCR+", metric), pe(other, metric)
        com = [e for e in a if e in b]
        d = np.array([a[e] - b[e] for e in com])
        p = float(wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
        m, lo, hi = ci(d)
        rows.append(dict(metric=metric, vs=other, n=len(d), mean=m, lo=lo, hi=hi, p=p,
                         win=int((d > 0).sum()), lose=int((d < 0).sum())))
        ps.append(p)
order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
for i, j in enumerate(order):
    run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1, run)
for r, pa in zip(rows, adj):
    r["p_holm"] = float(pa)
    print(f"{r['metric']:5s} R-SCR+ tru {r['vs']:16s} {r['mean']:+.3f} [{r['lo']:+.3f};{r['hi']:+.3f}] "
          f"p={r['p']:.2g} pHolm={pa:.2g} thang/thua {r['win']}/{r['lose']} n={r['n']}")
json.dump(dict(n_blocks=len(ents), n_cases=len(recs), tests=rows),
          open(RSCR_HOME + "/out/pooled_tests.json", "w"), indent=1)
