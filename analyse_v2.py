"""Phan tich ban 2: bang theo kho, kiem dinh gop SMD+PSM so voi R-SCR+, cam ket, dung som, thu tu kenh, BetaRCE ma goc.
Ghi /home/claude/rscr/out/summary_v2.json."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, glob
import numpy as np
from scipy.stats import bootstrap, wilcoxon

O = RSCR_HOME + "/out"
HELD = ["ocsvm", "lof", "ecod"]
GENS = ["Wachter-line", "CoMTE", "p-SCR", "SCR", "NG-variant", "m-SCR", "BetaRCE-B", "BetaRCE-T",
        "BetaRCE-T+margin", "R-SCR+ no margin", "R-SCR+ bootstrap", "R-SCR+"]
ORD = ["deviation", "granger", "intgrad", "occlusion", "consensus", "shapley", "lime", "random"]
MAIN = "R-SCR+"


def load(d):
    out = []
    for f in sorted(glob.glob(f"{O}/{d}/*.json")):
        out += json.load(open(f))["records"]
    return out


def metric(v, m):
    if m == "var":
        return [v["var"]] if v["var"] is not None else []
    if m == "cross":
        return [v["cross"][h] for h in HELD if v["cross"][h] is not None]
    if m in ("frac", "plaus"):
        return [v[m]] if v[m] is not None else []
    return [v[m]]


def pe(recs, get, m):
    acc = {}
    for r in recs:
        v = get(r)
        if v is not None:
            acc.setdefault(r["entity"], []).extend(metric(v, m))
    return {e: float(np.median(x) if m == "plaus" else np.mean(x)) for e, x in acc.items() if x}


def ci(v):
    v = np.asarray(v, float)
    if len(v) < 3 or np.ptp(v) == 0:
        return [float(v.mean()), None, None]
    r = bootstrap((v,), np.mean, method="BCa", n_resamples=10000, random_state=0)
    return [float(v.mean()), float(r.confidence_interval.low), float(r.confidence_interval.high)]


def holm(ps):
    o = np.argsort(ps); m = len(ps); a = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); a[j] = min(1, run)
    return a


def paired(a, b):
    es = [e for e in a if e in b]
    d = np.array([a[e] - b[e] for e in es])
    p = float(wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
    return dict(ci=ci(d), p=p, n=len(es), hi=int((d > 0).sum()), lo=int((d < 0).sum()))


MS = ("valid", "var", "cross", "frac", "plaus", "calls", "qc", "found")
out = {"bench": {}}
recs_all = []
for b in ("smd", "psm"):
    recs = load(b); recs_all += recs
    if not recs:
        continue
    tab = {}
    for g in GENS:
        tab[g] = {m: ci(list(pe(recs, lambda r, g=g: r["gen_res"][g], m).values())) for m in MS}
    xai = {}
    for o in ORD:
        xai[o] = {w: {m: ci(list(pe(recs, lambda r, o=o, w=w: r["xai"][o][w], m).values())) for m in ("var", "cross", "frac", "calls")}
                  for w in ("scr", "rscr")}
        pm = {}
        for r in recs:
            if r["pm"] is not None:
                pm.setdefault(r["entity"], []).append(r["pm"][o])
        xai[o]["pm"] = ci([np.mean(v) for v in pm.values()]) if pm else None
    cov = {}
    for g in ("R-SCR+", "R-SCR+ no margin", "R-SCR+ bootstrap", "BetaRCE-T", "BetaRCE-T+margin", "BetaRCE-B"):
        acc = [r["gen_res"][g]["var"] for r in recs if r["gen_res"][g]["found"] and r["upd_flag"]]
        cov[g] = dict(n=len(acc), rate=float(np.mean(acc)))
    tried = sum(r["gen_res"][MAIN]["tried"] for r in recs)
    qc = sum(r["gen_res"][MAIN]["qc"] for r in recs)
    dis = sum(r["gen_res"][MAIN]["disagree"] for r in recs)
    aps = dict(tried=tried, qc=qc, full=40 * tried, saving=1 - qc / (40 * tried), disagree=dis / tried)
    ents = sorted({r["entity"] for r in recs})
    out["bench"][b] = dict(table=tab, xai=xai, coverage=cov, aps=aps, n_cases=len(recs), n_entities=len(ents))
    print(f"\n== {b}: {len(recs)} ca, {len(ents)} thuc the")
    for g in GENS:
        t = tab[g]
        print(f"{g:18s} valid {t['valid'][0]:.3f} VaR {t['var'][0]:.3f} cross {t['cross'][0]:.3f} "
              f"frac {t['frac'][0]:.3f} plaus {t['plaus'][0]:.2f} calls {t['calls'][0]:7.1f} qc {t['qc'][0]:6.1f} found {t['found'][0]:.3f}")
    print("cam ket", {g: (v['n'], round(v['rate'], 3)) for g, v in cov.items()})
    print("dung som", aps)

# kiem dinh gop so voi R-SCR+
tests = []
for g in GENS:
    if g == MAIN:
        continue
    for m in ("var", "cross", "frac", "calls"):
        a = pe(recs_all, lambda r: r["gen_res"][MAIN], m); bb = pe(recs_all, lambda r, g=g: r["gen_res"][g], m)
        t = paired(a, bb); t.update(vs=g, metric=m); tests.append(t)
adj = holm(np.array([t["p"] for t in tests]))
for t, a in zip(tests, adj):
    t["p_holm"] = float(a)
out["pooled"] = dict(n_entities=len({r["entity"] for r in recs_all}), n_cases=len(recs_all), tests=tests)
print("\n== gop, so voi R-SCR+ (Holm tren", len(tests), "kiem dinh)")
for t in tests:
    if t["metric"] in ("var", "cross", "calls"):
        c_ = t["ci"]
        print(f"  {t['metric']:5s} vs {t['vs']:18s} {c_[0]:+.3f} [{c_[1] if c_[1] is None else round(c_[1],3)};"
              f"{c_[2] if c_[2] is None else round(c_[2],3)}] pHolm={t['p_holm']:.2g} {t['hi']}/{t['lo']} n={t['n']}")

# thu tu kenh: kiem dinh gop R-SCR+ theo tung thu tu so voi do lech
xt = []
for o in ORD[1:]:
    a = pe(recs_all, lambda r, o=o: r["xai"][o]["rscr"], "var"); bb = pe(recs_all, lambda r: r["xai"]["deviation"]["rscr"], "var")
    t = paired(a, bb); t.update(ordering=o); xt.append(t)
adj = holm(np.array([t["p"] for t in xt]))
for t, a in zip(xt, adj):
    t["p_holm"] = float(a)
out["xai_tests"] = xt
print("\nR-SCR+ VaR theo thu tu kenh tru do lech:", [(t["ordering"], round(t["ci"][0], 3), round(t["p_holm"], 3)) for t in xt])

# BetaRCE ma goc
off = {}
for b in ("smd", "psm"):
    for f in glob.glob(f"{O}/{b}_betarce/*.json"):
        for r in json.load(open(f))["records"]:
            off[(r["entity"], r["gen"], r["win"])] = r
if off:
    main_by = {(r["entity"], r["gen"], r["win"]): r for r in recs_all}
    common = [k for k in off if k in main_by]
    OFFN = ["BetaRCE-official default", "BetaRCE-official matched"]
    rows, otests = {}, []
    mk = lambda nm: (lambda r: off[(r["entity"], r["gen"], r["win"])]["gen_res"][nm])
    sub = [main_by[k] for k in common]
    for nm in OFFN + [MAIN]:
        get = mk(nm) if nm != MAIN else (lambda r: r["gen_res"][MAIN])
        rows[nm] = {m: ci(list(pe(sub, get, m).values())) for m in ("found", "var", "cross", "frac", "calls")}
    for nm in OFFN:
        for m in ("var", "cross", "calls"):
            t = paired(pe(sub, lambda r: r["gen_res"][MAIN], m), pe(sub, mk(nm), m)); t.update(vs=nm, metric=m); otests.append(t)
    adj = holm(np.array([t["p"] for t in otests]))
    for t, a in zip(otests, adj):
        t["p_holm"] = float(a)
    fails = {nm: float(np.mean([1 - off[k]["gen_res"][nm]["found"] for k in common])) for nm in OFFN}
    fcalls = [off[k]["gen_res"][OFFN[1]]["calls"] for k in common if not off[k]["gen_res"][OFFN[1]]["found"]]
    moved = [off[k]["gen_res"][OFFN[1]]["moved"] for k in common if off[k]["gen_res"][OFFN[1]]["found"]]
    out["official"] = dict(n_cases=len(common), n_entities=len({k[0] for k in common}), rows=rows, tests=otests,
                           fail=fails, median_calls_failed=float(np.median(fcalls)) if fcalls else 0.0,
                           moved_rate=float(np.mean(moved)) if moved else 0.0)
    print("\n== BetaRCE ma goc tren", len(common), "ca:", {nm: {m: round(v[0], 3) for m, v in r.items()} for nm, r in rows.items()})
    print("that bai", fails, "goi khi that bai (trung vi)", out["official"]["median_calls_failed"], "di chuyen", out["official"]["moved_rate"])
    for t in otests:
        print("  ", t["vs"], t["metric"], [round(x, 3) if x is not None else None for x in t["ci"]], round(t["p_holm"], 4), t["hi"], t["lo"], t["n"])
json.dump(out, open(f"{O}/summary_v2.json", "w"), indent=1, default=float)
