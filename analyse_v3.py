"""Phan tich ban 3. Kiem dinh suy dien chi tren SMD (28 may doc lap); PSM bao cao mo ta vi 16 khoi
dung chung mot bo phat hien. Ghi /home/claude/rscr/out/summary_v3.json."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, glob
import numpy as np
from scipy.stats import bootstrap, wilcoxon

O = RSCR_HOME + "/out"
HELD = ["ocsvm", "lof", "ecod"]
GENS = ["Wachter-line", "CoMTE", "p-SCR", "SCR", "NG-variant", "m-SCR", "m-SCR dev", "BetaRCE-B", "BetaRCE-T",
        "Full vote", "R-SCR+ d0.0", "R-SCR+ d0.1", "R-SCR+ d0.5", "R-SCR+ bootstrap", "R-SCR+",
        "R-SCR+ no curtail", "R-SCR+ Jeffreys", "BetaRCE-T uniform"]
ORD = ["deviation", "granger", "intgrad", "occlusion", "consensus", "shapley", "lime", "random"]
MS = ("valid", "var", "var_later", "cross", "frac", "plaus", "calls", "qc", "found")


def load(d):
    out = []
    for f in sorted(glob.glob(f"{O}/{d}/*.json")):
        out += json.load(open(f))["records"]
    return out


def metric(v, m):
    if m in ("var", "var_later"):
        return [v[m]] if v.get(m) is not None else []
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
    v = np.asarray(list(v), float)
    if len(v) == 0:
        return [None, None, None]
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
    p = float(wilcoxon(d, zero_method="zsplit").pvalue) if len(d) and np.any(d != 0) else 1.0
    return dict(ci=ci(d), p=p, n=len(es), hi=int((d > 0).sum()), lo=int((d < 0).sum()))


G = lambda g: (lambda r: r["gen_res"][g])
X = lambda o, w: (lambda r: r["xai"][o][w])
out = {"bench": {}}
recs = {b: load(b) for b in ("smd", "psm")}
for b, rr in recs.items():
    if not rr:
        continue
    tab = {g: {m: ci(pe(rr, G(g), m).values()) for m in MS} for g in GENS}
    xai = {}
    for o in ORD:
        xai[o] = {w: {m: ci(pe(rr, X(o, w), m).values()) for m in ("var", "var_later", "cross", "frac", "calls")}
                  for w in ("scr", "rscr")}
        pm = {}
        for r in rr:
            if r["pm"] is not None:
                pm.setdefault(r["entity"], []).append(r["pm"][o])
        xai[o]["pm"] = ci([np.mean(v) for v in pm.values()]) if pm else [None, None, None]
    cov = {}
    for g in ("R-SCR+", "R-SCR+ d0.0", "Full vote", "BetaRCE-T", "BetaRCE-B", "m-SCR dev"):
        for u, key in (("upd", "var"), ("later", "var_later")):
            acc = [r["gen_res"][g][key] for r in rr if r["gen_res"][g]["found"] and r["upd_flag"][u]]
            cov[f"{g}|{u}"] = dict(n=len(acc), rate=float(np.mean(acc)) if acc else None)
    tried = sum(r["gen_res"]["R-SCR+"]["tried"] for r in rr)
    qc = sum(r["gen_res"]["R-SCR+"]["qc"] for r in rr)
    dis = sum(r["gen_res"]["R-SCR+"]["disagree"] for r in rr)
    aps = dict(tried=tried, qc=qc, full=40 * tried, saving=1 - qc / (40 * tried), disagree=dis / tried)
    bygen = {gk: {g: {m: ci(pe([r for r in rr if r["gen"] == gk], G(g), m).values()) for m in ("var", "var_later", "cross", "calls")}
                  for g in ("SCR", "m-SCR", "BetaRCE-T", "R-SCR+")} for gk in ("pca", "iforest")}
    tried = {g: sum(r["gen_res"][g].get("tried", 0) for r in rr) for g in ("R-SCR+", "R-SCR+ d0.0", "R-SCR+ Jeffreys")}
    out["bench"][b] = dict(table=tab, xai=xai, coverage=cov, aps=aps, n_cases=len(rr), bygen=bygen, tried_counts=tried,
                           n_entities=len({r["entity"] for r in rr}))
    print(f"\n== {b}: {len(rr)} ca, {out['bench'][b]['n_entities']} thuc the")
    for g in GENS:
        t = tab[g]
        print(f"{g:17s} VaR {t['var'][0]:.3f} later {t['var_later'][0]:.3f} cross {t['cross'][0]:.3f} "
              f"size {t['frac'][0]:.3f} plaus {t['plaus'][0]:.2f} calls {t['calls'][0]:7.1f} found {t['found'][0]:.3f}")
    print("cam ket", {k: (v["n"], None if v["rate"] is None else round(v["rate"], 3)) for k, v in cov.items()})
    print("dung som", {k: round(v, 3) if isinstance(v, float) else v for k, v in aps.items()})
    print("thu tu kenh (R-SCR+):", {o: (round(xai[o]["rscr"]["var"][0], 3), round(xai[o]["rscr"]["cross"][0], 3),
                                       round(xai[o]["rscr"]["frac"][0], 3), round(xai[o]["rscr"]["calls"][0]),
                                       None if xai[o]["pm"][0] is None else round(xai[o]["pm"][0], 3)) for o in ORD})

# kiem dinh tren SMD
smd = recs["smd"]
COMP = [  # (nhan, getter A, getter B) ; hieu = A - B
    ("R-SCR+ vs BetaRCE-T", G("R-SCR+"), G("BetaRCE-T")),
    ("R-SCR+ vs BetaRCE-B", G("R-SCR+"), G("BetaRCE-B")),
    ("R-SCR+ vs Full vote", G("R-SCR+"), G("Full vote")),
    ("R-SCR+ vs m-SCR dev", G("R-SCR+"), G("m-SCR dev")),
    ("R-SCR+ vs m-SCR", G("R-SCR+"), G("m-SCR")),
    ("R-SCR+ vs SCR", G("R-SCR+"), G("SCR")),
    ("R-SCR+ vs NG-variant", G("R-SCR+"), G("NG-variant")),
    ("R-SCR+ vs Wachter-line", G("R-SCR+"), G("Wachter-line")),
    ("R-SCR+ vs no margin", G("R-SCR+"), G("R-SCR+ d0.0")),
    ("R-SCR+ vs bootstrap copies", G("R-SCR+"), G("R-SCR+ bootstrap")),
    ("R-SCR+ vs BetaRCE-T uniform", G("R-SCR+"), G("BetaRCE-T uniform")),
    ("prior: BetaRCE-T uniform vs BetaRCE-T", G("BetaRCE-T uniform"), G("BetaRCE-T")),
    ("prior: R-SCR+ vs R-SCR+ Jeffreys", G("R-SCR+"), G("R-SCR+ Jeffreys")),
    ("margin: m-SCR dev vs SCR dev", G("m-SCR dev"), X("deviation", "scr")),
    ("ordering: m-SCR dev vs m-SCR", G("m-SCR dev"), G("m-SCR")),
    ("ordering: R-SCR+ vs R-SCR+ occlusion", G("R-SCR+"), X("occlusion", "rscr")),
]
tests = []
for lab, ga, gb in COMP:
    for m in ("var", "var_later", "cross", "frac", "calls"):
        t = paired(pe(smd, ga, m), pe(smd, gb, m)); t.update(comp=lab, metric=m); tests.append(t)
adj = holm(np.array([t["p"] for t in tests]))
for t, a in zip(tests, adj):
    t["p_holm"] = float(a)
out["smd_tests"] = tests
print(f"\n== kiem dinh tren SMD ({len(tests)} kiem dinh, Holm)")
for t in tests:
    if t["metric"] != "frac" or "ordering" in t["comp"] or "margin" in t["comp"]:
        c_ = t["ci"]
        f_ = lambda x: "None" if x is None else f"{x:+.3f}"
        print(f"  {t['comp']:38s} {t['metric']:9s} {f_(c_[0])} [{f_(c_[1])};{f_(c_[2])}] pH={t['p_holm']:.2g} {t['hi']}/{t['lo']} n={t['n']}")

xt = []
for o in ORD[1:]:
    for m in ("var", "cross"):
        t = paired(pe(smd, X(o, "rscr"), m), pe(smd, X("deviation", "rscr"), m)); t.update(ordering=o, metric=m); xt.append(t)
adj = holm(np.array([t["p"] for t in xt]))
for t, a in zip(xt, adj):
    t["p_holm"] = float(a)
out["xai_tests"] = xt
print("\nthu tu kenh tru do lech (R-SCR+, SMD):", [(t["ordering"], t["metric"], round(t["ci"][0], 3), round(t["p_holm"], 3)) for t in xt])

# BetaRCE ma goc
off = {}
for b in ("smd", "psm"):
    for f in glob.glob(f"{O}/{b}_betarce/*.json"):
        for r in json.load(open(f))["records"]:
            off[(b, r["entity"], r["gen"], r["win"])] = r
if off:
    OFFN = ["BetaRCE-official default", "BetaRCE-official matched"]
    rows, otests = {}, {}
    out["official"] = {}
    for b in ("smd", "psm"):
        main_by = {(b, r["entity"], r["gen"], r["win"]): r for r in recs[b]}
        keys = [k for k in off if k[0] == b and k in main_by]
        if not keys:
            continue
        sub = [main_by[k] for k in keys]
        mk = lambda nm: (lambda r, b=b: off[(b, r["entity"], r["gen"], r["win"])]["gen_res"][nm])
        rws = {nm: {m: ci(pe(sub, mk(nm) if nm != "R-SCR+" else G("R-SCR+"), m).values())
                    for m in ("found", "var", "cross", "frac", "calls")} for nm in OFFN + ["R-SCR+"]}
        ts = []
        if b == "smd":
            for nm in OFFN:
                for m in ("var", "cross", "calls"):
                    t = paired(pe(sub, G("R-SCR+"), m), pe(sub, mk(nm), m)); t.update(vs=nm, metric=m); ts.append(t)
            adj = holm(np.array([t["p"] for t in ts]))
            for t, a in zip(ts, adj):
                t["p_holm"] = float(a)
        fails = {nm: float(np.mean([1 - off[k]["gen_res"][nm]["found"] for k in keys])) for nm in OFFN}
        fcalls = [off[k]["gen_res"][OFFN[1]]["calls"] for k in keys if not off[k]["gen_res"][OFFN[1]]["found"]]
        moved = [off[k]["gen_res"][OFFN[1]]["moved"] for k in keys if off[k]["gen_res"][OFFN[1]]["found"]]
        out["official"][b] = dict(n_cases=len(keys), n_entities=len({k[1] for k in keys}), rows=rws, tests=ts, fail=fails,
                                  median_calls_failed=float(np.median(fcalls)) if fcalls else 0.0,
                                  moved_rate=float(np.mean(moved)) if moved else 0.0)
        print(f"\n== BetaRCE ma goc {b}: {len(keys)} ca", {nm: {m: None if v[0] is None else round(v[0], 3) for m, v in r.items()} for nm, r in rws.items()})
        print("   that bai", fails, "goi khi that bai", out["official"][b]["median_calls_failed"], "giu nguyen ban goc", out["official"][b]["moved_rate"])
        for t in ts:
            print("  ", t["vs"], t["metric"], [None if x is None else round(x, 3) for x in t["ci"]], round(t["p_holm"], 4), t["hi"], t["lo"], t["n"])
json.dump(out, open(f"{O}/summary_v3.json", "w"), indent=1, default=float)
