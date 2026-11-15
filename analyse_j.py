"""Bang va macro cho ban tap chi: giao thuc chinh tren SMD, ASD, PSM; kiem dinh xac nhan tren ASD; bo phat hien cua so tho.
Doc out/{smd,asd,psm}/*.json, out/{smd,asd}_raw/*.json va out/{asd}_betarce. Ghi journal/numbers_j.tex,
journal/tab_main_j.tex, journal/tab_asd_tests.tex, journal/tab_raw.tex va out/summary_j.json."""
import os, glob, json
import numpy as np
from scipy.stats import wilcoxon, bootstrap
HOME = os.environ.get("RSCR_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
O = HOME + "/out"; J = HOME + "/journal"
HELD = ["ocsvm", "lof", "ecod"]
GEN = [("Wachter-line", "Wachter, line search"), ("CoMTE", "CoMTE variant"), ("p-SCR", "p-SCR"), ("SCR", "SCR"),
       ("NG-variant", "Native-Guide variant"), ("m-SCR", "m-SCR"), ("BetaRCE-B", "BetaRCE-B"), ("BetaRCE-T", "BetaRCE-T"),
       ("R-SCR+", "R-SCR+")]
TAG = {"Wachter-line": "WAC", "CoMTE": "CMT", "p-SCR": "PSCR", "SCR": "SCR", "NG-variant": "NG", "m-SCR": "MSCR",
       "m-SCR dev": "MSD", "BetaRCE-B": "BB", "BetaRCE-T": "BT", "R-SCR+": "RSP"}


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
    if m == "frac":
        return [v[m]] if v[m] is not None else []
    return [v[m]]


def pe(recs, g, m):
    acc = {}
    for r in recs:
        acc.setdefault(r["entity"], []).extend(metric(r["gen_res"][g], m))
    return {e: float(np.mean(x)) for e, x in acc.items() if x}


def mean(d):
    return float(np.mean(list(d.values()))) if d else None


def holm(ps):
    o = np.argsort(ps); m = len(ps); a = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); a[j] = min(1, run)
    return a


def ptex(p):
    return "%.3f" % p if p >= 0.001 else "\\ensuremath{%.1f\\times10^{%d}}" % (p / 10 ** np.floor(np.log10(p)), np.floor(np.log10(p)))


S, M = {}, {}
for ds in ("smd", "asd", "psm"):
    R = load(ds)
    if not R:
        continue
    S[ds] = {g: {m: mean(pe(R, g, m)) for m in ("var", "var_later", "cross", "frac", "calls", "found")} for g, _ in GEN + [("m-SCR dev", "")]}
    S[ds]["_n"] = len(R); S[ds]["_e"] = len({r["entity"] for r in R})
    dt = {"smd": "S", "asd": "A", "psm": "P"}[ds]
    M[f"J{dt}NWIN"] = str(len(R)); M[f"J{dt}NENT"] = str(S[ds]["_e"])
    for g, t in TAG.items():
        for m, mt in (("var", "VAR"), ("var_later", "LAT"), ("cross", "CR"), ("frac", "SIZE"), ("calls", "CALL")):
            v = S[ds][g][m]
            if v is not None:
                M[f"J{dt}{t}{mt}"] = "%.0f" % v if m == "calls" else "%.3f" % v
                if m != "calls":
                    M[f"J{dt}{t}{mt}PC"] = "%.1f\\%%" % (100 * v)
    # kiem dinh xac nhan tren ASD (va SMD de doi chieu)
    if ds in ("asd", "smd"):
        tests = []
        for b in ("BetaRCE-T", "BetaRCE-B", "m-SCR dev", "SCR"):
            for m in ("var", "var_later", "cross", "calls"):
                a_, b_ = pe(R, "R-SCR+", m), pe(R, b, m)
                ks = sorted(set(a_) & set(b_)); d = np.array([a_[k] - b_[k] for k in ks])
                p = float(wilcoxon(d, zero_method="zsplit").pvalue) if len(d) > 1 and np.any(d != 0) else 1.0
                tests.append(dict(base=b, metric=m, diff=float(d.mean()), up=int((d > 0).sum()), down=int((d < 0).sum()), n=len(d), p=p))
        for t, q in zip(tests, holm(np.array([t["p"] for t in tests]))):
            t["p_holm"] = float(q)
            k = f"J{dt}T{TAG[t['base']]}{dict(var='VAR', var_later='LAT', cross='CR', calls='CALL')[t['metric']]}"
            M[k + "D"] = ("%+.1f" if t["metric"] == "calls" else "%+.3f") % t["diff"]; M[k + "P"] = ptex(t["p_holm"]); M[k + "ABS"] = ("%.1f" if t["metric"] == "calls" else "%.3f") % abs(t["diff"])
            M[k + "UP"] = str(t["up"]); M[k + "DN"] = str(t["down"])
        S[ds]["_tests"] = tests

# BetaRCE ma goc tren ASD
for ds in ("asd",):
    R = load(f"{ds}_betarce")
    if R:
        keys = list(R[0].keys())
        S[ds + "_betarce_keys"] = keys

# ---------------------------------------------------------------- bang giao thuc chinh
L = ["\\begin{table}[!htb]\\centering",
     "\\caption{Every returned repair clears the running detector, while validity after full-history (VaR) and later-period (VaR-L) "
     "retraining and cross-architecture validity vary widely across the three benchmarks, with size and calls given for SMD.}",
     "\\label{tab:main}\\footnotesize\\setlength{\\tabcolsep}{2.4pt}",
     "\\begin{tabular}{@{}lccccccccccc@{}}\\toprule",
     " & \\multicolumn{3}{c}{VaR} & \\multicolumn{3}{c}{VaR-L} & \\multicolumn{3}{c}{Cross-arch.} & Size & Calls \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-11}\\cmidrule(lr){12-12}",
     "Generator & SMD & ASD & PSM & SMD & ASD & PSM & SMD & ASD & PSM & SMD & SMD \\\\\\midrule"]
for g, lab in GEN:
    c = []
    for m in ("var", "var_later", "cross"):
        for ds in ("smd", "asd", "psm"):
            v = S.get(ds, {}).get(g, {}).get(m)
            c.append("--" if v is None else "%.3f" % v)
    c.append("%.3f" % S["smd"][g]["frac"]); c.append("%.0f" % S["smd"][g]["calls"])
    L.append(f"{lab} & " + " & ".join(c) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(f"{J}/tab_main_j.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- bang kiem dinh ASD
if "asd" in S:
    L = ["\\begin{table}[!htb]\\centering",
         "\\caption{Paired differences between R-SCR+ and four baselines over the 12 nodes of ASD in the main protocol, "
         "with Holm-corrected Wilcoxon p values over the sixteen tests and the number of nodes on which R-SCR+ is higher or lower.}",
         "\\label{tab:asdtests}\\footnotesize\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}lcccccccc@{}}\\toprule",
         " & \\multicolumn{2}{c}{VaR} & \\multicolumn{2}{c}{VaR-L} & \\multicolumn{2}{c}{Cross-arch.} & \\multicolumn{2}{c}{Calls} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
         "Baseline & Diff. (H/L) & $p_{\\text{Holm}}$ & Diff. (H/L) & $p_{\\text{Holm}}$ & Diff. (H/L) & $p_{\\text{Holm}}$ & Diff. (H/L) & $p_{\\text{Holm}}$ \\\\\\midrule"]
    TT = {(t["base"], t["metric"]): t for t in S["asd"]["_tests"]}
    for b, lab in (("BetaRCE-T", "BetaRCE-T"), ("BetaRCE-B", "BetaRCE-B"), ("m-SCR dev", "m-SCR, deviation"), ("SCR", "SCR")):
        c = []
        for m in ("var", "var_later", "cross", "calls"):
            t = TT[(b, m)]
            c += [(("%+.1f" if m == "calls" else "%+.3f") % t["diff"]) + f" ({t['up']}/{t['down']})", ptex(t["p_holm"])]
        L.append(f"{lab} & " + " & ".join(c) + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(f"{J}/tab_asd_tests.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- bo phat hien cua so tho
RAWG = [("SCR", "SCR"), ("m-SCR", "m-SCR"), ("BetaRCE-T", "BetaRCE-T"), ("R-SCR+", "R-SCR+")]
RAWE = {"full": ["upd_s0"], "later": ["later"], "seed": ["same_s0", "same_s1", "same_s2"], "chain": ["chain1", "chain2", "chain3"],
        "cross": ["x_rawpca", "x_rawae"]}


def rawpe(R, g, keys):
    acc = {}
    for r in R:
        rr = r["res"][g]
        v = [rr[k] for k in keys if rr.get(k) is not None]
        if v:
            acc.setdefault(r["entity"], []).append(np.mean(v))
    return {e: float(np.mean(x)) for e, x in acc.items()}


RAW = {}
for ds in ("smd", "asd"):
    R = load(f"{ds}_raw")
    if not R:
        continue
    RAW[ds] = {"_n": len(R), "_e": len({r["entity"] for r in R})}
    for g, _ in RAWG:
        RAW[ds][g] = {m: mean(rawpe(R, g, ks)) for m, ks in RAWE.items()}
        RAW[ds][g]["size"] = mean({e: float(np.mean([r["res"][g]["frac"] for r in R if r["entity"] == e and r["res"][g]["frac"] is not None] or [np.nan])) for e in {r["entity"] for r in R}})
        RAW[ds][g]["calls"] = mean({e: float(np.mean([r["res"][g]["calls"] for r in R if r["entity"] == e])) for e in {r["entity"] for r in R}})
        RAW[ds][g]["found"] = mean({e: float(np.mean([r["res"][g]["found"] for r in R if r["entity"] == e])) for e in {r["entity"] for r in R}})
    dt = {"smd": "S", "asd": "A"}[ds]
    M[f"Raw{dt}NWIN"] = str(len(R)); M[f"Raw{dt}NENT"] = str(RAW[ds]["_e"])
    for g, t in (("SCR", "SCR"), ("m-SCR", "MSCR"), ("BetaRCE-T", "BT"), ("R-SCR+", "RS")):
        for m in ("full", "later", "seed", "chain", "cross", "size", "calls", "found"):
            v = RAW[ds][g][m]
            M[f"Raw{dt}{t}{m.upper()}"] = "--" if v is None or np.isnan(v) else ("%.0f" % v if m == "calls" else "%.3f" % v)
    if ds == "smd":
        tests = []
        for b in ("BetaRCE-T", "m-SCR", "SCR"):
            for m in ("full", "chain"):
                a_, b_ = rawpe(R, "R-SCR+", RAWE[m]), rawpe(R, b, RAWE[m])
                ks = sorted(set(a_) & set(b_)); d = np.array([a_[k] - b_[k] for k in ks])
                p = float(wilcoxon(d, zero_method="zsplit").pvalue) if len(d) > 1 and np.any(d != 0) else 1.0
                tests.append(dict(base=b, metric=m, diff=float(d.mean()), up=int((d > 0).sum()), down=int((d < 0).sum()), p=p))
        for t, q in zip(tests, holm(np.array([t["p"] for t in tests]))):
            k = f"RawST{dict(zip(('BetaRCE-T','m-SCR','SCR'),('BT','MSCR','SCR')))[t['base']]}{t['metric'].upper()}"
            M[k + "D"] = "%+.3f" % t["diff"]; M[k + "P"] = ptex(q); M[k + "UP"] = str(t["up"]); M[k + "DN"] = str(t["down"])
        RAW[ds]["_tests"] = tests
if RAW:
    L = ["\\begin{table}[!htb]\\centering",
         "\\caption{Validity of repairs for detectors that read raw windows, pooled over principal component and autoencoder detectors, "
         "under one full-history, one later-period, three seed-only and three chained refits and on the other raw family, with size and calls per window.}",
         "\\label{tab:raw}\\footnotesize\\setlength{\\tabcolsep}{2.6pt}",
         "\\begin{tabular}{@{}l" + "ccccccc" * len(RAW) + "@{}}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{7}}{{c}}{{{ds.upper()}}}" for ds in RAW) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 7 * i}-{8 + 7 * i}}}" for i in range(len(RAW))),
         "Generator & " + " & ".join(["Full", "Later", "Seed", "Chain", "Cross", "Size", "Calls"] * len(RAW)) + " \\\\\\midrule"]
    for g, lab in RAWG:
        c = []
        for ds in RAW:
            for m in ("full", "later", "seed", "chain", "cross", "size", "calls"):
                v = RAW[ds][g][m]
                c.append("--" if v is None or np.isnan(v) else ("%.0f" % v if m == "calls" else "%.3f" % v))
        L.append(f"{lab} & " + " & ".join(c) + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(f"{J}/tab_raw.tex", "w").write("\n".join(L) + "\n")
S["raw"] = RAW
json.dump(S, open(f"{O}/summary_j.json", "w"), indent=1, default=float)
open(f"{J}/numbers_j.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
for ds in ("smd", "asd", "psm"):
    if ds in S:
        print(ds, S[ds]["_n"], S[ds]["_e"])
        for g, _ in GEN:
            t = S[ds][g]; print("  %-14s var %.3f lat %.3f cross %.3f size %.3f calls %.0f" % (g, t["var"] or -1, t["var_later"] or -1, t["cross"] or -1, t["frac"] or -1, t["calls"]))
for ds, d in RAW.items():
    print("raw", ds, d["_n"], {g: {m: (round(v, 3) if v is not None else None) for m, v in d[g].items()} for g, _ in RAWG})

# ---------------------------------------------------------------- bang cua so tho theo tung bo phat hien, hop le co dieu kien
def rawcond(R, g, keys):
    acc = {}
    for r in R:
        rr = r["res"][g]
        if not rr["found"]:
            continue
        v = [rr[k] for k in keys if rr.get(k) is not None]
        if v:
            acc.setdefault(r["entity"], []).append(np.mean(v))
    return {e: float(np.mean(x)) for e, x in acc.items()}


RC = {}
for ds in ("smd", "asd"):
    R = load(f"{ds}_raw")
    for kind, kt in (("rawpca", "PCA"), ("rawae", "AE")):
        RR = [r for r in R if r["gen"] == kind]
        dt = {"smd": "S", "asd": "A"}[ds]
        M[f"Raw{dt}{kt}NWIN"] = str(len(RR))
        M[f"Raw{dt}{kt}PREC"] = "%.2f" % np.mean([r["prec"] for r in RR])
        for g, t in (("SCR", "SCR"), ("m-SCR", "MSCR"), ("BetaRCE-T", "BT"), ("R-SCR+", "RS")):
            row = {"found": mean({e: float(np.mean([r["res"][g]["found"] for r in RR if r["entity"] == e])) for e in {r["entity"] for r in RR}}),
                   "calls": mean({e: float(np.mean([r["res"][g]["calls"] for r in RR if r["entity"] == e])) for e in {r["entity"] for r in RR}})}
            for m in ("full", "later", "seed", "chain", "cross"):
                row[m] = mean(rawpe(RR, g, RAWE[m]))          # Eq. (3): ban sua thieu tinh la khong hop le
                row["c" + m] = mean(rawcond(RR, g, RAWE[m]))  # chi tren ban sua duoc tra ve
            RC.setdefault(ds, {}).setdefault(kind, {})[g] = row
            for m, v in row.items():
                M[f"Raw{dt}{kt}{t}{m.upper()}"] = "--" if v is None or np.isnan(v) else ("%.0f" % v if m == "calls" else "%.3f" % v)
L = ["\\begin{table}[!htb]\\centering",
     "\\caption{Share of SMD windows repaired and validity as defined in Eq.~(3), with a missing repair counted as invalid, for detectors that read raw windows "
     "under one full-history, one later-period, three seed-only and three chained refits and on the other raw family, with detector calls per window.}",
     "\\label{tab:raw}\\footnotesize\\setlength{\\tabcolsep}{2.8pt}",
     "\\begin{tabular}{@{}lccccccccccccc@{}}\\toprule",
     " & \\multicolumn{6}{c}{Raw principal components} & \\multicolumn{7}{c}{Raw autoencoder} \\\\",
     "\\cmidrule(lr){2-7}\\cmidrule(lr){8-14}",
     "Generator & Found & Full & Later & Chain & Cross & Calls & Found & Full & Later & Seed & Chain & Cross & Calls \\\\\\midrule"]
f3 = lambda v: "--" if v is None or np.isnan(v) else "%.3f" % v
for g, lab in RAWG:
    a, b = RC["smd"]["rawpca"][g], RC["smd"]["rawae"][g]
    c = [f3(a[m]) for m in ("found", "full", "later", "chain", "cross")] + ["%.0f" % a["calls"]]
    c += [f3(b[m]) for m in ("found", "full", "later", "seed", "chain", "cross")] + ["%.0f" % b["calls"]]
    L.append(f"{lab} & " + " & ".join(c) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(f"{J}/tab_raw.tex", "w").write("\n".join(L) + "\n")
open(f"{J}/numbers_j.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
for ds in RC:
    for k in RC[ds]:
        print(ds, k, {g: {m: (round(v, 3) if v is not None else None) for m, v in r.items()} for g, r in RC[ds][k].items()})
