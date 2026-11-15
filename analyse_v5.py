"""Phan tich ban 5 (ban tap chi): giao thuc mo rong voi RobX, T-Rex, ROAR va kiem tra cheo kien truc.
Doc out/{smd,asd,psm}_v5/*.json. Ghi out/summary_v5.json, journal/numbers_v5.tex, journal/tab_v5.tex, journal/tab_v5_tests.tex.
Moi do do la ty le ban sua con hop le duoi mot nhom bo cap nhat, chi tinh tren cua so ma bo do cung bao dong,
trung binh theo cua so trong thuc the roi trung binh qua thuc the."""
import os, glob, json
import numpy as np
from scipy.stats import wilcoxon
HOME = os.environ.get("RSCR_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
O = HOME + "/out"; J = HOME + "/journal"
GEN = [("SCR dev", "SCR"), ("Wachter", "Wachter"), ("Wachter margin", "Wachter + margin"), ("ROAR", "ROAR"),
       ("RobX", "RobX"), ("RobX strict", "RobX, $\\tau_R=0.75$"), ("T-Rex", "T-Rex"), ("T-Rex strict", "T-Rex, $\\tau_R=0.75$"),
       ("m-SCR dev", "m-SCR"), ("BetaRCE-T dev", "BetaRCE-T"), ("R-SCR+", "R-SCR+")]
EV = {"seed": [f"same_s{i}" for i in range(5)], "full": [f"upd_s{i}" for i in range(5)],
      "chain": ["chain1", "chain2", "chain3"], "cross": ["x_ocsvm", "x_lof", "x_ecod"]}
rng = np.random.default_rng(0)


def load(ds):
    R = []
    for f in sorted(glob.glob(f"{O}/{ds}_v5/*.json")):
        R += json.load(open(f))["records"]
    return R


def wscore(rr, keys):
    v = [rr[k] for k in keys if rr.get(k) is not None]
    return np.mean(v) if v else None


def per_entity(R, g, fn):
    out = {}
    for e in sorted({r["entity"] for r in R}):
        v = [fn(r["res"][g]) for r in R if r["entity"] == e and g in r["res"]]
        v = [x for x in v if x is not None]
        if v:
            out[e] = float(np.mean(v))
    return out


def metric_fns():
    return {"found": lambda rr: rr["found"], "calls": lambda rr: rr["calls"],
            "size": lambda rr: rr["frac"] if rr["frac"] is not None else None,
            **{m: (lambda keys: (lambda rr: wscore(rr, keys)))(keys) for m, keys in EV.items()}}


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); out = np.empty_like(ps); run = 0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); out[j] = min(1, run)
    return out


S = {}
F = metric_fns()
for ds in ("smd", "asd", "psm"):
    R = load(ds)
    if not R:
        continue
    S[ds] = {"n_records": len(R), "entities": len({r["entity"] for r in R}), "table": {}, "fam": {}}
    for g, _ in GEN:
        if not any(g in r["res"] for r in R):
            continue
        S[ds]["table"][g] = {m: (lambda d: float(np.mean(list(d.values()))) if d else None)(per_entity(R, g, fn)) for m, fn in F.items()}
        for fam in ("pca", "iforest", "ae"):
            RR = [r for r in R if r["gen"] == fam]
            S[ds]["fam"].setdefault(fam, {})[g] = {m: (lambda d: float(np.mean(list(d.values()))) if d else None)(per_entity(RR, g, fn))
                                                    for m, fn in F.items() if m in ("full", "chain", "seed", "cross", "size", "calls")}
    if ds in ("smd", "asd"):
        tests = []
        for g, lab in GEN:
            if g == "R-SCR+" or g not in S[ds]["table"]:
                continue
            for m in ("full", "chain", "cross", "calls"):
                a, b = per_entity(R, "R-SCR+", F[m]), per_entity(R, g, F[m])
                ks = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in ks])
                p = float(wilcoxon(d, zero_method="zsplit").pvalue) if len(d) > 1 and np.any(d != 0) else 1.0
                tests.append(dict(gen=g, metric=m, diff=float(d.mean()), up=int((d > 0).sum()), down=int((d < 0).sum()), n=len(d), p=p))
        ph = holm([t["p"] for t in tests])
        for t, q in zip(tests, ph):
            t["p_holm"] = float(q)
        S[ds]["tests"] = tests
json.dump(S, open(f"{O}/summary_v5.json", "w"), indent=1)

# ---------------------------------------------------------------- macro
T3 = {"SCR dev": "SCR", "Wachter": "WA", "Wachter margin": "WAM", "ROAR": "ROAR", "RobX": "RX", "RobX strict": "RXS",
      "T-Rex": "TR", "T-Rex strict": "TRS", "m-SCR dev": "MSCR", "BetaRCE-T dev": "BT", "R-SCR+": "RS"}
MT = {"found": "FND", "seed": "SEED", "full": "FULL", "chain": "CHAIN", "cross": "CROSS", "size": "SIZE", "calls": "CALLS"}
DT = {"smd": "S", "asd": "A", "psm": "P"}


def fmt(m, v):
    if v is None:
        return "--"
    if m == "calls":
        return "%.0f" % v
    return "%.3f" % v


M = {}
for ds, dt in DT.items():
    if ds not in S:
        continue
    for g, gt in T3.items():
        if g in S[ds]["table"]:
            for m, mt in MT.items():
                M[f"Vfive{dt}{gt}{mt}"] = fmt(m, S[ds]["table"][g][m])
    for t in S[ds].get("tests", []):
        k = f"Vfive{dt}T{T3[t['gen']]}{MT[t['metric']]}"
        M[k + "D"] = ("%+.0f" if t["metric"] == "calls" else "%+.3f") % t["diff"]
        M[k + "UP"] = str(t["up"]); M[k + "DN"] = str(t["down"]); M[k + "N"] = str(t["n"])
        p = t["p_holm"]
        M[k + "P"] = "%.3f" % p if p >= 0.001 else "\\ensuremath{%.1f\\times10^{%d}}" % (p / 10 ** np.floor(np.log10(p)), np.floor(np.log10(p)))
    M[f"Vfive{dt}NREC"] = str(S[ds]["n_records"]); M[f"Vfive{dt}NENT"] = str(S[ds]["entities"])
os.makedirs(J, exist_ok=True)
open(f"{J}/numbers_v5.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")

# ---------------------------------------------------------------- bang chinh: SMD va ASD
cols = [("seed", "Seed"), ("full", "Full"), ("chain", "Chain"), ("cross", "Cross")]
L = ["\\begin{table}[!htb]\\centering",
     "\\caption{Validity of repairs under seed-only, full-history and chained refits and on three other detector families, "
     "with share of windows repaired, repair size and detector calls per window, averaged over machines of SMD and nodes of ASD and pooled over three running detectors.}",
     "\\label{tab:v5}\\footnotesize\\setlength{\\tabcolsep}{2.6pt}",
     "\\begin{tabular}{@{}l" + "ccccccc" * 2 + "@{}}\\toprule",
     " & \\multicolumn{7}{c}{SMD} & \\multicolumn{7}{c}{ASD} \\\\\\cmidrule(lr){2-8}\\cmidrule(lr){9-15}",
     "Generator & " + " & ".join(["Found"] + [c for _, c in cols] + ["Size", "Calls"]) * 1 + " & " + " & ".join(["Found"] + [c for _, c in cols] + ["Size", "Calls"]) + " \\\\\\midrule"]
for g, lab in GEN:
    cells = []
    for ds in ("smd", "asd"):
        t = S.get(ds, {}).get("table", {}).get(g)
        if t is None:
            cells += ["--"] * 7; continue
        cells += [fmt("x", t["found"])] + [fmt(m, t[m]) for m, _ in cols] + [fmt("size", t["size"]), fmt("calls", t["calls"])]
    L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(f"{J}/tab_v5.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- bang kiem dinh SMD
L = ["\\begin{table}[!htb]\\centering",
     "\\caption{Paired differences between R-SCR+ and each baseline over the SMD machines that raise alarms in the evaluation period of the extended protocol, "
     "with Holm-corrected Wilcoxon p values and the number of machines on which R-SCR+ is higher or lower.}",
     "\\label{tab:v5tests}\\footnotesize\\setlength{\\tabcolsep}{3pt}",
     "\\begin{tabular}{@{}lcccccccc@{}}\\toprule",
     " & \\multicolumn{2}{c}{Full history} & \\multicolumn{2}{c}{Chain} & \\multicolumn{2}{c}{Cross-arch.} & \\multicolumn{2}{c}{Calls} \\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
     "Baseline & Diff. & $p_{\\text{Holm}}$ & Diff. & $p_{\\text{Holm}}$ & Diff. & $p_{\\text{Holm}}$ & Diff. & $p_{\\text{Holm}}$ \\\\\\midrule"]
TT = {(t["gen"], t["metric"]): t for t in S.get("smd", {}).get("tests", [])}
for g, lab in GEN:
    if g == "R-SCR+" or (g, "full") not in TT:
        continue
    cells = []
    for m in ("full", "chain", "cross", "calls"):
        t = TT[(g, m)]
        cells += [("%+.0f" if m == "calls" else "%+.3f") % t["diff"], M[f"VfiveST{T3[g]}{MT[m]}P"]]
    L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(f"{J}/tab_v5_tests.tex", "w").write("\n".join(L) + "\n")
for ds in S:
    print(ds, S[ds]["n_records"], S[ds]["entities"])
    for g, _ in GEN:
        t = S[ds]["table"].get(g)
        if t:
            print("  %-15s" % g, " ".join("%s=%s" % (m, fmt(m, t[m])) for m in MT))

# ---------------------------------------------------------------- bang theo ho bo phat hien (SMD)
FG = [("SCR dev", "SCR"), ("m-SCR dev", "m-SCR"), ("BetaRCE-T dev", "BetaRCE-T"), ("RobX strict", "RobX, $\\tau_R=0.75$"), ("R-SCR+", "R-SCR+")]
L = ["\\begin{table}[!htb]\\centering",
     "\\caption{Validity of repairs on SMD in the extended protocol separated by running detector, under seed-only, full-history and chained refits, "
     "where seed-only refits are omitted for principal components because they reproduce the running detector exactly.}",
     "\\label{tab:fam}\\footnotesize\\setlength{\\tabcolsep}{3pt}",
     "\\begin{tabular}{@{}lccccccccc@{}}\\toprule",
     " & \\multicolumn{3}{c}{Principal components} & \\multicolumn{3}{c}{Isolation Forest} & \\multicolumn{3}{c}{Autoencoder} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
     "Generator & Seed & Full & Chain & Seed & Full & Chain & Seed & Full & Chain \\\\\\midrule"]
for g, lab in FG:
    c = []
    for fam in ("pca", "iforest", "ae"):
        t = S["smd"]["fam"][fam][g]
        c += ["n/a" if fam == "pca" else fmt("x", t["seed"]), fmt("x", t["full"]), fmt("x", t["chain"])]
    L.append(f"{lab} & " + " & ".join(c) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(f"{J}/tab_fam.tex", "w").write("\n".join(L) + "\n")
