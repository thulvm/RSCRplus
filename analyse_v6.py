"""Phan tich ban 6 (sua theo phan bien). Doc out/{smd,asd,psm}_v6. Ghi out/summary_v6.json, journal/numbers_v6.tex,
journal/tab_pareto.tex (cua so bat thuong va bao dong gia tren SMD) va journal/tab_v6tests.tex."""
import os, glob, json
import numpy as np
from scipy.stats import wilcoxon
HOME = RSCR_HOME + ""; O = HOME + "/out"; J = HOME + "/journal"
EV = {"seed": [f"same_s{i}" for i in range(5)], "full": [f"upd_s{i}" for i in range(5)],
      "chain": ["chain1", "chain2", "chain3"], "cross": ["x_ocsvm", "x_lof", "x_ecod"]}
GEN = [("SCR dev", "SCR", "SCR"), ("m-SCR 0.25", "m-SCR, $\\delta=0.25$", "MA"), ("m-SCR 0.35", "m-SCR, $\\delta=0.35$", "MB"),
       ("m-SCR 0.5", "m-SCR, $\\delta=0.5$", "MC"), ("m-SCR 0.75", "m-SCR, $\\delta=0.75$", "MD"),
       ("BetaRCE-T dev", "BetaRCE-T", "BT"), ("RobX strict", "RobX, $\\tau_R=0.75$", "RXS"),
       ("R-SCR+ 0.25", "\\textbf{R-SCR+}, $\\delta=0.25$", "RA"), ("R-SCR+ 0.5", "\\textbf{R-SCR+}, $\\delta=0.5$", "RB")]


def load(ds):
    R, N = [], []
    for f in sorted(glob.glob(f"{O}/{ds}_v6/*.json")):
        d = json.load(open(f)); R += d["records"]; N += [dict(n, entity=d["entity"]) for n in d["normals"]]
    # RobX strict cho cua so bat thuong lay tu ban 5 (cung cua so)
    v5 = {}
    for f in glob.glob(f"{O}/{ds}_v5/*.json"):
        for r in json.load(open(f))["records"]:
            v5[(r["entity"], r["gen"], r["win"])] = r["res"].get("RobX strict")
    for r in R:
        if r["kind"] == "tp" and "RobX strict" not in r["res"]:
            x = v5.get((r["entity"], r["gen"], r["win"]))
            if x is not None:
                r["res"]["RobX strict"] = dict(x, plaus=None)
    return R, N


def wscore(rr, keys):
    v = [rr[k] for k in keys if rr.get(k) is not None]
    return float(np.mean(v)) if v else None


FN = {**{m: (lambda ks: (lambda rr: wscore(rr, ks)))(ks) for m, ks in EV.items()},
      "found": lambda rr: rr["found"], "calls": lambda rr: rr["calls"],
      "size": lambda rr: rr["frac"], "plaus": lambda rr: rr.get("plaus")}


def pe(R, g, fn):
    acc = {}
    for r in R:
        if g in r["res"]:
            v = fn(r["res"][g])
            if v is not None:
                acc.setdefault(r["entity"], []).append(v)
    return {e: float(np.mean(x)) for e, x in acc.items()}


def mean(d):
    return float(np.mean(list(d.values()))) if d else None


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); out = np.empty_like(ps); run = 0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); out[j] = min(1, run)
    return out


S = {}; M = {}
for ds in ("smd", "asd", "psm"):
    R, NOR = load(ds)
    if not R:
        continue
    S[ds] = {}
    for kind in ("tp", "fa"):
        RR = [r for r in R if r["kind"] == kind]
        T = {g: {m: mean(pe(RR, g, fn)) for m, fn in FN.items()} for g, _, _ in GEN}
        S[ds][kind] = dict(n=len(RR), ent=len({r["entity"] for r in RR}), table=T)
        # phan phoi kich thuoc
        for g, _, _ in GEN:
            fr = np.array([r["res"][g]["frac"] for r in RR if g in r["res"] and r["res"][g]["frac"] is not None])
            if len(fr):
                T[g]["size_med"] = float(np.median(fr)); T[g]["size_p90"] = float(np.percentile(fr, 90))
                T[g]["size_half"] = float((fr >= 0.5).mean())
        # kiem dinh: R-SCR+ voi m-SCR co kich thuoc gan nhat, va voi BetaRCE-T
        tests = []
        pairs = [("R-SCR+ 0.25", "m-SCR 0.35"), ("R-SCR+ 0.25", "m-SCR 0.5"), ("R-SCR+ 0.5", "m-SCR 0.75"), ("R-SCR+ 0.25", "BetaRCE-T dev")]
        for a, b in pairs:
            for m in ("full", "chain", "cross", "size", "calls"):
                A, B = pe(RR, a, FN[m]), pe(RR, b, FN[m]); ks = sorted(set(A) & set(B))
                d = np.array([A[k] - B[k] for k in ks])
                p = float(wilcoxon(d, zero_method="zsplit").pvalue) if len(d) > 1 and np.any(d != 0) else 1.0
                tests.append(dict(a=a, b=b, m=m, diff=float(d.mean()) if len(d) else 0.0, up=int((d > 0).sum()), dn=int((d < 0).sum()), n=len(d), p=p))
        for t, q in zip(tests, holm([t["p"] for t in tests])):
            t["ph"] = float(q)
        S[ds][kind]["tests"] = tests
    # tuong quan giua cac ban sao (phieu tren moi ban sua co ghi phieu)
    V = np.array([r["res"][g]["votes"] for r in R for g in ("SCR dev", "m-SCR 0.25", "R-SCR+ 0.25")
                  if g in r["res"] and r["res"][g].get("votes") is not None], float)
    rho = None
    if len(V) > 10:
        sd = V.std(0); keep = sd > 0
        C = np.corrcoef(V[:, keep].T); iu = np.triu_indices(C.shape[0], 1); rho = float(np.nanmean(C[iu]))
    S[ds]["rho"] = rho; S[ds]["neff"] = (40 / (1 + 39 * rho)) if rho is not None else None
    # ty le ban sao chap nhan (m-SCR 0.25) va do ben thuc te
    cal = []
    for lo, hi in ((0, 0.7), (0.7, 0.9), (0.9, 0.999), (0.999, 1.01)):
        xs = [(wscore(r["res"]["m-SCR 0.25"], EV["full"]), wscore(r["res"]["m-SCR 0.25"], EV["chain"])) for r in R
              if r["kind"] == "tp" and r["res"]["m-SCR 0.25"].get("votes") and lo <= np.mean(r["res"]["m-SCR 0.25"]["votes"]) < hi]
        f = [x[0] for x in xs if x[0] is not None]; ch = [x[1] for x in xs if x[1] is not None]
        cal.append(dict(lo=lo, hi=hi, n=len(xs), full=(float(np.mean(f)) if f else None), chain=(float(np.mean(ch)) if ch else None)))
    S[ds]["votecal"] = cal
    # co che: do dich z o cung muc ban dau, ban sua (SCR, m-SCR 0.25) so voi cua so binh thuong cua tap kiem tra
    zr = [(r["res"][g]["z_dep"], r["res"][g]["z_upd"]) for r in R if r["kind"] == "tp" for g in ("SCR dev", "m-SCR 0.25")
          if r["res"][g].get("z_dep") is not None]
    zn = [(a, b) for n in NOR for a, b in zip(n["z_dep"], n["z_upd"])]
    mech = []
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.0)):
        a = np.array([b - x for x, b in zr if lo <= x < hi]); b_ = np.array([b - x for x, b in zn if lo <= x < hi])
        mech.append(dict(lo=lo, hi=hi, nr=len(a), nn=len(b_),
                         r_drop=(float((a < -0.25).mean()) if len(a) else None), n_drop=(float((b_ < -0.25).mean()) if len(b_) else None),
                         r_inv=(float(((np.array([x for x, _ in zr if lo <= x < hi]) + a) < 0).mean()) if len(a) else None),
                         n_inv=(float(((np.array([x for x, _ in zn if lo <= x < hi]) + b_) < 0).mean()) if len(b_) else None)))
    S[ds]["mech"] = mech
json.dump(S, open(f"{O}/summary_v6.json", "w"), indent=1)

# ---------------------------------------------------------------- macro
f3 = lambda v: "n/a" if v is None else "%.3f" % v
for ds, dt in (("smd", "S"), ("asd", "A"), ("psm", "P")):
    if ds not in S:
        continue
    for kind, kt in (("tp", "T"), ("fa", "F")):
        X = S[ds][kind]
        M[f"Six{dt}{kt}N"] = str(X["n"]); M[f"Six{dt}{kt}E"] = str(X["ent"])
        for g, _, gt in GEN:
            for m, v in X["table"][g].items():
                M[f"Six{dt}{kt}{gt}{m.upper().replace('_', '').replace('P90', 'PN')}"] = ("%.0f" % v if m == "calls" and v is not None else
                                                                    ("%.2f" % v if m in ("size_med", "size_p90") and v is not None else "%.0f\\%%" % (100 * v) if m == "size_half" and v is not None else
                                                                     ("%.2f" % v if m == "plaus" and v is not None else f3(v))))
        for t in X["tests"]:
            ga = [x[2] for x in GEN if x[0] == t["a"]][0]; gb = [x[2] for x in GEN if x[0] == t["b"]][0]
            k = f"Six{dt}{kt}T{ga}{gb}{t['m'].upper()}"
            M[k + "D"] = ("%+.0f" if t["m"] == "calls" else "%+.3f") % t["diff"]
            M[k + "P"] = "%.3f" % t["ph"] if t["ph"] >= 0.001 else "\\ensuremath{%.1f\\times10^{%d}}" % (t["ph"] / 10 ** np.floor(np.log10(t["ph"])), np.floor(np.log10(t["ph"])))
            M[k + "UP"] = str(t["up"]); M[k + "DN"] = str(t["dn"]); M[k + "N"] = str(t["n"])
    if S[ds]["rho"] is not None:
        M[f"Six{dt}RHO"] = "%.2f" % S[ds]["rho"]; M[f"Six{dt}NEFF"] = "%.1f" % S[ds]["neff"]
    for i, c in enumerate(S[ds]["votecal"]):
        M[f"Six{dt}VC{'abcd'[i]}N"] = str(c["n"]); M[f"Six{dt}VC{'abcd'[i]}FULL"] = f3(c["full"]); M[f"Six{dt}VC{'abcd'[i]}CHAIN"] = f3(c["chain"])
    for i, c in enumerate(S[ds]["mech"]):
        for k in ("r_drop", "n_drop", "r_inv", "n_inv"):
            M[f"Six{dt}MB{'abc'[i]}{k.replace('_', '').upper()}"] = "n/a" if c[k] is None else "%.1f\\%%" % (100 * c[k])
        M[f"Six{dt}MB{'abc'[i]}NR"] = str(c["nr"]); M[f"Six{dt}MB{'abc'[i]}NN"] = str(c["nn"])
open(f"{J}/numbers_v6.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")

# ---------------------------------------------------------------- bang Pareto: SMD, cua so bat thuong va bao dong gia
L = ["\\begin{table}[!htb]\\centering",
     "\\caption{Validity of repairs on SMD for true alarms and false alarms when the margin of m-SCR is swept and R-SCR+ runs at two margins, "
     "with share of channels restored, distance to the nearest normal window relative to normal spacing, and detector calls per window.}",
     "\\label{tab:pareto}\\footnotesize\\setlength{\\tabcolsep}{2.4pt}",
     "\\begin{tabular}{@{}lcccccccccccc@{}}\\toprule",
     " & \\multicolumn{7}{c}{True alarms} & \\multicolumn{5}{c}{False alarms} \\\\\\cmidrule(lr){2-8}\\cmidrule(lr){9-13}",
     "Generator & Seed & Full & Chain & Cross & Size & Dist. & Calls & Full & Chain & Cross & Size & Calls \\\\\\midrule"]
fmt = lambda m, v: "n/a" if v is None else ("%.0f" % v if m == "calls" else ("%.2f" % v if m == "plaus" else "%.3f" % v))
for g, lab, _ in GEN:
    a = S["smd"]["tp"]["table"][g]; b = S["smd"]["fa"]["table"][g]
    c = [fmt(m, a[m]) for m in ("seed", "full", "chain", "cross", "size", "plaus", "calls")] + [fmt(m, b[m]) for m in ("full", "chain", "cross", "size", "calls")]
    L.append(f"{lab} & " + " & ".join(c) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(f"{J}/tab_pareto.tex", "w").write("\n".join(L) + "\n")
for ds in S:
    print(ds, "rho", S[ds]["rho"], "neff", S[ds]["neff"])
    for kind in ("tp", "fa"):
        print(" ", kind, S[ds][kind]["n"], S[ds][kind]["ent"])
        for g, _, _ in GEN:
            t = S[ds][kind]["table"][g]
            print("   %-14s" % g, " ".join("%s=%s" % (m, fmt(m, t.get(m))) for m in ("found", "seed", "full", "chain", "cross", "size", "plaus", "calls")))
        for t in S[ds][kind]["tests"]:
            print("    %s vs %s %s %+.3f %d/%d p=%.4f" % (t["a"], t["b"], t["m"], t["diff"], t["up"], t["dn"], t["ph"]))
    print(" votecal", S[ds]["votecal"]); print(" mech", S[ds]["mech"])
