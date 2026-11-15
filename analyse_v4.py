"""Phan tich ban 4: tach nhieu seed khoi hieu ung huan luyen lai, chuoi huan luyen lai, bo phat hien autoencoder.
Doc out/smd_v4/*.json (va out/psm_v4 neu co). Ghi out/summary_v4.json, paper/numbers_v4.tex, paper/tab_v4.tex, paper/figs/fig_v4.pdf.
Moi do do ben la ty le ban sua con hop le duoi mot bo cap nhat, chi tinh tren cua so ma bo cap nhat do cung bao dong.
Seed: trung binh 5 bo chi doi seed (cung du lieu voi bo dang chay). Full: trung binh 5 seed khop lai tren toan lich su.
Chain k: bo khop tren lat du lieu ngay cang moi, buoc 2 va 3 co them cua so binh thuong cua giai doan kiem tra."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob, json, sys
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
O = RSCR_HOME + "/out"; P = RSCR_HOME + "/paper"
GEN = [("SCR dev", "SCR"), ("Wachter", "Wachter"), ("Wachter margin", "Wachter + margin"), ("m-SCR dev", "m-SCR"),
       ("BetaRCE-T dev", "BetaRCE-T"), ("R-SCR+", "R-SCR+")]
FAM = [("pca", "PCA"), ("iforest", "Isolation forest"), ("ae", "Autoencoder")]
EV = {"seed": [f"same_s{i}" for i in range(5)], "full": [f"upd_s{i}" for i in range(5)], "c1": ["chain1"], "c2": ["chain2"], "c3": ["chain3"]}
rng = np.random.default_rng(0)


def load(ds):
    R = []
    for f in sorted(glob.glob(f"{O}/{ds}_v4/*.json")):
        R += json.load(open(f))["records"]
    return R


def window_score(r, g, keys):
    """trung binh tren cac bo cap nhat cung bao dong cua so nay; None neu khong bo nao bao dong"""
    v = [r["res"][g][k] for k in keys if r["res"][g][k] is not None]
    return np.mean(v) if v else None


def cluster_boot(vals, ents, B=2000):
    vals = np.asarray(vals, float); ents = np.asarray(ents); u = np.unique(ents)
    idx = {e: np.where(ents == e)[0] for e in u}
    bs = []
    for _ in range(B):
        pick = rng.choice(u, len(u)); ii = np.concatenate([idx[e] for e in pick]); bs.append(vals[ii].mean())
    return [float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


S = {}
for ds in ("smd", "psm"):
    R = load(ds)
    if not R: continue
    S[ds] = {"n_records": len(R), "entities": len({r["entity"] for r in R})}
    for fam in [f for f, _ in FAM] + ["all"]:
        RR = R if fam == "all" else [r for r in R if r["gen"] == fam]
        if not RR: continue
        T = {}
        for g, _ in GEN:
            row = dict(found=float(np.mean([r["res"][g]["found"] for r in RR])),
                       calls=float(np.mean([r["res"][g]["calls"] for r in RR])),
                       frac=float(np.nanmean([r["res"][g]["frac"] if r["res"][g]["frac"] is not None else np.nan for r in RR])))
            for e, keys in EV.items():
                pairs = [(window_score(r, g, keys), r["entity"]) for r in RR]
                pairs = [(v, en) for v, en in pairs if v is not None]
                row[e] = cluster_boot([v for v, _ in pairs], [en for _, en in pairs]) if pairs else [None] * 3
                row["n_" + e] = len(pairs)
            cm = [np.mean([window_score(r, g, EV[k]) for k in ("c1", "c2", "c3") if window_score(r, g, EV[k]) is not None])
                  for r in RR if any(window_score(r, g, EV[k]) is not None for k in ("c1", "c2", "c3"))]
            row["chain"] = float(np.mean(cm)) if cm else None
            T[g] = row
        # kiem dinh ghep cap R-SCR+ so voi tung baseline tren do ben trung binh chuoi, theo thuc the
        tests = {}
        for g, _ in GEN:
            if g == "R-SCR+": continue
            a, b = [], []
            for en in sorted({r["entity"] for r in RR}):
                rs = [r for r in RR if r["entity"] == en]
                def ent(gg):
                    v = [window_score(r, gg, EV[k]) for r in rs for k in ("c1", "c2", "c3")]
                    v = [x for x in v if x is not None]; return np.mean(v) if v else None
                x, y = ent("R-SCR+"), ent(g)
                if x is not None and y is not None: a.append(x); b.append(y)
            d = np.array(a) - np.array(b)
            p = float(wilcoxon(d, zero_method="zsplit").pvalue) if len(d) > 1 and np.any(d != 0) else 1.0
            tests[g] = dict(diff=float(d.mean()) if len(d) else None, p=p, n=len(d), up=int((d > 0).sum()), down=int((d < 0).sum()))
        S[ds][fam] = dict(table=T, tests=tests)
    # Holm tren cac kiem dinh 'all' cua ds
    if "all" in S[ds]:
        ks = list(S[ds]["all"]["tests"]); ps = np.array([S[ds]["all"]["tests"][k]["p"] for k in ks])
        o = np.argsort(ps); run = 0
        for i, j in enumerate(o):
            run = max(run, (len(ps) - i) * ps[j]); S[ds]["all"]["tests"][ks[j]]["p_holm"] = float(min(1, run))
json.dump(S, open(f"{O}/summary_v4.json", "w"), indent=1)

# macro
M = {}
GT = {"SCR dev": "SCR", "Wachter": "WA", "Wachter margin": "WAM", "m-SCR dev": "MSCR", "BetaRCE-T dev": "BT", "R-SCR+": "RS"}
FT = {"pca": "P", "iforest": "I", "ae": "A", "all": "X"}
for ds, dt in (("smd", "S"), ("psm", "Q")):
    if ds not in S: continue
    M[f"Vfour{dt}NREC"] = str(S[ds]["n_records"]); M[f"Vfour{dt}NENT"] = str(S[ds]["entities"])
    for fam, ft in FT.items():
        if fam not in S[ds]: continue
        for g, gt in GT.items():
            r = S[ds][fam]["table"][g]
            for e in ("seed", "full", "c1", "c2", "c3"):
                if r[e][0] is not None:
                    M[f"Vfour{dt}{ft}{gt}{ {'seed': 'SEED', 'full': 'FULL', 'c1': 'CHA', 'c2': 'CHB', 'c3': 'CHC'}[e]}"] = "%.3f" % r[e][0]
            M[f"Vfour{dt}{ft}{gt}CHAIN"] = "%.3f" % r["chain"] if r["chain"] is not None else "--"
            M[f"Vfour{dt}{ft}{gt}CALLS"] = "%.1f" % r["calls"]; M[f"Vfour{dt}{ft}{gt}FOUND"] = "%.3f" % r["found"]
            if fam == "all" and g != "R-SCR+":
                t = S[ds]["all"]["tests"][g]
                pp = t.get("p_holm", t["p"]); M[f"Vfour{dt}T{gt}P"] = "%.3f" % pp if pp >= 0.001 else "\\ensuremath{%.1f\\times10^{%d}}" % (pp / 10 ** np.floor(np.log10(pp)), np.floor(np.log10(pp))); M[f"Vfour{dt}T{gt}D"] = "%.3f" % t["diff"]
                M[f"Vfour{dt}T{gt}UP"] = str(t["up"]); M[f"Vfour{dt}T{gt}DN"] = str(t["down"]); M[f"Vfour{dt}T{gt}N"] = str(t["n"])
open(f"{P}/numbers_ext.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()))

# bang: SMD, gop ba ho bo phat hien
ds = "smd"; T = S[ds]["all"]["table"]
f3 = lambda v: "--" if v is None else "%.3f" % v
L = ["\\begin{table}[H]\\centering",
     "\\caption{Validity of repairs on SMD under five seed-only refits, five full-history refits and three chained refits on newer data, "
     "pooled over the PCA, isolation forest and autoencoder detectors, with detector calls per window.}",
     "\\label{tab:v4}\\scriptsize\\setlength{\\tabcolsep}{3.2pt}",
     "\\begin{tabular}{@{}lcccccccc@{}}\\toprule",
     "Generator & Calls & Size & Seed only & Full history & Chain 1 & Chain 2 & Chain 3 & R-SCR+ gain, H/L \\\\\\midrule"]
for g, lab in GEN:
    r = T[g]
    t = S[ds]["all"]["tests"].get(g)
    hl = "" if t is None else f"{t['up']}/{t['down']}"
    L.append(f"{lab} & {r['calls']:.1f} & {r['frac']:.3f} & {f3(r['seed'][0])} & {f3(r['full'][0])} & {f3(r['c1'][0])} & {f3(r['c2'][0])} & {f3(r['c3'][0])} & {hl} \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(f"{P}/tab_ext.tex", "w").write("\n".join(L))

# hinh: moi ho bo phat hien mot panel, moi bo sinh mot hang: seed only, full, chain trung binh
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                     "font.size": 12.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
for ax, (fam, title) in zip(axes, FAM):
    if fam not in S[ds]: continue
    T = S[ds][fam]["table"]
    for i, (g, lab) in enumerate(GEN):
        v = [T[g]["seed"][0], T[g]["full"][0], T[g]["chain"]]
        v = [x for x in v if x is not None]
        ax.plot([min(v), max(v)], [i, i], color="#C9C9C9", lw=2, zorder=1)
        ax.scatter([T[g]["seed"][0]], [i], s=44, color="#8C8C8C", marker="o", zorder=3)
        ax.scatter([T[g]["full"][0]], [i], s=48, color="#3D85C6", marker="s", zorder=3)
        if T[g]["chain"] is not None: ax.scatter([T[g]["chain"]], [i], s=52, color="#C0504D", marker="D", zorder=3)
    ax.set_title(title, fontsize=12.5); ax.set_xlim(0.4, 1.03); ax.set_xticks([0.4, 0.7, 1.0])
axes[0].set_yticks(range(len(GEN))); axes[0].set_yticklabels([l for _, l in GEN])
axes[1].set_xlabel("share of repairs still valid")
h = [plt.scatter([], [], s=44, color="#8C8C8C"), plt.scatter([], [], s=48, color="#3D85C6", marker="s"), plt.scatter([], [], s=52, color="#C0504D", marker="D")]
fig.legend(h, ["seed-only refit", "full-history refit", "chained refits"], loc="upper center", bbox_to_anchor=(0.58, 1.08), ncol=3,
           frameon=False, fontsize=12, handletextpad=0.2, columnspacing=1.0)
fig.tight_layout(w_pad=0.6)
fig.savefig(f"{P}/figs/fig_v4.pdf", bbox_inches="tight"); fig.savefig(f"{O}/figs/fig_v4.png", dpi=200, bbox_inches="tight")
for fam in [f for f, _ in FAM] + ["all"]:
    if fam not in S[ds]: continue
    print(fam)
    for g, _ in GEN:
        r = S[ds][fam]["table"][g]
        print("  %-15s calls %6.1f found %.2f seed %s full %s c1 %s c2 %s c3 %s chain %s" % (g, r["calls"], r["found"], f3(r["seed"][0]), f3(r["full"][0]), f3(r["c1"][0]), f3(r["c2"][0]), f3(r["c3"][0]), f3(r["chain"])))
print(json.dumps(S[ds]["all"]["tests"], indent=0))
