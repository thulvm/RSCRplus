"""Dung lai mot ca that de ve hinh bai toan: SCR hong sau huan luyen lai, R-SCR+ giu duoc.
Thiet lap giong het run_smd.py; kiem tra lai so kenh va phan quyet so voi tep JSON."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT, NAME, GEN, WIN = RSCR_HOME + "/data/smd", "machine-2-2", "iforest", 15720
B, ALPHA, GAMMA, NCAND = 40, 0.10, 0.05, 10
OUT = RSCR_HOME + "/out/figs"

tr, te, lab, segs = lib.load_entity(ROOT, NAME)
mu, sd = tr.mean(0), tr.std(0); sd[sd < 1e-3] = 1e-3
Xtr, Xte = (tr - mu) / sd, (te - mu) / sd
K = Xtr.shape[1]
_, Wtr = lib.windows(Xtr); Ftr = lib.feats(Wtr)
perm = np.random.RandomState(0).permutation(len(Ftr)); ncal = len(Ftr) // 5
cal_idx, fit_idx = np.sort(perm[:ncal]), np.sort(perm[ncal:])
fm, fs = Ftr[fit_idx].mean(0), Ftr[fit_idx].std(0); fs[fs < 1e-6] = 1.0
S = lambda F: (F - fm) / fs
Ffit, Fcal = S(Ftr[fit_idx]), S(Ftr[cal_idx]); Fdep = Ffit[: int(0.7 * len(Ffit))]
base = np.median(Ffit, 0); fsd = Ffit.std(0) + 1e-6
held = {k: lib.Det(k, 0).fit(Ffit, Fcal) for k in ("ocsvm", "lof", "ecod")}
gr = lib.Granger(Xtr)
dep = lib.Det(GEN, 0).fit(Fdep, Fcal); upd = lib.Det(GEN, 7).fit(Ffit, Fcal)
copies, L = [], int(0.6 * len(Fdep))
for b in range(1, B + 1):
    rb = np.random.RandomState(1000 + b); rb.choice(len(Fdep), len(Fdep), replace=True)
    s0 = rb.randint(0, len(Fdep) - L + 1); copies.append(lib.Det(GEN, 100 + b).fit(Fdep[s0:s0 + L], Fcal))
tidx, Wte = lib.windows(Xte); wi = int(np.where(tidx == WIN)[0][0])
f = S(lib.feats(Wte[wi:wi + 1]))[0]
dev_all = np.abs(f[:K] - base[:K]) / fsd[:K] + np.abs(f[K:] - base[K:]) / fsd[K:]
dev_order = list(np.argsort(-dev_all)); cand = dev_order[:NCAND]
ext = lambda r: list(r) + [c for c in dev_order if c not in r]
occ, _ = lib.rank_occlusion(dep, f, base, cand, K)
cons = lib.borda([occ, lib.rank_ig(dep, f, base, cand, K)[0], lib.rank_shapley(dep, f, base, cand, K)[0],
                  gr.rank(Xte, WIN, cand)])
g1, k1, _ = lib.scr(dep, f, base, ext(occ), K)
r = lib.rscr(dep, copies, f, base, ext(cons), K, deltas=(0.25, 0.5, 1.0), alpha=ALPHA, gamma=GAMMA,
             rs=np.random.RandomState(wi))
g2, k2 = r["g"], r["k"]
ch1, ch2 = [int(c) for c in ext(occ)[:k1]], [int(c) for c in ext(cons)[:k2]]
rec = [x for x in json.load(open(fRSCR_HOME + "/out/smd/{NAME}.json"))["records"]
       if x["gen"] == GEN and x["win"] == WIN][0]["gen_res"]
print("SCR kenh", [c + 1 for c in ch1], "k", k1, "JSON k", rec["SCR"]["k"])
print("R-SCR+ kenh", [c + 1 for c in ch2], "k", k2, "JSON k", rec["R-SCR+"]["k"])
sc = {}
for nm, g in (("SCR", g1), ("R-SCR+", g2)):
    sc[nm] = dict(dep=float(dep.score(g)[0] / dep.tau), upd=float(upd.score(g)[0] / upd.tau),
                  copies=int(sum(c.score(g)[0] <= c.tau for c in copies)),
                  held={h: bool(d.score(g)[0] <= d.tau) for h, d in held.items()})
    print(nm, sc[nm])
labelled = [s for s in segs if s[0] <= WIN + 15 <= s[1]]
print("nhan chuyen gia", [[c + 1 for c in s[2]] for s in labelled])
json.dump(dict(entity=NAME, gen=GEN, win=WIN, scr_channels=[c + 1 for c in ch1],
               rscr_channels=[c + 1 for c in ch2], scores=sc,
               expert=[[c + 1 for c in s[2]] for s in labelled]),
          open(RSCR_HOME + "/out/case_study.json", "w"), indent=1)

# ---------- hinh
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
pad = 40
t = np.arange(WIN - pad, WIN + 30 + pad)
fig = plt.figure(figsize=(10.5, 3.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1.0], wspace=0.32)
show = sorted(ch2[:4])
for j, (title, chans) in enumerate((("SCR", ch1), ("R-SCR+", ch2))):
    ax = fig.add_subplot(gs[0, j])
    for i, c in enumerate(show):
        off = -1.5 * i
        y = Xte[t, c].copy()
        seg = Xte[WIN:WIN + 30, c]
        m_n, s_n = base[c] * fs[c] + fm[c], base[K + c] * fs[K + c] + fm[K + c]
        rep = (seg - seg.mean()) / (seg.std() + 1e-9) * s_n + m_n
        lo, hi = np.percentile(np.r_[y, rep], [2, 98]); rng = max(hi - lo, 1e-6)
        nz = lambda v: np.clip((v - lo) / rng, -0.1, 1.1)
        ax.plot(t, nz(y) + off, color="#6F6F6F", lw=1.0)
        if c in chans:
            ax.plot(np.arange(WIN, WIN + 30), nz(rep) + off, color="#3D85C6", lw=1.8)
        ax.text(t[0] - 2, off + 0.5, f"ch {c + 1}", ha="right", va="center", fontsize=9.5)
    ax.axvspan(WIN, WIN + 29, color="#FBEBD2", zorder=0)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlim(t[0] - 1, t[-1] + 1); ax.set_xticks([WIN - 30, WIN, WIN + 30, WIN + 60])
    ax.set_xlabel("time")
    ax.text(0.0, 1.03, f"({'ab'[j]}) {title} repairs {len(chans)} channel" + ("s" if len(chans) > 1 else ""),
            transform=ax.transAxes, fontsize=10.5)
ax = fig.add_subplot(gs[0, 2])
vals = [1 - sc["SCR"]["dep"], 1 - sc["SCR"]["upd"], 1 - sc["R-SCR+"]["dep"], 1 - sc["R-SCR+"]["upd"]]
labs = ["SCR\nrunning", "SCR\nretrained", "R-SCR+\nrunning", "R-SCR+\nretrained"]
cols = ["#D9D9D9", "#F4CCCC", "#9FC5E8", "#9FC5E8"]
bars = ax.bar(range(4), vals, color=cols, edgecolor="#555555", lw=0.6)
ax.axhline(0.0, color="#555555", lw=0.9)
top = max(vals)
for bb, v in zip(bars, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, v + (0.006 if v >= 0 else -0.006), f"{v:+.2f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=9.5, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8.8)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("margin below alarm"); ax.set_ylim(-0.07, top * 1.3)
ax.text(0.0, 1.03, "(c) margin after retraining", transform=ax.transAxes, fontsize=10.5)
fig.savefig(f"{OUT}/fig_problem.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_problem.png", dpi=220, bbox_inches="tight")
print("ok")
