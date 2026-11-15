"""Ca minh hoa ban 2 (thiet lap common.py, R-SCR+ thu tu do lech, bien 0.25, tham do mot lan)."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, common
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT, NAME, GEN, WIN = RSCR_HOME + "/data/smd", "machine-1-2", "iforest", 4640
OUT = RSCR_HOME + "/out/figs"
c = common.setup(ROOT, NAME); K = c.K; Xte = c.Xte; base = c.base; fs = c.fs; fm = c.fm
dep, upd, cop_t, _ = common.detectors(c, GEN)
wi = int(np.where(c.tidx == WIN)[0][0]); f = c.Fte[wi]
dev_all = np.abs(f[:K] - base[:K]) / c.fsd[:K] + np.abs(f[K:] - base[K:]) / c.fsd[K:]
dev_order = [int(x) for x in np.argsort(-dev_all)]
occ, _ = lib.rank_occlusion(dep, f, base, dev_order[:common.NCAND], K)
occ = [int(x) for x in occ] + [x for x in dev_order if x not in occ]
g1, k1, _ = lib.scr(dep, f, base, occ, K)
r = lib.rscr_v2(dep, cop_t, f, base, dev_order, K, 0.25, 0.10, 0.05, np.random.RandomState(wi))
g2, k2 = r["g"], r["k"]
ch1, ch2 = occ[:k1], dev_order[:k2]
rec = [x for x in json.load(open(fRSCR_HOME + "/out/smd/{NAME}.json"))["records"] if x["gen"] == GEN and x["win"] == WIN][0]["gen_res"]
assert k1 == rec["SCR"]["k"] and k2 == rec["R-SCR+"]["k"], (k1, k2, rec["SCR"]["k"], rec["R-SCR+"]["k"])
sc = {}
for nm, g in (("SCR", g1), ("R-SCR+", g2)):
    sc[nm] = dict(dep=float(dep.score(g)[0] / dep.tau), upd=float(upd.score(g)[0] / upd.tau),
                  copies=int(sum(e.score(g)[0] <= e.tau for e in cop_t)),
                  held={h: bool(d.score(g)[0] <= d.tau) for h, d in c.held.items()})
    print(nm, [x + 1 for x in (ch1 if nm == "SCR" else ch2)], sc[nm])
lab = [s for s in c.segs if s[0] <= WIN + 15 <= s[1]]
print("nhan chuyen gia", [[x + 1 for x in s[2]] for s in lab])
json.dump(dict(entity=NAME, gen=GEN, win=WIN, scr_channels=[x + 1 for x in ch1], rscr_channels=[x + 1 for x in ch2],
               scores=sc, expert=[[x + 1 for x in s[2]] for s in lab]), open(RSCR_HOME + "/out/case_study.json", "w"), indent=1)

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
