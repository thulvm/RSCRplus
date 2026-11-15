"""Ca minh hoa ban 3: tu dong chon mot cua so SMD ma SCR mat hieu luc sau huan luyen lai con R-SCR+ giu duoc,
tinh lai tu dau va doi chieu voi ket qua chinh, ve fig_problem va ghi macro cho bai."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, common
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = RSCR_HOME + "/data/smd"; OUT = RSCR_HOME + "/out/figs"
names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

# ---------- chon ca
cands = []
for fn in sorted(glob.glob(RSCR_HOME + "/out/smd/*.json")):
    for r in json.load(open(fn))["records"]:
        s, q = r["gen_res"]["SCR"], r["gen_res"]["R-SCR+"]
        if not (r["upd_flag"]["upd"] and r["upd_flag"]["later"] and r["pm"] is not None):
            continue
        if not (s["found"] and q["found"] and s["var"] == 0 and s["var_later"] == 0 and q["var"] == 1 and q["var_later"] == 1):
            continue
        cs = [v for v in s["cross"].values() if v is not None]; cq = [v for v in q["cross"].values() if v is not None]
        if len(cq) < 3 or sum(cq) < 3 or sum(cs) > 0 or q["k"] > 6 or s["k"] > 3:
            continue
        cands.append((-r["pm"]["deviation"], q["k"], r["entity"], r["gen"], r["win"]))
cands.sort()
print("so ca phu hop", len(cands), cands[:5])
_, _, NAME, GEN, WIN = cands[0]

c = common.setup(ROOT, NAME); K = c.K; Xte = c.Xte; base = c.base; fs = c.fs; fm = c.fm
dep, upd, cop_t, _ = common.detectors(c, GEN)
wi = int(np.where(c.tidx == WIN)[0][0]); f = c.Fte[wi]
dev_all = np.abs(f[:K] - base[:K]) / c.fsd[:K] + np.abs(f[K:] - base[K:]) / c.fsd[K:]
dev_order = [int(x) for x in np.argsort(-dev_all)]
occ, _ = lib.rank_occlusion(dep, f, base, dev_order[:common.NCAND], K)
occ = [int(x) for x in occ] + [x for x in dev_order if x not in occ]
g1, k1, _ = lib.scr(dep, f, base, occ, K)
r = lib.rscr_v2(dep, cop_t, f, base, dev_order, K, 0.25, 0.10, 0.05, np.random.RandomState(wi), track_full=True)
g2, k2 = r["g"], r["k"]
ch1, ch2 = occ[:k1], dev_order[:k2]
rec = [x for x in json.load(open(fRSCR_HOME + "/out/smd/{NAME}.json"))["records"] if x["gen"] == GEN and x["win"] == WIN][0]["gen_res"]
assert k1 == rec["SCR"]["k"] and k2 == rec["R-SCR+"]["k"], (k1, k2, rec["SCR"]["k"], rec["R-SCR+"]["k"])
sc = {}
for nm, g in (("SCR", g1), ("R-SCR+", g2)):
    sc[nm] = dict(dep=float(dep.score(g)[0] / dep.tau), upd=float(upd["upd"].score(g)[0] / upd["upd"].tau),
                  later=float(upd["later"].score(g)[0] / upd["later"].tau),
                  copies=int(sum(e.score(g)[0] <= e.tau for e in cop_t)),
                  held={h: bool(d.score(g)[0] <= d.tau) for h, d in c.held.items()})
    print(nm, [x + 1 for x in (ch1 if nm == "SCR" else ch2)], sc[nm])
lab = [s for s in c.segs if s[0] <= WIN + lib.W // 2 <= s[1]]
expert = sorted({x + 1 for s in lab for x in s[2]})
print("nhan chuyen gia", expert)
ov = len(set(x + 1 for x in ch2) & set(expert))
fam = "Isolation Forest" if GEN == "iforest" else "PCA detector"
mac = dict(CASEENT=NAME, CASEDET=fam, CASEWIN=WIN, CASESCRK=names[k1], CASERK=names[k2], CASEOV=names[ov],
           CASESCRMARG=f"{100 * (1 - sc['SCR']['dep']):.0f}", CASECOP=sc["SCR"]["copies"], CASERCOP=sc["R-SCR+"]["copies"],
           CASESCRHELD=names[sum(sc["SCR"]["held"].values())], CASERHELD=names[sum(sc["R-SCR+"]["held"].values())],
           CASESCRCH=", ".join(str(x + 1) for x in sorted(ch1)), CASERCH=", ".join(str(x + 1) for x in sorted(ch2)),
           CASEEXP=", ".join(map(str, expert)))
json.dump(dict(entity=NAME, gen=GEN, win=int(WIN), scr_channels=[x + 1 for x in ch1], rscr_channels=[x + 1 for x in ch2],
               scores=sc, expert=expert, macros=mac), open(RSCR_HOME + "/out/case_study_v3.json", "w"), indent=1, default=int)

# ---------- hinh
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 12.5,
                     "axes.spines.top": False, "axes.spines.right": False})
pad = 40
t = np.arange(max(0, WIN - pad), min(len(Xte), WIN + 30 + pad))
fig = plt.figure(figsize=(7.6, 2.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.55)
show = sorted(set(ch1) | set(ch2))[:6]
for j, (title, chans) in enumerate((("SCR", ch1), ("R-SCR+", ch2))):
    ax = fig.add_subplot(gs[0, j])
    for i, ch in enumerate(show):
        off = -2.0 * i
        y = Xte[t, ch].copy()
        seg = Xte[WIN:WIN + 30, ch]
        m_n, s_n = base[ch] * fs[ch] + fm[ch], base[K + ch] * fs[K + ch] + fm[K + ch]
        rep = (seg - seg.mean()) / (seg.std() + 1e-9) * s_n + m_n
        lo, hi = np.percentile(np.r_[y, rep], [2, 98]); rng = max(hi - lo, 1e-6)
        nz = lambda v: np.clip((v - lo) / rng, -0.05, 1.05)
        ax.plot(t, nz(y) + off, color="#6F6F6F", lw=1.0)
        if ch in chans:
            ax.plot(np.arange(WIN, WIN + 30), nz(rep) + off, color="#3D85C6", lw=1.8)
        ax.text(t[0] - 2, off + 0.5, f"ch {ch + 1}", ha="right", va="center", fontsize=12)
        if i:
            ax.axhline(off + 1.5, color="#D0D0D0", lw=0.7, zorder=0)
    ax.axvspan(WIN, WIN + 29, color="#FBEBD2", zorder=0)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlim(t[0] - 1, t[-1] + 1); ax.set_xticks([])
    ax.set_xlabel("time, window shaded")
    ax.text(0.0, 1.02, f"({'ab'[j]})", transform=ax.transAxes, fontsize=12.5)
ax = fig.add_subplot(gs[0, 2])
vals = [1 - sc["SCR"]["dep"], 1 - sc["SCR"]["upd"], 1 - sc["R-SCR+"]["dep"], 1 - sc["R-SCR+"]["upd"]]
labs = ["now", "after", "now", "after"]
cols = ["#D9D9D9", "#F4CCCC", "#9FC5E8", "#9FC5E8"]
bars = ax.bar(range(4), vals, color=cols, edgecolor="#555555", lw=0.6)
ax.axhline(0.0, color="#555555", lw=0.9)
top, bot = max(vals), min(vals)
sp = 0.03 * (top - min(bot, 0))
for bb, v in zip(bars, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, v + (sp if v >= 0 else -sp), f"{v:+.2f}".replace("-", "\u2212"), ha="center",
            va="bottom" if v >= 0 else "top", fontsize=12.5, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=12)
ax.tick_params(axis="x", length=0)
for xc, gl in ((0.5, "SCR"), (2.5, "R-SCR+")):
    ax.text(xc, -0.13, gl, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12.5)
ax.set_ylabel(r"margin, fraction of $\tau$"); ax.set_ylim(min(bot, 0) - 4 * sp, top + 5 * sp)
ax.text(0.0, 1.02, "(c)", transform=ax.transAxes, fontsize=12.5)
fig.savefig(f"{OUT}/fig_problem.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_problem.png", dpi=220, bbox_inches="tight")
pass
print(mac)
