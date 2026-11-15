"""Kiem tra ro ri o ranh gioi: chuoi huan luyen lai thu ba dung cac cua so binh thuong cua tap kiem tra co diem dau
truoc moc 0.6 NT, con cac cua so duoc giai thich bat dau tu moc nay. Cua so dai W = 30 nen cac cua so bat dau trong
khoang [moc, moc + W - 1) co the chung du lieu tho voi du lieu huan luyen cua chuoi. Script tinh lai do ben chuoi khi
bo cac cua so do va ghi journal/numbers_leak.tex."""
import json, glob, os
import numpy as np

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = 30
G = {"SCR dev": "SCR", "m-SCR 0.25": "MSCR", "BetaRCE-T dev": "BT", "R-SCR+ 0.25": "RS"}
M, out = {}, {}
deltas, nnear, ntot = [], 0, 0
for ds in ("smd", "asd", "psm"):
    acc = {g: {"all": [], "clean": []} for g in G}
    for f in glob.glob(f"{HOME}/out/{ds}_v6/*.json"):
        e = os.path.basename(f)[:-5]
        NT = sum(1 for _ in open(f"{HOME}/data/{ds}/test/{e}.txt"))
        cut = int(0.6 * NT)
        for r in json.load(open(f))["records"]:
            ntot += 1
            near = r["win"] < cut + W - 1
            nnear += near
            if r["kind"] != "tp":
                continue
            for g in G:
                rr = r["res"].get(g)
                if not rr:
                    continue
                v = [rr[k] for k in ("chain1", "chain2", "chain3") if rr.get(k) is not None]
                if v:
                    acc[g]["all"].append(np.mean(v))
                    if not near:
                        acc[g]["clean"].append(np.mean(v))
    out[ds] = {g: (float(np.mean(a["all"])), float(np.mean(a["clean"]))) for g, a in acc.items() if a["all"]}
    deltas += [c - a for a, c in out[ds].values()]
M["LeakNear"] = str(nnear); M["LeakTot"] = str(ntot)
M["LeakShare"] = "%.0f\\%%" % (100 * nnear / ntot)
M["LeakMaxD"] = "%.3f" % max(abs(x) for x in deltas)
M["LeakAllUp"] = "yes" if all(x >= 0 for x in deltas) else "no"
json.dump(out, open(f"{HOME}/out/leak_check.json", "w"), indent=1)
open(f"{HOME}/journal/numbers_leak.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
print(M); print(json.dumps(out, indent=1))
