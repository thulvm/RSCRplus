"""Tong hop chat luong phat hien (out/det_quality.json) va do nhay cua baseline (out/smd_sens/*.json).
Ghi journal/numbers_rev.tex va out/summary_rev.json. Ten macro khong chua chu so."""
import json, glob, os
import numpy as np
HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = HOME + "/out"; J = HOME + "/journal"
M, S = {}, {}
FAM = {"pca": "PCA", "iforest": "IF", "ae": "AE", "ocsvm": "OC", "lof": "LOF", "ecod": "ECOD"}
D = json.load(open(f"{O}/det_quality.json")) if os.path.exists(f"{O}/det_quality.json") else {}
for ds, dt in (("smd", "S"), ("asd", "A"), ("psm", "P")):
    rows = [v for k, v in D.items() if k.startswith(ds + "|")]
    S[ds] = {"n": len(rows)}
    M[f"Det{dt}N"] = str(len(rows))
    for fam, ft in FAM.items():
        vals = {m: [r[fam][m] for r in rows if fam in r and r[fam].get(m) is not None and not np.isnan(r[fam][m])]
                for m in ("f1", "auc", "fa_share", "p", "r")}
        S[ds][fam] = {m: (float(np.mean(v)) if v else None) for m, v in vals.items()}
        for m, mt in (("f1", "FONE"), ("auc", "AUC"), ("fa_share", "FA"), ("p", "PREC"), ("r", "REC")):
            v = S[ds][fam][m]
            M[f"Det{dt}{ft}{mt}"] = "n/a" if v is None else ("%.0f\\%%" % (100 * v) if m == "fa_share" else "%.2f" % v)
    run = [S[ds][f]["f1"] for f in ("pca", "iforest", "ae") if S[ds][f]["f1"] is not None]
    if run:
        M[f"Det{dt}FONELO"] = "%.2f" % min(run); M[f"Det{dt}FONEHI"] = "%.2f" % max(run)
    fa = [S[ds][f]["fa_share"] for f in ("pca", "iforest", "ae") if S[ds][f]["fa_share"] is not None]
    if fa:
        M[f"Det{dt}FALO"] = "%.0f\\%%" % (100 * min(fa)); M[f"Det{dt}FAHI"] = "%.0f\\%%" % (100 * max(fa))
    au = [S[ds][f]["auc"] for f in ("pca", "iforest", "ae") if S[ds][f]["auc"] is not None]
    if au:
        M[f"Det{dt}AUCLO"] = "%.2f" % min(au); M[f"Det{dt}AUCHI"] = "%.2f" % max(au)

# do nhay baseline
F = sorted(glob.glob(f"{O}/smd_sens/*.json"))
LET = "abcdefghijkl"
if F:
    conf = json.load(open(F[0]))["conf"]
    per = {i: {m: [] for m in ("found", "size", "full", "chain", "cross", "calls")} for i in range(len(conf))}
    for f in F:
        d = json.load(open(f))
        acc = {i: {m: [] for m in per[i]} for i in per}
        for r in d["records"]:
            for i in per:
                rr = r["res"][str(i)]
                acc[i]["found"].append(rr["found"]); acc[i]["calls"].append(rr["calls"])
                if rr["frac"] is not None: acc[i]["size"].append(rr["frac"])
                for m, keys in (("full", ["upd_s0", "upd_s1", "upd_s2"]), ("chain", ["chain1", "chain2", "chain3"]),
                                ("cross", ["x_ocsvm", "x_lof", "x_ecod"])):
                    v = [rr[k] for k in keys if rr.get(k) is not None]
                    if v: acc[i][m].append(np.mean(v))
        for i in per:
            for m in per[i]:
                if acc[i][m]: per[i][m].append(np.mean(acc[i][m]))
    S["sens"] = {"n_ent": len(F), "conf": conf, "res": {}}
    M["SensNE"] = str(len(F))
    for i in per:
        row = {m: (float(np.mean(v)) if v else None) for m, v in per[i].items()}
        S["sens"]["res"][i] = row
        for m, mt in (("found", "FOUND"), ("size", "SIZE"), ("full", "FULL"), ("chain", "CHAIN"), ("cross", "CROSS"), ("calls", "CALLS")):
            v = row[m]
            M[f"Sens{LET[i]}{mt}"] = "n/a" if v is None else ("%.0f" % v if m == "calls" else "%.3f" % v)
    for name, idx in (("RX", range(0, 6)), ("TR", range(6, 9)), ("RO", range(9, 12))):
        for m, mt in (("chain", "CHAIN"), ("full", "FULL"), ("size", "SIZE"), ("found", "FOUND"), ("cross", "CROSS")):
            v = [S["sens"]["res"][i][m] for i in idx if S["sens"]["res"][i][m] is not None]
            if v:
                M[f"Sens{name}{mt}LO"] = "%.3f" % min(v); M[f"Sens{name}{mt}HI"] = "%.3f" % max(v)

# doi chieu: R-SCR+ va m-SCR tren cung 10 may, cung cach tinh (giao thuc v6, cua so bat thuong that)
ENTS = [l.strip() for i, l in enumerate(open(os.path.dirname(os.path.abspath(__file__)) + "/entities.txt")) if i % 3 == 0][:10]
for g, gt in (("R-SCR+ 0.25", "RS"), ("m-SCR 0.25", "MS")):
    per = {m: [] for m in ("full", "chain", "cross", "size")}
    for e in ENTS:
        fp = f"{O}/smd_v6/{e}.json"
        if not os.path.exists(fp): continue
        acc = {m: [] for m in per}
        for r in json.load(open(fp))["records"]:
            if r["kind"] != "tp" or g not in r["res"]: continue
            rr = r["res"][g]
            if rr["frac"] is not None: acc["size"].append(rr["frac"])
            for m, keys in (("full", ["upd_s0", "upd_s1", "upd_s2"]), ("chain", ["chain1", "chain2", "chain3"]), ("cross", ["x_ocsvm", "x_lof", "x_ecod"])):
                v = [rr[k] for k in keys if rr.get(k) is not None]
                if v: acc[m].append(np.mean(v))
        for m in per:
            if acc[m]: per[m].append(np.mean(acc[m]))
    for m, mt in (("full", "FULL"), ("chain", "CHAIN"), ("cross", "CROSS"), ("size", "SIZE")):
        M[f"Sens{gt}{mt}"] = "%.3f" % np.mean(per[m]) if per[m] else "n/a"
    print(g, {m: round(float(np.mean(v)), 3) for m, v in per.items() if v})
json.dump(S, open(f"{O}/summary_rev.json", "w"), indent=1)
open(f"{J}/numbers_rev.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
for ds in ("smd", "asd", "psm"):
    print(ds, S[ds]["n"], {f: (None if S[ds][f]["f1"] is None else (round(S[ds][f]["f1"], 2), round(S[ds][f]["auc"], 2), round(S[ds][f]["fa_share"], 2))) for f in FAM})
if "sens" in S:
    for i, c in enumerate(S["sens"]["conf"]):
        print(i, c, {k: (None if v is None else round(v, 3)) for k, v in S["sens"]["res"][i].items()})
