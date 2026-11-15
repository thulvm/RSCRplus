"""Chat luong phat hien cua moi bo phat hien o muc cua so (nhan bat thuong khi it nhat nua so diem mang nhan).
Ghi out/det_quality.json: theo bo du lieu, ho bo phat hien, thuc the: precision, recall, F1 tai nguong, AUC-ROC, ty le bao dong gia."""
import os as _os; RSCR_HOME = _os.environ.get("RSCR_HOME", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
D = RSCR_HOME + "/data"; O = RSCR_HOME + "/out"
LISTS = {"smd": "entities.txt", "asd": "../asd_entities.txt", "psm": "psm_entities.txt"}
res = json.load(open(f"{O}/det_quality.json")) if os.path.exists(f"{O}/det_quality.json") else {}
for ds, L in LISTS.items():
    for e in open(L).read().split():
        if f"{ds}|{e}" in res:
            continue
        c = common.setup(f"{D}/{ds}", e)
        y = c.ylab.astype(int); row = {}
        dets = {gk: common.detectors(c, gk)[0] for gk in ("pca", "iforest", "ae")}
        dets.update(c.held)
        for gk, d in dets.items():
            s = d.score(c.Fte); fl = s > d.tau
            tp = int((fl & (y == 1)).sum()); fp = int((fl & (y == 0)).sum()); fn = int((~fl & (y == 1)).sum())
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            row[gk] = dict(p=p, r=r, f1=(2 * p * r / max(p + r, 1e-9)), auc=(roc_auc_score(y, s) if 0 < y.sum() < len(y) else None),
                           fa_share=fp / max(tp + fp, 1), n_alarm=int(fl.sum()))
        res[f"{ds}|{e}"] = row
        json.dump(res, open(f"{O}/det_quality.json", "w"))
        print(ds, e, {k: round(v["f1"], 2) for k, v in row.items()}, flush=True)
