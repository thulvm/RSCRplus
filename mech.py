"""Phan tich co che (ban tap chi): vi sao ban sua mat hieu luc sau huan luyen lai.
Cach dung: python3 mech.py <ten_kho> <thuc_the...>  (bien moi truong RSCR_ROOT, RSCR_OUT)
Voi moi cua so duoc chon, ghi bien chuan hoa z = (tau - s(g)) / (tau - s_med) cua ban sua SCR va m-SCR
duoi bo dang chay va duoi bo cap nhat toan lich su, cung voi hop le sau cap nhat. Them vao do, voi cac cua so
hieu chuan binh thuong, ghi do dich chuyen z_upd - z_dep de do do lon dao dong ma mot lan huan luyen lai gay ra."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, common

ROOT = os.environ.get("RSCR_ROOT"); OUT = os.environ.get("RSCR_OUT")


def z(det, X):
    return (det.tau - det.score(X)) / max(det.tau - det.med, 1e-12)


def main(names):
    os.makedirs(OUT, exist_ok=True)
    for name in names:
        if os.path.exists(f"{OUT}/{name}.json"):
            continue
        c = common.setup(ROOT, name); K = c.K
        rec = []; shift = {}
        for gk in ("pca", "iforest", "ae"):
            dep = lib.Det(gk, 0).fit(c.Fdep, c.Fcal_dep)
            upd = lib.Det(gk, 7).fit(c.Ffit, c.Fcal)
            sub = c.Fcal_dep[np.random.RandomState(0).permutation(len(c.Fcal_dep))[:300]]
            shift[gk] = (z(upd, sub) - z(dep, sub)).tolist()
            for wi in common.picks(c, dep, 15):
                f = c.Fte[wi]; base = c.base
                dev = np.abs(f[:K] - base[:K]) / c.fsd[:K] + np.abs(f[K:] - base[K:]) / c.fsd[K:]
                order = [int(x) for x in np.argsort(-dev)]
                for nm, d in (("SCR", 0.0), ("m-SCR", 0.25)):
                    g, k, _ = lib.scr(dep, f, base, order, K, delta=d)
                    if g is None or not upd.flags(f)[0]:
                        continue
                    rec.append(dict(gen=gk, rep=nm, z_dep=float(z(dep, g)[0]), z_upd=float(z(upd, g)[0]),
                                    valid=int(upd.score(g)[0] <= upd.tau), k=int(k)))
        json.dump(dict(entity=name, records=rec, shift=shift), open(f"{OUT}/{name}.json", "w"))
        print(name, len(rec), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
