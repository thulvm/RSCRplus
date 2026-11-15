"""Thiet lap dung chung (ban 3, sau hai vong ra soat doc lap).

Giai doan trien khai = 70% dau cua chuoi huan luyen theo thoi gian (hang).
  - Moi buoc chuan hoa (thang z tung kenh, chuan hoa dac trung) chi dung du lieu giai doan trien khai.
  - Cua so chia theo khoi 20 cua so lien nhau: 80% khoi de khop, 20% khoi de hieu chuan; bo cac cua so
    nam trong 2 buoc truot cua mot khoi khac nhom, nen cua so khop va cua so hieu chuan khong chong len nhau.
  - Bo dang chay: khop va hieu chuan chi tren cua so thuoc giai doan trien khai.
  - Ban sao: khoi lien nhau 60% cua cua so khop giai doan trien khai, hieu chuan nhu bo dang chay.
  - Bo cap nhat 'upd': khop tren moi cua so khop (toan bo lich su), hieu chuan tren moi cua so hieu chuan.
  - Bo cap nhat 'later': chi khop va hieu chuan tren 70% sau cua chuoi (tu moc 30%), it trung voi giai doan trien khai.
  - Ba kien truc giu lai (OCSVM, LOF, ECOD): khop va hieu chuan nhu 'upd'.
"""
import numpy as np
import lib

B, NCAND, CHUNK = 40, 10, 20


class Ctx:
    pass


def setup(root, name):
    c = Ctx()
    tr, te, lab, segs = lib.load_entity(root, name)
    N = len(tr)
    cut_row, later_row = int(0.7 * N), int(0.3 * N)
    mu, sd = tr[:cut_row].mean(0), tr[:cut_row].std(0); sd[sd < 1e-3] = 1e-3
    c.Xtr, c.Xte = (tr - mu) / sd, (te - mu) / sd
    c.K = K = c.Xtr.shape[1]
    c.segs = segs
    starts, Wtr = lib.windows(c.Xtr)
    Fraw = lib.feats(Wtr)
    n = len(starts)
    chunk = np.arange(n) // CHUNK
    nch = chunk.max() + 1
    is_cal_chunk = np.zeros(nch, bool)
    is_cal_chunk[np.random.RandomState(0).permutation(nch)[: max(1, nch // 5)]] = True
    is_cal = is_cal_chunk[chunk]
    keep = np.ones(n, bool)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        if np.any(is_cal[lo:hi] != is_cal[i]):
            keep[i] = False
    end = starts + lib.W
    dep_w = end <= cut_row
    later_w = starts >= later_row
    fit = keep & ~is_cal; cal = keep & is_cal
    fm, fs = Fraw[fit & dep_w].mean(0), Fraw[fit & dep_w].std(0); fs[fs < 1e-6] = 1.0
    c.fm, c.fs = fm, fs
    S = lambda F: (F - fm) / fs
    F = S(Fraw)
    c.Ffit, c.Fcal = F[fit], F[cal]
    c.Fdep, c.Fcal_dep = F[fit & dep_w], F[cal & dep_w]
    c.Flater, c.Fcal_later = F[fit & later_w], F[cal & later_w]
    c.base = np.median(c.Fdep, 0)
    c.fsd = c.Fdep.std(0) + 1e-6
    c.held = {k: lib.Det(k, 0).fit(c.Ffit, c.Fcal) for k in ("ocsvm", "lof", "ecod")}
    tidx, Wte = lib.windows(c.Xte)
    c.tidx = tidx
    c.Fte = S(lib.feats(Wte))
    c.ylab = np.array([lab[i:i + lib.W].mean() >= 0.5 for i in tidx])
    nn = lib.NearestNeighbors(n_neighbors=2).fit(c.Fcal)
    med_nn = np.median(nn.kneighbors(c.Fcal)[0][:, 1])
    c.plaus = lambda g: float(nn.kneighbors(g[None], n_neighbors=1)[0][0, 0] / med_nn)
    c.sizes = dict(rows=N, fit=int(fit.sum()), cal=int(cal.sum()), dep=len(c.Fdep), cal_dep=len(c.Fcal_dep),
                   later=len(c.Flater), dropped=int((~keep).sum()))
    return c


def detectors(c, gk):
    dep = lib.Det(gk, 0).fit(c.Fdep, c.Fcal_dep)
    upd = {"upd": lib.Det(gk, 7).fit(c.Ffit, c.Fcal), "later": lib.Det(gk, 11).fit(c.Flater, c.Fcal_later)}
    copies_t, copies_b = [], []
    L = int(0.6 * len(c.Fdep))
    for b in range(1, B + 1):
        rb = np.random.RandomState(1000 + b)
        boot = c.Fdep[rb.choice(len(c.Fdep), len(c.Fdep), replace=True)]
        copies_b.append(lib.Det(gk, b).fit(boot, c.Fcal_dep))
        s0 = rb.randint(0, len(c.Fdep) - L + 1)
        copies_t.append(lib.Det(gk, 100 + b).fit(c.Fdep[s0:s0 + L], c.Fcal_dep))
    return dep, upd, copies_t, copies_b


def picks(c, dep, maxcase=15):
    fl = np.where(c.ylab & dep.flags(c.Fte))[0]
    if len(fl) == 0:
        return []
    return list(fl[np.linspace(0, len(fl) - 1, min(maxcase, len(fl))).astype(int)])


def judge(c, dep, upd, f, g, k, calls, qc, dense=False):
    """Phan quyet ban sua: hop le, VaR tren hai ban cap nhat (chi khi ban do cung bao dong), cheo kien truc, kich thuoc."""
    K = c.K
    o = {h: bool(d.flags(f)[0]) for h, d in c.held.items()}
    uf = {u: bool(d.flags(f)[0]) for u, d in upd.items()}
    if g is None:
        r = dict(found=0, valid=0, cross={h: (0 if o[h] else None) for h in c.held},
                 k=None, frac=None, plaus=None, calls=int(calls), qc=int(qc))
        for u in upd:
            r["var" if u == "upd" else "var_" + u] = 0 if uf[u] else None
        return r
    if dense or k is None:
        k = int(len(np.where((np.abs(g[:K] - f[:K]) > 1e-9) | (np.abs(g[K:] - f[K:]) > 1e-9))[0]))
    r = dict(found=1, valid=int(dep.score(g)[0] <= dep.tau),
             cross={h: (int(d.score(g)[0] <= d.tau) if o[h] else None) for h, d in c.held.items()},
             k=int(k), frac=k / K, plaus=c.plaus(g), calls=int(calls), qc=int(qc))
    for u, d in upd.items():
        r["var" if u == "upd" else "var_" + u] = int(d.score(g)[0] <= d.tau) if uf[u] else None
    return r
