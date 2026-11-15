"""Thu vien thuc nghiem R-SCR: du lieu, bo phat hien, XAI, bo sinh phan thuc.

Moi so bao cao deu sinh tu ham trong tep nay; khong co so viet tay.
"""
import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.linear_model import Ridge
from pyod.models.ecod import ECOD

W, STRIDE, Q = 30, 10, 0.99


# ---------------------------------------------------------------- du lieu
def load_entity(root, name):
    tr = np.loadtxt(f"{root}/train/{name}.txt", delimiter=",")
    te = np.loadtxt(f"{root}/test/{name}.txt", delimiter=",")
    lab = np.loadtxt(f"{root}/test_label/{name}.txt").astype(int)
    segs = []
    for line in open(f"{root}/interpretation_label/{name}.txt"):
        line = line.strip()
        if not line:
            continue
        rng, ch = line.split(":")
        a, b = rng.split("-")
        segs.append((int(a), int(b), [int(c) - 1 for c in ch.split(",")]))
    return tr, te, lab, segs


def windows(X, stride=STRIDE):
    idx = np.arange(0, len(X) - W + 1, stride)
    return idx, np.stack([X[i:i + W] for i in idx])


def feats(Wn):
    """Dac trung cua so: trung binh va do lech chuan tung kenh."""
    return np.concatenate([Wn.mean(1), Wn.std(1)], axis=1)


# ---------------------------------------------------------------- bo phat hien
class Det:
    """Bo phat hien co diem cang lon cang bat thuong, nguong rieng tu tap hieu chuan."""

    def __init__(self, kind, seed=0):
        self.kind, self.seed = kind, seed

    def fit(self, F, Fcal):
        k, rs = self.kind, np.random.RandomState(self.seed)
        if k == "pca":
            self.m = PCA(n_components=0.9, random_state=self.seed).fit(F)
        elif k == "iforest":
            self.m = IsolationForest(n_estimators=100, random_state=self.seed).fit(F)
        elif k == "ocsvm":
            sub = F[rs.choice(len(F), min(2000, len(F)), replace=False)]
            self.m = OneClassSVM(nu=0.05, gamma="scale").fit(sub)
        elif k == "lof":
            sub = F[rs.choice(len(F), min(3000, len(F)), replace=False)]
            self.m = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(sub)
        elif k == "ecod":
            self.m = ECOD().fit(F)
        elif k == "ae":
            from sklearn.neural_network import MLPRegressor
            self.m = MLPRegressor(hidden_layer_sizes=(32, 8, 32), max_iter=300, early_stopping=True, random_state=self.seed).fit(F, F)
        self.tau = np.quantile(self.score(Fcal), Q)
        self.med = np.median(self.score(Fcal))
        return self

    def score(self, F):
        F = np.atleast_2d(F)
        k = self.kind
        if k == "pca":
            R = self.m.inverse_transform(self.m.transform(F))
            return ((F - R) ** 2).sum(1)
        if k == "iforest":
            return -self.m.score_samples(F)
        if k in ("ocsvm", "lof"):
            return -self.m.decision_function(F)
        if k == "ecod":
            return self.m.decision_function(F)
        if k == "ae":
            return ((F - self.m.predict(F)) ** 2).sum(1)

    def flags(self, F):
        return self.score(F) > self.tau


# ---------------------------------------------------------------- phep sua
def repair(f, base, chans, K):
    """Dua cac kenh trong chans ve muc binh thuong (ca trung binh lan do lech chuan)."""
    g = f.copy()
    for c in chans:
        g[c] = base[c]
        g[K + c] = base[K + c]
    return g


def repair_batch(f, base, masks, cand, K):
    """masks: (M, len(cand)) nhi phan, 1 = kenh duoc sua."""
    G = np.repeat(f[None], len(masks), 0)
    for j, c in enumerate(cand):
        sel = masks[:, j] == 1
        G[sel, c] = base[c]
        G[sel, K + c] = base[K + c]
    return G


# ---------------------------------------------------------------- XAI
def rank_deviation(f, base, cand, K, sd):
    d = [abs(f[c] - base[c]) / sd[c] + abs(f[K + c] - base[K + c]) / sd[K + c] for c in cand]
    return [cand[i] for i in np.argsort(-np.array(d))]


def rank_occlusion(det, f, base, cand, K):
    s0 = det.score(f)[0]
    G = np.stack([repair(f, base, [c], K) for c in cand])
    drop = s0 - det.score(G)
    return [cand[i] for i in np.argsort(-drop)], len(cand) + 1


def rank_shapley(det, f, base, cand, K):
    """Shapley chinh xac bang liet ke moi tap con cua tap ung vien."""
    m = len(cand)
    masks = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(np.int8)
    v = det.score(f)[0] - det.score(repair_batch(f, base, masks, cand, K))
    from math import factorial
    wsize = np.array([factorial(s) * factorial(m - s - 1) / factorial(m) for s in range(m)])
    size = masks.sum(1)
    idx = {tuple(r): i for i, r in enumerate(map(tuple, masks))}
    phi = np.zeros(m)
    for i in range(2 ** m):
        r = masks[i]
        for j in range(m):
            if r[j] == 0:
                r2 = r.copy(); r2[j] = 1
                phi[j] += wsize[size[i]] * (v[idx[tuple(r2)]] - v[i])
    return [cand[i] for i in np.argsort(-phi)], 2 ** m + 1


def rank_ig(det, f, base, cand, K, steps=16, h=0.05):
    """Integrated Gradients theo nhom kenh, dao ham bang sai phan huu han (dung cho ca Isolation Forest)."""
    att = np.zeros(len(cand))
    pts = []
    for t in (np.arange(steps) + 0.5) / steps:
        x_t = base + t * (f - base)
        pts.append(x_t)
        for c in cand:
            y = x_t.copy()
            y[c] -= h * (f[c] - base[c]); y[K + c] -= h * (f[K + c] - base[K + c])
            pts.append(y)
    S = det.score(np.stack(pts)).reshape(steps, len(cand) + 1)
    att = ((S[:, :1] - S[:, 1:]) / h).mean(0)
    return [cand[i] for i in np.argsort(-att)], steps * (len(cand) + 1)


def rank_lime(det, f, base, cand, K, n=200, seed=0):
    rs = np.random.RandomState(seed)
    masks = rs.binomial(1, 0.5, (n, len(cand))).astype(np.int8)
    y = det.score(repair_batch(f, base, masks, cand, K))
    dist = masks.sum(1) / len(cand)
    w = np.exp(-(dist ** 2) / 0.25 ** 2 * 0.5) + 1e-6
    coef = Ridge(alpha=1.0).fit(masks, y, sample_weight=w).coef_
    # sua kenh co loi lam GIAM diem, nen kenh co he so am nhat dung dau
    return [cand[i] for i in np.argsort(coef)], n


class Granger:
    """Thu tu phan du cua mo hinh tu hoi quy vec to bac 3 hoc tren du lieu binh thuong."""

    def __init__(self, Xtr, p=3, n=6000, seed=0):
        rs = np.random.RandomState(seed)
        t = rs.choice(np.arange(p, len(Xtr)), min(n, len(Xtr) - p), replace=False)
        Z = np.concatenate([Xtr[t - l] for l in range(1, p + 1)], 1)
        self.p = p
        self.m = Ridge(alpha=1.0).fit(Z, Xtr[t])
        self.sd = (Xtr[t] - self.m.predict(Z)).std(0) + 1e-6

    def rank(self, Xte, start, cand):
        p = self.p
        t = np.arange(max(start, p), start + W)
        Z = np.concatenate([Xte[t - l] for l in range(1, p + 1)], 1)
        r = np.abs(Xte[t] - self.m.predict(Z)) / self.sd
        sc = r.mean(0)[cand]
        return [cand[i] for i in np.argsort(-sc)]


def borda(rankings):
    pts = {}
    for r in rankings:
        n = len(r)
        for i, c in enumerate(r):
            pts[c] = pts.get(c, 0) + (n - i)
    return sorted(pts, key=lambda c: -pts[c])


# ---------------------------------------------------------------- bo sinh phan thuc
def stop_level(det, delta):
    return det.tau - delta * (det.tau - det.med)


def scr(det, f, base, order, K, delta=0.0, start=1):
    """Sua dan tung kenh theo thu tu; tra ve ban sua va so kenh, hoac None."""
    lvl, q = stop_level(det, delta), 0
    for k in range(start, len(order) + 1):
        g = repair(f, base, order[:k], K); q += 1
        if det.score(g)[0] <= lvl:
            return g, k, q
    return None, len(order), q


def p_scr(det, f, guide, order, K):
    """Thay kenh bang gia tri cua cua so binh thuong gan nhat (khuyen nghi bam du lieu that)."""
    q = 0
    for k in range(1, len(order) + 1):
        g = repair(f, guide, order[:k], K); q += 1
        if det.score(g)[0] <= det.tau:
            return g, k, q
    return None, len(order), q


def native_guide(det, f, guide, K):
    """Native-Guide muc kenh: sua cac kenh khac nguide nhieu nhat truoc."""
    d = np.abs(f[:K] - guide[:K]) + np.abs(f[K:] - guide[K:])
    return p_scr(det, f, guide, list(np.argsort(-d)), K)


def comte(det, f, guide, K, maxk=None):
    """CoMTE: moi buoc chon kenh hoan doi tu mau binh thuong lam diem giam nhieu nhat."""
    chosen, q, cur = [], 0, f.copy()
    left = list(range(K))
    maxk = maxk or K
    while left and len(chosen) < maxk:
        G = np.stack([repair(cur, guide, [c], K) for c in left]); q += len(left)
        s = det.score(G); j = int(np.argmin(s))
        chosen.append(left.pop(j)); cur = G[j]
        if s[j] <= det.tau:
            return cur, len(chosen), q
    return None, len(chosen), q


def wachter_line(det, f, base, iters=20):
    """Wachter kieu tim theo duong thang: dich ca vec to ve muc binh thuong it nhat de qua nguong."""
    lo, hi, q = 0.0, 1.0, 1          # tinh ca lan cham diem muc binh thuong
    if det.score(base)[0] > det.tau:
        return None, None, q
    for _ in range(iters):
        mid = (lo + hi) / 2; q += 1
        if det.score(base + (1 - mid) * (f - base))[0] <= det.tau:
            hi = mid
        else:
            lo = mid
    return base + (1 - hi) * (f - base), None, q


def beta_lower(s, n, gamma=0.05):
    return beta_dist.ppf(gamma, 1 + s, 1 + n - s)


def beta_upper(s, n, gamma=0.05):
    return beta_dist.ppf(1 - gamma, 1 + s, 1 + n - s)


def betarce_scr(det, copies, f, base, order, K, alpha=0.1, gamma=0.05):
    """BetaRCE (KDD 2025) ban cai lai: bat dau tu ban sua SCR, hoi DU moi phien ban,
    them kenh theo thu tu cho toi khi can duoi Beta dat 1 - alpha."""
    g, k, q = scr(det, f, base, order, K)
    if g is None:
        return None, k, q, 0
    qc = 0
    for kk in range(k, len(order) + 1):
        g = repair(f, base, order[:kk], K)
        ok = np.array([c.score(g)[0] <= c.tau for c in copies]); qc += len(copies)
        if beta_lower(ok.sum(), len(ok), gamma) >= 1 - alpha:
            return g, kk, q, qc
    return None, len(order), q, qc


def rscr(det, copies, f, base, order, K, deltas=(0.0, 0.25, 0.5), alpha=0.1, gamma=0.05,
         rs=None, track_full=False):
    """R-SCR: sua dan tung kenh co bien, tham do tuan tu cac phien ban va dung som theo can Beta."""
    rs = rs or np.random.RandomState(0)
    q, qc, disagree, tried = 0, 0, 0, 0
    k0 = 1
    for k in range(1, len(order) + 1):
        for d in deltas:
            g = repair(f, base, order[:k], K); q += 1
            if det.score(g)[0] > stop_level(det, d):
                continue
            tried += 1
            perm = rs.permutation(len(copies)); s = n = 0; dec = None
            for i in perm:
                n += 1; qc += 1
                s += int(copies[i].score(g)[0] <= copies[i].tau)
                if beta_lower(s, n, gamma) >= 1 - alpha:
                    dec = True; break
                if beta_upper(s, n, gamma) < 1 - alpha:
                    dec = False; break
            if dec is None:
                dec = beta_lower(s, n, gamma) >= 1 - alpha
            if track_full:
                okf = sum(int(c.score(g)[0] <= c.tau) for c in copies)
                full = beta_lower(okf, len(copies), gamma) >= 1 - alpha
                disagree += int(full != dec)
            if dec:
                return dict(g=g, k=k, q=q, qc=qc, delta=d, disagree=disagree, tried=tried)
    return dict(g=None, k=len(order), q=q, qc=qc, delta=None, disagree=disagree, tried=tried)


# ================================================================ ban 2 (sau ra soat doc lap)
def beta_lb(s, n, gamma, prior=(1.0, 1.0)):
    return beta_dist.ppf(gamma, prior[0] + s, prior[1] + n - s)


def beta_ub(s, n, gamma, prior=(1.0, 1.0)):
    return beta_dist.ppf(1 - gamma, prior[0] + s, prior[1] + n - s)


def rscr_v2(det, copies, f, base, order, K, delta=0.25, alpha=0.1, gamma=0.05, rs=None, track_full=False,
            prior=(1.0, 1.0), curtail=True):
    """R-SCR+ ban 2: moi ung vien (k kenh dau) duoc tham do DUNG MOT LAN.
    Ung vien chi duoc dua di tham do khi diem duoi muc dung co bien; bi bac thi sua them mot kenh."""
    rs = rs or np.random.RandomState(0)
    lvl = stop_level(det, delta)
    q = qc = disagree = tried = 0
    for k in range(1, len(order) + 1):
        g = repair(f, base, order[:k], K); q += 1
        if det.score(g)[0] > lvl:
            continue
        tried += 1
        s = n = 0; dec = None
        for i in rs.permutation(len(copies)):
            n += 1; qc += 1
            s += int(copies[i].score(g)[0] <= copies[i].tau)
            if beta_lb(s, n, gamma, prior) >= 1 - alpha:
                dec = True; break
            if beta_ub(s, n, gamma, prior) < 1 - alpha:
                dec = False; break
            # cat ngan (ban 3, sau ra soat): neu ke ca moi ban sao con lai deu chap nhan ma can duoi van khong dat
            # thi khong the chap nhan nua, dung luon. Quyet dinh giu nguyen, chi bot lan goi.
            rem = len(copies) - n
            if curtail and beta_lb(s + rem, n + rem, gamma, prior) < 1 - alpha:
                dec = False; break
        if dec is None:
            dec = beta_lb(s, n, gamma, prior) >= 1 - alpha
        if track_full:
            okf = sum(int(c.score(g)[0] <= c.tau) for c in copies)
            disagree += int((beta_lb(okf, len(copies), gamma, prior) >= 1 - alpha) != dec)
        if dec:
            return dict(g=g, k=k, q=q, qc=qc, disagree=disagree, tried=tried)
    return dict(g=None, k=None, q=q, qc=qc, disagree=disagree, tried=tried)


def full_vote(det, copies, f, base, order, K, delta=0.0, alpha=0.1, gamma=0.05, start_k=1,
              prior=(0.5, 0.5)):
    """Kiem dinh kieu BetaRCE: hoi DU moi ban sao, tien nghiem Jeffreys nhu ma goc; them kenh toi khi dat."""
    lvl = stop_level(det, delta)
    q = qc = 0
    for k in range(start_k, len(order) + 1):
        g = repair(f, base, order[:k], K); q += 1
        if det.score(g)[0] > lvl:
            continue
        ok = sum(int(c.score(g)[0] <= c.tau) for c in copies); qc += len(copies)
        if beta_lb(ok, len(copies), gamma, prior) >= 1 - alpha:
            return dict(g=g, k=k, q=q, qc=qc)
    return dict(g=None, k=None, q=q, qc=qc)
