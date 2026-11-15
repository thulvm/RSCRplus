"""Ban 5 (ban tap chi): ba baseline ben vung gan day cho bo phat hien khong giam sat, va bo phat hien doc cua so tho.

Moi bo phat hien co diem s(x), bao dong khi s(x) > tau. De dung cac phuong phap viet cho bo phan loai xac suat,
ta dinh nghia muc "binh thuong" m(x) = sigmoid(4 (tau - s(x)) / (tau - s_med)), bang 0,5 tai nguong va
khoang 0,98 tai diem trung vi cua cua so binh thuong. Moi lan cham diem mot diem dem la mot lan goi.

  RobX   (Dutta et al., ICML 2022): do on dinh R(x) = trung binh m tren K mau Gauss quanh x tru do lech chuan;
         neu ban sua chua on dinh thi dich dan ve diem binh thuong on dinh gan nhat cho toi khi R >= tau_R.
  T-Rex  (Hamman et al., ICML 2023, ban T-Rex:I): do on dinh R(x) = trung binh [m(x_i) - |m(x) - m(x_i)|];
         leo doc theo uoc luong gradient bang chinh cac mau Gauss cho toi khi R >= tau_R.
  ROAR   (Upadhyay et al., NeurIPS 2021): xap xi tuyen tinh dia phuong kieu LIME, toi uu ban sua chiu duoc
         dich chuyen tham so trong qua cau l-vo-cung ban kinh delta, chi phi l1; giam lambda khi chua hop le.
Ca ba bat dau tu, hoac huong ve, khong gian dac trung day du nen ban sua thuong cham moi kenh (giong Wachter).
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
import lib


SLOPE = 4.0   # do doc cua phep bien doi diem thanh muc binh thuong; doi trong phan tich do nhay


def normality(det, X):
    s = det.score(X)
    z = (det.tau - s) / max(det.tau - det.med, 1e-9)
    return 1.0 / (1.0 + np.exp(-SLOPE * z))


# ---------------------------------------------------------------- RobX
def robx_stability(det, x, K, sigma, rs):
    X = x[None] + sigma * rs.randn(K, len(x))
    m = normality(det, X)
    return float(m.mean() - m.std()), K


def robx(det, f, start, normals, K=100, sigma=0.3, tau_r=0.5, n_targets=10, steps=20, rs=None):
    """start: ban sua ban dau (vd SCR); normals: cua so binh thuong de chon dich on dinh."""
    rs = rs or np.random.RandomState(0)
    q = 0
    if start is None:
        return None, q
    r, c = robx_stability(det, start, K, sigma, rs); q += c
    if r >= tau_r:
        return start, q
    # chon diem binh thuong on dinh gan ban sua nhat
    d = np.linalg.norm(normals - start, axis=1)
    target = None
    for j in np.argsort(d)[:n_targets]:
        rj, c = robx_stability(det, normals[j], K, sigma, rs); q += c
        if rj >= tau_r:
            target = normals[j]; break
    if target is None:
        return None, q
    for a in np.linspace(0, 1, steps + 1)[1:]:
        g = start + a * (target - start)
        q += 1
        if det.score(g)[0] > det.tau:
            continue
        r, c = robx_stability(det, g, K, sigma, rs); q += c
        if r >= tau_r:
            return g, q
    return target, q


# ---------------------------------------------------------------- T-Rex:I
def trex(det, f, start, K=100, sigma=0.3, tau_r=0.5, lr=0.5, iters=40, max_dist=None, rs=None):
    rs = rs or np.random.RandomState(0)
    q = 0
    if start is None:
        return None, q
    g = start.copy()
    max_dist = max_dist or 2.0 * np.linalg.norm(start - f) + 1.0
    for _ in range(iters):
        E = rs.randn(K, len(g))
        X = g[None] + sigma * E
        mi = normality(det, X); m0 = normality(det, g)[0]; q += K + 1
        vals = mi - np.abs(m0 - mi)
        R = float(vals.mean())
        if R >= tau_r and det.score(g)[0] <= det.tau:
            return g, q
        grad = (vals - vals.mean()) @ E / (K * sigma)          # uoc luong ham diem cua gradient
        nrm = np.linalg.norm(grad)
        if nrm < 1e-12:
            break
        step = g + lr * grad / nrm
        if np.linalg.norm(step - f) > max_dist:
            break
        g = step
    return None, q


# ---------------------------------------------------------------- ROAR
def roar(det, f, n_lime=200, width=1.0, delta=0.1, lam0=0.1, outer=8, inner=200, lr=0.05, rounds=5, rs=None):
    """ROAR voi tuyen tinh hoa lap: neu ban sua tren mo hinh thay the chua hop le voi bo phat hien that,
    khop lai mo hinh thay the quanh diem hien tai (toi da `rounds` lan); chi phi l1 luon tinh tu f.
    Ban goc chi tuyen tinh hoa mot lan quanh f; tuyen tinh hoa lap la thay doi co loi cho ROAR."""
    rs = rs or np.random.RandomState(0)
    q = 0; center = f.copy()
    tgt = det.tau - 0.5 * (det.tau - det.med)
    for _ in range(rounds):
        X = center[None] + width * rs.randn(n_lime, len(f))
        s = det.score(X); q += n_lime
        w8 = np.exp(-np.sum((X - center) ** 2, 1) / (2 * (0.75 * np.sqrt(len(f))) ** 2))
        sur = Ridge(alpha=1.0).fit(X, s, sample_weight=w8 + 1e-12)
        w, b = sur.coef_, sur.intercept_
        lam = lam0; best = None
        for _ in range(outer):
            x = center.copy()
            for _ in range(inner):
                u = x - f
                worst = w @ x + b + delta * (np.abs(x).sum() + 1.0)   # cuc dai theo dich chuyen tham so trong qua cau l-vo-cung
                g_loss = (w + delta * np.sign(x)) if worst > tgt else np.zeros_like(x)
                x = x - lr * (g_loss + lam * np.sign(u))
            q += 1
            if det.score(x)[0] <= det.tau:
                return x, q
            best = x
            lam *= 0.5
        center = best
    return None, q


# ---------------------------------------------------------------- bo phat hien doc cua so tho
class RawDet:
    """Bo phat hien tren cua so tho duoc trai phang (W x K), diem lon la bat thuong."""

    def __init__(self, kind, seed=0):
        self.kind, self.seed = kind, seed

    def fit(self, X, Xcal):
        X = X.reshape(len(X), -1); Xcal = Xcal.reshape(len(Xcal), -1)
        if self.kind == "rawpca":
            self.m = PCA(n_components=0.9, random_state=self.seed).fit(X)
        else:
            self.m = MLPRegressor(hidden_layer_sizes=(64, 16, 64), max_iter=150, early_stopping=True,
                                  random_state=self.seed).fit(X, X)
        sc = self.score(Xcal)
        self.tau, self.med = np.quantile(sc, lib.Q), np.median(sc)
        return self

    def score(self, X):
        X = np.asarray(X).reshape(-1, self.m.n_features_in_)
        R = self.m.inverse_transform(self.m.transform(X)) if self.kind == "rawpca" else self.m.predict(X)
        return ((X - R) ** 2).sum(1)

    def flags(self, X):
        return self.score(X) > self.tau


def raw_repair(x, guide, chans):
    g = x.copy(); g[:, chans] = guide[:, chans]
    return g


def generic_scr(det, rep, n, delta=0.0):
    lvl, q = lib.stop_level(det, delta), 0
    for k in range(1, n + 1):
        g = rep(k); q += 1
        if det.score(g[None])[0] <= lvl:
            return g, k, q
    return None, None, q


def generic_rscr(det, copies, rep, n, delta=0.25, alpha=0.1, gamma=0.05, rs=None, prior=(1.0, 1.0)):
    rs = rs or np.random.RandomState(0)
    lvl = lib.stop_level(det, delta); q = qc = 0
    for k in range(1, n + 1):
        g = rep(k); q += 1
        if det.score(g[None])[0] > lvl:
            continue
        s = m = 0; dec = None
        for i in rs.permutation(len(copies)):
            m += 1; qc += 1
            s += int(copies[i].score(g[None])[0] <= copies[i].tau)
            if lib.beta_lb(s, m, gamma, prior) >= 1 - alpha:
                dec = True; break
            if lib.beta_ub(s, m, gamma, prior) < 1 - alpha:
                dec = False; break
            rem = len(copies) - m
            if lib.beta_lb(s + rem, m + rem, gamma, prior) < 1 - alpha:
                dec = False; break
        if dec is None:
            dec = lib.beta_lb(s, m, gamma, prior) >= 1 - alpha
        if dec:
            return g, k, q + qc
    return None, None, q + qc


def generic_fullvote(det, copies, rep, n, start_k=1, alpha=0.1, gamma=0.05, prior=(0.5, 0.5)):
    q = 0
    for k in range(start_k, n + 1):
        g = rep(k); q += 1
        if det.score(g[None])[0] > det.tau:
            continue
        ok = sum(int(c.score(g[None])[0] <= c.tau) for c in copies); q += len(copies)
        if lib.beta_lb(ok, len(copies), gamma, prior) >= 1 - alpha:
            return g, k, q
    return None, None, q
