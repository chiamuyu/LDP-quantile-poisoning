"""Simulation library: LDP quantile protocols, CLIFF attack, baselines, defenses.

Protocols (all epsilon-LDP):
  NBS     - noisy binary search with batched randomized response (DpNaiveNBS in Aamand et al. ICML'25)
  BAYESS  - DpBayeSS (Aamand et al. ICML'25), Bayesian screening search + padding reduction to the median
  SGD     - online LDP quantile SGD with self-normalized CI (Liu et al. ICML'23), run on log2 scale
  HH      - hierarchical histogram, OUE oracle, fan-out 4, consistency post-processing (Cormode et al. PVLDB'19)

Contamination model: every queried participant is adversarial independently with prob. beta.
Fake strategies (answer to the threshold query 1[x <= t]):
  0 none, 1 cliff (noise-free 1[t>=m*]), 2 stealthy cliff (noise-free 0 on [L,m*), honest RR of m* elsewhere),
  3 ipa (honest RR of value m*), 4 maxval (noise-free 0 for t<B), 5 random bit
"""
import numpy as np
from numba import njit
import math

DATA = 'data'
DOMAIN = {'creditcard': 2**20, 'flights': 2**12, 'power': 2**14, 'pareto': 2**20,
          'lognormal': 2**20, 'exponential': 2**20, 'uniform': 2**20}
ATT = {'none': 0, 'cliff': 1, 'scliff': 2, 'ipa': 3, 'maxval': 4, 'random': 5, 'flood': 6, 'tcliff': 7, 'curve': 8, 'fga': 9, 'lowcliff': 10, 'lowscliff': 11, 'fgac': 12, 'tcliffs': 13}


def synth(name, size=2_000_000, seed=12345, param=None):
    rng = np.random.default_rng(seed)
    B = DOMAIN.get(name, 2**20)
    if name == 'pareto':
        a = 1.5 if param is None else param
        v = rng.pareto(a, size) * 2000
    elif name == 'lognormal':
        s = 1.5 if param is None else param
        v = rng.lognormal(mean=8.0, sigma=s, size=size)
    elif name == 'exponential':
        v = rng.exponential(5000.0, size)
    elif name == 'uniform':
        v = rng.uniform(0, B * 0.8, size)
    else:
        raise ValueError(name)
    v = np.clip(np.round(v).astype(np.int64) + 1, 1, B)
    return v


class Pop:
    """Population = empirical distribution of a dataset over [1, B]."""

    def __init__(self, name, param=None, B=None):
        self.name = name
        if name in ('creditcard', 'flights', 'power'):
            v = np.load(f'{DATA}/{name}_vals.npy')
        else:
            v = synth(name, param=param)
        self.B = int(B or DOMAIN.get(name, 2**20))
        v = np.clip(v, 1, self.B)
        self.vals = np.sort(v).astype(np.int64)
        self.N = len(self.vals)

    def F(self, t):
        return np.searchsorted(self.vals, t, side='right') / self.N

    def quantile(self, q):
        # smallest t with F(t) >= q
        k = int(math.ceil(q * self.N))
        k = min(max(k, 1), self.N)
        return int(self.vals[k - 1])

    def target(self, q, kappa):
        xq = self.quantile(q)
        return int(min(self.B, max(xq + 1, round(kappa * xq))))


def consts(eps):
    rho = 1.0 / (1.0 + math.exp(eps))
    c = 1.0 / (math.exp(eps) - 1.0)
    return rho, c


def beta_star(pop, q, mstar, eps, eta=0.0):
    """Theorem 1 threshold (OPA / CLIFF)."""
    _, c = consts(eps)
    F = pop.F(mstar - 1)
    return max(0.0, (F - q + eta) / (F + c))


def beta_star_ipa(pop, q, mstar, eta=0.0):
    F = pop.F(mstar - 1)
    return max(0.0, (F - q + eta) / F)


# ----------------------------------------------------------------------------------------------
# fake answer (bit for the query 1[x <= t]) -- used by numba kernels
@njit(cache=True)
def cdf_sorted(vals, t):
    lo, hi = 0, vals.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if vals[mid] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo / vals.shape[0]


@njit(cache=True)
def fgac_s(t, mstar, beta, rho, vals):
    """FGA with a margin (FGA-C): the target CDF is CLIFF's own expected curve clipped to [0, 1];
    returns the probability that a fake user answers 1 to 1[x <= t]."""
    c = rho / (1 - 2 * rho)
    F = cdf_sorted(vals, t)
    gc = -c if t < mstar else 1 + c
    H = (1 - beta) * F + beta * gc
    if H < 0.0:
        H = 0.0
    if H > 1.0:
        H = 1.0
    g = (H - (1 - beta) * F) / beta
    if g < -c:
        g = -c
    if g > 1 + c:
        g = 1 + c
    return rho + (1 - 2 * rho) * g


@njit(cache=True)
def fga_s(t, mstar, xq, beta, rho, vals):
    """FGA (fine-grained attack of Li et al. adapted to quantiles): fake answers that make the expected debiased curve
    equal the CDF T of the target population (honest data with the mass of [x_q, m*) moved to m*), clipped to [-c, 1+c]."""
    c = rho / (1 - 2 * rho)
    F = cdf_sorted(vals, t)
    if t >= xq and t < mstar:
        T = cdf_sorted(vals, xq - 1)
    else:
        T = F
    g = (T - (1 - beta) * F) / beta
    if g < -c:
        g = -c
    if g > 1 + c:
        g = 1 + c
    return rho + (1 - 2 * rho) * g


@njit(cache=True)
def curve_s(t, mstar, L, rho):
    """P(answer 1[x<=t] = 1) on the step curve: G = 0 below a = mstar // 10, then two levels encoded in L
    (L = S2 + 10^7 * S3 with s = S / 10^7) on [a, mstar) and [mstar, B]."""
    a = mstar // 10
    if t < a:
        return rho
    Li = int(L)
    if t < mstar:
        return (Li % 10000000) / 1e7
    return (Li // 10000000) / 1e7


@njit(cache=True)
def fake_bit(kind, t, mstar, L, B, keep):
    if kind == 8:  # calibration curve (all users answer from a synthetic step curve, see stepcal.py)
        return 1 if np.random.random() < curve_s(t, mstar, L, 1.0 - keep) else 0
    if kind == 1:  # cliff
        return 1 if t >= mstar else 0
    elif kind == 2:  # stealthy cliff
        if t >= L and t < mstar:
            return 0
        b = 1 if t >= mstar else 0
        return b if np.random.random() < keep else 1 - b
    elif kind == 3:  # ipa
        b = 1 if t >= mstar else 0
        return b if np.random.random() < keep else 1 - b
    elif kind == 4:  # maxval
        return 1 if t >= B else 0
    elif kind == 5:
        return 1 if np.random.random() < 0.5 else 0
    return 0


@njit(cache=True)
def seed_nb(s):
    np.random.seed(s)


# ----------------------------------------------------------------------------------------------
# NBS: batched noisy binary search (exact aggregate simulation, statistical setting)
def run_nbs(pop, n, eps, q, beta, kind=0, mstar=0, L=0, rng=None, B=None, return_trace=False):
    B = B or pop.B
    rho, c = consts(eps)
    keep = 1 - rho
    R = int(math.ceil(math.log2(B)))
    b = n // R
    extra = n - b * R
    lo, hi = 1, B
    trace = []
    for r in range(R):
        if lo >= hi:
            break
        bs = b + (1 if r < extra else 0)
        mid = (lo + hi) // 2
        f = rng.binomial(bs, beta) if beta > 0 else 0
        k = bs - f
        a = rng.binomial(k, pop.F(mid))
        yes = rng.binomial(a, keep) + rng.binomial(k - a, rho)
        if f > 0:
            if kind == 1:
                yes += f * (1 if mid >= mstar else 0)
            elif kind == 2:
                if L <= mid < mstar:
                    pass
                else:
                    yes += rng.binomial(f, keep if mid >= mstar else rho)
            elif kind == 3:
                yes += rng.binomial(f, keep if mid >= mstar else rho)
            elif kind == 4:
                yes += 0
            elif kind == 5:
                yes += rng.binomial(f, 0.5)
            elif kind == 8:
                yes += rng.binomial(f, curve_s(mid, mstar, L, rho))
            elif kind == 9:
                yes += rng.binomial(f, fga_s(mid, mstar, L, beta, rho, pop.vals))
            elif kind == 12:
                yes += rng.binomial(f, fgac_s(mid, mstar, beta, rho, pop.vals))
            elif kind == 10:
                # LOW-CLIFF: "no" below m*, "yes" w.p. L/1e6 from m* on (a lower cliff top)
                if mid >= mstar:
                    yes += rng.binomial(f, L / 1e6)
            elif kind == 11:
                # LOW-CLIFF with S-CLIFF's honest answers below L: L encodes Lval * 10**7 + round(p_low * 1e6)
                Lv, pl = L // 10**7, (L % 10**7) / 1e6
                if mid < Lv:
                    yes += rng.binomial(f, rho)          # honest RR answer of a user holding m* (> mid)
                elif mid >= 2 * mstar:
                    yes += rng.binomial(f, keep)         # honest RR answer of a holder of m* again from 2 m* on
                elif mid >= mstar:
                    yes += rng.binomial(f, pl)
        G = (yes / bs - rho) / (1 - 2 * rho)
        if return_trace:
            trace.append((mid, G))
        if G >= q:
            hi = mid
        else:
            lo = mid + 1
    if return_trace:
        return lo, trace
    return lo


# ----------------------------------------------------------------------------------------------
# BayeSS (DpBayeSS of Aamand et al.) with padding reduction to the median.
# Segment tree with lazy multiplication over K = B intervals [t, t+1], t = 0..B-1 (coins 0..B).

@njit(cache=True)
def _st_push(sm, lz, node):
    if lz[node] != 1.0:
        for ch in (2 * node, 2 * node + 1):
            sm[ch] *= lz[node]
            lz[ch] *= lz[node]
        lz[node] = 1.0


@njit(cache=True)
def _st_mul(sm, lz, node, nl, nr, l, r, f):
    # multiply leaves in [l, r] by f
    if r < nl or nr < l or l > r:
        return
    if l <= nl and nr <= r:
        sm[node] *= f
        lz[node] *= f
        return
    _st_push(sm, lz, node)
    mid = (nl + nr) // 2
    _st_mul(sm, lz, 2 * node, nl, mid, l, r, f)
    _st_mul(sm, lz, 2 * node + 1, mid + 1, nr, l, r, f)
    sm[node] = sm[2 * node] + sm[2 * node + 1]


@njit(cache=True)
def _st_set(sm, lz, size, idx, val):
    # descend with pushes
    node = 1
    nl, nr = 0, size - 1
    path = np.empty(64, np.int64)
    d = 0
    while nl != nr:
        _st_push(sm, lz, node)
        path[d] = node
        d += 1
        mid = (nl + nr) // 2
        if idx <= mid:
            node = 2 * node
            nr = mid
        else:
            node = 2 * node + 1
            nl = mid + 1
    sm[node] = val
    for k in range(d - 1, -1, -1):
        nd = path[k]
        sm[nd] = sm[2 * nd] + sm[2 * nd + 1]


@njit(cache=True)
def _st_find(sm, lz, size, target):
    # first leaf i with prefix sum >= target; returns (i, prefix before i, leaf weight)
    node = 1
    nl, nr = 0, size - 1
    acc = 0.0
    while nl != nr:
        _st_push(sm, lz, node)
        mid = (nl + nr) // 2
        if acc + sm[2 * node] >= target:
            node = 2 * node
            nr = mid
        else:
            acc += sm[2 * node]
            node = 2 * node + 1
            nl = mid + 1
    return nl, acc, sm[node]


@njit(cache=True)
def _respond_padded(t, B, q, beta, kind, mstar, L, keep, popvals):
    # one participant of the padded population answers query 1[x <= t] under eps-RR
    u = np.random.random()
    if u < 0.5:
        if np.random.random() < beta:
            if kind == 9:
                return 1 if np.random.random() < fga_s(t, mstar, L, beta, 1.0 - keep, popvals) else 0
            if kind == 12:
                return 1 if np.random.random() < fgac_s(t, mstar, beta, 1.0 - keep, popvals) else 0
            return fake_bit(kind, t, mstar, L, B, keep)
        x = popvals[np.random.randint(popvals.shape[0])]
        b = 1 if x <= t else 0
    elif u < 0.5 + 0.5 * (1 - q):
        b = 1 if t >= 1 else 0          # virtual participant holding value 1
    else:
        b = 1 if t >= B else 0          # virtual participant holding value B
    return b if np.random.random() < keep else 1 - b


@njit(cache=True)
def _bayes_learn_tree(M, B, a, q, beta, kind, mstar, L, keep, popvals, size):
    # iterative lazy segment tree: sm = subtree sum (already including own lazy), lz = pending factor for children
    sm = np.zeros(2 * size)
    lz = np.ones(2 * size)
    K = B
    for i in range(K):
        sm[size + i] = 1.0 / K
    for nd in range(size - 1, 0, -1):
        sm[nd] = sm[2 * nd] + sm[2 * nd + 1]
    Lvis = np.empty(M, np.int64)
    d10, d11, d00, d01 = 1 + 2 * a, 1 - 2 * a, 1 - 2 * a, 1 + 2 * a
    path = np.empty(64, np.int64)
    for s in range(M):
        # ---- find weighted median interval
        node = 1
        acc = 0.0
        while node < size:
            f = lz[node]
            if f != 1.0:
                l = 2 * node
                sm[l] *= f; lz[l] *= f
                sm[l + 1] *= f; lz[l + 1] *= f
                lz[node] = 1.0
            l = 2 * node
            if acc + sm[l] >= 0.5:
                node = l
            else:
                acc += sm[l]
                node = l + 1
        i = node - size
        if i >= K:
            i = K - 1
            node = size + i
        Wp = acc
        wi = sm[node]
        if s == 0:
            i = K // 2
            node = size + i
            Wp = i / K
            wi = 1.0 / K
        coin = i if (0.5 - Wp) / max(wi, 1e-300) <= 0.5 else i + 1
        Lvis[s] = i
        y = _respond_padded(coin, B, q, beta, kind, mstar, L, keep, popvals)
        dl = d10 if y == 1 else d00
        dr = d11 if y == 1 else d01
        newi = dl * (0.5 - Wp) + dr * (Wp + wi - 0.5)
        if newi < 0.0:
            newi = 0.0
        # ---- update: siblings left of path *= dl, right of path *= dr, leaf i := newi
        node = 1
        nl = 0
        nr = size - 1
        d = 0
        while node < size:
            f = lz[node]
            if f != 1.0:
                l = 2 * node
                sm[l] *= f; lz[l] *= f
                sm[l + 1] *= f; lz[l + 1] *= f
                lz[node] = 1.0
            path[d] = node
            d += 1
            mid = (nl + nr) // 2
            l = 2 * node
            if i <= mid:
                sm[l + 1] *= dr; lz[l + 1] *= dr
                node = l
                nr = mid
            else:
                sm[l] *= dl; lz[l] *= dl
                node = l + 1
                nl = mid + 1
        sm[node] = newi
        for k in range(d - 1, -1, -1):
            nd = path[k]
            sm[nd] = sm[2 * nd] + sm[2 * nd + 1]
        tot = sm[1]
        if tot > 0:
            sm[1] = 1.0
            lz[1] /= tot
    return Lvis


@njit(cache=True)
def _bayes_learn_list(M, coinsL, coinsR, a, q, beta, kind, mstar, L, B, keep, popvals):
    # small-list BayesLearn over intervals [coinsL[j], coinsR[j]] (O(K) per step)
    K = coinsL.shape[0]
    w = np.ones(K) / K
    Lvis = np.empty(M, np.int64)
    d10, d11, d00, d01 = 1 + 2 * a, 1 - 2 * a, 1 - 2 * a, 1 + 2 * a
    for s in range(M):
        acc = 0.0
        i = K - 1
        for j in range(K):
            if acc + w[j] >= 0.5:
                i = j
                break
            acc += w[j]
        Wp = acc
        wi = w[i]
        coin = coinsL[i] if (0.5 - Wp) / max(wi, 1e-300) <= 0.5 else coinsR[i]
        Lvis[s] = i
        y = _respond_padded(coin, B, q, beta, kind, mstar, L, keep, popvals)
        dl = d10 if y == 1 else d00
        dr = d11 if y == 1 else d01
        newi = dl * (0.5 - Wp) + dr * (Wp + wi - 0.5)
        for j in range(i):
            w[j] *= dl
        for j in range(i + 1, K):
            w[j] *= dr
        w[i] = max(newi, 0.0)
        tot = w.sum()
        w /= tot
    return Lvis


@njit(cache=True)
def _nbs_coins(coins, Ms, q, beta, kind, mstar, L, B, keep, rho, popvals):
    # binary search over a sorted coin list for the first coin with debiased padded CDF >= 1/2
    K = coins.shape[0]
    lo, hi = 0, K - 1
    R = max(1, int(math.ceil(math.log2(max(K, 2)))))
    per = max(1, Ms // R)
    for r in range(R):
        if lo >= hi:
            break
        mid = (lo + hi) // 2
        t = coins[mid]
        yes = 0
        for k in range(per):
            yes += _respond_padded(t, B, q, beta, kind, mstar, L, keep, popvals)
        G = (yes / per - rho) / (1 - 2 * rho)
        if G >= 0.5:
            hi = mid
        else:
            lo = mid + 1
    return coins[lo]


def run_bayess(pop, n, eps, q, beta, kind=0, mstar=0, L=0, seed=0, B=None, cconst=0.6):
    B = B or pop.B
    seed_nb(seed)
    rho, _ = consts(eps)
    keep = 1 - rho
    M = 2 * n  # padded population: n real + n virtual participants
    lnB = math.log(B)
    den = lnB + math.log(lnB) + 1
    MB1 = int(M * lnB / den)
    MB2 = int(M * math.log(lnB) / den)
    a = min(0.24, cconst * math.sqrt(lnB / M))
    size = 1
    while size < B:
        size *= 2
    Lv = _bayes_learn_tree(MB1, B, a, q, beta, kind, mstar, L, keep, pop.vals, size)
    gamma = 1.0 / (lnB ** 2)
    qs = [gamma * i for i in range(1, int(math.floor(1 / gamma)) + 1)]
    R = sorted(set(int(round(np.quantile(Lv, x))) for x in qs))
    used = MB1
    if len(R) > 13:
        cl = np.array([R[0] - 0] + R + [R[-1]], dtype=np.int64)
        # intervals: [0, R0], [R_j, R_j+1] ... , [R_last+1, B]
        coinsL = np.array([0] + R + [R[-1] + 1], dtype=np.int64)
        coinsR = np.array([R[0]] + [r + 1 for r in R] + [B], dtype=np.int64)
        Lv2 = _bayes_learn_list(MB2, coinsL, coinsR, a, q, beta, kind, mstar, L, B, keep, pop.vals)
        g2 = 1 / 13
        qs2 = [g2 * i for i in range(1, 14)]
        idx = sorted(set(int(round(np.quantile(Lv2, x))) for x in qs2))
        coins = sorted(set([int(coinsL[j]) for j in idx] + [int(coinsR[j]) for j in idx]))
        used += MB2
    else:
        coins = sorted(set(R + [r + 1 for r in R]))
    coins = np.array(coins, dtype=np.int64)
    Ms = M - used
    return int(_nbs_coins(coins, Ms, q, beta, kind, mstar, L, B, keep, rho, pop.vals))


# ----------------------------------------------------------------------------------------------
# SGD with self-normalized inference (Liu et al. ICML'23), on z = log2(value)
@njit(cache=True)
def _sgd_kernel(n, r, tau, z0, scale, beta, kind, mstar, L, B, popvals):
    keep = (1 + r) / 2
    q = z0
    Qbar = 0.0
    va = 0.0
    vb = 0.0
    for k in range(1, n + 1):
        thr = 2.0 ** q
        if np.random.random() < beta:
            # fake: produce answer s = 1[x > thr]
            if kind == 1:
                s = 1 if mstar > thr else 0
            elif kind == 2:
                if thr >= L and thr < mstar:
                    s = 1
                else:
                    s0 = 1 if mstar > thr else 0
                    s = s0 if np.random.random() < keep else 1 - s0
            elif kind == 3:
                s0 = 1 if mstar > thr else 0
                s = s0 if np.random.random() < keep else 1 - s0
            elif kind == 4:
                s = 1 if B > thr else 0
            elif kind == 8:
                # calibration curve: P(1[x <= thr] = 1) = curve_s; s reports the complement 1[x > thr]
                s = 0 if np.random.random() < curve_s(thr, mstar, L, (1 - r) / 2) else 1
            elif kind == 9:
                s = 0 if np.random.random() < fga_s(thr, mstar, L, beta, (1 - r) / 2, popvals) else 1
            elif kind == 12:
                s = 0 if np.random.random() < fgac_s(thr, mstar, beta, (1 - r) / 2, popvals) else 1
            else:
                s = 1 if np.random.random() < 0.5 else 0
        else:
            x = popvals[np.random.randint(popvals.shape[0])]
            u = np.random.random()
            v = 1 if np.random.random() < 0.5 else 0
            if u < r:
                s = 1 if x > thr else 0
            else:
                s = v
        d = scale * 2.0 / (k ** 0.51 + 100.0)
        if s == 1:
            q = q + (1 - r + 2 * tau * r) / 2 * d
        else:
            q = q - (1 + r - 2 * tau * r) / 2 * d
        Qbar = ((k - 1) * Qbar + q) / k
        va += k * k * Qbar * Qbar
        vb += k * k * Qbar
    Nn = (va - 2 * Qbar * vb + Qbar * Qbar * n * (n + 1) * (2 * n + 1) / 6.0) / n
    return Qbar, Nn


U975 = 6.747


def run_sgd(pop, n, eps, q, beta, kind=0, mstar=0, L=0, seed=0, B=None, scale=1.0):
    B = B or pop.B
    seed_nb(seed)
    r = math.tanh(eps / 2)
    z0 = math.log2(B) / 2
    Qbar, Nn = _sgd_kernel(n, r, q, z0, scale, beta, kind, float(mstar), float(L), float(B), pop.vals)
    W = U975 * math.sqrt(max(Nn, 0.0)) / n
    est = int(min(B, max(1, math.ceil(2.0 ** Qbar))))
    return est, Qbar, Qbar - W, Qbar + W


# ----------------------------------------------------------------------------------------------
# HH: hierarchical histogram with OUE, fan-out 4, consistency (Hay et al.) -- aggregate simulation
class HHCache:
    def __init__(self, pop, fan=4):
        self.fan = fan
        B = pop.B
        h = int(round(math.log(B, fan)))
        assert fan ** h == B
        self.h = h
        # leaf probabilities (values 1..B -> leaf index 0..B-1)
        cnt = np.bincount(pop.vals - 1, minlength=B).astype(np.float64)
        self.leafp = cnt / cnt.sum()
        self.levelp = []
        for l in range(1, h + 1):
            width = fan ** (h - l)
            self.levelp.append(self.leafp.reshape(-1, width).sum(axis=1))


def _consistency(levels, fan):
    # levels[0] = root (exact), levels[l] has fan**l nodes; equal variance assumed (Hay et al.)
    h = len(levels) - 1
    fbar = [None] * (h + 1)
    fbar[h] = levels[h].copy()
    for l in range(h - 1, 0, -1):
        i = h - l + 1  # height counting leaves as 1
        child = fbar[l + 1].reshape(-1, fan).sum(axis=1)
        wa = (fan ** i - fan ** (i - 1)) / (fan ** i - 1)
        wb = (fan ** (i - 1) - 1) / (fan ** i - 1)
        fbar[l] = wa * levels[l] + wb * child
    fhat = [None] * (h + 1)
    fhat[0] = np.array([1.0])
    for l in range(1, h + 1):
        sums = fbar[l].reshape(-1, fan).sum(axis=1)
        diff = (fhat[l - 1] - sums) / fan
        fhat[l] = fbar[l] + np.repeat(diff, fan)
    return fhat


def run_hh(pop, cache, n, eps, q, beta, kind=0, mstar=0, L=0, rng=None, ci=True, return_cdf=False, audit=False):
    fan, h = cache.fan, cache.h
    po, qo = 0.5, 1.0 / (1.0 + math.exp(eps))
    levels = [np.array([1.0])]
    zmin = np.inf
    per = rng.multinomial(n, [1.0 / h] * h)
    for l in range(1, h + 1):
        Nl = int(per[l - 1])
        f = rng.binomial(Nl, beta) if beta > 0 else 0
        k = Nl - f
        probs = cache.levelp[l - 1]
        cnt = rng.multinomial(k, probs)
        ones = rng.binomial(cnt, po) + rng.binomial(k - cnt, qo)
        ones = ones.astype(np.float64)
        if f > 0:
            nn = fan ** l
            width = fan ** (h - l)
            if kind in (1, 3, 4):
                tgt = (pop.B if kind == 4 else mstar)
                node = (tgt - 1) // width
                if kind == 1 or kind == 4:      # noise-free point mass at target (one-hot, no flips)
                    ones[node] += f
                else:                            # IPA: honest OUE report of the target value
                    ones[node] += rng.binomial(f, po)
                    oth = rng.binomial(f, qo, size=nn)
                    oth[node] = 0
                    ones += oth
            elif kind == 2:   # stealthy (count-camouflaged) cliff: own bit 1, nodes left of m* zero, others ~ q_o
                node = (mstar - 1) // width
                ones[node] += f
                right = rng.binomial(f, qo, size=nn)
                right[:node + 1] = 0
                ones += right
            elif kind == 5:
                ones += rng.binomial(f, 0.5, size=fan ** l)
            elif kind == 6:   # flood: nodes left of m* -> 0, node of m* and right -> 1 (noise-free)
                node = (mstar - 1) // width
                ones[node:] += f
            elif kind == 7:   # tree-consistent cliff: inject the signed measure -c_H*delta_1 + (1+c_H)*delta_{m*}
                # expected debiased node value w = (s - qo)/(po - qo); s = 0 -> -c_H, s = po + qo -> 1 + c_H,
                # s = po -> 1 (node holds both atoms), s = qo -> 0 (all other nodes). Consistent across levels.
                node = (mstar - 1) // width
                bg = rng.binomial(f, qo, size=nn)
                if node == 0:
                    bg[0] = rng.binomial(f, po)
                else:
                    bg[0] = 0
                    bg[node] = rng.binomial(f, po + qo)
                ones += bg
            elif kind == 13:  # spread T-CLIFF: signed measure -c_H*Unif[1,L] + (1+c_H)*delta_{m*}, c_H = q_o/(p_o-q_o)
                a_ = po - qo
                cH = qo / a_
                lo_leaf = np.arange(nn) * width + 1            # node covers leaves [lo_leaf, lo_leaf+width-1]
                hi_leaf = lo_leaf + width - 1
                ov = np.clip(np.minimum(hi_leaf, L) - lo_leaf + 1, 0, None) / float(L)
                w = -cH * ov
                w[(mstar - 1) // width] += 1.0 + cH
                sprob = np.clip(qo + a_ * w, 0.0, 1.0)
                ones += rng.binomial(f, sprob)
        est = (ones / max(Nl, 1) - qo) / (po - qo)
        if audit:
            se = math.sqrt(qo * (1 - qo) / max(Nl, 1)) / (po - qo)
            zmin = min(zmin, float(est.min()) / se)
        levels.append(est)
    if ci:
        fhat = _consistency(levels, fan)
        leaves = fhat[h]
    else:
        leaves = levels[h]
    cdf = np.cumsum(leaves)
    idx = np.argmax(cdf >= q) if np.any(cdf >= q) else pop.B - 1
    est = int(idx + 1)
    if audit:
        return est, zmin
    if return_cdf:
        return est, cdf
    return est
