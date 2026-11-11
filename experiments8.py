"""Sixth follow-up (second external check).
  jump2   : audits against CLIFF, S-CLIFF, IPA, MGA and LOW-CLIFF, where LOW-CLIFF now answers like S-CLIFF below
            L = x_q/2 (honest RR of a holder of m*), "no" on [L, m*), and "yes" with a lowered probability from m* on.
            Three audits on the outputs of NBS (CreditCard, eps = 1, beta = 1%): range audit (31 log-spaced thresholds),
            jump audit (mass of [m_hat, 1.1 m_hat]), and a boundary audit at t = 0 and t = B, where every honest
            user's debiased answer has mean 0 and 1 (half the cohort each; flag beyond 3 SE). LOW-CLIFF answers
            honestly again from 2 m* on.   -> jump2.json
  tc15    : T-CLIFF against HH at beta = 1.5% (n = 1e7, four datasets) with raw outputs (exact-hit rate)  -> tc15.json
  plateau : NBS on a population with a CDF plateau at q (mass 0.99 at x_q = 100, 0.01 at B), with the budget
            calibrated on the synthetic Pareto(1.5) of Algorithm 1 (true kappa = 10)                 -> plateau.json
  plateau_cert : the same population attacked with the budget certified by Remark 1 with eta_NBS and worst-case ranks
            max{(1-q+eta)(1-e^-eps), eta/(1+c-q)}                                                   -> plateau.json
Usage: python3 experiments8.py jump2 tc15 plateau plateau_cert
"""
import sys, math, time, json
import numpy as np
from multiprocessing import Pool
from experiments import tasks_for, batch, log, Q0, KAP0, TOL, EPS0
from runexp import getpop, summarize, save, consts, run_nbs
import runexp
import experiments4 as e4


def load(name):
    try:
        return json.load(open(f'{runexp.RES}/{name}.json'))
    except FileNotFoundError:
        return {}


def eta_nbs_step(n):
    r = load('eta_step')
    return (r[f'nbs|creditcard|{n}|1.0|0.99|203610|lo|0.1'], r[f'nbs|creditcard|{n}|1.0|0.99|203610|up|0.1'])


def fake_yes(kind, t, mstar, L, p_low, B, f, rng, rho):
    keep = 1 - rho
    if f == 0:
        return 0
    if kind == 'cliff':
        return f if t >= mstar else 0
    if kind == 'scliff':
        return 0 if L <= t < mstar else rng.binomial(f, keep if t >= mstar else rho)
    if kind == 'ipa':
        return rng.binomial(f, keep if t >= mstar else rho)
    if kind == 'maxval':
        return f if t >= B else 0
    if kind == 'low':
        if t < L:
            return rng.binomial(f, rho)
        if t >= 2 * mstar:
            return rng.binomial(f, keep)
        return rng.binomial(f, p_low) if t >= mstar else 0
    return 0


def estimate(pop, k_each, beta, kind, t, mstar, L, p_low, rng, rho):
    f = rng.binomial(k_each, beta) if beta > 0 and kind != 'none' else 0
    k = k_each - f
    a = rng.binomial(k, pop.F(t)) if t > 0 else 0
    yes = rng.binomial(a, 1 - rho) + rng.binomial(k - a, rho) + fake_yes(kind, t, mstar, L, p_low, pop.B, f, rng, rho)
    pi = yes / k_each
    G = (pi - rho) / (1 - 2 * rho)
    se = math.sqrt(max(pi * (1 - pi), 1e-12) / k_each) / (1 - 2 * rho)
    return G, se


def jump_audit(pop, n_a, q, beta, kind, mstar, L, p_low, mhat, eta, rng, rho, z=3.0):
    t1 = max(int(mhat) - 1, 1)
    t2 = min(int(1.1 * mhat), pop.B)
    k_each = n_a // 2
    g1, s1 = estimate(pop, k_each, beta, kind, t1, mstar, L, p_low, rng, rho)
    g2, s2 = estimate(pop, k_each, beta, kind, t2, mstar, L, p_low, rng, rho)
    return (g2 - g1) > (1 - q) + eta + z * math.sqrt(s1 * s1 + s2 * s2), g2 - g1


def range_audit(pop, n_a, beta, kind, mstar, L, p_low, rng, rho, grid_pts=32, z=3.0):
    grid = np.unique(np.round(np.geomspace(1, pop.B - 1, grid_pts)).astype(np.int64))
    k_each = n_a // len(grid)
    for t in grid:
        G, se = estimate(pop, k_each, beta, kind, int(t), mstar, L, p_low, rng, rho)
        if G < -z * se or G > 1 + z * se:
            return True
    return False


def boundary_audit(pop, n_a, beta, kind, mstar, L, p_low, rng, rho, z=3.0):
    # half the cohort at t = 0 (honest mean 0), half at t = B (honest mean 1)
    G0, se0 = estimate(pop, n_a // 2, beta, kind, 0, mstar, L, p_low, rng, rho)
    GB, seB = estimate(pop, n_a // 2, beta, kind, pop.B, mstar, L, p_low, rng, rho)
    return G0 < -z * se0 or abs(GB - 1) > z * seB


def _nbs(a):
    pop = getpop('creditcard')
    return run_nbs(pop, a['n'], EPS0, Q0, a['beta'], a['kindc'], a['mstar'], L=a['L'], rng=np.random.default_rng(a['seed']))


def jump2():
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    xq = pop.quantile(Q0)
    Lv = xq // 2
    rho, c = consts(EPS0)
    beta = 0.01
    res = {}
    kinds = ['none', 'cliff', 'scliff', 'ipa', 'maxval', 'low']
    code = {'none': 0, 'cliff': 1, 'scliff': 2, 'ipa': 3, 'maxval': 4, 'low': 11}
    R = 200
    mh = {}
    with Pool(2) as pool:
        for n in [10**6, 10**7, 10**8]:
            elo, eup = eta_nbs_step(n)
            for kind in kinds:
                v = p_low = None
                Lp = Lv if kind == 'scliff' else 0
                if kind == 'low':      # data-free: lowest top that reaches q + eta_+ at m* if F(m*) = 1
                    v = min((Q0 + eup - (1 - beta)) / beta, 1 + c)
                    p_low = rho + (1 - 2 * rho) * v
                    Lp = Lv * 10**7 + int(round(p_low * 1e6))
                args = [dict(n=n, kindc=code[kind], beta=beta if kind != 'none' else 0.0, mstar=ms, L=Lp, seed=9100 + 13 * i + n % 997)
                        for i in range(R)]
                m = np.array(pool.map(_nbs, args, chunksize=10), float)
                mh[(n, kind)] = (m, p_low)
                res[f'hit|{n}|{kind}'] = dict(n=n, kind=kind, hit=float(np.mean(np.abs(m - ms) <= TOL * ms)), exact=float(np.mean(m == ms)),
                                              ge=float(np.mean(m >= 0.9 * ms)), v=v, p_low=p_low, infl=float(np.mean(m / xq)))
                log('nbs', n, kind, round(res[f'hit|{n}|{kind}']['hit'], 3), v)
    rng = np.random.default_rng(78)
    for n, frac in [(10**6, 0.05), (10**7, 0.05), (10**7, 0.2), (10**8, 0.2)]:
        elo, eup = eta_nbs_step(n)
        n_a = int(n * frac)
        for kind in kinds:
            m, p_low = mh[(n, kind)]
            b = beta if kind != 'none' else 0.0
            fj, fr, fb, Ds = [], [], [], []
            for mhat in m:
                fl_, D_ = jump_audit(pop, n_a, Q0, b, kind, ms, Lv, p_low, mhat, elo, rng, rho)
                fj.append(fl_); Ds.append(D_)
                fr.append(range_audit(pop, n_a, b, kind, ms, Lv, p_low, rng, rho))
                fb.append(boundary_audit(pop, n_a, b, kind, ms, Lv, p_low, rng, rho))
            out = dict(n=n, frac=frac, kind=kind, D_mean=float(np.mean(Ds)), D_std=float(np.std(Ds)), thr=(1 - Q0) + elo)
            for nm, arr in [('jump', fj), ('range', fr), ('bound', fb)]:
                arr = np.array(arr, float)
                bl = np.array([x.mean() for x in np.array_split(arr, 5)])
                out[nm + '_mean'] = float(bl.mean()); out[nm + '_std'] = float(bl.std())
            res[f'audit|{n}|{frac}|{kind}'] = out
            log('audit', n, frac, kind, round(out['range_mean'], 3), round(out['jump_mean'], 3), round(out['bound_mean'], 3))
    res['meta'] = dict(mstar=ms, xq=xq, L=Lv, c=c, beta=beta)
    save('jump2', res)


def tc15():
    res = {}
    cfgs = []
    for data in ['creditcard', 'flights', 'power', 'pareto']:
        pop = getpop(data)
        ms = pop.target(Q0, KAP0)
        cfgs.append((data, tasks_for('hh', data, 10**7, EPS0, Q0, 0.015, 'tcliff', ms, seed0=315)))
    outs = batch(cfgs)
    for data, o in outs.items():
        pop = getpop(data)
        ms = pop.target(Q0, KAP0)
        m = np.array([x['m'] for x in o])
        s = summarize(o, pop, Q0, ms, TOL)
        s.update(exact=float(np.mean(m == ms)), mstar=ms)
        res[data] = s
        log('tc15', data, round(s['hit_mean'], 3), 'exact', s['exact'])
    save('tc15', res)


class Plateau:
    """Honest mass 0.99 at x_q = 100 and 0.01 at B (CDF plateau at q on [100, B))."""
    def __init__(self, B=2**20):
        self.B = B
        self.vals = np.array([100] * 99 + [B])

    def F(self, t):
        t = np.asarray(t)
        return np.where(t >= self.B, 1.0, np.where(t >= 100, 0.99, 0.0))


def _nbs_plateau(a):
    return run_nbs(Plateau(), a['n'], EPS0, Q0, a['beta'], 1, a['mstar'], rng=np.random.default_rng(a['seed']))


def plateau():
    B = 2**20
    ms = 1000
    res = {}
    with Pool(2) as pool:
        lo_b, hi_b = 0.0, 0.06
        spop = ('sur', B, 100)
        for s in range(12):
            mid = (lo_b + hi_b) / 2
            m = e4.batch(pool, dict(proto='nbs', pop=spop, n=10**6, eps=EPS0, q=Q0, beta=mid, kind='cliff', mstar=ms), 200, 71 + s)
            if np.mean(np.abs(m - ms) <= TOL * ms) >= 0.95:
                hi_b = mid
            else:
                lo_b = mid
        res['plan'] = hi_b
        for mult in [1.0, 1.25]:
            b = hi_b * mult
            args = [dict(n=10**6, beta=b, mstar=ms, seed=5000 + i) for i in range(2000)]
            m = np.array(pool.map(_nbs_plateau, args, chunksize=50), float)
            ms_syn = e4.batch(pool, dict(proto='nbs', pop=spop, n=10**6, eps=EPS0, q=Q0, beta=b, kind='cliff', mstar=ms), 2000, 900 + int(mult * 10))
            res[f'mult|{mult}'] = dict(beta=b, hit_plateau=float(np.mean(np.abs(m - ms) <= TOL * ms)),
                                       hit_synthetic=float(np.mean(np.abs(ms_syn - ms) <= TOL * ms)))
            log('plateau', mult, round(100 * b, 4), res[f'mult|{mult}'])
    save('plateau', res)


def plateau_cert():
    from experiments7 import eta_nbs
    res = load('plateau')
    B, n, ms = 2**20, 10**6, 1000
    rho, c = consts(EPS0)
    eta = eta_nbs(n, EPS0, B)
    b = max((1 - Q0 + eta) / (1 + c), eta / (1 + c - Q0))
    with Pool(2) as pool:
        args = [dict(n=n, beta=b, mstar=ms, seed=100 + i) for i in range(2000)]
        m = np.array(pool.map(_nbs_plateau, args, chunksize=50), float)
    res['cert'] = dict(eta=eta, beta=b, exact=float(np.mean(m == ms)), hit=float(np.mean(np.abs(m - ms) <= TOL * ms)))
    log('plateau_cert', res['cert'])
    save('plateau', res)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        log('start', name)
        globals()[name]()
        log('end', name, f'{time.time() - t:.0f}s')
