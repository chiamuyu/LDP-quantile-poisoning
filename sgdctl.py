"""RQ6(c) control: SGD-SN self-normalized CI on continuous data, log-scale and raw-scale iterates.

Honest values are made continuous by spreading each CreditCard bucket v uniformly over (v-1, v]
(a positive density everywhere in the support, as the analysis of Liu et al. (ICML'23) assumes).
  log : the paper's variant, iterate on log2(value), start at log2(B)/2
  raw : the original raw-scale iterate on value / x_q (data in units of the true p99, so the
        default step size suits the data), start at 1024 / x_q (the same start as 'log')
Fake users run CLIFF with m* = 10 x_q (noise-free answers 1[x > thr] of a holder of m*).
Usage: python3 sgdctl.py            -> results/sgdctl.json
"""
import math, json, time
import numpy as np
from numba import njit
from multiprocessing import Pool
from runexp import getpop
import runexp

U975 = 6.747


@njit(cache=True)
def _kernel(n, r, tau, z0, logscale, beta, mstar, vals, xscale, jit, seed):
    np.random.seed(seed)
    q = z0
    Qbar = 0.0
    va = 0.0
    vb = 0.0
    N = vals.shape[0]
    for k in range(1, n + 1):
        thr = 2.0 ** q if logscale else q
        if np.random.random() < beta:
            s = 1 if mstar > thr else 0
        else:
            x = (vals[np.random.randint(N)] - jit * np.random.random()) / xscale
            if np.random.random() < r:
                s = 1 if x > thr else 0
            else:
                s = 1 if np.random.random() < 0.5 else 0
        d = 2.0 / (k ** 0.51 + 100.0)
        if s == 1:
            q = q + (1 - r + 2 * tau * r) / 2 * d
        else:
            q = q - (1 + r - 2 * tau * r) / 2 * d
        Qbar = ((k - 1) * Qbar + q) / k
        va += k * k * Qbar * Qbar
        vb += k * k * Qbar
    Nn = (va - 2 * Qbar * vb + Qbar * Qbar * n * (n + 1) * (2 * n + 1) / 6.0) / n
    return Qbar, Nn


def cont_quantile(vals, q):
    """q-quantile of the continuous version (bucket v spread over (v-1, v])."""
    v = np.sort(vals)
    cnt = np.bincount(v)
    F = np.cumsum(cnt) / len(v)
    j = int(np.searchsorted(F, q))          # first bucket with F(j) >= q
    Fm = F[j - 1] if j > 0 else 0.0
    return (j - 1) + (q - Fm) / (F[j] - Fm)


POP = None


def _one(a):
    global POP
    if POP is None:
        POP = getpop('creditcard')
    vals = POP.vals.astype(np.float64)
    r = math.tanh(a['eps'] / 2)
    xq = a['xq']
    if a['mode'] in ('log', 'disc'):
        if a['mode'] == 'disc':
            xq = float(POP.quantile(a['q']))
        z0, xs, ms, truth = math.log2(POP.B) / 2, 1.0, 10 * xq, math.log2(xq)
        Qb, Nn = _kernel(a['n'], r, a['q'], z0, True, a['beta'], ms, vals, xs, 0.0 if a['mode'] == 'disc' else 1.0, a['seed'])
        tgt = math.log2(ms)
        infl = 2.0 ** Qb / xq
    else:
        z0, xs, ms, truth = 1024.0 / xq, xq, 10.0, 1.0
        Qb, Nn = _kernel(a['n'], r, a['q'], z0, False, a['beta'], ms, vals, xs, 1.0, a['seed'])
        tgt = ms
        infl = Qb
    W = U975 * math.sqrt(max(Nn, 0.0)) / a['n']
    return dict(cov=float(Qb - W <= truth <= Qb + W), covt=float(Qb - W <= tgt <= Qb + W), infl=float(infl), width=float(2 * W))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return cen - half, cen + half


def main():
    pop = getpop('creditcard')
    xq = cont_quantile(pop.vals, 0.99)
    res = {'xq_cont': xq}
    jobs, keys = [], []
    for mode in ['disc', 'log', 'raw']:
        for beta, R in ([(0.0, 1000)] if mode == 'disc' else [(0.0, 1000), (0.005, 200), (0.01, 200), (0.02, 200), (0.05, 200)]):
            for i in range(R):
                jobs.append(dict(mode=mode, n=10**6, eps=1.0, q=0.99, beta=beta, xq=xq, seed=70000 + 1000 * int(beta * 1000) + i + {'disc': 900000, 'log': 0, 'raw': 500000}[mode]))
                keys.append((mode, beta))
    t = time.time()
    with Pool(2) as p:
        outs = p.map(_one, jobs, chunksize=20)
    tmp = {}
    for k, o in zip(keys, outs):
        tmp.setdefault(k, []).append(o)
    for (mode, beta), v in tmp.items():
        c = np.array([o['cov'] for o in v]); ct = np.array([o['covt'] for o in v])
        lo, hi = wilson(int(c.sum()), len(c))
        res[f'{mode}|{beta}'] = dict(mode=mode, beta=beta, runs=len(v), cov=float(c.mean()), cov_lo=lo, cov_hi=hi, covt=float(ct.mean()),
                                     infl_mean=float(np.mean([o['infl'] for o in v])), infl_std=float(np.std([o['infl'] for o in v])),
                                     width=float(np.mean([o['width'] for o in v])))
        print(mode, beta, len(v), round(c.mean(), 3), (round(lo, 3), round(hi, 3)), round(ct.mean(), 3), round(res[f'{mode}|{beta}']['infl_mean'], 2), flush=True)
    print('time', time.time() - t)
    json.dump(res, open(f'{runexp.RES}/sgdctl.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
