"""Tenth follow-up: stealth of T-CLIFF on HH.
  hhaudit : T-CLIFF (negative mass at value 1) vs. spread T-CLIFF (negative mass spread over [1, x_q/2], as S-CLIFF
            assumes a rough lower bound on x_q) on HH, with a distribution-free non-negativity audit on the raw node
            estimates: flag if some node is below -z* standard errors, z* = Bonferroni at 1% over all nodes.
            n = 1e7 (beta 1%, 1.5%) and 1e8 (beta 0.75%), eps = 1, q = 0.99, kappa = 10, 200 runs per cell.  -> hhaudit.json
Usage: python3 experiments12.py hhaudit
"""
import sys, time, json, math
import numpy as np
from scipy.stats import norm
from experiments import log, TOL
from runexp import getpop, save
from ldpq import run_hh, HHCache


def hhaudit():
    res = {}
    R = 200
    for d in ['creditcard', 'flights', 'power', 'pareto']:
        pop = getpop(d)
        cache = HHCache(pop)
        ms = pop.target(0.99, 10)
        xq = pop.quantile(0.99)
        L = max(1, xq // 2)
        nodes = sum(cache.fan ** l for l in range(1, cache.h + 1))
        zs = norm.ppf(1 - 0.01 / nodes)
        for n, betas in [(10**7, [0.0, 0.01, 0.015]), (10**8, [0.0, 0.0075])]:
            for beta in betas:
                for kind, nm in ([(0, 'clean')] if beta == 0 else [(7, 'tcliff'), (13, 'tcliffs')]):
                    rng = np.random.default_rng(1000 * kind + int(beta * 1e4) + (7 if n > 10**7 else 0))
                    out = [run_hh(pop, cache, n, 1.0, 0.99, beta, kind, ms, L, rng=rng, audit=True) for _ in range(R)]
                    m = np.array([o[0] for o in out], float)
                    z = np.array([o[1] for o in out])
                    key = f'{d}|{n}|{beta}|{nm}'
                    res[key] = dict(data=d, n=n, beta=beta, attack=nm, L=int(L), zstar=float(zs), runs=R,
                                    hit=float(np.mean(np.abs(m - ms) <= TOL * ms)), exact=float(np.mean(m == ms)),
                                    flag=float(np.mean(z < -zs)), zmin_med=float(np.median(z)))
                    log('hhaudit', key, res[key]['hit'], res[key]['exact'], res[key]['flag'], round(res[key]['zmin_med'], 1))
    save('hhaudit', res)


def hhaudit_big():
    """spread T-CLIFF at larger beta*sqrt(n) on Flights: the audit's power grows with beta*sqrt(n)."""
    pop = getpop('flights')
    cache = HHCache(pop)
    ms = pop.target(0.99, 10)
    L = max(1, pop.quantile(0.99) // 2)
    nodes = sum(cache.fan ** l for l in range(1, cache.h + 1))
    zs = norm.ppf(1 - 0.01 / nodes)
    res = {}
    for n, beta, R in [(10**8, 0.015, 200), (10**9, 0.0075, 100), (10**8, 0.0, 200)]:
        rng = np.random.default_rng(4242 + int(beta * 1e5) + n % 997)
        out = [run_hh(pop, cache, n, 1.0, 0.99, beta, 13 if beta > 0 else 0, ms, L, rng=rng, audit=True) for _ in range(R)]
        z = np.array([o[1] for o in out]); m = np.array([o[0] for o in out], float)
        key = f'flights|{n}|{beta}|' + ('tcliffs' if beta > 0 else 'clean')
        res[key] = dict(n=n, beta=beta, runs=R, flag=float(np.mean(z < -zs)), hit=float(np.mean(np.abs(m - ms) <= TOL * ms)))
        log('hhaudit_big', key, res[key])
    save('hhaudit_big', res)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        log('start', name)
        globals()[name]()
        log('end', name, f'{time.time() - t:.0f}s')
