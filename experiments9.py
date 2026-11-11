"""Seventh follow-up (third external check).
  fgac  : FGA whose target CDF is CLIFF's own expected curve clipped to [0, 1] (FGA-C, attack kind 12) on the RQ3 budget grid
          (CreditCard, n = 1e6, kappa = 10; same runs per protocol as RQ3)                              -> rq3fgac.json
  valid : independent validation of every empirical budget of Table III (NBS, n = 1e7, CreditCard): 1000 fresh runs at
          the reported mean budget; hit rate with a 95% Wilson interval                                  -> valid.json
Usage: python3 experiments9.py fgac valid
"""
import sys, math, time, json
import numpy as np
from multiprocessing import Pool
from experiments import tasks_for, batch, log, Q0, N0, EPS0, KAP0, TOL
from runexp import getpop, summarize, save, run_nbs
import runexp


def load(name):
    try:
        return json.load(open(f'{runexp.RES}/{name}.json'))
    except FileNotFoundError:
        return {}


def fgac():
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    bs = load('rq3')['beta_star']
    res = {'beta_star': bs}
    cfgs = []
    for proto in ['nbs', 'bayess', 'sgd']:
        for ratio in [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]:
            cfgs.append(((proto, ratio), tasks_for(proto, 'creditcard', N0, EPS0, Q0, ratio * bs, 'fgac', ms, seed0=int(ratio * 100) + 13)))
    outs = batch(cfgs)
    for (proto, ratio), o in outs.items():
        s = summarize(o, pop, Q0, ms, TOL)
        m = np.array([x['m'] for x in o])
        s.update(proto=proto, attack='fgac', ratio=ratio, beta=ratio * bs, exact=float(np.mean(m == ms)))
        res[f'{proto}|fgac|{ratio}'] = s
        log('fgac', proto, ratio, round(s['hit_mean'], 3), round(s['infl_mean'], 2))
    save('rq3fgac', res)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return cen - half, cen + half


def _nbs(a):
    pop = getpop('creditcard')
    return run_nbs(pop, a['n'], a['eps'], a['q'], a['beta'], runexp.ATT[a['kind']], a['mstar'], rng=np.random.default_rng(a['seed']))


def valid():
    pop = getpop('creditcard')
    r = load('rq1b')
    res = {}
    R = 1000
    with Pool(2) as pool:
        for key, v in r.items():
            q, eps, kind, b = v['q'], v['eps'], v['kind'], v['mean']
            ms = v['mstar']
            cap = 4 * max(v['theory'], v['central']) + 0.01
            capped = abs(b - cap) < 2e-4
            for ratio in [0.9, 1.0, 1.1]:
                args = [dict(n=10**7, eps=eps, q=q, beta=ratio * b, kind=kind, mstar=ms, seed=20000 + int(ratio * 10) * R + i) for i in range(R)]
                m = np.array(pool.map(_nbs, args, chunksize=50), float)
                k = int(np.sum(np.abs(m - ms) <= TOL * ms))
                lo, hi = wilson(k, R)
                res[f'{key}|{ratio}'] = dict(q=q, eps=eps, kind=kind, beta=ratio * b, ratio=ratio, capped=capped, hit=k / R, lo=lo, hi=hi,
                                             exact=float(np.mean(m == ms)))
                log('valid', key, ratio, capped, round(100 * b, 3), k / R, round(lo, 3), round(hi, 3))
    save('valid', res)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        log('start', name)
        globals()[name]()
        log('end', name, f'{time.time() - t:.0f}s')
