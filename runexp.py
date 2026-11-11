import numpy as np, math, json, os, sys, time, itertools
from multiprocessing import Pool
from ldpq import *

RES = 'results'
os.makedirs(RES, exist_ok=True)
_POP, _HH = {}, {}


def getpop(name, param=None):
    key = (name, param)
    if key not in _POP:
        _POP[key] = Pop(name, param=param)
    return _POP[key]


def gethh(name, param=None):
    key = (name, param)
    if key not in _HH:
        _HH[key] = HHCache(getpop(name, param))
    return _HH[key]


def one(args):
    """args: dict with proto, data, param, n, eps, q, beta, kind, mstar, L, seed, qrun(optional)"""
    a = args
    pop = getpop(a['data'], a.get('param'))
    qrun = a.get('qrun', a['q'])
    kind = ATT[a.get('kind', 'none')]
    beta = a['beta'] if kind != 0 else 0.0
    seed = a['seed']
    out = {}
    if a['proto'] == 'nbs':
        rng = np.random.default_rng(seed)
        out['m'] = run_nbs(pop, a['n'], a['eps'], qrun, beta, kind, a.get('mstar', 0), a.get('L', 0), rng=rng)
    elif a['proto'] == 'bayess':
        out['m'] = run_bayess(pop, a['n'], a['eps'], qrun, beta, kind, a.get('mstar', 0), a.get('L', 0), seed=seed)
    elif a['proto'] == 'sgd':
        m, Q, lo, hi = run_sgd(pop, a['n'], a['eps'], qrun, beta, kind, a.get('mstar', 0), a.get('L', 0), seed=seed)
        out.update(m=m, Q=Q, lo=lo, hi=hi)
    elif a['proto'] == 'hh':
        rng = np.random.default_rng(seed)
        out['m'] = run_hh(pop, gethh(a['data'], a.get('param')), a['n'], a['eps'], qrun, beta, kind,
                          a.get('mstar', 0), a.get('L', 0), rng=rng)
    else:
        raise ValueError(a['proto'])
    return out


def run_grid(tasks, procs=2):
    with Pool(procs) as p:
        return p.map(one, tasks, chunksize=1)


def summarize(outs, pop, q, mstar=None, tol=0.10, blocks=5):
    m = np.array([o['m'] for o in outs], dtype=float)
    xq = pop.quantile(q)
    res = {'xq': xq, 'T': len(m)}
    res['rank_mean'] = float(np.mean(pop.F(m.astype(np.int64))))
    res['rank_std'] = float(np.std(pop.F(m.astype(np.int64))))
    infl = m / xq
    res['infl_mean'] = float(np.mean(infl)); res['infl_std'] = float(np.std(infl))
    res['log2infl_mean'] = float(np.mean(np.log2(infl))); res['log2infl_std'] = float(np.std(np.log2(infl)))
    if mstar:
        hit = (np.abs(m - mstar) <= tol * mstar).astype(float)
        bl = np.array_split(hit, blocks)
        br = np.array([b.mean() for b in bl])
        res['hit_mean'] = float(br.mean()); res['hit_std'] = float(br.std())
        err = np.abs(np.log2(m / mstar))
        res['logerr_mean'] = float(err.mean()); res['logerr_std'] = float(err.std())
        ge = (m >= (1 - tol) * mstar).astype(float)          # one-sided success: reached the target
        res['ge_mean'] = float(ge.mean())
        res['edge_mean'] = float(np.mean(m >= pop.B / 2))   # landed in the top half of the domain
    res['m_raw'] = [int(x) for x in m]
    return res


def save(name, obj):
    with open(f'{RES}/{name}.json', 'w') as f:
        json.dump(obj, f, indent=1)
