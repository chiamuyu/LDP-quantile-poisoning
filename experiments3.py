"""Second follow-up (addresses: SGD-SN at 1%, finite-sample gap, beta=0.5%, HH pinning, budget planning, q=0.999).
eta     : clean runs (raw outputs) used to measure each protocol's accuracy eta (Def. 1) on the real data and on a
          synthetic surrogate (Pareto(1.5)); the finite-sample cost law is Thm. 1 with this eta.
sgd     : SGD-SN at n=1e7 (hit rates on all datasets; empirical budget for Fig. 2(d)).
hh      : tree-consistent CLIFF against HH (RQ2 grid).
plan    : data-free budget planning: bisect on the surrogate, then attack the real datasets with that budget.
"""
import sys, json, math, time
import numpy as np
from multiprocessing import Pool
import experiments as E
from experiments import *

SUR = 'pareto'


def _one(t):
    return one(t)


def runs(tasks):
    with Pool(2) as p:
        return p.map(_one, tasks, chunksize=4)


def load(name):
    import runexp
    try:
        return json.load(open(f'{runexp.RES}/{name}.json'))
    except FileNotFoundError:
        return {}


def eta():
    res = load('eta')
    cells = []
    for data in ['creditcard', SUR]:
        for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]:
            for eps in [0.25, 0.5, 1.0, 2.0, 4.0]:
                cells.append(('nbs', data, 10**7, eps, q, 200))
    for data in ['flights', 'power']:
        for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]:
            cells.append(('nbs', data, 10**7, 1.0, q, 200))
    for data in ['creditcard', SUR]:
        for n in [10**5, 3 * 10**5, 10**6, 3 * 10**6, 3 * 10**7, 10**8]:
            cells.append(('nbs', data, n, 1.0, 0.99, 200))
        for n in [10**5, 3 * 10**5, 10**6]:
            cells.append(('bayess', data, n, 1.0, 0.99, 100))
        for n in [10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7]:
            cells.append(('sgd', data, n, 1.0, 0.99, 100))
        cells.append(('hh', data, 10**7, 1.0, 0.99, 100))
    for (proto, data, n, eps, q, T) in cells:
        key = f'{proto}|{data}|{n}|{eps}|{q}'
        if key in res:
            continue
        ts = [dict(proto=proto, data=data, n=n, eps=eps, q=q, beta=0.0, kind='none', mstar=0, seed=900001 + 7919 * i) for i in range(T)]
        o = runs(ts)
        res[key] = [int(x['m']) for x in o]
        save('eta', res)
    log('eta done', len(res))


def sgd():
    res = load('sgd1e7')
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    for data in ['creditcard', 'flights', 'power', 'pareto']:
        p_ = getpop(data); m_ = p_.target(Q0, KAP0)
        for b in [0.005, 0.01, 0.02]:
            key = f'hit|{data}|{b}'
            if key in res:
                continue
            o = runs(tasks_for('sgd', data, 10**7, EPS0, Q0, b, 'cliff', m_, T=40, seed0=int(b * 1e4) + 301))
            s = summarize(o, p_, Q0, m_, TOL); s.update(data=data, beta=b)
            res[key] = s
            save('sgd1e7', res)
            log('sgd hit', data, b, s['hit_mean'], s['infl_mean'])
    if 'bisect' not in res:
        jobs = [dict(proto='sgd', data='creditcard', n=10**7, eps=EPS0, q=Q0, kind='cliff', mstar=ms, seed=rep + 11, T=20, steps=9, hi=0.06)
                for rep in range(3)]
        with Pool(2) as p:
            vals = p.map(E._ebs, jobs, chunksize=1)
        res['bisect'] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), vals=vals)
        save('sgd1e7', res)
    log('sgd done', res['bisect'])


def hh():
    res = load('hhcc')
    for data in ['creditcard', 'flights', 'power', 'pareto']:
        p_ = getpop(data); m_ = p_.target(Q0, KAP0)
        for b in [0.005, 0.01, 0.015, 0.02, 0.03]:
            key = f'{data}|{b}'
            if key in res:
                continue
            o = runs(tasks_for('hh', data, 10**7, EPS0, Q0, b, 'tcliff', m_, T=40, seed0=int(b * 1e4) + 401))
            s = summarize(o, p_, Q0, m_, TOL); s.update(data=data, beta=b)
            res[key] = s
            save('hhcc', res)
    log('hh done')


PLAN = [('nbs', 'cliff', 10**6, 40, 13), ('bayess', 'cliff', 10**6, 20, 9), ('sgd', 'cliff', 10**6, 20, 9),
        ('sgd', 'cliff', 10**7, 20, 9), ('hh', 'tcliff', 10**7, 40, 13)]


def plan():
    res = load('plan')
    # 1) calibration on the surrogate: one bisection, as an attacker would run it
    for proto, kind, n, T, steps in PLAN:
        key = f'cal|{proto}|{n}'
        if key in res:
            continue
        p_ = getpop(SUR); m_ = p_.target(Q0, KAP0)
        b = emp_beta_star(proto, SUR, n, EPS0, Q0, kind, m_, T=T, steps=steps, hi=0.06, seed=77)
        res[key] = b
        save('plan', res)
        log('cal', proto, n, b)
    # 2) attack the real datasets with the calibrated budget (and 1.25x)
    for proto, kind, n, T, steps in PLAN:
        b0 = res[f'cal|{proto}|{n}']
        for data in ['creditcard', 'flights', 'power']:
            p_ = getpop(data); m_ = p_.target(Q0, KAP0)
            for mult in [1.0, 1.25]:
                key = f'atk|{proto}|{n}|{data}|{mult}'
                if key in res:
                    continue
                o = runs(tasks_for(proto, data, n, EPS0, Q0, b0 * mult, kind, m_, T=40, seed0=int(mult * 100) + 503))
                s = summarize(o, p_, Q0, m_, TOL); s.update(proto=proto, n=n, data=data, mult=mult, beta=b0 * mult)
                res[key] = s
                save('plan', res)
                log('atk', proto, n, data, mult, round(s['hit_mean'], 3))
    log('plan done')


def eta2():
    """Clean runs at the reference level q_ref = 0.9 for the n-sweep (surrogate and CreditCard)."""
    res = load('eta')
    cells = []
    for data in [SUR, 'creditcard']:
        for n in [10**5, 3 * 10**5, 10**6, 3 * 10**6, 3 * 10**7, 10**8]:
            cells.append(('nbs', data, n, 1.0, 0.9, 200))
        for n in [10**5, 3 * 10**5, 10**6]:
            cells.append(('bayess', data, n, 1.0, 0.9, 100))
        for n in [10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7]:
            cells.append(('sgd', data, n, 1.0, 0.9, 100))
        cells.append(('hh', data, 10**7, 1.0, 0.9, 100))
    for (proto, data, n, eps, q, T) in cells:
        key = f'{proto}|{data}|{n}|{eps}|{q}'
        if key in res:
            continue
        ts = [dict(proto=proto, data=data, n=n, eps=eps, q=q, beta=0.0, kind='none', mstar=0, seed=900001 + 7919 * i) for i in range(T)]
        o = runs(ts)
        res[key] = [int(x['m']) for x in o]
        save('eta', res)
    log('eta2 done', len(res))


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        log('start', name)
        globals()[name]()
        log('end', name, f'{time.time() - t:.0f}s')
