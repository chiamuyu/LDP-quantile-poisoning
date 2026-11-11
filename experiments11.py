"""Ninth follow-up.
  rq1c40 : Fig. 2(d) budgets of BayeSS and SGD-SN-log re-estimated with 40 trials per bisection step (as for NBS; the
           earlier 10/20 trials target a hit rate of about 0.84/0.87 instead of 0.9)                    -> rq1c40.json
  leadk  : LEAD's step position calibrated with a guessed inflation factor kappa_hat (the attacker's guess of x_q),
           SGD-SN-log, n = 1e6, beta = 1.4%, 40 runs per cell                                             -> leadk.json
Usage: python3 experiments11.py leadk rq1c40
"""
import sys, time, json
import numpy as np
from multiprocessing import Pool
from experiments import log, Q0, EPS0, KAP0, TOL, _ebs, beta_star
from runexp import getpop, save
import runexp
import experiments4 as e4


def load(name):
    try:
        return json.load(open(f'{runexp.RES}/{name}.json'))
    except FileNotFoundError:
        return {}


def rq1c40():
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    res = load('rq1c40')
    for proto, ns in [('sgd', [10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7]), ('bayess', [10**5, 3 * 10**5, 10**6])]:
        for n in ns:
            key = f'{proto}|{n}'
            if key in res:
                continue
            jobs = [dict(proto=proto, data='creditcard', n=n, eps=EPS0, q=Q0, kind='cliff', mstar=ms, seed=rep + 11, T=40, steps=9, hi=0.06)
                    for rep in range(3)]
            with Pool(2) as p:
                vals = p.map(_ebs, jobs, chunksize=1)
            res[key] = dict(proto=proto, n=n, mean=float(np.mean(vals)), std=float(np.std(vals)), vals=[float(v) for v in vals],
                            theory=beta_star(pop, Q0, ms, EPS0), T=40)
            save('rq1c40', res)
            log('rq1c40', key, round(100 * res[key]['mean'], 3), round(100 * res[key]['std'], 3))


def leadk():
    res = {}
    with Pool(2) as pool:
        for d in ['creditcard', 'flights', 'pareto']:
            p_ = getpop(d)
            ms = p_.target(Q0, 10)
            B = p_.B
            base = dict(proto='sgd', n=10**6, eps=EPS0, q=Q0, beta=0.014, kind='cliff', mstar=ms)
            for khat in [5, 10, 20, 100, 1000]:
                xs = max(2, int(round(ms / khat)))
                m_sur = e4.batch(pool, dict(base, pop=('sur', B, xs)), 40, 811)
                lam = ms / np.exp(np.mean(np.log(m_sur)))
                mlead = int(min(B, round(lam * ms)))
                m_lead = e4.batch(pool, dict(base, pop=('real', d), mstar=mlead), 40, 813)
                hit = float(np.mean(np.abs(m_lead - ms) <= TOL * ms))
                res[f'{d}|{khat}'] = dict(data=d, khat=khat, lam=float(lam), hit=hit, infl=float(np.mean(m_lead) / p_.quantile(Q0)))
                log('leadk', d, khat, round(lam, 3), hit)
    save('leadk', res)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        log('start', name)
        globals()[name]()
        log('end', name, f'{time.time() - t:.0f}s')
