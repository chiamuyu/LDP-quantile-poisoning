"""Third follow-up.
step   : step-curve calibration of eta (Def. 1) without data, per side (Remark 1), for predictions.
lead   : SGD-SN at n=1e6 -- CLIFF with the step placed at lambda*m*, lambda planned on a synthetic population.
hh8    : HH at n=1e8 (T-Cliff and generic CLIFF).
half   : what beta = 0.5% buys (NBS; target sweep over n; kappa=10 at small eps).
plan2  : data-free budget planning on a synthetic population on the deployment's domain, for 95% success.
"""
import sys, json, math, time
import numpy as np
from multiprocessing import Pool
from runexp import getpop, summarize, save, one
import runexp
from ldpq import consts, run_nbs, run_bayess, run_sgd, run_hh, HHCache, ATT
from stepcal import eta_step

Q0, EPS0, TOL = 0.99, 1.0, 0.10


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load(name):
    try:
        return json.load(open(f'{runexp.RES}/{name}.json'))
    except FileNotFoundError:
        return {}


class SurPop:
    """Synthetic Pareto(1.5) population on [1, B], scaled so that its p99 equals xq (attacker-side surrogate)."""
    def __init__(self, B, xq, N=2_000_000, seed=4242):
        rng = np.random.default_rng(seed)
        raw = (rng.pareto(1.5, N) + 1.0)
        s = xq / np.quantile(raw, 0.99)
        v = np.clip(np.round(raw * s).astype(np.int64), 1, B)
        self.vals = np.sort(v); self.N = N; self.B = B; self.name = f'sur{B}_{xq}'

    def F(self, t):
        return np.searchsorted(self.vals, t, side='right') / self.N

    def quantile(self, q):
        k = min(max(int(math.ceil(q * self.N)), 1), self.N)
        return int(self.vals[k - 1])

    def target(self, q, kappa):
        xq = self.quantile(q)
        return int(min(self.B, max(xq + 1, round(kappa * xq))))


_SUR, _HHC = {}, {}


def sur(B, xq):
    k = (B, xq)
    if k not in _SUR:
        _SUR[k] = SurPop(B, xq)
    return _SUR[k]


def run_any(a):
    """a: dict(proto, pop=('real', name) or ('sur', B, xq), n, eps, q, beta, kind, mstar, seed)."""
    pk = a['pop']
    pop = getpop(pk[1]) if pk[0] == 'real' else sur(pk[1], pk[2])
    kind = ATT[a['kind']] if a['beta'] > 0 else 0
    if a['proto'] == 'nbs':
        return run_nbs(pop, a['n'], a['eps'], a['q'], a['beta'], kind, a['mstar'], rng=np.random.default_rng(a['seed']))
    if a['proto'] == 'bayess':
        return run_bayess(pop, a['n'], a['eps'], a['q'], a['beta'], kind, a['mstar'], seed=a['seed'])
    if a['proto'] == 'sgd':
        return run_sgd(pop, a['n'], a['eps'], a['q'], a['beta'], kind, a['mstar'], seed=a['seed'])[0]
    if a['proto'] == 'hh':
        if pop.name not in _HHC:
            _HHC[pop.name] = HHCache(pop)
        return run_hh(pop, _HHC[pop.name], a['n'], a['eps'], a['q'], a['beta'], kind, a['mstar'], rng=np.random.default_rng(a['seed']))
    raise ValueError(a['proto'])


def batch(pool, base, T, seed0):
    return np.array(pool.map(run_any, [dict(base, seed=seed0 * 100003 + 7919 * i + 17) for i in range(T)], chunksize=2), float)


# ------------------------------------------------------------------ step calibration
def step():
    res = load('eta_step')
    cells = []
    cc = getpop('creditcard')
    rb = load('rq1b')
    for q in [0.9, 0.95, 0.99, 0.999]:
        for e in [0.25, 0.5, 1.0, 2.0, 4.0]:
            cells.append(('nbs', 'creditcard', 10**7, e, q, rb[f'{q}|{e}|cliff']['mstar'], 200, 11, 0.1))
    for d in ['creditcard', 'flights', 'power', 'pareto']:
        p_ = getpop(d)
        for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]:
            ms = p_.target(q, 10) if p_.target(q, 10) > p_.quantile(q) + 1 else p_.B
            cells.append(('nbs', d, 10**7, 1.0, q, ms, 200, 11, 0.1))
    ms = cc.target(Q0, 10)
    for n in [10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7, 3 * 10**7, 10**8]:
        cells.append(('nbs', 'creditcard', n, 1.0, Q0, ms, 200, 11, 0.1))
    for n in [10**5, 3 * 10**5, 10**6, 3 * 10**6]:
        cells.append(('sgd', 'creditcard', n, 1.0, Q0, ms, 100, 10, 0.1))
    for n in [10**5, 3 * 10**5]:
        cells.append(('bayess', 'creditcard', n, 1.0, Q0, ms, 60, 9, 0.1))
    cells.append(('bayess', 'creditcard', 10**6, 1.0, Q0, ms, 60, 9, 0.1))
    with Pool(2) as pool:
        for (proto, d, n, e, q, b, T, steps, delta) in cells:
            for side in ['lo', 'up']:
                key = f'{proto}|{d}|{n}|{e}|{q}|{b}|{side}|{delta}'
                if key in res:
                    continue
                res[key] = eta_step(proto, getpop(d), n, e, q, b, delta=delta, T=T, steps=steps, hi=0.15, pool=pool, side=side)
                save('eta_step', res)
            log('step', proto, d, n, e, q, res[f'{proto}|{d}|{n}|{e}|{q}|{b}|lo|{delta}'], res[f'{proto}|{d}|{n}|{e}|{q}|{b}|up|{delta}'])


# ------------------------------------------------------------------ lead-compensated CLIFF against SGD-SN
def lead():
    res = load('lead')
    with Pool(2) as pool:
        for d in ['creditcard', 'flights', 'pareto']:
            p_ = getpop(d); ms = p_.target(Q0, 10); B = p_.B
            for beta in [0.01, 0.012, 0.014]:
                key = f'{d}|{beta}'
                if key in res:
                    continue
                base = dict(proto='sgd', n=10**6, eps=EPS0, q=Q0, beta=beta, kind='cliff', mstar=ms)
                ms_sur = ms   # the surrogate shares the domain and the target position (p99 at m*/10)
                m_sur = batch(pool, dict(base, pop=('sur', B, int(round(ms / 10)))), 40, 11)
                lam = ms_sur / np.exp(np.mean(np.log(m_sur)))
                mlead = int(min(B, round(lam * ms)))
                m_plain = batch(pool, dict(base, pop=('real', d)), 40, 12)
                m_lead = batch(pool, dict(base, pop=('real', d), mstar=mlead), 40, 13)
                def hitb(m):
                    h = (np.abs(m - ms) <= TOL * ms).astype(float)
                    br = np.array([x.mean() for x in np.array_split(h, 5)])
                    return float(br.mean()), float(br.std())
                res[key] = dict(data=d, beta=beta, lam=float(lam), plain_hit=hitb(m_plain), lead_hit=hitb(m_lead),
                                plain_infl=float(np.mean(m_plain) / p_.quantile(Q0)), lead_infl=float(np.mean(m_lead) / p_.quantile(Q0)),
                                lead_infl_std=float(np.std(m_lead) / p_.quantile(Q0)))
                save('lead', res)
                log('lead', d, beta, round(lam, 3), res[key]['plain_hit'], res[key]['lead_hit'])


# ------------------------------------------------------------------ HH at n = 1e8
def hh8():
    res = load('hh8')
    with Pool(2) as pool:
        for d in ['creditcard', 'flights', 'power', 'pareto']:
            p_ = getpop(d); ms = p_.target(Q0, 10)
            for kind, beta in [('none', 0.0), ('tcliff', 0.005), ('tcliff', 0.0075), ('tcliff', 0.01), ('cliff', 0.01)]:
                key = f'{d}|{kind}|{beta}'
                if key in res:
                    continue
                m = batch(pool, dict(proto='hh', pop=('real', d), n=10**8, eps=EPS0, q=Q0, beta=beta, kind=kind, mstar=ms), 40, 21)
                s = summarize([{'m': x} for x in m], p_, Q0, ms, TOL)
                s.update(data=d, kind=kind, beta=beta)
                res[key] = s
                save('hh8', res)
            log('hh8', d)


# ------------------------------------------------------------------ what 0.5% buys
def half():
    res = load('half')
    cc = getpop('creditcard'); xq = cc.quantile(Q0)
    with Pool(2) as pool:
        for n in [10**6, 10**7, 10**8]:
            for kap in [1.1, 1.25, 1.5, 2, 3, 5, 10]:
                key = f'kap|{n}|{kap}'
                if key in res:
                    continue
                ms = cc.target(Q0, kap)
                m = batch(pool, dict(proto='nbs', pop=('real', 'creditcard'), n=n, eps=EPS0, q=Q0, beta=0.005, kind='cliff', mstar=ms), 40, 31)
                s = summarize([{'m': x} for x in m], cc, Q0, ms, TOL)
                lo = eta_step('nbs', cc, n, EPS0, Q0, ms, T=200, steps=11, hi=0.15, pool=pool, side='lo')
                up = eta_step('nbs', cc, n, EPS0, Q0, ms, T=200, steps=11, hi=0.15, pool=pool, side='up')
                s.update(n=n, kappa=kap, mstar=ms, eta_lo=lo, eta_up=up)
                res[key] = s
                save('half', res)
                log('half', n, kap, s['hit_mean'])
        ms = cc.target(Q0, 10)
        for e in [0.25, 0.5, 1.0]:
            key = f'eps|{e}'
            if key in res:
                continue
            m = batch(pool, dict(proto='nbs', pop=('real', 'creditcard'), n=10**7, eps=e, q=Q0, beta=0.005, kind='cliff', mstar=ms), 40, 32)
            s = summarize([{'m': x} for x in m], cc, Q0, ms, TOL); s.update(eps=e)
            res[key] = s
            save('half', res)
            log('half eps', e, s['hit_mean'])


# ------------------------------------------------------------------ data-free planning, 95% success
PLAN = [('nbs', 'cliff', 10**6), ('bayess', 'cliff', 10**6), ('sgd', 'cliff', 10**6), ('sgd', 'cliff', 10**7), ('hh', 'tcliff', 10**7)]


def plan2():
    res = load('plan2')
    with Pool(2) as pool:
        for proto, kind, n in PLAN:
            for d in ['creditcard', 'flights', 'power']:
                p_ = getpop(d); ms = p_.target(Q0, 10); B = p_.B
                key = f'cal|{proto}|{n}|{d}'
                if key not in res:
                    spop = ('sur', B, int(round(ms / 10)))
                    lo_b, hi_b = 0.0, 0.06
                    for s in range(9):
                        mid = (lo_b + hi_b) / 2
                        m = batch(pool, dict(proto=proto, pop=spop, n=n, eps=EPS0, q=Q0, beta=mid, kind=kind, mstar=ms), 40, 41 + s)
                        if np.mean(np.abs(m - ms) <= TOL * ms) >= 0.95:
                            hi_b = mid
                        else:
                            lo_b = mid
                    res[key] = hi_b
                    save('plan2', res)
                    log('cal', proto, n, d, hi_b)
                for mult in [1.0, 1.25]:
                    key2 = f'atk|{proto}|{n}|{d}|{mult}'
                    if key2 in res:
                        continue
                    m = batch(pool, dict(proto=proto, pop=('real', d), n=n, eps=EPS0, q=Q0, beta=res[key] * mult, kind=kind, mstar=ms), 40, 51 + int(mult * 10))
                    s = summarize([{'m': x} for x in m], p_, Q0, ms, TOL)
                    s.update(proto=proto, n=n, data=d, mult=mult, beta=res[key] * mult)
                    res[key2] = s
                    save('plan2', res)
                    log('atk', proto, n, d, mult, round(s['hit_mean'], 3))


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        log('start', name)
        globals()[name]()
        log('end', name, f'{time.time() - t:.0f}s')
