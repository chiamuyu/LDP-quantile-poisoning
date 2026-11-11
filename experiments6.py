"""Fourth follow-up.
  rq2_15  : CLIFF at beta = 1.5% for every protocol/dataset of Table IV (adds keys to rq2.json)
  jump    : the jump audit (a tail-saturation test) and the range audit against CLIFF, S-CLIFF, IPA, MGA and LOW-CLIFF
            (NBS outputs at n = 1e6, 1e7, 1e8)                                    -> jump.json
  rq3raw  : RQ3 budget grid re-run with raw outputs: one-sided success P[m >= 0.9 m*] and the edge rate P[m >= B/2]
            (same seeds as experiments.py rq3 / experiments5.py)                 -> rq3raw.json
  plan3   : data-free planning when the attacker's guess of x_q is wrong (kappa_hat = 2, 5, 1000)  -> plan3.json
Usage: python3 experiments6.py rq2_15 jump rq3raw plan3
"""
import sys, math, time, json
import numpy as np
from multiprocessing import Pool
from experiments import tasks_for, batch, log, Q0, N0, EPS0, KAP0, TOL
from runexp import getpop, summarize, save, consts, run_nbs
import runexp
import experiments4 as e4


def load(name):
    try:
        return json.load(open(f'{runexp.RES}/{name}.json'))
    except FileNotFoundError:
        return {}


# ------------------------------------------------------------------ Table IV: beta = 1.5%
def rq2_15():
    res = load('rq2')
    cfgs = []
    for data in ['creditcard', 'flights', 'power', 'pareto']:
        pop = getpop(data)
        ms = pop.target(Q0, KAP0)
        for proto in ['nbs', 'bayess', 'sgd', 'hh']:
            n = 10**7 if proto == 'hh' else N0
            b = 0.015
            cfgs.append(((data, proto, 'cliff', b), tasks_for(proto, data, n, EPS0, Q0, b, 'cliff', ms, seed0=int(b * 1e4) + 5)))
    outs = batch(cfgs)
    for (data, proto, kind, b), o in outs.items():
        pop = getpop(data)
        ms = pop.target(Q0, KAP0)
        s = summarize(o, pop, Q0, ms, TOL)
        s.update(data=data, proto=proto, kind=kind, beta=b, mstar=ms, B=pop.B)
        res[f'{data}|{proto}|{kind}|{b}'] = s
    save('rq2', res)
    log('rq2_15 done')


# ------------------------------------------------------------------ jump audit
ETA = {}  # step-calibrated eta of NBS (CreditCard, eps = 1, q = 0.99): (lo, up)


def eta_nbs(n):
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
    if kind.startswith('low'):
        return rng.binomial(f, p_low) if t >= mstar else 0
    return 0


def estimate(pop, k_each, beta, kind, t, mstar, L, p_low, rng, rho):
    f = rng.binomial(k_each, beta) if beta > 0 and kind != 'none' else 0
    k = k_each - f
    a = rng.binomial(k, pop.F(t))
    yes = rng.binomial(a, 1 - rho) + rng.binomial(k - a, rho) + fake_yes(kind, t, mstar, L, p_low, pop.B, f, rng, rho)
    pi = yes / k_each
    G = (pi - rho) / (1 - 2 * rho)
    se = math.sqrt(max(pi * (1 - pi), 1e-12) / k_each) / (1 - 2 * rho)
    return G, se


def jump_audit(pop, n_a, eps, q, beta, kind, mstar, L, p_low, mhat, eta, rng, z=3.0):
    """Tail-saturation test: the honest mass on [mhat, 1.1 mhat] is at most 1 - F(mhat - 1) <= 1 - q + eta."""
    rho, _ = consts(eps)
    t1 = max(int(mhat) - 1, 1)
    t2 = min(int(1.1 * mhat), pop.B)
    k_each = n_a // 2
    g1, s1 = estimate(pop, k_each, beta, kind, t1, mstar, L, p_low, rng, rho)
    g2, s2 = estimate(pop, k_each, beta, kind, t2, mstar, L, p_low, rng, rho)
    D = g2 - g1
    return D > (1 - q) + eta + z * math.sqrt(s1 * s1 + s2 * s2), D


def range_audit(pop, n_a, eps, beta, kind, mstar, L, p_low, rng, grid_pts=32, z=3.0):
    rho, _ = consts(eps)
    grid = np.unique(np.round(np.geomspace(1, pop.B - 1, grid_pts)).astype(np.int64))
    k_each = n_a // len(grid)
    for t in grid:
        G, se = estimate(pop, k_each, beta, kind, int(t), mstar, L, p_low, rng, rho)
        if G < -z * se or G > 1 + z * se:
            return True
    return False


def _nbs(a):
    pop = getpop('creditcard')
    kind = runexp.ATT['lowcliff'] if a['kind'].startswith('low') else (runexp.ATT[a['kind']] if a['beta'] > 0 else 0)
    L = a['L']
    return run_nbs(pop, a['n'], EPS0, Q0, a['beta'], kind, a['mstar'], L=L, rng=np.random.default_rng(a['seed']))


def jump():
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    xq = pop.quantile(Q0)
    rho, c = consts(EPS0)
    beta = 0.01
    Fm = float(pop.F(ms))
    res = {}
    kinds = ['none', 'cliff', 'scliff', 'ipa', 'maxval', 'lowfree', 'lowone', 'lowinf']
    R = 200
    mh = {}
    with Pool(2) as pool:
        for n in [10**6, 10**7, 10**8]:
            elo, eup = eta_nbs(n)
            for kind in kinds:
                v = None
                if kind == 'lowfree':      # data-free: assumes only F(m*) >= q
                    v = min(Q0 + eup / beta, 1 + c)
                elif kind == 'lowone':     # data-free for deep targets: assumes F(m*) = 1
                    v = min((Q0 + eup - (1 - beta)) / beta, 1 + c)
                elif kind == 'lowinf':     # informed: knows F(m*)
                    v = min((Q0 + eup - (1 - beta) * Fm) / beta, 1 + c)
                p_low = rho + (1 - 2 * rho) * v if v is not None else 0.0
                L = int(round(p_low * 1e6)) if kind.startswith('low') else (xq // 2 if kind == 'scliff' else 0)
                args = [dict(n=n, kind=kind, beta=beta if kind != 'none' else 0.0, mstar=ms, L=L, seed=7000 + 13 * i + n % 997)
                        for i in range(R)]
                m = np.array(pool.map(_nbs, args, chunksize=10), float)
                mh[(n, kind)] = (m, p_low)
                hit = float(np.mean(np.abs(m - ms) <= TOL * ms))
                res[f'hit|{n}|{kind}'] = dict(n=n, kind=kind, hit=hit, ge=float(np.mean(m >= 0.9 * ms)), v=v, p_low=p_low,
                                              infl=float(np.mean(m / xq)), jump_th=(beta * (v + c) if v is not None else None))
                log('nbs', n, kind, round(hit, 3), v)
    rng = np.random.default_rng(77)
    for n, frac in [(10**6, 0.05), (10**7, 0.05), (10**7, 0.2), (10**8, 0.2)]:
        elo, eup = eta_nbs(n)
        n_a = int(n * frac)
        for kind in kinds:
            m, p_low = mh[(n, kind)]
            b = beta if kind != 'none' else 0.0
            L = xq // 2
            fl, Ds, fr = [], [], []
            for mhat in m:
                f_, D = jump_audit(pop, n_a, EPS0, Q0, b, kind, ms, L, p_low, mhat, elo, rng)
                fl.append(f_); Ds.append(D)
                fr.append(range_audit(pop, n_a, EPS0, b, kind, ms, L, p_low, rng))
            fl = np.array(fl, float); fr = np.array(fr, float)
            bj = np.array([x.mean() for x in np.array_split(fl, 5)])
            br = np.array([x.mean() for x in np.array_split(fr, 5)])
            res[f'audit|{n}|{frac}|{kind}'] = dict(n=n, frac=frac, kind=kind, jump_mean=float(bj.mean()), jump_std=float(bj.std()),
                                                  range_mean=float(br.mean()), range_std=float(br.std()),
                                                  D_mean=float(np.mean(Ds)), D_std=float(np.std(Ds)),
                                                  thr=(1 - Q0) + elo)
            log('audit', n, frac, kind, round(bj.mean(), 3), round(br.mean(), 3), round(np.mean(Ds), 4))
    res['meta'] = dict(Fm=Fm, mstar=ms, xq=xq, c=c, beta=beta)
    save('jump', res)
    log('jump done')


# ------------------------------------------------------------------ RQ3 with raw outputs
def rq3raw():
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    xq = pop.quantile(Q0)
    r3 = load('rq3')
    bs = r3['beta_star']
    res = {'beta_star': bs}
    cfgs = []
    ratios = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
    for proto in ['nbs', 'bayess', 'sgd']:
        for att in ['cliff', 'scliff', 'maxval', 'ipa']:
            for rr in ratios:
                cfgs.append(((proto, att, rr), tasks_for(proto, 'creditcard', N0, EPS0, Q0, rr * bs, att, ms, L=xq // 2, seed0=int(rr * 100) + 7)))
        for rr in ratios:
            cfgs.append(((proto, 'fga', rr), tasks_for(proto, 'creditcard', N0, EPS0, Q0, rr * bs, 'fga', ms, L=xq, seed0=int(rr * 100) + 9)))
    outs = batch(cfgs)
    for (proto, att, rr), o in outs.items():
        s = summarize(o, pop, Q0, ms, TOL)
        s.update(proto=proto, attack=att, ratio=rr, beta=rr * bs)
        old = r3.get(f'{proto}|{att}|{rr}')
        if old is not None:
            s['same_as_rq3'] = abs(old['hit_mean'] - s['hit_mean']) < 1e-12 and abs(old['infl_mean'] - s['infl_mean']) < 1e-9
        res[f'{proto}|{att}|{rr}'] = s
    save('rq3raw', res)
    log('rq3raw done')


# ------------------------------------------------------------------ planning with a wrong guess of x_q
def plan3():
    res = load('plan3')
    order = [('nbs', 'cliff', 10**6), ('sgd', 'cliff', 10**6), ('hh', 'tcliff', 10**7), ('sgd', 'cliff', 10**7), ('bayess', 'cliff', 10**6)]
    with Pool(2) as pool:
        for proto, kind, n in order:
            for kap in [1000, 2, 5]:
                for d in ['creditcard', 'flights', 'power']:
                    p_ = getpop(d); ms = p_.target(Q0, 10); B = p_.B
                    key = f'cal|{proto}|{n}|{d}|{kap}'
                    if key not in res:
                        spop = ('sur', B, max(1, int(round(ms / kap))))
                        lo_b, hi_b = 0.0, 0.06
                        for s in range(9):
                            mid = (lo_b + hi_b) / 2
                            m = e4.batch(pool, dict(proto=proto, pop=spop, n=n, eps=EPS0, q=Q0, beta=mid, kind=kind, mstar=ms), 40, 41 + s)
                            if np.mean(np.abs(m - ms) <= TOL * ms) >= 0.95:
                                hi_b = mid
                            else:
                                lo_b = mid
                        res[key] = hi_b
                        save('plan3', res)
                        log('cal', proto, n, d, kap, hi_b)
                    for mult in [1.0, 1.25]:
                        key2 = f'atk|{proto}|{n}|{d}|{kap}|{mult}'
                        if key2 in res:
                            continue
                        m = e4.batch(pool, dict(proto=proto, pop=('real', d), n=n, eps=EPS0, q=Q0, beta=res[key] * mult, kind=kind, mstar=ms), 40,
                                     51 + int(mult * 10))
                        s = summarize([{'m': x} for x in m], p_, Q0, ms, TOL)
                        s.update(proto=proto, n=n, data=d, mult=mult, kap=kap, beta=res[key] * mult)
                        res[key2] = s
                        save('plan3', res)
                        log('atk', proto, n, d, kap, mult, round(s['hit_mean'], 3))


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        log('start', name)
        globals()[name]()
        log('end', name, f'{time.time() - t:.0f}s')
