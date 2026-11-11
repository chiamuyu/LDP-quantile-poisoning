"""All experiments for RQ1--RQ6. Usage: python3 experiments.py rq1a rq1b ...  (results -> results/*.json)"""
import sys, math, time, json, os
import numpy as np
from runexp import *

EPS0, Q0, N0, KAP0 = 1.0, 0.99, 10**6, 10
TOL = 0.10
BGRID = [0, .001, .002, .003, .004, .005, .006, .007, .008, .009, .010, .012, .014, .016, .020, .025, .030]
BGRID_S = [0, .002, .004, .005, .006, .007, .008, .010, .012, .016, .020, .030]


FAST = bool(os.environ.get('FAST'))
if FAST:
    import runexp
    runexp.RES = 'results_fast'


def T_of(proto):
    if FAST:
        return 2
    return 20 if proto == 'bayess' else 40


def tasks_for(proto, data, n, eps, q, beta, kind, mstar, L=0, T=None, seed0=0, qrun=None, param=None):
    T = T or T_of(proto)
    if FAST:
        T = min(T, 2)
        n = min(n, 10**5) if proto != 'hh' else n
    out = []
    for t in range(T):
        d = dict(proto=proto, data=data, n=n, eps=eps, q=q, beta=beta, kind=kind, mstar=mstar, L=L,
                 seed=int(seed0 * 100003 + t * 7919 + 17), param=param)
        if qrun is not None:
            d['qrun'] = qrun
        out.append(d)
    return out


def batch(configs):
    """configs: list of (key, taskslist). Runs all tasks in one pool, returns dict key->outs."""
    allt, idx = [], []
    for key, ts in configs:
        idx.append((key, len(allt), len(allt) + len(ts)))
        allt += ts
    outs = run_grid(allt)
    return {k: outs[a:b] for k, a, b in idx}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- RQ1a: hit rate vs beta for several displacements
def rq1a():
    pop = getpop('creditcard')
    res = {}
    for proto, kaps, grid in [('nbs', [2, 5, 10, 50], BGRID), ('sgd', [2, 10, 50], BGRID_S), ('bayess', [2, 10, 50], BGRID_S)]:
        cfgs = []
        for kap in kaps:
            ms = pop.target(Q0, kap)
            for b in grid:
                cfgs.append(((proto, kap, b), tasks_for(proto, 'creditcard', N0, EPS0, Q0, b, 'cliff' if b > 0 else 'none', ms,
                                                       seed0=int(b * 1e4) + kap)))
        outs = batch(cfgs)
        for (p, kap, b), o in outs.items():
            ms = pop.target(Q0, kap)
            s = summarize(o, pop, Q0, ms, TOL)
            s.update(proto=p, kappa=kap, beta=b, mstar=ms, theory=beta_star(pop, Q0, ms, EPS0))
            res[f'{p}|{kap}|{b}'] = s
        log('rq1a', proto, 'done')
    save('rq1a', res)


# ---------------------------------------------------------------- bisection for empirical beta*
def emp_beta_star(proto, data, n, eps, q, kind, mstar, T=40, steps=13, hi=None, seed=0, L=0, param=None, crit='tol'):
    pop = getpop(data, param)
    lo_b, hi_b = 0.0, hi or min(0.9, 4 * max(beta_star(pop, q, mstar, eps), beta_star_ipa(pop, q, mstar)) + 0.01)
    for s in range(steps):
        mid = (lo_b + hi_b) / 2
        outs = [one(t) for t in tasks_for(proto, data, n, eps, q, mid, kind, mstar, L=L, T=T, seed0=seed * 1000 + s, param=param)]
        if crit == 'up':
            hit = np.mean([o['m'] >= mstar for o in outs])
        elif crit == 'down':
            hit = np.mean([o['m'] <= mstar for o in outs])
        else:
            hit = np.mean([abs(o['m'] - mstar) <= TOL * mstar for o in outs])
        if hit >= 0.9:
            hi_b = mid
        else:
            lo_b = mid
    return hi_b


def _ebs(args):
    return emp_beta_star(**args)


def rq1b():
    """NBS cost law: empirical beta* (CLIFF, IPA) vs theory over q x eps; n = 1e7."""
    pop = getpop('creditcard')
    res = {}
    jobs, keys = [], []
    for q in [0.9, 0.95, 0.99, 0.999]:
        ms = pop.target(q, KAP0)
        for eps in [0.25, 0.5, 1.0, 2.0, 4.0]:
            for kind in ['cliff', 'ipa']:
                for rep in range(5):
                    jobs.append(dict(proto='nbs', data='creditcard', n=10**7, eps=eps, q=q, kind=kind, mstar=ms, seed=rep + 1))
                    keys.append((q, eps, kind, rep))
    with Pool(2) as p:
        vals = p.map(_ebs, jobs, chunksize=4)
    tmp = {}
    for k, v in zip(keys, vals):
        tmp.setdefault(k[:3], []).append(v)
    for (q, eps, kind), v in tmp.items():
        ms = pop.target(q, KAP0)
        th = beta_star(pop, q, ms, eps) if kind == 'cliff' else beta_star_ipa(pop, q, ms)
        res[f'{q}|{eps}|{kind}'] = dict(q=q, eps=eps, kind=kind, mean=float(np.mean(v)), std=float(np.std(v)), theory=th,
                                         central=beta_star_ipa(pop, q, ms), univ=(1 - q) * (1 - math.exp(-eps)), mstar=ms)
    save('rq1b', res)
    log('rq1b done')


def rq1c():
    """Convergence of empirical beta* to theory as n grows (q=.99, eps=1, kappa=10)."""
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    res = {}
    jobs, keys = [], []
    for proto, ns, T in [('nbs', [10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7, 3 * 10**7, 10**8], 40),
                         ('sgd', [10**5, 3 * 10**5, 10**6, 3 * 10**6], 20),
                         ('bayess', [10**5, 3 * 10**5, 10**6], 10)]:
        for n in ns:
            for rep in range(3 if proto != 'nbs' else 5):
                jobs.append(dict(proto=proto, data='creditcard', n=n, eps=EPS0, q=Q0, kind='cliff', mstar=ms, seed=rep + 11,
                                 T=T, steps=9 if proto != 'nbs' else 13, hi=0.06))
                keys.append((proto, n, rep))
    with Pool(2) as p:
        vals = p.map(_ebs, jobs, chunksize=1)
    tmp = {}
    for k, v in zip(keys, vals):
        tmp.setdefault(k[:2], []).append(v)
    for (proto, n), v in tmp.items():
        res[f'{proto}|{n}'] = dict(proto=proto, n=n, mean=float(np.mean(v)), std=float(np.std(v)), theory=beta_star(pop, Q0, ms, EPS0))
    save('rq1c', res)
    log('rq1c done')


# ---------------------------------------------------------------- RQ2: protocols x datasets
def rq2():
    res = {}
    cfgs = []
    for data in ['creditcard', 'flights', 'power', 'pareto']:
        pop = getpop(data)
        ms = pop.target(Q0, KAP0)
        for proto in ['nbs', 'bayess', 'sgd', 'hh']:
            n = 10**7 if proto == 'hh' else N0
            kinds = ['cliff', 'flood'] if proto == 'hh' else ['cliff']
            cfgs.append(((data, proto, 'none', 0.0), tasks_for(proto, data, n, EPS0, Q0, 0.0, 'none', ms, seed0=3)))
            for kind in kinds:
                for b in [0.005, 0.01, 0.02, 0.03]:
                    cfgs.append(((data, proto, kind, b), tasks_for(proto, data, n, EPS0, Q0, b, kind, ms, seed0=int(b * 1e4) + 5)))
    outs = batch(cfgs)
    for (data, proto, kind, b), o in outs.items():
        pop = getpop(data)
        ms = pop.target(Q0, KAP0)
        s = summarize(o, pop, Q0, ms, TOL)
        s.update(data=data, proto=proto, kind=kind, beta=b, mstar=ms, theory=beta_star(pop, Q0, ms, EPS0), B=pop.B)
        res[f'{data}|{proto}|{kind}|{b}'] = s
    save('rq2', res)
    log('rq2 done')


# ---------------------------------------------------------------- RQ3: knowledge / ablation (budget mis-provisioning)
def rq3():
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    bs = beta_star(pop, Q0, ms, EPS0)
    _, c = consts(EPS0)
    res = {'beta_star': bs}
    cfgs = []
    ratios = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
    for proto in ['nbs', 'bayess', 'sgd']:
        for att in ['cliff', 'scliff', 'maxval', 'ipa']:
            for r in ratios:
                b = r * bs
                # maxval aims at m* by tuning beta: it is "cliff at B" whose landing is F^-1((q+beta c)/(1-beta))
                cfgs.append(((proto, att, r), tasks_for(proto, 'creditcard', N0, EPS0, Q0, b, att, ms, L=pop.quantile(Q0) // 2,
                                                        seed0=int(r * 100) + 7)))
        cfgs.append(((proto, 'random', 3.0), tasks_for(proto, 'creditcard', N0, EPS0, Q0, 3.0 * bs, 'random', ms, seed0=9)))
    outs = batch(cfgs)
    for (proto, att, r), o in outs.items():
        s = summarize(o, pop, Q0, ms, TOL)
        s.update(proto=proto, attack=att, ratio=r, beta=r * bs)
        res[f'{proto}|{att}|{r}'] = s
    # knowledge experiment for MaxVal: tune beta from an auxiliary sample of size s
    rng = np.random.default_rng(2024)
    know = {}
    for s_aux in [10**3, 10**4, 10**5, 10**6]:
        bt = []
        for rep in range(40):
            aux = np.sort(rng.choice(pop.vals, size=s_aux, replace=True))
            Fa = np.searchsorted(aux, ms - 1, side='right') / s_aux
            bt.append(max(0.0, (Fa - Q0) / (Fa + c)))
        know[s_aux] = bt
    cfgs = []
    for s_aux, bt in know.items():
        for proto in ['nbs', 'bayess']:
            for att in ['maxval', 'cliff']:
                ts = []
                for rep, b in enumerate(bt[: (20 if proto == 'bayess' else 40)]):
                    ts += tasks_for(proto, 'creditcard', N0, EPS0, Q0, b * (1.0 if att == 'maxval' else 1.2), att, ms, T=1,
                                    seed0=rep * 31 + s_aux % 997)
                cfgs.append(((s_aux, proto, att), ts))
    outs = batch(cfgs)
    for (s_aux, proto, att), o in outs.items():
        s = summarize(o, pop, Q0, ms, TOL)
        s.update(aux=s_aux, proto=proto, attack=att, beta_tuned_mean=float(np.mean(know[s_aux])), beta_tuned_std=float(np.std(know[s_aux])))
        res[f'know|{s_aux}|{proto}|{att}'] = s
    save('rq3', res)
    log('rq3 done')


# ---------------------------------------------------------------- RQ4: boundary (q, tails, direction)
def rq4():
    res = {}
    # 4a: empirical beta* (NBS, n=1e7) across q for 4 datasets
    jobs, keys = [], []
    for data in ['creditcard', 'flights', 'power', 'pareto']:
        pop = getpop(data)
        for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]:
            ms = pop.target(q, KAP0) if pop.target(q, KAP0) > pop.quantile(q) + 1 else pop.B
            for rep in range(5):
                jobs.append(dict(proto='nbs', data=data, n=10**7, eps=EPS0, q=q, kind='cliff', mstar=ms, seed=rep + 21, hi=0.9))
                keys.append((data, q, rep))
    with Pool(2) as p:
        vals = p.map(_ebs, jobs, chunksize=4)
    tmp = {}
    for k, v in zip(keys, vals):
        tmp.setdefault(k[:2], []).append(v)
    for (data, q), v in tmp.items():
        pop = getpop(data)
        ms = pop.target(q, KAP0) if pop.target(q, KAP0) > pop.quantile(q) + 1 else pop.B
        res[f'q|{data}|{q}'] = dict(data=data, q=q, mean=float(np.mean(v)), std=float(np.std(v)), theory=beta_star(pop, q, ms, EPS0),
                                     univ=(1 - q) * (1 - math.exp(-EPS0)), kappa=ms / pop.quantile(q))
    log('rq4a done')
    # 4b: achievable inflation vs beta for tails (cliff placed at B, NBS n=1e7)
    dists = [('pareto', 1.1), ('pareto', 1.5), ('pareto', 2.5), ('lognormal', 1.0), ('lognormal', 2.0), ('exponential', None),
             ('uniform', None), ('creditcard', None), ('flights', None), ('power', None)]
    cfgs = []
    for data, par in dists:
        pop = getpop(data, par)
        for b in [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01]:
            cfgs.append(((data, par, b), tasks_for('nbs', data, 10**7, EPS0, Q0, b, 'cliff' if b > 0 else 'none', pop.B, param=par,
                                                   seed0=int(b * 1e4) + 3)))
    outs = batch(cfgs)
    _, c = consts(EPS0)
    for (data, par, b), o in outs.items():
        pop = getpop(data, par)
        s = summarize(o, pop, Q0, None)
        lvl = (Q0 + b * c) / (1 - b)
        th = pop.B if lvl >= 1 else pop.quantile(min(lvl, 1.0))
        s.update(data=data, param=par, beta=b, theory_infl=th / pop.quantile(Q0), headroom=pop.B / pop.quantile(Q0))
        res[f'tail|{data}|{par}|{b}'] = s
    log('rq4b done')
    # 4c: direction asymmetry at q=.95: inflate to x_{q+d} vs deflate to x_{q-d}
    pop = getpop('creditcard')
    q = 0.95
    jobs, keys = [], []
    for d in [0.002, 0.005, 0.01, 0.02, 0.04]:
        for direction in ['up', 'down']:
            ms = pop.quantile(q + d) + 1 if direction == 'up' else pop.quantile(q - d)
            for rep in range(5):
                jobs.append(dict(proto='nbs', data='creditcard', n=10**7, eps=EPS0, q=q, kind='cliff', mstar=ms, seed=rep + 41, hi=0.9))
                keys.append((d, direction, rep))
    with Pool(2) as p:
        vals = p.map(_ebs, jobs, chunksize=2)
    tmp = {}
    for k, v in zip(keys, vals):
        tmp.setdefault(k[:2], []).append(v)
    for (d, direction), v in tmp.items():
        ms = pop.quantile(q + d) + 1 if direction == 'up' else pop.quantile(q - d)
        F = pop.F(ms - 1) if direction == 'up' else pop.F(ms)
        th = (F - q) / (F + c) if direction == 'up' else (q - F) / (1 + c - F)
        res[f'dir|{d}|{direction}'] = dict(d=d, direction=direction, mean=float(np.mean(v)), std=float(np.std(v)), theory=th, mstar=ms)
    save('rq4', res)
    log('rq4 done')


# ---------------------------------------------------------------- RQ5: defenses
def range_audit(pop, n_audit, eps, beta, kind, mstar, L, rng, grid_pts=32, z=3.0):
    rho, c = consts(eps)
    keep = 1 - rho
    grid = np.unique(np.round(np.geomspace(1, pop.B - 1, grid_pts)).astype(np.int64))
    k_each = n_audit // len(grid)
    flag = False
    for t in grid:
        f = rng.binomial(k_each, beta) if beta > 0 else 0
        k = k_each - f
        a = rng.binomial(k, pop.F(t))
        yes = rng.binomial(a, keep) + rng.binomial(k - a, rho)
        if f:
            if kind == 'cliff':
                yes += f * (1 if t >= mstar else 0)
            elif kind == 'scliff':
                if not (L <= t < mstar):
                    yes += rng.binomial(f, keep if t >= mstar else rho)
            elif kind == 'ipa':
                yes += rng.binomial(f, keep if t >= mstar else rho)
            elif kind == 'maxval':
                yes += 0
        G = (yes / k_each - rho) / (1 - 2 * rho)
        se = math.sqrt(rho * (1 - rho) / k_each) / (1 - 2 * rho)
        if G < -z * se or G > 1 + z * se:
            flag = True
    return flag


def rq5():
    res = {}
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    rng = np.random.default_rng(55)
    # 5a: range audit detection rate
    for n, frac in [(10**6, 0.05), (10**7, 0.05), (10**7, 0.2), (10**8, 0.2)]:
        for kind in ['none', 'cliff', 'scliff', 'ipa', 'maxval']:
            for b in [0.005, 0.01, 0.02]:
                fl = [range_audit(pop, int(n * frac), EPS0, b if kind != 'none' else 0.0, kind, ms, pop.quantile(Q0) // 2, rng)
                      for _ in range(200)]
                fl = np.array(fl, float)
                br = np.array([x.mean() for x in np.array_split(fl, 5)])
                res[f'audit|{n}|{frac}|{kind}|{b}'] = dict(n=n, frac=frac, kind=kind, beta=b, det_mean=float(br.mean()), det_std=float(br.std()))
    log('rq5a done')
    # 5b: CTQ certified quantile
    cfgs = []
    for eps in [0.5, 1.0, 2.0]:
        _, c = consts(eps)
        for bbar in [0.005, 0.01, 0.02]:
            qd = Q0 - bbar * (Q0 + c) - 0.001
            for proto in ['nbs', 'bayess', 'sgd']:
                for mult in [0.0, 1.0, 2.0]:
                    b = mult * bbar
                    kind = 'cliff' if b > 0 else 'none'
                    cfgs.append(((eps, bbar, proto, mult, 'ctq'), tasks_for(proto, 'creditcard', N0, eps, Q0, b, kind, ms, qrun=qd,
                                                                             seed0=int(b * 1e4) + int(eps * 10))))
                    if bbar == 0.01:
                        cfgs.append(((eps, bbar, proto, mult, 'plain'), tasks_for(proto, 'creditcard', N0, eps, Q0, b, kind, ms,
                                                                                   seed0=int(b * 1e4) + int(eps * 10) + 1)))
    outs = batch(cfgs)
    xq = pop.quantile(Q0)
    for (eps, bbar, proto, mult, mode), o in outs.items():
        m = np.array([x['m'] for x in o], float)
        ok = (m <= xq).astype(float)
        br = np.array([x.mean() for x in np.array_split(ok, 5)])
        _, c = consts(eps)
        s = summarize(o, pop, Q0, ms, TOL)
        s.update(eps=eps, bbar=bbar, proto=proto, mult=mult, mode=mode, safe_mean=float(br.mean()), safe_std=float(br.std()),
                 qdag=Q0 - bbar * (Q0 + c) - 0.001, qlow=Q0 - bbar * (1 + Q0 + 2 * c))
        res[f'ctq|{eps}|{bbar}|{proto}|{mult}|{mode}'] = s
    save('rq5', res)
    log('rq5 done')


# ---------------------------------------------------------------- RQ6: downstream consequences
def rq6():
    res = {}
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    xq = pop.quantile(Q0)
    cls = np.load(f'{DATA}/creditcard_class.npy')
    raw = np.clip(np.load(f'{DATA}/creditcard_vals.npy'), 1, pop.B)
    fraud = raw[cls == 1]
    rng = np.random.default_rng(7)
    attacker_tx = np.round(np.exp(rng.uniform(np.log(xq + 1), np.log(ms), 5000))).astype(np.int64)
    cfgs = []
    for proto in ['nbs', 'bayess', 'sgd']:
        for b in [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02]:
            kind = 'cliff' if b > 0 else 'none'
            cfgs.append(((proto, b, 'plain'), tasks_for(proto, 'creditcard', N0, EPS0, Q0, b, kind, ms, seed0=int(b * 1e4) + 61)))
            _, c = consts(EPS0)
            qd = Q0 - 0.01 * (Q0 + c) - 0.001
            cfgs.append(((proto, b, 'ctq'), tasks_for(proto, 'creditcard', N0, EPS0, Q0, b, kind, ms, qrun=qd, seed0=int(b * 1e4) + 62)))
    outs = batch(cfgs)
    for (proto, b, mode), o in outs.items():
        T = np.array([x['m'] for x in o])
        ev = np.array([(attacker_tx <= t).mean() for t in T])
        rec = np.array([(fraud > t).mean() for t in T])
        alert = np.array([(raw > t).mean() for t in T])
        res[f'cc|{proto}|{b}|{mode}'] = dict(proto=proto, beta=b, mode=mode, evasion_mean=float(ev.mean()), evasion_std=float(ev.std()),
                                             recall_mean=float(rec.mean()), recall_std=float(rec.std()),
                                             alert_mean=float(alert.mean()), alert_std=float(alert.std()),
                                             thr_mean=float(T.mean() * 0.05), thr_std=float(T.std() * 0.05))
    log('rq6a done')
    # 6b: flights SLO false alarm: alarm if estimated p99 delay > 2 x clean p99 (attacker inflates 3x)
    fpop = getpop('flights')
    fx = fpop.quantile(Q0)
    fms = fpop.target(Q0, 3)
    cfgs = []
    for proto in ['nbs', 'bayess', 'sgd']:
        for b in [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02]:
            cfgs.append(((proto, b), tasks_for(proto, 'flights', N0, EPS0, Q0, b, 'cliff' if b > 0 else 'none', fms, seed0=int(b * 1e4) + 71)))
    outs = batch(cfgs)
    for (proto, b), o in outs.items():
        T = np.array([x['m'] for x in o], float)
        alarm = (T > 2 * fx).astype(float)
        br = np.array([x.mean() for x in np.array_split(alarm, 5)])
        res[f'fl|{proto}|{b}'] = dict(proto=proto, beta=b, alarm_mean=float(br.mean()), alarm_std=float(br.std()),
                                      p99_mean=float(np.mean(T - 87)), p99_std=float(np.std(T)), clean_p99=fx - 87)
    log('rq6b done')
    # 6c: confidence hijacking of the self-normalized CI (SGD)
    cfgs = []
    for data in ['creditcard', 'flights', 'pareto']:
        p_ = getpop(data)
        m_ = p_.target(Q0, KAP0)
        for b in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]:
            cfgs.append(((data, b), tasks_for('sgd', data, N0, EPS0, Q0, b, 'cliff' if b > 0 else 'none', m_, T=100, seed0=int(b * 1e4) + 81)))
    outs = batch(cfgs)
    for (data, b), o in outs.items():
        p_ = getpop(data)
        m_ = p_.target(Q0, KAP0)
        zt = math.log2(p_.quantile(Q0))
        zt_lo = math.log2(max(p_.quantile(Q0) - 1, 1))
        zm = math.log2(m_)
        lo = np.array([x['lo'] for x in o]); hi = np.array([x['hi'] for x in o])
        cov_true = ((lo <= zt) & (hi >= zt_lo)).astype(float)
        cov_tgt = ((lo <= zm) & (hi >= math.log2(m_ - 1))).astype(float)
        w = hi - lo
        bt = np.array([x.mean() for x in np.array_split(cov_true, 5)])
        bm = np.array([x.mean() for x in np.array_split(cov_tgt, 5)])
        Q = np.array([x['Q'] for x in o])
        res[f'ci|{data}|{b}'] = dict(data=data, beta=b, cov_true_mean=float(bt.mean()), cov_true_std=float(bt.std()),
                                     cov_tgt_mean=float(bm.mean()), cov_tgt_std=float(bm.std()),
                                     width_mean=float(w.mean()), width_std=float(w.std()),
                                     est_infl_mean=float(np.mean(2 ** Q / p_.quantile(Q0))), est_infl_std=float(np.std(2 ** Q / p_.quantile(Q0))))
    save('rq6', res)
    log('rq6 done')


def rq5c():
    """CTQ with the margin 2*eta matched to each protocol's clean error (eta = 2 x clean rank std, from RQ5 'plain' runs)."""
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    xq = pop.quantile(Q0)
    etas = {(0.5, 'nbs'): 0.011, (0.5, 'bayess'): 0.010, (0.5, 'sgd'): 0.004,
            (1.0, 'nbs'): 0.0085, (1.0, 'bayess'): 0.0065, (1.0, 'sgd'): 0.002,
            (2.0, 'nbs'): 0.0026, (2.0, 'bayess'): 0.0034, (2.0, 'sgd'): 0.001}
    cfgs = []
    for eps in [0.5, 1.0, 2.0]:
        _, c = consts(eps)
        for bbar in [0.005, 0.01, 0.02]:
            for proto in ['nbs', 'bayess', 'sgd']:
                qd = Q0 - bbar * (Q0 + c) - 2 * etas[(eps, proto)]
                for mult in [0.0, 1.0, 2.0]:
                    b = mult * bbar
                    cfgs.append(((eps, bbar, proto, mult), tasks_for(proto, 'creditcard', N0, eps, Q0, b, 'cliff' if b > 0 else 'none', ms,
                                                                     qrun=qd, seed0=int(b * 1e4) + int(eps * 10) + 91)))
    outs = batch(cfgs)
    res = {}
    for (eps, bbar, proto, mult), o in outs.items():
        _, c = consts(eps)
        m = np.array([x['m'] for x in o], float)
        ok = (m <= xq).astype(float)
        br = np.array([x.mean() for x in np.array_split(ok, 5)])
        s_ = summarize(o, pop, Q0, ms, TOL)
        s_.update(eps=eps, bbar=bbar, proto=proto, mult=mult, eta=etas[(eps, proto)], safe_mean=float(br.mean()), safe_std=float(br.std()),
                  qdag=Q0 - bbar * (Q0 + c) - 2 * etas[(eps, proto)])
        res[f'{eps}|{bbar}|{proto}|{mult}'] = s_
    save('rq5c', res)
    log('rq5c done')


def rq4c():
    pop = getpop('creditcard')
    _, c = consts(EPS0)
    res = {}
    jobs, keys = [], []
    for q in [0.95, 0.99]:
        for d in [0.002, 0.005, 0.01, 0.02, 0.04]:
            for direction in ['up', 'down']:
                if direction == 'up' and q + d >= 1:
                    continue
                ms = pop.quantile(q + d) if direction == 'up' else pop.quantile(q - d)
                for rep in range(5):
                    jobs.append(dict(proto='nbs', data='creditcard', n=10**7, eps=EPS0, q=q, kind='cliff', mstar=ms, seed=rep + 41,
                                     hi=0.9, crit=direction))
                    keys.append((q, d, direction, rep))
    with Pool(2) as p:
        vals = p.map(_ebs, jobs, chunksize=2)
    tmp = {}
    for k, v in zip(keys, vals):
        tmp.setdefault(k[:3], []).append(v)
    for (q, d, direction), v in tmp.items():
        ms = pop.quantile(q + d) if direction == 'up' else pop.quantile(q - d)
        if direction == 'up':
            F = pop.F(ms - 1); th = (F - q) / (F + c)
        else:
            F = pop.F(ms); th = (q - F) / (1 + c - F)
        res[f'{q}|{d}|{direction}'] = dict(q=q, d=d, direction=direction, mean=float(np.mean(v)), std=float(np.std(v)), theory=th,
                                            mstar=ms, ratio=ms / pop.quantile(q))
    save('rq4c', res)
    log('rq4c done')


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        log('start', name)
        globals()[name]()
        log('end', name, f'{time.time() - t:.0f}s')
