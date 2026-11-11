"""Follow-up experiments after internal review.
rq6a2: RQ6(a) with the CTQ margins of RQ5 (calibrated eta) and alert volume on legitimate traffic only.
rq6b2: RQ6(b) with alarm/target defined on the delay in minutes (undo the +87 shift).
rq5d : adaptive attacks on CTQ: CLIFF targets just above x_q.
rq1a_bay: BayeSS rerun of RQ1a with kappa in {2,5,10,50} and 40 trials.
"""
import sys, json, math, time
import numpy as np
import experiments as E
from experiments import *

ETAS = {(0.5, 'nbs'): 0.011, (0.5, 'bayess'): 0.010, (0.5, 'sgd'): 0.004,
        (1.0, 'nbs'): 0.0085, (1.0, 'bayess'): 0.0065, (1.0, 'sgd'): 0.002,
        (2.0, 'nbs'): 0.0026, (2.0, 'bayess'): 0.0034, (2.0, 'sgd'): 0.001}


def load(name):
    with open(f'{runexp_RES()}/{name}.json') as f:
        return json.load(f)


def runexp_RES():
    import runexp
    return runexp.RES


def rq6a2():
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    xq = pop.quantile(Q0)
    cls = np.load(f'{DATA}/creditcard_class.npy')
    raw = np.clip(np.load(f'{DATA}/creditcard_vals.npy'), 1, pop.B)
    fraud = raw[cls == 1]
    legit = raw[cls == 0]
    rng = np.random.default_rng(7)
    attacker_tx = np.round(np.exp(rng.uniform(np.log(xq + 1), np.log(ms), 5000))).astype(np.int64)
    _, c = consts(EPS0)
    cfgs = []
    for proto in ['nbs', 'bayess', 'sgd']:
        qd = Q0 - 0.01 * (Q0 + c) - 2 * ETAS[(EPS0, proto)]
        for b in [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02]:
            kind = 'cliff' if b > 0 else 'none'
            cfgs.append(((proto, b, 'plain'), tasks_for(proto, 'creditcard', N0, EPS0, Q0, b, kind, ms, seed0=int(b * 1e4) + 61)))
            cfgs.append(((proto, b, 'ctq'), tasks_for(proto, 'creditcard', N0, EPS0, Q0, b, kind, ms, qrun=qd, seed0=int(b * 1e4) + 62)))
    outs = batch(cfgs)
    res = load('rq6')
    for (proto, b, mode), o in outs.items():
        T = np.array([x['m'] for x in o])
        ev = np.array([(attacker_tx <= t).mean() for t in T])
        rec = np.array([(fraud > t).mean() for t in T])
        alert = np.array([(legit > t).mean() for t in T])
        safe = (T <= xq).astype(float)
        res[f'cc|{proto}|{b}|{mode}'] = dict(proto=proto, beta=b, mode=mode, evasion_mean=float(ev.mean()), evasion_std=float(ev.std()),
                                             recall_mean=float(rec.mean()), recall_std=float(rec.std()),
                                             alert_mean=float(alert.mean()), alert_std=float(alert.std()),
                                             thr_mean=float(T.mean() * 0.05), thr_std=float(T.std() * 0.05),
                                             safe_mean=float(safe.mean()), eta=ETAS[(EPS0, proto)] if mode == 'ctq' else None)
    res['cc_meta'] = dict(xq_eur=xq * 0.05, mstar_eur=ms * 0.05, n_legit=int(len(legit)), n_fraud=int(len(fraud)))
    save('rq6', res)
    log('rq6a2 done')


def rq6b2():
    fpop = getpop('flights')
    SHIFT = 87
    fx = fpop.quantile(Q0)
    d99 = fx - SHIFT                      # true p99 delay in minutes
    fms = SHIFT + 3 * d99                 # target: 3x the p99 delay
    cfgs = []
    for proto in ['nbs', 'bayess', 'sgd']:
        for b in [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02]:
            cfgs.append(((proto, b), tasks_for(proto, 'flights', N0, EPS0, Q0, b, 'cliff' if b > 0 else 'none', fms, seed0=int(b * 1e4) + 71)))
    outs = batch(cfgs)
    res = load('rq6')
    for k in [k for k in res if k.startswith('fl|')]:
        del res[k]
    for (proto, b), o in outs.items():
        D = np.array([x['m'] for x in o], float) - SHIFT
        alarm = (D > 2 * d99).astype(float)
        br = np.array([x.mean() for x in np.array_split(alarm, 5)])
        hit = (np.abs(D + SHIFT - fms) <= TOL * fms).astype(float)
        res[f'fl|{proto}|{b}'] = dict(proto=proto, beta=b, alarm_mean=float(br.mean()), alarm_std=float(br.std()),
                                      p99_mean=float(D.mean()), p99_std=float(D.std()), clean_p99=float(d99),
                                      alarm_at=float(2 * d99), target_delay=float(fms - SHIFT))
    save('rq6', res)
    log('rq6b2 done')


def rq5d():
    pop = getpop('creditcard')
    xq = pop.quantile(Q0)
    _, c = consts(EPS0)
    targets = {'x+1': xq + 1, '1.1x': int(math.ceil(1.1 * xq)), '2x': 2 * xq, '10x': pop.target(Q0, KAP0)}
    cfgs = []
    for bbar in [0.005, 0.01, 0.02]:
        for proto in ['nbs', 'bayess', 'sgd']:
            qd = Q0 - bbar * (Q0 + c) - 2 * ETAS[(EPS0, proto)]
            for tname, ms in targets.items():
                if tname == '10x':
                    continue
                cfgs.append(((bbar, proto, tname), tasks_for(proto, 'creditcard', N0, EPS0, Q0, bbar, 'cliff', ms, qrun=qd,
                                                              seed0=int(bbar * 1e4) + 97 + len(tname))))
    outs = batch(cfgs)
    res = {}
    for (bbar, proto, tname), o in outs.items():
        m = np.array([x['m'] for x in o], float)
        ok = (m <= xq).astype(float)
        br = np.array([x.mean() for x in np.array_split(ok, 5)])
        res[f'{bbar}|{proto}|{tname}'] = dict(bbar=bbar, proto=proto, target=tname, mstar=int(targets[tname]), safe_mean=float(br.mean()),
                                               safe_std=float(br.std()), rank_mean=float(np.mean(pop.F(m.astype(np.int64)))),
                                               rank_std=float(np.std(pop.F(m.astype(np.int64)))), T=len(m))
    save('rq5d', res)
    log('rq5d done')


def rq1a_bay():
    pop = getpop('creditcard')
    res = load('rq1a')
    for k in [k for k in res if k.startswith('bayess|')]:
        del res[k]
    cfgs = []
    for kap in [2, 5, 10, 50]:
        ms = pop.target(Q0, kap)
        for b in BGRID_S:
            cfgs.append((('bayess', kap, b), tasks_for('bayess', 'creditcard', N0, EPS0, Q0, b, 'cliff' if b > 0 else 'none', ms, T=40,
                                                       seed0=int(b * 1e4) + kap)))
    outs = batch(cfgs)
    for (p, kap, b), o in outs.items():
        ms = pop.target(Q0, kap)
        s = summarize(o, pop, Q0, ms, TOL)
        s.update(proto=p, kappa=kap, beta=b, mstar=ms, theory=beta_star(pop, Q0, ms, EPS0))
        res[f'{p}|{kap}|{b}'] = s
    save('rq1a', res)
    log('rq1a_bay done')




def rq5e():
    """Non-monotone strategy against CTQ: S-CLIFF (noise-free only on [L, m*), L = x_q/2) at beta = bbar."""
    pop = getpop('creditcard')
    xq = pop.quantile(Q0)
    _, c = consts(EPS0)
    targets = {'x+1': xq + 1, '10x': pop.target(Q0, KAP0)}
    cfgs = []
    for bbar in [0.005, 0.01, 0.02]:
        for proto in ['nbs', 'bayess', 'sgd']:
            qd = Q0 - bbar * (Q0 + c) - 2 * ETAS[(EPS0, proto)]
            for tname, ms in targets.items():
                cfgs.append(((bbar, proto, tname), tasks_for(proto, 'creditcard', N0, EPS0, Q0, bbar, 'scliff', ms, L=xq // 2, qrun=qd,
                                                              seed0=int(bbar * 1e4) + 131 + len(tname))))
    outs = batch(cfgs)
    res = {}
    for (bbar, proto, tname), o in outs.items():
        m = np.array([x['m'] for x in o], float)
        ok = (m <= xq).astype(float)
        br = np.array([x.mean() for x in np.array_split(ok, 5)])
        res[f'{bbar}|{proto}|{tname}'] = dict(bbar=bbar, proto=proto, target=tname, safe_mean=float(br.mean()), safe_std=float(br.std()),
                                               rank_mean=float(np.mean(pop.F(m.astype(np.int64)))), T=len(m))
    save('rq5e', res)
    log('rq5e done')


def rq4c_top():
    """Direction table, missing cell: inflate the p99 to x_{1.0} (the data maximum), rank distance d = 0.01."""
    pop = getpop('creditcard')
    _, c = consts(EPS0)
    q, d = 0.99, 0.01
    ms = pop.quantile(1.0)
    jobs = [dict(proto='nbs', data='creditcard', n=10**7, eps=EPS0, q=q, kind='cliff', mstar=ms, seed=rep + 41, hi=0.9, crit='up')
            for rep in range(5)]
    from multiprocessing import Pool
    with Pool(2) as p:
        vals = p.map(E._ebs, jobs, chunksize=1)
    res = load('rq4c')
    F = pop.F(ms - 1)
    res[f'{q}|{d}|up'] = dict(q=q, d=d, direction='up', mean=float(np.mean(vals)), std=float(np.std(vals)), theory=(F - q) / (F + c),
                              mstar=int(ms), ratio=ms / pop.quantile(q))
    save('rq4c', res)
    log('rq4c_top done', vals)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        log('start', name)
        globals()[name]()
        log('end', name, f'{time.time() - t:.0f}s')
