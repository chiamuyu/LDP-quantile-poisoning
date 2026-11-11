"""Fifth follow-up: an analytic, data-free accuracy bound for NBS and what it certifies.

For NBS with R = ceil(log2 B) batches of at least k = floor(n/R) users, Hoeffding's inequality on every batch and a
union bound over the R batches give, with probability >= 1 - delta, |G_hat(t) - G(t)| <= eta_NBS at every probed
threshold, for any (even non-monotone) curve and adaptively chosen thresholds, where
    eta_NBS = sqrt(log(2R/delta) / (2k)) / (1 - 2 rho_eps),   1 - 2 rho_eps = tanh(eps/2).
With this eta, Definition 1 holds for NBS, so Theorem 1 (with Remark 1) certifies a CLIFF budget and Theorem 3 certifies CTQ.

  cert : validate the certified CLIFF budget of Theorem 1 / Remark 1 (CreditCard, q = 0.99, kappa = 10, eps = 1)
         at n = 3e5 ... 1e8, 200 runs each                                                        -> cert.json
  ctq  : CTQ for NBS with the analytic eta (certified; q_dagger = q - bbar (q + c) - eta), CreditCard, n = 1e6 (eps in {0.5, 1, 2}) and n = 1e7 (eps = 1),
         bbar in {0.5, 1, 2}%, attack at beta = 0, bbar, 2 bbar                                  -> cert.json
Usage: python3 experiments7.py cert ctq
"""
import sys, math, time, json
import numpy as np
from experiments import tasks_for, batch, log, Q0, KAP0, TOL
from runexp import getpop, summarize, save, consts
import runexp

DELTA = 0.1


def eta_nbs(n, eps, B, delta=DELTA):
    R = int(math.ceil(math.log2(B)))
    k = n // R
    return math.sqrt(math.log(2 * R / delta) / (2 * k)) / math.tanh(eps / 2)


def cert_budget(pop, q, ms, eps, eta):
    _, c = consts(eps)
    F1, Fm = float(pop.F(ms - 1)), float(pop.F(ms))
    b1 = (F1 - q + eta) / (F1 + c)
    b2 = (q + eta - Fm) / (1 + c - Fm)
    return max(b1, b2), b1, b2


def load(name):
    try:
        return json.load(open(f'{runexp.RES}/{name}.json'))
    except FileNotFoundError:
        return {}


def cert():
    res = load('cert')
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    cfgs = []
    for n in [3 * 10**5, 10**6, 3 * 10**6, 10**7, 10**8]:
        eta = eta_nbs(n, 1.0, pop.B)
        b, b1, b2 = cert_budget(pop, Q0, ms, 1.0, eta)
        cfgs.append(((n, eta, b), tasks_for('nbs', 'creditcard', n, 1.0, Q0, b, 'cliff', ms, T=200, seed0=n % 1009 + 3)))
    outs = batch(cfgs)
    for (n, eta, b), o in outs.items():
        m = np.array([x['m'] for x in o])
        s = summarize(o, pop, Q0, ms, TOL)
        s.update(n=n, eta=eta, beta=b, exact=float(np.mean(m == ms)))
        res[f'cert|{n}'] = s
        log('cert', n, round(eta, 4), round(100 * b, 3), 'exact', s['exact'], 'hit', round(s['hit_mean'], 3))
    save('cert', res)


def ctq():
    res = load('cert')
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    xq = pop.quantile(Q0)
    cfgs = []
    for n, eps in [(10**6, 0.5), (10**6, 1.0), (10**6, 2.0), (10**7, 1.0)]:
        _, c = consts(eps)
        eta = eta_nbs(n, eps, pop.B)
        for bbar in [0.005, 0.01, 0.02]:
            qd = Q0 - bbar * (Q0 + c) - eta          # Theorem 3: q_dagger = q - bbar (q + c) - eta
            for mult in [0.0, 1.0, 2.0]:
                b = mult * bbar
                cfgs.append(((n, eps, bbar, mult, eta, qd), tasks_for('nbs', 'creditcard', n, eps, Q0, b, 'cliff' if b > 0 else 'none', ms,
                                                                     qrun=qd, T=40, seed0=int(b * 1e4) + int(eps * 10) + n % 997 + 5)))
    outs = batch(cfgs)
    for (n, eps, bbar, mult, eta, qd), o in outs.items():
        _, c = consts(eps)
        m = np.array([x['m'] for x in o], float)
        ok = (m <= xq).astype(float)
        br = np.array([x.mean() for x in np.array_split(ok, 5)])
        s = summarize(o, pop, Q0, ms, TOL)
        qlow = (Q0 - bbar * (1 + Q0 + 2 * c) - 2 * eta) / (1 - bbar)
        lo_ok = float(np.mean(pop.F(m.astype(np.int64)) >= qlow)) if qlow > 0 else 1.0
        s.update(n=n, eps=eps, bbar=bbar, mult=mult, eta=eta, qdag=qd, qlow=qlow, safe_mean=float(br.mean()), safe_std=float(br.std()),
                 lower_ok=lo_ok)
        res[f'ctq|{n}|{eps}|{bbar}|{mult}'] = s
        log('ctq', n, eps, bbar, mult, 'eta', round(eta, 4), 'qdag', round(qd, 4), 'safe', round(s['safe_mean'], 3), 'rank', round(s['rank_mean'], 4))
    save('cert', res)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        log('start', name)
        globals()[name]()
        log('end', name, f'{time.time() - t:.0f}s')
