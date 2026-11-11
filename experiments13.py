"""Thirteenth follow-up.
  headline : NBS headline cell (CreditCard, n=1e6, eps=1, q=0.99, kappa=10, beta=1%) with 10,000 runs; Wilson CI
  clean    : clean NBS base rates at n=1e6 and 1e7 (10,000 runs): P[m >= 2 x_q], hit and reach rates of the 10x target
  fgactest : FGA-C vs CLIFF under NBS on the RQ3 budget grid with 2,000 runs per cell; Fisher exact test
  ramp     : change monitor (alarm if a period's estimate exceeds tau times the previous one) against
             (i) CLIFF ramped by a factor r per period toward 10 * m0, (ii) one-shot CLIFF, (iii) MGA, (iv) clean periods
All runs use the exact binomial NBS simulator.                                              -> exp13.json
Usage: python3 experiments13.py
"""
import json, math, time
import numpy as np
from scipy.stats import fisher_exact
from runexp import getpop, RES
from ldpq import run_nbs, beta_star, ATT

EPS, Q, KAP = 1.0, 0.99, 10


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def headline(pop, out):
    ms = pop.target(Q, KAP)
    T = 10000
    rng = np.random.default_rng(1301)
    m = np.array([run_nbs(pop, 10**6, EPS, Q, 0.01, ATT['cliff'], ms, rng=rng) for _ in range(T)])
    hit = int(np.sum(np.abs(m - ms) <= 0.1 * ms))
    ex = int(np.sum(m == ms))
    out['headline'] = dict(T=T, mstar=int(ms), hit=hit / T, hit_ci=wilson(hit, T), exact=ex / T, exact_ci=wilson(ex, T))
    print('headline', out['headline'], flush=True)


def clean(pop, out):
    xq = pop.quantile(Q)
    ms = pop.target(Q, KAP)
    res = {}
    for n in [10**6, 10**7]:
        T = 10000
        rng = np.random.default_rng(1302 + n % 97)
        m = np.array([run_nbs(pop, n, EPS, Q, 0.0, 0, 0, rng=rng) for _ in range(T)])
        k2 = int(np.sum(m >= 2 * xq))
        res[str(n)] = dict(T=T, ge2=k2 / T, ge2_ci=wilson(k2, T), reach=float(np.mean(m >= 0.9 * ms)),
                           hit=float(np.mean(np.abs(m - ms) <= 0.1 * ms)), ge1p5=float(np.mean(m >= 1.5 * xq)))
        print('clean', n, res[str(n)], flush=True)
    out['clean'] = res


def fgactest(pop, out):
    ms = pop.target(Q, KAP)
    bs = beta_star(pop, Q, ms, EPS)
    T = 2000
    res = {'beta_star': bs}
    for ratio in [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]:
        hits = {}
        for att in ['cliff', 'fgac']:
            rng = np.random.default_rng(int(ratio * 1000) + (7 if att == 'fgac' else 3))
            m = np.array([run_nbs(pop, 10**6, EPS, Q, ratio * bs, ATT[att], ms, rng=rng) for _ in range(T)])
            hits[att] = int(np.sum(np.abs(m - ms) <= 0.1 * ms))
        a, b = hits['cliff'], hits['fgac']
        p = fisher_exact([[a, T - a], [b, T - b]])[1]
        res[str(ratio)] = dict(cliff=a / T, fgac=b / T, diff=(b - a) / T, p=float(p))
        print('fgac', ratio, res[str(ratio)], flush=True)
    out['fgactest'] = res


def ramp(pop, out, n=10**7, beta=0.01, r=1.6, tau=2.0, K=8, T=2000):
    xq = pop.quantile(Q)
    true_goal = pop.target(Q, KAP)
    B = pop.B
    rng = np.random.default_rng(1304)
    ramp_ok = ramp_alarm = 0
    periods = []
    oneshot_alarm = mga_alarm = 0
    clean_alarm_runs = 0
    clean_alarm_periods = 0
    for _ in range(T):
        m0 = run_nbs(pop, n, EPS, Q, 0.0, 0, 0, rng=rng)
        goal = min(KAP * m0, B)
        # (i) ramped CLIFF
        prev, alarm, k = m0, False, 0
        while k < K:
            k += 1
            tgt = int(min(math.floor(r * prev), goal))
            m = run_nbs(pop, n, EPS, Q, beta, ATT['cliff'], tgt, rng=rng)
            if m > tau * prev:
                alarm = True
                break
            prev = m
            if tgt == goal:
                break
        if alarm:
            ramp_alarm += 1
        elif abs(prev - true_goal) <= 0.1 * true_goal:
            ramp_ok += 1
            periods.append(k)
        # (ii) one-shot CLIFF to the goal
        m = run_nbs(pop, n, EPS, Q, beta, ATT['cliff'], int(goal), rng=rng)
        oneshot_alarm += int(m > tau * m0)
        # (iii) MGA (untargeted, answers as a holder of B)
        m = run_nbs(pop, n, EPS, Q, beta, ATT['maxval'], 0, rng=rng)
        mga_alarm += int(m > tau * m0)
        # (iv) clean periods
        prev, any_alarm = m0, False
        for _k in range(K):
            m = run_nbs(pop, n, EPS, Q, 0.0, 0, 0, rng=rng)
            if m > tau * prev:
                any_alarm = True
                clean_alarm_periods += 1
            prev = m
        clean_alarm_runs += int(any_alarm)
    res = dict(n=n, beta=beta, r=r, tau=tau, K=K, T=T, xq=int(xq), goal=int(true_goal),
               ramp_success=ramp_ok / T, ramp_success_ci=wilson(ramp_ok, T), ramp_alarm=ramp_alarm / T,
               ramp_periods_median=float(np.median(periods)) if periods else None,
               ramp_periods_max=int(max(periods)) if periods else None,
               oneshot_alarm=oneshot_alarm / T, mga_alarm=mga_alarm / T,
               clean_alarm_any=clean_alarm_runs / T, clean_alarm_per_period=clean_alarm_periods / (T * K))
    print('ramp', res, flush=True)
    out[f'ramp_{n}'] = res


if __name__ == '__main__':
    pop = getpop('creditcard')
    out = {}
    t0 = time.time()
    headline(pop, out)
    clean(pop, out)
    fgactest(pop, out)
    ramp(pop, out, n=10**7)
    ramp(pop, out, n=10**6)
    json.dump(out, open(f'{RES}/exp13.json', 'w'), indent=1)
    print('done', round(time.time() - t0, 1), 's')
