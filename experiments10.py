"""Eighth follow-up (fourth external check).
  valid2 : fine-grid validation of every Table II budget (NBS, n = 1e7, CreditCard): hit rate over 1000 fresh runs at
           0.80, 0.85, ..., 1.50 times the reported bisection mean beta-hat; the 90%-hit budget beta_90 is read off the
           isotonic hit curve; the 'validated' budget is the smallest grid budget whose one-sided 95% Wilson lower bound
           reaches 0.9. Cells capped at the search range (IPA, q = 0.999, eps <= 1) are run at 1.4..20%.   -> valid2.json
  path   : exact NBS example on B = 8 (all honest values 1, q = 0.99, eps = 1, n = 3e5): equal ranks, different search
           paths, different finite-n budgets                                                                     -> path.json
Usage: python3 experiments10.py valid2 path
"""
import sys, math, time, json
import numpy as np
from multiprocessing import Pool
from experiments import log, TOL
from runexp import getpop, save, run_nbs
import runexp


def load(name):
    try:
        return json.load(open(f'{runexp.RES}/{name}.json'))
    except FileNotFoundError:
        return {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return cen - half, cen + half


def pava(y):
    """isotonic (non-decreasing) regression with equal weights"""
    blocks = [[v, 1] for v in y]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2 = out.pop()
            v1, w1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    res = []
    for v, w in out:
        res += [v] * w
    return res


def _nbs(a):
    pop = getpop('creditcard')
    return run_nbs(pop, a['n'], a['eps'], a['q'], a['beta'], runexp.ATT[a['kind']], a['mstar'], rng=np.random.default_rng(a['seed']))


def hits(pool, q, eps, kind, beta, ms, R, seed0):
    args = [dict(n=10**7, eps=eps, q=q, beta=beta, kind=kind, mstar=ms, seed=seed0 + i) for i in range(R)]
    m = np.array(pool.map(_nbs, args, chunksize=100), float)
    return int(np.sum(np.abs(m - ms) <= TOL * ms))


def valid2():
    r = load('rq1b')
    res = {}
    R = 1000
    ratios = [round(0.8 + 0.05 * i, 2) for i in range(15)]
    with Pool(2) as pool:
        for ci, (key, v) in enumerate(r.items()):
            q, eps, kind, b, ms = v['q'], v['eps'], v['kind'], v['mean'], v['mstar']
            cap = 4 * max(v['theory'], v['central']) + 0.01
            if abs(b - cap) < 2e-4:
                pts = []
                for bb in [0.014, 0.02, 0.03, 0.05, 0.1, 0.2]:
                    k = hits(pool, q, eps, kind, bb, ms, R, 500000 + 10000 * ci + int(bb * 1000))
                    pts.append((bb, k / R))
                res[key] = dict(q=q, eps=eps, kind=kind, capped=True, beta_hat=b, pts=pts)
                log('valid2 capped', key, pts)
                continue
            hr, lo1 = [], []
            for j, rt in enumerate(ratios):
                k = hits(pool, q, eps, kind, rt * b, ms, R, 300000 + 10000 * ci + 100 * j)
                hr.append(k / R)
                lo1.append(wilson(k, R, 1.645)[0])
            iso = pava(hr)
            b90 = None
            for j in range(len(ratios)):
                if iso[j] >= 0.9:
                    if j == 0:
                        b90 = ratios[0]
                    else:
                        x0, x1, y0, y1 = ratios[j - 1], ratios[j], iso[j - 1], iso[j]
                        b90 = x0 + (0.9 - y0) * (x1 - x0) / (y1 - y0) if y1 > y0 else x1
                    break
            vr = next((rt for rt, l in zip(ratios, lo1) if l >= 0.9), None)
            res[key] = dict(q=q, eps=eps, kind=kind, capped=False, beta_hat=b, ratios=ratios, hit=hr, iso=iso, lo1=lo1,
                            r90=b90, beta90=None if b90 is None else b90 * b, rval=vr, beta_val=None if vr is None else vr * b)
            log('valid2', key, round(100 * b, 3), 'r90', None if b90 is None else round(b90, 3), 'rval', vr)
    save('valid2', res)


def path():
    from scipy.stats import binom
    from scipy.optimize import brentq
    eps, q, b = 1.0, 0.99, 100000
    rho = 1 / (1 + math.e ** eps)
    keep = 1 - rho
    k = math.ceil((rho + (1 - 2 * rho) * q) * b - 1e-12)

    def below(beta):
        return binom.cdf(k - 1, b, (1 - beta) * keep)

    def above(beta):
        return binom.sf(k - 1, b, (1 - beta) * keep + beta)

    h2 = lambda x: above(x) ** 2 * below(x)    # m* = 2: queries 4, 2 (above), 1 (below)
    h7 = lambda x: below(x) ** 2 * above(x)    # m* = 7: queries 4, 6 (below), 7 (above)
    b2 = brentq(lambda x: h2(x) - 0.9, 1e-4, 0.05)
    b7 = brentq(lambda x: h7(x) - 0.9, 1e-4, 0.05)
    out = dict(b2=b2, b7=b7, hit7_at_b2=h7(b2), asym=(1 - q) * (1 - math.exp(-eps)))
    log('path', out)
    save('path', out)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        log('start', name)
        globals()[name]()
        log('end', name, f'{time.time() - t:.0f}s')
