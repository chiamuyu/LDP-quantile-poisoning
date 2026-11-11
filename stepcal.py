"""Step-curve calibration of the protocol error eta (Def. 1) without any data.
Every user answers from the synthetic step curve G(t) = 0 (t < b/10), q - Delta (b/10 <= t < b), q + Delta (t >= b),
the curve that CLIFF creates around its target b, through attack kind 8 at rate 1.
eta_step(delta) = smallest Delta for which the protocol returns b (within the +-10% hit window) w.p. >= 1 - delta."""
import math
import numpy as np
from ldpq import consts, run_nbs, run_bayess, run_sgd, ATT


def encode(eps, q, Delta, side='both'):
    """side='lo': margin Delta below b, maximal above (G = 1 + c); 'up': margin Delta above b, maximal below (G = -c)."""
    rho, _ = consts(eps)
    s2 = rho + (1 - 2 * rho) * (q - Delta) if side != 'up' else 0.0
    s3 = min(1.0, rho + (1 - 2 * rho) * (q + Delta)) if side != 'lo' else 1.0
    S2 = int(round(s2 * 1e7)); S3 = int(round(s3 * 1e7))
    S2 = min(max(S2, 0), 10**7 - 1); S3 = min(max(S3, 0), 10**7 - 1)
    return S2 + 10**7 * S3


def run_step(args):
    proto, pop, n, eps, q, b, Delta, seed, side = args
    L = encode(eps, q, Delta, side)
    if proto == 'nbs':
        m = run_nbs(pop, n, eps, q, 1.0, ATT['curve'], b, L, rng=np.random.default_rng(seed))
    elif proto == 'bayess':
        m = run_bayess(pop, n, eps, q, 1.0, ATT['curve'], b, L, seed=seed)
    elif proto == 'sgd':
        m = run_sgd(pop, n, eps, q, 1.0, ATT['curve'], b, L, seed=seed)[0]
    else:
        raise ValueError(proto)
    return abs(m - b) <= 0.1 * b


def eta_step(proto, pop, n, eps, q, b, delta=0.1, T=100, steps=11, hi=0.1, seed=0, pool=None, side='both'):
    lo_d, hi_d = 0.0, hi
    for s in range(steps):
        mid = (lo_d + hi_d) / 2
        args = [(proto, pop, n, eps, q, b, mid, seed * 100003 + s * 7919 + i, side) for i in range(T)]
        ok = pool.map(run_step, args, chunksize=4) if pool else [run_step(x) for x in args]
        if np.mean(ok) >= 1 - delta:
            hi_d = mid
        else:
            lo_d = mid
    return hi_d


def eta_sides(proto, pop, n, eps, q, b, **kw):
    """(eta_lo, eta_up): margins needed below and above the target, each with the other side maximal."""
    return eta_step(proto, pop, n, eps, q, b, side='lo', **kw), eta_step(proto, pop, n, eps, q, b, side='up', **kw)
